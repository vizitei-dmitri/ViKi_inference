"""Bounded pull sources for causal processing; camera ownership stays explicit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from viki.contracts import CameraIntrinsics, Episode, Frame, SyncedFrameGroup
from .sync import MultiCameraSync


def realsense_color_pixel_to_ray(intrinsics: dict, u: float, v: float) -> np.ndarray | None:
    """Exact librealsense ray from a recorded color factory calibration."""
    import pyrealsense2 as rs

    calibrated = rs.intrinsics()
    for attr in ("width", "height", "fx", "fy", "ppx", "ppy"):
        setattr(calibrated, attr, intrinsics[attr])
    model = str(intrinsics.get("model", "none")).removeprefix("distortion.")
    if model not in ("none", "brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"):
        raise ValueError(f"unsupported RealSense color distortion {model!r}")
    calibrated.model = getattr(rs.distortion, model)
    calibrated.coeffs = list(intrinsics.get("coeffs", [0.] * 5))
    try:
        ray = np.asarray(rs.rs2_deproject_pixel_to_point(calibrated, [float(u), float(v)], 1.), float)
    except RuntimeError:
        return None
    if ray.shape != (3,) or not np.isfinite(ray).all() or ray[2] <= 0:
        return None
    return ray / ray[2]


def _episode(value: Episode | str | Path) -> Episode:
    return value if isinstance(value, Episode) else Episode(Path(value))


class RecordedFrameSource:
    """Decode only arrived RGB-D groups, retimed onto the current host clock.

    On overload old *due* groups are dropped. Native device timestamps are
    retained for deduplication. Neither future images nor derived artifacts
    are read. ``poll`` never waits for a future group; ``close`` is idempotent.
    """

    def __init__(self, episode: Episode | str | Path, *, speed: float = 1.0,
                 device_ids: list[str] | tuple[str, ...] | None = None,
                 clock_us=None) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("recorded playback speed must be finite and positive")
        self.episode = _episode(episode)
        raw = self.episode.raw_dir
        self._timestamps = json.loads((raw / "timestamps.json").read_text())
        if not self._timestamps:
            raise ValueError("recorded frame source requires nonempty timestamps")
        ticks = [int(row["sync_us"]) for row in self._timestamps]
        if any(b <= a for a, b in zip(ticks, ticks[1:])):
            raise ValueError("recorded sync timestamps must increase strictly")
        intrinsics = json.loads((raw / "intrinsics.json").read_text())
        available = sorted(p.stem for p in raw.glob("*.mp4"))
        self.device_ids = tuple(device_ids or available)
        if len(self.device_ids) < 2 or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("live triangulation requires at least two distinct cameras")
        if any(dev not in available or dev not in intrinsics for dev in self.device_ids):
            raise ValueError("reference recording does not contain every required camera")
        self.speed = float(speed)
        self.fps = 1e6 / float(np.median(np.diff(ticks))) if len(ticks) > 1 else 0.
        self.recorded_duration = (ticks[-1] - ticks[0]) / 1e6
        self.source_summary = {"kind": "recorded", "episode": str(self.episode.root),
                               "device_ids": list(self.device_ids), "playback_speed": self.speed,
                               "recorded_fps": self.fps, "frames": len(ticks)}
        self._clock_us = clock_us or (lambda: time.monotonic_ns() // 1000)
        # Start the clock after video setup, so setup is not observation latency.
        self._captures = {}
        self._intrinsics = {}
        self._raw = raw
        self._index = 0
        self._closed = False
        self.dropped_groups = 0
        try:
            for dev in self.device_ids:
                capture = cv2.VideoCapture(str(raw / f"{dev}.mp4"))
                self._captures[dev] = capture
                if not capture.isOpened():
                    raise RuntimeError(f"cannot decode reference camera {dev}")
                self._intrinsics[dev] = {
                    key: CameraIntrinsics(**{
                        k: (np.asarray(v) if k == "dist_coeffs" else v)
                        for k, v in intrinsics[dev][key].items()
                        if k in CameraIntrinsics.__dataclass_fields__
                    }) for key in ("color", "depth")
                }
        except BaseException:
            self.close()
            raise
        self._start_us = int(self._clock_us())
        self._recorded_start_us = ticks[0]
        self.source_summary["clock"] = {
            "host_start_us": self._start_us,
            "recorded_start_us": self._recorded_start_us,
            "playback_speed": self.speed,
        }

    @property
    def exhausted(self) -> bool:
        return self._closed or self._index >= len(self._timestamps)

    def _retime(self, timestamp_us: int) -> int:
        return self._start_us + int((timestamp_us - self._recorded_start_us) / self.speed)

    def poll(self) -> SyncedFrameGroup | None:
        if self.exhausted:
            return None
        now_us = int(self._clock_us())
        if self._retime(int(self._timestamps[self._index]["sync_us"])) > now_us:
            return None
        selected = self._index
        while selected + 1 < len(self._timestamps):
            if self._retime(int(self._timestamps[selected + 1]["sync_us"])) > now_us:
                break
            selected += 1
        skipped = selected - self._index
        for dev, capture in self._captures.items():
            for _ in range(skipped):
                if not capture.grab():
                    raise RuntimeError(f"incomplete RGB recording for {dev}")
        self.dropped_groups += skipped
        row = self._timestamps[selected]
        sync_us = self._retime(int(row["sync_us"]))
        frames, offsets = {}, {}
        for dev, capture in self._captures.items():
            decoded, color = capture.read()
            depth_path = self._raw / f"{dev}_depth" / f"{selected:06d}.npy"
            if not decoded or not depth_path.is_file():
                raise RuntimeError(f"missing reference RGB-D: {dev} frame {selected}")
            depth = np.load(depth_path, allow_pickle=False)
            if depth.ndim != 2 or depth.dtype != np.uint16:
                raise ValueError(f"invalid recorded depth for {dev}")
            native = row.get("device_timestamps_us", {}).get(dev)
            if native is None:
                raise ValueError(f"reference recording lacks native timestamp for {dev}")
            host_us = self._retime(int(row["sync_us"]) + int(row["offsets_us"][dev]))
            frames[dev] = Frame(
                color=color, depth=depth, timestamp_us=int(native), device_id=dev,
                host_timestamp_us=host_us,
                color_intrinsics=self._intrinsics[dev]["color"],
                depth_intrinsics=self._intrinsics[dev]["depth"],
            )
            offsets[dev] = host_us - sync_us
        self._index = selected + 1
        return SyncedFrameGroup(frames=frames, sync_timestamp_us=sync_us, offsets_us=offsets)

    read = poll

    def close(self) -> None:
        if not self._closed:
            for capture in self._captures.values():
                capture.release()
            self._closed = True


class CameraFrameSource:
    """Pull new synchronized frames from an existing manager, never own SDKs."""

    def __init__(self, manager, *, device_ids: list[str] | tuple[str, ...],
                 sync_fps: int = 20, max_age_s: float = .25, clock_us=None) -> None:
        self.device_ids = tuple(device_ids)
        if len(self.device_ids) < 2 or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("at least two distinct camera IDs are required")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if not set(self.device_ids).issubset(manager.active_device_ids()):
            raise ValueError("every required camera must already be active")
        self._manager = manager
        self._sync = MultiCameraSync(manager, sync_fps=sync_fps,
                                     required_devices=list(self.device_ids))
        self._clock_us = clock_us or (lambda: time.monotonic_ns() // 1000)
        self._max_age_us = int(max_age_s * 1e6)
        self._last_native: dict[str, int] = {}
        self._closed = False
        self.fps = sync_fps
        self.source_summary = {"kind": "cameras", "device_ids": list(self.device_ids),
                               "sync_fps": sync_fps}

    @property
    def exhausted(self) -> bool:
        return self._closed

    def poll(self) -> SyncedFrameGroup | None:
        if self._closed:
            return None
        if not set(self.device_ids).issubset(self._manager.active_device_ids()):
            raise RuntimeError("required camera stopped; create a new live session")
        group = self._sync.get_synced_frame()
        if group is None:
            return None
        now_us = int(self._clock_us())
        for dev in self.device_ids:
            frame = group.frames[dev]
            previous = self._last_native.get(dev)
            if previous is not None and frame.timestamp_us < previous:
                raise RuntimeError(f"camera clock reset for {dev}; create a new session")
            if previous is not None and frame.timestamp_us == previous:
                return None
            age = now_us - int(frame.host_timestamp_us)
            if age < 0 or age > self._max_age_us:
                return None
        self._last_native = {dev: int(group.frames[dev].timestamp_us) for dev in self.device_ids}
        return group

    read = poll

    def close(self) -> None:
        # The server/CLI owner decides when to stop the shared cameras.
        self._closed = True
