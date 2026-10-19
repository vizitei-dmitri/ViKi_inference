"""Stateful causal multi-view hand estimation, separate from offline profiles.

No gap filling, centered smoothing, episode median or hand-fit is used. Exact
factory color rays are rectified once; depth-frame ranges are deliberately not
used as color-frame depth priors. Single-hand scenes are required by the current
MediaPipe backend; this module does not claim multi-person identity association.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from viki.cameras import realsense_color_pixel_to_ray
from viki.contracts import CalibrationExtrinsics, Episode, LiveHandEstimate, SyncedFrameGroup
from .camera_prep import prepare_frame
from .hand_angles import compute_palm_rotation
from .triangulate import _Cam, TriConfig, triangulate_joint


ARM_LANDMARKS = (0, 1, 4, 5, 8, 9, 17)
PALM_LANDMARKS = (0, 5, 9, 17)
LIVE_PROFILE = "live-human-sdk-rays-causal-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class LiveCameraGeometry:
    device_id: str
    camera: _Cam
    projector: object
    intrinsics: dict
    reference: dict
    factory_hash: str
    factory_data: dict | None = None

    def color_ray(self, u, v):
        if self.factory_data is not None:
            return realsense_color_pixel_to_ray(self.factory_data["color"], u, v)
        return self.projector.color_pixel_to_ray(u, v)


def load_live_geometry(episode: Episode | str | Path) -> tuple[dict[str, LiveCameraGeometry], str]:
    """Load measured cached camera geometry without opening any camera.

    Missing factory calibration or SDK is an error; there is no identity or
    guessed pinhole fallback. Only raw calibration and metadata are read.
    """
    ep = episode if isinstance(episode, Episode) else Episode(Path(episode))
    raw = ep.raw_dir
    meta = json.loads(ep.meta_path.read_text())
    intrinsics = json.loads((raw / "intrinsics.json").read_text())
    extrinsics = json.loads((raw / "extrinsics.json").read_text())
    devices = sorted(meta.get("cameras", {}))
    if len(devices) < 2:
        raise ValueError("live geometry requires at least two reference cameras")
    digest = hashlib.sha256()
    for path in (ep.meta_path, raw / "intrinsics.json", raw / "extrinsics.json"):
        digest.update(path.read_bytes())
    anchor = raw / "world_anchor.json"
    if anchor.is_file():
        digest.update(anchor.read_bytes())
    contexts = {}
    try:
        for dev in devices:
            reference = meta["cameras"][dev]
            intr = intrinsics[dev]
            color = intr["color"]
            K = np.array([[color["fx"], 0., color["cx"]],
                          [0., color["fy"], color["cy"]], [0., 0., 1.]])
            if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError(f"invalid color intrinsics for {dev}")
            extr = extrinsics[dev]
            T = CalibrationExtrinsics(np.asarray(extr["rvec"], float), np.asarray(extr["tvec"], float)).transform_matrix
            if not np.isfinite(T).all():
                raise ValueError(f"invalid camera extrinsics for {dev}")
            sensor = reference.get("type")
            factory_data = None
            if sensor == "kinect":
                from .k4a_offline import K4ACalibration
                path = raw / reference.get("k4a_calib", f"{dev}_k4a_calib.bin")
                projector = K4ACalibration.from_episode(raw, dev, meta)
            elif sensor == "realsense":
                from .rs_offline import RealSenseCalibration
                path = raw / reference.get("rs_calib", f"{dev}_rs_calib.json")
                projector = RealSenseCalibration.from_episode(raw, dev, meta)
                if path.is_file():
                    factory_data = json.loads(path.read_text())
                    # Validate SDK distortion support before processing begins.
                    realsense_color_pixel_to_ray(factory_data["color"], color["cx"], color["cy"])
            else:
                raise ValueError(f"unsupported reference sensor {sensor!r} for {dev}")
            if projector is None or not path.is_file():
                raise RuntimeError(f"exact cached factory calibration unavailable for {dev}; no fallback")
            digest.update(path.read_bytes())
            # SDK rays become synthetic ideal pixels. Zero distortion prevents
            # the triangulator from correcting already corrected rays again.
            camera = _Cam(dev, {"K": K, "dist": np.zeros(5), "T_wc": T,
                                "image_size": [color["width"], color["height"]]})
            contexts[dev] = LiveCameraGeometry(dev, camera, projector, intr, reference,
                                                _sha256(path), factory_data)
    except BaseException:
        for context in contexts.values():
            close = getattr(context.projector, "close", None)
            if close:
                close()
        raise
    return contexts, digest.hexdigest()


class _OneEuro:
    """Independent landmark states; missing observations are never propagated."""

    def __init__(self, min_cutoff: float = 2., beta: float = .5):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.reset()

    def reset(self):
        self.values = np.full((21, 3), np.nan)
        self.raw = np.full((21, 3), np.nan)
        self.derivative = np.zeros((21, 3))
        self.times = np.zeros(21, dtype=np.int64)

    @staticmethod
    def _alpha(cutoff, dt):
        return 1. / (1. + 1. / (2. * np.pi * cutoff * dt))

    def update(self, points, observed, now_us):
        result = np.full((21, 3), np.nan)
        for lm in np.flatnonzero(observed):
            dt = (now_us - self.times[lm]) / 1e6
            if self.times[lm] == 0 or dt > .25:
                filtered = points[lm].copy()
                self.derivative[lm] = 0.
            else:
                derivative = (points[lm] - self.raw[lm]) / dt
                a = self._alpha(1., dt)
                self.derivative[lm] = a * derivative + (1 - a) * self.derivative[lm]
                cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.derivative[lm])
                a = self._alpha(cutoff, dt)
                filtered = a * points[lm] + (1 - a) * self.values[lm]
            result[lm] = filtered
            self.raw[lm], self.values[lm], self.times[lm] = points[lm], filtered, now_us
        return result


class LiveHandEstimator:
    """Persistent per-camera VIDEO detectors and current-frame-only geometry."""

    def __init__(self, geometry: dict[str, LiveCameraGeometry], detectors: dict,
                 *, calibration_id: str, landmark_mode: str = "all21",
                 max_age_s: float = .25, max_skew_s: float = .02,
                 target_anchor: str = "pinch_center", hand: str = "right",
                 warmup_frames: int = 10, clock_us=None, model_sha256: str = ""):
        if landmark_mode not in ("all21", "arm7"):
            raise ValueError("landmark_mode must be all21 or arm7")
        if target_anchor not in ("pinch_center", "wrist"):
            raise ValueError("target_anchor must be pinch_center or wrist")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if not math.isfinite(max_skew_s) or max_skew_s <= 0:
            raise ValueError("max_skew_s must be finite and positive")
        if warmup_frames < 1 or len(geometry) < 2 or set(geometry) != set(detectors):
            raise ValueError("require matching detectors for at least two cameras and positive warmup")
        self.geometry, self.detectors = geometry, detectors
        self.calibration_id = calibration_id
        self.device_ids = tuple(sorted(geometry))
        self.landmark_mode, self.target_anchor, self.hand = landmark_mode, target_anchor, hand
        self.landmarks = tuple(range(21)) if landmark_mode == "all21" else ARM_LANDMARKS
        self._clock_us = clock_us or (lambda: time.monotonic_ns() // 1000)
        self._max_age_us, self._max_skew_us = int(max_age_s * 1e6), int(max_skew_s * 1e6)
        self._last_native = {}
        self._last_source_us = 0
        self._sequence = 0
        self._filter = _OneEuro()
        self._rotation = None
        self._rotation_time_us = 0
        self._last_wrist = None
        self._last_wrist_time_us = 0
        self._warmup_frames = warmup_frames
        self._warmup_lengths = []
        self._palm_reference = None
        self._closed = False
        # Independent live recipe; none of the immutable offline profiles change.
        self._tri = TriConfig({"min_score": 0., "time_align": False, "min_ray_deg": 5.,
                               "reproj_inlier_mrad": 25., "depth_lambda": 0.,
                               "geometry_cameras": [], "quality_detector_score": False})
        self.manifest = {
            "profile": LIVE_PROFILE, "landmark_mode": landmark_mode,
            "landmark_ids": list(self.landmarks), "target_anchor": target_anchor,
            "calibration_id": calibration_id, "model_sha256": model_sha256,
            "camera_factory_sha256": {d: g.factory_hash for d, g in geometry.items()},
            "detector": "persistent per-camera VIDEO", "single_hand_scene_required": True,
            "optics": "factory SDK color rays -> synthetic K pixels -> zero distortion",
            "depth_prior": "disabled: native depth z is not color-frame z",
            "filter": {"kind": "causal One Euro", "min_cutoff_hz": 2., "beta": .5,
                       "derivative_cutoff_hz": 1., "orientation_cutoff_hz": 3.},
            "triangulation": self._tri.as_dict(), "warmup_frames": warmup_frames,
            "max_age_s": max_age_s, "max_skew_s": max_skew_s,
        }

    @classmethod
    def from_episode(cls, episode: Episode | str | Path, *, landmark_mode="all21",
                     model="mediapipe", max_age_s=.25, target_anchor="pinch_center", **kwargs):
        if model != "mediapipe":
            raise ValueError("initial live implementation supports mediapipe only")
        ep = episode if isinstance(episode, Episode) else Episode(Path(episode))
        geometry, calibration_id = load_live_geometry(ep)
        detectors = {}
        try:
            from .backends.mediapipe import MediaPipeHandBackend, _ensure_model
            model_path = Path(_ensure_model("models")).resolve()
            for dev in geometry:
                detectors[dev] = MediaPipeHandBackend(mode="video", model_path=str(model_path),
                                                     min_confidence=.3, tracking_confidence=.3)
            meta = json.loads(ep.meta_path.read_text())
            return cls(geometry, detectors, calibration_id=calibration_id,
                       landmark_mode=landmark_mode, max_age_s=max_age_s,
                       target_anchor=target_anchor, hand=meta.get("hand", "right"),
                       model_sha256=_sha256(model_path), **kwargs)
        except BaseException:
            for detector in detectors.values():
                detector.close()
            for context in geometry.values():
                close = getattr(context.projector, "close", None)
                if close:
                    close()
            raise

    def verify_cameras(self, manager) -> None:
        """Refuse a different device factory calibration, mode or image geometry."""
        for dev, context in self.geometry.items():
            info = manager.get_info(dev)
            backend = manager.get_backend(dev)
            if not info or not info.get("running") or not info.get("has_frame") or backend is None:
                raise RuntimeError(f"camera {dev} must have a fresh frame before live startup")
            for shape in ("color_shape", "depth_shape"):
                if list(info.get(shape, [])) != list(context.reference.get(shape, [])):
                    raise RuntimeError(f"camera {dev} {shape} differs from reference calibration")
            requested, actual = context.reference.get("requested", {}), info.get("config", {})
            for key in ("depth_mode", "color_width", "color_height"):
                if key in requested and actual.get(key) != requested[key]:
                    raise RuntimeError(f"camera {dev} {key} differs from reference calibration")
            if context.reference.get("type") == "kinect":
                blob = backend.get_raw_calibration()
                if not blob or hashlib.sha256(blob).hexdigest() != context.factory_hash:
                    raise RuntimeError(f"camera {dev} factory calibration changed (device/order mismatch)")
            else:
                data = backend.get_rs_calibration()
                if data != context.factory_data:
                    raise RuntimeError(f"camera {dev} RealSense factory calibration changed")

    def _clear_tracking(self):
        self._filter.reset()
        self._rotation = None
        self._rotation_time_us = 0

    def _result(self, source_time, points, observed, quality, reason="", *, valid=False,
                rotation=None, orientation_valid=False, camera_timestamps=None):
        self._sequence += 1
        position = np.full(3, np.nan)
        if valid:
            position = (points[4] + points[8]) * .5 if self.target_anchor == "pinch_center" else points[0].copy()
        opening = float(np.linalg.norm(points[4] - points[8])) if observed[4] and observed[8] else float("nan")
        return LiveHandEstimate(
            sequence=self._sequence, source_time_us=source_time,
            generated_time_us=int(self._clock_us()), points=points.copy(), observed=observed.copy(),
            quality=quality.copy(), position=position, rotation=(np.full((3, 3), np.nan)
                     if rotation is None else rotation.copy()), position_valid=bool(valid),
            orientation_valid=bool(orientation_valid), calibration_id=self.calibration_id,
            reason=reason, opening=opening, camera_timestamps=camera_timestamps or {},
        )

    def process(self, group: SyncedFrameGroup) -> LiveHandEstimate | None:
        if self._closed:
            raise RuntimeError("live estimator is closed")
        if set(group.frames) != set(self.device_ids):
            raise ValueError("frame camera IDs differ from fixed live geometry")
        now_us = int(self._clock_us())
        source_time = min(int(frame.host_timestamp_us) for frame in group.frames.values())
        native = {dev: int(group.frames[dev].timestamp_us) for dev in self.device_ids}
        if source_time <= self._last_source_us or any(
            ts <= self._last_native.get(dev, -1) for dev, ts in native.items()
        ):
            return None
        if source_time <= 0 or any(now_us - int(f.host_timestamp_us) < 0 or
                                   now_us - int(f.host_timestamp_us) > self._max_age_us
                                   for f in group.frames.values()):
            return None
        self._last_native, self._last_source_us = native, source_time
        points, observed, quality = np.full((21, 3), np.nan), np.zeros(21, bool), np.zeros(21)
        skew = max(int(f.host_timestamp_us) for f in group.frames.values()) - source_time
        if skew > self._max_skew_us:
            self._clear_tracking()
            return self._result(source_time, points, observed, quality, "camera_skew", camera_timestamps=native)
        observations = {}
        for dev, context in self.geometry.items():
            frame = group.frames[dev]
            color = context.intrinsics["color"]
            depth = context.intrinsics["depth"]
            if frame.color.shape != (color["height"], color["width"], 3) or frame.depth.shape != (depth["height"], depth["width"]):
                raise ValueError(f"camera {dev} resolution changed during session")
            detection = self.detectors[dev].detect(prepare_frame(frame), self.hand)
            if detection is None:
                continue
            scores = detection.lm_score
            if scores is None:
                scores = np.full(21, detection.confidence)
            row = {}
            for lm in self.landmarks:
                uv = detection.points.get(lm)
                if uv is None or not np.isfinite(uv).all() or not np.isfinite(scores[lm]) or scores[lm] <= 0:
                    continue
                if not (0 <= uv[0] < color["width"] and 0 <= uv[1] < color["height"]):
                    continue
                ray = context.color_ray(float(uv[0]), float(uv[1]))
                if ray is None or not np.isfinite(ray).all() or ray[2] <= 0:
                    continue
                rectified = context.camera.K @ (np.asarray(ray) / ray[2])
                row[lm] = {"camera_id": dev, "uv": rectified[:2], "score": float(scores[lm]),
                           "depth_m": float("nan"), "depth_valid": False,
                           "depth_spread_m": float("nan")}
            observations[dev] = row
        for lm in self.landmarks:
            views = [row[lm] for row in observations.values() if lm in row]
            result = triangulate_joint(views, {d: c.camera for d, c in self.geometry.items()}, lm, self._tri)
            if result is not None and np.isfinite(result["xyz"]).all() and result["quality"] >= .1:
                points[lm], quality[lm], observed[lm] = result["xyz"], result["quality"], True
        if int(self._clock_us()) - source_time > self._max_age_us:
            self._clear_tracking()
            return self._result(source_time, points, observed, quality, "processing_stale", camera_timestamps=native)
        required = (0, 4, 8) if self.target_anchor == "pinch_center" else (0,)
        if not all(observed[list(required)]):
            self._clear_tracking()
            return self._result(source_time, points, observed, quality, "missing_position", camera_timestamps=native)
        rotation = None
        if all(observed[list(PALM_LANDMARKS)]):
            lengths = np.array([np.linalg.norm(points[9] - points[0]), np.linalg.norm(points[17] - points[5])])
            if not (.03 <= lengths[0] <= .15 and .02 <= lengths[1] <= .14):
                self._clear_tracking()
                return self._result(source_time, points, observed, quality, "palm_anatomy", camera_timestamps=native)
            if self._palm_reference is not None and np.any((lengths < .65 * self._palm_reference) | (lengths > 1.5 * self._palm_reference)):
                self._clear_tracking()
                return self._result(source_time, points, observed, quality, "palm_reference", camera_timestamps=native)
            rotation = compute_palm_rotation(*[points[lm] for lm in PALM_LANDMARKS])
            if rotation is None:
                self._clear_tracking()
                return self._result(source_time, points, observed, quality, "degenerate_palm", camera_timestamps=native)
            if self._palm_reference is None:
                self._warmup_lengths.append(lengths)
                if len(self._warmup_lengths) < self._warmup_frames:
                    return self._result(source_time, points, observed, quality, "warming_up", camera_timestamps=native)
                self._palm_reference = np.median(self._warmup_lengths, axis=0)
                self._warmup_lengths.clear()
        elif self._palm_reference is None:
            return self._result(source_time, points, observed, quality, "missing_palm_warmup", camera_timestamps=native)
        if self._last_wrist is not None:
            dt = (source_time - self._last_wrist_time_us) / 1e6
            if dt <= self._max_age_us / 1e6 and np.linalg.norm(points[0] - self._last_wrist) > .03 + 1.5 * dt:
                self._clear_tracking()
                return self._result(source_time, points, observed, quality, "position_jump", camera_timestamps=native)
        self._last_wrist, self._last_wrist_time_us = points[0].copy(), source_time
        filtered = self._filter.update(points, observed, source_time)
        if rotation is not None:
            if self._rotation is not None:
                delta = Rotation.from_matrix(self._rotation.T @ rotation).as_rotvec()
                if np.linalg.norm(delta) > 1.2:
                    rotation = None
                else:
                    dt = (source_time - self._rotation_time_us) / 1e6
                    alpha = _OneEuro._alpha(3., dt)
                    rotation = self._rotation @ Rotation.from_rotvec(alpha * delta).as_matrix()
            if rotation is not None:
                self._rotation, self._rotation_time_us = rotation, source_time
        return self._result(source_time, filtered, observed, quality, valid=True, rotation=rotation,
                            orientation_valid=rotation is not None, camera_timestamps=native)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for detector in self.detectors.values():
            detector.close()
        for context in self.geometry.values():
            close = getattr(context.projector, "close", None)
            if close:
                close()
