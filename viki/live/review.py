"""Read-only, clock-aligned inspection of recorded causal inference runs.

Only raw timestamps/cloud, hand.jsonl and measured executor logs are inputs.
No episode plan, prepared landmarks, future interpolation or robot connection.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import threading

import h5py
import numpy as np

from viki.contracts import LiveControlOptions
from .session import _safe


def _load_json(path):
    return json.loads(Path(path).read_text())


def _inside(path, roots):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("review path is outside configured data directories")
    return path


def _strict_times(values, label):
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError(f"{label} must be nonempty and strictly increasing")
    return values


def recorded_clock(source, rows, hands):
    """Exact saved clock, or checked legacy reconstruction from native stamps."""
    speed = float(source["playback_speed"])
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("invalid recorded playback speed")
    first = int(rows[0]["sync_us"])
    explicit = source.get("clock")
    if explicit:
        if (int(explicit["recorded_start_us"]) != first
                or float(explicit["playback_speed"]) != speed):
            raise ValueError("recorded clock disagrees with source timestamps")
        return int(explicit["host_start_us"]), "saved source clock"
    devices = tuple(sorted(source["device_ids"]))
    native = {}
    for row in rows:
        signature = tuple(int(row["device_timestamps_us"][dev]) for dev in devices)
        oldest = min(int(row["sync_us"]) + int(row["offsets_us"][dev]) for dev in devices)
        native.setdefault(signature, set()).add(oldest)
    recovered = []
    for hand in hands:
        signature = tuple(int(hand.get("camera_timestamps", {}).get(dev, -1)) for dev in devices)
        oldest_times = native.get(signature)
        if oldest_times is None:
            raise ValueError("online hand does not match reference native timestamps")
        recovered.append({int(hand["source_time_us"]) - int((oldest - first) / speed)
                          for oldest in oldest_times})
    candidates = set.intersection(*recovered) if recovered else set()
    if len(recovered) < 2 or len(candidates) != 1:
        raise ValueError("legacy source clock cannot be verified; record a new live session")
    return candidates.pop(), "verified reconstruction from native timestamps"


class RecordedReview:
    """An immutable completed run, with frame selection by availability time."""

    def __init__(self, directory, *, episode_roots, retargeter_factory=None):
        self.directory = Path(directory)
        self.manifest = _load_json(self.directory / "manifest.json")
        self.report = _load_json(self.directory / "session.json")
        final = self.report["summary"]
        if final.get("running") or final.get("controller", {}).get("running"):
            raise ValueError("review requires a completed online session")
        self.config = self.manifest["config"]
        if self.config["source"] != "recorded" or self.manifest["source"]["kind"] != "recorded":
            raise ValueError("combined review requires recorded RGB-D input")
        self.reference = _inside(self.config["reference"], episode_roots)
        self.rows = _load_json(self.reference / "raw/timestamps.json")
        self.ticks = _strict_times([row["sync_us"] for row in self.rows], "source timestamps")
        self.hands = [json.loads(row) for row in (self.directory / "hand.jsonl").read_text().splitlines()]
        if not self.hands:
            raise ValueError("session contains no online hand observations")
        self.hand_times = _strict_times([h["generated_time_us"] for h in self.hands], "hand publication times")
        _strict_times([h["sequence"] for h in self.hands], "hand sequences")
        self.options = LiveControlOptions(**self.config["options"])
        self.options.validate()
        self.start_us, self.clock_method = recorded_clock(self.manifest["source"], self.rows, self.hands)
        self.speed = float(self.manifest["source"]["playback_speed"])
        logs = list((self.directory / "executor").glob("*/execution.h5"))
        if len(logs) != 1:
            raise ValueError("session must contain exactly one executor log")
        log = _inside(logs[0], [self.directory])
        with h5py.File(log, "r") as file:
            samples = file["samples"]
            self.samples = {key: (samples[key].asstr()[:] if key in ("state", "cause") else samples[key][:])
                            for key in ("host_monotonic_us", "source_time_us", "sequence", "q_observed",
                                        "qd_observed", "q_commanded", "tcp_position", "tcp_rotation",
                                        "target_position", "solver_ms", "residual_rad", "state", "cause",
                                        "command_accepted")}
            self.events = [json.loads(row) for row in file["events_json"].asstr()[:]]
        self.sample_times = _strict_times(self.samples["host_monotonic_us"], "robot measurement times")
        q = self.samples["q_observed"]
        if q.shape != (len(self.sample_times), 6) or not np.isfinite(q).all():
            raise ValueError("invalid measured arm joints in executor log")
        self.executor = self.report.get("executor", self.report["summary"].get("controller", {}))
        self.end_us = max(int(self.sample_times[-1]), int(self.hand_times[-1]),
                          int(self.executor.get("ended_monotonic_us", self.sample_times[-1])))
        if self.end_us <= self.start_us:
            raise ValueError("executor log does not overlap recorded input")
        self.fps = int(self.options.frequency)
        self.n_frames = math.ceil((self.end_us - self.start_us) * self.fps / 1e6) + 1
        if self.n_frames > 30_000:
            raise ValueError("combined review supports runs up to 30000 timeline frames")
        if retargeter_factory is None:
            from viki.retarget import OnlineRetargeter
            retargeter_factory = OnlineRetargeter
        # For display only. Reconstruct FK of measured joints; never solve IK.
        self.kinematics = retargeter_factory(replace(self.options, initial_q=tuple(q[0])))
        saved_model = self.executor.get("kinematic_model", {})
        actual_model = getattr(self.kinematics, "manifest", {})
        if saved_model.get("urdf_sha256") and saved_model["urdf_sha256"] != actual_model.get("urdf_sha256"):
            raise ValueError("current URDF differs from the model used in the online run")
        self.rotation = np.asarray(self.options.rotation_robot_rig, float).T
        binds = [event["binding"] for event in self.events
                 if event.get("kind") == "operator_request" and event.get("command") == "bind"
                 and event.get("accepted") and event.get("binding")]
        if binds:
            binding = binds[0]
            anchor = np.asarray(binding["human_position"], float)
            robot_anchor = np.asarray(binding["robot_position"], float)
            self.alignment = "fixed display alignment at first bind; motion scale is preserved"
        else:
            first_hand = next((h for h in self.hands if h["position_valid"]), None)
            anchor = np.asarray(first_hand["position"], float) if first_hand else np.array([-1., 0., .5])
            robot_anchor = self.samples["tcp_position"][0]
            self.alignment = "display alignment only; no successful bind in this run"
        self.translation = anchor - self.rotation @ robot_anchor
        self.hand_anchor = anchor
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError("invalid display alignment")
        self.actual_path = self._points(self.samples["tcp_position"])
        self.target_path = self._points(self.samples["target_position"])
        self.target_path[self.samples["state"] != "active"] = np.nan
        self.cloud_meta_path = self.reference / "cloud/meta.json"
        self.cloud = _load_json(self.cloud_meta_path) if self.cloud_meta_path.exists() else None
        anchor_file = self.reference / "raw/world_anchor.json"
        self.display_transform = np.asarray(_load_json(anchor_file)["T_world_display"]
                                             if anchor_file.exists() else np.eye(4), dtype=float)
        if (self.display_transform.shape != (4, 4) or not np.isfinite(self.display_transform).all()
                or not np.allclose(self.display_transform[3], [0, 0, 0, 1])
                or not np.allclose(self.display_transform[:3, :3].T @ self.display_transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(self.display_transform[:3, :3]), 1, atol=1e-5)):
            raise ValueError("invalid source world-display transform")
        if self.cloud and self.cloud["n_frames"] != len(self.ticks):
            raise ValueError("cloud frame count differs from raw timestamps")
        self._lock = threading.RLock()
        self._scenes = OrderedDict()

    def _points(self, points):
        return np.asarray(points, float) @ self.rotation.T + self.translation

    def metadata(self):
        with self._lock:
            graph = self.kinematics.scene(self.samples["q_observed"][0])
        return _safe({
            "id": self.directory.name, "reference": str(self.reference), "episode_id": self.reference.name,
            "profile": self.manifest["perception"]["profile"], "backend": self.config["backend"],
            "hardware": self.manifest.get("hardware", False), "fps": self.fps, "n_frames": self.n_frames,
            "duration_s": (self.n_frames - 1) / self.fps, "source_frames": len(self.ticks),
            "recorded_speed": self.speed, "scale": self.options.scale, "orientation": self.options.orientation,
            "input_ttl_ms": self.options.input_ttl_s * 1000,
            "clock_method": self.clock_method, "alignment": self.alignment,
            "robot_base_rig": self.translation, "cloud": self.cloud,
            "robot_initial_points_rig": self._points(graph["points"]), "hand_anchor_rig": self.hand_anchor,
            "display_transform": self.display_transform,
            "phase": self.report["summary"]["phase"], "reason": self.report["summary"]["reason"],
            "robot_edges": graph["edges"], "actual_trajectory": self.actual_path,
            "target_trajectory": self.target_path,
            "scope": "recorded causal inference and measured robot-state replay; display registration is not physical calibration",
        })

    def frame(self, index):
        if not 0 <= index < self.n_frames:
            raise IndexError("review frame out of range")
        now = self.start_us + round(index * 1e6 / self.fps)
        recorded = int(self.ticks[0]) + int((now - self.start_us) * self.speed)
        source_index = min(len(self.ticks) - 1, max(0, bisect_right(self.ticks, recorded) - 1))
        hi = bisect_right(self.hand_times, now) - 1
        ri = bisect_right(self.sample_times, now) - 1
        hand = None
        if hi >= 0:
            estimate = self.hands[hi]
            age = (now - int(estimate["source_time_us"])) / 1000
            valid = bool(estimate["position_valid"] and 0 <= age < self.options.input_ttl_s * 1000)
            hand = {**estimate, "age_ms": age, "visible": valid,
                    "processing_ms": (int(estimate["generated_time_us"]) - int(estimate["source_time_us"])) / 1000}
        robot = None
        if ri >= 0:
            with self._lock:
                if ri not in self._scenes:
                    graph = self.kinematics.scene(self.samples["q_observed"][ri])
                    self._scenes[ri] = self._points(graph["points"])
                    if len(self._scenes) > 256:
                        self._scenes.popitem(last=False)
                points = self._scenes[ri]
            state, cause = self.samples["state"][ri], self.samples["cause"][ri]
            if now >= self.end_us:
                state, cause = self.executor.get("state", state), self.executor.get("reason", cause)
            robot = {"sample_index": ri, "points": points, "q_observed": self.samples["q_observed"][ri],
                     "q_commanded": self.samples["q_commanded"][ri], "tcp_position": self.actual_path[ri],
                     "tcp_rotation": self.rotation @ self.samples["tcp_rotation"][ri].reshape(3, 3),
                     "target_position": self.target_path[ri], "state": state, "reason": cause,
                     "measurement_age_ms": (now - self.sample_times[ri]) / 1000,
                     "solver_ms": self.samples["solver_ms"][ri],
                     "tracking_error_rad": self.samples["residual_rad"][ri],
                     "input_sequence": int(self.samples["sequence"][ri]),
                     "command_accepted": bool(self.samples["command_accepted"][ri])}
        return _safe({"frame": index, "time_s": index / self.fps, "source_frame": source_index,
                      "source_time_s": (self.ticks[source_index] - self.ticks[0]) / 1e6,
                      "hand": hand, "robot": robot})

    def cloud_path(self, frame):
        if not 0 <= frame < len(self.ticks):
            raise IndexError("cloud frame out of range")
        path = _inside(self.reference / "cloud" / f"{frame:06d}.bin", [self.reference])
        if not path.is_file():
            raise FileNotFoundError("source point cloud is not built; run viki cloud for the reference episode")
        return path


class RecordedReviewStore:
    """Bounded read-only cache; separate from the owner of any running robot."""

    def __init__(self, root="data/live_sessions", *, episode_roots=None, retargeter_factory=None):
        self.root = Path(root).resolve()
        self.episode_roots = tuple(Path(p).resolve() for p in (episode_roots or ("data/datasets", "data/episodes")))
        self.retargeter_factory = retargeter_factory
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _directory(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("invalid online session ID")
        directory = _inside(self.root / identifier, [self.root])
        if not directory.is_dir():
            raise FileNotFoundError("online session not found")
        return directory

    def list(self):
        result = []
        if not self.root.exists():
            return result
        for directory in self.root.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                directory = self._directory(directory.name)
                manifest = _load_json(directory / "manifest.json")
                report = _load_json(directory / "session.json")
                if report["summary"].get("running") or report["summary"].get("controller", {}).get("running"):
                    continue
                if manifest["config"]["source"] != "recorded" or not (directory / "hand.jsonl").stat().st_size:
                    continue
                reference = _inside(manifest["config"]["reference"], self.episode_roots)
                result.append({"id": directory.name, "episode_id": reference.name,
                               "backend": manifest["config"]["backend"],
                               "orientation": manifest["config"]["options"]["orientation"],
                               "recorded_speed": manifest["source"]["playback_speed"],
                               "phase": report["summary"]["phase"], "reason": report["summary"]["reason"],
                               "modified_ns": (directory / "session.json").stat().st_mtime_ns})
            except (KeyError, ValueError, OSError):
                continue
        return sorted(result, key=lambda row: row["modified_ns"], reverse=True)

    def get(self, identifier):
        with self._lock:
            if identifier not in self._cache:
                self._cache[identifier] = RecordedReview(self._directory(identifier), episode_roots=self.episode_roots,
                                                         retargeter_factory=self.retargeter_factory)
                if len(self._cache) > 2:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(identifier)
            return self._cache[identifier]
