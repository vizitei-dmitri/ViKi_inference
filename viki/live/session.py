"""Session supervision. Camera/perception and servo execution remain independent."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

from viki.contracts import Episode, HumanTarget, LiveControlOptions


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def load_mapping(path: str | Path) -> tuple[dict, dict]:
    """Read an operator-supplied rig→robot rotation and robot-frame floor."""
    file = Path(path)
    contents = file.read_bytes()
    payload = json.loads(contents)
    if not isinstance(payload, dict):
        raise ValueError("mapping must be a JSON object")
    if payload.get("coordinate_frame") != "rig" or payload.get("confirmed") is not True:
        raise ValueError("mapping must declare coordinate_frame='rig' and confirmed=true")
    for key in ("rotation_robot_rig", "floor_normal_robot", "floor_offset_m"):
        if key not in payload:
            raise ValueError(f"mapping missing {key}")
    changes = {
        "rotation_robot_rig": tuple(tuple(row) for row in payload["rotation_robot_rig"]),
        "floor_normal": tuple(payload["floor_normal_robot"]),
        "floor_offset_m": float(payload["floor_offset_m"]),
        "mapping_confirmed": True,
    }
    replace(LiveControlOptions(), **changes).validate()
    return changes, {"path": str(file), "sha256": hashlib.sha256(contents).hexdigest(),
                     "source_calibration_id": payload.get("source_calibration_id")}


@dataclass(frozen=True)
class LiveSessionConfig:
    reference: str
    source: str = "recorded"
    backend: str = "mujoco"
    perception_hz: float = 15.
    recorded_speed: float = 1.
    landmark_mode: str = "all21"
    seconds: float = 60.
    auto_bind: bool = False
    out_dir: str = "data/live_sessions"
    options: LiveControlOptions = field(default_factory=LiveControlOptions)
    mapping_manifest: dict = field(default_factory=dict)

    def validate(self):
        self.options.validate()
        if self.source not in ("recorded", "cameras") or self.backend not in ("mujoco", "ur10e"):
            raise ValueError("invalid live source/backend")
        if self.landmark_mode not in ("all21", "arm7"):
            raise ValueError("landmark_mode must be all21 or arm7")
        for name in ("perception_hz", "recorded_speed", "seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.perception_hz > 30 or self.recorded_speed > 2:
            raise ValueError("perception_hz<=30 and recorded_speed<=2 required")
        if self.seconds > 3600:
            raise ValueError("first live sessions are limited to one hour")
        if self.auto_bind and self.backend != "mujoco":
            raise ValueError("automatic bind is supported only in MuJoCo")
        if self.backend == "ur10e":
            if not self.options.hardware_enabled or not self.options.mapping_confirmed or not self.options.host:
                raise ValueError("UR live requires explicit execute, host and confirmed mapping")
            calibration_id = self.mapping_manifest.get("source_calibration_id")
            if not isinstance(calibration_id, str) or not calibration_id.strip():
                raise ValueError("UR mapping requires source_calibration_id from the reference check")
            if self.recorded_speed > 1:
                raise ValueError("accelerated recorded input is simulation-only")
        if not Path(self.reference).is_dir():
            raise FileNotFoundError(f"reference episode not found: {self.reference}")


class _InputWriter:
    """Bounded disk worker; camera results never wait for JSON encoding/fsync."""

    def __init__(self, directory):
        self.path = Path(directory) / "hand.jsonl"
        self.channel = queue.Queue(maxsize=64)
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, name="live-input-writer", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with self.path.open("w") as stream:
                while True:
                    item = self.channel.get()
                    if item is None:
                        break
                    stream.write(json.dumps(_safe(item), allow_nan=False) + "\n")
                    stream.flush()
        except Exception as exc:
            self.error = str(exc)

    def put(self, item):
        if self.error:
            raise RuntimeError("live input log failed: " + self.error)
        try:
            self.channel.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self.thread.is_alive():
            try:
                self.channel.put(None, timeout=1)
            except queue.Full:
                self.error = self.error or "input log queue failed to drain"
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = self.error or "input log worker did not stop"


class LiveSession:
    """One causal observation worker and one independent arm-owner process."""

    def __init__(self, config: LiveSessionConfig, *, camera_manager=None,
                 estimator_factory=None, source_factory=None, controller_factory=None):
        config.validate()
        self.config = config
        self.camera_manager = camera_manager
        self._estimator_factory = estimator_factory
        self._source_factory = source_factory
        self._controller_factory = controller_factory
        self.id = uuid.uuid4().hex
        self.directory = Path(config.out_dir) / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._controller = None
        self._hand = None
        self._latencies = deque(maxlen=256)
        self._ages = deque(maxlen=256)
        self._stats = {"estimates": 0, "position_valid": 0, "orientation_valid": 0,
                       "phase": "created", "reason": ""}

    def start(self):
        if self._thread is not None:
            raise RuntimeError("live session cannot be restarted")
        self._thread = threading.Thread(target=self._run, name="live-perception", daemon=True)
        self._thread.start()
        return self.snapshot()

    def _run(self):
        from viki.cameras import CameraFrameSource, RecordedFrameSource
        from viki.perception import LiveHandEstimator
        from viki.replay import LiveController

        cfg = self.config
        estimator = source = writer = controller = None
        owned_manager = None
        self._stats.update(phase="starting")
        metadata = {"schema_version": 1, "session_id": self.id,
                    "config": _safe(asdict(cfg)), "hardware": cfg.backend == "ur10e",
                    "scope": "live hand-only arm retarget; no grasp/obstacle/operator-arm certification"}
        try:
            factory = self._estimator_factory or LiveHandEstimator.from_episode
            estimator = factory(Episode(Path(cfg.reference)), landmark_mode=cfg.landmark_mode,
                                max_age_s=cfg.options.input_ttl_s)
            if self._stop.is_set():
                self._stats["reason"] = "operator_stop"
                return
            expected = cfg.mapping_manifest.get("source_calibration_id")
            if expected and expected != estimator.calibration_id:
                raise ValueError("mapping belongs to another camera calibration")
            metadata["perception"] = estimator.manifest
            self._controller = controller = (self._controller_factory or LiveController)(
                backend=cfg.backend, options=cfg.options, out_dir=self.directory / "executor")
            controller.start()
            if self._stop.is_set():
                self._stats["reason"] = "operator_stop"
                return
            # Create recorded clock only after detector/model/process warm-up.
            if self._source_factory:
                source = self._source_factory(estimator)
            elif cfg.source == "recorded":
                source = RecordedFrameSource(Episode(Path(cfg.reference)), speed=cfg.recorded_speed,
                                             device_ids=estimator.device_ids)
            else:
                manager = self.camera_manager
                if manager is None:
                    from viki.cameras import CameraManager
                    owned_manager = manager = CameraManager()
                    self._start_reference_cameras(manager, estimator.device_ids)
                self._wait_cameras(manager, estimator.device_ids)
                if self._stop.is_set():
                    self._stats["reason"] = "operator_stop"
                    return
                estimator.verify_cameras(manager)
                source = CameraFrameSource(manager, device_ids=estimator.device_ids,
                                           sync_fps=int(cfg.perception_hz), max_age_s=cfg.options.input_ttl_s)
            metadata["source"] = source.source_summary
            self._write_json("manifest.json", metadata)
            writer = _InputWriter(self.directory)
            self._stats.update(phase="observing", calibration_id=estimator.calibration_id)
            self._ready.set()
            end = time.monotonic() + cfg.seconds
            period = 1 / cfg.perception_hz
            auto_requested = False
            auto_sequence = None
            while not self._stop.is_set() and time.monotonic() < end:
                begin = time.monotonic()
                with self._control_lock:
                    control_state = controller.snapshot()
                if control_state.get("state") == "fault" or not control_state.get("running", True):
                    raise RuntimeError(control_state.get("reason", "live executor stopped"))
                if (cfg.auto_bind and not auto_requested and auto_sequence is not None
                        and control_state.get("sequence", -1) >= auto_sequence):
                    with self._control_lock:
                        controller.bind()
                    auto_requested = True
                if source.exhausted:
                    self._stats["reason"] = "source_exhausted"
                    break
                group = source.poll()
                if group is not None:
                    estimate = estimator.process(group)
                    if estimate is not None:
                        target = HumanTarget(
                            estimate.sequence, estimate.source_time_us,
                            estimate.source_time_us + int(cfg.options.input_ttl_s * 1e6),
                            estimate.position, estimate.rotation, estimate.position_valid,
                            estimate.orientation_valid, estimate.calibration_id,
                        )
                        with self._control_lock:
                            controller.submit(target)
                        hand = _safe(asdict(estimate))
                        hand["position_m"] = hand["position"]
                        with self._lock:
                            self._hand = hand
                            self._stats["estimates"] += 1
                            self._stats["position_valid"] += int(estimate.position_valid)
                            self._stats["orientation_valid"] += int(estimate.orientation_valid)
                            self._latencies.append((time.monotonic() - begin) * 1000)
                            self._ages.append((time.monotonic_ns() // 1000 - estimate.source_time_us) / 1000)
                        writer.put(hand)
                        if (cfg.auto_bind and not auto_requested and auto_sequence is None
                                and estimate.position_valid
                                and (estimate.orientation_valid or not cfg.options.orientation)):
                            # Wait for the separate input queue to be consumed before
                            # binding. Request once; never reacquire after input loss.
                            auto_sequence = estimate.sequence
                self._stop.wait(max(0., period - (time.monotonic() - begin)))
            if not self._stats["reason"]:
                self._stats["reason"] = "operator_stop" if self._stop.is_set() else "duration_limit"
        except Exception as exc:
            if (self._stop.is_set() and isinstance(exc, RuntimeError)
                    and str(exc) in ("operator_stop", "session_stop_requested",
                                    "a live controller cannot be restarted; create a new session")):
                self._stats["reason"] = "operator_stop"
            else:
                self._stats.update(phase="fault", reason=str(exc))
        finally:
            self._ready.set()
            def cleanup(action):
                try:
                    action()
                except Exception as exc:
                    self._stats.update(phase="fault", reason=f"cleanup: {exc}")
            def stop_controller():
                with self._control_lock:
                    final = controller.stop(timeout=5)
                metadata["executor"] = final
                if final.get("state") == "fault":
                    self._stats.update(phase="fault", reason=final.get("reason", "executor fault"))
                if final.get("stop_pending"):
                    self._stats.update(phase="fault", reason="executor stop pending")
            if controller is not None:
                cleanup(stop_controller)
            if source is not None:
                cleanup(source.close)
                self._stats["source_dropped_groups"] = getattr(source, "dropped_groups", 0)
            if estimator is not None:
                cleanup(estimator.close)
            if owned_manager is not None:
                cleanup(owned_manager.stop_all)
            if writer is not None:
                cleanup(writer.close)
                self._stats.update(input_log_dropped=writer.dropped, input_log_error=writer.error)
                if writer.error:
                    self._stats.update(phase="fault", reason="input log failed: " + writer.error)
            if self._stats["phase"] != "fault":
                self._stats["phase"] = "stopped"
            metadata["summary"] = self.snapshot()
            metadata["summary"]["running"] = False
            self._write_json("session.json", metadata)

    def _start_reference_cameras(self, manager, devices):
        from .sources import start_reference_cameras
        start_reference_cameras(self.config.reference, manager, devices)

    def _wait_cameras(self, manager, devices):
        from .sources import wait_reference_cameras
        if not self._stop.is_set():
            wait_reference_cameras(manager, devices, self._stop)

    def _write_json(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(_safe(value), indent=2, allow_nan=False))
        temporary.replace(path)

    def bind(self):
        if not self._ready.is_set() or self._stats["phase"] != "observing":
            raise RuntimeError("live session is not ready")
        with self._control_lock:
            return self._controller.bind()

    def clutch(self):
        if self._controller is None:
            raise RuntimeError("live controller has not started")
        with self._control_lock:
            return self._controller.clutch()

    def snapshot(self):
        with self._lock:
            result = {"session_id": self.id, "running": bool(self._thread and self._thread.is_alive()),
                      "reference": self.config.reference, "source": self.config.source,
                      "backend": self.config.backend, "directory": str(self.directory),
                      "hand": self._hand, **self._stats}
            if self._latencies:
                result["processing_ms"] = {"median": float(np.median(self._latencies)),
                                           "p95": float(np.percentile(self._latencies, 95))}
                result["observation_age_ms"] = {"median": float(np.median(self._ages)),
                                                "p95": float(np.percentile(self._ages, 95))}
        if self._controller is not None:
            with self._control_lock:
                result["controller"] = self._controller.snapshot()
        return _safe(result)

    def stop(self, timeout=10):
        self._stop.set()
        if self._controller is not None:
            with self._control_lock:
                self._controller.stop(timeout=1)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        return self.snapshot()


class LiveManager:
    """Application-lifespan owner, reusing its existing camera manager."""

    def __init__(self, camera_manager=None):
        self.camera_manager = camera_manager
        self.session = None
        self._lock = threading.RLock()

    def start(self, config: LiveSessionConfig):
        with self._lock:
            if self.session is not None:
                status = self.session.snapshot()
                if status["running"] or status.get("controller", {}).get("running"):
                    raise RuntimeError("another live session is still running")
            self.session = LiveSession(config, camera_manager=self.camera_manager)
            return self.session.start()

    def snapshot(self):
        with self._lock:
            return self.session.snapshot() if self.session is not None else {"running": False, "phase": "idle", "hand": None}

    def bind(self):
        with self._lock:
            if self.session is None:
                raise RuntimeError("no live session")
            return self.session.bind()

    def clutch(self):
        with self._lock:
            if self.session is None:
                raise RuntimeError("no live session")
            return self.session.clutch()

    def stop(self):
        with self._lock:
            return self.session.stop() if self.session is not None else self.snapshot()

    close = stop
