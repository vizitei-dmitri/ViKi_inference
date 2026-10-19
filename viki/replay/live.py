"""Bounded live arm control, isolated from cameras, ML inference and rendering.

The spawned process owns the robot interfaces and the persistent IK model.
Human inputs are observations with their original timestamps, never queued robot
waypoints. Following requires an explicit bind, and losing input clears that bind.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time
import uuid

import numpy as np

from viki.contracts import HumanTarget, LiveControlOptions


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _latest(channel, value):
    """A single replaceable message, with no wait in a servo period."""
    try:
        channel.put_nowait(value)
        return True
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            # A multiprocessing feeder may not have published its preceding
            # item yet. Drop this observation instead of waiting in a cycle.
            return False
        try:
            channel.put_nowait(value)
            return True
        except queue.Full:
            return False


class _TargetGate:
    """Validate input chronology; arrival never extends source validity."""

    def __init__(self, options):
        self.options = options
        self.target = None
        self.sequence = -1
        self.source_time_us = -1

    def accept(self, target, now_us):
        if not isinstance(target, HumanTarget):
            raise ValueError("expected a HumanTarget")
        if target.sequence <= self.sequence or target.source_time_us < self.source_time_us:
            return False, "out_of_order_input"
        if (target.source_time_us < 0 or target.source_time_us > now_us + 50_000
                or target.valid_until_us <= target.source_time_us
                or not isinstance(target.calibration_id, str) or not target.calibration_id):
            raise ValueError("invalid source time or calibration identity")
        position = np.asarray(target.position, dtype=float)
        rotation = np.asarray(target.rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("invalid human pose shape")
        if target.position_valid and not np.isfinite(position).all():
            raise ValueError("non-finite human position")
        if target.orientation_valid and (not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError("invalid human orientation")
        ttl_limit = target.source_time_us + int(self.options.input_ttl_s * 1e6)
        self.target = replace(target, position=position.copy(), rotation=rotation.copy(),
                              valid_until_us=min(target.valid_until_us, ttl_limit))
        self.sequence = target.sequence
        self.source_time_us = target.source_time_us
        return True, ""

    def reason(self, now_us, calibration_id=None):
        target = self.target
        if target is None:
            return "no_human_input"
        if now_us >= target.valid_until_us:
            return "stale_human_input"
        if not target.position_valid:
            return "invalid_human_position"
        if self.options.orientation and not target.orientation_valid:
            return "invalid_human_orientation"
        if calibration_id is not None and target.calibration_id != calibration_id:
            return "calibration_changed"
        return ""


class _LiveCore:
    """Small state machine, separate from transport and physical backends."""

    def __init__(self, options, retargeter, q):
        self.options = options
        self.ik = retargeter
        self.inputs = _TargetGate(options)
        self.state = "unbound"
        self.reason = "operator_bind_required"
        self.command = np.asarray(q, dtype=float).copy()
        self.velocity = np.zeros(6)
        self.binding = None
        self.last_result = {}

    def bind(self, q, qd, now_us):
        if self.state != "unbound":
            raise RuntimeError("bind requires unbound and stationary state")
        reason = self.inputs.reason(now_us)
        if reason:
            raise ValueError(reason)
        if np.max(np.abs(qd)) > 0.01:
            raise ValueError("robot_must_be_stationary")
        self.ik.model_validate(q)
        position, rotation = self.ik.pose(q)
        target = self.inputs.target
        self.command = np.asarray(q).copy()
        self.velocity = np.zeros(6)
        self.binding = {
            "human_position": target.position.copy(),
            "human_rotation": target.rotation.copy(),
            "robot_position": np.asarray(position).copy(),
            "robot_rotation": np.asarray(rotation).copy(),
            "calibration_id": target.calibration_id,
        }
        self.state, self.reason = "active", ""

    def clutch(self, reason="operator_clutch"):
        self.binding = None
        if self.state in ("active", "braking"):
            self.state, self.reason = "braking", reason
        elif self.state != "fault":
            self.state, self.reason = "unbound", reason

    def step(self, measured, now_us):
        dt = 1 / self.options.frequency
        if self.state == "active":
            reason = self.inputs.reason(now_us, self.binding["calibration_id"])
            if reason:
                self.clutch(reason)
            else:
                target = self.inputs.target
                rotation_map = np.asarray(self.options.rotation_robot_rig)
                delta = self.options.scale * rotation_map @ (target.position - self.binding["human_position"])
                # Do not accumulate distant requests or chase them later. A
                # workspace excursion clears the binding and requires rebind.
                if np.linalg.norm(delta) > self.options.max_displacement_m:
                    self.clutch("human_workspace_limit")
                else:
                    rotation = self.binding["robot_rotation"]
                    if self.options.orientation:
                        relative = target.rotation @ self.binding["human_rotation"].T
                        rotation = rotation_map @ relative @ rotation_map.T @ rotation
                    try:
                        result = self.ik.step(measured, self.command, self.velocity,
                            self.binding["robot_position"] + delta, rotation, dt)
                    except (ValueError, RuntimeError) as exc:
                        self.clutch("ik_rejected: " + str(exc))
                    else:
                        return self._result(result)
        if self.state == "braking":
            result = self._result(self.ik.brake(self.command, self.velocity, dt))
            if np.max(np.abs(self.velocity)) <= 1e-9:
                self.state = "unbound"
            return result
        return {"q": self.command.copy(), "velocity": self.velocity.copy()}

    def _result(self, result):
        command = np.asarray(result["q"], dtype=float)
        velocity = np.asarray(result["velocity"], dtype=float)
        dt = 1 / self.options.frequency
        if (command.shape != (6,) or velocity.shape != (6,)
                or not np.isfinite(command).all() or not np.isfinite(velocity).all()
                or np.max(np.abs(velocity)) > self.options.max_velocity + 1e-6
                or np.max(np.abs(velocity - self.velocity)) / dt > self.options.max_acceleration + 1e-5
                or not np.allclose(command - self.command, velocity * dt, atol=1e-8)):
            raise RuntimeError("IK emitted an invalid or unbounded command")
        self.command, self.velocity = command.copy(), velocity.copy()
        self.last_result = result
        return result


class _MujocoBackend:
    def __init__(self, options):
        import mujoco
        from scipy.spatial.transform import Rotation
        from viki.retarget import normalize_robot

        self.mj, self.options = mujoco, options
        path = Path(options.model_path)
        source_path = path.parent / "source.json"
        source = {}
        if source_path.exists():
            source = json.loads(source_path.read_text())
            directory = path.parent.resolve()
            for name, digest in source.get("sha256", {}).items():
                asset = (directory / name).resolve()
                if not asset.is_relative_to(directory) or hashlib.sha256(asset.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"MuJoCo model cache hash mismatch: {name}")
        specification = mujoco.MjSpec.from_file(str(path))
        for body in specification.bodies:
            body.gravcomp = 1
        self.model = specification.compile()
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        base_rotation = None
        if base >= 0:
            base_rotation = self.model.body_quat[base].copy().tolist()
            # Menagerie's example scene rotates the whole arm by pi about Z.
            # Live poses and the configured plane are in the URDF robot-base
            # frame, so remove the example-scene placement before simulation.
            self.model.body_quat[base] = [1., 0., 0., 0.]
            self.model.body_pos[base] = [0., 0., 0.]
        if self.model.nq != 6 or self.model.nv != 6 or self.model.nu != 6:
            raise ValueError("live MuJoCo requires a bare six-joint UR10e")
        names = normalize_robot("ur10e").joint_names
        joints = [self.model.joint(name).id for name in names]
        self.addresses = self.model.jnt_qposadr[joints]
        self.dofs = self.model.jnt_dofadr[joints]
        self.actuators = []
        for joint in joints:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError("each live arm joint needs one position actuator")
            self.actuators.append(int(matches[0]))
        self.ranges = self.model.jnt_range[joints]
        floor = self.model.geom("floor").id
        normal = np.asarray(options.floor_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        # n.p >= offset describes the free half-space in robot coordinates.
        self.model.geom_pos[floor] = normal * options.floor_offset_m
        rotation, _ = Rotation.align_vectors([normal], [[0, 0, 1]])
        self.model.geom_quat[floor] = rotation.as_quat()[[3, 0, 1, 2]]
        self.model.opt.timestep = 0.002
        # UR's joint controller compensates gravity. Give the simulated bare
        # arm the same property, rather than binding to a sagging position servo.
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.addresses] = options.initial_q
        self.data.ctrl[self.actuators] = options.initial_q
        mujoco.mj_forward(self.model, self.data)
        self.identity = {"model": "UR10e", "backend": "mujoco", "version": mujoco.__version__,
            "model_xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_source_commit": source.get("commit"), "gravity_compensation": True,
            "original_base_quaternion_wxyz": base_rotation, "frame": "robot_base",
            "scope": "bare arm and configured floor"}

    def read(self):
        q = self.data.qpos[self.addresses].copy()
        qd = self.data.qvel[self.dofs].copy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError("non-finite MuJoCo state")
        for contact in self.data.contact:
            if contact.dist < -1e-5:
                raise RuntimeError("MuJoCo penetrating contact")
        return {"q": q, "qd": qd, "timestamp_s": float(self.data.time), "safety_mode": 1}

    def bind(self, q):
        self._check_limits(q)

    def _check_limits(self, q):
        if np.any(q < self.ranges[:, 0]) or np.any(q > self.ranges[:, 1]):
            raise RuntimeError("MuJoCo joint range violation")

    def command(self, q):
        self._check_limits(q)
        self.data.ctrl[self.actuators] = q

    def advance(self, period, token=None):
        steps = round(period / self.model.opt.timestep)
        if steps < 1 or not np.isclose(steps * self.model.opt.timestep, period, atol=1e-10):
            raise RuntimeError("servo period must be an integer number of MuJoCo steps")
        for _ in range(steps):
            self.mj.mj_step(self.model, self.data)

    def begin_period(self):
        return None

    def release(self):
        return []

    def close(self):
        return []


class _URBackend:
    """RTDE control is uploaded only when an operator explicitly binds."""

    def __init__(self, options):
        from viki.replay import ur

        if not options.hardware_enabled or not options.mapping_confirmed or not options.host:
            raise ValueError("UR live needs explicit hardware enable, confirmed rig mapping and host")
        self.options, self.ur = options, ur
        self.control = None
        self.receive = None
        self.cleanup_errors = []
        self.lock = ur._robot_lock(options.host)
        self.lock.__enter__()
        self.last_timestamp = None
        self.fresh_at = time.monotonic()
        try:
            self.identity = ur._dashboard_info(options.host)
            self.receive = ur._receive(options.host, options.frequency)
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        ur, receive = self.ur, self.receive
        ur._ready(receive)
        timestamp = float(receive.getTimestamp())
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp < self.last_timestamp):
            raise RuntimeError("invalid or reversed RTDE timestamp")
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp, self.fresh_at = timestamp, time.monotonic()
        if time.monotonic() - self.fresh_at > 0.1:
            raise RuntimeError("stale RTDE state")
        if self.control is not None and (not self.control.isConnected() or not self.control.isProgramRunning()):
            raise RuntimeError("UR control program stopped or disconnected")
        return {"q": ur._vector(receive.getActualQ(), "actual_q"),
                "qd": ur._vector(receive.getActualQd(), "actual_qd"),
                "tcp_pose_controller": ur._vector(receive.getActualTCPPose(), "actual_tcp_pose"),
                "timestamp_s": timestamp, "safety_mode": int(receive.getSafetyMode())}

    def bind(self, q):
        if self.control is not None:
            raise RuntimeError("UR control already active")
        observed = self.read()
        if self.receive.getRuntimeState() not in (1, 4):
            raise RuntimeError("stop the existing robot program before live bind")
        if np.max(np.abs(observed["qd"])) > 0.01:
            raise RuntimeError("UR must be stationary before live bind")
        if np.max(np.abs(observed["q"] - q)) > 0.01:
            raise RuntimeError("UR moved during live bind")
        self.control = self.ur._control(self.options.host, self.options.frequency)
        try:
            if not self.control.isConnected():
                raise RuntimeError("UR control disconnected")
            if not self.control.isJointsWithinSafetyLimits(q.tolist()):
                raise RuntimeError("bind pose violates installed UR safety limits")
            if not self.control.setWatchdog(10):
                raise RuntimeError("UR watchdog initialization failed")
            after = self.read()
            if np.max(np.abs(after["q"] - q)) > 0.01:
                raise RuntimeError("UR moved during control upload")
        except BaseException:
            self.release()
            raise

    def begin_period(self):
        return self.control.initPeriod() if self.control is not None else None

    def command(self, q):
        if self.control is None:
            raise RuntimeError("UR command without an operator bind")
        if not self.control.isJointsWithinSafetyLimits(q.tolist()):
            raise RuntimeError("command violates installed UR safety limits")
        if not self.control.kickWatchdog():
            raise RuntimeError("UR watchdog kick failed")
        if not self.control.servoJ(q.tolist(), 0, 0, 1 / self.options.frequency, 0.1, 300):
            raise RuntimeError("UR servoJ rejected")

    def advance(self, period, token=None):
        if token is not None and self.control is not None:
            self.control.waitPeriod(token)

    def release(self):
        errors = []
        control, self.control = self.control, None
        if control is not None:
            for name in ("servoStop", "stopScript", "disconnect"):
                try:
                    result = (control.servoStop(self.options.max_acceleration) if name == "servoStop"
                              else getattr(control, name)())
                    if result is False:
                        errors.append(f"{name} returned false")
                except BaseException as exc:
                    errors.append(f"{name}: {exc}")
        self.cleanup_errors.extend(errors)
        return errors

    def close(self):
        self.release()
        errors = self.cleanup_errors.copy()
        if self.receive is not None:
            try:
                self.receive.disconnect()
            except BaseException as exc:
                errors.append(f"receive.disconnect: {exc}")
            self.receive = None
        if self.lock is not None:
            try:
                self.lock.__exit__(None, None, None)
            except BaseException as exc:
                errors.append(f"robot lock release: {exc}")
            self.lock = None
        return errors


class _LiveLog:
    """A bounded writer process also keeps HDF5 Python work off the servo GIL."""

    VECTORS = {"q_observed": 6, "qd_observed": 6, "q_commanded": 6, "velocity_commanded": 6,
               "human_position": 3, "target_position": 3, "tcp_position": 3,
               "tcp_pose_controller": 6, "tcp_rotation": 9}
    SCALARS = ("host_monotonic_us", "source_time_us", "sequence", "controller_timestamp_s",
               "residual_rad", "solver_ms", "cycle_compute_ms", "safety_mode", "command_accepted",
               "command_host_monotonic_us", "command_completed_monotonic_us")

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        context = mp.get_context("spawn")
        self.queue = context.Queue(maxsize=512)
        self.results = context.Queue(maxsize=2)
        self.error = None
        self.rows = self.events = 0
        self.process = context.Process(target=self._write,
            args=(self.directory, metadata, self.queue, self.results), name="live-hdf-log", daemon=False)
        self.process.start()
        try:
            started = self.results.get(timeout=15)
        except queue.Empty:
            self.process.terminate()
            self.process.join(5)
            raise RuntimeError("live logger failed to initialize") from None
        if not started.get("ready"):
            self.process.join(5)
            raise RuntimeError("live logger failed: " + started.get("error", "startup failure"))

    def _collect(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return
        self.error = result.get("error")
        self.rows, self.events = result.get("rows", 0), result.get("events", 0)

    def check(self):
        self._collect()
        if self.error:
            raise RuntimeError("live logger failed: " + self.error)
        if not self.process.is_alive():
            raise RuntimeError("live logger exited before session close")

    def put(self, kind, payload):
        self.check()
        try:
            self.queue.put_nowait((kind, payload))
        except queue.Full:
            raise RuntimeError("live log queue overflow") from None

    @staticmethod
    def _write(directory, metadata, messages, results):
        # Ctrl-C belongs to the arm owner, which brakes and then closes this
        # queue. Interrupting the writer itself would discard the partial log.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        rows = event_count = 0
        try:
            import h5py
            with h5py.File(directory / "execution.h5", "w") as file:
                file.attrs["schema_version"] = 1
                file.attrs["metadata_json"] = json.dumps(_json_safe(metadata), allow_nan=False)
                samples = file.create_group("samples")
                dtype = h5py.string_dtype("utf-8")
                datasets = {}
                for key in _LiveLog.SCALARS:
                    datasets[key] = samples.create_dataset(key, (0,), maxshape=(None,), chunks=(128,), dtype="f8")
                for key, width in _LiveLog.VECTORS.items():
                    datasets[key] = samples.create_dataset(key, (0, width), maxshape=(None, width), chunks=(128, width), dtype="f8")
                for key in ("state", "cause", "margins_json"):
                    datasets[key] = samples.create_dataset(key, (0,), maxshape=(None,), chunks=(128,), dtype=dtype)
                events = file.create_dataset("events_json", (0,), maxshape=(None,), chunks=(64,), dtype=dtype)
                results.put({"ready": True})
                pending, pending_events = [], []
                done = False
                while not done:
                    try:
                        item = messages.get(timeout=0.2)
                    except queue.Empty:
                        item = ("flush", None)
                    kind, payload = item
                    if kind == "sample": pending.append(payload)
                    if kind == "event": pending_events.append(payload)
                    if kind == "close": done = True
                    if len(pending) >= 128 or kind in ("flush", "close"):
                        if pending:
                            start, end = rows, rows + len(pending)
                            for key, dataset in datasets.items():
                                dataset.resize((end, *dataset.shape[1:]))
                                if key in _LiveLog.VECTORS:
                                    dataset[start:end] = np.asarray([row.get(key, np.full(_LiveLog.VECTORS[key], np.nan)) for row in pending])
                                elif key in _LiveLog.SCALARS:
                                    dataset[start:end] = [row.get(key, np.nan) for row in pending]
                                elif key == "margins_json":
                                    dataset[start:end] = [json.dumps(_json_safe(row.get("margins", {})), allow_nan=False) for row in pending]
                                else:
                                    dataset[start:end] = [str(row.get(key, "")) for row in pending]
                            rows, pending = end, []
                        if pending_events:
                            end = event_count + len(pending_events)
                            events.resize((end,))
                            events[event_count:end] = [json.dumps(_json_safe(item), allow_nan=False) for item in pending_events]
                            event_count, pending_events = end, []
                        file.flush()
                file.attrs["closed_cleanly"] = True
            results.put({"rows": rows, "events": event_count, "error": None})
        except BaseException as exc:
            results.put({"rows": rows, "events": event_count, "error": str(exc)})

    def close(self, report):
        try:
            self.queue.put(("close", None), timeout=5)
        except queue.Full:
            self.error = self.error or "log queue did not drain"
        self.process.join(10)
        self._collect()
        if self.process.is_alive():
            self.error = self.error or "log writer did not close"
            self.process.terminate()
            self.process.join(5)
        # If a writer faulted, its queue may still hold pending messages. Do not
        # block process cleanup joining a feeder whose consumer has exited.
        self.queue.cancel_join_thread()
        self.queue.close()
        self.results.cancel_join_thread()
        self.results.close()
        report.update(log_rows=self.rows, log_events=self.events, log_error=self.error,
                      execution_log=str(self.directory / "execution.h5"))
        if self.error:
            report.update(state="fault", reason="logging_failure: " + self.error)
        temporary = self.directory / ".session.json.tmp"
        temporary.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False))
        os.replace(temporary, self.directory / "session.json")


def _worker(backend_name, options, directory, targets, controls, snapshots, stop_event,
            backend_factory=None, retargeter_factory=None):
    """Process entrypoint. Test factories never cross the public hardware path."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop_event.set())
    backend = core = log = limiter = None
    state = {"running": True, "state": "starting", "reason": "", "pid": os.getpid(),
             "backend": backend_name, "run_dir": str(directory), "hardware": backend_name == "ur10e",
             "frequency_hz": options.frequency, "servo_dt_s": 1 / options.frequency}
    _latest(snapshots, state.copy())
    request_id = None
    request_result = None
    stopping = False
    gc_was_enabled = gc.isenabled()
    last_snapshot = 0.0
    max_compute_ms = 0.0
    measured = None
    try:
        log = _LiveLog(directory, {"backend": backend_name, "options": asdict(options),
                "scope": "live arm-only relative human following; no gripper control"})
        if retargeter_factory is None:
            from viki.retarget import OnlineRetargeter
            retargeter_factory = OnlineRetargeter
        backend_factory = backend_factory or {"mujoco": _MujocoBackend, "ur10e": _URBackend}[backend_name]
        backend = backend_factory(options)
        from threadpoolctl import threadpool_limits

        # Native six-variable solves do not benefit from the offline image's
        # several BLAS workers. Apply this once in the isolated owner process.
        limiter = threadpool_limits(limits=1)
        state["identity"] = backend.identity
        measured = backend.read()
        # initial_q describes a simulation start. The nominal IK constructor
        # must validate the actual observed pose for hardware, never veto a
        # valid physical pose because a hypothetical simulation pose intersects
        # the configured plane. The backend remains receive-only until bind.
        ik_options = (replace(options, initial_q=tuple(measured["q"]))
                      if backend_name == "ur10e" else options)
        ik = retargeter_factory(ik_options)
        state["kinematic_model"] = getattr(ik, "manifest", {})
        ik.model_validate(measured["q"])
        core = _LiveCore(options, ik, measured["q"])
        position, rotation = ik.pose(measured["q"])
        # Import/initialize the solver and native kernels before any bind or
        # controller upload. These outputs are intentionally never commanded.
        ik.step(measured["q"], measured["q"], np.zeros(6), position, rotation, 1 / options.frequency)
        # Temporary vectors/messages are freed by reference counting. Cyclic
        # collection of the whole imported stack can pause an otherwise small
        # native solve for tens of milliseconds, so do it after arm cleanup.
        gc.disable()
        state["cyclic_gc_during_control"] = False
        log.put("event", {"kind": "ready", "host_monotonic_us": time.monotonic_ns() // 1000,
                          "identity": backend.identity})
        period = 1 / options.frequency
        deadline = time.monotonic()
        while True:
            cycle_start = time.monotonic()
            if core.state in ("active", "braking") and cycle_start - deadline > options.max_lateness:
                raise RuntimeError("live control deadline missed; no catch-up")
            deadline = cycle_start + period
            token = backend.begin_period()
            now_us = time.monotonic_ns() // 1000
            measured = backend.read()
            while True:
                try:
                    target = targets.get_nowait()
                except queue.Empty:
                    break
                try:
                    accepted, cause = core.inputs.accept(target, now_us)
                except (ValueError, TypeError) as exc:
                    accepted, cause = False, "invalid_input: " + str(exc)
                    core.clutch(cause)
                log.put("event", {"kind": "human_input", "accepted": accepted, "cause": cause,
                    "host_monotonic_us": now_us, "input": asdict(target) if isinstance(target, HumanTarget) else str(type(target))})
            was_bound = core.state in ("active", "braking")
            while True:
                try:
                    request_id, command = controls.get_nowait()
                except queue.Empty:
                    break
                try:
                    if command == "bind":
                        if stopping or stop_event.is_set():
                            raise RuntimeError("session_stop_requested")
                        core.bind(measured["q"], measured["qd"], now_us)
                        try:
                            backend.bind(measured["q"])
                        except BaseException:
                            core.clutch("backend_bind_rejected")
                            raise
                    elif command == "clutch":
                        core.clutch()
                    else:
                        raise ValueError("unknown live control request")
                    request_result = {"accepted": True, "command": command}
                except (ValueError, RuntimeError) as exc:
                    request_result = {"accepted": False, "command": command, "reason": str(exc)}
                log.put("event", {"kind": "operator_request", "host_monotonic_us": now_us,
                                  "request_id": request_id, "binding": core.binding, **request_result})
                # A bind/control script upload has no servo budget. Start a new
                # epoch after it, without replaying elapsed human observations.
                cycle_start = time.monotonic()
                deadline = cycle_start + period
                token = backend.begin_period()
                now_us = time.monotonic_ns() // 1000
                measured = backend.read()
            if stop_event.is_set():
                stopping = True
                core.clutch("operator_stop")
            active = core.state in ("active", "braking")
            residual = float(np.max(np.abs(measured["q"] - core.command)))
            if active and residual > options.tracking_tolerance:
                raise RuntimeError("live joint tracking fault")
            if active and np.max(np.abs(measured["qd"])) > options.max_velocity + 0.025:
                raise RuntimeError("measured live joint speed exceeded bound")
            solve_start = time.monotonic()
            if active:
                result = core.step(measured["q"], now_us)
            else:
                # An unbound robot may be repositioned on its pendant. It is
                # never pulled back to an old reference at the next bind.
                core.command = measured["q"].copy()
                result = {"q": core.command, "velocity": np.zeros(6)}
            solver_ms = (time.monotonic() - solve_start) * 1000
            compute_ms = (time.monotonic() - cycle_start) * 1000
            max_compute_ms = max(max_compute_ms, compute_ms)
            state.update(solver_ms=solver_ms, cycle_compute_ms=compute_ms,
                         max_cycle_compute_ms=max_compute_ms, request_id=request_id,
                         request_result=request_result)
            if active and compute_ms > period * 1000:
                raise RuntimeError("live computation exceeded servo period; no catch-up")
            command_accepted = False
            command_time_us = completed_time_us = float("nan")
            if active:
                command_time_us = time.monotonic_ns() // 1000
                try:
                    backend.command(core.command)
                    command_accepted = True
                    completed_time_us = time.monotonic_ns() // 1000
                except BaseException as exc:
                    log.put("event", {"kind": "command_rejected", "host_monotonic_us": now_us,
                        "command_host_monotonic_us": command_time_us,
                        "q_commanded": core.command.copy(), "q_observed": measured["q"].copy(), "cause": str(exc)})
                    raise
            tcp_position, tcp_rotation = ik.pose(measured["q"])
            target = core.inputs.target
            binding = core.binding
            target_position = tcp_position
            if target is not None and binding is not None:
                target_position = binding["robot_position"] + options.scale * np.asarray(options.rotation_robot_rig) @ (target.position - binding["human_position"])
            row = {"host_monotonic_us": now_us, "source_time_us": target.source_time_us if target else -1,
                "sequence": target.sequence if target else -1, "q_observed": measured["q"].copy(),
                "qd_observed": measured["qd"].copy(), "q_commanded": core.command.copy(),
                "velocity_commanded": core.velocity.copy(), "controller_timestamp_s": measured["timestamp_s"],
                "tcp_pose_controller": measured.get("tcp_pose_controller", np.full(6, np.nan)),
                "tcp_position": tcp_position, "tcp_rotation": np.asarray(tcp_rotation).reshape(9),
                "human_position": target.position.copy() if target else np.full(3, np.nan),
                "target_position": target_position, "residual_rad": residual, "solver_ms": solver_ms,
                "cycle_compute_ms": compute_ms, "safety_mode": measured["safety_mode"],
                "state": core.state, "cause": core.reason, "margins": result.get("margins", {}),
                "command_accepted": command_accepted,
                "command_host_monotonic_us": command_time_us,
                "command_completed_monotonic_us": completed_time_us}
            log.put("sample", row)
            state.update(state=core.state, reason=core.reason, q=measured["q"].tolist(),
                qd=measured["qd"].tolist(), command_q=core.command.tolist(), tcp_position=np.asarray(tcp_position).tolist(),
                tcp_rotation=np.asarray(tcp_rotation).tolist(), sequence=target.sequence if target else -1,
                source_time_us=target.source_time_us if target else None, host_monotonic_us=now_us,
                solver_ms=solver_ms, cycle_compute_ms=compute_ms, max_cycle_compute_ms=max_compute_ms,
                tracking_error_rad=residual, margins=result.get("margins", {}), request_id=request_id,
                request_result=request_result, stopping=stopping,
                binding_human_position=binding["human_position"] if binding else None,
                binding_robot_position=binding["robot_position"] if binding else None,
                binding_robot_rotation=binding["robot_rotation"] if binding else None)
            if cycle_start - last_snapshot >= 0.1 or stopping:
                if hasattr(ik, "scene"):
                    scene = ik.scene(measured["q"])
                    state.update(robot_points=scene["points"], robot_edges=scene["edges"])
                _latest(snapshots, _json_safe(state.copy()))
                last_snapshot = cycle_start
            elapsed = time.monotonic() - cycle_start
            max_compute_ms = max(max_compute_ms, elapsed * 1000)
            if active and elapsed > period:
                raise RuntimeError("live full cycle exceeded servo period; no catch-up")
            backend.advance(period, token)
            if active and backend_name == "mujoco" and time.monotonic() - cycle_start > period:
                raise RuntimeError("live simulation exceeded servo period; no catch-up")
            if (was_bound or active) and core.state == "unbound" and backend_name == "ur10e":
                errors = backend.release()
                if errors:
                    raise RuntimeError("live controller cleanup failed: " + "; ".join(errors))
            if stopping and core.state == "unbound":
                break
            remainder = deadline - time.monotonic()
            if remainder > 0:
                stop_event.wait(remainder) if core.state == "unbound" else time.sleep(remainder)
    except BaseException as exc:
        state.update(state="fault", reason=str(exc) or type(exc).__name__)
        if log is not None:
            try:
                log.put("event", {"kind": "fault", "host_monotonic_us": time.monotonic_ns() // 1000,
                    "cause": state["reason"], "solver_ms": state.get("solver_ms"),
                    "cycle_compute_ms": state.get("cycle_compute_ms"),
                    "candidate_q": core.command if core else None,
                    "q_observed": measured["q"] if measured else None})
            except BaseException:
                pass
    finally:
        if backend is not None:
            errors = backend.close()
            if errors:
                state.update(state="fault", reason="live cleanup failed: " + "; ".join(errors), cleanup_errors=errors)
        if gc_was_enabled:
            gc.enable()
        if state["state"] != "fault":
            state.update(state="stopped", reason="operator_stop")
        state.update(running=False, ended_monotonic_us=time.monotonic_ns() // 1000)
        if log is not None:
            log.close(state)
        if limiter is not None:
            limiter.restore_original_limits()
        _latest(snapshots, _json_safe(state.copy()))
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class LiveController:
    """Nonblocking façade for the sole servo/IK process owner.

    ``bind`` and ``clutch`` return request IDs. Their accepted/rejected outcome
    appears in ``snapshot``; submitting a pose alone never enables following.
    """

    def __init__(self, *, backend="mujoco", options=None, out_dir="data/live"):
        self.options = options or LiveControlOptions()
        self.options.validate()
        if backend not in ("mujoco", "ur10e"):
            raise ValueError("live backend must be mujoco or ur10e")
        if backend == "ur10e" and (not self.options.hardware_enabled or not self.options.mapping_confirmed or not self.options.host):
            raise ValueError("UR live needs explicit hardware enable, confirmed rig mapping and host")
        self.backend = backend
        self.run_dir = Path(out_dir) / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12])
        self.context = mp.get_context("spawn")
        self.targets = self.context.Queue(maxsize=1)
        self.controls = self.context.Queue(maxsize=16)
        self.snapshots = self.context.Queue(maxsize=1)
        self.stop_event = self.context.Event()
        self.process = None
        self._process_lock = threading.RLock()
        self._queues_closed = False
        self._snapshot = {"running": False, "state": "created", "backend": backend, "run_dir": str(self.run_dir)}

    def start(self, timeout=30):
        with self._process_lock:
            if self.process is not None or self.stop_event.is_set():
                raise RuntimeError("a live controller cannot be restarted; create a new session")
            self.process = self.context.Process(target=_worker,
                args=(self.backend, self.options, self.run_dir, self.targets, self.controls, self.snapshots, self.stop_event),
                name="viki-live-arm", daemon=False)
            self.process.start()
            self._snapshot.update(running=True, state="starting", pid=self.process.pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.get("state") == "unbound":
                return self
            if not snapshot.get("running") or snapshot.get("state") == "fault":
                self.stop()
                raise RuntimeError(snapshot.get("reason", "live process failed to start"))
            time.sleep(0.02)
        self.stop()
        raise RuntimeError("live controller startup timed out")

    def submit(self, target: HumanTarget):
        with self._process_lock:
            if self.process is None or not self.process.is_alive() or self.stop_event.is_set():
                return False
            if not isinstance(target, HumanTarget):
                raise TypeError("submit expects a HumanTarget")
            # Copy now: asynchronous serialization must not observe arrays
            # later mutated by a perception worker.
            return _latest(self.targets, replace(target, position=np.asarray(target.position).copy(),
                                                 rotation=np.asarray(target.rotation).copy()))

    def _request(self, command):
        with self._process_lock:
            if self.process is None or not self.process.is_alive() or self.stop_event.is_set():
                raise RuntimeError("live controller is not running")
            request_id = uuid.uuid4().hex
            try:
                self.controls.put_nowait((request_id, command))
            except queue.Full:
                raise RuntimeError("live operator request queue is full") from None
            return request_id

    def bind(self):
        return self._request("bind")

    def clutch(self):
        return self._request("clutch")

    def snapshot(self):
        with self._process_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        if not self._queues_closed:
            while True:
                try:
                    self._snapshot = self.snapshots.get_nowait()
                except queue.Empty:
                    break
        if self.process is not None and not self.process.is_alive() and self._snapshot.get("running"):
            report = self.run_dir / "session.json"
            if report.exists():
                self._snapshot = json.loads(report.read_text())
            else:
                self._snapshot.update(running=False, state="fault", reason="live process exited without a final report",
                                      exitcode=self.process.exitcode)
        return _json_safe(self._snapshot.copy())

    def stop(self, timeout=5):
        self.stop_event.set()
        with self._process_lock:
            process = self.process
        if process is not None:
            process.join(timeout)
        with self._process_lock:
            result = self._snapshot_locked()
            if process is not None and process.is_alive():
                result["stop_pending"] = True
            elif not self._queues_closed:
                for channel in (self.targets, self.controls, self.snapshots):
                    channel.cancel_join_thread()
                    channel.close()
                self._queues_closed = True
        return result

