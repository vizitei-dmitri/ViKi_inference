"""Live chronology, binding, interruption and log tests without hardware."""

from contextlib import nullcontext
from dataclasses import replace
import json
import os
import queue
import signal
import threading
import time

import h5py
import numpy as np
import pytest

from viki.contracts import HumanTarget, LiveControlOptions, LiveIKTuning
from viki.replay import LiveController
from viki.replay.live import _LiveCore, _LiveLog, _MujocoBackend, _TargetGate, _URBackend, _worker


def target(sequence=1, *, source=1_000_000, position=(0., 0., 0.), calibration="rig-a", valid=True):
    return HumanTarget(sequence, source, source + 250_000, np.array(position), np.eye(3), valid, True, calibration)


class FakeIK:
    manifest = {"bare_arm": True}

    def __init__(self, options):
        self.options = options

    def model_validate(self, q):
        if np.max(np.abs(q)) > 2:
            raise RuntimeError("invalid model state")
        return {"joint": 1., "floor": .1, "collision": .1}

    def pose(self, q):
        return np.asarray(q)[:3].copy(), np.eye(3)

    def step(self, measured, q, velocity, position, rotation, dt):
        desired = np.clip((position - q[:3]) / dt, -.1, .1)
        new = np.r_[desired, np.zeros(3)]
        new = np.clip(new, velocity - self.options.max_acceleration * dt, velocity + self.options.max_acceleration * dt)
        return {"q": q + dt * new, "velocity": new, "margins": self.model_validate(q)}

    def brake(self, q, velocity, dt):
        new = np.sign(velocity) * np.maximum(np.abs(velocity) - self.options.max_acceleration * dt, 0)
        return {"q": q + dt * new, "velocity": new, "margins": self.model_validate(q)}


@pytest.fixture
def core():
    options = LiveControlOptions(frequency=125)
    return _LiveCore(options, FakeIK(options), np.zeros(6))


def test_arrival_and_duplicate_sequence_never_renew_source_expiry():
    gate = _TargetGate(LiveControlOptions())
    original = replace(target(), valid_until_us=9_000_000)
    assert gate.accept(original, 1_000_000) == (True, "")
    assert gate.target.valid_until_us == 1_250_000
    assert gate.accept(replace(original, source_time_us=1_200_000), 1_200_000)[0] is False
    assert gate.reason(1_250_000) == "stale_human_input"


def test_feedforward_is_cleared_on_loss_and_manual_rebind():
    class VelocityIK(FakeIK):
        tuning = LiveIKTuning(feedforward_gain=1.)

        def step(self, measured, q, velocity, position, rotation, dt, **kwargs):
            self.feedforward = kwargs["target_velocity"]
            return super().step(measured, q, velocity, position, rotation, dt)

    options = LiveControlOptions()
    ik = VelocityIK(options)
    core = _LiveCore(options, ik, np.zeros(6))
    q = np.zeros(6)
    core.inputs.accept(target(), 1_000_000)
    core.bind(q, np.zeros(6), 1_000_000)
    q = core.step(q, 1_000_000)["q"]
    core.inputs.accept(target(2, source=1_100_000, position=(.01, 0, 0)), 1_100_000)
    q = core.step(q, 1_100_000)["q"]
    assert ik.feedforward[0] > 0
    q = core.step(q, 1_360_000)["q"]
    assert core.binding is None and core.target_velocity.position is None
    for _ in range(20):
        q = core.step(q, 1_370_000)["q"]
    core.inputs.accept(target(3, source=1_400_000, position=(.1, 0, 0)), 1_400_000)
    core.bind(q, np.zeros(6), 1_400_000)
    core.step(q, 1_400_000)
    np.testing.assert_array_equal(ik.feedforward, 0)


def test_bind_uses_current_fk_and_following_remains_relative(core):
    q = np.array([.4, .2, .1, 0, 0, 0])
    core.inputs.accept(target(position=(2., 3., 4.)), 1_000_000)
    core.bind(q, np.zeros(6), 1_000_000)
    result = core.step(q, 1_010_000)
    np.testing.assert_allclose(result["q"], q)
    core.inputs.accept(target(2, source=1_020_000, position=(2.1, 3., 4.)), 1_020_000)
    result = core.step(q, 1_020_000)
    assert 0 < result["q"][0] - q[0] <= .5 * .008**2 + 1e-10
    assert core.binding["human_position"][0] == 2


@pytest.mark.parametrize("cause", ["stale", "invalid", "calibration", "workspace", "clutch"])
def test_lost_input_brakes_and_requires_a_new_explicit_bind(core, cause):
    core.inputs.accept(target(), 1_000_000)
    core.bind(np.zeros(6), np.zeros(6), 1_000_000)
    core.inputs.accept(target(2, source=1_020_000, position=(.1, 0., 0.)), 1_020_000)
    q = np.zeros(6)
    for _ in range(8):
        q = core.step(q, 1_030_000)["q"]
    old_velocity = core.velocity.copy()
    now = 1_040_000
    if cause == "stale": now = 2_000_000
    elif cause == "invalid": core.inputs.accept(target(3, source=now, valid=False), now)
    elif cause == "calibration": core.inputs.accept(target(3, source=now, calibration="rig-b"), now)
    elif cause == "workspace": core.inputs.accept(target(3, source=now, position=(2., 0., 0.)), now)
    else: core.clutch()
    for _ in range(50):
        result = core.step(q, now)
        assert np.max(np.abs(result["velocity"] - old_velocity)) <= .5 * .008 + 1e-8
        old_velocity, q = result["velocity"].copy(), result["q"].copy()
    assert core.state == "unbound" and core.binding is None
    core.inputs.accept(target(4, source=now + 1_000), now + 1_000)
    assert core.step(q, now + 1_000)["velocity"].max() == 0
    assert core.state == "unbound"


def test_bind_rejects_missing_input_moving_robot_and_invalid_model(core):
    with pytest.raises(ValueError, match="no_human"):
        core.bind(np.zeros(6), np.zeros(6), 1_000_000)
    core.inputs.accept(target(), 1_000_000)
    with pytest.raises(ValueError, match="stationary"):
        core.bind(np.zeros(6), np.ones(6), 1_000_000)
    with pytest.raises(RuntimeError, match="invalid model"):
        core.bind(np.ones(6) * 3, np.zeros(6), 1_000_000)


def test_nonfinite_valid_input_and_malformed_so3_are_rejected():
    gate = _TargetGate(LiveControlOptions())
    with pytest.raises(ValueError, match="non-finite"):
        gate.accept(target(position=(np.nan, 0., 0.)), 1_000_000)
    with pytest.raises(ValueError, match="orientation"):
        gate.accept(replace(target(), rotation=np.zeros((3, 3))), 1_000_000)
    assert gate.target is None


def test_bad_solver_command_is_never_admitted(core):
    with pytest.raises(RuntimeError, match="unbounded"):
        core._result({"q": np.ones(6), "velocity": np.ones(6)})
    assert np.max(np.abs(core.command)) == 0


def test_async_log_is_incremental_and_preserves_unknown_measurements(tmp_path):
    directory = tmp_path / "run"
    log = _LiveLog(directory, {"options": {"test": True}})
    for index in range(260):
        log.put("sample", {"host_monotonic_us": index, "q_commanded": np.ones(6) * index,
                            "state": "active", "margins": {"floor": .1}})
    log.put("event", {"kind": "operator_clutch", "cause": "test", "position": np.zeros(3)})
    report = {"state": "stopped", "reason": "operator_stop"}
    log.close(report)
    assert report["log_rows"] == 260 and report["log_error"] is None
    with h5py.File(directory / "execution.h5") as file:
        assert file.attrs["closed_cleanly"]
        assert file["samples/q_commanded"].shape == (260, 6)
        assert np.isnan(file["samples/q_observed"][:]).all()
        assert json.loads(file["events_json"][0])["kind"] == "operator_clutch"
    assert json.loads((directory / "session.json").read_text())["log_rows"] == 260
    with pytest.raises(FileExistsError):
        _LiveLog(directory, {})


def test_interrupting_terminal_does_not_interrupt_partial_log_flush(tmp_path):
    log = _LiveLog(tmp_path / "run", {})
    log.put("sample", {"state": "active", "q_commanded": np.ones(6)})
    os.kill(log.process.pid, signal.SIGINT)
    log.put("event", {"kind": "operator_stop", "cause": "SIGINT"})
    report = {"state": "stopped"}
    log.close(report)
    assert report["log_rows"] == 1 and report["log_error"] is None


@pytest.mark.parametrize("settings", [{}, {"hardware_enabled": True}, {"hardware_enabled": True, "host": "robot"}])
def test_hardware_enable_requires_all_explicit_gates_before_any_connection(settings, monkeypatch):
    monkeypatch.setattr("viki.replay.ur._dashboard_info", lambda *_: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="explicit"):
        LiveController(backend="ur10e", options=LiveControlOptions(**settings))
    with pytest.raises(ValueError, match="explicit"):
        _URBackend(LiveControlOptions(**settings))


class FakeReceive:
    def __init__(self):
        self.q = np.zeros(6)
        self.qd = np.zeros(6)
        self.runtime = 1
        self.safety = 1
        self.timestamp = 1.
        self.connected = True

    def isConnected(self): return self.connected
    def getSafetyMode(self): return self.safety
    def getRobotMode(self): return 7
    def getRuntimeState(self): return self.runtime
    def getTimestamp(self): return self.timestamp
    def getActualQ(self): return self.q
    def getActualQd(self): return self.qd
    def getActualTCPPose(self): return np.zeros(6)
    def disconnect(self): self.connected = False


class FakeControl:
    def __init__(self):
        self.calls = []
        self.rejected_watchdog = False
        self.failed_stop = False

    def isConnected(self): return True
    def isProgramRunning(self): return True
    def isJointsWithinSafetyLimits(self, q): return True
    def setWatchdog(self, frequency):
        self.calls.append(("watchdog", frequency))
        return not self.rejected_watchdog
    def kickWatchdog(self): return True
    def servoJ(self, *args):
        self.calls.append(("servo", args[0]))
        return True
    def initPeriod(self): return 0
    def waitPeriod(self, period): pass
    def servoStop(self, acceleration):
        self.calls.append("servoStop")
        if self.failed_stop: raise RuntimeError("stop failure")
    def stopScript(self): self.calls.append("stopScript")
    def disconnect(self): self.calls.append("disconnect")


@pytest.fixture
def ur_fake(monkeypatch):
    receive, control, uploaded = FakeReceive(), FakeControl(), []
    monkeypatch.setattr("viki.replay.ur._robot_lock", lambda *_: nullcontext())
    monkeypatch.setattr("viki.replay.ur._dashboard_info", lambda *_: {"model": "UR10e"})
    monkeypatch.setattr("viki.replay.ur._receive", lambda *_: receive)
    def upload(*_):
        uploaded.append(True)
        return control
    monkeypatch.setattr("viki.replay.ur._control", upload)
    options = LiveControlOptions(host="robot", hardware_enabled=True, mapping_confirmed=True)
    backend = _URBackend(options)
    return backend, receive, control, uploaded


def test_ur_control_upload_requires_bind_and_stationary_checked_state(ur_fake):
    backend, receive, control, uploaded = ur_fake
    assert not uploaded
    receive.qd[:] = .2
    with pytest.raises(RuntimeError, match="stationary"):
        backend.bind(np.zeros(6))
    assert not uploaded
    receive.qd[:] = 0
    receive.runtime = 2
    with pytest.raises(RuntimeError, match="existing"):
        backend.bind(np.zeros(6))
    assert not uploaded
    receive.runtime = 1
    backend.bind(np.zeros(6))
    assert uploaded and ("watchdog", 10) in control.calls
    backend.command(np.zeros(6))
    assert control.calls[-1][0] == "servo"
    backend.close()


def test_ur_watchdog_failure_and_cleanup_attempt_each_operation(ur_fake):
    backend, receive, control, _ = ur_fake
    control.rejected_watchdog = True
    control.failed_stop = True
    with pytest.raises(RuntimeError, match="watchdog"):
        backend.bind(np.zeros(6))
    assert "servoStop" in control.calls and "stopScript" in control.calls and "disconnect" in control.calls
    backend.close()
    assert not receive.connected


def test_stale_and_reversed_robot_state_is_a_fault(ur_fake, monkeypatch):
    backend, _, _, _ = ur_fake
    monkeypatch.setattr("viki.replay.live.time.monotonic", lambda: backend.fresh_at + .2)
    with pytest.raises(RuntimeError, match="stale"):
        backend.read()
    backend.close()


class FakeBackend:
    identity = {"model": "fake"}

    def __init__(self, options, stop_event):
        self.options = options
        self.q, self.qd = np.zeros(6), np.zeros(6)
        self.steps = 0
        self.closed = False
        self.stop_event = stop_event
        self.commanded = self.q.copy()
        self.commands = []

    def read(self): return {"q": self.q.copy(), "qd": self.qd.copy(), "timestamp_s": self.steps / self.options.frequency, "safety_mode": 1}
    def bind(self, q): pass
    def command(self, q):
        self.commanded = q.copy()
        self.commands.append(q.copy())
    def begin_period(self): return None
    def advance(self, period, token=None):
        self.qd = (self.commanded - self.q) / period
        self.q = self.commanded.copy()
        self.steps += 1
        if self.steps >= 12: self.stop_event.set()
    def release(self): return []
    def close(self):
        self.closed = True
        return []


def test_process_owner_stops_and_keeps_partial_command_state_log(tmp_path):
    stop_event = threading.Event()
    targets, controls, snapshots = queue.Queue(1), queue.Queue(16), queue.Queue(1)
    controls.put(("request-a", "bind"))
    instances = []
    def make_backend(options):
        targets.put(target(source=time.monotonic_ns() // 1000))
        instances.append(FakeBackend(options, stop_event))
        return instances[-1]
    directory = tmp_path / "run"
    _worker("mujoco", LiveControlOptions(), directory, targets, controls, snapshots, stop_event,
            backend_factory=make_backend, retargeter_factory=FakeIK)
    report = json.loads((directory / "session.json").read_text())
    assert report["state"] == "stopped" and not report["running"]
    assert report["request_result"]["accepted"]
    assert report["log_rows"] >= 12 and instances[0].closed
    with h5py.File(directory / "execution.h5") as file:
        assert "active" in file["samples/state"].asstr()[:]
        assert file["samples/command_accepted"][:].all()


def test_startup_failure_also_writes_a_durable_rejected_run(tmp_path):
    stop_event = threading.Event()
    def unavailable(_options): raise RuntimeError("model unavailable")
    directory = tmp_path / "run"
    _worker("mujoco", LiveControlOptions(), directory, queue.Queue(), queue.Queue(), queue.Queue(1), stop_event,
            backend_factory=unavailable, retargeter_factory=FakeIK)
    report = json.loads((directory / "session.json").read_text())
    assert report["state"] == "fault" and "model unavailable" in report["reason"]
    assert report["log_rows"] == 0 and report["log_events"] == 1


def test_hardware_model_initialization_uses_observed_q_not_simulation_default(tmp_path):
    actual = np.full(6, .1)
    stop = threading.Event()
    stop.set()
    observed_options = []
    instances = []

    class ObservedIK(FakeIK):
        def __init__(self, options):
            np.testing.assert_array_equal(options.initial_q, actual)
            observed_options.append(options)
            super().__init__(options)

    def backend(options):
        arm = FakeBackend(options, stop)
        arm.q = arm.commanded = actual.copy()
        instances.append(arm)
        return arm

    directory = tmp_path / "observed-hardware-fake"
    _worker("ur10e", LiveControlOptions(), directory, queue.Queue(), queue.Queue(),
            queue.Queue(1), stop, backend_factory=backend, retargeter_factory=ObservedIK)
    report = json.loads((directory / "session.json").read_text())
    assert observed_options and report["state"] == "stopped"
    assert not instances[0].commands and instances[0].closed


def test_overdue_solver_command_is_rejected_before_actuation(tmp_path):
    class SlowIK(FakeIK):
        def step(self, *args):
            time.sleep(.03)
            return super().step(*args)
    stop = threading.Event()
    targets, controls = queue.Queue(1), queue.Queue(16)
    controls.put(("request", "bind"))
    instances = []
    def make_backend(options):
        targets.put(target(source=time.monotonic_ns() // 1000))
        instances.append(FakeBackend(options, stop))
        return instances[-1]
    directory = tmp_path / "run"
    _worker("mujoco", LiveControlOptions(frequency=50), directory, targets, controls, queue.Queue(1), stop,
            backend_factory=make_backend, retargeter_factory=SlowIK)
    report = json.loads((directory / "session.json").read_text())
    assert report["state"] == "fault" and "servo period" in report["reason"]
    assert report["cycle_compute_ms"] >= 30
    assert not instances[0].commands and instances[0].closed


def test_stop_during_process_creation_never_joins_an_unstarted_process(tmp_path):
    controller = LiveController(out_dir=tmp_path)
    entered, release = threading.Event(), threading.Event()
    errors = []
    class Process:
        pid, exitcode = 123, 0
        started = False
        def start(self):
            entered.set()
            assert release.wait(3)
            self.started = True
        def join(self, _timeout):
            assert self.started, "cannot join a process before start"
        def is_alive(self): return False
    class Context:
        def Process(self, **_kwargs): return Process()
    controller.context = Context()
    def start():
        try: controller.start(timeout=1)
        except RuntimeError: pass  # The fake process has no readiness report.
        except BaseException as exc: errors.append(exc)
    def stop():
        try: controller.stop()
        except BaseException as exc: errors.append(exc)
    starter, stopper = threading.Thread(target=start), threading.Thread(target=stop)
    starter.start()
    assert entered.wait(1)
    stopper.start()
    assert controller.stop_event.wait(1)
    release.set()
    starter.join(2); stopper.join(2)
    assert not starter.is_alive() and not stopper.is_alive() and not errors
    assert controller.snapshot()["running"] is False


def test_mujoco_holds_unbound_pose_with_compiled_gravity_compensation(tmp_path):
    pytest.importorskip("mujoco")
    from viki.retarget import normalize_robot

    names = normalize_robot("ur10e").joint_names
    body = "".join(f'<body pos="0 0 .1"><joint name="{name}" axis="0 1 0" range="-6.28 6.28"/><geom type="sphere" size=".01" mass="1"/>' for name in names)
    actuators = "".join(f'<position joint="{name}" kp="1000" kv="100"/>' for name in names)
    xml = tmp_path / "scene.xml"
    xml.write_text(f'<mujoco><compiler angle="radian"/><option integrator="implicitfast"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body name="base" quat="0 0 0 1">{body}{"</body>"*7}</worldbody><actuator>{actuators}</actuator></mujoco>')
    options = LiveControlOptions(model_path=str(xml), initial_q=(.3, .1, .1, .1, .1, .1), floor_offset_m=-1.)
    backend = _MujocoBackend(options)
    assert backend.model.ngravcomp > 0
    np.testing.assert_allclose(backend.model.body("base").quat, [1, 0, 0, 0])
    for _ in range(100): backend.advance(.008)
    np.testing.assert_allclose(backend.read()["q"], options.initial_q, atol=1e-9)
    assert np.max(np.abs(backend.read()["qd"])) < 1e-9
