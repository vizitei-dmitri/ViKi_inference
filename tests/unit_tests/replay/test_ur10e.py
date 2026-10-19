"""Exercise RTDE failure paths without ever importing or connecting real RTDE."""

from contextlib import nullcontext
import json
import os
import signal
import threading

import numpy as np
import pytest

from viki.contracts import Episode
from viki.episode import mark_stage, read_status, stage_done
from viki.replay import ReplayManager, URReplayOptions, check_replay, inspect_ur10e, replay_episode
from viki.replay.plan import servo_commands
from viki.replay.ur import ReplayAborted, UR10eDriver
from viki.retarget import load_archive, normalize_robot, write_hdf5_archive


@pytest.fixture
def planned(tmp_path):
    ep = Episode(root=tmp_path / "ep")
    q = np.zeros((4, 6))
    q[:, 0] = np.arange(4) * 0.001
    data = {
        "q": q, "q_approach": np.zeros((2, 6)), "dt": 0.05,
        "robot": "ur10e_description", "gripper_model": "linear",
        "gripper_command": np.linspace(1, 0, 4)[:, None],
        "gripper_command_names_json": '["opening"]',
        "joint_names_json": json.dumps(normalize_robot("ur10e").joint_names),
        "solver_status": "converged", "config_json": "{}",
        "metrics_json": json.dumps({
            "min_floor_margin_mm": 5.0, "min_collision_margin": 0.001,
            "min_joint_limit_margin_rad": 1.0, "min_velocity_limit_margin_rad_s": 1.0,
        }),
    }
    write_hdf5_archive(ep.plan_h5, data)
    mark_stage(ep, "retarget")
    return ep, data


class FakeReceive:
    def __init__(self, clock):
        self.clock = clock
        self.q = np.zeros(6)
        self.safety = 1
        self.runtime = 1
        self.disconnected = False
        self.stale = False

    def isConnected(self): return not self.disconnected
    def getActualQ(self): return self.q.copy()
    def getActualQd(self): return np.zeros(6)
    def getActualTCPPose(self): return np.zeros(6)
    def getSafetyMode(self): return self.safety
    def getRobotMode(self): return 7
    def getRuntimeState(self): return self.runtime
    def getTimestamp(self): return 1.0 if self.stale else self.clock[0] + 1
    def disconnect(self): self.disconnected = True


class FakeControl:
    def __init__(self, receive, clock):
        self.receive = receive
        self.clock = clock
        self.calls = []
        self.failure = None

    def isConnected(self): return True
    def isProgramRunning(self): return True
    def isJointsWithinSafetyLimits(self, q): return True
    def setWatchdog(self, frequency):
        self.calls.append("watchdog")
        return True
    def kickWatchdog(self): return True
    def initPeriod(self): return self.clock[0]
    def servoJ(self, q, speed, acceleration, dt, lookahead, gain):
        self.calls.append("servo")
        if self.failure == "interrupt": raise KeyboardInterrupt()
        if self.failure == "command": return False
        self.receive.q = np.asarray(q)
        if self.failure == "tracking": self.receive.q += 0.1
        return True
    def waitPeriod(self, cycle):
        self.clock[0] = max(self.clock[0], cycle + 1 / 125)
        if self.failure == "deadline": self.clock[0] += 0.1
        if self.failure == "safety": self.receive.safety = 3
    def servoStop(self): self.calls.append("servoStop")
    def stopScript(self): self.calls.append("stopScript")
    def disconnect(self): self.calls.append("disconnect")


@pytest.fixture
def rtde(monkeypatch):
    clock = [0.0]
    receive = FakeReceive(clock)
    control = FakeControl(receive, clock)
    connected = []
    monkeypatch.setattr("viki.replay.ur._robot_lock", lambda host: nullcontext())
    monkeypatch.setattr("viki.replay.ur._dashboard_info", lambda host: {"model": "UR10", "version": "5.23"})
    monkeypatch.setattr("viki.replay.ur._receive", lambda *args: receive)
    def connect(*args):
        connected.append(True)
        return control
    monkeypatch.setattr("viki.replay.ur._control", connect)
    monkeypatch.setattr("viki.replay.ur.time.monotonic", lambda: clock[0])
    return receive, control, connected


def test_offline_check_has_no_connections_or_writes(planned, rtde):
    ep, _ = planned
    report = check_replay(ep, options=URReplayOptions(arm_only=True))
    assert report["robot"] == "ur10e"
    assert report["approach_frames"] == 2
    assert not rtde[2] and not ep.replay_h5.exists()


@pytest.mark.parametrize("change", [
    {"robot": "ur10_official_description"},
    {"joint_names_json": '["wrong"]'}, {"solver_status": "max_iterations"},
    {"metrics_json": '{}'}, {"q": np.full((4, 6), np.nan)}, {"dt": 0},
    {"gripper_command": np.full((4, 1), 2.0)},
])
def test_invalid_plan_never_connects(planned, rtde, change):
    ep, data = planned
    write_hdf5_archive(ep.plan_h5, {**data, **change})
    with pytest.raises(ValueError):
        replay_episode(ep, driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    assert not rtde[2] and "servo" not in rtde[1].calls


def test_limits_include_start_and_end_and_no_angle_wrap():
    q = np.zeros((2, 6)); q[1, 0] = 0.1
    with pytest.raises(ValueError, match="command limits"):
        servo_commands(q, 0.1, URReplayOptions(speed_scale=1))
    q[0, 0], q[1, 0] = 6.2, -6.2
    with pytest.raises(ValueError, match="command limits"):
        servo_commands(q, 0.1, URReplayOptions())
    times, commands, v, a = servo_commands(np.zeros((1, 6)), 0.1, URReplayOptions())
    assert len(times) == len(commands) and v == a == 0


@pytest.mark.parametrize("change", ["pose", "program", "safety"])
def test_live_preflight_does_not_upload_control_on_failure(rtde, change):
    receive, control, connected = rtde
    if change == "pose": receive.q[:] = 0.2
    if change == "program": receive.runtime = 2
    if change == "safety": receive.safety = 3
    with pytest.raises(ReplayAborted):
        UR10eDriver(host="robot", options=URReplayOptions(arm_only=True)).execute(np.zeros((2, 6)), np.ones(2), 0.1)
    assert not connected and not control.calls
    assert receive.disconnected


def test_inspection_is_read_only(rtde):
    report = inspect_ur10e("robot", start_q=np.zeros(6))
    assert report["at_start"] and not rtde[2]
    assert rtde[0].disconnected


def test_arm_only_writes_measurements_and_keeps_approach_in_log(planned, rtde):
    ep, _ = planned
    replay_episode(ep, driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    with load_archive(ep.replay_h5) as arc:
        assert arc["verdict"] == "arm-only"
        assert arc["hardware"] and arc["completed"] and not arc["gripper_measured"]
        assert np.isnan(arc["gripper_attained"]).all()
        assert arc["q_attained"].shape == (4, 6)
        assert arc["dt"] == 0.5 and arc["source_dt"] == 0.05
        log = ep.root / arc["execution_log"]
    with load_archive(log) as arc:
        assert arc["q_source"].shape == (6, 6)
        assert arc["q_commanded"].shape[0] > 6
        assert arc["q_observed"].shape == arc["q_commanded"].shape
        assert np.diff(arc["controller_timestamp_s"]).min() > 0
    assert rtde[1].calls[-3:] == ["servoStop", "stopScript", "disconnect"]


@pytest.mark.parametrize("fault", ["command", "tracking", "interrupt", "deadline", "safety", "stale"])
def test_abort_stops_and_preserves_partial_log(planned, rtde, fault):
    ep, _ = planned
    receive, control, _ = rtde
    if fault == "stale": receive.stale = True
    else: control.failure = fault
    mark_stage(ep, "replay", verdict="pass")
    mark_stage(ep, "export")
    with pytest.raises(RuntimeError, match="partial log"):
        replay_episode(ep, driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    assert not stage_done(ep, "replay") and not stage_done(ep, "export")
    status = read_status(ep)["stages"]["replay"]
    assert status["verdict"] == "reject"
    assert (ep.root / status["execution_log"]).is_file()
    assert control.calls[-3:] == ["servoStop", "stopScript", "disconnect"]
    assert receive.disconnected


def test_no_implicit_gripper_or_resolve(planned):
    ep, _ = planned
    with pytest.raises(ValueError, match="arm-only"):
        replay_episode(ep, driver="ur10e", host="robot")
    with pytest.raises(NotImplementedError):
        replay_episode(ep, max_resolves=1)


def test_each_replay_retains_an_immutable_log(planned):
    ep, _ = planned
    replay_episode(ep)
    logs = list((ep.root / "intermediates/replay").glob("*/execution.h5"))
    first = logs[0].read_bytes()
    replay_episode(ep)
    assert len(list((ep.root / "intermediates/replay").glob("*/execution.h5"))) == 2
    assert logs[0].read_bytes() == first


def test_cleanup_failure_rejects_even_a_completed_motion(planned, rtde, monkeypatch):
    ep, _ = planned
    def failed_stop():
        rtde[1].calls.append("servoStop")
        return False
    monkeypatch.setattr(rtde[1], "servoStop", failed_stop)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        replay_episode(ep, driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    assert rtde[1].calls[-3:] == ["servoStop", "stopScript", "disconnect"]
    assert not stage_done(ep, "replay")


def test_http_requires_explicit_hardware_settings_and_offline_check_is_read_only(planned, rtde, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from viki.server.routes.replay import router

    app = FastAPI(); app.include_router(router)
    app.state.replay_manager = ReplayManager()
    client = TestClient(app)
    payload = {"episode": str(planned[0].root), "driver": "ur10e"}
    def no_job(*args, **kwargs):
        pytest.fail("invalid request queued a robot job")
    monkeypatch.setattr("viki.server.routes.replay.jobs.submit", no_job)
    assert client.post("/replay", json=payload).status_code == 400
    response = client.post("/replay/check", json={**payload, "options": {"arm_only": True}})
    assert response.status_code == 200 and response.json()["robot"] == "ur10e"
    assert not rtde[2] and not planned[0].replay_h5.exists()
    bad = client.post("/replay/check", json={**payload, "options": {"speed_scale": 0}})
    assert bad.status_code == 400


def test_stop_request_runs_cleanup_and_retains_data(planned, rtde, monkeypatch):
    stop = threading.Event()
    original_wait = rtde[1].waitPeriod
    def stop_after_sample(cycle):
        original_wait(cycle)
        stop.set()
    monkeypatch.setattr(rtde[1], "waitPeriod", stop_after_sample)
    with pytest.raises(RuntimeError, match="requested replay stop"):
        replay_episode(planned[0], driver="ur10e", host="robot",
                       options=URReplayOptions(arm_only=True), stop_event=stop)
    assert rtde[1].calls[-3:] == ["servoStop", "stopScript", "disconnect"]
    assert not stage_done(planned[0], "replay")


def test_cli_signal_handler_overrides_inherited_ignore_and_restores_it():
    from viki.replay.ur import _motion_signals

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with pytest.raises(KeyboardInterrupt):
            with _motion_signals():
                os.kill(os.getpid(), signal.SIGINT)
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_manager_cancels_queued_motion_before_driver_connection(planned, rtde):
    manager = ReplayManager()
    token = manager.prepare()
    manager.bind_job("job", token)
    assert manager.stop_job("job")
    with pytest.raises(RuntimeError, match="cancelled"):
        manager.run(token, planned[0], driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    assert not rtde[2]
    assert not manager.stop_job("job")


def test_telemetry_collection_uses_period_budget_without_cumulative_drift(planned, rtde, monkeypatch):
    receive, control, _ = rtde
    def delayed_getter():
        control.clock[0] += 0.002
        return np.zeros(6)
    monkeypatch.setattr(receive, "getActualTCPPose", delayed_getter)
    replay_episode(planned[0], driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    assert read_status(planned[0])["stages"]["replay"]["verdict"] == "arm-only"


def test_failed_connection_preserves_rejected_attempt(planned, rtde, monkeypatch):
    def unavailable(_host):
        raise TimeoutError("dashboard unavailable")
    monkeypatch.setattr("viki.replay.ur._dashboard_info", unavailable)
    with pytest.raises(RuntimeError, match="dashboard unavailable"):
        replay_episode(planned[0], driver="ur10e", host="robot", options=URReplayOptions(arm_only=True))
    status = read_status(planned[0])["stages"]["replay"]
    assert not status["done"] and status["verdict"] == "reject"
    assert (planned[0].root / status["execution_log"]).exists()
    assert not rtde[2]


def test_bounded_hardware_test_stops_without_claiming_a_full_replay(planned, rtde):
    replay_episode(planned[0], driver="ur10e", host="robot",
                   options=URReplayOptions(arm_only=True, stop_after=0.05))
    status = read_status(planned[0])["stages"]["replay"]
    assert status["verdict"] == "test-stop" and not status["completed"]
    assert not stage_done(planned[0], "replay")
    assert rtde[1].clock[0] < 0.07
    assert rtde[1].calls[-3:] == ["servoStop", "stopScript", "disconnect"]
