"""Session integration without cameras, models, RTDE or robot movement."""

from dataclasses import replace
import json
import threading
import time

import numpy as np
import pytest

from viki.cameras import HardwareSyncError
from viki.contracts import LiveControlOptions, LiveHandEstimate
from viki.live import LiveManager, LiveSession, LiveSessionConfig, load_mapping


class FakeEstimator:
    device_ids = ("a", "b")
    calibration_id = "known-calibration"
    manifest = {"profile": "injected-test"}

    def __init__(self, events, *, fail_processing=False, fail_close=False):
        self.events = events
        self.closed = False
        self.fail_processing, self.fail_close = fail_processing, fail_close
        events.append("estimator-ready")

    def process(self, sequence):
        if self.fail_processing:
            raise RuntimeError("injected perception failure")
        now = time.monotonic_ns() // 1000
        return LiveHandEstimate(
            sequence=sequence, source_time_us=now - 80_000,
            generated_time_us=now, points=np.full((21, 3), np.nan),
            observed=np.zeros(21, bool), quality=np.zeros(21),
            position=np.array([.1, .2, .3]), rotation=np.full((3, 3), np.nan),
            position_valid=True, orientation_valid=False,
            calibration_id=self.calibration_id,
            camera_timestamps={"a": 100 + sequence, "b": 200 + sequence},
        )

    def close(self):
        self.closed = True
        self.events.append("estimator-closed")
        if self.fail_close:
            raise RuntimeError("injected estimator cleanup failure")


class FakeSource:
    source_summary = {"kind": "injected-readonly-source"}
    dropped_groups = 2

    def __init__(self, events, groups=4, *, fail_close=False):
        self.events, self.groups = events, groups
        self.index = 0
        self.closed = False
        self.fail_close = fail_close
        events.append("source-clock-start")

    @property
    def exhausted(self):
        return self.closed or self.groups is not None and self.index >= self.groups

    def poll(self):
        self.index += 1
        return self.index

    def close(self):
        self.closed = True
        self.events.append("source-closed")
        if self.fail_close:
            raise RuntimeError("injected source cleanup failure")


class FakeController:
    def __init__(self, events, *, delayed_consumption=False, fail_stop=False, **kwargs):
        self.events, self.delayed_consumption = events, delayed_consumption
        self.fail_stop = fail_stop
        self.running = False
        self.targets = []
        self.sequence = -1
        self.binds = []
        self.snapshots = 0
        self.state = "unbound"
        self.request_result = None
        self.events.append("controller-created")

    def start(self):
        self.running = True
        self.events.append("controller-ready")

    def snapshot(self):
        self.snapshots += 1
        if self.targets and (not self.delayed_consumption or len(self.targets) >= 2):
            self.sequence = self.targets[-1].sequence
        # Simulate input loss/released tracking after every bind. Automatic
        # reacquisition must still never occur merely because input returned.
        return {"running": self.running, "state": self.state, "sequence": self.sequence,
                "request_result": self.request_result}

    def submit(self, target):
        self.targets.append(target)
        return True

    def bind(self):
        self.binds.append(self.sequence)
        self.request_result = {"command": "bind", "accepted": True}
        return f"request-bind-{len(self.binds)}"

    def clutch(self):
        self.request_result = {"command": "clutch", "accepted": True}
        return "request-clutch"

    def stop(self, timeout=5):
        self.running = False
        self.events.append("controller-stopped")
        if self.fail_stop:
            raise RuntimeError("injected controller cleanup failure")
        return {"running": False, "state": "stopped"}


def build(tmp_path, *, groups=4, config_changes=None, fail_processing=False,
          fail_source_close=False, fail_estimator_close=False, fail_controller_stop=False,
          delayed_consumption=False):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "sentinel.txt").write_text("reference must stay unchanged")
    events = []
    estimator = FakeEstimator(events, fail_processing=fail_processing,
                              fail_close=fail_estimator_close)
    controller = FakeController(events, delayed_consumption=delayed_consumption,
                                fail_stop=fail_controller_stop)
    source = None

    def source_factory(_estimator):
        nonlocal source
        assert _estimator is estimator
        assert controller.running, "recorded clock started before executor became ready"
        source = FakeSource(events, groups, fail_close=fail_source_close)
        return source

    cfg = LiveSessionConfig(reference=str(reference), out_dir=str(tmp_path / "runs"),
                            perception_hz=30, seconds=2, **(config_changes or {}))
    session = LiveSession(cfg, estimator_factory=lambda *a, **kw: estimator,
                           source_factory=source_factory, controller_factory=lambda **kw: controller)
    return session, estimator, controller, lambda: source, events


def completed(session):
    session.start()
    session._thread.join(timeout=3)
    assert not session._thread.is_alive(), "injected session did not finish"
    return session.snapshot()


def wait_for(predicate, limit=1):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("session condition was not reached")


def test_recorded_clock_after_executor_start_ttl_source_time_and_durable_nan_safe_logs(tmp_path):
    session, estimator, controller, source, events = build(tmp_path)
    result = completed(session)
    assert result["phase"] == "stopped" and not result["running"]
    assert events.index("source-clock-start") > events.index("controller-ready")
    assert estimator.closed and source().closed and not controller.running
    assert controller.binds == []
    assert len(controller.targets) == 4
    for target in controller.targets:
        assert target.valid_until_us - target.source_time_us == 250_000
    assert result["observation_age_ms"]["median"] >= 80
    assert result["source_dropped_groups"] == 2
    summary = json.loads((session.directory / "session.json").read_text())
    assert summary["summary"]["running"] is False
    assert summary["summary"]["phase"] == "stopped"
    payload = (session.directory / "hand.jsonl").read_text()
    assert "NaN" not in payload and "Infinity" not in payload
    lines = [json.loads(line) for line in payload.splitlines()]
    assert len(lines) == 4 and lines[0]["rotation"][0][0] is None
    assert lines[0]["points"][0][0] is None
    assert (tmp_path / "reference" / "sentinel.txt").read_text() == "reference must stay unchanged"


def test_perception_fault_saves_partial_run_and_closes_every_component(tmp_path):
    session, estimator, controller, source, events = build(tmp_path, fail_processing=True)
    result = completed(session)
    assert result["phase"] == "fault" and "perception failure" in result["reason"]
    assert estimator.closed and source().closed and not controller.running
    assert json.loads((session.directory / "session.json").read_text())["summary"]["phase"] == "fault"


def test_cleanup_failures_are_independent_and_final_run_is_persisted(tmp_path):
    session, estimator, controller, source, events = build(
        tmp_path, groups=1, fail_source_close=True, fail_estimator_close=True,
        fail_controller_stop=True)
    result = completed(session)
    assert result["phase"] == "fault"
    assert estimator.closed and source().closed and not controller.running
    assert all(name in events for name in ("controller-stopped", "source-closed", "estimator-closed"))
    assert (session.directory / "session.json").is_file()
    assert len((session.directory / "hand.jsonl").read_text().splitlines()) == 1


def test_manual_bind_and_clutch_are_explicit_requests_visible_in_status(tmp_path):
    session, estimator, controller, source, events = build(tmp_path, groups=None)
    with pytest.raises(RuntimeError, match="not ready"):
        session.bind()
    session.start()
    try:
        wait_for(lambda: session.snapshot()["phase"] == "observing"
                 and session.snapshot()["hand"] is not None)
        assert controller.binds == []
        assert session.bind() == "request-bind-1"
        assert session.snapshot()["controller"]["request_result"]["accepted"]
        assert session.clutch() == "request-clutch"
    finally:
        result = session.stop()
    assert not result["running"] and result["phase"] == "stopped"
    assert source().closed and estimator.closed


def test_simulation_auto_bind_waits_for_consumed_sequence_and_never_reacquires(tmp_path):
    session, estimator, controller, source, events = build(
        tmp_path, groups=6, config_changes={"auto_bind": True}, delayed_consumption=True)
    result = completed(session)
    assert result["phase"] == "stopped"
    assert len(controller.binds) == 1
    assert controller.binds[0] >= controller.targets[0].sequence
    assert len(controller.targets) == 6


def test_hardware_rejects_auto_bind_before_any_hardware_or_session_is_created(tmp_path):
    options = replace(LiveControlOptions(), host="192.0.2.1", mapping_confirmed=True,
                      hardware_enabled=True)
    config = LiveSessionConfig(reference=str(tmp_path), backend="ur10e", auto_bind=True,
                               options=options, out_dir=str(tmp_path / "runs"))
    with pytest.raises(ValueError, match="automatic bind"):
        LiveSession(config)
    assert not (tmp_path / "runs").exists()


def test_foreign_mapping_calibration_closes_estimator_before_source_and_controller_start(tmp_path):
    session, estimator, controller, source, events = build(
        tmp_path, config_changes={"mapping_manifest": {"source_calibration_id": "foreign-calibration"}})
    result = completed(session)
    assert result["phase"] == "fault" and "another camera calibration" in result["reason"]
    assert estimator.closed and source() is None
    assert "controller-ready" not in events
    assert (session.directory / "session.json").is_file()


def test_camera_readiness_retries_native_timestamp_warmup(tmp_path):
    session, _, _, _, _ = build(tmp_path)
    calls = []

    class Manager:
        def latest_frame(self, device):
            return object()

        def require_hardware_sync_ready(self):
            calls.append(True)
            if len(calls) < 3:
                raise HardwareSyncError("timestamp buffer is warming up")
            return {"ready": True}

    session._wait_cameras(Manager(), ("a", "b"))
    assert len(calls) == 3


@pytest.mark.parametrize("component", ["estimator", "controller"])
def test_operator_cancel_during_startup_never_starts_frame_source_or_reports_fault(tmp_path, component):
    session, estimator, controller, source, _ = build(tmp_path)
    entered, release = threading.Event(), threading.Event()
    if component == "estimator":
        def estimator_factory(*args, **kwargs):
            entered.set()
            assert release.wait(2)
            return estimator
        session._estimator_factory = estimator_factory
    else:
        def start():
            entered.set()
            assert release.wait(2)
            raise RuntimeError("operator_stop")
        controller.start = start
    session.start()
    assert entered.wait(1)
    session.stop(timeout=.01)
    release.set()
    session._thread.join(2)
    result = session.snapshot()
    assert result["phase"] == "stopped" and result["reason"] == "operator_stop"
    assert not result["running"] and not controller.running
    assert source() is None and estimator.closed


def test_live_manager_rejects_second_owner_even_when_session_thread_finished(monkeypatch, tmp_path):
    import viki.live.session as module

    class Session:
        def __init__(self, config, camera_manager=None):
            self.running, self.controller_running = False, False

        def start(self):
            self.running = self.controller_running = True
            return self.snapshot()

        def snapshot(self):
            return {"running": self.running, "controller": {"running": self.controller_running}}

        def stop(self):
            self.running = self.controller_running = False
            return self.snapshot()

        def bind(self):
            return "bind-request"

        def clutch(self):
            return "clutch-request"

    monkeypatch.setattr(module, "LiveSession", Session)
    manager = LiveManager()
    config = LiveSessionConfig(reference=str(tmp_path))
    assert manager.start(config)["running"]
    with pytest.raises(RuntimeError, match="another live session"):
        manager.start(config)
    manager.session.running = False
    with pytest.raises(RuntimeError, match="another live session"):
        manager.start(config)
    assert manager.bind() == "bind-request" and manager.clutch() == "clutch-request"
    assert not manager.stop()["controller"]["running"]
    assert manager.start(config)["running"]
    manager.close()


def test_mapping_requires_explicit_confirmation_rig_axes_and_proper_rotation(tmp_path):
    path = tmp_path / "mapping.json"
    valid = {"coordinate_frame": "rig", "confirmed": True,
             "rotation_robot_rig": np.eye(3).tolist(), "floor_normal_robot": [0, 0, 1],
             "floor_offset_m": 0., "source_calibration_id": "known-calibration"}
    path.write_text(json.dumps(valid))
    changes, manifest = load_mapping(path)
    assert changes["mapping_confirmed"] and manifest["source_calibration_id"] == "known-calibration"
    path.write_text(json.dumps({**valid, "confirmed": False}))
    with pytest.raises(ValueError, match="confirmed"):
        load_mapping(path)
    path.write_text(json.dumps({**valid, "rotation_robot_rig": [[1, 0, 0], [0, 2, 0], [0, 0, 1]]}))
    with pytest.raises(ValueError, match="proper SO"):
        load_mapping(path)
