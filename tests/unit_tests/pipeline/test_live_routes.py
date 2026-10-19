"""HTTP session lifecycle and hardware mapping gates without hardware."""

import asyncio
import hashlib
import importlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from viki.server.deps import get_live_manager
from viki.server.routes.live import router


class SessionOwner:
    def __init__(self):
        self.configs = []
        self.controls = []
        self.running = False

    def start(self, config):
        if self.running:
            raise RuntimeError("another live session is still running")
        self.configs.append(config)
        self.running = True
        return self.snapshot()

    def snapshot(self):
        return {"session_id": "test-session", "running": self.running,
                "phase": "observing" if self.running else "idle"}

    def bind(self):
        if not self.running:
            raise RuntimeError("no live session")
        self.controls.append("bind")
        return "bind-request-id"

    def clutch(self):
        if not self.running:
            raise RuntimeError("no live session")
        self.controls.append("clutch")
        return "clutch-request-id"

    def stop(self):
        self.controls.append("stop")
        self.running = False
        return self.snapshot()


@pytest.fixture
def client_and_owner():
    application = FastAPI()
    application.include_router(router, prefix="/api")
    owner = SessionOwner()
    application.dependency_overrides[get_live_manager] = lambda: owner
    return TestClient(application), owner


def _mapping(tmp_path, **updates):
    path = tmp_path / "mapping.json"
    payload = {"coordinate_frame": "rig", "confirmed": True,
               "rotation_robot_rig": [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
               "floor_normal_robot": [0., 0., 1.], "floor_offset_m": -0.1,
               "source_calibration_id": "camera-calibration-test"}
    payload.update(updates)
    path.write_text(json.dumps(payload))
    return path


def test_status_is_read_only_and_simulation_start_never_auto_binds(client_and_owner, tmp_path):
    client, owner = client_and_owner
    assert client.get("/api/live/status").json()["phase"] == "idle"
    assert client.get("/api/live").status_code == 200
    assert owner.configs == owner.controls == []
    response = client.post("/api/live/start", json={"reference": str(tmp_path)})
    assert response.status_code == 200
    config = owner.configs[-1]
    assert config.backend == "mujoco" and config.source == "recorded"
    assert not config.auto_bind and not config.options.hardware_enabled
    assert owner.controls == []
    assert client.get("/api/live/status").json()["running"]


def test_operator_requests_return_pending_ids_and_stop_status(client_and_owner, tmp_path):
    client, owner = client_and_owner
    assert client.post("/api/live/bind").status_code == 409
    assert client.post("/api/live/clutch").status_code == 409
    assert client.post("/api/live/start", json={"reference": str(tmp_path)}).status_code == 200
    assert client.post("/api/live/bind").json() == {
        "request_id": "bind-request-id", "status": "bind-requested"}
    assert client.post("/api/live/clutch").json() == {
        "request_id": "clutch-request-id", "status": "clutch-requested"}
    assert client.post("/api/live/stop").json()["running"] is False
    assert owner.controls == ["bind", "clutch", "stop"]
    # Stop on an idle owner remains harmless and idempotent.
    assert client.post("/api/live/stop").status_code == 200


def test_second_session_is_rejected_without_extra_start(client_and_owner, tmp_path):
    client, owner = client_and_owner
    body = {"reference": str(tmp_path)}
    assert client.post("/api/live/start", json=body).status_code == 200
    assert client.post("/api/live/start", json=body).status_code == 409
    assert len(owner.configs) == 1


@pytest.mark.parametrize("body", [
    {"backend": "ur10e"},
    {"backend": "ur10e", "options": {"host": "192.168.0.100"}},
    {"backend": "ur10e", "options": {"host": "192.168.0.100",
                                      "hardware_enabled": True, "mapping_confirmed": True}},
])
def test_client_confirmed_boolean_cannot_bypass_mapping_file(client_and_owner, tmp_path, body):
    client, owner = client_and_owner
    response = client.post("/api/live/start", json={"reference": str(tmp_path), **body})
    assert response.status_code == 400
    assert owner.configs == []


def test_confirmed_mapping_is_applied_and_hashed_before_hardware_start(client_and_owner, tmp_path):
    client, owner = client_and_owner
    path = _mapping(tmp_path)
    response = client.post("/api/live/start", json={
        "reference": str(tmp_path), "backend": "ur10e", "source": "cameras",
        "mapping_json": str(path),
        "options": {"host": "192.168.0.100", "hardware_enabled": True,
                    "floor_offset_m": 5.0, "mapping_confirmed": False},
    })
    assert response.status_code == 200
    config = owner.configs[-1]
    assert config.options.mapping_confirmed and config.options.hardware_enabled
    assert config.options.floor_offset_m == -0.1
    assert config.options.rotation_robot_rig[0] == (0., -1., 0.)
    assert config.mapping_manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config.mapping_manifest["source_calibration_id"] == "camera-calibration-test"
    assert owner.controls == []


def test_mapping_file_alone_does_not_enable_hardware(client_and_owner, tmp_path):
    client, owner = client_and_owner
    response = client.post("/api/live/start", json={
        "reference": str(tmp_path), "backend": "ur10e", "mapping_json": str(_mapping(tmp_path)),
        "options": {"host": "192.168.0.100"},
    })
    assert response.status_code == 400
    assert owner.configs == []


@pytest.mark.parametrize("calibration_id", [None, "", "   ", 42])
def test_hardware_mapping_requires_a_calibration_identity(client_and_owner, tmp_path, calibration_id):
    client, owner = client_and_owner
    response = client.post("/api/live/start", json={
        "reference": str(tmp_path), "backend": "ur10e",
        "mapping_json": str(_mapping(tmp_path, source_calibration_id=calibration_id)),
        "options": {"host": "192.168.0.100", "hardware_enabled": True},
    })
    assert response.status_code == 400
    assert "source_calibration_id" in response.json()["detail"]
    assert owner.configs == []


@pytest.mark.parametrize("updates", [
    {"confirmed": False}, {"coordinate_frame": "robot_base"},
    {"rotation_robot_rig": [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]},
])
def test_invalid_mapping_never_starts_session(client_and_owner, tmp_path, updates):
    client, owner = client_and_owner
    response = client.post("/api/live/start", json={
        "reference": str(tmp_path), "mapping_json": str(_mapping(tmp_path, **updates))})
    assert response.status_code == 400
    assert owner.configs == []


def test_missing_reference_and_mapping_are_reported_before_start(client_and_owner, tmp_path):
    client, owner = client_and_owner
    assert client.post("/api/live/start", json={"reference": str(tmp_path / "absent")}).status_code == 404
    assert client.post("/api/live/start", json={"reference": str(tmp_path),
        "mapping_json": str(tmp_path / "absent.json")}).status_code == 404
    assert owner.configs == []


def test_mapping_json_must_be_an_object(client_and_owner, tmp_path):
    client, owner = client_and_owner
    path = tmp_path / "mapping-list.json"
    path.write_text("[]")
    response = client.post("/api/live/start", json={
        "reference": str(tmp_path), "mapping_json": str(path)})
    assert response.status_code == 400
    assert owner.configs == []


def test_typed_request_and_hardware_auto_bind_rejection(client_and_owner, tmp_path):
    client, owner = client_and_owner
    assert client.post("/api/live/start", json={"reference": str(tmp_path), "backend": "other"}).status_code == 422
    assert client.post("/api/live/start", json={"reference": str(tmp_path), "unrecognised": True}).status_code == 422
    assert client.post("/api/live/start", json={"reference": str(tmp_path), "seconds": -1}).status_code == 400
    response = client.post("/api/live/start", json={"reference": str(tmp_path), "backend": "ur10e",
        "mapping_json": str(_mapping(tmp_path)), "auto_bind": True,
        "options": {"hardware_enabled": True, "host": "192.168.0.100"}})
    assert response.status_code == 400
    assert owner.configs == []


@pytest.mark.parametrize("close_failure", [False, True])
def test_application_lifespan_shares_cameras_and_closes_live_first(monkeypatch, tmp_path, close_failure):
    module = importlib.import_module("viki.server.app")
    events = []

    class Cameras:
        def stop_all(self):
            events.append("cameras-close")

    class Calibration:
        def __init__(self, manager):
            self.manager = manager

        def load_all_extrinsics(self):
            pass

    class Replay:
        def close(self):
            events.append("replay-close")

    class Live:
        def __init__(self, camera_manager):
            self.camera_manager = camera_manager

        def close(self):
            events.append("live-close")
            if close_failure:
                raise RuntimeError("test live cleanup fault")

    monkeypatch.setattr(module, "CameraManager", Cameras)
    monkeypatch.setattr(module, "CalibrationManager", Calibration)
    monkeypatch.setattr(module, "_prefetch_models", lambda: None)
    monkeypatch.setattr("viki.calibration.presets.apply_active_on_startup", lambda: None)
    monkeypatch.setattr("viki.replay.ReplayManager", Replay)
    monkeypatch.setattr("viki.live.LiveManager", Live)
    application = FastAPI()

    async def run():
        async with module.lifespan(application):
            assert application.state.live_manager.camera_manager is application.state.manager
            assert application.state.calibrator.manager is application.state.manager

    if close_failure:
        with pytest.raises(RuntimeError, match="cleanup fault"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert events == ["live-close", "replay-close", "cameras-close"]
    # Check real router wiring through HTTP instead of depending on FastAPI's
    # eager/lazy included-router representation.
    owner = SessionOwner()
    monkeypatch.setitem(module.app.dependency_overrides, get_live_manager, lambda: owner)
    assembled = TestClient(module.app)
    response = assembled.get("/api/live/status")
    assert response.status_code == 200 and response.json()["phase"] == "idle"
    assert assembled.get("/api/live").status_code == 200
    assert assembled.post("/api/live/start", json={"reference": str(tmp_path)}).status_code == 200
    assert assembled.post("/api/live/bind").json()["request_id"] == "bind-request-id"
    assert assembled.post("/api/live/clutch").json()["request_id"] == "clutch-request-id"
    assert assembled.post("/api/live/stop").json()["running"] is False
