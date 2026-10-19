"""Review must preserve availability/measurement time, scale and invalid gaps."""

from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from viki.contracts import LiveControlOptions
from viki.live import RecordedReviewStore
from viki.live.review import recorded_clock
from viki.server.deps import get_recorded_reviews
from viki.server.routes.live import router


class Graph:
    manifest = {"urdf_sha256": "same-model"}

    def __init__(self, options):
        self.options = options

    def scene(self, q):
        return {"points": np.array([[0, 0, 0], q[:3]]), "edges": [[0, 1]]}


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    episode = tmp_path / "episodes" / "source"
    raw = episode / "raw"
    raw.mkdir(parents=True)
    # Native stamps repeat in held raw frames; their host arrival stays fixed.
    rows = []
    for i in range(8):
        native = max(0, i-1) if i == 1 else i
        rows.append({"sync_us": 1_000_000+i*100_000,
                     "offsets_us": {"a": -1000 if i != 1 else -101000, "b": -2000 if i != 1 else -102000},
                     "device_timestamps_us": {"a": 100+native*100, "b": 110+native*100}})
    write_json(raw / "timestamps.json", rows)
    cloud = episode / "cloud"; cloud.mkdir()
    write_json(cloud / "meta.json", {"n_frames": len(rows), "bounds": [-1,-1,0,1,1,2]})
    for i in range(len(rows)):
        # A real valid tiny cloud payload, not an inference observation.
        (cloud / f"{i:06d}.bin").write_bytes(np.array([1], "<i4").tobytes()
            + np.array([[float(i),0,1]], "<f4").tobytes() + bytes([255,0,0]))
    directory = tmp_path / "runs" / ("a"*32)
    directory.mkdir(parents=True)
    cfg = {"reference": str(episode), "source": "recorded", "backend": "mujoco",
           "options": asdict(LiveControlOptions(scale=.25))}
    source = {"kind": "recorded", "playback_speed": 1., "device_ids": ["a", "b"],
              "clock": {"host_start_us": 10_000_000, "recorded_start_us": 1_000_000, "playback_speed": 1.}}
    write_json(directory / "manifest.json", {"config": cfg, "source": source,
                                            "hardware": False, "perception": {"profile": "online-fixture"}})
    q = np.array([[0.,0.,0.,0.,0.,0.], [.1,0,0,0,0,0], [.2,0,0,0,0,0], [.3,0,0,0,0,0]])
    executor = {"ended_monotonic_us": 10_700_000, "state": "stopped", "reason": "operator_stop",
                "kinematic_model": {"urdf_sha256": "same-model"}}
    write_json(directory / "session.json", {"executor": executor,
        "summary": {"phase": "stopped", "reason": "operator_stop", "controller": executor}})
    # First hand is from source row 0, but only becomes available at 120ms.
    hands = []
    for sequence, row_index, ready, valid in [(1,0,10_120_000,True),(2,2,10_330_000,False),(3,4,10_560_000,True)]:
        row = rows[row_index]
        hands.append({"sequence": sequence,
            "source_time_us": 10_000_000+row_index*100_000-2000,
            "generated_time_us": ready, "position_valid": valid, "orientation_valid": valid,
            "position": [1.,2.,3.], "rotation": np.eye(3).tolist(),
            "points": np.tile([1.,2.,3.], (21,1)).tolist(), "observed": [True]*21,
            "reason": "missing_position" if not valid else "", "camera_timestamps": row["device_timestamps_us"]})
    (directory / "hand.jsonl").write_text("".join(json.dumps(h)+"\n" for h in hands))
    log_dir = directory / "executor" / "test"; log_dir.mkdir(parents=True)
    with h5py.File(log_dir / "execution.h5", "w") as file:
        samples = file.create_group("samples")
        arrays = {"host_monotonic_us": [9_980_000,10_200_000,10_400_000,10_600_000],
            "source_time_us": [-1,9_998_000,10_198_000,10_398_000], "sequence": [-1,1,2,3],
            "q_observed": q, "q_commanded": q+1, "qd_observed": np.zeros_like(q),
            "tcp_position": q[:,:3], "tcp_rotation": np.tile(np.eye(3).reshape(9),(4,1)),
            "target_position": q[:,:3]+.1, "solver_ms": [0,4,4,4], "residual_rad": [0,.01,.02,.03],
            "command_accepted": [0,1,1,0]}
        for key, value in arrays.items(): samples[key] = value
        dt = h5py.string_dtype()
        samples.create_dataset("state", data=["unbound","active","braking","unbound"], dtype=dt)
        samples.create_dataset("cause", data=["","","invalid_human_position","operator_clutch"], dtype=dt)
        binding = {"human_position": [1.,2.,3.], "robot_position": [.1,0,0]}
        file.create_dataset("events_json", data=[json.dumps({"kind": "operator_request", "command": "bind",
            "accepted": True, "binding": binding})], dtype=dt)
    store = RecordedReviewStore(directory.parent, episode_roots=[tmp_path / "episodes"], retargeter_factory=Graph)
    return store, directory, episode, rows, hands


def test_availability_timeline_never_backdates_hand_or_uses_future_robot_state(artifacts):
    store, directory, _, _, _ = artifacts
    review = store.get(directory.name)
    assert review.frame(0)["hand"] is None
    early = review.frame(5)  # 100ms: first estimate is not yet available.
    assert early["hand"] is None and early["source_frame"] == 1
    assert early["robot"]["q_observed"] == [0.]*6
    ready = review.frame(6)
    assert ready["hand"]["visible"] and ready["hand"]["sequence"] == 1
    assert ready["robot"]["q_observed"] != ready["robot"]["q_commanded"]
    # Repeated seeks are not filters or forward-only stateful interpolation.
    review.frame(30)
    assert review.frame(6) == ready


def test_stale_and_invalid_hand_are_hidden_even_if_coordinates_are_finite(artifacts):
    store, directory, _, _, _ = artifacts
    review = store.get(directory.name)
    assert not review.frame(14)["hand"]["visible"]  # 280ms; first sample TTL is over.
    invalid = review.frame(17)  # 340ms, rejected estimate arrived.
    assert invalid["hand"]["reason"] == "missing_position" and not invalid["hand"]["visible"]
    assert review.frame(28)["hand"]["visible"]  # A later fresh estimate.
    assert review.frame(35)["robot"]["state"] == "stopped"


def test_fixed_registration_preserves_robot_motion_scale_and_real_measurements(artifacts):
    store, directory, _, _, _ = artifacts
    review = store.get(directory.name)
    meta = review.metadata()
    np.testing.assert_allclose(meta["robot_base_rig"], [.9,2,3])
    p0 = np.asarray(review.frame(0)["robot"]["points"])[1]
    p1 = np.asarray(review.frame(10)["robot"]["points"])[1]
    np.testing.assert_allclose(p1-p0,[.1,0,0])  # No 1/scale magnification of the arm.
    assert meta["scale"] == .25 and meta["profile"] == "online-fixture"
    assert meta["target_trajectory"][0] == [None]*3
    assert meta["target_trajectory"][2] == [None]*3


def test_legacy_clock_is_reconstructed_from_native_stamps_including_held_raw_frames(artifacts):
    _, directory, _, rows, hands = artifacts
    manifest = json.loads((directory/"manifest.json").read_text())
    source = manifest["source"]; source.pop("clock")
    epoch, method = recorded_clock(source, rows, hands)
    assert epoch == 10_000_000 and "native" in method
    hands[1]["source_time_us"] += 50_000
    with pytest.raises(ValueError, match="cannot be verified"):
        recorded_clock(source, rows, hands)


def test_recorded_clock_respects_source_speed_and_rejects_inconsistent_explicit_mapping(artifacts):
    _, directory, _, rows, hands = artifacts
    source = json.loads((directory/"manifest.json").read_text())["source"]
    source.pop("clock"); source["playback_speed"] = .5
    for hand in hands:
        hand["source_time_us"] = 20_000_000+int((hand["source_time_us"]-10_000_000)/.5)
    assert recorded_clock(source, rows, hands)[0] == 20_000_000
    source["clock"] = {"host_start_us":20_000_000,"recorded_start_us":0,"playback_speed":.5}
    with pytest.raises(ValueError, match="disagrees"):
        recorded_clock(source, rows, hands)


def test_model_hash_mismatch_is_explicit_and_no_offline_artifact_is_needed(artifacts):
    store, directory, episode, _, _ = artifacts
    assert not (episode/"plan.h5").exists() and not (episode/"cln.npz").exists()
    assert store.get(directory.name).metadata()["n_frames"] > 1
    class Wrong(Graph): manifest = {"urdf_sha256":"changed"}
    store.retargeter_factory = Wrong; store._cache.clear()
    with pytest.raises(ValueError, match="URDF differs"):
        store.get(directory.name)


def test_review_api_is_readonly_with_explicit_frame_and_path_boundaries(artifacts, tmp_path):
    store, directory, episode, _, _ = artifacts
    app = FastAPI(); app.include_router(router, prefix="/api")
    app.dependency_overrides[get_recorded_reviews] = lambda: store
    client = TestClient(app)
    listed = client.get("/api/live/reviews").json()["sessions"]
    assert listed[0]["id"] == directory.name
    base = f"/api/live/reviews/{directory.name}"
    assert client.get(base).json()["clock_method"] == "saved source clock"
    assert client.get(base+"/frames/5").json()["hand"] is None
    assert client.get(base+"/cloud/2").content == (episode/"cloud/000002.bin").read_bytes()
    assert client.get(base+"/frames/-1").status_code == 404
    assert client.get(base+"/frames/99999").status_code == 404
    assert client.get(base+"/cloud/99999").status_code == 404
    assert client.get("/api/live/reviews/not-a-session").status_code == 400
    external = tmp_path/"external.bin"; external.write_bytes(b"private")
    (episode/"cloud/000002.bin").unlink()
    (episode/"cloud/000002.bin").symlink_to(external)
    assert client.get(base+"/cloud/2").status_code == 400
    assert client.post(base+"/frames/0").status_code == 405


def test_store_rejects_reference_escape_and_session_symlink(artifacts, tmp_path):
    store, directory, _, _, _ = artifacts
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["config"]["reference"] = str(tmp_path)
    write_json(directory/"manifest.json", manifest)
    assert store.list() == []
    with pytest.raises(ValueError, match="outside"):
        store.get(directory.name)
    (store.root/("b"*32)).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        store.get("b"*32)


def test_display_transform_is_separate_from_logged_rig_coordinates(artifacts):
    store, directory, episode, _, _ = artifacts
    transform = [[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]]
    write_json(episode/"raw/world_anchor.json", {"T_world_display":transform})
    review = store.get(directory.name)
    assert review.metadata()["display_transform"] == transform
    np.testing.assert_allclose(review.frame(10)["hand"]["position"],[1,2,3])
    np.testing.assert_allclose(review.frame(10)["robot"]["points"][1],[1,2,3])


def test_running_logs_and_invalid_display_transform_are_rejected(artifacts):
    store, directory, episode, _, _ = artifacts
    report = json.loads((directory/"session.json").read_text())
    report["summary"]["running"] = True
    write_json(directory/"session.json",report)
    assert store.list() == []
    with pytest.raises(ValueError,match="completed"):
        store.get(directory.name)
    report["summary"]["running"] = False
    write_json(directory/"session.json",report)
    write_json(episode/"raw/world_anchor.json",{"T_world_display":(np.eye(4)*2).tolist()})
    with pytest.raises(ValueError,match="world-display"):
        store.get(directory.name)
