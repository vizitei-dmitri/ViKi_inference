"""A comparison must retain input availability, expiry and source provenance."""

from dataclasses import asdict
import json

import h5py
import numpy as np
import pytest

from viki.contracts import LiveControlOptions, LiveIKTuning
from viki.replay import benchmark_live_ik
from viki.replay.benchmark import _load_targets, _simulate


@pytest.fixture
def session(tmp_path):
    directory = tmp_path / "source"
    log_dir = directory / "executor" / "run"
    log_dir.mkdir(parents=True)
    cfg = {"backend": "mujoco", "reference": "fixture", "recorded_speed": 1.,
           "options": asdict(LiveControlOptions(initial_q=(0.,)*6))}
    (directory / "manifest.json").write_text(json.dumps({"config": cfg, "hardware": False}))
    (directory / "session.json").write_text(json.dumps({"summary": {"controller": {
        "kinematic_model": {"urdf_sha256": "fixture"}}}}))
    with h5py.File(log_dir / "execution.h5", "w") as f:
        f.attrs["closed_cleanly"] = True
        s = f.create_group("samples")
        s["host_monotonic_us"] = [1_000_000, 1_025_000, 1_040_000, 1_060_000, 1_080_000]
        s["source_time_us"] = [990_000, 1_015_000, 1_030_000, 1_050_000, 1_070_000]
        s["target_position"] = [[.01, 0, 0], [.02, 0, 0], [.03, 0, 0], [.04, 0, 0], [.05, 0, 0]]
        s["q_observed"] = np.zeros((5, 6))
        s["qd_observed"] = np.zeros((5, 6))
        s["state"] = np.array(["active", "active", "active", "unbound", "active"], dtype=h5py.string_dtype())
        s["command_accepted"] = [1, 1, 1, 0, 1]
    return directory, log_dir / "execution.h5"


def test_freeze_uses_only_available_targets_and_never_bridges_lost_binding(session):
    directory, _ = session
    _, frozen, info, _ = _load_targets(directory)
    np.testing.assert_array_equal(frozen["original_row"], [0, 0, 2])
    np.testing.assert_array_equal(frozen["source_time_us"], [990_000, 990_000, 1_030_000])
    assert (frozen["input_available_us"] <= frozen["host_monotonic_us"]).all()
    assert frozen["target_position"][1, 0] == .01  # .02 is available only at 25ms.
    assert info["first_segment_samples"] == 3 and info["original_active_samples"] == 4


def test_expired_target_is_rejected_instead_of_renewed(session):
    directory, path = session
    with h5py.File(path, "r+") as f:
        f["samples/source_time_us"][0] = 700_000
    with pytest.raises(ValueError, match="stale"):
        _load_targets(directory)


@pytest.mark.parametrize("change", [{"backend": "ur10e"}, {"options": {"orientation": True}},
                                  {"options": {"host": "192.168.0.100"}}])
def test_benchmark_rejects_hardware_and_unlogged_orientation(session, change):
    directory, _ = session
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if "options" in change:
        manifest["config"]["options"].update(change["options"])
    else:
        manifest["config"].update(change)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _load_targets(directory)


def test_existing_source_cannot_be_used_as_output(session):
    directory, _ = session
    with pytest.raises(ValueError, match="outside"):
        benchmark_live_ik(directory, directory / "results", [("baseline", LiveIKTuning())])


def test_open_log_is_not_a_valid_experiment_input(session):
    directory, path = session
    with h5py.File(path, "r+") as f:
        f.attrs["closed_cleanly"] = False
    with pytest.raises(ValueError, match="closed"):
        _load_targets(directory)


def test_failed_variant_retains_partial_trace_and_cannot_look_successful():
    class Backend:
        identity = {"backend": "fake"}
        closed = False

        def __init__(self, options):
            pass

        def read(self):
            return {"q": np.zeros(6), "qd": np.zeros(6)}

        def command(self, q):
            pass

        def advance(self, dt):
            pass

        def close(self):
            Backend.closed = True

    class IK:
        manifest = {"urdf_sha256": "fixture"}
        calls = 0

        def __init__(self, options, tuning):
            pass

        def pose(self, q):
            return np.zeros(3), np.eye(3)

        def step(self, q, command, v, p, r, dt):
            self.calls += 1
            if self.calls == 3:  # one cache warmup, one accepted sample, then failure
                raise RuntimeError("fixture rejection")
            return {"q": command, "velocity": v, "margins": {"joint": 1., "floor": 1., "collision": 1.}}

    result = _simulate(LiveControlOptions(), {"target_position": np.zeros((3, 3))},
        LiveIKTuning(), {"urdf_sha256": "fixture"}, backend_factory=Backend, retargeter_factory=IK)
    assert Backend.closed and not result["passed"] and not result["completed"]
    assert "fixture rejection" in result["error"]
    assert result["samples"] == 1 and result["trace"]["q_observed"].shape == (1, 6)
