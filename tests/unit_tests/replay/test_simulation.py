"""MuJoCo mapping/validation on a small model, without robot connections."""

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from viki.contracts import Episode
from viki.replay import preview_mujoco
from viki.retarget import normalize_robot, write_hdf5_archive


@pytest.fixture
def scene(tmp_path, monkeypatch):
    ep = Episode(root=tmp_path / "ep")
    q = np.zeros((4, 6)); q[:, 0] = np.arange(4) * 0.001
    names = normalize_robot("ur10e").joint_names
    write_hdf5_archive(ep.plan_h5, {
        "q": q, "dt": 0.05, "robot": "ur10e_description",
        "joint_names_json": json.dumps(names), "gripper_model": "linear",
        "gripper_command": np.ones((4, 1)),
    })
    body = "".join(f'<body pos="0 0 0.1"><joint name="{name}" axis="0 0 1" range="-6.28 6.28"/><geom type="sphere" size="0.01"/>' for name in names)
    actuators = "".join(f'<position joint="{name}" kp="1000" kv="10"/>' for name in names)
    xml = tmp_path / "scene.xml"
    xml.write_text(f'<mujoco><compiler angle="radian"/><option integrator="implicitfast"/><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>{body}{"</body>"*6}</worldbody><actuator>{actuators}</actuator></mujoco>')
    def no_hardware(*_args, **_kwargs):
        pytest.fail("simulation tried connecting to hardware")
    monkeypatch.setattr("viki.replay.ur._control", no_hardware)
    monkeypatch.setattr("viki.replay.ur._receive", no_hardware)
    return ep, xml


@pytest.mark.parametrize("mode", ["kinematic", "position"])
def test_preview_records_model_scoped_report_and_no_hardware(scene, mode):
    ep, xml = scene
    report = preview_mujoco(ep, model_path=str(xml), speed_scale=0.1, mode=mode)
    assert report["completed"] and report["passed_model_checks"]
    assert report["max_penetration_m"] == 0
    assert "no real rig" in report["scope"]
    assert (ep.root / "intermediates/mujoco" / f"{mode}_report.json").exists()


def test_corrupted_model_cache_is_rejected(scene):
    ep, xml = scene
    (xml.parent / "source.json").write_text(json.dumps({"sha256": {"scene.xml": "bad"}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        preview_mujoco(ep, model_path=str(xml))


def test_unknown_joint_mapping_is_rejected(scene):
    ep, xml = scene
    xml.write_text(xml.read_text().replace("shoulder_pan_joint", "wrong_joint"))
    with pytest.raises(KeyError):
        preview_mujoco(ep, model_path=str(xml))
