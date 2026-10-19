"""A synthetic or incomplete replay cannot become robot training evidence."""

import hashlib
import numpy as np
import pytest

from viki.contracts import Episode, EpisodeLabels
from viki.episode import mark_stage
from viki.export.run import _eligible
from viki.labeling import save_labels
from viki.retarget import write_hdf5_archive


@pytest.mark.parametrize("verdict", ["dry-run", "arm-only", "reject", None])
def test_only_pass_can_be_eligible(tmp_path, verdict):
    ep = Episode(root=tmp_path)
    mark_stage(ep, "retarget")
    mark_stage(ep, "replay", verdict=verdict)
    ok, reason = _eligible(ep)
    assert not ok and "measured complete replay" in reason


@pytest.mark.parametrize("missing", ["hardware", "completed", "gripper_measured", "plan_sha256"])
def test_claimed_pass_requires_measured_artifact_and_matching_plan(tmp_path, missing):
    ep = Episode(root=tmp_path)
    write_hdf5_archive(ep.plan_h5, {"q": np.zeros((2, 6))})
    replay = {
        "verdict": "pass", "hardware": True, "completed": True,
        "gripper_measured": True,
        "plan_sha256": hashlib.sha256(ep.plan_h5.read_bytes()).hexdigest(),
    }
    replay.pop(missing)
    write_hdf5_archive(ep.replay_h5, replay)
    mark_stage(ep, "retarget")
    mark_stage(ep, "replay", verdict="pass")
    assert not _eligible(ep)[0]


def test_changed_plan_invalidates_replay_for_training(tmp_path):
    ep = Episode(root=tmp_path)
    write_hdf5_archive(ep.plan_h5, {"q": np.zeros((2, 6))})
    write_hdf5_archive(ep.replay_h5, {
        "verdict": "pass", "hardware": True, "completed": True,
        "gripper_measured": True,
        "plan_sha256": hashlib.sha256(ep.plan_h5.read_bytes()).hexdigest(),
    })
    mark_stage(ep, "retarget")
    mark_stage(ep, "replay", verdict="pass")
    save_labels(ep, EpisodeLabels(task="pick cube", outcome="good"))
    assert _eligible(ep)[0]
    write_hdf5_archive(ep.plan_h5, {"q": np.ones((2, 6))})
    assert _eligible(ep) == (False, "replay belongs to a different plan")
