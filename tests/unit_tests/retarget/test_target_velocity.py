"""Observation cadence, cap, fading and loss are distinct from servo cadence."""

from dataclasses import replace

import numpy as np
import pytest

from viki.contracts import LiveIKTuning
from viki.retarget import CausalTargetVelocity


def test_velocity_uses_source_interval_not_time_between_servo_reads():
    estimator = CausalTargetVelocity(LiveIKTuning(feedforward_gain=1.))
    np.testing.assert_array_equal(estimator.update([0, 0, 0], 1_000_000, 1_090_000), 0)
    first = estimator.update([.005, 0, 0], 1_100_000, 1_190_000)
    assert 0 < first[0] < .05
    # Repeated reads do not differentiate the same observation again.
    np.testing.assert_array_equal(estimator.update([.005, 0, 0], 1_100_000, 1_210_000), first)
    np.testing.assert_array_equal(estimator.update([.005, 0, 0], 1_100_000, 1_320_000), 0)
    assert estimator.arrival_us == 1_190_000


def test_velocity_norm_is_capped_and_reset_discards_previous_direction():
    estimator = CausalTargetVelocity(LiveIKTuning(feedforward_max_speed=.05))
    estimator.update([0, 0, 0], 0, 0)
    velocity = estimator.update([1, 1, 1], 100_000, 100_000)
    assert 0 < np.linalg.norm(velocity) <= .05
    estimator.reset()
    np.testing.assert_array_equal(estimator.update([-1, 0, 0], 200_000, 200_000), 0)
    np.testing.assert_array_equal(estimator.update([1, 0, 0], 500_001, 500_001), 0)


def test_same_source_cannot_gain_new_position_or_go_backwards():
    estimator = CausalTargetVelocity(LiveIKTuning())
    estimator.update([0, 0, 0], 100, 200)
    with pytest.raises(ValueError, match="same source"):
        estimator.update([1, 0, 0], 100, 201)
    with pytest.raises(ValueError, match="backwards"):
        estimator.update([0, 0, 0], 99, 202)
    with pytest.raises(ValueError, match="chronology"):
        estimator.update([0, 0, 0], 300, 203)


@pytest.mark.parametrize("changes", [{"feedforward_gain": 2}, {"feedforward_max_speed": .2},
    {"feedforward_hold_s": .3}, {"feedforward_cutoff_hz": 0}])
def test_invalid_feedforward_tuning_is_rejected(changes):
    with pytest.raises(ValueError):
        CausalTargetVelocity(replace(LiveIKTuning(), **changes))
