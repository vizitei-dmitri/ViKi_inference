"""The independent finger display cannot consume expired or invalid targets."""
import numpy as np
import pytest

from viki.contracts import HandJointTarget
from viki.live import LiveFingerSession


def test_latest_target_expiry_and_invalid_replace_previous(tmp_path):
    s = LiveFingerSession("unused",out=tmp_path)
    t = HandJointTarget(1,100,200,(),np.zeros(20),np.zeros(5),np.zeros(7),True)
    s.latest = t
    assert s.current(150) is t
    assert s.current(201) is None
    t.valid = False
    assert s.current(150) is None


@pytest.mark.parametrize("settings",[{"source":"oculus"},{"perception_hz":0},{"ttl_s":2},{"seconds":float("nan")}])
def test_invalid_settings_rejected(tmp_path,settings):
    with pytest.raises(ValueError):
        LiveFingerSession("unused",out=tmp_path,**settings)
