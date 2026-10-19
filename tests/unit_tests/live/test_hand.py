"""The independent finger display cannot consume expired or invalid targets."""
import numpy as np
import pytest
from types import SimpleNamespace

from viki.contracts import HandJointTarget
from viki.live import LiveFingerSession
from viki.live import run_hand_cli


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


@pytest.mark.parametrize("backend,execute,source,auto_arm",[
    ("tesollo",False,"cameras",False),
    ("tesollo",True,"recorded",False),
    ("tesollo",True,"cameras",True),
    ("mock",True,"recorded",False),
])
def test_cli_rejects_unauthorized_hardware_paths_before_start(backend,execute,source,auto_arm):
    with pytest.raises(ValueError):
        run_hand_cli(SimpleNamespace(backend=backend,execute=execute,source=source,auto_arm=auto_arm))
