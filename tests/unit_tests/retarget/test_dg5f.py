"""Camera adapter preserves the original Hybrid input, units and fault map."""
from pathlib import Path
import numpy as np
import pytest

from viki.contracts import LiveHandEstimate
from viki.retarget import DG5FHandRetargeter


def estimate(sequence=1, source=1_000_000):
    p = np.zeros((21,3))
    for root,y in zip((1,5,9,13,17),(-.045,-.03,0.,.025,.045)):
        for k in range(4):
            p[root+k] = [.03+.025*k,y,.01*k]
    return LiveHandEstimate(sequence,source,source,p,np.ones(21,bool),np.ones(21),p[0],np.eye(3),True,True,"test")


class Engine:
    def __init__(self):
        self.inputs = []

    def retarget(self, points):
        self.inputs.append(points.copy())
        q = np.arange(20,dtype=float); q[16] = 0
        return q,np.full(5,.8),np.full(7,.02)


def test_exact_xyz_and_degree_radian_boundary():
    e = Engine()
    wrapper = DG5FHandRetargeter(engine_factory=lambda:e,clock_us=lambda:1_100_000)
    original = estimate()
    target = wrapper.step(original)
    assert target.valid
    np.testing.assert_array_equal(e.inputs[0],original.points)
    np.testing.assert_allclose(target.q_rad,np.deg2rad(np.r_[np.arange(16),0,np.arange(17,20)]))
    assert target.valid_until_us == 1_250_000


@pytest.mark.parametrize("kind",["missing","nan","invalid","stale","future"])
def test_bad_input_never_reaches_hybrid_or_zero_fills(kind):
    e = Engine()
    obs = estimate()
    now = 1_100_000
    if kind == "missing": obs.observed[16] = False
    if kind == "nan": obs.points[16] = np.nan
    if kind == "invalid": obs.position_valid = False
    if kind == "stale": now = 1_300_000
    if kind == "future": now = 999_999
    wrapper = DG5FHandRetargeter(engine_factory=lambda:e,clock_us=lambda:now)
    result = wrapper.step(obs)
    assert not result.valid and not e.inputs
    assert np.isnan(result.q_rad).all()


def test_expired_solution_and_duplicate_are_not_published():
    now = [1_100_000]
    class Slow(Engine):
        def retarget(self,p):
            result = super().retarget(p)
            now[0] += 200_000
            return result
    wrapper = DG5FHandRetargeter(engine_factory=Slow,clock_us=lambda:now[0])
    assert wrapper.step(estimate()).reason == "expired_during_retarget"
    assert wrapper.step(estimate()).reason == "duplicate_or_out_of_order"


def test_loss_resets_warm_start_before_reacquisition():
    engines = []
    def factory():
        engines.append(Engine())
        return engines[-1]
    wrapper = DG5FHandRetargeter(engine_factory=factory,clock_us=lambda:1_200_000)
    assert wrapper.step(estimate()).valid
    bad = estimate(2,1_100_000); bad.observed[20] = False
    assert not wrapper.step(bad).valid
    assert wrapper.step(estimate(3,1_150_000)).valid
    assert len(engines) == 2 and len(engines[1].inputs) == 1


def test_original_hybrid_equivalence_when_external_source_available():
    root = Path("/opt/dg5f-hybrid")
    if not (root/"src/dg5f_demo/retarget.py").exists():
        pytest.skip("run in live-hand image with the original Hybrid source mounted")
    pytest.importorskip("dex_retargeting")
    adapter = DG5FHandRetargeter(root,clock_us=lambda:1_100_000)
    from dg5f_demo.retarget import HybridRetargeter
    reference = HybridRetargeter(low_pass_alpha=1.)
    obs = estimate()
    q_deg,weights,distances = reference.retarget(obs.points)
    actual = adapter.step(obs)
    assert actual.valid
    np.testing.assert_array_equal(actual.q_rad,np.deg2rad(q_deg))
    np.testing.assert_array_equal(actual.contact_weights,weights)
    np.testing.assert_array_equal(actual.fingertip_distances_m,distances)
