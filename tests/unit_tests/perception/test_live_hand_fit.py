"""Causal calibration, tangent derivatives, model bounds and missing support."""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from viki.contracts import LiveHandFitOptions
from viki.perception import CausalHandFitter
from scripts.benchmark_live_anatomy_synthetic import generate


def calibrated():
    truth, _, _, _ = generate("isotropic", 0, n=12)
    fitter = CausalHandFitter(LiveHandFitOptions(warmup_frames=2))
    for i in range(2):
        result = fitter.update(truth[i], np.ones(21,bool), np.ones(21), 1+i*66667)
    assert result.accepted
    return fitter, truth[0], result


def test_fit_recovers_model_and_freezes_calibration():
    fitter, points, result = calibrated()
    assert np.max(np.linalg.norm(result.points-points,axis=1)) < .001
    digest = fitter.manifest["calibration_sha256"]
    for i in range(4):
        moved = points + [i*.001,0,0]
        result = fitter.update(moved,np.ones(21,bool),np.ones(21),200000+i*66667)
        assert result.joint_margin_rad >= -1e-8
    assert fitter.manifest["calibration_sha256"] == digest


def test_geometry_jacobian_nonzero_tangent():
    fitter, _, result = calibrated()
    delta = np.random.default_rng(2).normal(size=26)*.02
    _, _, jac = fitter._geometry(result.q,delta,np.arange(21))
    numerical = np.empty_like(jac)
    for col in range(26):
        eps = np.eye(26)[col]*1e-6
        plus = fitter._geometry(result.q,delta+eps,np.arange(21))[1]
        minus = fitter._geometry(result.q,delta-eps,np.arange(21))[1]
        numerical[:,col] = ((plus-minus)/2e-6).ravel()
    np.testing.assert_allclose(jac,numerical,atol=2e-7,rtol=2e-5)


def test_missing_noncontrol_can_be_inferred_but_control_cannot():
    fitter, points, _ = calibrated()
    mask = np.ones(21,bool); mask[13:17] = False
    partial = points.copy(); partial[~mask] = np.nan
    result = fitter.update(partial,mask,mask.astype(float),200000)
    assert result.accepted
    np.testing.assert_array_equal(result.inferred,~mask)
    mask[4] = False; partial[4] = np.nan
    result = fitter.update(partial,mask,mask.astype(float),300000)
    assert not result.accepted and result.reason == "missing_control_support"
    assert not np.isfinite(result.points).any()


def test_incomplete_calibration_never_uses_generic_finger():
    truth, _, _, _ = generate("isotropic", 0, n=3)
    fitter = CausalHandFitter(LiveHandFitOptions(warmup_frames=2))
    mask = np.ones(21,bool); mask[16] = False
    for i in range(3):
        result = fitter.update(truth[i],mask,mask.astype(float),i+1)
        assert result.reason == "calibration_incomplete"
    assert fitter.hand is None
