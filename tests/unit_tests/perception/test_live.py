"""Real triangulation, causal filtering and liveness with injected cameras."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.contracts import Frame, HandDetection, SyncedFrameGroup
from viki.perception.live import CausalLandmarkFilter, LiveCameraGeometry, LiveHandEstimator, load_live_geometry
from viki.perception.triangulate import _Cam


def skeleton():
    points = np.array([[.01 * (i % 4), .015 * (i // 4), 1.] for i in range(21)])
    points[0] = [0, 0, 1]
    points[5] = [-.03, .07, 1]
    points[9] = [0, .08, 1]
    points[17] = [.04, .07, 1]
    points[4] = [-.04, .10, 1]
    points[8] = [-.03, .12, 1]
    return points


class Detector:
    def __init__(self, camera, points, shift=0):
        self.camera, self.points, self.shift = camera, points, shift
        self.missing = set()
        self.calls = 0
        self.closed = False

    def detect(self, frame, hand):
        self.calls += 1
        return HandDetection(
            points={i: self.camera.project(p)[0] + [self.shift, 0]
                    for i, p in enumerate(self.points) if i not in self.missing},
            lm_z_rel=np.zeros(21), confidence=1., device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )

    def close(self):
        self.closed = True


def estimator(points=None, *, warmup=1, shift=0, anchor="pinch_center"):
    points = skeleton() if points is None else points
    contexts, detectors = {}, {}
    intr = {"color": {"fx": 300, "fy": 300, "cx": 320, "cy": 240,
                      "height": 480, "width": 640},
            "depth": {"height": 2, "width": 2}}
    for dev, x in (("a", -.15), ("b", .15)):
        T = np.eye(4)
        T[0, 3] = x
        camera = _Cam(dev, {"K": [[300, 0, 320], [0, 300, 240], [0, 0, 1]],
                           "dist": np.zeros(5), "T_wc": T, "image_size": [640, 480]})
        projector = SimpleNamespace(color_pixel_to_ray=lambda u, v:
                                    np.array([(u - 320 - shift) / 300, (v - 240) / 300, 1.]))
        contexts[dev] = LiveCameraGeometry(dev, camera, projector, intr, {}, "fake")
        detectors[dev] = Detector(camera, points, shift)
    clock = [1_000_000]
    instance = LiveHandEstimator(contexts, detectors, calibration_id="fixed-calibration",
                                 warmup_frames=warmup, clock_us=lambda: clock[0],
                                 target_anchor=anchor)
    return instance, detectors, clock


def group(host, native=None):
    return SyncedFrameGroup(
        frames={d: Frame(np.zeros((480, 640, 3), np.uint8), np.zeros((2, 2), np.uint16),
                         native or host, d, host_timestamp_us=host) for d in ("a", "b")},
        sync_timestamp_us=host, offsets_us={"a": 0, "b": 0},
    )


def test_exact_factory_rays_are_corrected_once_and_all21_are_observed():
    est, detectors, clock = estimator(shift=18.)
    result = est.process(group(clock[0]))
    assert result.position_valid and result.orientation_valid
    assert result.observed.all()
    np.testing.assert_allclose(result.points, skeleton(), atol=1e-6)
    np.testing.assert_allclose(result.rotation.T @ result.rotation, np.eye(3), atol=1e-6)
    assert result.calibration_id == "fixed-calibration"
    assert all(d.calls == 1 for d in detectors.values())


def test_raw_capture_keeps_pre_filter_geometry_and_default_profile():
    points = skeleton()
    est, _, clock = estimator(points)
    est.capture_raw = True
    first = est.process(group(clock[0]))
    saved = first.raw_points.copy()
    points[:, 0] += .01
    clock[0] += 66_667
    current = est.process(group(clock[0]))
    np.testing.assert_allclose(current.raw_points, points, atol=1e-6)
    assert np.linalg.norm(current.points[0]-current.raw_points[0]) > .001
    np.testing.assert_array_equal(first.raw_points, saved)
    assert est.manifest["profile"] == "live-human-sdk-rays-causal-v1"


def test_filter_prefix_is_causal_and_loss_reset_cannot_keep_old_velocity():
    raw = np.tile(skeleton(), (6, 1, 1))
    raw[:, :, 0] += np.arange(6)[:, None]*.005
    a, b = CausalLandmarkFilter(1., 10.), CausalLandmarkFilter(1., 10.)
    for i in range(4):
        np.testing.assert_array_equal(a.update(raw[i], np.ones(21, bool), 1_000_000+i*66_667),
                                      b.update(raw[i], np.ones(21, bool), 1_000_000+i*66_667))
    a.update(raw[-1]+10, np.ones(21, bool), 1_400_000)
    b.reset()
    reacquired = b.update(raw[-1], np.ones(21, bool), 1_400_000)
    np.testing.assert_array_equal(reacquired, raw[-1])


@pytest.mark.parametrize("recovery", [False, True])
def test_large_orientation_reacquires_only_after_three_coherent_fresh_frames(recovery):
    points = skeleton()
    est, detectors, clock = estimator(points)
    est.orientation_recovery = recovery
    assert est.process(group(clock[0])).orientation_valid
    # Only orientation is lost: wrist/pinch observations remain valid.
    for detector in detectors.values():
        detector.missing.add(17)
    clock[0] += 300_000
    assert not est.process(group(clock[0])).orientation_valid
    for detector in detectors.values():
        detector.missing.clear()
    wrist = points[0].copy()
    points[:] = (points-wrist) @ Rotation.from_rotvec([1.8, 0., 0.]).as_matrix().T+wrist
    outcomes = []
    for _ in range(3):
        clock[0] += 66_667
        outcomes.append(est.process(group(clock[0])).orientation_valid)
    assert outcomes == ([False, False, True] if recovery else [False, False, False])


def test_quality_weight_reduces_response_to_uncertain_observation_without_filling_missing():
    points = skeleton()
    a, b = CausalLandmarkFilter(2., .5), CausalLandmarkFilter(2., .5, quality_power=1.)
    observed = np.ones(21, dtype=bool)
    a.update(points, observed, 1_000_000)
    b.update(points, observed, 1_000_000)
    moved = points+np.array([.01, 0., 0.])
    observed[20] = False
    normal = a.update(moved, observed, 1_066_667, quality=np.full(21, .2))
    uncertain = b.update(moved, observed, 1_066_667, quality=np.full(21, .2))
    assert 0 < uncertain[0, 0]-points[0, 0] < normal[0, 0]-points[0, 0]
    assert np.isnan(uncertain[20]).all()


def test_orientation_position_loss_requires_confirmation_in_recovery_mode():
    est, detectors, clock = estimator()
    est.orientation_recovery = True
    assert est.process(group(clock[0])).orientation_valid
    for detector in detectors.values():
        detector.missing.add(4)
    clock[0] += 66_667
    assert not est.process(group(clock[0])).position_valid
    for detector in detectors.values():
        detector.missing.clear()
    states = []
    for _ in range(3):
        clock[0] += 66_667
        states.append(est.process(group(clock[0])).orientation_valid)
    assert states == [False, False, True]


def test_degenerate_palm_is_invalid_before_filtering():
    points = skeleton()
    est, _, clock = estimator(points)
    assert est.process(group(clock[0])).orientation_valid
    points[5] = [0, .04, 1]
    points[17] = [0, .10, 1]
    clock[0] += 66_667
    result = est.process(group(clock[0]))
    assert result.reason == "degenerate_palm" and not result.orientation_valid
    assert not result.position_valid and np.isnan(result.rotation).all()


def test_missing_palm_interrupts_orientation_recovery_confirmation():
    points = skeleton()
    est, detectors, clock = estimator(points)
    est.orientation_recovery = True
    assert est.process(group(clock[0])).orientation_valid
    wrist = points[0].copy()
    points[:] = (points-wrist) @ Rotation.from_rotvec([1.8, 0, 0]).as_matrix().T+wrist
    clock[0] += 66_667
    assert not est.process(group(clock[0])).orientation_valid
    for detector in detectors.values():
        detector.missing.add(17)
    clock[0] += 66_667
    assert not est.process(group(clock[0])).orientation_valid
    for detector in detectors.values():
        detector.missing.clear()
    recovered = []
    for _ in range(3):
        clock[0] += 66_667
        recovered.append(est.process(group(clock[0])).orientation_valid)
    assert recovered == [False, False, True]


def test_bone_projection_calibrates_from_prefix_and_preserves_missingness():
    from scripts.benchmark_live_bones import project_stream

    points = np.zeros((21, 3))
    for finger, root in enumerate((1, 5, 9, 13, 17)):
        for joint in range(4):
            points[root+joint] = [(finger-2)*.015, .04+joint*.02, 0.]
    raw = np.tile(points, (20, 1, 1))
    observed = np.ones((20, 21), dtype=bool)
    observed[12, 8] = False
    raw[12, 8] = np.nan
    valid = np.ones(20, dtype=bool)
    original = project_stream(raw, raw.copy(), observed, valid)
    future = raw.copy()
    future[15:, 4, 1] += .1
    changed = project_stream(future, future.copy(), observed, valid)
    np.testing.assert_allclose(original[0][:15], changed[0][:15], equal_nan=True)
    assert not original[1][:9].any() and original[1][9]
    assert not original[1][12] and np.isnan(original[0][12]).all()


def test_duplicate_stale_out_of_order_frames_never_call_detector_or_refresh_source():
    est, detectors, clock = estimator()
    first = est.process(group(clock[0], native=10))
    clock[0] += 50_000
    assert est.process(group(clock[0], native=10)) is None
    assert est.process(group(clock[0], native=9)) is None
    assert est.process(group(clock[0] - 300_000, native=11)) is None
    assert est.process(group(clock[0] + 1, native=11)) is None
    assert all(d.calls == 1 for d in detectors.values())
    assert first.source_time_us == 1_000_000


def test_missing_palm_has_no_identity_or_held_rotation():
    est, detectors, clock = estimator()
    assert est.process(group(clock[0])).orientation_valid
    for detector in detectors.values():
        detector.missing.add(17)
    clock[0] += 33_000
    result = est.process(group(clock[0]))
    assert result.position_valid
    assert not result.orientation_valid
    assert np.isnan(result.rotation).all()
    assert np.isnan(result.points[17]).all()
    assert not result.observed[17]


def test_missing_position_does_not_repeat_last_target():
    est, detectors, clock = estimator()
    assert est.process(group(clock[0])).position_valid
    for detector in detectors.values():
        detector.missing.add(4)
    clock[0] += 33_000
    result = est.process(group(clock[0]))
    assert not result.position_valid
    assert result.reason == "missing_position"
    assert np.isnan(result.position).all()


def test_warmup_references_are_frozen_and_bones_jump_is_rejected():
    points = skeleton()
    est, detectors, clock = estimator(points, warmup=2)
    assert est.process(group(clock[0])).reason == "warming_up"
    clock[0] += 33_000
    assert est.process(group(clock[0])).position_valid
    reference = est._palm_reference.copy()
    points[9, 1] = .145
    clock[0] += 33_000
    result = est.process(group(clock[0]))
    assert not result.position_valid and result.reason == "palm_reference"
    np.testing.assert_array_equal(est._palm_reference, reference)


def test_causal_output_prefix_does_not_depend_on_future_hand_motion():
    a, _, ca = estimator(skeleton(), warmup=2)
    b, _, cb = estimator(skeleton(), warmup=2)
    prefix_a, prefix_b = [], []
    for index in range(5):
        ca[0] = cb[0] = 1_000_000 + index * 33_000
        prefix_a.append(a.process(group(ca[0])))
        prefix_b.append(b.process(group(cb[0])))
        for detector in a.detectors.values():
            detector.points[:, 0] += .001
        for detector in b.detectors.values():
            detector.points[:, 0] += .001
    for detector in b.detectors.values():
        detector.points[:, 0] += .2
    ca[0] = cb[0] = 1_200_000
    a.process(group(ca[0]))
    b.process(group(cb[0]))
    for x, y in zip(prefix_a, prefix_b):
        np.testing.assert_array_equal(x.points, y.points)
        assert x.position_valid == y.position_valid and x.source_time_us == y.source_time_us


def test_skew_and_processing_age_guards_publish_invalid_observation():
    est, detectors, clock = estimator()
    paired = group(clock[0])
    paired.frames["b"].host_timestamp_us -= 25_000
    result = est.process(paired)
    assert result.reason == "camera_skew" and not result.position_valid
    assert all(d.calls == 0 for d in detectors.values())
    clock[0] += 33_000
    original = detectors["b"].detect

    def slow(*args):
        detection = original(*args)
        clock[0] += 300_000
        return detection

    detectors["b"].detect = slow
    result = est.process(group(clock[0]))
    assert result.reason == "processing_stale" and not result.position_valid


def test_missing_cached_calibration_is_an_error_without_guessing(tmp_path):
    import json
    raw = tmp_path / "raw"
    raw.mkdir()
    cameras = {d: {"type": "kinect"} for d in ("a", "b")}
    (tmp_path / "meta.json").write_text(json.dumps({"cameras": cameras}))
    intr = {d: {"color": {"fx": 300, "fy": 300, "cx": 320, "cy": 240,
                          "height": 480, "width": 640}} for d in cameras}
    (raw / "intrinsics.json").write_text(json.dumps(intr))
    (raw / "extrinsics.json").write_text(json.dumps({d: {"rvec": [0, 0, 0],
                                                         "tvec": [0, 0, 0]} for d in cameras}))
    with pytest.raises(RuntimeError, match="no fallback"):
        load_live_geometry(tmp_path)


def test_live_camera_verification_refuses_different_physical_device():
    import hashlib
    est, _, _ = estimator()
    reference = {"type": "kinect", "color_shape": [480, 640, 3], "depth_shape": [2, 2],
                 "requested": {"depth_mode": "NFOV_UNBINNED", "color_width": 640,
                               "color_height": 480}}
    good_blob = b"recorded factory calibration"
    for context in est.geometry.values():
        context.reference = reference
        context.factory_hash = hashlib.sha256(good_blob).hexdigest()
    info = {"running": True, "has_frame": True, "color_shape": [480, 640, 3],
            "depth_shape": [2, 2], "config": reference["requested"]}
    blob = [good_blob]
    manager = SimpleNamespace(get_info=lambda d: info,
                              get_backend=lambda d: SimpleNamespace(get_raw_calibration=lambda: blob[0]))
    est.verify_cameras(manager)
    blob[0] = b"factory calibration of another physical camera"
    with pytest.raises(RuntimeError, match="device/order mismatch"):
        est.verify_cameras(manager)


def test_position_jump_rejected_before_causal_filter_can_hide_it():
    points = skeleton()
    est, _, clock = estimator(points)
    assert est.process(group(clock[0])).position_valid
    points[:, 0] += .2
    clock[0] += 33_000
    result = est.process(group(clock[0]))
    assert result.reason == "position_jump" and not result.position_valid
    assert np.isnan(result.position).all()
