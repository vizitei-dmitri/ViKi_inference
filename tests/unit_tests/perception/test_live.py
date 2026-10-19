"""Real triangulation, causal filtering and liveness with injected cameras."""

from types import SimpleNamespace

import numpy as np
import pytest

from viki.contracts import Frame, HandDetection, SyncedFrameGroup
from viki.perception.live import LiveCameraGeometry, LiveHandEstimator, load_live_geometry
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
