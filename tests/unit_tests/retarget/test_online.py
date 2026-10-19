"""Command continuity tests with independently inspectable toy geometry."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.contracts import LiveControlOptions, LiveIKTuning
from viki.retarget import OnlineIKError, OnlineRetargeter


class CartesianArm:
    nq = 6
    q_min = np.full(6, -2 * np.pi)
    q_max = np.full(6, 2 * np.pi)
    velocity_limit = np.ones(6)

    def pose(self, q):
        return np.asarray(q[:3]).copy(), Rotation.from_rotvec(q[3:]).as_matrix()

    def pose_jacobian(self, q):
        p, r = self.pose(q)
        jp, jr = np.zeros((3, 6)), np.zeros((3, 6))
        jp[:, :3] = np.eye(3)
        jr[:, 3:] = np.eye(3)
        return p, r, jp, jr

    def floor_margins(self, q):
        return np.array([1.0 + q[2]])

    def floor_linearization(self, q):
        row = np.zeros((1, 6))
        row[0, 2] = 1.0
        return self.floor_margins(q), row

    def collision_margins(self, q):
        return np.array([1.0])

    def collision_linearization(self, q):
        return self.collision_margins(q), np.zeros((1, 6))


@pytest.fixture
def options():
    return LiveControlOptions(frequency=125, initial_q=(0.,) * 6)


def test_same_fk_reference_produces_no_initial_correction(options):
    arm = CartesianArm()
    ik = OnlineRetargeter(options=options, kinematics=arm)
    q = np.zeros(6)
    p, r = ik.pose(q)
    result = ik.step(q, q, np.zeros(6), p, r, 0.008)
    np.testing.assert_allclose(result["q"], q, atol=1e-15)
    np.testing.assert_allclose(result["velocity"], 0, atol=1e-15)
    assert result["position_error_m"] == 0
    assert result["orientation_error_rad"] == 0
    assert result["margins"]["floor"] == 1


@pytest.mark.parametrize("field,value", [("position_gain", 0), ("orientation_gain", -1),
    ("damping", 0), ("damping", float("nan")), ("acceleration_weight", -1)])
def test_invalid_objective_tuning_is_rejected_before_model_use(options, field, value):
    with pytest.raises(ValueError, match=field):
        OnlineRetargeter(options, CartesianArm(), tuning=replace(LiveIKTuning(), **{field: value}))


def test_gain_experiment_improves_unsaturated_tracking_without_relaxing_bounds(options):
    errors = []
    for gain in (6., 15.):
        ik = OnlineRetargeter(options, CartesianArm(), tuning=LiveIKTuning(position_gain=gain))
        q, v = np.zeros(6), np.zeros(6)
        for _ in range(150):
            row = ik.step(q, q, v, np.array([.01, 0., 0.]), np.eye(3), .008)
            assert np.max(np.abs(row["velocity"]-v)) <= options.max_acceleration*.008+1e-10
            assert np.max(np.abs(row["velocity"])) <= options.max_velocity+1e-10
            q, v = row["q"], row["velocity"]
        errors.append(row["position_error_m"])
        assert ik.manifest["objective_tuning"]["position_gain"] == gain
    assert errors[1] < errors[0]/2


def test_target_replacement_and_stop_bound_every_accepted_sample(options):
    ik = OnlineRetargeter(options, CartesianArm())
    q, previous = np.zeros(6), np.zeros(6)
    dt = 0.008
    velocities = []
    for index in range(200):
        target = np.array([0.1 if index < 100 else -0.1, 0., 0.])
        result = ik.step(q, q, previous, target, np.eye(3), dt)
        velocity = result["velocity"]
        np.testing.assert_allclose(result["q"] - q, dt * velocity, atol=1e-14)
        assert np.max(np.abs(velocity)) <= options.max_velocity + 1e-10
        assert np.max(np.abs(velocity - previous)) <= options.max_acceleration * dt + 1e-10
        q, previous = result["q"], velocity
        velocities.append(velocity)
    for _ in range(100):
        result = ik.brake(q, previous, dt)
        assert np.max(np.abs(result["velocity"] - previous)) <= options.max_acceleration * dt + 1e-10
        assert np.all(np.abs(result["velocity"]) <= np.abs(previous) + 1e-10)
        q, previous = result["q"], result["velocity"]
        if np.max(np.abs(previous)) == 0:
            break
    np.testing.assert_array_equal(previous, np.zeros(6))
    assert velocities[0][0] > 0
    # Sudden target reversal does not instantly reverse a travelling arm.
    assert abs(velocities[100][0] - velocities[99][0]) <= 0.004 + 1e-10


def test_physical_winding_is_not_wrapped_or_pulled_to_home(options):
    q = np.array([5.8, 0., 0., 0., 0., 4.9])
    ik = OnlineRetargeter(replace(options, initial_q=tuple(q)), CartesianArm())
    p, r = ik.pose(q)
    for _ in range(20):
        result = ik.step(q, q, np.zeros(6), p, r, 0.008)
        np.testing.assert_allclose(result["q"], q, atol=1e-12)
    assert result["q"][0] > np.pi
    assert result["q"][5] > np.pi


def test_fk_uses_measurement_but_integration_uses_command_boundary(options):
    class RecordedArm(CartesianArm):
        seen = None

        def pose_jacobian(self, q):
            self.seen = q.copy()
            return super().pose_jacobian(q)

    arm = RecordedArm()
    ik = OnlineRetargeter(options, arm)
    measured = np.zeros(6)
    command = np.array([0.002, 0., 0., 0., 0., 0.])
    previous = np.array([0.1, 0., 0., 0., 0., 0.])
    result = ik.step(measured, command, previous, np.array([0.05, 0., 0.]), np.eye(3), 0.008)
    np.testing.assert_array_equal(arm.seen, measured)
    np.testing.assert_allclose(result["q"], command + 0.008 * result["velocity"])
    assert abs(result["velocity"][0] - previous[0]) <= 0.004 + 1e-10


def test_already_violated_floor_and_collision_are_faults(options):
    class CollidingArm(CartesianArm):
        def collision_margins(self, q):
            return np.array([-0.001])

    with pytest.raises(OnlineIKError, match="collision"):
        OnlineRetargeter(options, CollidingArm())
    with pytest.raises(OnlineIKError, match="floor"):
        OnlineRetargeter(replace(options, initial_q=(0., 0., -1.1, 0., 0., 0.)), CartesianArm())


def test_missing_constraints_do_not_pretend_model_passed(options):
    arm = CartesianArm()
    arm.floor_margins = lambda q: np.empty(0)
    with pytest.raises(OnlineIKError, match="missing"):
        OnlineRetargeter(options, arm)


def test_nonlinear_midpoint_violation_is_rejected_even_when_qp_accepts(options):
    class InteriorCollision(CartesianArm):
        def collision_margins(self, q):
            return np.array([abs(q[0] - 0.000016) - 0.000008])

        def collision_linearization(self, q):
            # A deliberately inaccurate local approximation: exact nonlinear
            # admission checks must still catch the interior violation.
            return np.ones(1), np.zeros((1, 6))

    ik = OnlineRetargeter(options, InteriorCollision())
    with pytest.raises(OnlineIKError, match="collision"):
        ik.step(np.zeros(6), np.zeros(6), np.zeros(6), np.array([0.1, 0., 0.]), np.eye(3), 0.008)


def test_linear_floor_barrier_and_brake_margins(options):
    ik = OnlineRetargeter(options, CartesianArm())
    q = np.array([0., 0., -1., 0., 0., 0.])
    result = ik.step(q, q, np.zeros(6), np.array([0., 0., -2.]), np.eye(3), 0.008)
    assert result["q"][2] >= -1. - 1e-10
    assert result["margins"]["floor"] >= -1e-10
    with pytest.raises(OnlineIKError, match="floor"):
        ik.brake(q, np.array([0., 0., -0.1, 0., 0., 0.]), 0.008)


@pytest.mark.parametrize("dt", [0., -0.01, 0.02, float("nan"), float("inf")])
def test_no_catch_up_step_or_invalid_dt(options, dt):
    ik = OnlineRetargeter(options, CartesianArm())
    with pytest.raises(OnlineIKError, match="dt"):
        ik.brake(np.zeros(6), np.zeros(6), dt)


def test_tracking_fault_never_causes_recovery_jump(options):
    ik = OnlineRetargeter(options, CartesianArm())
    measured = np.zeros(6)
    command = np.array([0.06, 0., 0., 0., 0., 0.])
    with pytest.raises(OnlineIKError, match="tracking"):
        ik.step(measured, command, np.zeros(6), np.zeros(3), np.eye(3), 0.008)


def test_orientation_motion_has_bounded_history_and_reduces_geodesic_error(options):
    ik = OnlineRetargeter(options, CartesianArm())
    q, previous = np.zeros(6), np.zeros(6)
    target = Rotation.from_rotvec([0.02, -0.01, 0.015]).as_matrix()
    for _ in range(60):
        result = ik.step(q, q, previous, np.zeros(3), target, 0.008)
        assert np.max(np.abs(result["velocity"] - previous)) <= 0.004 + 1e-10
        q, previous = result["q"], result["velocity"]
    assert result["orientation_error_rad"] < np.linalg.norm([0.02, -0.01, 0.015])
    np.testing.assert_allclose(q[:3], 0, atol=1e-12)


def test_nonfinite_history_and_invalid_rotation_cannot_form_commands(options):
    ik = OnlineRetargeter(options, CartesianArm())
    with pytest.raises(ValueError, match="previous_velocity"):
        ik.brake(np.zeros(6), np.full(6, np.nan), 0.008)
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        ik.step(np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(3), np.eye(3) * 1.001, 0.008)


def test_joint_stopping_envelope_rejects_state_that_cannot_brake(options):
    ik = OnlineRetargeter(options, CartesianArm())
    q = np.array([2 * np.pi - 0.0001, 0., 0., 0., 0., 0.])
    previous = np.array([0.1, 0., 0., 0., 0., 0.])
    p, r = ik.pose(q)
    with pytest.raises(OnlineIKError, match="stopping bounds"):
        ik.step(q, q, previous, p + np.array([1., 0., 0.]), r, 0.008)


def test_native_ur10e_fk_initial_equilibrium_and_short_live_motion():
    # The Docker image supplies Pinocchio and the local robot model cache. This
    # exercises the real mesh floor supports and all retained collision pairs.
    pytest.importorskip("pinocchio")
    options = LiveControlOptions()
    ik = OnlineRetargeter(options)
    q = np.asarray(options.initial_q)
    p, r = ik.pose(q)
    dt = 1. / options.frequency
    result = ik.step(q, q, np.zeros(6), p, r, dt)
    np.testing.assert_allclose(result["q"], q, atol=1e-12)
    previous = np.zeros(6)
    for _ in range(25):
        result = ik.step(q, q, previous, p + np.array([0.01, 0., 0.]), r, dt)
        assert min(result["margins"].values()) >= -1e-9
        assert np.max(np.abs(result["velocity"] - previous)) <= options.max_acceleration * dt + 1e-9
        q, previous = result["q"], result["velocity"]
    assert np.linalg.norm(ik.pose(q)[0] - p) > 0
    scene = ik.scene(q)
    assert len(scene["points"]) > 6
    assert scene["frame"] == "robot_base"
    assert len(scene["edges"]) == len(scene["points"]) - 1
    assert ik.manifest["task_frame"] == "tool0"
    assert len(ik.manifest["urdf_sha256"]) == 64


def test_native_broadphase_bounds_distance_and_analytic_jacobian():
    pytest.importorskip("pinocchio")
    ik = OnlineRetargeter(LiveControlOptions())
    model = ik.kinematics
    q = np.asarray(ik.options.initial_q)
    rng = np.random.default_rng(42)
    for _ in range(5):
        angles = q + rng.uniform(-0.03, 0.03, 6)
        admitted = model.collision_margins(angles)
        # Independent all-pair native distances prove the broadphase result
        # never overstates body separation, including pairs it didn't query.
        model.pin.computeDistances(model.collision_model, model.collision_data)
        exact = np.array([result.min_distance - ik.options.collision_distance_m
                          for result in model.collision_data.distanceResults])
        assert np.all(admitted <= exact + 1e-9)
        assert np.all((admitted < 0) == (exact < 0))
    # Avoid symmetric AABB edge ties when checking an ordinary derivative;
    # exactly at a tie the analytical result is a valid one-sided subgradient.
    q = q + np.array([0.013, -0.016, 0.009, 0.017, -0.011, 0.014])
    margins, jacobian = model.collision_linearization(q)
    finite_difference = np.empty_like(jacobian)
    epsilon = 1e-6
    for joint in range(6):
        delta = np.zeros(6)
        delta[joint] = epsilon
        finite_difference[:, joint] = (
            model.collision_margins(q + delta)
            - model.collision_margins(q - delta)) / (2 * epsilon)
    np.testing.assert_allclose(jacobian, finite_difference, atol=2e-4, rtol=2e-3)
