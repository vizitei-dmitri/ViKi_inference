"""State-local, bounded IK for the live bare UR10e arm.

This API consumes an already mapped robot-base pose. It does not fetch camera
frames, renew an observation's expiry, wrap joint angles, or connect to a robot.
The caller owns command history: update it only after command acceptance.
"""

from __future__ import annotations

import hashlib
import importlib
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from viki.contracts import LiveControlOptions, LiveIKTuning
from viki.retarget.robots import normalize_robot
from viki.retarget.run import _load_robot_description
from viki.retarget.solver import (PinocchioKinematics, _floor_support_descriptor,
                                 _local_floor_support)


class OnlineIKError(RuntimeError):
    """An input state or proposed live command cannot be admitted."""


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _rotation(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (3, 3) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(result), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("rotation must be a proper SO(3) matrix")
    return result


class _LivePinocchioKinematics(PinocchioKinematics):
    """Full pair scope with conservative AABB culling before FCL distances.

    Pink's generic configuration queries every mesh pair even when separated
    by half a metre. A distance between enclosing world AABBs lower-bounds
    body distance; pairs cleared by that bound need no mesh narrowphase. Near
    pairs still use FCL's signed distance and nearest-point derivatives.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._live_descriptors = tuple(_floor_support_descriptor(g.geometry)
                                       for g in self.collision_model.geometryObjects)
        self._live_cache = OrderedDict()

    def _live_collision_query(self, q):
        angles = _vector(q, 6, "q")
        cache_key = angles.tobytes()
        if cache_key in self._live_cache:
            self._live_cache.move_to_end(cache_key)
            return self._live_cache[cache_key]
        self.pin.forwardKinematics(self.model, self.data, self._configuration(angles))
        self.pin.updateGeometryPlacements(self.model, self.data,
                                          self.collision_model, self.collision_data)
        low, high, low_points, high_points = [], [], [], []
        axes = np.eye(3)
        for geometry, placement, descriptor in zip(
                self.collision_model.geometryObjects, self.collision_data.oMg,
                self._live_descriptors):
            rotation = np.asarray(placement.rotation)
            translation = np.asarray(placement.translation)
            if descriptor[0] == "vertices":
                vertices = descriptor[1]
                world = vertices @ rotation.T + translation
                low_p = world[np.argmin(world, axis=0)]
                high_p = world[np.argmax(world, axis=0)]
            else:
                low_p = np.stack([translation + rotation @ _local_floor_support(
                    descriptor, rotation.T @ axis) for axis in axes])
                high_p = np.stack([translation + rotation @ _local_floor_support(
                    descriptor, -rotation.T @ axis) for axis in axes])
            low_points.append(low_p)
            high_points.append(high_p)
            low.append(np.diag(low_p))
            high.append(np.diag(high_p))
        margins, records = [], []
        minimum = float(self.collision_barrier.d_min)
        for index, pair in enumerate(self.collision_model.collisionPairs):
            first, second = int(pair.first), int(pair.second)
            plus = low[first] - high[second]
            minus = low[second] - high[first]
            gap = np.maximum(np.maximum(plus, minus), 0)
            distance = float(np.linalg.norm(gap))
            # Keep a narrowphase band outside the mandatory clearance. Using
            # the coarser bound right at d_min would introduce an artificial
            # near-zero barrier where actual meshes are still well separated.
            if distance > minimum + 0.02:
                margins.append(distance - minimum)
                records.append(("bound", first, second, gap / distance,
                                plus > minus, low_points, high_points))
            else:
                native = self.pin.computeDistance(self.collision_model, self.collision_data, index)
                value = float(native.min_distance)
                margins.append(value - minimum)
                records.append(("native", first, second,
                                np.asarray(native.getNearestPoint1()).copy(),
                                np.asarray(native.getNearestPoint2()).copy()))
        result = np.asarray(margins), records
        self._live_cache[cache_key] = result
        if len(self._live_cache) > 8:
            self._live_cache.popitem(last=False)
        return result

    def collision_margins(self, q):
        return self._live_collision_query(q)[0].copy()

    def collision_linearization(self, q):
        margins, records = self._live_collision_query(q)
        self.pin.computeJointJacobians(self.model, self.data, self._configuration(q))
        joint_rows = {}
        for geometry in self.collision_model.geometryObjects:
            parent = int(geometry.parentJoint)
            if parent not in joint_rows:
                jacobian = (np.zeros((6, self.model.nv)) if parent == 0 else
                            np.asarray(self.pin.getJointJacobian(
                                self.model, self.data, parent,
                                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)))
                joint_rows[parent] = jacobian[:, self._active_v_indices]

        def point_row(geometry_index, point, direction):
            parent = int(self.collision_model.geometryObjects[geometry_index].parentJoint)
            offset = point - self.data.oMi[parent].translation
            jacobian = joint_rows[parent]
            return direction @ jacobian[:3] + np.cross(offset, direction) @ jacobian[3:]

        rows = []
        axes = np.eye(3)
        for record in records:
            kind, first, second = record[:3]
            row = np.zeros(6)
            if kind == "native":
                point1, point2 = record[3:]
                delta = point1 - point2
                length = float(np.linalg.norm(delta))
                if length > 1e-12:
                    direction = delta / length
                    row = (point_row(first, point1, direction)
                           - point_row(second, point2, direction))
            else:
                weights, first_above, low_points, high_points = record[3:]
                for axis_index, weight in enumerate(weights):
                    if weight == 0:
                        continue
                    axis = axes[axis_index]
                    if first_above[axis_index]:
                        difference = (point_row(first, low_points[first][axis_index], axis)
                                      - point_row(second, high_points[second][axis_index], axis))
                    else:
                        difference = (point_row(second, low_points[second][axis_index], axis)
                                      - point_row(first, high_points[first][axis_index], axis))
                    row += weight * difference
            rows.append(row)
        return margins.copy(), np.asarray(rows)


def _load_bare_kinematics(options: LiveControlOptions) -> PinocchioKinematics:
    config = normalize_robot("ur10e")
    if options.urdf_path:
        import pinocchio as pin

        urdf = Path(options.urdf_path).resolve(strict=True)
        # Expanded robot_descriptions URDFs contain absolute mesh filenames.
        # These roots also admit a supplied URDF with package-relative meshes.
        roots = [str(urdf.parent), str(urdf.parent.parent),
                 str(Path("models/robot_descriptions").resolve())]
        robot = pin.RobotWrapper.BuildFromURDF(str(urdf), roots)
    else:
        robot = _load_robot_description(config.description)
    if int(robot.model.nv) != 6:
        raise ValueError("online UR10e requires a fixed-base bare six-joint arm")
    # Preserve the existing model's pair scope, classified at a fixed registered
    # reference, never at a possibly colliding live initial pose. All retained
    # pairs are evaluated, rather than just a changing nearest-pair subset.
    return _LivePinocchioKinematics(
        robot, "tool0", collision_pairs=10_000,
        collision_min_distance_m=options.collision_distance_m,
        actuated_joint_names=config.joint_names,
        actuated_position_limits=config.position_limits,
        reference_q=config.home_q,
        position_points=(("tool0", (0.0, 0.0, 0.0)),),
        floor_normal=options.floor_normal,
        floor_offset_m=options.floor_offset_m,
    )


class OnlineRetargeter:
    """Persistent six-variable QP with exact checks before admitting a sample.

    ``q_next = command_q + dt * velocity`` preserves the physical joint-angle
    winding. FK/Jacobians use measured_q; an explicit command-to-measurement
    offset participates in the objective and model barriers. Constraints bound
    every command velocity and its change from the last accepted velocity.

    Model margins are metres for ``floor``/``collision`` and radians for
    ``joint``. Floor covers movable collision bodies and tool0. Self-collision
    uses all retained pairs: conservative enclosing-box distance bounds for
    cleared pairs and signed FCL distances for near pairs. Real surrounding
    obstacles are absent.
    Three configurations on each short segment are tested. This is a sampled
    nonlinear check, not a continuous swept-volume certificate.
    """

    def __init__(self, options: LiveControlOptions, kinematics=None, *,
                 tuning: LiveIKTuning | None = None) -> None:
        options.validate()
        if tuning is None:
            from viki.config import load

            cfg = load()
            tuning = LiveIKTuning(position_gain=cfg.LIVE_IK_POSITION_GAIN,
                orientation_gain=cfg.LIVE_IK_ORIENTATION_GAIN, damping=cfg.LIVE_IK_DAMPING,
                acceleration_weight=cfg.LIVE_IK_ACCELERATION_WEIGHT)
            tuning = LiveIKTuning(**{**asdict(tuning),
                "feedforward_gain": cfg.LIVE_IK_FEEDFORWARD_GAIN,
                "feedforward_cutoff_hz": cfg.LIVE_IK_FEEDFORWARD_CUTOFF_HZ,
                "feedforward_max_speed": cfg.LIVE_IK_FEEDFORWARD_MAX_SPEED,
                "feedforward_hold_s": cfg.LIVE_IK_FEEDFORWARD_HOLD_S})
        tuning.validate()
        self.tuning = tuning
        self.options = options
        self.kinematics = (kinematics if kinematics is not None
                           else _load_bare_kinematics(options))
        model = self.kinematics
        if int(model.nq) != 6:
            raise ValueError("online IK requires six physical arm angles")
        self.q_min = _vector(model.q_min, 6, "joint lower limits").copy()
        self.q_max = _vector(model.q_max, 6, "joint upper limits").copy()
        if np.any(self.q_min >= self.q_max):
            raise ValueError("invalid joint limits")
        registered = np.asarray(normalize_robot("ur10e").position_limits)
        self.q_min = np.maximum(self.q_min, registered[:, 0])
        self.q_max = np.minimum(self.q_max, registered[:, 1])
        if np.any(self.q_min >= self.q_max):
            raise ValueError("UR10e joint limits do not intersect model limits")
        model_velocity = np.asarray(model.velocity_limit, dtype=np.float64)
        if (model_velocity.shape != (6,) or np.isnan(model_velocity).any()
                or np.any(model_velocity <= 0)):
            raise ValueError("invalid model velocity limits")
        self.v_max = np.minimum(options.max_velocity, model_velocity)
        self.a_max = float(options.max_acceleration)
        # Import/initialise the native QP binding before any execution loop.
        try:
            import quadprog
        except ImportError as exc:
            raise RuntimeError("online retarget requires quadprog") from exc
        self._quadprog = quadprog
        self._scene_data = (model.model.createData()
                            if isinstance(model, PinocchioKinematics) else None)
        source = None
        if kinematics is None:
            if options.urdf_path:
                source = Path(options.urdf_path).resolve()
            else:
                description = importlib.import_module("robot_descriptions.ur10e_description")
                loader = importlib.import_module("robot_descriptions.loaders.pinocchio")
                source = Path(loader.get_urdf_path(description))
        self.manifest = {
            "objective_tuning": asdict(tuning),
            "robot": "ur10e", "task_frame": "tool0", "bare_arm": True,
            "joint_names": list(normalize_robot("ur10e").joint_names),
            "urdf_path": str(source) if source else None,
            "urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
            "self_collision_pairs": (len(model.collision_model.collisionPairs)
                                     if isinstance(model, PinocchioKinematics) else None),
            "collision_query": "conservative world AABB lower bounds; signed FCL for near pairs",
            "segment_samples": [0., 0.5, 1.],
            "scope": "movable URDF collision bodies, retained self-pairs and configured plane",
        }
        self.model_validate(np.asarray(options.initial_q))

    def _dt(self, dt: float) -> float:
        if not np.isfinite(dt) or not 0 < dt <= 2.0 / self.options.frequency:
            raise OnlineIKError("IK dt must be positive and at most two servo periods")
        return float(dt)

    def pose(self, q) -> tuple[np.ndarray, np.ndarray]:
        """FK of the bare-arm tool0 in the model robot-base frame."""
        p, r = self.kinematics.pose(_vector(q, 6, "q"))
        return _vector(p, 3, "model position").copy(), _rotation(r).copy()

    def scene(self, q) -> dict:
        """Low-rate body-frame skeleton using separate FK data for the viewer.

        This method does not mutate the execution solver's kinematics cache.
        The caller still owns its viewer-thread serialization and rate limit.
        """
        angles = _vector(q, 6, "q")
        model = self.kinematics
        if self._scene_data is None:
            return {"points": [], "edges": [], "frame": "robot_base"}
        pin, data = model.pin, self._scene_data
        pin.forwardKinematics(model.model, data, model._configuration(angles))
        pin.updateFramePlacements(model.model, data)
        points = np.zeros((len(model._scene_frame_ids), 3))
        for index, frame_id in enumerate(model._scene_frame_ids[1:], start=1):
            points[index] = data.oMf[frame_id].translation
        return {"points": points.tolist(), "edges": model.link_edges.tolist(),
                "frame": "robot_base"}

    def model_validate(self, q) -> dict[str, float]:
        """Reject already invalid states; never calculate a recovery motion."""
        angles = _vector(q, 6, "q")
        margins = {"joint": float(np.min(np.minimum(
            angles - self.q_min, self.q_max - angles)))}
        for name in ("floor", "collision"):
            values = np.asarray(getattr(self.kinematics, f"{name}_margins")(angles),
                                dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise OnlineIKError(f"{name} model margins are missing or nonfinite")
            margins[name] = float(np.min(values))
        for name, value in margins.items():
            if value < -1e-9:
                raise OnlineIKError(f"{name} model constraint violated: {value:.8g}")
        return margins

    def _history(self, command_q, previous_velocity, dt):
        period = self._dt(dt)
        q = _vector(command_q, 6, "command_q")
        velocity = _vector(previous_velocity, 6, "previous_velocity")
        if np.any(np.abs(velocity) > self.v_max + 1e-9):
            raise OnlineIKError("previous command velocity exceeds limits")
        margins = self.model_validate(q)
        return q, velocity, period, margins

    def _segment(self, start, end, start_margins=None, end_margins=None) -> dict[str, float]:
        """Exact nonlinear checks at both ends and midpoint of a tiny step."""
        result = start_margins if start_margins is not None else self.model_validate(start)
        if np.max(np.abs(start - end)) <= 1e-12:
            return result.copy()
        for alpha in (0.5, 1.0):
            margins = (end_margins if alpha == 1 and end_margins is not None
                       else self.model_validate(start + alpha * (end - start)))
            result = {key: min(result[key], margins[key]) for key in result}
        return result

    def _bounds(self, q, previous_velocity, dt):
        lower = np.maximum(-self.v_max, previous_velocity - self.a_max * dt)
        upper = np.minimum(self.v_max, previous_velocity + self.a_max * dt)
        # In addition to next-sample joint limits, reserve enough joint travel
        # for a future bounded stop. The continuous stop distance is conservative
        # for the endpoint-velocity integration used by brake().
        distance_low = np.maximum(q - self.q_min, 0.0)
        distance_high = np.maximum(self.q_max - q, 0.0)
        stop_low = np.sqrt((self.a_max * dt) ** 2
                           + 2 * self.a_max * distance_low) - self.a_max * dt
        stop_high = np.sqrt((self.a_max * dt) ** 2
                            + 2 * self.a_max * distance_high) - self.a_max * dt
        lower = np.maximum(lower, -stop_low)
        upper = np.minimum(upper, stop_high)
        lower = np.maximum(lower, (self.q_min - q) / dt)
        upper = np.minimum(upper, (self.q_max - q) / dt)
        if np.any(lower > upper + 1e-10):
            raise OnlineIKError("no velocity satisfies continuity and joint stopping bounds")
        return lower, upper

    def _barriers(self, measured_q, command_q, dt):
        rows, bounds = [], []
        offset = command_q - measured_q
        for name in ("floor", "collision"):
            margin, jacobian = getattr(self.kinematics, f"{name}_linearization")(measured_q)
            margin = np.asarray(margin, dtype=np.float64)
            jacobian = np.asarray(jacobian, dtype=np.float64)
            if (margin.ndim != 1 or not len(margin)
                    or jacobian.shape != (len(margin), 6)
                    or not np.isfinite(margin).all() or not np.isfinite(jacobian).all()):
                raise OnlineIKError(f"invalid {name} model barrier")
            rows.append(dt * jacobian)
            bounds.append(-margin - jacobian @ offset)
        return np.vstack(rows), np.concatenate(bounds)

    def _result(self, q, velocity, previous_velocity, dt, margins, position, rotation,
                singular_value=float("nan")):
        if (np.any(np.abs(velocity) > self.v_max + 1e-8)
                or np.any(np.abs(velocity - previous_velocity) > self.a_max * dt + 1e-8)):
            raise OnlineIKError("solver violated command velocity/acceleration bounds")
        p, r = self.pose(q)
        return {
            "q": q.copy(), "velocity": velocity.copy(), "margins": margins,
            "position_error_m": float(np.linalg.norm(position - p)),
            "orientation_error_rad": float(np.linalg.norm(
                Rotation.from_matrix(rotation @ r.T).as_rotvec())),
            "min_jacobian_singular_value": float(singular_value),
        }

    def step(self, measured_q, command_q, previous_velocity, position, rotation, dt, *,
             target_velocity=None):
        """Compute one bounded command using the current measured configuration."""
        command, previous, period, command_margins = self._history(command_q, previous_velocity, dt)
        measured = _vector(measured_q, 6, "measured_q")
        if np.max(np.abs(measured - command)) > self.options.tracking_tolerance:
            raise OnlineIKError("measured/command joint tracking error exceeds tolerance")
        measured_margins = (command_margins if np.max(np.abs(measured - command)) <= 1e-12
                            else self.model_validate(measured))
        # Check the model-space lag segment too; the actual arm need not have
        # reached the last accepted command when the new sample is calculated.
        lag_margins = self._segment(measured, command, measured_margins, command_margins)
        target_p, target_r = _vector(position, 3, "position"), _rotation(rotation)
        p, r, jp, jr = self.kinematics.pose_jacobian(measured)
        p, r = _vector(p, 3, "model position"), _rotation(r)
        jp, jr = np.asarray(jp, dtype=float), np.asarray(jr, dtype=float)
        if jp.shape != (3, 6) or jr.shape != (3, 6) or not np.isfinite([jp, jr]).all():
            raise OnlineIKError("invalid model pose Jacobian")
        # World angular error matches LOCAL_WORLD_ALIGNED Jacobian axes.
        error_r = Rotation.from_matrix(target_r @ r.T).as_rotvec()
        weighted_j = np.vstack((jp, 0.15 * jr))
        desired = np.concatenate((self.tuning.position_gain * (target_p - p),
                                  0.15 * self.tuning.orientation_gain * error_r))
        feedforward = np.zeros(3)
        if target_velocity is not None:
            feedforward = _vector(target_velocity, 3, "target_velocity")
            if np.linalg.norm(feedforward) > self.tuning.feedforward_max_speed+1e-9:
                raise ValueError("target_velocity exceeds configured Cartesian cap")
            desired[:3] += self.tuning.feedforward_gain*feedforward
        desired -= weighted_j @ (command - measured) / period
        # Damping limits singularity amplification; acceleration regularisation
        # refers to local accepted history, never to the registered home pose.
        damping, acceleration_weight = self.tuning.damping, self.tuning.acceleration_weight
        hessian = weighted_j.T @ weighted_j + (damping + acceleration_weight) * np.eye(6)
        linear = weighted_j.T @ desired + acceleration_weight * previous
        lower, upper = self._bounds(command, previous, period)
        model_rows, model_bounds = self._barriers(measured, command, period)
        rows = np.vstack((np.eye(6), -np.eye(6), model_rows))
        bounds = np.concatenate((lower, -upper, model_bounds))
        try:
            velocity = self._quadprog.solve_qp(hessian, linear, rows.T, bounds)[0]
        except (ValueError, RuntimeError) as exc:
            raise OnlineIKError(f"online IK QP is infeasible: {exc}") from exc
        velocity = _vector(velocity, 6, "solver velocity")
        q_next = command + period * velocity
        margins = self._segment(command, q_next, command_margins)
        margins = {key: min(margins[key], lag_margins[key]) for key in margins}
        smallest = float(np.linalg.svd(weighted_j, compute_uv=False)[-1])
        result = self._result(q_next, velocity, previous, period, margins,
                              target_p, target_r, smallest)
        result["feedforward_velocity"] = self.tuning.feedforward_gain*feedforward
        return result

    def brake(self, command_q, previous_velocity, dt):
        """A controlled bounded stop, with no hand-target extrapolation."""
        command, previous, period, command_margins = self._history(command_q, previous_velocity, dt)
        velocity = np.sign(previous) * np.maximum(np.abs(previous) - self.a_max * period, 0)
        q_next = command + period * velocity
        margins = self._segment(command, q_next, command_margins)
        p, r = self.pose(q_next)
        return self._result(q_next, velocity, previous, period, margins, p, r)
