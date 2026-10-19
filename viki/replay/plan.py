"""Read plans and check the complete UR10e command sequence before connecting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from viki.contracts import Episode
from viki.retarget import load_archive, normalize_robot


@dataclass(frozen=True)
class URReplayOptions:
    """Per-execution settings; angles are radians, time is seconds."""

    speed_scale: float = 0.1
    frequency: int = 125
    max_velocity: float = 0.25
    max_acceleration: float = 0.5
    start_tolerance: float = 0.01
    tracking_tolerance: float = 0.05
    max_lateness: float = 0.04
    lookahead_time: float = 0.1
    gain: int = 300
    arm_only: bool = False
    stop_after: float | None = None

    def validate(self) -> None:
        for name in (
            "speed_scale", "max_velocity", "max_acceleration", "start_tolerance",
            "tracking_tolerance", "max_lateness", "lookahead_time",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.speed_scale > 1:
            raise ValueError("speed_scale must be at most 1")
        if self.stop_after is not None and (not np.isfinite(self.stop_after) or self.stop_after <= 0):
            raise ValueError("stop_after must be finite and positive")
        if self.frequency not in (125, 250, 500):
            raise ValueError("frequency must be 125, 250 or 500 Hz")
        if not 0.03 <= self.lookahead_time <= 0.2 or not 100 <= self.gain <= 2000:
            raise ValueError("invalid servoJ lookahead_time or gain")


@dataclass
class ReplayPlan:
    q: np.ndarray
    gripper: np.ndarray
    dt: float
    robot: str
    approach_frames: int
    demo_frames: int
    metadata: dict
    sha256: str


def validate_commands(q: np.ndarray, gripper: np.ndarray, dt: float) -> None:
    if q.ndim != 2 or len(q) == 0 or q.shape[1] == 0 or not np.isfinite(q).all():
        raise ValueError("q must be a nonempty finite (T, nq) trajectory")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if gripper.shape != (len(q),) or not np.isfinite(gripper).all():
        raise ValueError("gripper opening must be a finite (T,) trajectory")
    if np.any(gripper < 0) or np.any(gripper > 1):
        raise ValueError("gripper opening must stay in [0, 1]")


def read_plan(ep: Episode) -> ReplayPlan:
    if not ep.plan_h5.exists():
        raise FileNotFoundError(f"no plan.h5 for episode {ep.id}; run retarget first")
    # Hash the same immutable snapshot that is read below. Detect a concurrent
    # retarget rather than attributing execution to a different plan.
    digest = hashlib.sha256(ep.plan_h5.read_bytes()).hexdigest()
    with load_archive(ep.plan_h5) as arc:
        q = np.asarray(arc["q"], dtype=np.float64)
        if q.ndim != 2 or not len(q):
            raise ValueError("q must be a nonempty (T, nq) trajectory")
        approach = np.asarray(
            arc["q_approach"] if "q_approach" in arc else np.empty((0, q.shape[1])),
            dtype=np.float64,
        )
        if approach.ndim != 2 or approach.shape[1:] != q.shape[1:]:
            raise ValueError("q_approach must have the same joint dimension as q")
        command = np.asarray(arc["gripper_command"], dtype=np.float64)
        if command.shape != (len(q), 1):
            raise ValueError("replay currently accepts one-dimensional gripper commands")
        model = str(arc["gripper_model"])
        names = json.loads(str(arc["gripper_command_names_json"])) if "gripper_command_names_json" in arc else []
        if names and names not in (["opening"], ["closed"]):
            raise ValueError("replay requires opening or legacy closed command semantics")
        legacy = names == ["closed"] or (not names and model == "binary")
        gripper = 1 - command[:, 0] if legacy else command[:, 0]
        metadata = {
            key: arc[key] for key in (
                "joint_names_json", "solver_status", "metrics_json", "config_json",
                "robot_key", "schema_version",
            ) if key in arc
        }
        dt, robot = float(arc["dt"]), str(arc["robot"])
    if hashlib.sha256(ep.plan_h5.read_bytes()).hexdigest() != digest:
        raise RuntimeError("plan.h5 changed during replay preparation")
    full_q = np.concatenate((approach, q))
    full_g = np.concatenate((np.full(len(approach), gripper[0]), gripper))
    validate_commands(full_q, full_g, dt)
    return ReplayPlan(full_q, full_g, dt, robot, len(approach), len(q), metadata, digest)


def servo_commands(q: np.ndarray, dt: float, options: URReplayOptions):
    """Linear interpolation in joint space; preserve absolute joint winding."""
    options.validate()
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("UR10e commands require six joints")
    source_time = np.arange(len(q)) * dt / options.speed_scale
    period = 1 / options.frequency
    if source_time[-1] > 3600:
        raise ValueError("replay exceeds one hour; select a shorter episode or increase speed_scale")
    times = np.arange(int(np.ceil(source_time[-1] / period)) + 1) * period
    commands = np.column_stack([np.interp(times, source_time, col) for col in q.T])
    # Hold the final setpoint for the lookahead interval before ending servoing.
    hold = int(np.ceil(options.lookahead_time / period))
    commands = np.concatenate((commands, np.repeat(commands[-1:], hold, axis=0)))
    times = np.arange(len(commands)) * period
    limits = np.asarray(normalize_robot("ur10e").position_limits)
    if np.any(commands < limits[:, 0]) or np.any(commands > limits[:, 1]):
        raise ValueError("UR10e commands violate physical joint limits")
    # Include starting and stopping from rest in the discrete command bounds.
    velocity = np.diff(np.concatenate((commands[:1], commands, commands[-1:])), axis=0) / period
    acceleration = np.diff(velocity, axis=0) / period
    vmax = float(np.max(np.abs(velocity), initial=0))
    amax = float(np.max(np.abs(acceleration), initial=0))
    if vmax > options.max_velocity + 1e-9 or amax > options.max_acceleration + 1e-9:
        raise ValueError(
            f"command limits exceeded: velocity={vmax:.4g} rad/s, acceleration={amax:.4g} rad/s²; "
            "reduce --speed-scale or regenerate the plan"
        )
    return times, commands, vmax, amax


def check_ur_plan(plan: ReplayPlan, options: URReplayOptions) -> dict:
    options.validate()
    robot = normalize_robot(plan.robot)
    if robot.description != normalize_robot("ur10e").description:
        raise ValueError(f"UR10e replay requires a UR10e plan, got {plan.robot!r}; retarget with --robot ur10e")
    if json.loads(str(plan.metadata.get("joint_names_json", "[]"))) != list(robot.joint_names):
        raise ValueError("plan joint order must exactly match UR10e RTDE joint order")
    if plan.metadata.get("solver_status") != "converged":
        raise ValueError("hardware replay requires a converged retarget plan")
    if "robot_key" in plan.metadata and str(plan.metadata["robot_key"]) != "ur10e":
        raise ValueError("robot_key contradicts the UR10e plan description")
    metrics = json.loads(str(plan.metadata.get("metrics_json", "{}")))
    for key in ("min_floor_margin_mm", "min_collision_margin", "min_joint_limit_margin_rad", "min_velocity_limit_margin_rad_s"):
        value = metrics.get(key)
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError(f"hardware replay requires a nonnegative measured plan margin: {key}")
    _, commands, vmax, amax = servo_commands(plan.q, plan.dt, options)
    return {
        "robot": "ur10e", "plan_sha256": plan.sha256,
        "demo_frames": plan.demo_frames, "approach_frames": plan.approach_frames,
        "start_q_rad": plan.q[0].tolist(), "end_q_rad": plan.q[-1].tolist(),
        "control_samples": len(commands), "duration_s": len(commands) / options.frequency,
        "max_command_velocity_rad_s": vmax, "max_command_acceleration_rad_s2": amax,
        "plan_margins": {k: metrics[k] for k in metrics if k.startswith("min_")},
        "options": asdict(options),
        "gripper": "uncontrolled" if options.arm_only else "driver required",
        "collision_scope": "archived planner sample checks; no environment or continuous-path validation",
    }


def check_replay(ep: Episode, *, options: URReplayOptions | None = None) -> dict:
    """Offline check, with no RTDE connection, movement, or artifact writes."""
    return check_ur_plan(read_plan(ep), options or URReplayOptions())
