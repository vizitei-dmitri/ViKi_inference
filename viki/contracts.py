"""
viki.contracts
--------------
The single shared module between pipeline stages: every DTO, enum, Protocol,
and file-artifact schema constant lives here.

Rule of the refactor: a stage imports from ``viki.contracts`` and from the
public ``__init__`` of a neighbouring stage — never from another stage's
internal modules. This module depends only on numpy/cv2; nothing from
``viki.*`` may be imported here.

Artifact chain (see ``Episode``)::

    episodes/<id>/raw/      raw synced RGB-D  (cameras.record)
        -> rec.npz          per-camera hand landmark trajectories  (perception.extract)
        -> cln.npz          fused + smoothed wrist trajectory + EE pose  (prepare.run)
        -> plan.h5          synthesised robot joint trajectory  (retarget.run)
        -> replay.h5        proprioception attained on hardware  (replay.run)
    datasets/<name>/        LeRobot dataset  (export.run)  [stub]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

# ─────────────────────────────── aliases ────────────────────────────────

Hand = Literal["left", "right"]
Outcome = Literal["good", "bad", "unrated"]
Verdict = Literal["pass", "reject", "dry-run", "unverified"]


# ──────────────────────────── hand landmarks ────────────────────────────


class LM(IntEnum):
    """
    MediaPipe Hands landmark indices, 0..20.

    ViKi tracks a single hand only; the arm landmarks that older revisions
    reserved (ELBOW/SHOULDER) are gone. Use ``HAND_LM_COUNT`` for the count —
    it is deliberately *not* an enum member so ``list(LM)`` stays clean.
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_LM_COUNT = 21


# ──────────────────────────── camera / capture ──────────────────────────


@dataclass
class CameraIntrinsics:
    """SDK-reported pinhole intrinsics for one colour or depth stream."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Distortion coefficients (k1,k2,p1,p2[,k3]) — optional.
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def matrix(self) -> np.ndarray:
        """3x3 pinhole camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class Frame:
    """
    A single frame from one camera.

    Fields
    ------
    color             : HxWx3 uint8 BGR (OpenCV convention)
    depth             : HxW   uint16 millimetres
    timestamp_us      : device monotonic clock, microseconds
    device_id         : serial or ``kinect_<n>``
    aligned_depth     : HxW uint16 SDK-aligned depth (optional)
    host_timestamp_us : host **monotonic** clock (µs, arbitrary epoch) stamped by
                        the worker on arrival; used for cross-camera sync — only
                        ever compared as a difference, never as wall time
    color_intrinsics / depth_intrinsics : CameraIntrinsics | None
    """

    color: np.ndarray
    depth: np.ndarray
    timestamp_us: int
    device_id: str
    aligned_depth: Optional[np.ndarray] = None
    host_timestamp_us: int = 0
    color_intrinsics: Optional[CameraIntrinsics] = None
    depth_intrinsics: Optional[CameraIntrinsics] = None

    def has_depth(self) -> bool:
        return self.depth is not None and self.depth.size > 0


@dataclass
class SyncedFrameGroup:
    """
    One frame per camera, aligned to a common host monotonic-clock tick.

    ``sync_timestamp_us`` is ``time.monotonic_ns() // 1000`` at the tick;
    ``offsets_us[device_id] = frame.host_timestamp_us - sync_timestamp_us``
    (negative = frame arrived before the tick).
    """

    frames: dict
    sync_timestamp_us: int
    offsets_us: dict

    @property
    def device_ids(self) -> list:
        return list(self.frames.keys())

    def has_depth(self) -> bool:
        return all(f.has_depth() for f in self.frames.values())


@runtime_checkable
class DepthProjector(Protocol):
    """
    Minimal slice of a camera backend that ``perception`` needs for lifting:
    map a colour pixel to the depth image plane at range ``z``.
    """

    def project_color_to_depth(
        self, u: float, v: float, z: float
    ) -> tuple[float, float] | None: ...


@dataclass
class LiveHandEstimate:
    """Causal hand estimate in rig axes; source time never advances on a hold."""

    sequence: int
    source_time_us: int
    generated_time_us: int
    points: np.ndarray
    observed: np.ndarray
    quality: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    position_valid: bool
    orientation_valid: bool
    calibration_id: str
    reason: str = ""
    opening: float = float("nan")
    camera_timestamps: dict = field(default_factory=dict)
    raw_points: np.ndarray | None = None  # opt-in current-frame geometry for filter experiments
    camera_observations: dict | None = None  # opt-in rectified 2-D evidence; local captures only
    triangulation_details: dict | None = None


@dataclass
class HumanTarget:
    """Latest human pose, not a robot command; all host times are monotonic µs."""

    sequence: int
    source_time_us: int
    valid_until_us: int
    position: np.ndarray
    rotation: np.ndarray
    position_valid: bool
    orientation_valid: bool
    calibration_id: str


@dataclass(frozen=True)
class LiveHandFitOptions:
    """Experimental causal landmark-only articulated fit, not batch/cloud fit."""

    warmup_frames: int = 10
    quality_power: float = 0.0
    max_nfev: int = 20
    joint_prior: float = 1e-5
    robust_scale_m: float = 0.008
    correction_cap_m: float = 0.03

    def validate(self):
        if self.warmup_frames < 2 or self.max_nfev < 1:
            raise ValueError("fit warmup>=2 and max_nfev>=1 required")
        if not np.isfinite(self.quality_power) or not 0 <= self.quality_power <= 2:
            raise ValueError("quality_power must be in [0, 2]")
        for name in ("joint_prior", "robust_scale_m", "correction_cap_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name != "joint_prior" and value == 0):
                raise ValueError(f"invalid {name}")


@dataclass
class LiveHandFitEstimate:
    source_time_us: int
    points: np.ndarray
    observed: np.ndarray
    inferred: np.ndarray
    accepted: bool
    reason: str
    q: np.ndarray
    compute_ms: float
    nfev: int
    joint_margin_rad: float
    max_correction_m: float
    converged: bool = False


@dataclass(frozen=True)
class LiveIKTuning:
    """Objective tuning; joint, acceleration and collision bounds stay separate."""

    position_gain: float = 6.0  # s^-1
    orientation_gain: float = 4.0  # s^-1
    damping: float = 1e-4
    acceleration_weight: float = 1e-3
    feedforward_gain: float = 0.0
    feedforward_cutoff_hz: float = 5.0
    feedforward_max_speed: float = 0.1  # robot-base m/s
    feedforward_hold_s: float = 0.12  # availability age; source TTL remains independent

    def validate(self) -> None:
        for name in ("position_gain", "orientation_gain", "damping", "acceleration_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name != "acceleration_weight" and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'acceleration_weight' else 'positive'}")
        if not np.isfinite(self.feedforward_gain) or not 0 <= self.feedforward_gain <= 1:
            raise ValueError("feedforward_gain must be in [0, 1]")
        for name in ("feedforward_cutoff_hz", "feedforward_max_speed", "feedforward_hold_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.feedforward_max_speed > .15 or self.feedforward_hold_s > .25:
            raise ValueError("feedforward max speed <=0.15 m/s and hold <=0.25 s required")


@dataclass(frozen=True)
class LiveControlOptions:
    """Per-session arm-only settings, independent of frozen offline profiles."""

    frequency: int = 50
    max_velocity: float = 0.25
    max_acceleration: float = 0.5
    tracking_tolerance: float = 0.05
    max_lateness: float = 0.04
    input_ttl_s: float = 0.25
    scale: float = 0.25
    max_displacement_m: float = 0.15
    orientation: bool = False
    rotation_robot_rig: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    floor_normal: tuple = (0., 0., 1.)
    floor_offset_m: float = 0.
    collision_distance_m: float = 0.02
    initial_q: tuple = (0., -1.2, 1.6, -1.9, -1.2, 0.)
    host: str | None = None
    hardware_enabled: bool = False
    mapping_confirmed: bool = False
    model_path: str = "models/mujoco/ur10e/scene.xml"
    urdf_path: str | None = None

    def validate(self) -> None:
        if self.frequency not in (50, 125):
            raise ValueError("live executor supports 50 Hz or experimental 125 Hz")
        for key in ("max_velocity", "max_acceleration", "tracking_tolerance",
                    "max_lateness", "input_ttl_s", "scale", "max_displacement_m"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.max_velocity > 0.25 or self.max_acceleration > 0.5 or self.scale > 1:
            raise ValueError("live bring-up bounds: velocity<=0.25, acceleration<=0.5, scale<=1")
        rotation = np.asarray(self.rotation_robot_rig, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            raise ValueError("rotation_robot_rig must be a proper SO(3) rotation")
        normal = np.asarray(self.floor_normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            raise ValueError("floor_normal must be finite and nonzero")
        if not np.isfinite(self.floor_offset_m) or not np.isfinite(self.collision_distance_m) or self.collision_distance_m <= 0:
            raise ValueError("invalid model floor/collision settings")
        initial = np.asarray(self.initial_q, dtype=float)
        if initial.shape != (6,) or not np.isfinite(initial).all():
            raise ValueError("initial_q must contain six finite joint angles")


# ──────────────────────────── calibration ───────────────────────────────


@dataclass
class BoardParameters:
    """Physical parameters of a plain chessboard."""

    board_size: Tuple[int, int]  # internal corners (cols, rows)
    square_size: float  # metres


@dataclass
class ArucoBoardParameters(BoardParameters):
    """Physical parameters of a ChArUco board."""

    marker_size: float
    aruco_dict: int  # cv2.aruco predefined dictionary id


@dataclass
class CalibrationSample:
    """One accepted board observation for one camera."""

    frame: "Frame"
    corners: np.ndarray
    resolution: Tuple[int, int]  # (width, height)
    board_params: BoardParameters


@dataclass
class ArucoCalibrationSample(CalibrationSample):
    """ChArUco sample: also carries detected chessboard-corner ids."""

    c_ids: np.ndarray


@dataclass
class CalibrationIntrinsics:
    """Solved intrinsics for one camera (calibration result / stored form)."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass
class CalibrationExtrinsics:
    """
    Pose of a ChArUco board relative to one camera (Rodrigues ``rvec`` +
    ``tvec``). ``transform_matrix`` is the 4x4 camera→world (board) transform
    consumed by ``perception.lift``.
    """

    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def rotation_matrix(self) -> np.ndarray:
        import cv2

        R, _ = cv2.Rodrigues(self.rvec)
        return np.array(R)

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 homogeneous transform from the camera frame to the board frame."""
        R = self.rotation_matrix
        R_inv = R.T
        t = self.tvec.flatten()
        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = -R_inv @ t
        return np.array(T)


# ──────────────────────────── perception DTOs ───────────────────────────


@dataclass
class PreparedFrame:
    """
    A camera frame ready for a pose backend.

    rgb          : (H, W, 3) uint8, RGB
    depth_m      : (H, W) float32, metres (0 → NaN)
    depth_K      : (3, 3) depth intrinsic matrix, or None
    base_depth_m : optional static-background depth (metres) for scene subtraction
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    depth_K: Optional[np.ndarray]
    device_id: str
    timestamp_us: int
    base_depth_m: Optional[np.ndarray] = None


@dataclass
class HandDetection:
    """
    Pixel-space output of a pose backend for one camera, one hand.

    points   : {LM: (u, v)} subpixel pixel coordinates (NaN where missing)
    lm_z_rel : (21,) float32 backend-relative z (NOT metric); wrist-relative
    confidence : overall detection score in [0, 1]
    lm_score : (21,) float32 per-landmark detector score in [0, 1], or None when
               the backend only reports one score for the whole hand (callers
               then fall back to broadcasting ``confidence``).
    """

    points: dict
    lm_z_rel: np.ndarray
    confidence: float
    device_id: str
    timestamp_us: int
    lm_score: np.ndarray | None = None


@dataclass
class Landmarks3D:
    """Per-landmark 3-D positions in one camera's frame (metres)."""

    points: dict  # {LM: (3,) float32}
    device_id: str
    timestamp_us: int
    # Per-landmark fusion weight (paper §3.5, eq. 2). Stub: detector visibility
    # only — the range and incidence factors are not computed yet.
    weights: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {int(k): v.tolist() for k, v in self.points.items()},
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class EndEffectorPose:
    """
    World-frame wrist pose.

    position     : (3,) float32 — WRIST world XYZ (m)
    R_world_palm : (3, 3) float32 — palm frame → world. Built from the MCP
                   knuckle spread (INDEX→PINKY), not the thumb.
    rpy_deg      : (3,) float32 — roll/pitch/yaw, extrinsic XYZ
    valid        : all required landmarks present and the palm frame resolved
    """

    position: np.ndarray
    R_world_palm: np.ndarray
    rpy_deg: np.ndarray
    valid: bool
    timestamp_us: int

    def as_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "R_world_palm": self.R_world_palm.tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
            "valid": self.valid,
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class SkeletonFrame:
    """Per-camera 3-D hand skeleton in world coordinates."""

    device_id: str
    points: dict  # {LM: (3,) world XYZ, metres}
    timestamp_us: int
    end_effector: Optional[EndEffectorPose] = None

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {int(k): v.tolist() for k, v in self.points.items()},
            "timestamp_us": self.timestamp_us,
            "end_effector": (
                self.end_effector.as_dict() if self.end_effector is not None else None
            ),
        }


@dataclass
class DepthDebug:
    """Per-camera, per-frame depth diagnostics kept alongside a recording."""

    device_id: str
    depth_valid_fraction: float
    depth_median_m: float
    depth_mean_m: float
    hand_detected: bool
    wrist_depth_m: float


@dataclass
class PipelineResult:
    """Result of one ``perception.extract`` step over a synced frame group."""

    frames: list  # list[SkeletonFrame], one per camera that saw a hand
    detections: dict  # {device_id: HandDetection | None}
    debug_depth_marks: Optional[dict] = None
    depth_debug: Optional[dict] = None


# ──────────────────────────── gripper ───────────────────────────────────


@dataclass(frozen=True)
class GripperState:
    """
    Estimated gripper state for one frame.

    closed     : True = closed / grasping
    width      : 0..1 normalised opening (for continuous grippers; binary sets 0/1)
    confidence : 0..1
    """

    closed: bool
    width: float
    confidence: float


# ──────────────────────────── labelling ─────────────────────────────────


@dataclass
class Segment:
    """A frame range within an episode tagged with a phase label."""

    start: int
    end: int
    label: str  # approach / grasp / transport / release / ...


@dataclass
class EpisodeLabels:
    """
    Human annotation for one episode, persisted under ``meta.json["labels"]``.
    ``task`` must be non-empty before the episode can be exported.
    """

    task: str = ""
    hand: Hand = "right"
    segments: list = field(default_factory=list)  # list[Segment]
    outcome: Outcome = "unrated"
    notes: str = ""


# ──────────────────────────── episode paths ─────────────────────────────


@dataclass(frozen=True)
class Episode:
    """
    Filesystem view of one demonstration. Every stage reads this directory and
    writes exactly one more artifact into it.
    """

    root: Path

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def rec_npz(self) -> Path:
        return self.root / "rec.npz"

    @property
    def cln_npz(self) -> Path:
        return self.root / "cln.npz"

    @property
    def intermediates_dir(self) -> Path:
        """Lossless stage checkpoints used for A/B inspection and diagnostics."""
        return self.root / "intermediates"

    @property
    def plan_h5(self) -> Path:
        return self.root / "plan.h5"

    @property
    def replay_h5(self) -> Path:
        return self.root / "replay.h5"

    @property
    def cloud_dir(self) -> Path:
        """Per-frame coloured point cloud (``<i:06d>.bin`` + ``meta.json``)."""
        return self.root / "cloud"

    @property
    def segmentation_dir(self) -> Path:
        """Versioned scene-segmentation artifact used by the object stage."""
        return self.intermediates_dir / "segmentation" / "sam2.1_hiera_small"

    @property
    def scene_prompts_path(self) -> Path:
        """Auditable automatic/manual prompt manifest for scene segmentation."""
        return self.intermediates_dir / "segmentation" / "auto_prompts.json"

    @property
    def object_models_npz(self) -> Path:
        """Compact rigid-object models and tracks derived from semantic RGB-D."""
        return self.segmentation_dir / "object_models.npz"

    @property
    def object_relative_npz(self) -> Path:
        """Post-IK object-relative hand and TCP trajectories."""
        return self.intermediates_dir / "object_relative.npz"


# ──────────────────────── artifact schema keys ─────────────────────────
# The exact array keys each stage writes. Writers and readers assert against
# these so a schema drift fails loudly instead of silently.

REC_KEYS: tuple[str, ...] = (
    "device_ids",
    "timestamps",
    "points",  # (N, 21, 3)
    "landmark_ids",  # (21,)
    "confidence",  # (N, 21)  — per-landmark fusion weight w (paper §3.5 eq. 2)
)

CLN_KEYS: tuple[str, ...] = (
    "timestamps",  # (T,)
    "positions",  # (T, 3)
    "rotations",  # (T, 3, 3)
    "valid",  # (T,)
    "omega",  # (T,)  — per-frame confidence weight ω_t (paper §3.5 eq. 5)
    "gripper",  # (T,) float32 — normalised opening: 0 closed, 1 fully open
    "coordinate_frame",
    "raw_points",
    "smoothed_points",
    "landmark_ids",
)
CLN_OPTIONAL_KEYS: tuple[str, ...] = (
    "landmark_confidence",  # (T, L) — fused confidence retained for hand-fit anchors
    "observed_points",  # (T, L, 3) — fusion output before any gap filling
    "filled_points",  # (T, L, 3) — after gap fill, before temporal smoothing
    "observed_mask",  # (T, L) — joints directly present at the fusion boundary
    "interpolated_mask",  # (T, L) — joints fabricated by the gap-fill stage
    "perception_fuse_mode",  # scalar string — xyz_mean | triangulate
    "fused_interpolation",  # scalar string — cubic | linear
    "fused_extrapolate_edges",  # scalar bool — fill outside observed support
    "checkpoint_stage",  # scalar string — observed | filled | smoothed | hand_fit
    "checkpoint_params_json",  # scalar JSON — exact knobs used for this artifact
    "T_world_obj",  # (T, 4, 4)  — object pose track  (stub: absent)
    "T_obj_hand",  # (T, 4, 4)  — object-relative form  (stub: absent)
    "hand_fit_positions",  # (T, 3) — fitted pose; landmark positions remain untouched
    "hand_fit_rotations",  # (T, 3, 3)
    "hand_fit_joint_angles",  # (T, nq) — complete articulated trajectory
    "hand_fit_model_nq",  # scalar int — nq of the fitted hand model
    "hand_fit_capsules",  # (T, C, 2, 3) — world capsule endpoints per frame
    "hand_fit_capsule_radii",  # (C,) — capsule radii (m)
    "hand_fit_metrics_json",  # scalar JSON — episode-level diagnostics
)

# Canonical retarget output.  Scene geometry is intentionally part of the plan:
# the browser never needs Pinocchio or a second interpretation of the URDF.
PLAN_KEYS: tuple[str, ...] = (
    "schema_version",
    "coordinate_frame",
    "timestamps",
    "fps",
    "dt",
    "robot",
    "robot_key",
    "ee_frame",
    "joint_names_json",
    "link_edges",
    "link_groups_json",
    "point_groups_json",
    "T_calibration_rig",
    "base_position_calibration",
    "base_rpy_deg_calibration",
    "target_position_anchor",
    "q",
    "q_approach",
    "joint_velocity",
    "joint_acceleration",
    "gripper_model",
    "gripper_description",
    "gripper_tcp_frame",
    "gripper_drive_joint",
    "adapter_kind",
    "adapter_translation_m",
    "adapter_rpy_deg",
    "adapter_radius_m",
    "gripper_command_names_json",
    "gripper_command",
    "gripper_closed",
    "gripper_opening",
    "gripper_joint_position",
    "gripper_opening_m",
    "omega",
    "target_position_calibration",
    "target_rotation_calibration",
    "target_position_robot",
    "achieved_position_calibration",
    "achieved_rotation_calibration",
    "position_error_m",
    "orientation_error_rad",
    "sequential_baseline_q",
    "sequential_baseline_achieved_position_calibration",
    "sequential_baseline_achieved_rotation_calibration",
    "sequential_baseline_position_error_m",
    "sequential_baseline_orientation_error_rad",
    "sequential_baseline_metrics_json",
    "link_positions_calibration",
    "robot_joint_placements_calibration",
    "robot_visuals_json",
    "solver_status",
    "config_json",
    "metrics_json",
    "source_pose",
)


def cln_pose_keys(files, source: str) -> tuple[str, str]:
    """Resolve the configured wrist-pose source without mutating the artifact.

    ``hand_fit`` gracefully falls back for old/unfitted episodes so changing the
    global source does not make historical datasets unreadable.
    """
    available = set(files)
    if str(source).strip().lower() == "hand_fit" and {
        "hand_fit_positions", "hand_fit_rotations"
    }.issubset(available):
        return "hand_fit_positions", "hand_fit_rotations"
    return "positions", "rotations"

REPLAY_KEYS: tuple[str, ...] = (
    "q_attained",  # (T, nq)
    "gripper_attained",  # (T,)
    "controller_residual",  # (T,)  NaN under dry-run
    "verdict",
    "rejection_cause",
    "resolve_attempts",
    "robot",
    "dt",
    "source_dt",  # source plan time step before hardware slowdown
    "driver",
    "hardware",
    "completed",
    "gripper_measured",  # false for dryrun and arm-only execution
    "plan_sha256",
    "execution_log",  # immutable full control-rate log, relative to episode
)
