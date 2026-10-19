"""Measured joint/tracking checks (§3.8); no collision or singularity verdict."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from viki.retarget import robot_joint_limits

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    verdict: str  # pass | reject | dry-run | arm-only | test-stop
    cause: str


def screen(q_attained, controller_residual, robot_description: str, *, residual_threshold: float = 0.05) -> Verdict:
    q = np.asarray(q_attained, dtype=np.float64)
    residual = np.asarray(controller_residual, dtype=np.float64)
    if q.ndim != 2 or not len(q) or not np.isfinite(q).all() or residual.shape != (len(q),):
        return Verdict("reject", "invalid_telemetry")
    limits_available = False
    try:
        lower, upper = robot_joint_limits(robot_description)
        limits_available = q.shape[1] == len(lower)
        if limits_available and (np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6)):
            return Verdict("reject", "joint_limit")
    except Exception as exc:
        logger.warning("joint limits unavailable (%s): %s", robot_description, exc)
    if np.isnan(residual).all():
        return Verdict("dry-run", "")
    if not np.isfinite(residual).all() or np.any(residual < 0):
        return Verdict("reject", "invalid_telemetry")
    if np.max(residual) > residual_threshold:
        return Verdict("reject", "tracking_fault")
    if not limits_available:
        return Verdict("reject", "joint_limits_unavailable")
    return Verdict("pass", "")
