"""plan.h5 -> replay.h5 plus immutable control-rate execution logs (§3.8)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import uuid

from viki.contracts import Episode, REPLAY_KEYS
from viki.episode import clear_stage, mark_stage
from viki.replay.driver import load_driver
from viki.replay.plan import URReplayOptions, check_ur_plan, read_plan
from viki.replay.screen import Verdict, screen
from viki.replay.ur import ReplayAborted
from viki.retarget import write_hdf5_archive

logger = logging.getLogger(__name__)


def replay_episode(
    ep: Episode,
    *,
    driver: str = "dryrun",
    max_resolves: int = 0,
    host: str | None = None,
    options: URReplayOptions | None = None,
    stop_event=None,
) -> str:
    """Execute a prepared plan. Hardware errors retain a rejected partial log."""
    if max_resolves:
        raise NotImplementedError("replay re-solve loop is not implemented (paper §3.8)")
    plan = read_plan(ep)
    hardware = driver == "ur10e"
    settings = options or URReplayOptions()
    if hardware:
        report = check_ur_plan(plan, settings)
        drv = load_driver(driver, host=host, options=settings, stop_event=stop_event)
    else:
        if host is not None or options is not None:
            raise ValueError("host/options apply only to driver='ur10e'")
        report = {}
        drv = load_driver(driver)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    log_path = ep.root / "intermediates" / "replay" / run_id / "execution.h5"
    clear_stage(ep, "replay")
    clear_stage(ep, "export")
    error = None
    try:
        execution_log = drv.execute(plan.q, plan.gripper, plan.dt)
    except ReplayAborted as exc:
        execution_log, error = exc.log, exc
    finally:
        drv.close()

    completed = error is None
    if completed and len(execution_log.q_attained) != len(plan.q):
        raise RuntimeError("replay driver returned a trajectory with the wrong length")
    effective_dt = plan.dt / settings.speed_scale if hardware else plan.dt
    v = (
        Verdict("test-stop" if error.intentional else "reject", str(error)) if error else
        screen(execution_log.q_attained, execution_log.controller_residual, plan.robot,
               residual_threshold=settings.tracking_tolerance if hardware else 0.05)
    )
    if hardware and settings.arm_only and v.verdict == "pass":
        v = Verdict("arm-only", "")

    # Full motion including the approach, never conflated with source video.
    values = {
        "schema_version": 1, "driver": driver, "hardware": hardware,
        "completed": completed, "verdict": v.verdict, "rejection_cause": v.cause,
        "plan_sha256": plan.sha256, "source_dt": plan.dt, "dt": effective_dt,
        "robot": plan.robot, "approach_frames": plan.approach_frames,
        "q_source": plan.q, "gripper_command_source": plan.gripper,
        "q_attained_source_frames": execution_log.q_attained,
        "gripper_attained_source_frames": execution_log.gripper_attained,
        "controller_residual_source_frames": execution_log.controller_residual,
        "options_json": json.dumps(asdict(settings) if hardware else {}),
        "identity_json": json.dumps(getattr(drv, "identity", {})),
        "preflight_json": json.dumps(report),
        **execution_log.telemetry,
    }
    write_hdf5_archive(log_path, values, schema="viki_execution_hdf5_v1")
    approach = plan.approach_frames
    archive = {
        "q_attained": execution_log.q_attained[approach:],
        "gripper_attained": execution_log.gripper_attained[approach:],
        "controller_residual": execution_log.controller_residual[approach:],
        "verdict": v.verdict, "rejection_cause": v.cause, "resolve_attempts": 0,
        "robot": plan.robot, "dt": effective_dt, "source_dt": plan.dt,
        "driver": driver, "hardware": hardware, "completed": completed,
        "gripper_measured": False,  # dryrun echoes; arm-only has no sensor
        "plan_sha256": plan.sha256,
        "execution_log": str(log_path.relative_to(ep.root)),
    }
    assert set(archive) == set(REPLAY_KEYS)
    write_hdf5_archive(ep.replay_h5, archive)
    mark_stage(
        ep, "replay", done=completed, verdict=v.verdict, rejection_cause=v.cause,
        driver=driver, hardware=hardware, completed=completed, gripper_measured=False,
        execution_log=archive["execution_log"], plan_sha256=plan.sha256,
    )
    logger.info("replay %s: verdict=%s log=%s", ep.id, v.verdict, log_path)
    if error and not error.intentional:
        raise RuntimeError(f"UR replay aborted: {error}; partial log: {log_path}") from error
    return str(ep.replay_h5)
