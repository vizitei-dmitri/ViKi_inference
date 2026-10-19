"""viki.server.routes.replay — plan checks and execution (paper §3.8)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from viki.contracts import Episode
from viki.replay import ReplayManager, URReplayOptions, check_replay, inspect_ur10e
from viki.server import jobs
from viki.server.deps import get_replay_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/replay", tags=["replay"])


class ReplayRequest(BaseModel):
    episode: str
    driver: Literal["dryrun", "ur3", "ur10e"] = "dryrun"
    max_resolves: int = 0
    host: str | None = None
    options: URReplayOptions | None = None


@router.post("/check")
async def check_plan(req: ReplayRequest):
    try:
        return await run_in_threadpool(check_replay, Episode(root=Path(req.episode)), options=req.options)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/inspect")
async def inspect_robot(req: ReplayRequest):
    try:
        report = await run_in_threadpool(check_replay, Episode(root=Path(req.episode)), options=req.options)
        report["live"] = await run_in_threadpool(
            inspect_ur10e, req.host, start_q=report["start_q_rad"],
            tolerance=(req.options or URReplayOptions()).start_tolerance,
        )
        return report
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("")
async def start_replay(req: ReplayRequest, manager: ReplayManager = Depends(get_replay_manager)):
    ep = Episode(root=Path(req.episode))
    if not ep.plan_h5.exists():
        raise HTTPException(404, f"no plan.h5 for episode {req.episode}; run retarget first")
    if req.max_resolves:
        raise HTTPException(400, "replay re-solve loop is not implemented")
    if req.driver == "ur3":
        raise HTTPException(400, "UR3 replay remains a stub")
    if req.driver == "ur10e":
        if not req.host or not req.options or not req.options.arm_only:
            raise HTTPException(400, "UR10e replay requires host and options.arm_only=true")
        try:
            await run_in_threadpool(check_replay, ep, options=req.options)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    elif req.host is not None or req.options is not None:
        raise HTTPException(400, "host/options apply only to UR10e execution")

    logger.info("replay: episode=%s driver=%s max_resolves=%d", ep.id, req.driver, req.max_resolves)

    token = manager.prepare()

    def _job():
        return manager.run(token, ep, driver=req.driver, max_resolves=req.max_resolves, host=req.host, options=req.options)

    job_id = jobs.submit("replay", _job, episode=ep.id)
    manager.bind_job(job_id, token)
    return {"job_id": job_id}


@router.post("/jobs/{job_id}/stop")
async def stop_replay(job_id: str, manager: ReplayManager = Depends(get_replay_manager)):
    job = jobs.get(job_id)
    if job is None or job["kind"] != "replay":
        raise HTTPException(404, "no replay job with this id")
    requested = manager.stop_job(job_id)
    if jobs.cancel(job_id):
        manager.forget_job(job_id)
        return {"status": "cancelled"}
    if not requested:
        raise HTTPException(409, "replay is already finished")
    return {"status": "stop-requested"}


@router.get("/jobs/{job_id}")
async def replay_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j
