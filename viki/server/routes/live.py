"""Session transport for causal hand estimation and explicit live following."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from viki.contracts import LiveControlOptions
from viki.live import LiveManager, LiveSessionConfig, RecordedReviewStore, load_mapping
from viki.server.deps import get_live_manager, get_recorded_reviews

router = APIRouter(prefix="/live", tags=["live"])


async def _review_call(action):
    try:
        return await run_in_threadpool(action)
    except (FileNotFoundError, IndexError) as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/reviews")
async def reviews(store: RecordedReviewStore = Depends(get_recorded_reviews)):
    return {"sessions": await _review_call(store.list)}


@router.get("/reviews/{session_id}")
async def review_metadata(session_id: str, store: RecordedReviewStore = Depends(get_recorded_reviews)):
    return await _review_call(lambda: store.get(session_id).metadata())


@router.get("/reviews/{session_id}/frames/{frame}")
async def review_frame(session_id: str, frame: int, store: RecordedReviewStore = Depends(get_recorded_reviews)):
    return await _review_call(lambda: store.get(session_id).frame(frame))


@router.get("/reviews/{session_id}/cloud/{frame}")
async def review_cloud(session_id: str, frame: int, store: RecordedReviewStore = Depends(get_recorded_reviews)):
    path = await _review_call(lambda: store.get(session_id).cloud_path(frame))
    return FileResponse(path, media_type="application/octet-stream")


class LiveStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str = Field(min_length=1)
    source: Literal["recorded", "cameras"] = "recorded"
    backend: Literal["mujoco", "ur10e"] = "mujoco"
    seconds: float = 60.
    perception_hz: float = 15.
    recorded_speed: float = 1.
    landmark_mode: Literal["all21", "arm7"] = "all21"
    auto_bind: bool = False
    mapping_json: str | None = None
    options: LiveControlOptions = Field(default_factory=LiveControlOptions)


def _start(req: LiveStartRequest, manager: LiveManager):
    # A client boolean cannot certify a rig mapping. Only the explicitly
    # supplied operator file may set this field, and its digest is archived.
    options = replace(req.options, mapping_confirmed=False)
    manifest = {}
    if req.mapping_json:
        changes, manifest = load_mapping(req.mapping_json)
        options = replace(options, **changes)
    config = LiveSessionConfig(
        reference=req.reference, source=req.source, backend=req.backend,
        seconds=req.seconds, perception_hz=req.perception_hz,
        recorded_speed=req.recorded_speed, landmark_mode=req.landmark_mode,
        auto_bind=req.auto_bind, options=options, mapping_manifest=manifest,
    )
    config.validate()
    return manager.start(config)


@router.get("")
@router.get("/status")
async def status(manager: LiveManager = Depends(get_live_manager)):
    return await run_in_threadpool(manager.snapshot)


@router.post("/start")
async def start(req: LiveStartRequest, manager: LiveManager = Depends(get_live_manager)):
    try:
        return await run_in_threadpool(_start, req, manager)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, TypeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/bind")
async def bind(manager: LiveManager = Depends(get_live_manager)):
    try:
        request_id = await run_in_threadpool(manager.bind)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"request_id": request_id, "status": "bind-requested"}


@router.post("/clutch")
async def clutch(manager: LiveManager = Depends(get_live_manager)):
    try:
        request_id = await run_in_threadpool(manager.clutch)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"request_id": request_id, "status": "clutch-requested"}


@router.post("/stop")
async def stop(manager: LiveManager = Depends(get_live_manager)):
    try:
        return await run_in_threadpool(manager.stop)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
