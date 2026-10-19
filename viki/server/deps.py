"""
viki.server.deps
----------------
FastAPI dependencies. The camera manager and calibrator are built once in the
app lifespan and stored on ``app.state``; handlers get them via ``Depends`` so
they don't reach into app state directly. The live session owner reuses these
camera resources and supervises an independent perception worker and arm
executor process.
"""

from __future__ import annotations

from fastapi import Request

from viki.calibration.manager import CalibrationManager
from viki.cameras.manager import CameraManager
from viki.replay import ReplayManager
from viki.live import LiveManager, RecordedReviewStore


def get_manager(request: Request) -> CameraManager:
    """The application's CameraManager."""
    return request.app.state.manager


def get_calibrator(request: Request) -> CalibrationManager:
    """The application's CalibrationManager."""
    return request.app.state.calibrator


def get_replay_manager(request: Request) -> ReplayManager:
    return request.app.state.replay_manager


def get_live_manager(request: Request) -> LiveManager:
    return request.app.state.live_manager


def get_recorded_reviews(request: Request) -> RecordedReviewStore:
    return request.app.state.recorded_reviews
