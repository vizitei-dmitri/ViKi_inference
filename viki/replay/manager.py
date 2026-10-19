"""Cancellation for queued/server replay; the driver keeps RTDE in one thread."""

import threading
import uuid

from viki.replay.run import replay_episode


class ReplayManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._events = {}
        self._jobs = {}

    def prepare(self) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._events[token] = threading.Event()
        return token

    def bind_job(self, job_id: str, token: str):
        with self._lock:
            if token in self._events:
                self._jobs[job_id] = token

    def run(self, token: str, ep, **kwargs):
        with self._lock:
            event = self._events[token]
        try:
            if event.is_set():
                raise RuntimeError("replay cancelled before execution")
            return replay_episode(ep, stop_event=event, **kwargs)
        finally:
            with self._lock:
                self._events.pop(token, None)
                self._jobs = {job: value for job, value in self._jobs.items() if value != token}

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            token = self._jobs.get(job_id)
            event = self._events.get(token)
            if event is None:
                return False
            event.set()
        return True

    def forget_job(self, job_id: str):
        with self._lock:
            token = self._jobs.pop(job_id, None)
            self._events.pop(token, None)

    def close(self):
        """Request stop on server shutdown; never call RTDE from this thread."""
        with self._lock:
            for event in self._events.values():
                event.set()
