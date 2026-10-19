"""Source-clock gating, reliable overwrite/clear and original guarded runtime."""
from dataclasses import dataclass
from types import SimpleNamespace
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time

import numpy as np
import pytest

from viki.contracts import HandJointTarget
from viki.replay import DG5FHandController
from viki.replay.dg5f import _LatestMailbox, _target_for_runtime, _shutdown_runtime


NAMES = tuple(f"rj_dg_{f}_{j}" for f in range(1,6) for j in range(1,5))


def target(sequence=1,now_us=None):
    now_us = time.monotonic_ns()//1000 if now_us is None else now_us
    q = np.zeros(20); q[5:8] = .3
    return HandJointTarget(sequence,now_us,now_us+250000,NAMES,q,np.zeros(5),np.full(7,.08),True)


@dataclass
class ExternalTarget:
    q_deg: np.ndarray
    captured_at: float
    weights: np.ndarray
    distances: np.ndarray


def test_runtime_target_keeps_capture_clock_and_canonical_units():
    t = target(now_us=1_000_000)
    result = _target_for_runtime(t,1_200_000,ExternalTarget)
    assert result.captured_at == 1.
    np.testing.assert_allclose(result.q_deg,np.rad2deg(t.q_rad))
    assert _target_for_runtime(t,1_250_001,ExternalTarget) is None
    assert _target_for_runtime(t,999_999,ExternalTarget) is None
    t.joint_names = NAMES[::-1]
    assert _target_for_runtime(t,1_100_000,ExternalTarget) is None


@pytest.mark.parametrize("kind",["invalid","nan","broken","weights","distances"])
def test_runtime_boundary_rejects_bad_targets(kind):
    t = target(now_us=1_000_000)
    if kind == "invalid": t.valid = False
    if kind == "nan": t.q_rad[0] = np.nan
    if kind == "broken": t.q_rad[16] = 1
    if kind == "weights": t.contact_weights[0] = 2
    if kind == "distances": t.fingertip_distances_m[0] = -1
    assert _target_for_runtime(t,1_100_000,ExternalTarget) is None


def test_shared_latest_slot_cannot_drop_invalidating_clear():
    slot = _LatestMailbox(mp.get_context("spawn"),8192)
    for sequence in range(100):
        slot.put_nowait(target(sequence))
    slot.put_nowait(None)
    assert slot.get_nowait() is None
    with pytest.raises(queue.Empty):
        slot.get_nowait()


def test_physical_connection_requires_explicit_execute_before_any_process():
    with pytest.raises(ValueError,match="--execute"):
        DG5FHandController(backend="tesollo")


@pytest.mark.parametrize("connected,errors",[(True,[]),(False,["vendor stop failed"])])
def test_shutdown_does_not_claim_success_after_cleanup_failure(connected,errors):
    runtime = SimpleNamespace(close=lambda:None,engine=SimpleNamespace(backend=SimpleNamespace(is_connected=connected)))
    with pytest.raises(RuntimeError):
        _shutdown_runtime(runtime,errors)


def wait_for(controller,predicate,timeout=3):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        value = controller.snapshot()
        if predicate(value):
            return value
        time.sleep(.01)
    raise AssertionError(f"DG5F state timeout: {controller.snapshot()}")


@pytest.fixture
def controller(tmp_path):
    if not Path("/opt/dg5f-hybrid/src/dg5f_demo/runtime.py").exists():
        pytest.skip("original DG5F runtime mounted in live-hand image")
    c = DG5FHandController(out_dir=tmp_path)
    try:
        c.start()
        yield c
    finally:
        c.close()


def test_mock_owner_requires_arm_holds_on_source_expiry_and_closes(controller):
    c = controller
    assert c.process.pid != mp.current_process().pid
    assert c.snapshot()["runtime_state"] == "DISARMED"
    c.submit(target())
    time.sleep(.05)
    np.testing.assert_allclose(c.snapshot()["telemetry"]["pos"],np.zeros(20))
    c.request("arm")
    wait_for(c,lambda s:s.get("armed"))
    c.submit(target(2))
    active = wait_for(c,lambda s:s.get("runtime_state")=="ACTIVE")
    assert active["arm_count"] == 1
    held = wait_for(c,lambda s:s.get("runtime_state")=="TRACKING_HOLD")
    q = held["q_cmd_deg"]
    time.sleep(.1)
    np.testing.assert_allclose(c.snapshot()["q_cmd_deg"],q)
    c.submit(target(3))
    wait_for(c,lambda s:s.get("runtime_state")=="ACTIVE")
    c.submit(None)
    wait_for(c,lambda s:s.get("runtime_state")=="TRACKING_HOLD")
    c.request("disarm")
    wait_for(c,lambda s:s.get("runtime_state")=="DISARMED")
    c.submit(target(4))
    time.sleep(.05)
    assert not c.snapshot()["armed"]
    final = c.close()
    assert final["closed"] and not final["armed"] and not c.process.is_alive()


def test_explicit_recover_stays_disarmed(controller):
    c = controller
    request = c.request("recover")
    state = wait_for(c,lambda s:s.get("request",{}).get("id")==request if s.get("request") else False)
    assert state["request"]["accepted"]
    assert state["runtime_state"] == "DISARMED" and not state["armed"]


def test_owner_ignores_terminal_interrupt_and_handles_termination(controller):
    c = controller
    os.kill(c.process.pid,signal.SIGINT)
    time.sleep(.05)
    assert c.process.is_alive()
    os.kill(c.process.pid,signal.SIGTERM)
    c.process.join(5)
    assert not c.process.is_alive()
    assert c.close()["closed"]


def _producer_for_parent_loss(pipe, out_dir):
    c = DG5FHandController(out_dir=out_dir)
    c.start()
    pipe.send(c.process.pid)
    while True:
        time.sleep(1)


def test_mock_sdk_owner_stops_when_inference_parent_dies(tmp_path):
    if not Path("/opt/dg5f-hybrid/src/dg5f_demo/runtime.py").exists():
        pytest.skip("original DG5F runtime required")
    ctx = mp.get_context("spawn")
    receive,send = ctx.Pipe(duplex=False)
    producer = ctx.Process(target=_producer_for_parent_loss,args=(send,str(tmp_path)))
    owner_pid = None
    def alive(pid):
        stat = Path(f"/proc/{pid}/stat")
        return stat.exists() and stat.read_text().split()[2] != "Z"
    try:
        producer.start()
        assert receive.poll(10), "producer did not initialize"
        owner_pid = receive.recv()
        producer.terminate()
        producer.join(3)
        end = time.monotonic()+5
        while alive(owner_pid) and time.monotonic()<end:
            time.sleep(.02)
        assert not alive(owner_pid), "SDK owner outlived its inference parent"
    finally:
        if producer.is_alive():
            producer.terminate(); producer.join(3)
        if owner_pid is not None and alive(owner_pid):
            os.kill(owner_pid,15)
        receive.close(); send.close()
