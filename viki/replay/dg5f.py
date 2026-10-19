"""Isolated SDK owner around the user's existing 60 Hz guarded DG5F Runtime.

Camera/inference/display never touch the backend. The original servo, current,
contact, telemetry, measured-pose ARM and recovery policy remain authoritative.
"""
import hashlib
import importlib
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid

import numpy as np

from viki.contracts import HandJointTarget


_JOINT_NAMES = tuple(f"rj_dg_{f}_{j}" for f in range(1,6) for j in range(1,5))


def _load_runtime(root):
    source = Path(root).resolve()/"src"
    if not (source/"dg5f_demo/runtime.py").is_file():
        raise FileNotFoundError(f"original DG5F control runtime missing: {source}")
    sys.path.insert(0,str(source))
    runtime = importlib.import_module("dg5f_demo.runtime")
    backends = importlib.import_module("lerobot_robot_dg5f.backends")
    for module in (runtime,backends):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("another DG5F package is already imported")
    return runtime,backends


def _target_for_runtime(target, now_us, target_factory):
    """Check original source time; never make a delayed target fresh again."""
    if target is None or not target.valid:
        return None
    if (tuple(target.joint_names) != _JOINT_NAMES or
            not 0 <= target.source_time_us <= now_us <= target.valid_until_us):
        return None
    q,w,d = (np.asarray(a,float) for a in (target.q_rad,target.contact_weights,target.fingertip_distances_m))
    if (q.shape != (20,) or w.shape != (5,) or d.shape != (7,) or
            not all(np.isfinite(a).all() for a in (q,w,d)) or
            np.any((w<0)|(w>1)) or np.any(d<0) or abs(q[16])>1e-10):
        return None
    return target_factory(np.rad2deg(q),target.source_time_us/1e6,w,d)


def _latest_put(mailbox, value):
    mailbox.put_nowait(value)  # Shared slot overwrites; no Queue feeder involved.


class _LatestMailbox:
    """Overwrite one shared slot, including explicit None; no Queue feeder race."""
    def __init__(self, context, capacity):
        self.buffer = context.RawArray("B",capacity)
        self.length = context.Value("i",0,lock=False)
        self.lock = context.Lock()

    def put_nowait(self, value):
        payload = pickle.dumps(value,protocol=5)
        if len(payload)>len(self.buffer):
            raise ValueError("DG5F mailbox payload exceeds fixed capacity")
        with self.lock:
            self.buffer[:len(payload)] = payload
            self.length.value = len(payload)

    def get_nowait(self):
        with self.lock:
            if not self.length.value:
                raise queue.Empty
            payload = bytes(self.buffer[:self.length.value])
            self.length.value = 0
        return pickle.loads(payload)

    def close(self):
        pass

    def cancel_join_thread(self):
        pass


def _verify_sdk(root):
    provenance = Path("/opt/dg5f-sdk/provenance.json")
    if not provenance.exists():
        raise RuntimeError("DG5F SDK build missing; use the live-hand-hardware image")
    for relative,digest in json.loads(provenance.read_text()).items():
        source = Path(root)/"vendor/tesollo_control"/relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"DG5F SDK source changed: {relative}; rebuild live-hand-hardware")


def _shutdown_runtime(runtime, worker_errors):
    runtime.close()
    # Original Runtime joins its thread, but an exception in that thread's
    # finally block is reported through excepthook rather than close().
    if worker_errors:
        raise RuntimeError("SDK worker cleanup failed: " + "; ".join(worker_errors))
    if runtime.engine.backend.is_connected:
        raise RuntimeError("SDK backend is still connected after shutdown")


def _owner(config, targets, commands, status, stop):
    terminated = [False]
    signal.signal(signal.SIGINT,signal.SIG_IGN)  # Parent handles terminal Ctrl+C.
    # Event.set() can deadlock if a signal interrupts Event.wait() under its
    # non-reentrant condition lock. Signal handlers must only set a Python flag.
    signal.signal(signal.SIGTERM,lambda *_:terminated.__setitem__(0,True))
    runtime = None
    worker_errors = []
    last_sequence = -1
    last_action = None
    try:
        module,backends = _load_runtime(config["root"])
        if config["backend"] == "tesollo":
            _verify_sdk(config["root"])
        backend = (backends.MockDg5fBackend() if config["backend"] == "mock" else
                   backends.TesolloDg5fBackend(config["host"],config["port"],config["slave_id"],servo_keepalive=True))
        runtime = module.Runtime(backend,record_path=config["log"],
            max_velocity_deg_s=config["speed"],max_acceleration_deg_s2=config["acceleration"])
        runtime.engine.tracking_timeout_s = config["ttl_s"]
        if stop.is_set() or terminated[0]:
            runtime = None  # No SDK thread was started, so there is nothing to join.
            _latest_put(status,dict(runtime_state="CLOSED",closed=True,armed=False,owner_running=False))
            return
        def record_thread_error(args):
            if args.thread is runtime.thread:
                worker_errors.append(f"{args.exc_type.__name__}: {args.exc_value}")
            threading.__excepthook__(args)
        threading.excepthook = record_thread_error
        runtime.start()
        parent = mp.parent_process()
        while not stop.is_set() and not terminated[0] and (parent is None or parent.is_alive()):
            while True:
                try:
                    target = targets.get_nowait()
                except queue.Empty:
                    break
                if target is None:
                    runtime.publish_target(None)
                elif target.sequence > last_sequence:
                    last_sequence = target.sequence
                    runtime.publish_target(_target_for_runtime(target,time.monotonic_ns()//1000,module.Target))
            try:
                request_id,action,expires = commands.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    if stop.is_set() or time.monotonic()>expires:
                        raise RuntimeError("operator action expired")
                    result = runtime.request(action)
                    last_action = dict(id=request_id,action=action,accepted=True,result=result)
                except Exception as exc:
                    last_action = dict(id=request_id,action=action,accepted=False,error=str(exc))
            snapshot = runtime.snapshot()
            snapshot = dict(snapshot,owner_running=runtime.thread.is_alive(),request=last_action,
                            source_sequence=last_sequence,backend=config["backend"])
            _latest_put(status,snapshot)
            if not runtime.thread.is_alive():
                break
            stop.wait(.01)
    except Exception as exc:
        _latest_put(status,dict(runtime_state="FAULT",error=f"{type(exc).__name__}: {exc}",owner_running=False,closed=runtime is None))
    finally:
        if runtime is not None:
            try:
                _shutdown_runtime(runtime,worker_errors)
                snapshot = runtime.snapshot()
                disconnected = not runtime.engine.backend.is_connected
                _latest_put(status,dict(snapshot,owner_running=False,closed=disconnected,armed=False,
                    state_before_close=snapshot.get("runtime_state"),
                    runtime_state="FAULT" if snapshot.get("runtime_state")=="FAULT" else "CLOSED",
                    request=last_action,source_sequence=last_sequence,backend=config["backend"]))
            except Exception as exc:
                _latest_put(status,dict(runtime_state="FAULT",error=f"shutdown: {exc}",owner_running=False))


class DG5FHandController:
    def __init__(self, root="/opt/dg5f-hybrid", *, backend="mock", execute=False,
                 host="169.254.186.72", port=502, slave_id=1, ttl_s=.25,
                 speed=120., acceleration=720., out_dir="data/dg5f_executor"):
        if backend not in ("mock","tesollo"):
            raise ValueError("DG5F backend must be mock or tesollo")
        if backend == "tesollo" and not execute:
            raise ValueError("physical DG5F requires --execute; starts DISARMED")
        if not all(np.isfinite(x) and x>0 for x in (ttl_s,speed,acceleration)) or ttl_s>1:
            raise ValueError("invalid DG5F control limits/TTL")
        if not isinstance(host,str) or not host or not 1 <= port <= 65535 or not 1 <= slave_id <= 247:
            raise ValueError("invalid DG5F connection settings")
        self.config = dict(root=str(Path(root).resolve()),backend=backend,host=host,port=port,
            slave_id=slave_id,ttl_s=ttl_s,speed=speed,acceleration=acceleration,
            log=str(Path(out_dir).resolve()/"servo.jsonl"))
        self.context = mp.get_context("spawn")
        self.targets = _LatestMailbox(self.context,8192)
        self.commands = self.context.Queue(maxsize=8)
        self.status = _LatestMailbox(self.context,262144)
        self.stop = self.context.Event()
        self.process = None
        self._snapshot = {"runtime_state":"STARTING","owner_running":False}
        self._last_sequence = -1
        self._last_source = -1
        self._closed = False

    def start(self, timeout=20.):
        if self.process is not None or self._closed:
            raise RuntimeError("create a new DG5F controller for each session")
        self.process = self.context.Process(target=_owner,args=(self.config,self.targets,self.commands,self.status,self.stop),daemon=True)
        self.process.start()
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline and not self.stop.is_set():
            data = self.snapshot()
            if data.get("runtime_state") == "FAULT":
                raise RuntimeError(data.get("error","DG5F startup fault"))
            if data.get("hardware",{}).get("transport_connected"):
                return data
            if not self.process.is_alive():
                raise RuntimeError("DG5F owner exited during startup")
            self.stop.wait(.02)
        if self.stop.is_set():
            raise RuntimeError("operator_stop")
        raise RuntimeError("DG5F startup timeout")

    def submit(self, target: HandJointTarget | None):
        if self.process is None or self._closed or not self.process.is_alive():
            raise RuntimeError("DG5F owner is not running")
        if target is not None:
            if target.sequence <= self._last_sequence or target.source_time_us <= self._last_source:
                _latest_put(self.targets,None)
                return
            self._last_sequence,self._last_source = target.sequence,target.source_time_us
        _latest_put(self.targets,target)

    def request(self, action):
        if action not in ("arm","disarm","recover"):
            raise ValueError("unknown DG5F action")
        if self._closed or self.process is None or not self.process.is_alive():
            raise RuntimeError("DG5F owner is not running")
        request_id = uuid.uuid4().hex
        self.commands.put_nowait((request_id,action,time.monotonic()+2.))
        return request_id

    def snapshot(self):
        while True:
            try:
                self._snapshot = self.status.get_nowait()
            except queue.Empty:
                break
        return dict(self._snapshot)

    def close(self):
        if self._closed:
            return self.snapshot()
        self.stop.set()
        if self.process is not None:
            self.process.join(timeout=10.)
            if self.process.is_alive():
                raise RuntimeError("DG5F SDK owner failed to stop; shutdown is not confirmed")
        self._closed = True
        data = self.snapshot()
        if self.process is None:
            data.update(closed=True,armed=False,runtime_state="CLOSED")
            self._snapshot = data
        for mailbox in (self.targets,self.commands,self.status):
            mailbox.cancel_join_thread()
            mailbox.close()
        if not data.get("closed"):
            raise RuntimeError("DG5F owner exited without confirming SDK disconnect")
        return data

    def manifest(self):
        root = Path(self.config["root"])
        paths = [root/"src/dg5f_demo/runtime.py",root/"src/dg5f_demo/servo.py"]
        paths += [root/f"src/lerobot_robot_dg5f/{n}.py" for n in
                  ("backends","servo_controller","command_shaper","current_guard","object_contact","contact_kinematics","constants")]
        result = {"config":self.config,"control":"original guarded Runtime, single SDK owner process, manual ARM",
            "sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        sdk = Path("/opt/dg5f-sdk/provenance.json")
        if sdk.exists():
            result["sdk_source_sha256"] = json.loads(sdk.read_text())
        return result
