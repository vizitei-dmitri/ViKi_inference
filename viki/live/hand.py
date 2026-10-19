"""Finger-only causal camera sessions using the original DG5F Hybrid solver.

Perception/retarget run in one observation worker. The main thread displays
latest kinematic states independently; it never sends a hardware command.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import signal
import select
import sys
import threading
import time
import uuid

import numpy as np

from viki.contracts import Episode
from viki.cameras import CameraFrameSource, CameraManager, RecordedFrameSource
from viki.perception import LiveHandEstimator
from viki.retarget import DG5FHandRetargeter, DG5F_JOINT_NAMES
from viki.replay import DG5FHandPreview, DG5FHandController
from .session import _safe
from .sources import start_reference_cameras, wait_reference_cameras


class LiveFingerSession:
    def __init__(self, reference, *, source="recorded", hybrid_root="/opt/dg5f-hybrid",
                 seconds=60., perception_hz=15., ttl_s=.25, filter_beta=30., out="data/live_hand_sessions", controller=None):
        for value in (seconds,perception_hz,ttl_s,filter_beta):
            if not np.isfinite(value):
                raise ValueError("hand settings must be finite")
        if source not in ("recorded","cameras") or not 0 < seconds <= 3600 or not 0 < perception_hz <= 30 or not 0 < ttl_s <= 1 or filter_beta < 0:
            raise ValueError("invalid hand-session settings")
        self.reference,self.source = str(reference),source
        self.seconds,self.hz,self.ttl,self.beta = seconds,perception_hz,ttl_s,filter_beta
        self.hybrid_root = hybrid_root
        self.controller = controller
        self.directory = Path(out)/uuid.uuid4().hex
        self.directory.mkdir(parents=True,exist_ok=False)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.worker = threading.Thread(target=self._run,name="finger-observation",daemon=True)
        self.report = {"scope":"camera landmarks -> original Hybrid -> kinematic DG5F preview; no hardware",
            "source_kind":source,"reference":self.reference,"perception_hz":perception_hz,
            "ttl_s":ttl_s,"filter":{"min_cutoff_hz":2.,"beta":filter_beta,"quality_power":0.},
            "anatomical_fit":False,"counts":{},"error":None}
        self.model_path = None

    def _run(self):
        estimator = source = manager = None
        counts = Counter()
        try:
            retarget = DG5FHandRetargeter(self.hybrid_root,ttl_s=self.ttl)
            self.model_path = retarget.model_path
            estimator = LiveHandEstimator.from_episode(Episode(Path(self.reference)),landmark_mode="all21",
                max_age_s=self.ttl,min_cutoff_hz=2.,beta=self.beta,quality_power=0.)
            self.report.update(retarget=retarget.manifest,perception=estimator.manifest)
            if self.controller is not None:
                self.report["executor"] = self.controller.manifest()
                self.controller.start()
            if self.stop_event.is_set():
                return
            if self.source == "recorded":
                source = RecordedFrameSource(Episode(Path(self.reference)),device_ids=estimator.device_ids)
            else:
                manager = CameraManager()
                start_reference_cameras(self.reference,manager,estimator.device_ids)
                wait_reference_cameras(manager,estimator.device_ids,self.stop_event)
                estimator.verify_cameras(manager)
                source = CameraFrameSource(manager,device_ids=estimator.device_ids,
                    sync_fps=int(self.hz),max_age_s=self.ttl)
            self.report["source"] = source.source_summary
            self.ready.set()
            end = time.monotonic()+self.seconds
            with (self.directory/"hand.jsonl").open("w") as hand_log, (self.directory/"targets.jsonl").open("w") as target_log:
                while not self.stop_event.is_set() and time.monotonic() < end and not source.exhausted:
                    begin = time.monotonic()
                    group = source.poll()
                    if group is not None:
                        estimate = estimator.process(group)
                        if estimate is not None:
                            started = time.monotonic()
                            target = retarget.step(estimate)
                            record = asdict(target)
                            record["retarget_compute_ms"] = (time.monotonic()-started)*1000
                            hand_log.write(json.dumps(_safe(asdict(estimate)),allow_nan=False)+"\n")
                            target_log.write(json.dumps(_safe(record),allow_nan=False)+"\n")
                            counts["estimates"] += 1
                            counts["valid_targets"] += int(target.valid)
                            counts["contact_frames"] += int(target.valid and np.max(target.contact_weights)>0)
                            if not target.valid:
                                counts[target.reason] += 1
                            with self.lock:
                                self.latest = target
                            if self.controller is not None:
                                self.controller.submit(target)
                    self.stop_event.wait(max(0.,1/self.hz-(time.monotonic()-begin)))
            self.report["reason"] = "source_exhausted" if source.exhausted else "stopped_or_duration_limit"
        except Exception as exc:
            if not (self.stop_event.is_set() and str(exc) == "operator_stop"):
                self.report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if (self.controller is not None and self.controller.process is not None
                    and self.controller.process.is_alive() and not self.controller.stop.is_set()):
                try:
                    self.controller.submit(None)
                except Exception as exc:
                    if not self.controller.stop.is_set():
                        self.report["error"] = f"clear target: {exc}"
            for resource in (source,estimator):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.report["error"] = f"cleanup: {exc}"
            if manager is not None:
                try:
                    manager.stop_all()
                except Exception as exc:
                    self.report["error"] = f"camera cleanup: {exc}"
            self.report["counts"] = dict(counts)
            self.ready.set()

    def current(self, now_us=None):
        now_us = time.monotonic_ns()//1000 if now_us is None else now_us
        with self.lock:
            target = self.latest
        if target is None or not target.valid or now_us > target.valid_until_us:
            return None
        return target


def run_hand_cli(args):
    if args.backend == "tesollo" and (not args.execute or args.source != "cameras"):
        raise ValueError("physical DG5F requires --execute and --source cameras")
    if args.auto_arm and args.backend != "mock":
        raise ValueError("--auto-arm is available only with --backend mock")
    if args.execute and args.backend != "tesollo":
        raise ValueError("--execute applies only to the physical tesollo backend")
    session = LiveFingerSession(args.reference,source=args.source,hybrid_root=args.hybrid_root,
        seconds=args.seconds,perception_hz=args.perception_hz,ttl_s=args.input_ttl,
        filter_beta=args.filter_beta,out=args.out)
    controller = None
    if args.backend != "preview":
        controller = DG5FHandController(args.hybrid_root,backend=args.backend,execute=args.execute,
            host=args.hand_ip,port=args.hand_port,slave_id=args.hand_id,ttl_s=args.input_ttl,
            speed=args.servo_speed,acceleration=args.servo_acceleration,out_dir=session.directory/"executor")
        session.controller = controller
        session.report["scope"] = "camera landmarks -> original Hybrid -> original guarded DG5F servo"
    session.report["backend"] = args.backend
    session.report["hardware"] = args.backend == "tesollo"
    interrupted = [False]
    def request_stop():
        session.stop_event.set()
        if controller is not None:
            controller.stop.set()
    old_handlers = {s:signal.signal(s,lambda *_:interrupted.__setitem__(0,True)) for s in (signal.SIGINT,signal.SIGTERM)}
    viewer = renderer = video = None
    preview = None
    states = None
    last_sequence = -1
    counts = Counter()
    try:
        session.worker.start()
        print(json.dumps({"directory":str(session.directory),"source":args.source,"hardware":args.backend == "tesollo"}),flush=True)
        while not session.ready.wait(.05):
            if interrupted[0]:
                request_stop()
            if session.stop_event.is_set():
                break
        if interrupted[0]:
            request_stop()
        if session.report["error"]:
            raise RuntimeError(session.report["error"])
        if session.stop_event.is_set():
            return
        preview = DG5FHandPreview(session.model_path,DG5F_JOINT_NAMES)
        states = (session.directory/"preview.jsonl").open("w")
        if controller is not None:
            print("DG5F DISARMED: a + Enter = ARM, d = DISARM, r = RECOVER (stays disarmed), q = stop.",flush=True)
            if args.auto_arm:
                controller.request("arm")
        last_control = None
        if args.gui:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(preview.model,preview.data)
            viewer.cam.lookat[:] = preview.camera.lookat
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = preview.camera.distance,preview.camera.azimuth,preview.camera.elevation
        if args.video:
            import cv2
            renderer = preview.mj.Renderer(preview.model,height=480,width=640)
            video = cv2.VideoWriter(str(session.directory/"preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),30.,(640,480))
            if not video.isOpened():
                raise RuntimeError("could not open preview video encoder")
        while session.worker.is_alive() and not session.stop_event.is_set():
            if interrupted[0]:
                request_stop()
                break
            begin = time.monotonic()
            target = session.current()
            if controller is not None:
                snapshot = controller.snapshot()
                status_key = (snapshot.get("runtime_state"),snapshot.get("error"),repr(snapshot.get("request")))
                if status_key != last_control:
                    print(json.dumps({k:snapshot.get(k) for k in ("runtime_state","error","request")}),flush=True)
                    last_control = status_key
                if not controller.process.is_alive() or not snapshot.get("owner_running",True):
                    raise RuntimeError(snapshot.get("error") or "DG5F owner stopped")
                if sys.stdin.isatty() and select.select([sys.stdin],[],[],0)[0]:
                    key = sys.stdin.readline().strip().lower()
                    if key == "q":
                        session.stop_event.set()
                    elif key in ("a","d","r"):
                        controller.request({"a":"arm","d":"disarm","r":"recover"}[key])
                measured = np.asarray(snapshot.get("telemetry",{}).get("pos"),float)
                if (measured.shape == (20,) and np.isfinite(measured).all() and
                        snapshot.get("hardware",{}).get("last_position_sample_age_ms",float("inf"))<=100):
                    counts["clipped_joints"] += preview.apply(np.deg2rad(measured))
                    counts["measured_updates"] += 1
                    states.write(json.dumps({"host_time_us":time.monotonic_ns()//1000,
                        "source":"SDK telemetry","q_rad":preview.q.tolist(),"runtime_state":snapshot.get("runtime_state")})+"\n")
            elif target is not None and target.sequence != last_sequence:
                counts["clipped_joints"] += preview.apply(target.q_rad)
                counts["applied_targets"] += 1
                last_sequence = target.sequence
                states.write(json.dumps({"sequence":target.sequence,"source_time_us":target.source_time_us,
                    "host_time_us":time.monotonic_ns()//1000,"q_rad":preview.q.tolist()})+"\n")
            if viewer is not None:
                if not viewer.is_running():
                    session.stop_event.set()
                    break
                viewer.sync()
            if video is not None:
                video.write(preview.render(renderer)[...,::-1])
            session.stop_event.wait(max(0.,1/30-(time.monotonic()-begin)))
        if renderer is not None:
            import cv2
            cv2.imwrite(str(session.directory/"preview.png"),preview.render(renderer)[...,::-1])
    except Exception as exc:
        if not (session.stop_event.is_set() and str(exc) in ("operator_stop","DG5F owner stopped")):
            session.report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.stop_event.set()
        if controller is not None:
            controller.stop.set()
        session.worker.join(timeout=15)
        if controller is not None:
            try:
                final = controller.close()
                (session.directory/"executor").mkdir(parents=True,exist_ok=True)
                (session.directory/"executor/final.json").write_text(json.dumps(_safe(final),indent=2,allow_nan=False)+"\n")
                session.report["executor_final"] = {k:final.get(k) for k in
                    ("runtime_state","state_before_close","closed","armed","arm_count","error","source_sequence")}
                if final.get("runtime_state") == "FAULT":
                    session.report["error"] = final.get("error") or "DG5F executor fault"
            except Exception as exc:
                session.report["error"] = f"executor shutdown: {exc}"
        for resource in (states,video,renderer,viewer):
            if resource is not None:
                (resource.release if resource is video else resource.close)()
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
        session.report["preview"] = dict(counts)
        if session.worker.is_alive():
            session.report["error"] = "observation worker did not stop"
        (session.directory/"session.json").write_text(json.dumps(_safe(session.report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(_safe(session.report),indent=2,allow_nan=False),flush=True)
    if session.report["error"]:
        raise SystemExit(1)
