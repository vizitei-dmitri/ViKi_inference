"""Interactive CLI orchestration through the same public session API as HTTP."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import select
import signal
import sys
import threading
import time

from viki.contracts import Episode, LiveControlOptions
from .session import LiveSession, LiveSessionConfig, load_mapping


def run_live_cli(args):
    changes = {"frequency": args.control_hz, "scale": args.scale,
               "orientation": args.orientation, "host": args.host,
               "hardware_enabled": args.execute, "model_path": args.model,
               "urdf_path": args.urdf}
    if args.initial_q is not None:
        changes["initial_q"] = tuple(args.initial_q)
    mapping_manifest = {}
    if args.mapping_json:
        mapping, mapping_manifest = load_mapping(args.mapping_json)
        changes.update(mapping)
    options = replace(LiveControlOptions(), **changes)
    cfg = LiveSessionConfig(reference=args.reference, source=args.source, backend=args.backend,
        perception_hz=args.perception_hz, recorded_speed=args.recorded_speed,
        landmark_mode=args.landmarks, seconds=args.seconds, auto_bind=args.auto_bind,
        out_dir=args.out, options=options, mapping_manifest=mapping_manifest)
    cfg.validate()
    if args.check:
        from viki.perception import LiveHandEstimator
        from viki.retarget import OnlineRetargeter

        estimator = LiveHandEstimator.from_episode(Episode(Path(cfg.reference)),
            landmark_mode=cfg.landmark_mode, max_age_s=options.input_ttl_s)
        try:
            expected = mapping_manifest.get("source_calibration_id")
            if expected and expected != estimator.calibration_id:
                raise ValueError("mapping belongs to another camera calibration")
            ik = OnlineRetargeter(options)
            print(json.dumps({"scope": "offline reference/model check; no camera or robot connection",
                "perception": estimator.manifest, "retarget": ik.manifest,
                "initial_model_margins": ik.model_validate(options.initial_q)}, indent=2))
        finally:
            estimator.close()
        return

    stop = threading.Event()
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    session = LiveSession(cfg)
    viewer = None
    final = None
    last_state = None
    try:
        if args.gui:
            from .viewer import LiveViewer
            viewer = LiveViewer(options)
            viewer.start()
        session.start()
        print("Live: b + Enter = bind, c + Enter = clutch, q + Enter = stop.", flush=True)
        if viewer:
            print("MuJoCo window: Space = bind/clutch, B/C/Q = bind/clutch/stop.", flush=True)
        while not stop.is_set():
            status = session.snapshot()
            controller = status.get("controller", {})
            key = (status.get("phase"), controller.get("state"), controller.get("reason"),
                   repr(controller.get("request_result")))
            if key != last_state:
                print(json.dumps({"phase": status.get("phase"), "state": controller.get("state"),
                    "reason": controller.get("reason") or status.get("reason"),
                    "request_result": controller.get("request_result"),
                    "frequency_hz": controller.get("frequency_hz"),
                    "directory": status["directory"]}), flush=True)
                last_state = key
            if not status["running"]:
                break
            commands = []
            if viewer:
                viewer.update(controller.get("q"))
                commands.extend(viewer.commands())
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                command = {"b": "bind", "c": "clutch", "q": "stop"}.get(sys.stdin.readline().strip())
                if command:
                    commands.append(command)
            for command in commands:
                try:
                    if command == "toggle":
                        command = "clutch" if controller.get("state") in ("active", "braking") else "bind"
                    if command == "stop":
                        stop.set()
                    elif command == "bind":
                        print("Bind requested: " + session.bind(), flush=True)
                    elif command == "clutch":
                        print("Clutch requested: " + session.clutch(), flush=True)
                except (ValueError, RuntimeError) as exc:
                    print(str(exc), file=sys.stderr, flush=True)
            stop.wait(.05)
    finally:
        final = session.stop(timeout=15)
        if viewer:
            viewer.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps(final, indent=2, allow_nan=False))
    if final.get("phase") == "fault" or final.get("running") or final.get("controller", {}).get("running"):
        raise SystemExit(1)
