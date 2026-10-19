#!/usr/bin/env python3
"""Measure frame-local perception components from a recorded RGB-D episode.

Run inside the normal ViKi Docker image, for example::

    docker compose run --rm --entrypoint python3 test \
        scripts/benchmark_live_perception.py data/datasets/<name>/<episode> \
        --out data/live_perception_benchmark.json

No camera is opened, no robot is connected, and no episode/cache/baseline is
modified. Only the explicit JSON output is written. Video decoding and depth
file reads are excluded from component timings; initialization and warmup are
reported separately. Two persistent per-camera MediaPipe VIDEO detectors feed
the existing V4 depth sampler and frame-local triangulator. The 21-landmark
and seven-landmark paths use the same detections, without downscaling images.

This is a component benchmark, not a complete causal live implementation,
an end-to-end latency/FPS measurement, or a ground-truth accuracy evaluation.
It excludes sensor/USB latency, synchronization, causal filters, hand fitting,
IK, networking, servo control, logging and rendering. Exact cached SDK
color-to-depth projectors are required; there is no identity/preset fallback.
The current triangulator still uses its existing saved K/dist lens model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace


ARM_LANDMARKS = (0, 1, 4, 5, 8, 9, 17)
PROFILE = "stable-fused-hand-v4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values, np) -> dict:
    samples = np.asarray(values, dtype=float)
    return {
        "samples": int(samples.size),
        "mean_ms": float(samples.mean() * 1000.0),
        "median_ms": float(np.median(samples) * 1000.0),
        "p95_ms": float(np.percentile(samples, 95) * 1000.0),
        "p99_ms": float(np.percentile(samples, 99) * 1000.0),
        "max_ms": float(samples.max() * 1000.0),
    }


def benchmark(episode: Path, *, frames: int, warmup: int, detector_mode: str) -> dict:
    imports_started = time.perf_counter()
    import cv2
    import numpy as np

    from viki.contracts import CalibrationExtrinsics, PreparedFrame
    from viki.perception.backends.mediapipe import MediaPipeHandBackend
    from viki.perception.k4a_offline import K4ACalibration
    from viki.perception.observations import collect_row
    from viki.perception.profiles import get_profile
    from viki.perception.rs_offline import RealSenseCalibration
    from viki.perception.triangulate import _Cam, TriConfig, triangulate_joint

    import_s = time.perf_counter() - imports_started
    raw = episode / "raw"
    meta = json.loads((episode / "meta.json").read_text())
    intrinsics = json.loads((raw / "intrinsics.json").read_text())
    extrinsics = json.loads((raw / "extrinsics.json").read_text())
    timestamps = json.loads((raw / "timestamps.json").read_text())
    videos = sorted(raw.glob("*.mp4"))
    if len(videos) != 2:
        raise ValueError(f"benchmark requires exactly two recorded cameras, found {len(videos)}")
    if len(timestamps) < warmup + frames:
        raise ValueError(f"need {warmup + frames} raw groups, found {len(timestamps)}")
    profile = get_profile(PROFILE)
    tri_config = TriConfig(profile.triangulation)
    if tri_config.time_align:
        raise ValueError("single-group benchmark cannot apply future-frame time alignment")

    inputs = [episode / "meta.json", raw / "intrinsics.json", raw / "extrinsics.json",
              raw / "timestamps.json"]
    contexts = []
    setup_started = time.perf_counter()
    try:
        for video in videos:
            device = video.stem
            if device not in intrinsics or device not in extrinsics:
                raise ValueError(f"missing recorded intrinsics/extrinsics for {device}")
            camera_meta = meta.get("cameras", {}).get(device, {})
            sensor_type = camera_meta.get("type", "")
            calibration_started = time.perf_counter()
            if sensor_type == "kinect":
                projector = K4ACalibration.from_episode(raw, device, meta)
                calibration_path = raw / camera_meta.get("k4a_calib", f"{device}_k4a_calib.bin")
            elif sensor_type == "realsense":
                projector = RealSenseCalibration.from_episode(raw, device, meta)
                calibration_path = raw / camera_meta.get("rs_calib", f"{device}_rs_calib.json")
            else:
                raise ValueError(f"unsupported recorded sensor type {sensor_type!r} for {device}")
            if projector is None:
                raise RuntimeError(f"cannot rebuild cached SDK calibration for {device}; no fallback")
            calibration_s = time.perf_counter() - calibration_started
            inputs.append(calibration_path)
            color = intrinsics[device]["color"]
            depth = intrinsics[device]["depth"]
            color_k = np.array([[color["fx"], 0.0, color["cx"]],
                                [0.0, color["fy"], color["cy"]], [0.0, 0.0, 1.0]])
            depth_k = np.array([[depth["fx"], 0.0, depth["cx"]],
                                [0.0, depth["fy"], depth["cy"]], [0.0, 0.0, 1.0]])
            extr = extrinsics[device]
            world = CalibrationExtrinsics(
                rvec=np.asarray(extr["rvec"]), tvec=np.asarray(extr["tvec"]),
            ).transform_matrix
            tri_camera = _Cam(device, {
                "K": color_k.tolist(), "dist": color.get("dist_coeffs", [0.0] * 5),
                "T_wc": world.tolist(), "image_size": [color["width"], color["height"]],
            })
            radius = profile.depth_radius_px
            if profile.depth_radius_mrad is not None:
                focal = 0.5 * (depth["fx"] + depth["fy"])
                radius = max(1, int(round(profile.depth_radius_mrad / 1000.0 * focal)))
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"cannot decode {video}")
            # Register cleanup before model initialization, which may fail.
            context = {"capture": capture, "projector": projector, "detector": None}
            contexts.append(context)
            detector_started = time.perf_counter()
            detector = MediaPipeHandBackend(
                mode=detector_mode, min_confidence=profile.min_confidence,
                tracking_confidence=profile.tracking_confidence,
                model_path=str(Path("models/hand_landmarker.task").resolve()),
            )
            detector_s = time.perf_counter() - detector_started
            context.update({
                "device": device, "projector": projector, "depth_k": depth_k,
                "tri_camera": tri_camera, "depth_radius_px": radius, "capture": capture,
                "detector": detector, "calibration_init_s": calibration_s,
                "detector_init_s": detector_s,
                "requested": camera_meta.get("requested"),
                "video": str(video), "color_shape": camera_meta.get("color_shape"),
                "depth_shape": camera_meta.get("depth_shape"),
            })

        setup_s = time.perf_counter() - setup_started
        hashes = {str(path): _sha256(path) for path in inputs}
        cameras = {c["device"]: c["tri_camera"] for c in contexts}
        measured = []
        warmup_totals = []
        all_started = time.perf_counter()
        for index in range(warmup + frames):
            row = {"frame_index": index, "sync_us": int(timestamps[index]["sync_us"]),
                   "detected": {}, "per_camera_ms": {}}
            detections = {}
            prepared_by_device = {}
            preparation_s = detection_s = 0.0
            for context in contexts:
                device = context["device"]
                # Disk/decoder work happens entirely before this camera's timed stages.
                decoded, bgr = context["capture"].read()
                depth_path = raw / f"{device}_depth" / f"{index:06d}.npy"
                if not decoded or not depth_path.is_file():
                    raise RuntimeError(f"missing complete recorded RGB-D pair: {device} frame {index}")
                depth_mm = np.load(depth_path, allow_pickle=False)
                started = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth_m[depth_mm == 0] = np.nan
                offsets = timestamps[index].get("offsets_us", {})
                timestamp_us = row["sync_us"] + int(offsets.get(device, 0))
                prepared = PreparedFrame(
                    rgb=rgb, depth_m=depth_m, depth_K=context["depth_k"],
                    device_id=device, timestamp_us=timestamp_us,
                )
                preparation_elapsed = time.perf_counter() - started
                started = time.perf_counter()
                detection = context["detector"].detect(prepared, meta.get("hand", "right"))
                detection_elapsed = time.perf_counter() - started
                preparation_s += preparation_elapsed
                detection_s += detection_elapsed
                detections[device] = detection
                prepared_by_device[device] = prepared
                row["detected"][device] = detection is not None
                row["per_camera_ms"][device] = {
                    "prepare": preparation_elapsed * 1000.0,
                    "detect": detection_elapsed * 1000.0,
                }

            row["prepare_pair_ms"] = preparation_s * 1000.0
            row["detect_pair_ms"] = detection_s * 1000.0
            # Alternate the branches to avoid consistently charging cold caches to one.
            branches = (("all21", tuple(range(21))), ("arm7", ARM_LANDMARKS))
            if index % 2:
                branches = tuple(reversed(branches))
            for name, landmarks in branches:
                started = time.perf_counter()
                observations = []
                for context in contexts:
                    device = context["device"]
                    detection = detections[device]
                    if detection is None:
                        continue
                    filtered = SimpleNamespace(
                        points={lm: value for lm, value in detection.points.items()
                                if int(lm) in landmarks},
                        confidence=detection.confidence,
                        lm_score=getattr(detection, "lm_score", None),
                    )
                    observations.append(collect_row(
                        camera_id=device, frame_index=index,
                        host_timestamp_us=prepared_by_device[device].timestamp_us,
                        detection=filtered, depth_m=prepared_by_device[device].depth_m,
                        projector=context["projector"],
                        depth_radius=context["depth_radius_px"],
                    ))
                sampling_s = time.perf_counter() - started
                started = time.perf_counter()
                solved = 0
                quality = []
                for landmark in landmarks:
                    views = [{
                        "camera_id": observation["camera_id"],
                        "uv": observation["uv"][landmark].copy(),
                        "score": float(observation["lm_score"][landmark]),
                        "depth_m": float(observation["depth_m"][landmark]),
                        "depth_valid": bool(observation["depth_valid"][landmark]),
                        "depth_spread_m": float(observation["depth_spread_m"][landmark]),
                    } for observation in observations]
                    result = triangulate_joint(views, cameras, landmark, tri_config)
                    if result is not None:
                        solved += 1
                        quality.append(result["quality"])
                triangulation_s = time.perf_counter() - started
                row[name] = {
                    "sampling_pair_ms": sampling_s * 1000.0,
                    "triangulation_ms": triangulation_s * 1000.0,
                    "component_total_ms": (preparation_s + detection_s + sampling_s
                                           + triangulation_s) * 1000.0,
                    "solved_landmarks": solved,
                    "quality_mean": float(np.mean(quality)) if quality else 0.0,
                }
            if index < warmup:
                warmup_totals.append(row)
            else:
                measured.append(row)
            if (index + 1) % 10 == 0:
                print(f"processed {index + 1}/{warmup + frames} recorded groups", file=sys.stderr,
                      flush=True)
        run_s = time.perf_counter() - all_started
        for path in inputs:
            if _sha256(path) != hashes[str(path)]:
                raise RuntimeError(f"recorded calibration/metadata changed during benchmark: {path}")
        summary = {
            "prepare_pair": _summary([r["prepare_pair_ms"] / 1000.0 for r in measured], np),
            "detect_pair": _summary([r["detect_pair_ms"] / 1000.0 for r in measured], np),
        }
        for name, landmarks in (("all21", tuple(range(21))), ("arm7", ARM_LANDMARKS)):
            summary[name] = {
                "landmarks": list(landmarks),
                **{stage: _summary([r[name][stage + "_ms"] / 1000.0 for r in measured], np)
                   for stage in ("sampling_pair", "triangulation", "component_total")},
                "mean_solved_landmarks": float(np.mean([r[name]["solved_landmarks"]
                                                       for r in measured])),
                "frames_all_landmarks_solved": sum(r[name]["solved_landmarks"] == len(landmarks)
                                                   for r in measured),
            }
        versions = {"python": sys.version.split()[0], "platform": platform.platform(),
                    "machine": platform.machine(), "cpu_count": os.cpu_count()}
        for distribution in ("numpy", "opencv-python", "opencv-python-headless", "mediapipe",
                             "scipy", "threadpoolctl", "pin", "ur-rtde"):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        source_files = (
            "scripts/benchmark_live_perception.py", "viki/perception/observations.py",
            "viki/perception/triangulate.py", "viki/perception/backends/mediapipe.py",
            "viki/perception/profiles.py", "viki/perception/k4a_offline.py",
            "viki/perception/rs_offline.py",
        )
        return {
            "schema": 1, "episode": str(episode), "profile": profile.manifest(),
            "detector_mode": detector_mode, "triangulation_config": tri_config.as_dict(),
            "scope": "recorded RGB-D component benchmark; no hardware or ground truth",
            "excludes": ["RGB/depth decode and file reads", "sensor/USB capture and camera sync",
                         "causal temporal filters", "articulated hand fit", "IK", "networking",
                         "robot servo", "execution logging", "rendering"],
            "notes": [
                "Totals are measured sums per group, not sums of independent p95 quantiles.",
                "Both branches share the same two persistent detectors and detections.",
                "No RGB downscaling, identity projector, missing-frame skip or detection-success selection.",
                "Sampling uses cached SDK calibration at the existing 1 m projection depth estimate.",
                "Triangulation retains existing K/dist optics; it is not exact SDK ray triangulation.",
                "The arm7 subset is a component experiment, not a modified frozen perception profile.",
            ],
            "measured_groups": frames, "warmup_groups": warmup,
            "import_s": import_s, "setup_s": setup_s, "run_wall_s_including_io": run_s,
            "warmup_component_totals_s": {
                name: sum(row[name]["component_total_ms"] for row in warmup_totals) / 1000.0
                for name in ("all21", "arm7")
            },
            "cameras": [{key: c[key] for key in (
                "device", "requested", "video", "color_shape", "depth_shape", "depth_radius_px",
                "calibration_init_s", "detector_init_s",
            )} for c in contexts],
            "versions": versions, "input_sha256": hashes,
            "source_sha256": {name: _sha256(Path(name)) for name in source_files},
            "model_sha256": _sha256(Path("models/hand_landmarker.task")),
            "summary": summary, "warmup_rows": warmup_totals, "measured_rows": measured,
        }
    finally:
        for context in contexts:
            context["capture"].release()
            if context["detector"] is not None:
                context["detector"].close()
            close = getattr(context["projector"], "close", None)
            if close is not None:
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--detector-mode", choices=("video", "image"), default="video",
                        help="default VIDEO matches V4; IMAGE is a separate timing experiment")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup nonnegative")
    if args.out.resolve().is_relative_to(args.episode.resolve()):
        parser.error("--out must be outside the source episode")
    if not Path("models/hand_landmarker.task").is_file():
        parser.error("cached models/hand_landmarker.task required; benchmark will not download weights")
    result = benchmark(args.episode, frames=args.frames, warmup=args.warmup,
                       detector_mode=args.detector_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.out.parent, prefix=args.out.name + ".", delete=False) as temp:
        json.dump(result, temp, indent=2, allow_nan=False)
        temporary = Path(temp.name)
    temporary.chmod(0o644)
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
