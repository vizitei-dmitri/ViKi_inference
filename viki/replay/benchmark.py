"""Deterministic, simulation-only comparisons on frozen live executor targets.

No camera, RTDE owner or live worker is constructed. Virtual time advances at
the configured servo period; compute timings are not real-time guarantees.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from viki.contracts import LiveControlOptions, LiveIKTuning
from viki.retarget import OnlineRetargeter
from .live import _MujocoBackend


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()),
            "rmse": float(np.sqrt(np.mean(values**2)))}


def _load_targets(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest["config"]
    if cfg["backend"] != "mujoco" or manifest.get("hardware"):
        raise ValueError("benchmark accepts MuJoCo session artifacts only")
    options = LiveControlOptions(**cfg["options"])
    options.validate()
    if options.hardware_enabled or options.host or options.orientation:
        raise ValueError("benchmark requires simulation-only position following")
    paths = sorted((directory / "executor").glob("*/execution.h5"))
    if len(paths) != 1:
        raise ValueError("expected exactly one execution.h5")
    path = paths[0]
    with h5py.File(path, "r") as file:
        if not file.attrs.get("closed_cleanly"):
            raise ValueError("execution log must be closed cleanly")
        s = file["samples"]
        active = (s["state"].asstr()[:] == "active") & s["command_accepted"][:].astype(bool)
        indices = np.flatnonzero(active)
        if len(indices) < 2:
            raise ValueError("need at least two active accepted samples")
        # Compare the first continuous binding only; never bridge a lost input
        # or silently substitute a later, more successful binding.
        gaps = np.flatnonzero(np.diff(indices) != 1)
        if gaps.size:
            indices = indices[:gaps[0]+1]
        if len(indices) < 2:
            raise ValueError("first active segment is too short")
        times = s["host_monotonic_us"][:][indices].astype(np.int64)
        sources = s["source_time_us"][:][indices].astype(np.int64)
        positions = s["target_position"][:][indices]
        initial_q = s["q_observed"][int(indices[0])]
        initial_qd = s["qd_observed"][int(indices[0])]
        if (not np.isfinite([initial_q, initial_qd]).all() or
                np.max(np.abs(initial_qd)) > .01):
            raise ValueError("first binding must start from a stationary measured arm")
        if not np.isfinite(positions).all() or np.any(np.diff(times) <= 0):
            raise ValueError("invalid positions or non-increasing availability times")
        if np.any(sources > times) or np.any(np.diff(sources) < 0):
            raise ValueError("invalid source chronology")
        controller = json.loads((directory / "session.json").read_text())["summary"]["controller"]
        expected_model = dict(controller["kinematic_model"])
        expected_model["simulation_identity"] = controller.get("identity", {})
    options = replace(options, initial_q=tuple(initial_q))
    period_us = int(1e6 / options.frequency)
    ticks = np.arange(times[0], times[-1]+1, period_us, dtype=np.int64)
    # Hold only values already consumed by the original executor. No temporal
    # interpolation, future target or refreshed source timestamp is introduced.
    selected = np.searchsorted(times, ticks, side="right") - 1
    if np.any(ticks - sources[selected] >= options.input_ttl_s * 1e6):
        raise ValueError("frozen input becomes stale on the nominal servo grid")
    frozen = {"host_monotonic_us": ticks, "source_time_us": sources[selected],
              "input_available_us": times[selected], "target_position": positions[selected],
              "original_row": indices[selected], "initial_q": initial_q}
    provenance = {"session": directory.name, "reference": cfg["reference"],
        "recorded_speed": cfg.get("recorded_speed"), "source_log": str(path),
        "source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "original_active_samples": int(active.sum()), "first_segment_samples": len(indices),
        "resampled_samples": len(ticks), "frequency_hz": options.frequency,
        "nominal_duration_s": float((ticks[-1]-ticks[0])/1e6),
        "selection": "first continuous active accepted segment; causal zero-order hold at original availability times"}
    return options, frozen, provenance, expected_model


def benchmark_live_ik(directory, out_dir, variants, *, scenario="recorded"):
    """Compare named LiveIKTuning values against identical recorded targets.

    All variants retain the source session's physical/model constraints. A
    rejected variant keeps its partial trace and is never ranked as successful.
    Defaults are not changed by running this function.
    """
    variants = list(variants)
    if not variants or len({name for name, _ in variants}) != len(variants):
        raise ValueError("need nonempty uniquely named tuning variants")
    for name, tuning in variants:
        if not isinstance(name, str) or not name:
            raise ValueError("variant name must be a nonempty string")
        tuning.validate()
    options, frozen, provenance, expected_model = _load_targets(directory)
    if scenario not in ("recorded", "step", "ramp"):
        raise ValueError("scenario must be recorded, step or ramp")
    if scenario != "recorded":
        ik = OnlineRetargeter(options, tuning=LiveIKTuning())
        p0, _ = ik.pose(options.initial_q)
        seconds = np.arange(0., 7., 1/options.frequency)
        if scenario == "step":
            offset = np.select([seconds < .5, seconds < 2.5, seconds < 4.5],
                               [0., .02, -.02], default=0.)
            stopped = np.logical_or.reduce([(seconds >= 1.5) & (seconds < 2.5),
                (seconds >= 3.5) & (seconds < 4.5), seconds >= 5.5])
        else:
            offset = np.interp(seconds, [0., 1., 2., 4., 5., 6., 7.],
                               [0., .02, .02, -.02, -.02, 0., 0.])
            stopped = np.logical_or.reduce([(seconds >= 1.5) & (seconds < 2.),
                (seconds >= 4.5) & (seconds < 5.), seconds >= 6.5])
        positions = np.tile(p0, (len(seconds), 1))
        positions[:, 0] += offset
        ticks = np.rint(seconds*1e6).astype(np.int64)
        frozen = {"host_monotonic_us": ticks, "input_available_us": ticks.copy(),
            "source_time_us": ticks.copy(), "target_position": positions,
            "original_row": np.full(len(ticks), -1), "initial_q": np.asarray(options.initial_q),
            "stationary_evaluation": stopped}
        provenance = {"configuration_source": provenance, "scenario": scenario,
            "target_source": "synthetic robot-base X displacement +/-20mm with fresh samples at every servo tick; no perception",
            "frequency_hz": options.frequency, "nominal_duration_s": float(seconds[-1]),
            "resampled_samples": len(ticks),
            "stationary_evaluation": "holding windows start 1s after steps or 0.5s after ramps"}
    out = Path(out_dir)
    if out.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("benchmark output must be outside the source session")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "inputs.npz", **frozen)
    report = {"schema_version": 1, "hardware": False,
        "scope": "nominal-time MuJoCo TCP tracking; no hand ground truth or hardware timing claim",
        "scenario": scenario,
        "provenance": provenance, "options": asdict(options),
        "inputs_sha256": hashlib.sha256((out / "inputs.npz").read_bytes()).hexdigest(),
        "variants": []}
    for index, (name, tuning) in enumerate(variants):
        result = _simulate(options, frozen, tuning, expected_model)
        trace = result.pop("trace")
        trace_path = out / f"variant_{index:03d}.npz"
        np.savez_compressed(trace_path, **trace)
        result.update(name=name, tuning=asdict(tuning), trace=trace_path.name)
        report["variants"].append(result)
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def _simulate(options, inputs, tuning, expected_model, *, backend_factory=_MujocoBackend,
              retargeter_factory=OnlineRetargeter):
    backend = backend_factory(options)
    rows, timings, margin_rows = [], [], []
    error = None
    try:
        expected_identity = expected_model.get("simulation_identity", {})
        for key in ("model_xml_sha256", "model_source_commit", "version", "gravity_compensation", "frame"):
            if key in expected_identity and backend.identity.get(key) != expected_identity[key]:
                raise ValueError(f"MuJoCo {key} differs from the recorded session")
        ik = retargeter_factory(options, tuning=tuning)
        if ik.manifest["urdf_sha256"] != expected_model["urdf_sha256"]:
            raise ValueError("URDF hash differs from the recorded session")
        initial = backend.read()
        command, velocity = initial["q"].copy(), np.zeros(6)
        _, fixed_rotation = ik.pose(initial["q"])
        period = 1 / options.frequency
        # Initialize native/model caches outside component timing, as live does.
        ik.step(command, command, velocity, ik.pose(command)[0], fixed_rotation, period)
        for target in inputs["target_position"]:
            measured = backend.read()
            position, _ = ik.pose(measured["q"])
            started = time.perf_counter()
            step = ik.step(measured["q"], command, velocity, target, fixed_rotation, period)
            timings.append((time.perf_counter()-started)*1000)
            backend.command(step["q"])
            rows.append((measured["q"].copy(), measured["qd"].copy(), step["q"].copy(),
                         step["velocity"].copy(), position.copy(), target.copy()))
            margin_rows.append(step["margins"])
            command, velocity = step["q"], step["velocity"]
            backend.advance(period)
    except (ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()
    keys = ("q_observed", "qd_observed", "q_commanded", "velocity_commanded", "tcp_position", "target_position")
    trace = {key: np.array([row[i] for row in rows]).reshape((-1, 6 if i < 4 else 3))
             for i, key in enumerate(keys)}
    qd = trace["velocity_commanded"]
    acceleration = np.diff(np.vstack([np.zeros((1, 6)), qd]), axis=0) * options.frequency
    error_mm = np.linalg.norm(trace["tcp_position"]-trace["target_position"], axis=1)*1000
    stopped = inputs.get("stationary_evaluation", np.zeros(len(inputs["target_position"]), dtype=bool))[:len(rows)]
    stop_speed = np.linalg.norm(np.diff(trace["tcp_position"], axis=0), axis=1)*options.frequency*1000
    stop_pairs = stopped[1:] & stopped[:-1]
    margins = {key: min(row[key] for row in margin_rows)
               for key in ("joint", "floor", "collision") if margin_rows}
    peak_v = float(np.max(np.abs(qd))) if qd.size else 0.
    peak_a = float(np.max(np.abs(acceleration))) if acceleration.size else 0.
    observed_v = float(np.max(np.abs(trace["qd_observed"]))) if rows else 0.
    checks = {"all_samples_completed": len(rows) == len(inputs["target_position"]) and error is None,
              "velocity_bound": peak_v <= options.max_velocity+1e-6,
              "acceleration_bound": peak_a <= options.max_acceleration+1e-5,
              "observed_velocity_bound": observed_v <= options.max_velocity+.025,
              "sampled_model_margins": bool(margins) and min(margins.values()) >= -1e-8}
    return {"completed": checks["all_samples_completed"], "error": error,
        "samples": len(rows), "tcp_error_mm": _stats(error_mm),
        "solver_ms": _stats(timings), "peak_velocity_rad_s": peak_v,
        "peak_observed_velocity_rad_s": observed_v,
        "stationary_tcp_error_mm": _stats(error_mm[stopped]),
        "stationary_tcp_speed_mm_s": _stats(stop_speed[stop_pairs]),
        "peak_acceleration_rad_s2": peak_a,
        "velocity_bound_fraction": float(np.mean(np.any(np.abs(qd) >= options.max_velocity-1e-6, axis=1))) if qd.size else None,
        "acceleration_bound_fraction": float(np.mean(np.any(np.abs(acceleration) >= options.max_acceleration-1e-6, axis=1))) if acceleration.size else None,
        "minimum_model_margins": margins, "checks": checks, "passed": all(checks.values()),
        "kinematic_model": ik.manifest if 'ik' in locals() else None,
        "simulation": backend.identity, "trace": trace}
