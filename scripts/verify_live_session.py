#!/usr/bin/env python3
"""Inspect durable live artifacts without opening cameras or robot interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def verify(directory: Path, require_motion: bool = False) -> dict:
    report = json.loads((directory / "session.json").read_text())
    summary, cfg = report["summary"], report["config"]
    controls = cfg["options"]
    executor = summary.get("controller", report.get("executor", {}))
    log = Path(executor["execution_log"])
    if not log.is_file():
        candidates = list((directory / "executor").glob("*/execution.h5"))
        if len(candidates) != 1:
            raise ValueError("expected exactly one execution log")
        log = candidates[0]
    observations = [json.loads(row) for row in (directory / "hand.jsonl").read_text().splitlines()]
    dt = 1 / controls["frequency"]
    with h5py.File(log, "r") as file:
        samples = file["samples"]
        q, velocity = samples["q_commanded"][:], samples["velocity_commanded"][:]
        accepted = samples["command_accepted"][:].astype(bool)
        state = samples["state"].asstr()[:]
        host = samples["host_monotonic_us"][:]
        source = samples["source_time_us"][:]
        cycle_ms = samples["cycle_compute_ms"][:]
        solver_ms = samples["solver_ms"][:]
        contiguous = accepted[1:] & accepted[:-1]
        acceleration = np.diff(velocity, axis=0)[contiguous] / dt
        integration = np.diff(q, axis=0)[contiguous] - velocity[1:][contiguous] * dt
        peak_velocity = float(np.max(np.abs(velocity[accepted]))) if accepted.any() else 0.
        peak_acceleration = float(np.max(np.abs(acceleration))) if acceleration.size else 0.
        max_integration_error = float(np.max(np.abs(integration))) if integration.size else 0.
        excursion = float(np.max(np.ptp(q[accepted], axis=0))) if accepted.any() else 0.
        margins = [json.loads(value) for value in samples["margins_json"].asstr()[:]]
        minima = {key: min(float(row[key]) for row in margins if key in row)
                  for key in ("joint", "floor", "collision") if any(key in row for row in margins)}
        active = state == "active"
        ages = (host[active] - source[active]) / 1000
        checks = {
            "completed_without_fault": summary["phase"] == "stopped" and executor.get("state") == "stopped",
            "owners_stopped": not summary["running"] and not executor.get("running"),
            "log_closed": bool(file.attrs.get("closed_cleanly")),
            "log_counts_match": len(q) == executor["log_rows"] and len(file["events_json"]) == executor["log_events"],
            "no_logging_error": not executor.get("log_error") and not summary.get("input_log_error"),
            "hand_log_count": len(observations) + summary.get("input_log_dropped", 0) == summary["estimates"],
            "finite_commands": bool(np.isfinite(q).all() and np.isfinite(velocity).all()),
            "velocity_bound": peak_velocity <= controls["max_velocity"] + 1e-6,
            "acceleration_bound": peak_acceleration <= controls["max_acceleration"] + 1e-5,
            "command_integration": max_integration_error <= 1e-8,
            "command_compute_budget": bool((cycle_ms[accepted] <= dt * 1000).all()),
            "active_input_fresh_at_cycle_check": bool((ages < controls["input_ttl_s"] * 1000).all()),
            "model_margins_nonnegative": bool(all(value >= -1e-8 for value in minima.values())),
        }
        if require_motion:
            checks["motion_observed"] = bool(excursion > 1e-5 and np.count_nonzero(active) > 1)
        result = {"scope": "artifact consistency, discrete command bounds and sampled model checks; no physical accuracy claim",
            "directory": str(directory), "hardware": cfg["backend"] == "ur10e",
            "frequency_hz": controls["frequency"], "samples": len(q),
            "host_period_ms": {"median": float(np.median(np.diff(host)) / 1000) if len(host) > 1 else None,
                               "p95": float(np.percentile(np.diff(host), 95) / 1000) if len(host) > 1 else None},
            "accepted_commands": int(accepted.sum()), "hand_estimates": len(observations),
            "active_samples": int(active.sum()), "peak_command_velocity_rad_s": peak_velocity,
            "peak_command_acceleration_rad_s2": peak_acceleration,
            "joint_excursion_rad": excursion, "max_integration_error_rad": max_integration_error,
            "solver_ms": {"median": float(np.median(solver_ms[accepted])) if accepted.any() else None,
                          "p95": float(np.percentile(solver_ms[accepted], 95)) if accepted.any() else None},
            "active_source_age_max_ms": float(ages.max()) if ages.size else None,
            "model_margin_minima": minima, "checks": checks, "passed": all(checks.values())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-motion", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.require_motion)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(payload + "\n")
    print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
