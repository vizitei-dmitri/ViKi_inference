#!/usr/bin/env python3
"""Causal fixed-bone projection prototype; not a full anatomical hand fit.

Only directly observed endpoints are used. First ten complete valid frames
calibrate lengths; a 15mm correction cap rejects a frame. No missing finger is
invented. Fixed-length variation is self-consistency, not hand accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from viki.perception import CausalLandmarkFilter


EDGES = [edge for root in (1, 5, 9, 13, 17)
         for edge in [(0, root), (root, root+1), (root+1, root+2), (root+2, root+3)]]


def project_stream(raw, filtered, observed, valid, *, warmup=10, cap=.015):
    output = np.full_like(filtered, np.nan)
    accepted = np.zeros(len(raw), dtype=bool)
    ready = np.zeros(len(raw), dtype=bool)
    calibration = []
    reference = None
    rejected = 0
    for i in range(len(raw)):
        if not valid[i] or not observed[i].all():
            continue
        if reference is None:
            lengths = [np.linalg.norm(raw[i, b]-raw[i, a]) for a, b in EDGES]
            if np.min(lengths) <= .002:
                continue
            calibration.append(lengths)
            if len(calibration) < warmup:
                continue
            reference = np.median(calibration, axis=0)
        ready[i] = True
        points = filtered[i].copy()
        for (a, b), length in zip(EDGES, reference):
            direction = points[b]-points[a]
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                break
            points[b] = points[a]+length*direction/norm
        else:
            if np.max(np.linalg.norm(points-filtered[i], axis=1)) <= cap:
                output[i], accepted[i] = points, True
                continue
        rejected += 1
    return output, accepted, ready, rejected


def cv(points):
    if len(points) < 2:
        return None
    lengths = np.column_stack([np.linalg.norm(points[:, b]-points[:, a], axis=1) for a, b in EDGES])
    return float(np.mean(np.std(lengths, axis=0)/np.maximum(np.mean(lengths, axis=0), 1e-9))*100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = (args.capture/"hand.jsonl").read_bytes()
    rows = [json.loads(row) for row in data.decode().splitlines()]
    if not rows or any(row.get("raw_points") is None for row in rows):
        raise ValueError("raw geometry capture required")
    raw = np.array([row["raw_points"] for row in rows], dtype=float)
    observed = np.array([row["observed"] for row in rows], dtype=bool)
    valid = np.array([row["position_valid"] for row in rows], dtype=bool)
    report = {"scope": "fixed-length causal projection prototype; no joint-angle constraints, palm rigidity or hand ground truth",
        "hand_sha256": hashlib.sha256(data).hexdigest(), "samples": len(rows), "variants": []}
    for beta in (.5, 30.):
        processor = CausalLandmarkFilter(2., beta)
        filtered = np.full_like(raw, np.nan)
        for i, row in enumerate(rows):
            if not valid[i]:
                processor.reset()
            else:
                filtered[i] = processor.update(raw[i], observed[i], row["source_time_us"])
        projected, accepted, ready, rejected = project_stream(raw, filtered, observed, valid)
        before_target = (filtered[accepted, 4]+filtered[accepted, 8])*.5
        after_target = (projected[accepted, 4]+projected[accepted, 8])*.5
        shifts = np.linalg.norm(after_target-before_target, axis=1)*1000
        report["variants"].append({"min_cutoff_hz": 2., "beta": beta,
            "ready_frames": int(ready.sum()), "accepted_frames": int(accepted.sum()),
            "rejected_by_correction_cap": rejected,
            "common_window_bone_cv_before_percent": cv(filtered[accepted]),
            "common_window_bone_cv_after_percent": cv(projected[accepted]),
            "target_shift_mean_mm": float(shifts.mean()) if shifts.size else None,
            "target_shift_max_mm": float(shifts.max()) if shifts.size else None})
        np.savez_compressed(args.out/f"beta_{beta:g}.npz", filtered=filtered, projected=projected,
                            accepted=accepted, ready=ready)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
