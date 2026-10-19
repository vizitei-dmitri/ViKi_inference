#!/usr/bin/env python3
"""Compare causal One Euro settings on one frozen pre-filter XYZ stream.

Real captures have no hand ground truth. Synthetic noise/motion has known
positions and is reported separately; none of these are robot TCP errors.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from viki.perception import CausalLandmarkFilter


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {"mean": float(values.mean()), "rmse": float(np.sqrt(np.mean(values**2))),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def synthetic():
    t = np.arange(0., 12., 1/15)
    x = np.interp(t, [0, 2, 4, 5, 7, 9, 10, 12], [0, 0, .04, .04, .04, .04, 0, 0])
    fast = (t >= 5) & (t < 7)
    x[fast] += .06*np.sin(2*np.pi*(t[fast]-5))
    truth = np.tile(np.arange(21)[None, :, None]*np.array([.002, .003, .001]), (len(t), 1, 1))
    truth[:, :, 0] += x[:, None]
    rng = np.random.default_rng(20261019)
    raw = truth+rng.normal(0, .002, truth.shape)+rng.normal(0, .001, (len(t), 1, 3))
    valid = ~((t >= 10.) & (t < 10.4))
    observed = np.tile(valid[:, None], (1, 21))
    raw[~observed] = np.nan
    stationary = ((t > .5) & (t < 2)) | ((t > 4.5) & (t < 5)) | ((t > 7.5) & (t < 9)) | (t > 11)
    return np.rint((t+1)*1e6).astype(np.int64), raw, observed, valid, truth, stationary


def apply_filter(times, raw, observed, valid, cutoff, beta, quality=None, quality_power=0.):
    processor = CausalLandmarkFilter(cutoff, beta, quality_power=quality_power)
    result = np.full_like(raw, np.nan)
    for i, now in enumerate(times):
        if not valid[i]:
            processor.reset()
        else:
            result[i] = processor.update(raw[i], observed[i], int(now),
                quality=quality[i] if quality is not None else None)
    return result


def lag_proxy(times, raw_target, filtered_target, valid):
    """Offline diagnostic only: align with past raw estimates, never ground truth."""
    t = (times-times[0])/1e6
    indices = np.flatnonzero(valid)
    chunks = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    chunks = [c for c in chunks if len(c) >= 6]
    losses = []
    for tau in np.arange(0., .251, .005):
        residuals = []
        for c in chunks:
            eligible = c[t[c] >= t[c[0]]+.25]
            if not eligible.size:
                continue
            shifted = np.column_stack([np.interp(t[eligible]-tau, t[c], raw_target[c, axis]) for axis in range(3)])
            residuals.extend(np.sum((shifted-filtered_target[eligible])**2, axis=1))
        losses.append(np.mean(residuals) if residuals else np.inf)
    return float(np.argmin(losses)*5) if np.isfinite(losses).any() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quality", action="store_true", help="also test geometric-quality alpha weights on real captures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.capture is not None) == args.synthetic:
        parser.error("choose a capture directory or --synthetic")
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = truth = stationary = quality = None
    if args.synthetic:
        times, raw, observed, valid, truth, stationary = synthetic()
        source = {"kind": "synthetic", "seed": 20261019, "independent_noise_std_mm": 2., "shared_noise_std_mm": 1.}
    else:
        data = (args.capture/"hand.jsonl").read_bytes()
        rows = [json.loads(row) for row in data.decode().splitlines()]
        if not rows or any(row.get("raw_points") is None for row in rows):
            raise ValueError("capture must contain opt-in raw triangulated points")
        times = np.array([row["source_time_us"] for row in rows], dtype=np.int64)
        raw = np.array([row["raw_points"] for row in rows], dtype=float)
        observed = np.array([row["observed"] for row in rows], dtype=bool) & np.isfinite(raw).all(axis=2)
        valid = np.array([row["position_valid"] for row in rows], dtype=bool)
        baseline = np.array([row["points"] for row in rows], dtype=float)
        quality = np.array([row["quality"] for row in rows], dtype=float)
        manifest = json.loads((args.capture/"manifest.json").read_text())
        source = {"kind": "recorded_raw_geometry", "reference": manifest["reference"],
            "hand_sha256": hashlib.sha256(data).hexdigest(),
            "perception": manifest["perception"], "capture": str(args.capture)}
    if np.any(np.diff(times) <= 0):
        raise ValueError("source times must increase")
    np.savez_compressed(args.out/"inputs.npz", times=times, raw=raw, observed=observed, valid=valid,
                        **({"quality": quality} if quality is not None else {}),
                        **({"truth": truth, "stationary": stationary} if truth is not None else {}))
    raw_target = (raw[:, 4]+raw[:, 8])*.5
    report = {"scope": "human/rig-space causal smoothing; real-capture residuals and alignment are self-consistency measures",
              "source": source, "samples": len(times), "valid_samples": int(valid.sum()), "variants": []}
    settings = [(cutoff, beta, power) for cutoff in (.5, 1., 2., 4., 8.)
                for beta in (0., .5, 2., 10., 30.)
                for power in ((0., .5, 1.) if args.quality and quality is not None else (0.,))]
    for cutoff, beta, power in settings:
        result = apply_filter(times, raw, observed, valid, cutoff, beta, quality, power)
        target = (result[:, 4]+result[:, 8])*.5
        pairs = valid[1:] & valid[:-1]
        row = {"min_cutoff_hz": cutoff, "beta": beta, "quality_power": power,
            "target_correction_mm": stats(np.linalg.norm(target[valid]-raw_target[valid], axis=1)*1000),
            "output_step_mm": stats(np.linalg.norm(np.diff(target, axis=0)[pairs], axis=1)*1000),
            "alignment_lag_proxy_ms": lag_proxy(times, raw_target, target, valid)}
        if baseline is not None and cutoff == 2. and beta == .5 and power == 0.:
            row["baseline_reproduction_max_m"] = float(np.nanmax(np.abs(result[valid]-baseline[valid])))
            if row["baseline_reproduction_max_m"] > 1e-10:
                raise ValueError("baseline filter does not reproduce capture")
        if truth is not None:
            true_target = (truth[:, 4]+truth[:, 8])*.5
            error = np.linalg.norm(target-true_target, axis=1)*1000
            row["known_target_error_mm"] = stats(error[valid])
            row["known_stationary_error_mm"] = stats(error[valid & stationary])
            row["known_moving_error_mm"] = stats(error[valid & ~stationary])
        trace_name = f"variant_{len(report['variants']):03d}.npz"
        np.savez_compressed(args.out/trace_name, filtered_points=result, target_position=target)
        row["trace"] = trace_name
        report["variants"].append(row)
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.out), "samples": len(times), "variants": len(report["variants"])}))


if __name__ == "__main__":
    main()
