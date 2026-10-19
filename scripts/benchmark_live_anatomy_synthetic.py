#!/usr/bin/env python3
"""Known-pose stress cases; quality is constructed, not calibrated real visibility."""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from viki.perception import hand_model as hm
from scripts.benchmark_live_anatomy_dataset import run_capture


def known_hand():
    roots = {f: np.array([.025 if f == "thumb" else .06, y, 0.])
             for f, y in zip(hm.FINGERS, [-.04, -.03, 0., .025, .045])}
    angles = {f: np.array([0., 0., np.arctan2(p[1], p[0]) if f == "thumb" else 0.])
              for f, p in roots.items()}
    lengths = {f: np.array([.038, .03, .024] if f == "thumb" else [.04, .026, .02]) for f in roots}
    return hm.build(hm.HandParams(roots, angles, lengths,
                    {f: np.full(3, .008) for f in roots}))


def generate(case, seed, n=100):
    rng = np.random.default_rng(seed)
    hand = known_hand()
    truth, measured, qualities, masks = [], [], [], []
    for i in range(n):
        q = pin.neutral(hand.model)
        if i >= 10:
            t = (i-10)/15
            q[:3] = [.04*np.sin(t*2), .02*np.sin(t), .01*np.sin(t*3)]
            for finger in hm.FINGERS:
                for joint, scale in zip(hm._JOINTS[finger][1:], [.6,.8,.45]):
                    q[hand.model.joints[hand.model.getJointId(joint)].idx_q] = scale*(1+np.sin(t*2))*.5
        points = hm.fk_landmark_positions(hand,q,list(range(21)))
        # A deliberately unmodelled dynamic palm change tests model bias.
        if case == "palm_mismatch" and i >= 10:
            points[13:17,1] += .008*np.sin((i-10)/15)
        quality = np.full(21, .9)
        sigma = np.full(21, .002)
        observed = np.ones(21,bool)
        if i >= 10 and case in ("informative_quality", "shuffled_quality"):
            bad = rng.random(21) < .25
            sigma[bad] = .015
            quality[bad] = .15
            if case == "shuffled_quality":
                quality = rng.permutation(quality)
        noise = rng.normal(size=(21,3))*sigma[:,None]
        # Same, clean causal calibration prefix for all cases and seeds.
        if i < 10:
            noise[:] = 0
        sample = points+noise
        if case == "partial_visibility" and 30 <= i < 65:
            observed[13:17] = False
            sample[~observed] = np.nan
            quality[~observed] = 0
        truth.append(points); measured.append(sample); qualities.append(quality); masks.append(observed)
    return np.array(truth), np.array(measured), np.array(qualities), np.array(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"scope": "constructed quality, known synthetic hand; not evidence of real visibility calibration", "runs": []}
    for case in ("isotropic", "informative_quality", "shuffled_quality", "partial_visibility", "palm_mismatch"):
        for seed in range(3):
            directory = args.out / f"{case}_{seed}" / "input"
            directory.mkdir(parents=True)
            truth, points, quality, observed = generate(case, seed)
            rows = [dict(raw_points=p.tolist(), observed=m.tolist(),quality=c.tolist(),
                         source_time_us=1_000_000+round(i*1e6/15),position_valid=True)
                    for i,(p,c,m) in enumerate(zip(points,quality,observed))]
            (directory/"hand.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            np.savez_compressed(directory/"truth.npz",points=truth)
            report = run_capture(directory,directory.parent/"comparison")
            row = {"case":case,"seed":seed,"report":str((directory.parent/"comparison/report.json").relative_to(args.out)),
                   "common_frames":report["common_frames"]}
            summary["runs"].append(row)
            (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
