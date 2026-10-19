#!/usr/bin/env python3
"""Emit original Hybrid outputs for a frozen ViKi hand log in either image.

This script deliberately has no ViKi/ROS imports, so it also runs in the
original dg5f-camera-demo image. Compare emitted arrays across environments.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hand_log",type=Path)
    parser.add_argument("--hybrid-root",type=Path,default=Path("/opt/dg5f-hybrid"))
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.hybrid_root/"src"))
    from dg5f_demo.retarget import HybridRetargeter
    engine = HybridRetargeter(low_pass_alpha=1.)
    results = []
    for row in map(json.loads,args.hand_log.read_text().splitlines()):
        points = np.array(row["points"],float)
        if not all(row["observed"]) or not np.isfinite(points).all() or not row["position_valid"]:
            continue
        q,w,d = engine.retarget(points)
        results.append(dict(sequence=row["sequence"],q_deg=q.tolist(),weights=w.tolist(),distances_m=d.tolist()))
    args.out.write_text(json.dumps({"versions":{p:importlib.metadata.version(p) for p in ("pin","dex-retargeting","torch","numpy")},
        "frames":results},indent=2)+"\n")
    print(json.dumps({"frames":len(results),"out":str(args.out)}))


if __name__ == "__main__":
    main()
