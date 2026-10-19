#!/usr/bin/env python3
"""Compare online IK objectives in MuJoCo on one frozen recorded target stream."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from viki.contracts import LiveIKTuning
from viki.replay import benchmark_live_ik


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new output directory outside source session")
    parser.add_argument("--gains", type=float, nargs="+", default=[6., 10., 15.])
    parser.add_argument("--regularization", action="store_true", help="also sweep damping and acceleration weights at each gain")
    parser.add_argument("--scenario", choices=["recorded", "step", "ramp"], default="recorded",
                        help="recorded targets, or synthetic +/-20mm X targets with the session's model and limits")
    args = parser.parse_args()
    base = LiveIKTuning()
    variants = []
    for gain in args.gains:
        variants.append((f"gain={gain:g}", replace(base, position_gain=gain)))
        if args.regularization:
            for damping in (1e-5, 1e-3):
                variants.append((f"gain={gain:g},damping={damping:g}",
                                 replace(base, position_gain=gain, damping=damping)))
            for weight in (0., 1e-4, 1e-2):
                variants.append((f"gain={gain:g},acceleration_weight={weight:g}",
                                 replace(base, position_gain=gain, acceleration_weight=weight)))
    result = benchmark_live_ik(args.session, args.out, variants, scenario=args.scenario)
    print(json.dumps({"output": str(args.out), "variants": [
        {key: row[key] for key in ("name", "passed", "samples", "tcp_error_mm", "error")}
        for row in result["variants"]]}, indent=2))
    return 0 if all(row["passed"] for row in result["variants"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
