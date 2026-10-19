#!/usr/bin/env python3
"""Capture causal pre-filter XYZ from recorded RGB-D without robot execution.

One raw geometry pass supplies every filter candidate. Playback uses real host
time, bounded latest frames and original source ages, exactly as live does.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from viki.cameras import RecordedFrameSource
from viki.perception import LiveHandEstimator


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    estimator = LiveHandEstimator.from_episode(args.reference, capture_raw=True,
                                               min_cutoff_hz=2., beta=.5)
    source = None
    count, valid = 0, 0
    try:
        source = RecordedFrameSource(args.reference)
        with (args.out/"hand.jsonl").open("w") as stream:
            while not source.exhausted:
                begin = time.monotonic()
                group = source.poll()
                if group is not None:
                    result = estimator.process(group)
                    if result is not None:
                        stream.write(json.dumps(safe(asdict(result)), allow_nan=False)+"\n")
                        count += 1
                        valid += int(result.position_valid)
                time.sleep(max(0., 1/15-(time.monotonic()-begin)))
        manifest = {"scope": "recorded RGB-D -> causal raw geometry and baseline filter; no arm executor",
            "reference": str(args.reference), "perception": estimator.manifest,
            "source": source.source_summary, "samples": count, "valid_positions": valid,
            "hand_sha256": hashlib.sha256((args.out/"hand.jsonl").read_bytes()).hexdigest()}
        (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"output": str(args.out), "samples": count, "valid_positions": valid}))
    finally:
        estimator.close()
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()
