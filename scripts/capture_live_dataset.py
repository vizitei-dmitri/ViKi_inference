#!/usr/bin/env python3
"""Predeclare the complete available episode set and repeat causal geometry passes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-episodes", nargs="+", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    episodes = sorted(p.parent.parent for p in args.dataset.glob("*/raw/timestamps.json"))
    if not episodes or not set(args.train_episodes) < {p.name for p in episodes}:
        parser.error("need a nonempty training subset and at least one held-out episode")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {"scope": "same rig/participant dataset; repeated inference passes are not independent human trials",
        "repeats": args.repeats, "perception_hz": 15, "recorded_speed": 1,
        "episodes": [], "runs": []}
    for ep in episodes:
        ticks = json.loads((ep/"raw/timestamps.json").read_text())
        protocol["episodes"].append({"id": ep.name, "role": "train" if ep.name in args.train_episodes else "held_out",
            "frames": len(ticks), "duration_s": (ticks[-1]["sync_us"]-ticks[0]["sync_us"])/1e6,
            "task": json.loads((ep/"meta.json").read_text()).get("task"),
            "timestamps_sha256": hashlib.sha256((ep/"raw/timestamps.json").read_bytes()).hexdigest()})
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    for repeat in range(args.repeats):
        for ep in episodes:
            name = f"repeat_{repeat+1:02d}/{ep.name}"
            directory = args.out/name
            log = args.out/f"{repeat+1:02d}-{ep.name}.log"
            with log.open("w") as stream:
                result = subprocess.run([sys.executable, "scripts/capture_live_filter_inputs.py",
                    str(ep), "--out", str(directory)], stdout=stream, stderr=subprocess.STDOUT)
            row = {"episode": ep.name, "repeat": repeat+1, "directory": name, "returncode": result.returncode}
            if (directory/"manifest.json").is_file():
                manifest = json.loads((directory/"manifest.json").read_text())
                row.update(samples=manifest["samples"], valid_positions=manifest["valid_positions"],
                           hand_sha256=manifest["hand_sha256"], source_exhausted=manifest["source_exhausted"])
            protocol["runs"].append(row)
            (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
            print(json.dumps(row),flush=True)
    return int(any(row["returncode"] for row in protocol["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
