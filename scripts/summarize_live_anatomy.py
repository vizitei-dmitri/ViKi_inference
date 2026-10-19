#!/usr/bin/env python3
"""Episode-balanced summaries; repeated inference passes are not independent trials."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def real_summary(root):
    summary = json.loads((root/"summary.json").read_text())
    roles = {e["id"]:e["role"] for e in summary["episodes"]}
    episodes = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    counts = defaultdict(lambda:defaultdict(int))
    strata = defaultdict(int)
    for run in summary["runs"]:
        if run["error"]:
            raise ValueError(run["error"])
        report = json.loads((root/run["report"]).read_text())
        for label,n in report["quality_stratum_counts"].items():
            strata[label] += n
        for v in report["variants"]:
            name = v["setting"]["name"]
            m = v["metrics"]
            for key, value in {
                "correction_mm": (m["common_target_correction_mm"] or {}).get("mean"),
                "reprojection_mrad": (m["common_reprojection_mrad"] or {}).get("mean"),
                "phalanx_cv_percent": m["common_phalanx_cv_percent"],
                "palm_cv_percent": m["common_palm_cv_percent"],
                "lag_proxy_ms": m["alignment_lag_proxy_ms"],
                "mean_run_p95_compute_ms": v["compute_ms"]["p95"],
            }.items():
                episodes[run["episode"]][name][key].append(value)
            for key in ("finite_frames","accepted_frames","common_eval_frames"):
                counts[name][key] += m[key]
            for key in ("optimizer_budget_frames","optimizer_converged_frames"):
                counts[name][key] += v[key]
    per_episode = {e:{n:{k:mean(vals) for k,vals in metrics.items()} for n,metrics in variants.items()}
                   for e,variants in episodes.items()}
    by_role = {}
    for role in ("train", "held_out", "all"):
        selected = [v for e,v in per_episode.items() if role == "all" or roles[e] == role]
        by_role[role] = {n:{k:mean([v[n][k] for v in selected]) for k in selected[0][n]}
                         for n in selected[0]}
    paired = {}
    for candidate, base in [("model_raw_q1","model_raw_uniform"), ("model_oe_q1","model_oe_uniform"),
                            ("oe_fast_q1","oe_fast")]:
        values = [v[candidate]["reprojection_mrad"]-v[base]["reprojection_mrad"]
                  for e,v in per_episode.items() if roles[e] == "held_out"]
        rng = random.Random(20261019)
        bootstrap = sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000))
        paired[f"{candidate}_minus_{base}"] = {"metric":"reprojection_mrad, consistency with same input",
            "episodes":len(values),"mean":mean(values),"episode_bootstrap_95_interval":[bootstrap[49],bootstrap[1949]]}
    return {"scope":"per-episode mean over three passes, then equal episode weighting; no hand ground truth",
            "runs":len(summary["runs"]),"quality_stratum_counts":dict(strata),
            "counts":dict(counts),"per_episode":per_episode,"by_role":by_role,"held_out_paired":paired}


def synthetic_summary(root):
    summary = json.loads((root/"summary.json").read_text())
    groups = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for run in summary["runs"]:
        for v in json.loads((root/run["report"]).read_text())["variants"]:
            for key in ("known_common_target_error_mm", "known_all_fitted_landmark_error_mm", "known_inferred_landmark_error_mm"):
                value = v["metrics"].get(key)
                groups[run["case"]][v["setting"]["name"]][key].append(value["rmse"] if value else None)
    return {"scope":"mean of three seed RMSEs; constructed quality, known model, not real cameras",
            "runs":len(summary["runs"]),
            "cases":{c:{n:{k:mean(a) for k,a in m.items()} for n,m in variants.items()} for c,variants in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real", type=Path)
    parser.add_argument("synthetic", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps({"real":real_summary(args.real),"synthetic":synthetic_summary(args.synthetic)},indent=2)+"\n")


if __name__ == "__main__":
    main()
