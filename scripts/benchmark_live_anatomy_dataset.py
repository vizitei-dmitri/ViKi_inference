#!/usr/bin/env python3
"""Paired whole-dataset causal filter / articulated-model comparison.

Real recordings have no hand ground truth. Reprojection, model consistency,
lag and repeatability are separate diagnostics; none proves absolute accuracy.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from viki.contracts import LiveHandFitOptions
from viki.perception import CausalHandFitter, CausalLandmarkFilter
from scripts.benchmark_live_filters import lag_proxy
from scripts.benchmark_live_bones import project_stream


SETTINGS = [
    {"name":"raw"},
    {"name":"oe_legacy", "beta":.5, "filter_quality":0.},
    {"name":"oe_fast", "beta":30., "filter_quality":0.},
    {"name":"oe_fast_q05", "beta":30., "filter_quality":.5},
    {"name":"oe_fast_q1", "beta":30., "filter_quality":1.},
    {"name":"old_bones", "beta":30., "filter_quality":0., "bones":True},
    {"name":"model_raw_uniform", "model_quality":0.},
    {"name":"model_raw_q05", "model_quality":.5},
    {"name":"model_raw_q1", "model_quality":1.},
    {"name":"model_oe_uniform", "beta":30., "filter_quality":0., "model_quality":0.},
    {"name":"model_oe_q1", "beta":30., "filter_quality":0., "model_quality":1.},
]
EDGES = [(a,b) for root in (1,5,9,13,17)
         for a,b in [(root,root+1),(root+1,root+2),(root+2,root+3)]]
PALM_EDGES = [(a,b) for i,a in enumerate([0,1,5,9,13,17]) for b in [0,1,5,9,13,17][i+1:]]


def stats(values):
    x = np.asarray(values,float).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return {"n":int(x.size),"mean":float(x.mean()),"rmse":float(np.sqrt(np.mean(x*x))),
            "p95":float(np.percentile(x,95)),"max":float(x.max())}


def length_cv(points, mask, edges):
    selected = points[mask]
    if len(selected)<2:
        return None
    values = np.column_stack([np.linalg.norm(selected[:,b]-selected[:,a],axis=1) for a,b in edges])
    return float(np.mean(np.std(values,axis=0)/np.maximum(np.mean(values,axis=0),1e-9))*100)


def execute(raw, observed, quality, valid, times, setting):
    filtered = raw.copy()
    compute = np.zeros(len(raw))
    if "beta" in setting:
        processor = CausalLandmarkFilter(2.,setting["beta"],quality_power=setting["filter_quality"])
        filtered[:] = np.nan
        for i, now in enumerate(times):
            start=time.perf_counter()
            if valid[i]:
                filtered[i] = processor.update(raw[i],observed[i],int(now),quality=quality[i])
            else:
                processor.reset()
            compute[i]=(time.perf_counter()-start)*1000
    statuses=Counter()
    model_points = filtered.copy()
    accepted=valid.copy()
    inferred=np.zeros_like(observed)
    nfev=np.zeros(len(raw),int)
    margins=np.full(len(raw),np.nan)
    converged=np.zeros(len(raw),bool)
    manifest=None
    if "model_quality" in setting:
        fitter=CausalHandFitter(LiveHandFitOptions(quality_power=setting["model_quality"]))
        model_points[:]=np.nan
        accepted[:]=False
        for i,now in enumerate(times):
            estimate=fitter.update(filtered[i],observed[i],quality[i],int(now),valid=bool(valid[i]))
            model_points[i],accepted[i],inferred[i]=estimate.points,estimate.accepted,estimate.inferred
            compute[i]+=estimate.compute_ms
            nfev[i],margins[i]=estimate.nfev,estimate.joint_margin_rad
            converged[i]=estimate.converged
            statuses[estimate.reason or "accepted"]+=1
        manifest=fitter.manifest
    elif setting.get("bones"):
        model_points,accepted,ready,rejected=project_stream(raw,filtered,observed,valid)
        statuses.update(warmup_or_incomplete=int(np.count_nonzero(valid & ~ready)),
                        correction_cap=int(rejected),accepted=int(accepted.sum()))
    model_points[~valid]=np.nan
    return {"points":model_points,"accepted":accepted,"inferred":inferred,"compute_ms":compute,
            "nfev":nfev,"converged":converged,"joint_margin_rad":margins},dict(statuses),manifest


def reprojection(points, observed, rows, mask):
    errors=[]
    for i in np.flatnonzero(mask):
        for camera in (rows[i].get("camera_observations") or {}).values():
            P=np.asarray(camera["P"],float)
            for key,uv in camera["uv"].items():
                lm=int(key)
                if not observed[i,lm] or not np.isfinite(points[i,lm]).all():
                    continue
                projected=P @ np.r_[points[i,lm],1.]
                if projected[2]<=0:
                    continue
                errors.append(np.linalg.norm(projected[:2]/projected[2]-uv)/camera["focal_px"]*1000)
    return stats(errors)


def metrics(points,raw,observed,quality,times,valid,accepted,common,rows,truth=None):
    finite=np.isfinite(points).all(axis=(1,2)) & valid
    common = common & finite
    target=(points[:,4]+points[:,8])*.5
    reference=(raw[:,4]+raw[:,8])*.5
    obs_mask=observed & finite[:,None]
    corrections=np.linalg.norm(points-raw,axis=2)*1000
    pairs=finite[1:] & finite[:-1] & ((np.diff(times)/1e6)<.25)
    out={"common_eval_frames":int(common.sum()),"finite_frames":int(finite.sum()),"accepted_frames":int(accepted.sum()),
        "all_fitted_target_correction_mm":stats(np.linalg.norm(target[finite]-reference[finite],axis=1)*1000),
        "common_target_correction_mm":stats(np.linalg.norm(target[common]-reference[common],axis=1)*1000),
        "common_reprojection_mrad":reprojection(points,observed,rows,common),
        "common_phalanx_cv_percent":length_cv(points,common,EDGES),
        "common_palm_cv_percent":length_cv(points,common,PALM_EDGES),
        "output_step_mm":stats(np.linalg.norm(np.diff(target,axis=0)[pairs],axis=1)*1000),
        "alignment_lag_proxy_ms":lag_proxy(times,reference,target,finite),
        "quality_strata_correction_mm":{}}
    for label,lo,hi in [("low",.1,.5),("middle",.5,.8),("high",.8,1.000001)]:
        m=obs_mask & (quality>=lo) & (quality<hi)
        out["quality_strata_correction_mm"][label]=stats(corrections[m])
    if truth is not None:
        known_target=(truth[:,4]+truth[:,8])*.5
        out["known_common_target_error_mm"]=stats(np.linalg.norm(target[common]-known_target[common],axis=1)*1000)
        out["known_common_landmark_error_mm"]=stats(np.linalg.norm(points[common]-truth[common],axis=2).reshape(-1)*1000)
        out["known_all_fitted_target_error_mm"]=stats(np.linalg.norm(target[finite]-known_target[finite],axis=1)*1000)
        out["known_all_fitted_landmark_error_mm"]=stats(np.linalg.norm(points[finite]-truth[finite],axis=2).reshape(-1)*1000)
        missing = (~observed) & np.isfinite(points).all(axis=2) & valid[:,None]
        out["known_inferred_landmark_error_mm"]=stats(np.linalg.norm(points-truth,axis=2)[missing]*1000)
    return out


def run_capture(directory,out,settings=SETTINGS):
    directory,out=Path(directory),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    data=(directory/"hand.jsonl").read_bytes()
    rows=[json.loads(row) for row in data.decode().splitlines()]
    raw=np.array([row["raw_points"] for row in rows],float)
    observed=np.array([row["observed"] for row in rows],bool) & np.isfinite(raw).all(axis=2)
    quality=np.array([row["quality"] for row in rows],float)
    times=np.array([row["source_time_us"] for row in rows],np.int64)
    valid=np.array([row["position_valid"] for row in rows],bool)
    truth=np.load(directory/"truth.npz")["points"] if (directory/"truth.npz").exists() else None
    traces=[]
    reports=[]
    for i,setting in enumerate(settings):
        trace,status,model=execute(raw,observed,quality,valid,times,setting)
        traces.append(trace)
        np.savez_compressed(out/f"variant_{i:02d}.npz",**trace)
        reports.append({"setting":setting,"status_counts":status,
            "compute_ms":stats(trace["compute_ms"][valid]),
            "optimizer_budget_frames":int(np.count_nonzero(trace["nfev"]>=20)),
            "optimizer_converged_frames":int(trace["converged"].sum()),
            "joint_margin_min_rad":float(np.nanmin(trace["joint_margin_rad"])) if np.isfinite(trace["joint_margin_rad"]).any() else None,
            "model":model,"trace":f"variant_{i:02d}.npz"})
    # Include rejected-but-finite fits to avoid evaluating only convenient frames.
    common=valid.copy() & observed.all(axis=1)
    for setting, trace in zip(settings, traces):
        if "model_quality" in setting:
            common &= np.isfinite(trace["points"]).all(axis=(1,2))
    for trace,report in zip(traces,reports):
        report["metrics"]=metrics(trace["points"],raw,observed,quality,times,valid,trace["accepted"],common,rows,truth)
    report={"scope":"real recordings: geometric/model consistency, not hand ground truth",
        "input_sha256":hashlib.sha256(data).hexdigest(),"samples":len(rows),
        "valid_samples":int(valid.sum()),"common_frames":int(common.sum()),
        "quality_distribution":stats(quality[observed]),
        "quality_stratum_counts":{label:int(np.count_nonzero(observed & (quality>=lo) & (quality<hi)))
            for label,lo,hi in [("low",.1,.5),("middle",.5,.8),("high",.8,1.000001)]},
        "variants":reports}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.captures/"protocol.json").read_text())
    expected=len(protocol["episodes"])*protocol["repeats"]
    if len(protocol["runs"])!=expected:
        parser.error("capture protocol is incomplete")
    args.out.mkdir(parents=True,exist_ok=False)
    summary={"protocol_sha256":hashlib.sha256((args.captures/"protocol.json").read_bytes()).hexdigest(),
        "episodes":protocol["episodes"],"settings":SETTINGS,"runs":[]}
    for run in protocol["runs"]:
        row=dict(run)
        try:
            if run["returncode"] or not run.get("source_exhausted"):
                raise ValueError("capture did not complete")
            r=run_capture(args.captures/run["directory"],args.out/run["directory"])
            row.update(report=str(Path(run["directory"])/"report.json"),common_frames=r["common_frames"],error=None)
        except Exception as exc:
            row["error"]=f"{type(exc).__name__}: {exc}"
        summary["runs"].append(row)
        (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(row),flush=True)
    return int(any(row["error"] for row in summary["runs"]))


if __name__=="__main__":
    raise SystemExit(main())
