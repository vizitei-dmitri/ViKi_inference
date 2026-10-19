"""Adapter to the user's unchanged portable Vector + DexPilot Hybrid solver.

No ROS, camera SDK, transport, SDK commands or anatomical hallucinations here.
The external source/models remain in their original project, mounted read-only.
"""
import hashlib
import importlib
from pathlib import Path
import sys
import time

import numpy as np

from viki.contracts import HandJointTarget, LiveHandEstimate


DG5F_JOINT_NAMES = tuple(f"rj_dg_{finger}_{joint}" for finger in range(1,6) for joint in range(1,5))


class DG5FHandRetargeter:
    def __init__(self, root="/opt/dg5f-hybrid", *, ttl_s=.25, engine_factory=None,
                 clock_us=None):
        if not np.isfinite(ttl_s) or not 0 < ttl_s <= 1:
            raise ValueError("hand target TTL must be in (0, 1] seconds")
        self.ttl_us = int(ttl_s*1e6)
        self.clock_us = clock_us or (lambda:time.monotonic_ns()//1000)
        self._last_sequence = -1
        self._last_source = None
        self._needs_reset = False
        self.manifest = {"solver":"external DG5F Camera Demo HybridRetargeter",
            "input":"21 directly observed rig-frame XYZ in metres, right hand",
            "coordinates":"original landmarks_to_mano; no new reflection, scaling or axis transform",
            "output":"canonical DG5F 20-joint order, radians",
            "fixed_joint":"rj_dg_5_1 = 0 (existing physical fault map)",
            "ttl_s":ttl_s,"anatomical_fit":False}
        if engine_factory is None:
            root = Path(root).resolve()
            source = root/"src"
            module_file = source/"dg5f_demo/retarget.py"
            if not module_file.is_file():
                raise FileNotFoundError(f"DG5F Hybrid source missing: {module_file}; set DG5F_HYBRID_ROOT")
            sys.path.insert(0,str(source))
            module = importlib.import_module("dg5f_demo.retarget")
            if Path(module.__file__).resolve() != module_file.resolve():
                raise RuntimeError("another dg5f_demo is already imported; start an isolated hand process")
            engine_factory = lambda:module.HybridRetargeter(low_pass_alpha=1.)
            files = [module_file,source/"dg5f_teleop/hand_math.py",source/"dg5f_teleop/constants.py",
                source/"dg5f_demo/assets/dg5f_hybrid.yaml",source/"dg5f_demo/assets/dg5f_dexpilot.yaml",
                source/"dg5f_demo/assets/dg5f/urdf/dg5f_right.urdf"]
            self.manifest.update(root=str(root),sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
            self.model_path = source/"dg5f_demo/assets/dg5f/scene_dg.xml"
        else:
            self.model_path = None
        self._factory = engine_factory
        self._engine = engine_factory()

    def step(self, estimate: LiveHandEstimate) -> HandJointTarget:
        deadline = int(estimate.source_time_us)+self.ttl_us
        result = HandJointTarget(estimate.sequence,estimate.source_time_us,deadline,
            DG5F_JOINT_NAMES,np.full(20,np.nan),np.zeros(5),np.full(7,np.nan),False)
        def invalid(reason):
            self._needs_reset = True
            result.reason = reason
            return result
        if estimate.sequence <= self._last_sequence or (self._last_source is not None and estimate.source_time_us <= self._last_source):
            return invalid("duplicate_or_out_of_order")
        self._last_sequence,self._last_source = estimate.sequence,estimate.source_time_us
        now = int(self.clock_us())
        if now < estimate.source_time_us or now > deadline:
            return invalid("stale_or_future_observation")
        points = np.asarray(estimate.points,float)
        observed = np.asarray(estimate.observed,bool)
        if (points.shape != (21,3) or observed.shape != (21,) or not observed.all()
                or not np.isfinite(points).all() or not estimate.position_valid):
            return invalid("incomplete_or_invalid_observation")
        if self._needs_reset:
            self._engine = self._factory()
            self._needs_reset = False
        # Input metres and MANO ordering already match the original solver.
        q_deg,weights,distances = self._engine.retarget(points.copy())
        q = np.deg2rad(np.asarray(q_deg,float))
        weights,distances = np.asarray(weights,float),np.asarray(distances,float)
        if q.shape != (20,) or weights.shape != (5,) or distances.shape != (7,) or not all(np.isfinite(a).all() for a in (q,weights,distances)):
            raise ValueError("invalid DG5F Hybrid output")
        if abs(q[16]) > 1e-10:
            raise ValueError("DG5F Hybrid fault map does not hold rj_dg_5_1")
        if int(self.clock_us()) > deadline:
            return invalid("expired_during_retarget")
        result.q_rad,result.contact_weights,result.fingertip_distances_m = q,weights,distances
        result.valid = True
        return result
