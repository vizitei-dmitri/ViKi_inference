"""Experimental current-frame articulated landmark fit with a causal calibration.

Uses the project's existing 26-DoF hand model, fixed palm roots, bone lengths
and joint bounds. It does not call trajectory fitting, fill control gaps, read
depth clouds, or tune a model from future observations.
"""

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

from viki.contracts import LM, LiveHandFitEstimate, LiveHandFitOptions
from . import hand_model as hm
from .hand_angles import compute_palm_rotation


class CausalHandFitter:
    def __init__(self, options: LiveHandFitOptions | None = None):
        self.options = options or LiveHandFitOptions()
        self.options.validate()
        self.hand = None
        self._calibration = []
        self._last_time = None
        self._last_valid_time = None
        self._previous_q = None
        self.manifest = {"profile": "causal-articulated-landmarks-experimental-v1",
            "options": asdict(self.options), "geometry": "existing CapsuleHand, 26 DoF",
            "calibration": "first complete valid frames, rigid roots median in palm frame; neutral finger axes in palm plane",
            "scope": "model constraints, not validated subject-specific anatomical joint limits"}

    def reset_tracking(self):
        self._previous_q = None
        self._last_valid_time = None

    def _calibrate(self):
        # Require every finger in every chosen frame: calibrate_from_frames'
        # generic unseen-finger fallback is never used by this online branch.
        mappings = [{LM(i): points[i] for i in range(21)} for points in self._calibration]
        params = hm.calibrate_from_frames(mappings)
        # Early frames may be curled. Do not freeze that observed curl as a
        # straight reference; neutral flexion axes are tied to the rigid palm.
        for finger in hm.FINGERS:
            angle = 0.
            if finger == "thumb":
                root = params.root_xyz[finger]
                angle = float(np.arctan2(root[1], max(root[0], .005)))
            params.root_rpy[finger] = np.array([0., 0., angle])
        self.hand = hm.build(params)
        payload = params.as_dict()
        self.manifest.update(calibration_sha256=hashlib.sha256(
            json.dumps(payload,sort_keys=True).encode()).hexdigest(), params=payload,
            joint_bounds_rad=np.column_stack([self.hand.q_lo[7:], self.hand.q_hi[7:]]).tolist())
        self._calibration.clear()

    def _geometry(self, seed, delta, indices):
        import pinocchio as pin

        hand = self.hand
        q = pin.integrate(hand.model, seed, delta)
        pin.computeJointJacobians(hand.model, hand.data, q)
        pin.updateFramePlacements(hand.model, hand.data)
        jint = np.asarray(pin.dIntegrate(hand.model, seed, delta)[1])
        positions = np.array([hand.data.oMf[hand.lm_frames[int(i)]].translation for i in indices])
        jac = np.vstack([np.asarray(pin.getFrameJacobian(hand.model, hand.data,
            hand.lm_frames[int(i)], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:3] @ jint for i in indices])
        return q, positions, jac

    def update(self, points, observed, quality, source_time_us, *, valid=True):
        from scipy.optimize import least_squares

        started = time.perf_counter()
        points, quality = np.asarray(points,float), np.asarray(quality,float)
        observed = np.asarray(observed,bool).copy()
        if points.shape != (21,3) or observed.shape != (21,) or quality.shape != (21,):
            raise ValueError("hand fit requires 21 XYZ, observed masks and qualities")
        if self._last_time is not None and source_time_us <= self._last_time:
            raise ValueError("hand fit source time must increase")
        self._last_time = source_time_us
        observed &= np.isfinite(points).all(axis=1) & np.isfinite(quality) & (quality > 0)

        def result(reason, fitted=None, q=None, nfev=0, margin=float("nan"), correction=float("nan"), converged=False):
            accepted = reason == ""
            if not accepted:
                self.reset_tracking()
            return LiveHandFitEstimate(source_time_us,
                np.full((21,3),np.nan) if fitted is None else fitted.copy(), observed.copy(),
                (~observed if fitted is not None else np.zeros(21,bool)), accepted, reason,
                np.empty(0) if q is None else q.copy(), (time.perf_counter()-started)*1000,
                nfev, margin, correction, converged)

        if not valid or not observed[[0,4,8]].all():
            return result("missing_control_support")
        palm = [0,5,9,17]
        if not observed[palm].all() or compute_palm_rotation(*points[palm]) is None:
            return result("missing_or_degenerate_palm")
        if self.hand is None:
            if not observed.all() or np.any(quality < .1):
                return result("calibration_incomplete")
            lengths = [np.linalg.norm(points[int(b)]-points[int(a)])
                       for lms in hm.FINGERS.values() for a,b in zip(lms[:-1],lms[1:])]
            if min(lengths) < .002 or max(lengths) > .12:
                return result("calibration_bone_range")
            self._calibration.append(points.copy())
            if len(self._calibration) < self.options.warmup_frames:
                return result("warming_up")
            self._calibrate()

        if self._last_valid_time is not None and source_time_us-self._last_valid_time > 250_000:
            self.reset_tracking()
        mapping = {LM(i): points[i] for i in np.flatnonzero(observed)}
        seed = hm.q_from_landmarks(self.hand,mapping)
        if self._previous_q is not None:
            seed[7:] = self._previous_q[7:]
        seed[7:] = np.clip(seed[7:],self.hand.q_lo[7:]+1e-8,self.hand.q_hi[7:]-1e-8)
        indices = np.flatnonzero(observed)
        weights = np.sqrt(np.clip(quality[indices],.1,1.)**self.options.quality_power)
        prior_scale = np.sqrt(self.options.joint_prior)
        cache = {}

        def evaluate(delta):
            key = delta.tobytes()
            if key not in cache:
                q, pred, jac = self._geometry(seed,delta,indices)
                data_res = ((pred-points[indices])*weights[:,None]).reshape(-1)
                data_jac = jac*np.repeat(weights,3)[:,None]
                prior_res = prior_scale*(q[7:]-seed[7:])
                prior_jac = np.zeros((20,self.hand.nv))
                prior_jac[:,6:] = prior_scale*np.eye(20)
                cache.clear()
                cache[key] = (np.r_[data_res,prior_res],np.vstack([data_jac,prior_jac]))
            return cache[key]

        lower = np.r_[np.full(3,-.03),np.full(3,-.6),self.hand.q_lo[7:]-seed[7:]]
        upper = np.r_[np.full(3,.03),np.full(3,.6),self.hand.q_hi[7:]-seed[7:]]
        solved = least_squares(lambda x:evaluate(x)[0],np.zeros(self.hand.nv),
            jac=lambda x:evaluate(x)[1],bounds=(lower,upper),loss="soft_l1",
            f_scale=self.options.robust_scale_m,max_nfev=self.options.max_nfev,
            ftol=1e-5,xtol=1e-5,gtol=1e-6)
        q, fitted, _ = self._geometry(seed,solved.x,np.arange(21))
        correction = float(np.max(np.linalg.norm(fitted[observed]-points[observed],axis=1)))
        margin = float(min(np.min(q[7:]-self.hand.q_lo[7:]),np.min(self.hand.q_hi[7:]-q[7:])))
        reason = ("nonfinite_fit" if not np.isfinite(fitted).all() else
                  "joint_bound_violation" if margin < -1e-8 else
                  "correction_cap" if correction > self.options.correction_cap_m else "")
        if not reason:
            self._previous_q, self._last_valid_time = q.copy(), source_time_us
        return result(reason,fitted,q,int(solved.nfev),margin,correction,bool(solved.success))
