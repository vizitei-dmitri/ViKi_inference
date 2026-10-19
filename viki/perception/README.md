# viki.perception — video → 3-D hand skeleton

**Stage 2** · `episodes/<id>/raw/` → `rec.npz`

The durable episode stage is offline. Decode the recorded colour + depth, run a pluggable hand-pose backend
per camera per frame, lift the 2-D detection to 3-D with measured depth,
transform into the workspace frame with the recorded extrinsics, and write
per-camera landmark trajectories. Cross-camera fusion is **not** done here — it
is deferred to `viki.prepare`.

## Files

| file | what |
|---|---|
| `extract.py` | offline extraction: `raw/` → `rec.npz` + raw 2-D observations; uses cached/preset SDK color→depth projection, with a warned identity fallback for legacy recordings |
| `cloud.py` | offline Viewer cloud fusion with calibrated depth geometry, background subtraction, angular/metric depth-edge rejection, voxel deduplication and workspace crop |
| `backends/` | `HandPoseBackend` ABC + implementations; MediaPipe and RTMPose-Hand are implemented, while HaMeR/YOLO remain stubs — see `backends/README.md` |
| `camera_prep.py` | `prepare_frame` : `Frame` → `PreparedFrame` (RGB, depth in metres, depth K) |
| `geometry.py` | `lift_to_3d` (2-D + depth → camera-frame 3-D), `camera_landmarks_to_world` (apply extrinsics) |
| `segmentation.py` | prompted SAM 2.1 video masks and calibrated RGB-D lift |
| `object_model.py` | core/shell rigid models, robust SE(3) tracks, contact freeze and compact point decisions over a semantic cloud |
| `scene.py` | resumable episode stage: prompts → masks → semantic cloud → object models |
| `hand_angles.py` | `compute_end_effector_pose` — the single site that derives the wrist SE(3) pose from landmarks (also used by `prepare`) |
| `observations.py` / `triangulate.py` | stored raw 2-D evidence + multi-camera triangulation with geometric/angular gates |
| `profiles.py` / `run.py` | immutable named offline recipes and episode perception orchestration; current stable fused-hand recipe is V4 |
| `live.py` | persistent MediaPipe VIDEO detectors → exact factory color rays → current-frame triangulation → causal XYZ/SO(3) filters; public `LiveHandEstimator` |

## Contract

`cloud.py` also accepts an *experimental* per-frame depth-range bias and an
explicit alternate output directory. This never changes the canonical Viewer
cloud or raw depth. For the two-Kinect board-depth diagnosis, validation and
the corrected comparison artifact, see
[`docs/2026-10-04-two-kinect-cloud-handedness.md`](../../docs/2026-10-04-two-kinect-cloud-handedness.md).

- **in:** `PreparedFrame`, `DepthProjector` (Protocol), `CalibrationExtrinsics`.
- **out:** `rec.npz` — keys in `contracts.REC_KEYS`:
  `device_ids`, `timestamps`, `points (N,21,3)`, `landmark_ids (21,)`,
  `confidence (N,21)` *(stub: detector visibility only)*.

## Palm frame

`compute_end_effector_pose` builds the wrist rotation from the MCP knuckle
spread (INDEX→PINKY), **not** the thumb: `x = norm(MIDDLE_MCP − WRIST)`,
`z = norm(x × (PINKY_MCP − INDEX_MCP))`, `y = z × x`. Missing a required
landmark falls back to the centroid of the available palm landmarks with an
identity rotation.

## Causal live estimation

`LiveHandEstimator.from_episode(reference_episode, landmark_mode="all21")`
loads the reference's raw factory calibration, camera intrinsics/extrinsics
and one persistent MediaPipe VIDEO detector per camera. It does not read
`rec.npz`, `cln.npz`, `plan.h5` or future images, and never modifies the
reference episode. Model initialization/download happens before processing.
The public `load_live_geometry` API also exposes the cached calibrated camera
contexts and their calibration hash without starting devices.

`process(SyncedFrameGroup)` returns a `contracts.LiveHandEstimate` with 21
rig-frame XYZ slots, direct observation masks, geometric quality, source and
generation timestamps, pinch-center position, palm rotation and separate
validity flags. The default solves all 21 landmarks. `landmark_mode="arm7"`
solves indices `0,1,4,5,8,9,17`; other slots remain NaN/unobserved. Both modes
still use MediaPipe's 21-landmark detector. `target_anchor="wrist"` selects a
wrist position instead of the default midpoint of thumb/index fingertips.
Opening is the metric fingertip distance, not a measured robot tool state.
The current detector requires a scene containing one intended hand; it does
not implement multi-person identity association or elbow/shoulder estimation.

The independent live recipe `live-human-sdk-rays-causal-v1` uses this order:

1. Reject duplicate/reversed native camera times, old/non-increasing source
   times, stale/future arrivals and excessive camera skew before inference.
   Duplicate/stale frames return `None` and cannot refresh an input timeout.
2. Run the persistent detectors in native full color image coordinates.
3. Build exact cached SDK color rays, convert them into ideal pixels, and
   triangulate with zero distortion so rays are corrected exactly once.
   Missing factory geometry fails startup; there is no identity fallback.
   The angular gate is 25 mrad, with a minimum ray angle of 5 degrees.
   Native depth-frame ranges are not used as color-frame depth priors.
4. Check current geometry and palm anatomy. Freeze palm length references
   from ten valid past warmup observations; no episode median is used.
   Reject abrupt position jumps before filtering can conceal them.
5. Filter each directly observed XYZ using causal One Euro (2 Hz minimum
   cutoff, beta 0.5, 1 Hz derivative cutoff) and palm rotations using a
   causal 3 Hz SO(3) filter with measured observation intervals.

Only the four directly observed palm points `0,5,9,17` can produce an
orientation. Missing/degenerate orientation is NaN and invalid; the offline
centroid/identity fallback above is deliberately not used. Missing target
points invalidate the position. Partial landmark output remains explicit;
missing points are never copied forward with a fresh source timestamp.
The default freshness limit is 250 ms, including processing time; stale
results publish invalid flags. Position loss, anatomy/jump/skew failures and
processing-stale failures reset tracking filters.

Explicit experimental settings use profile
`live-human-sdk-rays-causal-tunable-v2`. Public `CausalLandmarkFilter` is the
same causal XYZ filter used by the estimator. `LIVE_FILTER_MIN_CUTOFF_HZ`
and `LIVE_FILTER_BETA` default to 2 and 0.5; constructor overrides are also
accepted. Optional `LIVE_FILTER_QUALITY_POWER` (default 0) scales filter alpha
by clipped geometric quality raised to that power. This weight is not a
calibrated noise covariance, and experiments found extra lag; it is disabled
by default. Changing parameters does not alter the canonical V1 recipe.

`LIVE_ORIENTATION_RECOVERY` defaults to false. The experimental mode confirms
three coherent fresh palm rotations after an aged reference or large jump,
including recovery after position loss. Missing orientation interrupts that
confirmation. It does not issue bind requests or resume robot movement.
V1's historical rotation behavior is retained when the option is off.

`capture_raw=True` adds pre-filter current-frame `raw_points` to the estimate
and hand log; it is opt-in and never reads a future frame. The capture/filter
benchmark scripts use those points to compare candidates without rerunning
MediaPipe. `scripts/benchmark_live_bones.py` is a separate fixed-length
projection prototype, not the live estimator or a full anatomical joint fit.

For real cameras, call `verify_cameras(existing_manager)` after the rig has
fresh frames and before starting control. It checks image shapes, recorded
depth/color modes and actual factory calibration identity, catching changed
Kinect device order. It cannot establish that camera mounts or the robot
base have remained physically fixed. That registration must be valid for
the session. Source classes and camera ownership are documented in
[`viki/cameras`](../cameras/README.md); the coordinating runtime uses these
public APIs. None of the immutable offline profiles or protected baselines
is edited by the live estimator.

## Scene perception

SAM 2.1 is intentionally isolated from the normal ViKi image because the
PyTorch CUDA wheels are several gigabytes. Build/run the dedicated profile:

The complete supported branch is one resumable command. It records `segment`
and `object_model` in `status.json`; rerunning it skips durable completed stages:

```bash
docker compose run --rm sam2 scene episodes/<id> --objects 3
```

It can also be included in the complete episode pipeline:

```bash
docker compose run --rm sam2 run episodes/<id> --scene-prompts data/prompts/<scene>.json
```

The explicit object count and dedicated image are deliberate: recordings may
contain no manipulable object, and the normal hand/IK path must not acquire a
multi-gigabyte CUDA dependency. The individual commands below remain available
for correction, inspection and recovery.

For an episode recorded against a stored empty-scene depth plate, frame-zero
prompts can first be proposed from fused 3-D foreground components:

```bash
docker compose run --rm cli auto-prompts episodes/<id> \
  --objects 3 --out data/prompts/<scene>.json
```

The proposal generator treats the largest foreground component as the operator
and compact, multi-view, colour-saturated components as object candidates. It
projects the same selected 3-D components into every RGB camera to produce a
box, a positive click, and negative clicks on the other components. Candidate
objects default to `other_dynamic`: frame-zero geometry alone cannot determine
their task role. Component measurements and thresholds remain in the prompt
JSON for audit.

```bash
docker compose build sam2
docker compose run --rm sam2 segment episodes/<id> \
  --prompts data/prompts/<scene>.json --lift-3d
```

If a long 2-D run completed but the optional 3-D lift was interrupted, reuse
the masks without rerunning SAM:

```bash
docker compose run --rm cli segment-lift episodes/<id>
```

Build the non-destructive rigid object model after the 3-D lift:

```bash
docker compose run --rm cli object-model episodes/<id>
```

The official `sam2.1_hiera_small.pt` checkpoint goes in `models/sam2/`; its
expected SHA-256 is
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.
The image pins upstream commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4` and PyTorch 2.8 CUDA 12.8.
The default 100-frame chunk is chosen for the repository's 11 GiB container
ceiling; reduce it when tracking more than three simultaneous instances.

Prompts use schema `viki_sam2_prompts_v1`: declare stable positive integer
object IDs and labels (`operator`, `manipulated_object`, `other_object`, or
`other_dynamic`), then give every object a frame-0 box and/or positive/negative
points for each camera. Later prompts are corrections. Coordinates are in the
original colour-video pixels.

Outputs live under
`intermediates/segmentation/sam2.1_hiera_small/`: bit-packed mask chunks,
per-camera overlay videos, prompt/model provenance, and—when `--lift-3d` is
used—mask-supported world-frame point clouds plus a provisional centroid and
visible-dimensions track. Nothing downstream reads these artifacts implicitly.
Missing or unlabelled points remain unknown, never free space.

Both the Viewer cloud builder and the semantic RGB-D lift reject mixed depth
samples near discontinuities. The neighbourhood is specified in ray angle
(`CLOUD_EDGE_RADIUS_RAD`) and converted through the recorded depth focal length,
so it keeps the same physical meaning when recording resolution changes. The
depth jump itself is metric (`max(CLOUD_EDGE_JUMP_MM,
CLOUD_EDGE_JUMP_RELATIVE * depth)`). Exact libk4a colour-to-depth warping is on
by default so RGB and SAM masks are sampled at the same depth pixels used for
XYZ. After its one-time SDK setup, measured warmed cost is about 4.9–6.7 ms per
camera/frame. Semantic lift also applies the saved empty-scene depth plate when
available; this prevents a correct 2-D mask from admitting static background
behind the object.

`object-model` adds `object_models.npz` beside those artifacts. It does not copy
or rewrite XYZ/RGB. The archive holds a temporally supported canonical
core/shell surface, `T_world_object(t)`, tracking residual/coverage/confidence,
rotation information, an operator-proximity contact flag, and compact
accepted/reassigned/rejected/ambiguous decisions referring to point indices in
each source semantic-cloud frame. Shell promotion is disabled on contact
frames. This is a supported perception artifact with explicit limitations, not
an unconditional source of physical truth: cube-aware retarget explicitly
requires one labelled `manipulated_object`, while hand-only IK ignores it.
Trajectory export copies a completed `object_models.npz` as a
sidecar and records its stage provenance. Object-relative IK remains separately
gated until pose observability and contact phases have acceptance tests.

Measured results and known failure modes from the first two real scenes are in
[`docs/2026-09-14-sam2-segmentation-probe.md`](../../docs/2026-09-14-sam2-segmentation-probe.md)
and the follow-on core/shell experiment is recorded in
[`docs/2026-09-14-object-model-filter-probe.md`](../../docs/2026-09-14-object-model-filter-probe.md).
The calibrated background-to-prompt experiment on the complete `pyramid` scene
is recorded in
[`docs/2026-09-14-pyramid-auto-prompts.md`](../../docs/2026-09-14-pyramid-auto-prompts.md).
