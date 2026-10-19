# viki.live — causal human-hand following

For **finger-only DG5F following**, use `viki live-hand`, provided by Compose's
`live-hand` service. It connects current 21-point stereo observations to the
existing Hybrid Vector/DexPilot solver and an independent MuJoCo kinematic
display. It does not instantiate the arm executor. Recorded RGB-D and actual
calibrated cameras share the same causal estimator; see
[finger live guide](../../docs/live_dg5f_hand.md). `--backend mock|tesollo`
adds the original guarded DG5F Runtime in a spawned SDK-owner process.
Tesollo requires `--execute --source cameras` and manual ARM; no automatic
hardware ARM or recorded-input execution is exposed. Its camera-to-physical
integration still requires validation on the actual rig.

This runtime connects current RGB-D observations to incremental retargeting
and a separate arm executor. It does not run the offline episode pipeline
repeatedly, and does not require a trained policy checkpoint.

The default observation contains the **21 wrist/finger landmarks**, a
pinch-center position and a directly observed palm orientation. There is no
elbow/shoulder/body pose estimator. The `arm7` option triangulates the seven
control anchors while keeping unknown slots explicit in the 21-point DTO.

## Runtime ownership and public API

| Module/API | Responsibility |
|---|---|
| `session.py` / `LiveSession` | One bounded observation worker; model setup, frame source, causal estimator, target publication, operator requests, shutdown and session provenance |
| `LiveSessionConfig` | Reference episode, recorded/camera source, MuJoCo/UR10e backend and immutable session options |
| `LiveManager` | Application-lifespan owner; reuses the web camera manager and rejects overlapping sessions |
| `load_mapping` | Validates an explicit operator mapping file and preserves its digest/calibration association |
| `cli.py` / `run_live_cli` | CLI interaction through the same session API; terminal bind/clutch/stop and optional independent viewer |
| `viewer.py` | Independent MuJoCo state display; it does not drive perception or servo timing |

Neighbour stages are used through their public APIs:

```text
cameras.RecordedFrameSource / CameraFrameSource
    → perception.LiveHandEstimator
    → contracts.HumanTarget
    → replay.LiveController
    → retarget.OnlineRetargeter + MuJoCo / RTDE backend
```

The default perception rate is **15 Hz**. The default executor rate is
**50 Hz**; `125 Hz` remains experimental and keeps strict timing faults
enabled. These are configured rates, not guarantees of delivered throughput
under every camera/render load. Expensive perception, rendering and disk
logging are separated from the arm owner's process. Target exchange is a
bounded latest-value mailbox; old targets are not queued for later motion.

Initialization loads persistent detectors and cached SDK geometry before
starting the executor. A recorded source starts its monotonic playback
epoch only after the executor is ready. A camera source reuses an existing
manager when provided; a standalone CLI session owns its new manager.
Native Kinect synchronization warms up before source creation, and actual
factory calibration/device order/image modes must match the reference.

`start()` begins observing. `bind()` and `clutch()` return asynchronous
request IDs; their accepted/rejected result is available in
`snapshot()["controller"]`. A posted request is not an acknowledgement that
following began. `stop()` requests independent controlled shutdown and
returns status, including unfinished executor shutdown if applicable.
The server's **Live** tab displays separate human-rig and robot-base views,
all landmark masks and palm/TCP axes, and these operator controls. Switching
tabs disposes the browser display without stopping the server session.

## Freshness, binding and motion

Every `HumanTarget.valid_until_us` is derived from the oldest source-frame
host arrival, not the end of perception processing. The default TTL is
250 ms. Duplicate or stale camera frames do not extend this deadline;
invalid geometry retains separate validity flags and cannot acquire a
fresh valid position by holding old numbers.

Binding requires a fresh valid target and a stationary robot. It records
the current human anchor and current measured arm state. Following then
uses relative displacement from those anchors, scaled by 0.25 by default;
binding a stationary hand does not introduce an absolute FK correction.
The default follows position; `orientation=True` also requires directly
observed, fresh palm orientation. Arm winding is preserved.

Clutch, input loss, workspace excursions or rejected IK release binding
and request bounded deceleration. Fresh input after loss does not silently
resume following: the operator must bind again. Simulation-only
`auto_bind=True` requests the first bind after the executor has consumed a
valid input sequence; it requests only once and never enables automatic
reacquisition. Hardware rejects this option.

Limits include six-arm-joint ranges, command velocity/acceleration,
tracking residual, model floor/self-collision margins and executor
deadlines. The default bring-up bounds are 0.25 rad/s and 0.5 rad/s² with a
0.15 m maximum mapped displacement. The model covers the arm and configured
floor; it does not reconstruct room obstacles or certify operator safety.
The physical tool is not controlled or measured by this arm-only runtime.

UR10e sessions require the robot IP, explicit hardware enable and an
operator-confirmed rig-to-robot mapping file with the reference calibration ID.
Starting observation creates
receive-only robot access; RTDE control/script upload happens only at a
manual bind. Replay and live share the same robot ownership lock. The
runtime never automatically homes the physical robot, changes its safety
settings or restores motion after a controller fault.

## Artifacts and validation

Every session has a unique directory under `data/live_sessions/` by default:

```text
<session-id>/
    manifest.json                 source/calibration/model/options provenance
    hand.jsonl                    filtered landmarks and original source times
    session.json                  final status, counters and executor report
    executor/<run-id>/execution.h5 controller inputs, sent commands and measured state
    executor/<run-id>/session.json executor final report
```

JSON represents unavailable numbers as `null`; missing physical tool
feedback remains unavailable. Partial/faulted runs are preserved. Bounded
input logging reports dropped rows instead of blocking acquisition.
Executor log failure/overflow is a fault rather than a silent success.
These artifacts do not claim grasp success or automatically become
eligible autonomous-policy training episodes.

Hardware-independent tests inject frame/estimator/controller components
to verify source-epoch ordering, source-based TTL, manual bind, one-time
simulation bind, calibration identity, independent failure cleanup,
ownership and durable NaN-safe logs. Perception tests verify prefix
causality and exact-ray correction; retarget/replay tests check command
bounds, binding and stop/fault behaviour. A real recorded-input run exercises
MediaPipe, cached factory SDK geometry and causal filtering without opening
cameras or moving a robot. Current recorded measurements and hardware
bring-up instructions live in the repository's `docs/`.

## Combined recorded online review

`RecordedReviewStore` and `RecordedReview` provide read-only playback of a
completed recorded-input run. Inputs are raw timestamps/cloud, `hand.jsonl`
and `execution.h5` measured states. No `rec.npz`, `cln.npz` or `plan.h5` is read.
The **Online review** tab overlays the source cloud, online filtered hand,
measured arm FK and target/achieved TCP on one runtime timeline.

New recorded sources persist their host/recorded clock mapping in the session
manifest. Legacy runs are accepted only when their epoch can be uniquely
recovered by matching saved native camera stamps and original per-camera
offsets across hand observations; nominal FPS is never used to guess it.
Hand geometry appears only after `generated_time_us`, is hidden when rejected
or stale, and is never interpolated using future estimates. Robot geometry
uses the last `q_observed` available at the selected time, not `q_commanded`.
The original binding losses, braking and faults remain visible.

Robot base geometry is registered for display at the first successful bind;
the configured motion scale remains unchanged. This is a fixed visual
alignment, not a measured physical base calibration. The raw world-display
transform is applied equally to cloud, hand and robot. The current URDF hash
must match the run. Completed logs only; caches are bounded, and endpoints
never construct a camera or robot control interface. See
[the review guide](../../docs/2026-10-19-online-review.md).
