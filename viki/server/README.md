# viki.server — HTTP/WS transport

Thin FastAPI wrapper over the pipeline packages and a static web UI. Domain
logic lives in stages and `viki.live`. Offline episode jobs and independent live
hand-following sessions are exposed through separate routers.

## Assembly (`app.py`)

`lifespan` builds `CameraManager`, `CalibrationManager`, `ReplayManager` and `LiveManager`
(on `app.state`). `deps.py` hands them to routes via `Depends`.
ReplayManager owns stop requests; the execution thread performs RTDE cleanup.
LiveManager reuses the camera manager and supervises perception plus the separate
arm-owner process. Shutdown stops live, then replay, then cameras; cleanup is
attempted for each resource even if a preceding cleanup fails.

## Routes

| router | endpoints | drives |
|---|---|---|
| `cameras.py` | device list, start/stop, info, colour/depth MJPEG | `CameraManager`, `render` |
| `calibration.py` | intrinsics/extrinsics capture + solve + board preview | `CalibrationManager` |
| `skeleton.py` | `POST /skeleton/capture_base/{id}` — static background depth for offline scene subtraction | — |
| `recording.py` | `POST /record/start` → `SceneRecorder` in a background job | `cameras.record` |
| `pipeline.py` | episode geometry/cloud/object-model diagnostics, batch `POST /pipeline/{perceive,retarget}`, retarget robot/plan scene APIs, and jobs | `perception`, `prepare`, `retarget`, `episode` |
| `replay.py` | check/inspect, `POST /replay` job, `POST /replay/jobs/{id}/stop` | `viki.replay` |
| `live.py` | `GET /live/status`, `POST /live/{start,bind,clutch,stop}` | `viki.live` |
| `live.py` review routes | `GET /live/reviews`, `/live/reviews/{id}`, `/live/reviews/{id}/frames/{frame}`, `/live/reviews/{id}/cloud/{frame}` | `viki.live.RecordedReviewStore`, read-only |
| `label.py` | `GET/POST /label` | `viki.labeling` |
| `export.py` | `POST /export` job | `viki.export` |
| `system.py` | config get/set/reset, restart | `viki.config` |

## Jobs

`jobs.py` — a minimal in-process registry (`submit` / `get` / `all_jobs`). Not
durable; fine for a single-user research tool. `pipeline` / `replay` / `export`
return a `job_id`; poll `…/jobs/{job_id}`.

## Frontend

`static/` — plain ES-module HTML/JS, no build step, served at `/`. Panels:
cameras, calibration, **episodes** (`episodes.js` — list, per-stage job buttons,
label form, record) and **3-D viewer** (`viewer.js` + `scene3d.js` — a three.js
orbit view of point clouds, hand/robot trajectories and object-model filter
diagnostics). The object layers use `/pipeline/episode/{id}/object-model` for
canonical core/shell + pose tracks and the binary `.../object-model/{frame}`
endpoint for accepted/rejected/reassigned/ambiguous observations. Viewer also
loads `/pipeline/episode/{id}/retarget` to overlay the saved offline robot plan
and target/achieved TCP paths on the cloud timeline. That plan belongs to the
episode's active retarget, including when inspecting a different hand geometry
variant; the status labels it as a saved offline plan. The Live panel
(`live.js`, `live_scene.js`) polls at 5 Hz without overlapping requests and
renders the 21-point hand/palm and robot links independently of the control
process. Switching panels disposes display resources while the session keeps
running. Bind/Clutch requests return IDs; the controller's matching request
result reports acceptance or rejection.

**Online review** (`online_review.js`, `review_scene.js`) is distinct from both
Viewer's offline plan and the live owner. It replays logged causal inference
alongside the original point cloud using one runtime clock. It shows measured
arm joints, filtered hand availability/age, target and achieved TCP, control
state and stop causes. Its requests are GET-only. A `RecordedReviewStore` is
built in app lifespan and injected through deps. Direct links use
`?online_session=<id>#online-review`; seeking never calls a hardware endpoint.
