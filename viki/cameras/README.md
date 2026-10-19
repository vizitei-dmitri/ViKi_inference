# viki.cameras — capture

**Stage 1** · live device → `episodes/<id>/raw/`

The only package that talks to camera SDKs (`pyrealsense2`, `libk4a`). Detects
and opens RealSense / Azure Kinect, streams colour (BGR uint8) + depth (uint16
mm) in per-camera worker threads, aligns frames across devices on a host clock,
and records whole scenes into an episode directory.

## Files

| file | what |
|---|---|
| `base.py` | `CameraBackend` ABC (compat re-export of `Frame` / `SyncedFrameGroup` / `CameraIntrinsics` from `viki.contracts`) |
| `realsense.py` / `kinect.py` | concrete backends, imported lazily by the manager so a missing SDK doesn't break other imports |
| `manager.py` | `CameraManager` — discovery, start/stop, per-camera ring buffer, backend factory |
| `sync.py` | `MultiCameraSync` — host-clock grouping; verified Kinect pairs use device timestamps → `SyncedFrameGroup` |
| `hw_sync.py` | strict multi-Kinect role/cable policy (SDK-free validation) |
| `record.py` | `SceneRecorder` — records synced RGB-D into `raw/` (colour `.mp4`, raw depth `.npy`, `timestamps.json`, the intrinsics/extrinsics in force) and marks `status.json` |
| `live.py` | public `CameraFrameSource` over an existing manager and read-only paced `RecordedFrameSource`; factory RealSense color-ray helper |

## Contract

- **out:** `Frame` (`latest_frame` / `nearest_frame`), `SyncedFrameGroup`, and the `raw/` directory layout.
- Consumers pull frames; there is no push pub/sub.
- The current two-Kinect rig uses one independent PCI xHCI controller per
  camera (Intel motherboard + Renesas PCIe). Separate hubs under one controller
  were not sufficient and previously hard-locked this host.
- One Kinect may run standalone. With two or more connected Kinects, ViKi is
  fail-closed: `KINECT_SYNC` must cover the rig exactly, the UI starts the whole
  rig subordinate-first, SDK jack detection must confirm every required cable,
  and the per-device timestamp offset must remain stable within 500 µs
  across buffered frames. Recording is refused unless all checks pass.
- Kinect quirks (`align_depth_to_color` disabled, WFOV capped at 15 fps, `stop()` sleeps 2 s) live in `kinect.py` and must not change without hardware.

## Live frame sources

`CameraFrameSource(manager, device_ids=[...], sync_fps=20)` pulls groups from
the existing `MultiCameraSync`. Required IDs are fixed for the session and
must already be running; a multi-Kinect rig must have verified hardware sync.
The source checks arrival freshness (250 ms by default) and native device
timestamps. A repeated camera frame is not a new observation, even if the
sync tick changed. Lost devices or reversed clocks require a new session;
they are not silently relabelled as new measurements.

This source never starts or stops SDKs. `close()` closes only the source;
the web lifespan or CLI owner manages `CameraManager`. Reuse the web's
manager instead of opening a second manager on the same cameras. Existing
worker ring buffers and hardware sync checks remain unchanged.

`RecordedFrameSource(Episode(...), speed=1.0)` provides the same pull API
from recorded `raw/` RGB-D pairs, without cameras or a robot connection.
Instantiate it after detector/model setup: its playback clock begins at
construction. Color and depth are decoded one due group at a time, not
loaded as an episode-sized image array. If processing falls behind, old due
groups are dropped; future images are not read. Original device timestamps
remain intact for deduplication. Recorded host arrival/sync times are retimed
onto the current monotonic clock, retaining per-camera offsets and the
configured playback speed. Nothing in the source episode is written.

Both sources expose `poll()` (also `read()`), `exhausted`, `device_ids`,
`fps`, `source_summary` and idempotent `close()`. `poll()` returns a
`SyncedFrameGroup` or `None` without waiting for a future capture. The
recorded source exposes `dropped_groups` as a diagnostic. Geometry belongs
to the reference calibration; the source does not infer new extrinsics.
Before live processing, public `perception.LiveHandEstimator.verify_cameras`
checks the actual backend factory calibration and image modes against that
reference, catching swapped devices. Physical camera mounting/robot
registration still requires a valid calibration.

## Adding a backend

Subclass `CameraBackend`, implement `start` / `stop` / `get_frame` / `device_id`
/ `is_running` / `project_color_to_depth`, then register it in
`CameraManager._make_backend()` and detection in `list_devices()`.
