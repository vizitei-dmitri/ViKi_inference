"""Reference-matched camera startup shared by arm and finger-only sessions."""
import json
from pathlib import Path
import time


def start_reference_cameras(reference, manager, devices):
    meta = json.loads((Path(reference)/"meta.json").read_text())
    kinect = [d for d in devices if d.startswith("kinect_")]
    if len(kinect) >= 2:
        request = meta["cameras"][kinect[0]]["requested"]
        manager.start_configured_kinect_rig(**{k:request[k] for k in ("fps","color_width","color_height","depth_mode")})
    for device in devices:
        if device in kinect and len(kinect) >= 2:
            continue
        request = meta["cameras"][device].get("requested",{})
        manager.start(device,**{k:request[k] for k in ("fps","color_width","color_height","depth_mode") if k in request})


def wait_reference_cameras(manager, devices, stop):
    from viki.cameras import HardwareSyncError
    end = time.monotonic()+8
    last_error = "no frames"
    while time.monotonic() < end and not stop.is_set():
        if all(manager.latest_frame(d) is not None for d in devices):
            try:
                manager.require_hardware_sync_ready()
                return
            except HardwareSyncError as exc:
                last_error = str(exc)
        stop.wait(.05)
    if stop.is_set():
        raise RuntimeError("operator_stop")
    raise RuntimeError("live camera synchronization not ready: " + last_error)
