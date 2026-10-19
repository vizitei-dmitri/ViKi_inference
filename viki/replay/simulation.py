"""Offline MuJoCo arm preview. Never connects to a robot or certifies a rig."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from viki.contracts import Episode
from viki.replay.plan import URReplayOptions, read_plan, servo_commands
from viki.retarget import normalize_robot
from viki.retarget import load_archive


def preview_mujoco(ep: Episode, *, model_path: str, speed_scale: float = 0.02,
                   mode: str = "position", gui: bool = False,
                   video: str | None = None, video_fps: int = 20,
                   playback_speed: float = 1.0) -> dict:
    """Position-actuator simulation or explicitly kinematic inspection."""
    import mujoco

    if mode not in ("position", "kinematic"):
        raise ValueError("mode must be position or kinematic")
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("playback_speed must be finite and positive")
    plan = read_plan(ep)
    robot = normalize_robot(plan.robot)
    if robot.description != normalize_robot("ur10e").description:
        raise ValueError("MuJoCo preview requires a UR10e plan")
    if json.loads(str(plan.metadata.get("joint_names_json", "[]"))) != list(robot.joint_names):
        raise ValueError("plan has an incorrect joint order")
    options = URReplayOptions(speed_scale=speed_scale, arm_only=True)
    times, commands, vmax, amax = servo_commands(plan.q, plan.dt, options)
    model_file = Path(model_path)
    source = {}
    source_file = model_file.parent / "source.json"
    if source_file.exists():
        source = json.loads(source_file.read_text())
        root = model_file.parent.resolve()
        for name, expected in source.get("sha256", {}).items():
            asset = (root / name).resolve()
            if not asset.is_relative_to(root) or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
                raise ValueError(f"MuJoCo model cache hash mismatch: {name}")
    model = mujoco.MjModel.from_xml_path(str(model_file))
    # The planner's floor is the calibration plane, expressed in robot axes.
    from scipy.spatial.transform import Rotation

    with load_archive(ep.plan_h5) as arc:
        base_position = np.asarray(arc["base_position_calibration"] if "base_position_calibration" in arc else np.zeros(3))
        base_rpy = np.asarray(arc["base_rpy_deg_calibration"] if "base_rpy_deg_calibration" in arc else np.zeros(3))
    calibration_to_robot = Rotation.from_euler("xyz", base_rpy, degrees=True).inv()
    floor = model.geom("floor").id
    model.geom_pos[floor] = calibration_to_robot.apply(-base_position)
    quat = calibration_to_robot.as_quat()
    model.geom_quat[floor] = quat[[3, 0, 1, 2]]
    if model.nq != 6 or model.nv != 6 or model.nu != 6:
        raise ValueError("preview expects a bare six-joint UR10e model")
    joints = [model.joint(name).id for name in robot.joint_names]
    addresses = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    actuators = []
    for joint in joints:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matches) != 1:
            raise ValueError("each arm joint must have exactly one position actuator")
        actuators.append(int(matches[0]))
    if np.any(commands < model.jnt_range[joints, 0]) or np.any(commands > model.jnt_range[joints, 1]):
        raise ValueError("command trajectory exceeds MuJoCo model joint ranges")
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    data.qpos[addresses] = commands[0]
    data.ctrl[actuators] = commands[0]
    mujoco.mj_forward(model, data)
    viewer = None
    paused = [True]
    if gui:
        import mujoco.viewer

        def key_callback(key):
            if key == 32:
                paused[0] = not paused[0]
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        viewer.cam.lookat[:] = [0, 0, 0.7]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
    renderer = writer = None
    if video:
        import cv2

        if video_fps <= 0 or video_fps > options.frequency:
            raise ValueError("video_fps must be positive and no greater than control frequency")
        path = Path(video); path.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
        renderer = mujoco.Renderer(model, height=720, width=960)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (960, 720))
        if not writer.isOpened():
            renderer.close()
            raise RuntimeError("cannot open preview video writer")
    traces, bad_contacts = [], {}
    completed = False
    next_video_time = 0.0
    epoch = time.monotonic()
    try:
        for scheduled, target in zip(times, commands):
            if viewer is not None:
                while viewer.is_running() and paused[0]:
                    viewer.sync(); time.sleep(0.01)
                    epoch = time.monotonic() - scheduled / playback_speed
                if not viewer.is_running():
                    break
            with viewer.lock() if viewer is not None else nullcontext():
                if mode == "kinematic":
                    data.qpos[addresses] = target
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                else:
                    data.ctrl[actuators] = target
                    for _ in range(4):
                        mujoco.mj_step(model, data)
                error = float(np.max(np.abs(data.qpos[addresses] - target)))
                penetration = 0.0
                for contact in data.contact:
                    if contact.dist < -1e-5:
                        names = tuple(sorted(model.geom(int(g)).name or f"geom-{g}" for g in contact.geom))
                        key = " / ".join(names)
                        bad_contacts[key] = bad_contacts.get(key, 0) + 1
                        penetration = max(penetration, -float(contact.dist))
                traces.append((scheduled, error, penetration, float(np.max(np.abs(data.qvel[dofs])))))
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("simulation produced non-finite state")
                if renderer is not None and scheduled + 1e-9 >= next_video_time:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = [0, 0, 0.7]
                    camera.distance = 2.8; camera.azimuth = 135; camera.elevation = -25
                    renderer.update_scene(data, camera=camera)
                    writer.write(renderer.render()[:, :, ::-1])
                    next_video_time += playback_speed / video_fps
            if viewer is not None:
                viewer.sync()
                delay = epoch + (scheduled + 1 / options.frequency) / playback_speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        else:
            completed = True
    finally:
        if writer is not None: writer.release()
        if renderer is not None: renderer.close()
        if viewer is not None: viewer.close()
    values = np.asarray(traces).reshape(-1, 4)
    out = ep.root / "intermediates/mujoco"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{mode}_trace.npz", scheduled_time_s=values[:, 0],
                        tracking_error_rad=values[:, 1], penetration_m=values[:, 2],
                        max_joint_speed_rad_s=values[:, 3])
    report = {
        "mode": mode, "completed": completed, "plan_sha256": plan.sha256,
        "model_xml_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "model_source_commit": source.get("commit"),
        "model_manifest_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file.exists() else None,
        "floor_position_robot_m": model.geom_pos[floor].tolist(),
        "floor_normal_robot": calibration_to_robot.apply([0, 0, 1]).tolist(),
        "mujoco_version": mujoco.__version__, "speed_scale": speed_scale,
        "playback_speed": playback_speed,
        "duration_s": len(commands) / options.frequency,
        "max_command_velocity_rad_s": vmax, "max_command_acceleration_rad_s2": amax,
        "max_simulated_tracking_error_rad": float(np.max(values[:, 1], initial=0)),
        "max_penetration_m": float(np.max(values[:, 2], initial=0)),
        "max_simulated_joint_speed_rad_s": float(np.max(values[:, 3], initial=0)),
        "penetrating_contact_pairs": bad_contacts,
        "passed_model_checks": bool(completed and not bad_contacts and np.max(values[:, 1], initial=0) <= 0.05),
        "scope": "bare arm and model floor; no real rig obstacles, tool, objects or grasp success",
        "video": video,
    }
    (out / f"{mode}_report.json").write_text(json.dumps(report, indent=2))
    return report
