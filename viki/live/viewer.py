"""Optional MuJoCo display process. It displays state and never computes actions."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time

import numpy as np

from viki.contracts import LiveControlOptions


def _display(options, states, requests, stop):
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import mujoco
    import mujoco.viewer
    from scipy.spatial.transform import Rotation
    from viki.retarget import normalize_robot

    model = mujoco.MjModel.from_xml_path(options.model_path)
    base = model.body("base").id
    model.body_quat[base] = [1, 0, 0, 0]
    model.body_pos[base] = 0
    normal = np.asarray(options.floor_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    floor = model.geom("floor").id
    model.geom_pos[floor] = normal * options.floor_offset_m
    rotation, _ = Rotation.align_vectors([normal], [[0, 0, 1]])
    model.geom_quat[floor] = rotation.as_quat()[[3, 0, 1, 2]]
    addresses = [model.jnt_qposadr[model.joint(n).id]
                 for n in normalize_robot("ur10e").joint_names]
    data = mujoco.MjData(model)
    data.qpos[addresses] = options.initial_q
    mujoco.mj_forward(model, data)

    def key(keycode):
        command = {32: "toggle", 66: "bind", 67: "clutch", 81: "stop"}.get(keycode)
        if command:
            try:
                requests.put_nowait(command)
            except queue.Full:
                pass

    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.lookat[:] = [0, 0, .7]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while not stop.is_set() and viewer.is_running():
            latest = None
            while True:
                try:
                    latest = states.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                with viewer.lock():
                    data.qpos[addresses] = latest
                    mujoco.mj_forward(model, data)
            viewer.sync()
            stop.wait(.04)


class LiveViewer:
    def __init__(self, options: LiveControlOptions):
        context = mp.get_context("spawn")
        self.states = context.Queue(maxsize=1)
        self.requests = context.Queue(maxsize=16)
        self.stop_event = context.Event()
        self.process = context.Process(target=_display,
            args=(options, self.states, self.requests, self.stop_event),
            name="viki-live-display", daemon=True)

    def start(self):
        self.process.start()

    def update(self, q):
        if q is None:
            return
        try:
            self.states.put_nowait(q)
        except queue.Full:
            try:
                self.states.get_nowait()
                self.states.put_nowait(q)
            except (queue.Empty, queue.Full):
                pass

    def commands(self):
        while True:
            try:
                yield self.requests.get_nowait()
            except queue.Empty:
                return

    def close(self):
        self.stop_event.set()
        self.process.join(2)
        if self.process.is_alive():
            # This process owns pixels only; stopping it cannot orphan a driver.
            self.process.terminate()
            self.process.join(1)
