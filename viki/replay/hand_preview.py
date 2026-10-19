"""DG5F kinematic joint preview. No SDK, contacts, actuator or hardware claims."""
from pathlib import Path
import numpy as np


class DG5FHandPreview:
    def __init__(self, model_path, joint_names):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        ids = [mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in joint_names]
        if len(ids) != 20 or min(ids) < 0:
            raise ValueError("DG5F model must contain all canonical joints")
        self.addresses = self.model.jnt_qposadr[ids]
        self.limits = self.model.jnt_range[ids].copy()
        if not np.all(self.model.jnt_limited[ids]):
            raise ValueError("DG5F preview needs finite joint ranges")
        self.q = np.clip(np.zeros(20),self.limits[:,0],self.limits[:,1])
        self.apply(self.q)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0,0,.17]
        self.camera.distance = .65
        self.camera.azimuth,self.camera.elevation = 130,-20

    def apply(self, q_rad):
        q = np.asarray(q_rad,float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError("preview expects 20 finite radians")
        clipped = np.clip(q,self.limits[:,0],self.limits[:,1])
        self.q = clipped.copy()
        self.data.qpos[self.addresses] = clipped
        self.mj.mj_forward(self.model,self.data)
        return int(np.count_nonzero(np.abs(q-clipped)>1e-8))

    def render(self, renderer):
        renderer.update_scene(self.data,camera=self.camera)
        return renderer.render().copy()
