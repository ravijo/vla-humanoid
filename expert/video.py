"""Episode video: a third-person camera tracking the pelvis next to the robot's head camera."""

import imageio
import mujoco
import numpy as np


class VideoRecorder:
    def __init__(self, h, path, every=2, size=(480, 640)):
        self.h, self.path, self.every, self.k = h, path, every, 0
        m = h.be.sim.model._model
        self.r = mujoco.Renderer(m, height=size[0], width=size[1])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_pelvis")
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.0, 150.0, -25.0
        self.ego = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_oak_egoview")

        # robosuite recolours collision geoms (group 0): robot green, hands blue. Hide them so the
        # video shows the visual meshes only, matching the camera image the policy sees.
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0
        self.frames = []

    def __call__(self, h):
        self.k += 1
        if self.k % self.every:
            return
        d = h.be.sim.data._data
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        a = self.r.render()
        self.r.update_scene(d, camera=self.ego, scene_option=self.opt)
        b = self.r.render()
        self.frames.append(np.concatenate([a, b], axis=1))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(round(1 / (self.h.dt * self.every)))
        imageio.mimwrite(self.path, self.frames, fps=fps)
        return self.path
