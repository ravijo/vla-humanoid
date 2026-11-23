"""Programmatic 'teleoperator' for the decoupled-WBC G1 sim (no ROS, no VR).

Each control step takes the same commands a teleop device produces:
  - wrist targets (4x4, world frame here; converted to the pelvis frame the IK expects),
  - hand open/close per side (Dex3 'index-close' grasp from NVIDIA's gripper IK),
  - base height and navigation (vx, vy, wz) commands,
runs NVIDIA's TeleopRetargetingIK -> decoupled WBC -> MuJoCo, and returns the observation plus the
`teleop_cmd` / `wbc_goal` dicts in exactly the shape the stock data exporter (generate_frame) needs.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT  # noqa: E402
from decoupled_wbc.control.main.teleop.configs.configs import SyncSimDataCollectionConfig  # noqa: E402
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy  # noqa: E402
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model  # noqa: E402
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (  # noqa: E402
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK  # noqa: E402
from decoupled_wbc.control.utils.sync_sim_utils import (  # noqa: E402
    get_body_teleoped_joint_groups,
    get_env,
    get_wbc_config,
)


def pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    T[:3, 3] = pos
    return T


def fingers(closed: bool) -> dict:
    """Fingertip keypoints in the format NVIDIA's G1 gripper IK reads: thumb tip (4) meeting the
    index tip (9) means 'index close' (power grasp); apart means open."""
    f = np.zeros((25, 4, 4))
    f[4, 0, 3] = 1.0
    if closed:
        f[9, 0, 3] = 1.0
    return {"position": f}


class SafetyStop(RuntimeError):
    """NVIDIA's joint safety monitor tripped (arm joint too fast / out of limits)."""


class G1Harness:
    HAND_SECS = 0.4  # time for a hand to fully close
    HAND_OPEN_SECS = 0.9  # opening is gentler: fast-opening fingers flick a freshly placed bottle over
    WAIST_WEIGHT = 30.0  # posture-task weight on the waist joints when a waist target is given

    def __init__(self, task: str, onscreen=False, offscreen=False, render_camera=None):
        # enable_waist: the waist joints join the arm IK, so the torso leans/turns to reach low and far
        #   (without it the right wrist reaches only ~0.3 m in front of the pelvis at pelvis height).
        # gravity compensation on the arms: without it the arms sag ~7 cm below their targets.
        self.cfg = SyncSimDataCollectionConfig(
            task_name=task,
            enable_onscreen=onscreen,
            save_img_obs=offscreen,
            enable_waist=True,
            enable_gravity_compensation=True,
            gravity_compensation_joints=["arms"],
        )
        self.robot_type, self.rm = get_robot_type_and_model(self.cfg.robot, enable_waist_ik=self.cfg.enable_waist)
        kw = {} if render_camera == "ego" else {"render_camera": render_camera}
        if offscreen:  # record only the head camera (the UNITREE_G1 layout's single video stream)
            kw.update(camera_names=["robot0_oak_egoview"], camera_heights=[480], camera_widths=[640])
        self.env = get_env(self.cfg, onscreen=onscreen, offscreen=offscreen, **kw)
        self.dt = 1.0 / self.cfg.control_frequency

        # The safety monitor calls sys.exit(1) on a violation; make it a per-episode failure instead.
        def _stop():
            raise SafetyStop("joint safety violation")

        self.env.safety_monitor.trigger_system_shutdown = _stop
        self.onscreen = onscreen

    @property
    def be(self):
        return self.env.base_env

    def reset(self, seed: int):
        self.seed = seed
        self.obs, _ = self.env.reset(seed)
        self.rm.set_initial_body_pose(self.obs["q"])
        l, r = instantiate_g1_hand_ik_solver()
        self.ik = TeleopRetargetingIK(
            robot_model=self.rm,
            left_hand_ik_solver=l,
            right_hand_ik_solver=r,
            body_active_joint_groups=get_body_teleoped_joint_groups(self.cfg.robot),
        )

        # The second IK for clean labels (noise injection) is built once and reused: its state is overwritten
        # every step anyway, and rebuilding the robot model per episode leaks native memory (segfaults).
        self.ik_label = getattr(self, "ik_label", None)
        self.label_upper = self.label_wrist_pose = None
        self.wbc = get_wbc_policy(self.robot_type, self.rm, get_wbc_config(self.cfg), init_time=0.0)
        self.wbc.activate_policy()
        self.t = 0.0

        # Grip level per hand in [0, 1], ramped toward the open/close command (no instant snaps).
        self.grip = {"left": 0.0, "right": 0.0}
        ub = list(self.rm.get_joint_group_indices("upper_body"))
        self.hand_slots = {s: [ub.index(j) for j in self.rm.get_hand_actuated_joint_indices(side=s)]
                           for s in ("left", "right")}
        return self.obs

    # ---- world-frame state from the simulator (privileged; the policy never sees these) ----
    def body_pos(self, name: str) -> np.ndarray:
        return self.be.sim.data.body_xpos[self.be.obj_body_id[name]].copy()

    def T_world_pelvis(self) -> np.ndarray:
        p = self.obs["floating_base_pose"]
        return pose_to_T(p[:3], p[3:7])

    def wrist_world(self, side: str) -> np.ndarray:
        """Actual wrist pose in world (from joint state FK + floating base)."""
        wp = self.obs["wrist_pose"][:7] if side == "left" else self.obs["wrist_pose"][7:]
        return self.T_world_pelvis() @ pose_to_T(wp[:3], wp[3:])

    def mj_body_world(self, body: str) -> np.ndarray:
        m, d = self.be.sim.model._model, self.be.sim.data._data
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        return d.xpos[i].copy()

    # ---- one 50 Hz control step ----
    def step(self, T_left_w, T_right_w, close_left=False, close_right=False,
             base_height=DEFAULT_BASE_HEIGHT, nav=(0.0, 0.0, 0.0), waist=None, T_right_label=None, nav_label=None):
        """waist: optional (yaw, roll, pitch) the IK should hold the torso at (strong posture weight), so
        the arms do the reaching. Without it the waist is a weakly-regularised free DoF and wanders
        inconsistently between episodes, which a learned policy cannot reproduce.
        T_right_label: noise injection (DART). T_right_w is the perturbed target actually executed; this is
        the clean expert target. Its IK solution is stored in self.label_upper / self.label_wrist_pose, so
        the dataset pairs perturbed states with corrective actions.
        nav_label: the same for walking: nav is executed, nav_label (the clean command) is stored in self.label_nav."""
        self.label_nav = np.asarray(nav if nav_label is None else nav_label, dtype=float)
        if waist is not None:
            self.set_waist_target(self.ik, waist)

        Tinv = np.linalg.inv(self.T_world_pelvis())
        lf, rf = self.rm.supplemental_info.hand_frame_names["left"], self.rm.supplemental_info.hand_frame_names["right"]
        T_l, T_r = Tinv @ T_left_w, Tinv @ T_right_w
        lfing, rfing = fingers(close_left), fingers(close_right)
        q_ik_prev = self.ik.body_ik_solver.configuration.q.copy()  # label IK starts from the same place
        self.ik.set_goal({"body_data": {lf: T_l, rf: T_r}, "left_hand_data": lfing, "right_hand_data": rfing})
        upper = self.ik.get_action().copy()

        # NVIDIA's gripper IK gives 0 (open) or a fixed closed pose; blend between them over HAND_SECS.
        for side, closed, fing in (("left", close_left, lfing), ("right", close_right, rfing)):
            q_closed = (self.ik.left_hand_ik_solver if side == "left" else self.ik.right_hand_ik_solver)(fingers(True))
            step_ = self.dt / (self.HAND_SECS if closed else self.HAND_OPEN_SECS)
            self.grip[side] = float(np.clip(self.grip[side] + (step_ if closed else -step_), 0.0, 1.0))
            upper[self.hand_slots[side]] = self.grip[side] * q_closed
        wrist_pose = self._wrist_pose(T_l, T_r)

        if T_right_label is not None:
            if self.ik_label is None:
                _, rm2 = get_robot_type_and_model(self.cfg.robot, enable_waist_ik=self.cfg.enable_waist)
                rm2.set_initial_body_pose(self.prev_obs["q"] if hasattr(self, "prev_obs") else self.obs["q"])
                l2, r2 = instantiate_g1_hand_ik_solver()
                self.ik_label = TeleopRetargetingIK(robot_model=rm2, left_hand_ik_solver=l2, right_hand_ik_solver=r2,
                                                    body_active_joint_groups=get_body_teleoped_joint_groups(self.cfg.robot))
            if waist is not None:
                self.set_waist_target(self.ik_label, waist)

            # Same start configuration and same solver steps as the executed IK this frame, so the label differs
            # from the executed command only through the target (the arm is redundant, so independent solvers
            # drift onto different elbow branches).
            lab_solver = self.ik_label.body_ik_solver
            lab_solver.configuration.q = q_ik_prev
            lab_solver.configuration.update()
            self.ik_label.in_warmup = False
            T_rl = Tinv @ T_right_label
            self.ik_label.set_goal({"body_data": {lf: T_l, rf: T_rl}, "left_hand_data": lfing, "right_hand_data": rfing})
            lab = self.ik_label.get_action().copy()
            for side in ("left", "right"):
                lab[self.hand_slots[side]] = upper[self.hand_slots[side]]

            self.label_upper, self.label_wrist_pose = lab, self._wrist_pose(T_l, T_rl)
        else:
            self.label_upper, self.label_wrist_pose = upper, wrist_pose

        self.teleop_cmd = {
            "left_wrist": T_l, "right_wrist": T_r, "left_fingers": lfing, "right_fingers": rfing,
            "wrist_pose": wrist_pose, "target_upper_body_pose": upper,
            "navigate_cmd": np.asarray(nav, dtype=float), "base_height_command": float(base_height),
        }
        self.wbc_goal = dict(self.teleop_cmd)
        self.state_info = self.env.get_mujoco_state_info()
        self.prev_obs = self.obs  # observation the action was computed from (recorded with it)
        self.wbc.set_observation(self.obs)
        self.wbc.set_goal(self.wbc_goal)
        self.wbc_action = self.wbc.get_action(time=self.t)
        self.obs, _, _, _, self.step_info = self.env.step(self.wbc_action)
        if self.onscreen:
            self.env.render()

        self.t += self.dt
        return self.obs

    @staticmethod
    def _wrist_pose(T_l, T_r):
        return np.concatenate([
            T_l[:3, 3], R.from_matrix(T_l[:3, :3]).as_quat(scalar_first=True),
            T_r[:3, 3], R.from_matrix(T_r[:3, :3]).as_quat(scalar_first=True),
        ])

    def set_waist_target(self, ik, waist):
        post = ik.body_ik_solver.tasks["posture"]
        ix = [ik.body.joint_to_dof_index[f"waist_{n}_joint"] for n in ("yaw", "roll", "pitch")]
        post.weights[ix] = self.WAIST_WEIGHT
        q = post.target_q.copy()
        q[ix] = waist
        post.set_target(q)

    def success(self) -> bool:
        return bool(self.env.is_success()["task"])

    def close(self):
        self.env.close()
