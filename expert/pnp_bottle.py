"""Scripted expert for LMPnPBottle: step up to the table, grasp the bottle with the right hand,
lift it, side-step about 1 m left along the tables while carrying it, set it upright on the target table,
release and return to rest.

The two lab tables stand side by side (the target table starts ~2 cm past the first one), with their near
edge 0.13 m in front of the robot, so walking means side-stepping along them, as a person would.
Walking is commanded with the same navigation velocities (vx, vy, wz) a joystick sends to the WBC's
lower-body policy; a P-controller on the privileged pelvis pose generates them.

It reads privileged object poses from the simulator and drives the robot only through teleop-style
commands (see harness.py), so the recorded data has the UNITREE_G1 layout.

Hand orientation is "natural": the fingers point along the shoulder->hand direction (tilted down a
little when reaching low) with the palm facing inward, like a person reaching. It is never held at a
fixed world orientation, which bends the wrist unnaturally once the hand comes back near the body.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from expert.harness import G1Harness, SafetyStop

# Right Dex3 hand, in the wrist_yaw_link frame: fingers point +x, the palm faces +y. The grasp centre
# (where the bottle axis should be) sits in front of the palm, between the thumb and the fingers.
GRASP_IN_WRIST = np.array([0.12, 0.04, 0.0])
GRASP_HEIGHT = 0.07  # above the bottle's base
GRASP_PITCH = 0.2  # max downward tilt of the hand at the grasp/place: more tilts the fingertips into the bottle
PREGRASP = 0.09  # start this far out along the palm normal, then move in palm-first
RELEASE_BACK = 0.03  # ...and start sliding back while the hand opens, so the thumb swings clear of the bottle
RETREAT_BACK = 0.06  # after release, also pull back along the fingers; straight off the palm, the thumb drags the bottle

STAND_H = 0.74
WORK_H = 0.72  # slight knee bend while working at the table
# Where the pelvis stands relative to the object it works on (world x/y offsets): the bottle ends up in
# front of the right shoulder.
REACH = np.array([0.40, -0.12])
# Place point on the target table, relative to its centre: near its near-right corner, so the walk is short.
PLACE_OFF = np.array([-0.16, -0.32])
CARRY = np.array([0.30, -0.16])  # carried grasp centre, pelvis frame x/y (closer to the body than the grasp)
WITHDRAW = np.array([0.18, -0.26, 0.12])  # pelvis frame, beside the right hip, after the release
STEP_BACK = 0.15
LIFT = 0.12  # bottle base clearance above the table while lifted/carried
WAIST = dict(gain=1.0, pitch=0.15)
WAIST_YAW_MAX = 0.9
# Walking (navigation command) limits and gains; a smooth P-controller, rate-limited so steps start gently.
V_MAX = np.array([0.25, 0.35, 0.5])  # vx, vy (m/s), wz (rad/s)
ACC = np.array([0.5, 0.6, 1.0])  # per second
K_NAV = np.array([1.5, 1.5, 2.0])
MAX_REAIM = 0.06  # a bigger place correction means the grasp went wrong
ARRIVE = (0.05, 0.05)  # m, rad: close enough to stop walking
V_MIN = 0.2  # m/s: the lower-body policy barely steps below ~0.1 (0.15 stalls going back sideways)
# Wrist DART for data collection: the executed right-wrist target wanders off the expert's by a smooth random
# offset (Ornstein-Uhlenbeck, std NOISE_STD m, correlation NOISE_TAU s) while the recorded action is the clean
# target, so the data teaches the policy to correct. Only in free arm motion (Seg.noise). 0 = off.
NOISE_STD, NOISE_TAU = 0.0, 0.5
# Navigation DART, on the side-step to the target table: OU noise on the executed walk command, and (with
# probability NAV_OVERSHOOT_P) a deliberate overshoot where the robot keeps stepping past the stop point for a
# while. The recorded command is always the clean P-controller output, so the demos show stopping and coming
# back. The first policy walked off past the table when it missed the stop (no such states in clean demos).
NAV_NOISE = False
NAV_STD = np.array([0.05, 0.08, 0.10])  # vx, vy, wz
NAV_TAU = 1.0
NAV_OVERSHOOT_P, NAV_OVERSHOOT_SECS, NAV_OVERSHOOT_V = 0.7, (0.5, 2.5), 0.25


class BadGrasp(RuntimeError):
    """The bottle is not where a good grasp would hold it: abort cleanly between sim steps."""


def natural_R(shoulder: np.ndarray, target: np.ndarray, max_pitch=0.6) -> np.ndarray:
    """Fingers along the horizontal shoulder->target direction, tilted down by half the reach angle;
    the palm normal (wrist +y) stays horizontal, so a vertical bottle is still grasped from the side."""
    v = target - shoulder
    yaw = np.arctan2(v[1], v[0])
    pitch = np.clip(0.5 * np.arctan2(-v[2], np.hypot(v[0], v[1])), 0.0, max_pitch)
    return Rot.from_euler("ZY", [yaw, pitch]).as_matrix()


def _interp(T0, T1, s):
    T = np.eye(4)
    T[:3, 3] = T0[:3, 3] + (T1[:3, 3] - T0[:3, 3]) * s
    T[:3, :3] = Slerp([0, 1], Rot.from_matrix([T0[:3, :3], T1[:3, :3]]))(s).as_matrix()
    return T


@dataclass
class Seg:
    secs: float  # duration (minimum for walking segments)
    label: str
    grasp: Optional[np.ndarray] = None  # world grasp-centre target, or pelvis-frame if rel
    R: Optional[np.ndarray] = None  # wrist orientation (world, or pelvis-frame if rel)
    rel: bool = False  # target expressed in the pelvis frame (follows the body while walking)
    close: bool = False
    height: float = STAND_H
    waist: np.ndarray = None
    goal: Optional[Callable] = None  # walking: pelvis xy at segment start -> (x, y, yaw) pelvis goal in world
    rest: bool = False  # right arm back to its rest pose
    noise: float = 0.0  # DART gain at the end of the segment (fades from the previous)

    def wrist(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.grasp - self.R @ GRASP_IN_WRIST
        return T


def _yaw(h: G1Harness) -> float:
    return Rot.from_matrix(h.T_world_pelvis()[:3, :3]).as_euler("ZYX")[0]


def _waist_to(shoulder, target):
    yaw = WAIST["gain"] * np.arctan2(target[1] - shoulder[1], target[0] - shoulder[0])
    return np.array([np.clip(yaw, -WAIST_YAW_MAX, WAIST_YAW_MAX), 0.0, WAIST["pitch"]])


def place_point(h: G1Harness) -> np.ndarray:
    tt = h.body_pos("table_target")
    top = tt[2] + h.be.table_target.mj_obj.top_offset[2]
    return np.array([tt[0] + PLACE_OFF[0], tt[1] + PLACE_OFF[1], top])  # bottle base


def plan(h: G1Harness) -> list[Seg]:
    bottle = h.body_pos("bottle")
    place = place_point(h)
    p_grasp = bottle[:2] - REACH  # pelvis stand points
    p_place = place[:2] - REACH
    up = np.array([0, 0, 1.0])
    g = bottle + [0, 0, GRASP_HEIGHT]
    pl = place + [0, 0, GRASP_HEIGHT]

    # The shoulder as it will be once the robot stands at each work point (same posture, shifted).
    sh0 = h.mj_body_world("robot0_right_shoulder_pitch_link")
    pel0 = h.T_world_pelvis()[:3, 3]
    sh_g = sh0 + np.r_[p_grasp - pel0[:2], 0]
    sh_p = sh0 + np.r_[p_place - pel0[:2], 0]
    Rg = natural_R(sh_g, g, max_pitch=GRASP_PITCH)
    Rp = natural_R(sh_p, pl, max_pitch=GRASP_PITCH)
    Wg, Wp = _waist_to(sh_g, g), _waist_to(sh_p, pl)
    out_g, out_p = -PREGRASP * Rg[:, 1], -PREGRASP * Rp[:, 1]
    out_r = out_p - RETREAT_BACK * Rp[:, 0]

    # Carry pose (pelvis frame): bottle upright, in front of the right hip, clear of the table top.
    carry_z = bottle[2] + LIFT + GRASP_HEIGHT - STAND_H
    carry = np.array([CARRY[0], CARRY[1], carry_z])
    Rc = natural_R(sh0 - pel0, carry, max_pitch=GRASP_PITCH)
    z0 = np.zeros(3)
    return [
        Seg(1.0, "step-in", goal=lambda p0: (*p_grasp, 0.0), rest=True, waist=z0),
        Seg(2.0, "raise", g + out_g + 0.10 * up, Rg, height=WORK_H, waist=Wg, noise=1.0),
        Seg(1.0, "pre-grasp", g + out_g, Rg, height=WORK_H, waist=Wg, noise=1.0),
        Seg(0.6, "settle", g + out_g, Rg, height=WORK_H, waist=Wg, noise=1.0),
        Seg(1.6, "approach", g, Rg, height=WORK_H, waist=Wg),
        Seg(0.8, "close", g, Rg, close=True, height=WORK_H, waist=Wg),
        Seg(1.2, "lift", g + LIFT * up, Rg, close=True, height=WORK_H, waist=Wg),
        Seg(1.2, "tuck", carry, Rc, rel=True, close=True, height=STAND_H, waist=z0),
        Seg(1.0, "walk", carry, Rc, rel=True, close=True, height=STAND_H, waist=z0,
            goal=lambda p0: (*p_place, 0.0)),  # no DART while holding/walking: it spins the robot or loosens the grip
        Seg(1.8, "carry", pl + LIFT * up, Rp, close=True, height=WORK_H, waist=Wp),
        Seg(1.2, "lower", pl, Rp, close=True, height=WORK_H, waist=Wp),
        Seg(1.0, "release", pl + 0.15 * out_p - RELEASE_BACK * Rp[:, 0], Rp, height=WORK_H, waist=Wp),
        Seg(1.0, "retreat", pl + out_r, Rp, height=WORK_H, waist=Wp),
        Seg(1.0, "raise-off", pl + out_r + 0.10 * up, Rp, height=WORK_H, waist=Wp, noise=1.0),
        # Pull the hand straight back to the body (pelvis frame) before turning the torso back: a direct
        # return to rest swings the hand across the freshly placed bottle as the waist untwists.
        Seg(1.2, "withdraw", WITHDRAW, Rc, rel=True, height=STAND_H, waist=Wp, noise=1.0),
        # Step back from the table (squares the stance after the walk; the standing policy otherwise slowly
        # pivots the robot and the rest pose's fingertips can catch the bottle), then lower the arm.
        Seg(0.8, "step-back", WITHDRAW, Rc, rel=True, height=STAND_H, waist=z0, noise=1.0,
            goal=lambda p0: (p0[0] - STEP_BACK, p0[1], 0.0)),
        Seg(1.8, "rest", rest=True, height=STAND_H, waist=z0, noise=1.0),
    ]


def reaim_place(h: G1Harness, segs: list[Seg], place: np.ndarray) -> np.ndarray:
    """After the walk, measure how the bottle sits in the hand (it is usually tilted a few degrees) and
    re-plan the place so the BOTTLE lands upright, base on the place point and just touching the table. The
    hand is rotated by the small correction needed. A bottle set down tilted rests on its rim, held up only
    by the hand, and rocks over when the fingers open."""
    T_w = h.wrist_world("right")
    T_b = np.eye(4)
    T_b[:3, :3] = Rot.from_quat(h.be.sim.data.body_xquat[h.be.obj_body_id["bottle"]], scalar_first=True).as_matrix()
    T_b[:3, 3] = h.body_pos("bottle")
    T_wb = np.linalg.inv(T_w) @ T_b
    u = T_b[:3, 2]
    axis = np.cross(u, [0, 0, 1.0])
    ang = np.arctan2(np.linalg.norm(axis), u[2])
    R_fix = Rot.from_rotvec(axis / max(np.linalg.norm(axis), 1e-9) * ang).as_matrix()
    T_b_des = np.eye(4)
    T_b_des[:3, :3] = R_fix @ T_b[:3, :3]
    T_b_des[:3, 3] = place + [0, 0, -0.003]
    T_w_des = T_b_des @ np.linalg.inv(T_wb)
    R_new = T_w_des[:3, :3]
    g_new = T_w_des[:3, 3] + R_new @ GRASP_IN_WRIST
    i_lower = next(i for i, s in enumerate(segs) if s.label == "lower")
    shift = g_new - segs[i_lower].grasp
    for s in segs[next(i for i, s in enumerate(segs) if s.label == "carry"):]:
        if s.grasp is not None and not s.rel:
            s.grasp = s.grasp + shift
            s.R = R_new
    return shift


def run_episode(h: G1Harness, seed: int, on_step=None, verbose=False) -> bool:
    try:
        return _run(h, seed, on_step, verbose)
    except (SafetyStop, BadGrasp) as e:
        if verbose:
            print(f"  aborted: {e}")
        return False


def _run(h: G1Harness, seed: int, on_step, verbose) -> bool:
    h.reset(seed)
    P_inv = np.linalg.inv(h.T_world_pelvis())
    TR_rest = P_inv @ h.wrist_world("right")
    TL_rest = P_inv @ h.wrist_world("left")
    TL = TL_rest
    corr = np.zeros(3)
    rng = np.random.default_rng(seed + 7919)
    noise = np.zeros(3)
    a_ou = np.exp(-h.dt / NOISE_TAU)
    nav = np.zeros(3)  # clean expert walk command (the label)
    nav_exec = np.zeros(3)  # executed (differs from nav only with NAV_NOISE)
    nav_ou = np.zeros(3)
    a_nav = np.exp(-h.dt / NAV_TAU)

    def step(T_des, close, height, waist, adapt=True, gain=0.0):
        nonlocal corr, noise
        ne = nav_exec if NAV_NOISE else nav
        T_cmd = T_des.copy()
        T_cmd[:3, 3] += corr
        P = h.T_world_pelvis()
        if NOISE_STD > 0:
            noise = a_ou * noise + np.sqrt(1 - a_ou**2) * NOISE_STD * rng.standard_normal(3)
            noise = np.clip(noise, -2 * NOISE_STD, 2 * NOISE_STD)
            T_exec = T_cmd.copy()
            T_exec[:3, 3] += gain * noise
            h.step(P @ TL, T_exec, close_right=close, base_height=height, waist=waist, nav=ne, nav_label=nav, T_right_label=T_cmd)
        else:
            h.step(P @ TL, T_cmd, close_right=close, base_height=height, waist=waist, nav=ne, nav_label=nav)
        err = T_des[:3, 3] + (gain * noise if NOISE_STD > 0 else 0) - h.wrist_world("right")[:3, 3]
        if adapt:
            corr = np.clip(corr + 0.04 * err, -0.08, 0.08)
        if on_step:
            on_step(h)

    def nav_update(goal):
        """Rate-limited P-control toward the pelvis goal (x, y, yaw); returns True once there."""
        nonlocal nav
        p = h.T_world_pelvis()
        yaw = _yaw(h)
        e_w = np.array(goal[:2]) - p[:2, 3]
        c, s_ = np.cos(yaw), np.sin(yaw)
        e = np.array([c * e_w[0] + s_ * e_w[1], -s_ * e_w[0] + c * e_w[1], np.arctan2(np.sin(goal[2] - yaw), np.cos(goal[2] - yaw))])
        want = np.clip(K_NAV * e, -V_MAX, V_MAX)
        lin = np.linalg.norm(want[:2])
        if np.hypot(*e[:2]) > ARRIVE[0] and lin < V_MIN:  # the WBC ignores slower commands
            want[:2] *= V_MIN / max(lin, 1e-6)
        nav = nav + np.clip(want - nav, -ACC * h.dt, ACC * h.dt)

        # Arrived when close: the WBC barely steps for commands under ~0.08 (and switches to its standing
        # policy under 0.05), so the last few cm never close. Where the place lands is re-planned from the
        # actual stop point anyway.
        done = np.linalg.norm(want) < 0.06 or (np.hypot(*e[:2]) < ARRIVE[0] and abs(e[2]) < ARRIVE[1])
        if done:
            nav = np.zeros(3)
        return done, e

    def target(seg, P_start, T_from_w, T_from_rel, s):
        if seg.rest:
            return h.T_world_pelvis() @ _interp(T_from_rel, TR_rest, s)
        if seg.rel:
            return h.T_world_pelvis() @ _interp(T_from_rel, seg.wrist(), s)
        return _interp(T_from_w, seg.wrist(), s)

    T_from = h.wrist_world("right")
    for _ in range(int(0.5 / h.dt)):
        step(T_from, False, STAND_H, np.zeros(3))
    segs = plan(h)
    place = place_point(h)
    height, waist, ng = STAND_H, np.zeros(3), 0.0
    for i, seg in enumerate(segs):
        n = int(seg.secs / h.dt)
        h0, w0, g0 = height, waist, ng
        T_from_rel = np.linalg.inv(h.T_world_pelvis()) @ T_from
        k, done, max_k = 0, seg.goal is None, int(20.0 / h.dt)
        goal = seg.goal(h.T_world_pelvis()[:2, 3].copy()) if seg.goal else None
        over_left = None  # overshoot steps left (navigation DART), None = not yet triggered
        while k < n or not done:
            s = 0.5 - 0.5 * np.cos(np.pi * min(1.0, (k + 1) / (0.8 * n)))
            if seg.goal is not None:
                done, e = nav_update(goal)
                if verbose and k % 50 == 0:
                    print(f"    walk k {k} err {np.round(e, 3)} nav {np.round(nav, 3)} pelvis {np.round(h.T_world_pelvis()[:2, 3], 3)}")
                nav_exec = nav.copy()
                if NAV_NOISE and seg.label == "walk":
                    if done and over_left is None:
                        over_left = int(rng.uniform(*NAV_OVERSHOOT_SECS) / h.dt) if rng.random() < NAV_OVERSHOOT_P else 0
                    if over_left:  # keep side-stepping past the stop point; the label says stop / come back
                        nav_exec = np.array([0.0, NAV_OVERSHOOT_V, 0.0])
                        over_left -= 1
                        done = False
                    elif not done:
                        nav_ou = a_nav * nav_ou + np.sqrt(1 - a_nav**2) * NAV_STD * rng.standard_normal(3)
                        nav_exec = nav + np.clip(nav_ou, -2 * NAV_STD, 2 * NAV_STD)
                if k > max_k:
                    raise BadGrasp(f"walk did not converge, err {np.round(e, 3)}")

            # The left arm stays at rest: parked out to the side, its fingers drop into the gap between
            # the tables while side-stepping and snag, twisting the robot. The right hand never crosses here.
            if seg.label == "rest":
                corr *= 1 - s
            T_cmd = target(seg, None, T_from, T_from_rel, s)
            walking = seg.goal is not None or seg.rel
            step(T_cmd, seg.close, h0 + (seg.height - h0) * s, w0 + (seg.waist - w0) * s,
                 adapt=seg.label != "release" and not walking, gain=g0 + (seg.noise - g0) * s)
            k += 1
        if seg.goal is not None:  # come to a full stop before working with the arms again
            nav = np.zeros(3)
            nav_exec = np.zeros(3)
            for _ in range(int(0.6 / h.dt)):
                step(target(seg, None, T_from, T_from_rel, 1.0), seg.close, seg.height, seg.waist, adapt=False)
        T_from = target(seg, None, T_from, T_from_rel, 1.0)
        height, waist, ng = seg.height, seg.waist, seg.noise
        if seg.label == "walk":

            # Place in front of where the robot actually stopped (clamped onto the target table's near-right
            # area), then re-aim for the bottle's measured in-hand pose.
            P = h.T_world_pelvis()
            new = place.copy()
            new[:2] = P[:2, 3] + P[:2, :2] @ REACH
            new[:2] = np.clip(new[:2], place[:2] - 0.06, place[:2] + 0.06)
            for sg in segs[i + 1:]:
                if sg.grasp is not None and not sg.rel:
                    sg.grasp = sg.grasp + (new - place)
            place = new
            shift = reaim_place(h, segs, place)
            if np.linalg.norm(shift) > MAX_REAIM:
                raise BadGrasp(f"re-aim {np.linalg.norm(shift):.3f} m")
            if verbose:
                print(f"  re-aim place by {np.round(shift, 3)}")
        if verbose:
            act = h.wrist_world("right")
            err = np.linalg.norm(act[:3, 3] - T_from[:3, 3])
            p = h.T_world_pelvis()[:3, 3]
            print(f"  {seg.label:10s} t {h.t:5.1f} wrist err {err * 100:4.1f} cm pelvis ({p[0]:+.3f},{p[1]:+.3f}) "
                  f"yaw {_yaw(h):+.3f} bottle {np.round(h.body_pos('bottle'), 3)}", flush=True)
    for _ in range(int(0.5 / h.dt)):
        step(h.T_world_pelvis() @ TR_rest, False, STAND_H, np.zeros(3))
    return h.success()
