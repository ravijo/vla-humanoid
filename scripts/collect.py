"""Record scripted-expert demos as a GR00T/LeRobot dataset (UNITREE_G1 layout) with NVIDIA's exporter.
Only successful episodes are saved.
  MUJOCO_GL=egl .venvs/wbc/bin/python scripts/collect.py --episodes 50 --out data/clean_0
Seeds start at --seed0 (default 1000) so they never overlap the evaluation seeds (0-999).
Resumes if --out already has episodes.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expert import pnp_bottle as expert  # noqa: E402
from expert.harness import G1Harness  # noqa: E402
from decoupled_wbc.control.utils.sync_sim_utils import generate_frame, get_data_exporter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--out", required=True)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--noise", type=float, default=0.0, help="DART noise std (m) on the right wrist target")
ap.add_argument("--nav-noise", action="store_true", help="navigation DART: walk noise and overshoots")
a = ap.parse_args()

task = "LMPnPBottle"
expert.NOISE_STD = a.noise
expert.NAV_NOISE = a.nav_noise
h = G1Harness(task, offscreen=True)
obs = h.reset(a.seed0)
instruction = obs["annotation.human.task_description"]
out = (ROOT / a.out).resolve()
exporter = get_data_exporter(h.cfg, obs, h.rm, save_path=out)
done = exporter.meta.total_episodes
seed = a.seed0 + done * 2  # rough resume point; seeds are only used once per run
print(f"task {task!r}: {instruction!r} -> {out} ({done} episodes already)", flush=True)


UB = list(h.rm.get_joint_group_indices("upper_body"))
WAIST = list(h.rm.get_joint_group_indices("waist"))
WAIST_SLOTS = [UB.index(j) for j in WAIST]


def record(h):
    st = h.state_info  # (max_len, len, state) taken before the step, like NVIDIA's loop
    frame = generate_frame(h.prev_obs, h.wbc_action, h.seed, st[2], st[1], st[0],
                           h.teleop_cmd, h.wbc_goal, save_img_obs=True)

    # The balance policy owns the waist and lags the command. Learn the COMMAND, because at deploy time
    # the policy's waist output is sent as the command (the stock exporter stores the lagged output).
    # With noise injection, the label is the clean expert command (h.label_upper), not the executed one.
    frame["action"] = frame["action"].copy()
    frame["action"][UB] = h.label_upper
    frame["observation.sim.target_upper_body_pose"] = h.label_upper.astype(np.float64)
    frame["action.eef"] = h.label_wrist_pose.astype(np.float64)
    frame["teleop.navigate_command"] = h.label_nav.astype(np.float64)  # clean walk command (navigation DART)
    frame["task"] = instruction
    exporter.add_frame(frame)


tried, t0 = 0, time.time()
while done < a.episodes:
    tried += 1
    ok = expert.run_episode(h, seed, on_step=record)
    if ok:
        exporter.save_episode()
        done += 1
    else:
        exporter.skip_and_start_new_episode()
    el = time.time() - t0
    print(f"seed {seed}: {'saved' if ok else 'discarded'} | {done}/{a.episodes} saved, "
          f"{tried} tried, {el / tried:.0f} s/episode", flush=True)
    seed += 1
print(f"done: {done} episodes in {out}; expert success during collection {done}/{tried}")
h.close()
