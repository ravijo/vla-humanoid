"""Closed-loop evaluation. GR00T (served by Isaac-GR00T's run_gr00t_server.py, GR00T venv)
drives the G1 in the WBC sim (this script, WBC venv) over ZMQ.

Per control cycle the policy sees the head camera, the joint state and the instruction, and returns a
40-step chunk of: arm joints, hand joints, waist, base height, navigation command. The first
--exec-steps of each chunk are sent to the WBC at 50 Hz (packed into the upper-body target exactly
like NVIDIA's n1_utils.concat_action), then the policy is queried again.

  # terminal 1 (GR00T venv): the policy server
  cd third_party/Isaac-GR00T && .venv/bin/python gr00t/eval/run_gr00t_server.py \
      --model-path <checkpoint> --embodiment-tag UNITREE_G1 --port 5555
  # terminal 2 (WBC venv): this client
  MUJOCO_GL=egl .venvs/wbc/bin/python scripts/closed_loop_eval.py --n 50 --max-secs 90 --video 5
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expert.harness import G1Harness, SafetyStop  # noqa: E402
from expert.video import VideoRecorder  # noqa: E402

STATE_KEYS = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
UPPER_KEYS = ["waist", "left_arm", "right_arm", "left_hand", "right_hand"]


class Client:
    """Minimal GR00T PolicyClient (same wire format: msgpack + msgpack_numpy ndarrays)."""

    def __init__(self, host="localhost", port=5555, timeout_ms=30000):
        self.sock = zmq.Context().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=mnp.encode))
        resp = msgpack.unpackb(self.sock.recv(), object_hook=self._decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and obj.get("__ndarray_class__"):
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return mnp.decode(obj)


def policy_obs(obs, instruction):
    return {
        "video": {"ego_view": obs["ego_view_image"][None, None]},  # (B=1, T=1, H, W, 3) uint8
        "state": {k: obs[f"state.{k}"].astype(np.float32)[None, None] for k in STATE_KEYS},
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="LMPnPBottle")
    ap.add_argument("--n", type=int, default=50, help="evaluate seeds 0..n-1 (fixed)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--exec-steps", type=int, default=20, help="steps of each 40-step chunk executed")
    ap.add_argument("--max-secs", type=float, default=90.0)
    ap.add_argument("--hold-steps", type=int, default=50, help="success must hold this long (1 s)")
    ap.add_argument("--video", type=int, default=0, help="record videos for the first K episodes")
    ap.add_argument("--out", default="outputs/closed_loop")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--log", action="store_true", help="save per-query diagnostics to log_seed*.npz")
    ap.add_argument("--no-safety", action="store_true", help="disable the joint safety stop (for videos only)")
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else list(range(a.n))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    client = Client(port=a.port)
    client.call("ping")
    h = G1Harness(a.task, offscreen=True)
    if a.no_safety:  # viewing only: let unsafe commands run so the behaviour can be seen
        h.env.safety_monitor.trigger_system_shutdown = lambda: None
    rm = h.rm
    ub = list(rm.get_joint_group_indices("upper_body"))
    slots = {k: [ub.index(j) for j in rm.get_joint_group_indices(k)] for k in UPPER_KEYS}

    results = []
    for i, seed in enumerate(seeds):
        obs = h.reset(seed)
        instruction = obs["annotation.human.task_description"]
        rec = VideoRecorder(h, out / f"seed{seed}.mp4") if i < a.video else None
        client.call("reset", {"options": None})
        held, success, t_inf, n_inf, stop = 0, False, 0.0, 0, None
        log = {"t": [], "wrist": [], "bottle": [], "state": [], "chunk_ra": [], "chunk_waist": [], "chunk_rh": [],
               "height": []}
        for step in range(int(a.max_secs / h.dt)):
            if step % a.exec_steps == 0:
                t0 = time.time()
                action, _ = client.call("get_action", {"observation": policy_obs(h.obs, instruction)})
                t_inf += time.time() - t0
                n_inf += 1
                chunk = {k: np.asarray(v)[0] for k, v in action.items()}  # (40, D)
                if a.log:
                    log["t"].append(h.t); log["wrist"].append(h.wrist_world("right")[:3, 3])
                    log["bottle"].append(h.body_pos("bottle")); log["state"].append(h.obs["q"].copy())
                    log["chunk_ra"].append(chunk["right_arm"]); log["chunk_waist"].append(chunk["waist"])
                    log["chunk_rh"].append(chunk["right_hand"]); log["height"].append(chunk["base_height_command"])
            j = step % a.exec_steps
            upper = np.zeros(len(ub))
            for k in UPPER_KEYS:
                upper[slots[k]] = chunk[k][j]
            goal = {
                "target_upper_body_pose": upper,
                "base_height_command": float(chunk["base_height_command"][j][0]),
                "navigate_cmd": chunk["navigate_command"][j].astype(float),
            }
            h.wbc.set_observation(h.obs)
            h.wbc.set_goal(goal)
            try:
                h.obs, *_ = h.env.step(h.wbc.get_action(time=h.t))
            except SafetyStop as e:  # the policy commanded unsafe joints: count as a failure, keep going
                stop = f"safety stop: {e}"
                break
            h.t += h.dt
            if rec:
                rec(h)
            held = held + 1 if h.success() else 0
            if held >= a.hold_steps:
                success = True
                break
            if h.obs["floating_base_pose"][2] < 0.4:  # fell
                stop = "fell"
                break
        if rec:
            rec.save()
        if a.log:
            np.savez(out / f"log_seed{seed}.npz", **{k: np.array(v) for k, v in log.items()})
        results.append({"seed": seed, "success": success, "secs": round(h.t, 1),
                        "infer_ms": round(1000 * t_inf / max(n_inf, 1), 1), "stop": stop})
        ok = sum(r["success"] for r in results)
        print(f"seed {seed}: {'SUCCESS' if success else 'fail'} ({h.t:.1f} s{', ' + stop if stop else ''}) | running {ok}/{len(results)}",
              flush=True)
    ok = sum(r["success"] for r in results)
    summary = {"task": a.task, "episodes": len(results), "success": ok, "rate": ok / len(results),
               "exec_steps": a.exec_steps, "results": results}
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    print(f"closed-loop success: {ok}/{len(results)} = {100 * ok / len(results):.0f}%  -> {out}/results.json")
    h.close()


if __name__ == "__main__":
    main()
