"""Run the scripted expert for N seeds and report success. Optional videos to outputs/expert/.
  MUJOCO_GL=egl .venvs/wbc/bin/python scripts/expert_eval.py --seeds 0 1 2 --video
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expert import pnp_bottle as expert  # noqa: E402
from expert.harness import G1Harness  # noqa: E402
from expert.video import VideoRecorder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--n", type=int, default=None, help="use seeds 0..n-1")
ap.add_argument("--video", action="store_true")
ap.add_argument("-v", "--verbose", action="store_true")
ap.add_argument("--noise", type=float, default=0.0, help="DART noise std (m) on the right wrist target")
ap.add_argument("--nav-noise", action="store_true", help="navigation DART: walk noise and overshoots")
a = ap.parse_args()
seeds = list(range(a.n)) if a.n else a.seeds

expert.NOISE_STD = a.noise
expert.NAV_NOISE = a.nav_noise

h = G1Harness("LMPnPBottle")
ok = 0
for s in seeds:
    rec = None
    if a.video:
        h.reset(s)  # the renderer needs this seed's scene; run_episode resets again with the same seed
        rec = VideoRecorder(h, ROOT / f"outputs/expert/seed{s}.mp4")
    try:
        succ = expert.run_episode(h, s, on_step=rec, verbose=a.verbose)
    finally:
        if rec:
            rec.save()
    ok += succ
    print(f"seed {s}: {'SUCCESS' if succ else 'fail'}", flush=True)
print(f"expert success: {ok}/{len(seeds)} = {100 * ok / len(seeds):.0f}%")
h.close()
