#!/usr/bin/env bash
# Native (non-Docker) install of NVIDIA's decoupled whole-body-control MuJoCo sim for the Unitree G1.
# Mirrors decoupled_wbc/docker/Dockerfile.deploy, minus the teleop/hardware extras.
# Creates its own venv at .venvs/wbc (Python 3.10, separate from GR00T's 3.12 venv).
# Needs the submodules checked out first: git submodule update --init
#
# What it installs / fetches:
#   - CycloneDDS 0.10.x, built from source into the venv (eclipse-cyclonedds on GitHub)
#   - unitree_sdk2_python (vendored in GR00T-WholeBodyControl/external_dependencies)
#   - decoupled_wbc[full] (vendored)
#   - robosuite (submodule third_party/robosuite: branch leo/support_g1_locomanip of github.com/xieleo5/robosuite)
#       NOTE: a personal-account fork; this is the exact repo/branch NVIDIA's Dockerfile installs.
#   - gr00trobocasa (vendored in decoupled_wbc/dexmg)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WBC="$ROOT/third_party/GR00T-WholeBodyControl"
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"
for d in "$WBC/decoupled_wbc" third_party/robosuite/robosuite; do
  [ -d "$d" ] || { echo "missing $d: run 'git submodule update --init' first"; exit 1; }
done

echo "== venv (Python 3.10)"
[ -d .venvs/wbc ] || uv venv --python 3.10 .venvs/wbc
# shellcheck disable=SC1091
source .venvs/wbc/bin/activate

echo "== CycloneDDS (build from source, ~1-2 min)"
# shellcheck disable=SC1091
source "$WBC/install_scripts/setup_cyclonedds.sh"

echo "== unitree_sdk2_python"
uv pip install -e "$WBC/external_dependencies/unitree_sdk2_python"

echo "== decoupled_wbc[full]"
# Upstream pyproject sets readme/license to ../README.md and ../LICENSE, which every setuptools
# version refuses to build. So install its declared deps directly, and put the package on the path
# with a .pth file (equivalent to the editable install; the package lives at $WBC/decoupled_wbc).
REQS="$ROOT/.venvs/wbc/decoupled_wbc_full_requirements.txt"
python3.12 -c '
import sys, tomllib
p = tomllib.load(open(sys.argv[1], "rb"))["project"]
print("\n".join(p["dependencies"] + p["optional-dependencies"]["full"]))
' "$WBC/decoupled_wbc/pyproject.toml" > "$REQS"
GIT_LFS_SKIP_SMUDGE=1 uv pip install -r "$REQS"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$WBC" > "$SITE/decoupled_wbc_local.pth"

echo "== robosuite (G1 locomanip branch)"
uv pip install -e third_party/robosuite

echo "== gr00trobocasa"
uv pip install -e "$WBC/decoupled_wbc/dexmg/gr00trobocasa"

echo
echo "Done. CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
python -c "import mujoco, robosuite, robocasa, decoupled_wbc; print('imports OK, mujoco', mujoco.__version__)"
