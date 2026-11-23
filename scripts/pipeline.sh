#!/usr/bin/env bash
# Collect data, fine-tune and evaluate, unattended. Rerunning skips datasets that are already done.
#   scripts/pipeline.sh v1     300 clean + 300 wrist-DART demos, train, eval on 50 seeds (~5.5 h)
#   scripts/pipeline.sh v2     add 300 walk-DART demos, retrain on everything, eval (~4.5 h, run v1 first)
#   scripts/pipeline.sh base   GR00T N1.7 without fine-tuning, eval on 20 seeds (needs a v1 or v2 checkpoint)
# Logs go to outputs/. To keep it running after the terminal closes:
#   mkdir -p outputs && nohup scripts/pipeline.sh v1 > outputs/pipeline_v1.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
SIM=.venvs/wbc/bin/python
GR00T=third_party/Isaac-GR00T/.venv/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
mkdir -p data outputs

log() { echo "[$(date +%T)] $*"; }

# collect NAME SEED0 NOISE [--nav-noise]: 50 successful expert demos in data/NAME, plus GR00T's dataset stats.
# The video encoder crashes now and then, so a failed run is retried with new seeds.
collect() {
  local name=$1 seed0=$2 noise=$3 extra=${4:-} d="data/$1"
  [ -f "$d/.done" ] && return 0
  for try in 0 1 2; do
    rm -rf "$d"
    if timeout 4000 $SIM scripts/collect.py --episodes 50 --out "$d" --seed0 $((seed0 + 10000 * try)) \
         --noise "$noise" $extra > "outputs/collect_$name.log" 2>&1; then
      (cd third_party/Isaac-GR00T && .venv/bin/python gr00t/data/stats.py \
         --dataset-path "$ROOT/$d" --embodiment-tag UNITREE_G1 > /dev/null 2>&1)
      touch "$d/.done"
      log "$name done"
      return 0
    fi
    log "$name failed (try $((try + 1)) of 3), see outputs/collect_$name.log"
  done
}

collect_all() {  # each argument is "NAME SEED0 NOISE [--nav-noise]"; three run at a time
  for job in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 10; done
    collect $job &
    sleep 5
  done
  wait
}

datasets() {  # finished data/PREFIX_* for each prefix, as absolute paths joined with ':'
  local p d out=""
  for p in "$@"; do
    for d in "$ROOT"/data/"$p"_*; do [ -f "$d/.done" ] && out="${out:+$out:}$d"; done
  done
  echo "$out"
}

train() {  # train OUT_DIR DATASETS
  log "training $1 on $2"
  mkdir -p "$1"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $GR00T scripts/finetune_g1.py \
    --base-model-path nvidia/GR00T-N1.7-3B --embodiment-tag UNITREE_G1 --dataset-path "$2" \
    --output-dir "$ROOT/$1" --global-batch-size 32 --max-steps 30000 --save-steps 2500 \
    --save-total-limit 8 --dataloader-num-workers 6 > "$1/train.log" 2>&1
}

evaluate() {  # evaluate CHECKPOINT NAME N_SEEDS: serve the checkpoint and run the closed-loop eval
  local ckpt=$1 name=$2 n=$3 server
  [ -d "$ckpt" ] || { log "no checkpoint at $ckpt"; return 1; }
  log "evaluating $ckpt on $n seeds"
  (cd third_party/Isaac-GR00T && exec .venv/bin/python gr00t/eval/run_gr00t_server.py \
     --model-path "$ROOT/$ckpt" --embodiment-tag UNITREE_G1 --port 5555) > "outputs/server_$name.log" 2>&1 &
  server=$!
  until grep -q "Server ready" "outputs/server_$name.log"; do
    kill -0 $server 2>/dev/null || { log "server died, see outputs/server_$name.log"; return 1; }
    sleep 3
  done
  $SIM scripts/closed_loop_eval.py --n "$n" --video 5 --out "outputs/closed_loop/$name" 2>&1 \
    | grep -E "^seed|closed-loop|Error|Traceback"
  kill $server
}

# The untouched NVIDIA weights plus the G1 modality config and normalisation stats of a fine-tuned checkpoint.
make_base() {  # make_base FROM_CHECKPOINT OUT_DIR
  local snap f
  snap=$($GR00T -c 'from huggingface_hub import snapshot_download; print(snapshot_download("nvidia/GR00T-N1.7-3B"))')
  mkdir -p "$2/experiment_cfg"
  for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors model.safetensors.index.json; do
    ln -sf "$snap/$f" "$2/$f"
  done
  for f in config.json embodiment_id.json processor_config.json statistics.json experiment_cfg/config.yaml; do
    cp "$1/$f" "$2/$f"
  done
}

case "${1:-}" in
  v1)
    sets=()
    for k in 0 1 2 3 4 5; do
      sets+=("clean_$k $((1000 + 100 * k)) 0" "wrist_dart_$k $((3000 + 100 * k)) 0.02")
    done
    collect_all "${sets[@]}"
    train outputs/v1_lora "$(datasets clean wrist_dart)"
    evaluate outputs/v1_lora/checkpoint-30000 v1 50
    ;;
  v2)
    sets=()
    for k in 0 1 2 3 4 5; do sets+=("nav_dart_$k $((4000 + 100 * k)) 0.02 --nav-noise"); done
    collect_all "${sets[@]}"
    train outputs/v2_lora "$(datasets clean wrist_dart nav_dart)"
    evaluate outputs/v2_lora/checkpoint-30000 v2 50
    ;;
  base)
    from=$(ls -d outputs/v2_lora/checkpoint-30000 outputs/v1_lora/checkpoint-30000 2>/dev/null | head -1)
    [ -n "$from" ] || { echo "run v1 first: the baseline borrows its G1 config and normalisation stats"; exit 1; }
    make_base "$from" outputs/base_zeroshot
    evaluate outputs/base_zeroshot base 20
    ;;
  *)
    sed -n '2,5p' "$0"
    exit 1
    ;;
esac
log "done"
