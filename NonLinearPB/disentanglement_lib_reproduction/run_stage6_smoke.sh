#!/bin/bash
# Step 6 — end-to-end smoke test + CPU timing for disentanglement_lib (original code, unmodified).
#
# Conda environment: dlib-tf1-cpu
#
# Usage (launch detached so it survives disconnects):
#   setsid nohup bash disentanglement_lib_reproduction/run_stage6_smoke.sh \
#       > reproduction_outputs/stage6_runner.log 2>&1 < /dev/null &
#
# Part 1 — SMOKE TEST (pipeline correctness):
#   dlib_train with examples/model.gin (5 steps, dsprites_full, beta_vae beta=4, 10 latents, seed 0)
#   then dlib_reproduce --model_dir on it (visualize -> postprocess mean/sampled -> 18 metrics).
#   -> reproduction_outputs/smoke_beta_vae_seed0/
# Part 2 — TIMING RUN (training speed only):
#   same examples/model.gin with gin binding model.training_steps=2000.
#   -> reproduction_outputs/timing_beta_vae_seed0_2000steps/
#
# Each part writes command.txt, environment.txt, stdout.log and a DONE/FAILED marker.

NLPB=/home/argha/Mat_embedding/code/NonLinearPB
ENV=/home/argha/miniconda3/envs/dlib-tf1-cpu/bin
DLIB=$NLPB/disentanglement_lib
OUT=$NLPB/reproduction_outputs
export DISENTANGLEMENT_LIB_DATA=/home/argha/Mat_embedding/data
export PATH=$ENV:$PATH
export CUDA_VISIBLE_DEVICES=""

write_env () {   # $1 = run dir
  {
    echo "date: $(date '+%F %T')"
    echo "host: $(hostname)"
    echo "repo commit: $(git -C "$DLIB" rev-parse HEAD)"
    echo "conda env: dlib-tf1-cpu"
    echo "DISENTANGLEMENT_LIB_DATA=$DISENTANGLEMENT_LIB_DATA"
    echo "CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\""
    "$ENV/python" --version 2>&1
    "$ENV/pip" freeze 2>/dev/null
  } > "$1/environment.txt"
}

run_step () {    # $1 = run dir, $2 = label, rest = command
  local dir=$1 label=$2; shift 2
  echo "$*" >> "$dir/command.txt"
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $label"
  "$@" >> "$dir/stdout.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "[$(date '+%F %T')] END   $label exit=$rc ${dt}s"
  printf "%s\t%s\t%s\n" "$label" "$rc" "$dt" >> "$dir/step_times.tsv"
  return $rc
}

# All dlib commands run from a scratch workdir so nothing is written to an unexpected cwd.
WORK=$OUT/workdir_stage6; mkdir -p "$WORK"; cd "$WORK" || exit 1

# ---------------- Part 1: smoke test ----------------
SMOKE=$OUT/smoke_beta_vae_seed0
if [ -e "$SMOKE" ]; then echo "refusing: $SMOKE already exists"; exit 1; fi
mkdir -p "$SMOKE"; write_env "$SMOKE"
printf "step\texit\tseconds\n" > "$SMOKE/step_times.tsv"
ok=1
run_step "$SMOKE" train "$ENV/python" "$ENV/dlib_train" \
    --gin_config="$DLIB/examples/model.gin" --model_dir="$SMOKE/model" || ok=0
if [ $ok -eq 1 ]; then
  run_step "$SMOKE" reproduce "$ENV/python" "$ENV/dlib_reproduce" \
      --model_dir="$SMOKE/model" --output_directory="$SMOKE" || ok=0
fi
if [ $ok -eq 1 ]; then touch "$SMOKE/DONE"; else touch "$SMOKE/FAILED"; fi

# ---------------- Part 2: timing run ----------------
TIMING=$OUT/timing_beta_vae_seed0_2000steps
if [ -e "$TIMING" ]; then echo "refusing: $TIMING already exists"; exit 1; fi
mkdir -p "$TIMING"; write_env "$TIMING"
printf "step\texit\tseconds\n" > "$TIMING/step_times.tsv"
if run_step "$TIMING" train_2000 "$ENV/python" "$ENV/dlib_train" \
    --gin_config="$DLIB/examples/model.gin" --gin_bindings="model.training_steps = 2000" \
    --model_dir="$TIMING/model"; then
  touch "$TIMING/DONE"
else
  touch "$TIMING/FAILED"
fi

echo "[$(date '+%F %T')] finished"
