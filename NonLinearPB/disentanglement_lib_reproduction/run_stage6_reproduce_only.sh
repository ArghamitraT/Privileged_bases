#!/bin/bash
# Step 6 (re-run) — run dlib_reproduce on an already-trained model directory
# (visualize -> postprocess mean/sampled -> 18 metrics), into a NEW output directory.
#
# Conda environment: dlib-tf1-cpu
#
# Usage (detached):
#   setsid nohup bash disentanglement_lib_reproduction/run_stage6_reproduce_only.sh MODEL_DIR OUT_DIR \
#       > reproduction_outputs/<name>_runner.log 2>&1 < /dev/null &
# Example:
#   ... run_stage6_reproduce_only.sh reproduction_outputs/smoke_beta_vae_seed0/model \
#                                    reproduction_outputs/smoke_beta_vae_seed0_sklearn020
#
# Writes command.txt, environment.txt, stdout.log, step_times.tsv and DONE/FAILED in OUT_DIR.

NLPB=/home/argha/Mat_embedding/code/NonLinearPB
ENV=/home/argha/miniconda3/envs/dlib-tf1-cpu/bin
DLIB=$NLPB/disentanglement_lib
export DISENTANGLEMENT_LIB_DATA=/home/argha/Mat_embedding/data
export PATH=$ENV:$PATH
export CUDA_VISIBLE_DEVICES=""

MODEL_DIR=$(readlink -f "$1"); OUT_DIR=$(readlink -m "$2")
if [ ! -d "$MODEL_DIR/tfhub" ]; then echo "no trained model at $MODEL_DIR"; exit 1; fi
if [ -e "$OUT_DIR" ]; then echo "refusing: $OUT_DIR already exists"; exit 1; fi
mkdir -p "$OUT_DIR"

{
  echo "date: $(date '+%F %T')"; echo "repo commit: $(git -C "$DLIB" rev-parse HEAD)"
  echo "conda env: dlib-tf1-cpu"; echo "model_dir: $MODEL_DIR"
  echo "DISENTANGLEMENT_LIB_DATA=$DISENTANGLEMENT_LIB_DATA"
  "$ENV/python" --version 2>&1; "$ENV/pip" freeze 2>/dev/null
} > "$OUT_DIR/environment.txt"

WORK=$NLPB/reproduction_outputs/workdir_stage6; mkdir -p "$WORK"; cd "$WORK" || exit 1
CMD=("$ENV/python" "$ENV/dlib_reproduce" --model_dir="$MODEL_DIR" --output_directory="$OUT_DIR")
echo "${CMD[*]}" > "$OUT_DIR/command.txt"
printf "step\texit\tseconds\n" > "$OUT_DIR/step_times.tsv"

t0=$(date +%s); echo "[$(date '+%F %T')] START reproduce"
"${CMD[@]}" > "$OUT_DIR/stdout.log" 2>&1; rc=$?
dt=$(( $(date +%s) - t0 )); echo "[$(date '+%F %T')] END   reproduce exit=$rc ${dt}s"
printf "reproduce\t%s\t%s\n" "$rc" "$dt" >> "$OUT_DIR/step_times.tsv"
if [ $rc -eq 0 ]; then touch "$OUT_DIR/DONE"; else touch "$OUT_DIR/FAILED"; fi
echo "[$(date '+%F %T')] finished"
