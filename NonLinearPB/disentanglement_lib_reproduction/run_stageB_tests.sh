#!/bin/bash
# Milestone B — run every test module listed in disentanglement_lib/bin/dlib_tests
# individually (dlib_tests uses `set -e`, so one failure would hide the rest).
#
# Conda environment: dlib-tf1-cpu
#
# Usage (launch detached so it survives disconnects):
#   setsid nohup bash disentanglement_lib_reproduction/run_stageB_tests.sh \
#       > reproduction_outputs/stageB_tests/runner.log 2>&1 &
#
# Outputs (reproduction_outputs/stageB_tests/):
#   <module>.log   stdout+stderr of each test module
#   summary.tsv    module <TAB> PASS/FAIL <TAB> exit code <TAB> seconds
#   DONE / FAILED  marker written at the end (FAILED if any module failed)

NLPB=/home/argha/Mat_embedding/code/NonLinearPB
PY=/home/argha/miniconda3/envs/dlib-tf1-cpu/bin/python
OUT=${OUT:-$NLPB/reproduction_outputs/stageB_tests}   # override with OUT=... for re-runs
export DISENTANGLEMENT_LIB_DATA=/home/argha/Mat_embedding/data
export PATH=/home/argha/miniconda3/envs/dlib-tf1-cpu/bin:$PATH
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/FAILED"
printf "module\tstatus\texit\tseconds\n" > "$OUT/summary.tsv"

# Tests write temp files to cwd in some cases — run from a scratch dir inside OUT.
mkdir -p "$OUT/workdir" && cd "$OUT/workdir" || exit 1

any_fail=0
for mod in $(grep '^python -m' "$NLPB/disentanglement_lib/bin/dlib_tests" | awk '{print $3}'); do
  start=$(date +%s)
  echo "[$(date '+%F %T')] START $mod"
  timeout 3600 "$PY" -m "$mod" > "$OUT/$mod.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then status=PASS; else status=FAIL; any_fail=1; fi
  printf "%s\t%s\t%s\t%s\n" "$mod" "$status" "$rc" "$secs" >> "$OUT/summary.tsv"
  echo "[$(date '+%F %T')] $status $mod (exit $rc, ${secs}s)"
done

if [ $any_fail -eq 0 ]; then touch "$OUT/DONE"; else touch "$OUT/FAILED"; fi
echo "[$(date '+%F %T')] finished"
