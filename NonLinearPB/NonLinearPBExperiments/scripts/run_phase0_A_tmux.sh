#!/usr/bin/env bash
# Conda environment: mrl_env_cuda12
# Usage (from code/):
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh phase0 --smoke
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh phase0 --device cuda:0
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh rotation --smoke
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh rotation --device cuda:1
# First argument picks Phase 0 (phase0) or Experiment A (rotation); the rest
# are passed to that entry point. Full runs start only without --smoke.
set -euo pipefail
PYTHON="/home/argha/miniconda3/envs/mrl_env_cuda12/bin/python"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
case "${1:-}" in
    phase0) ENTRY="$SCRIPT_DIR/../experiments/dsprites_phase0_audit.py" ;;
    rotation) ENTRY="$SCRIPT_DIR/../experiments/dsprites_rotation_basis.py" ;;
    *) printf 'First argument must be phase0 or rotation\n' >&2; exit 2 ;;
esac
KIND="$1"
shift
STAMP="$(date +%Y_%m_%d__%H_%M_%S)"
SESSION="dsprites-$KIND-$STAMP"
LOG_DIR="$PROJECT_ROOT/files/results/launch_logs/$SESSION"
command -v tmux >/dev/null
test -x "$PYTHON"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'Session already exists: %s\n' "$SESSION" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
arguments=("$PYTHON" -u "$ENTRY" "$@")
printf -v command_text '%q ' "${arguments[@]}"
printf -v log_quoted '%q' "$LOG_DIR/workflow.log"
printf -v shell_command 'bash -c %q' "set -o pipefail; $command_text 2>&1 | tee $log_quoted"
printf '%s\n' "$command_text" > "$LOG_DIR/command.txt"
tmux new-session -d -s "$SESSION" -n workflow -c "$CODE_ROOT" "$shell_command"
tmux set-window-option -t "$SESSION:workflow" remain-on-exit on
printf 'Session: %s\nLog: %s/workflow.log\nAttach: tmux attach -t %s\n' "$SESSION" "$LOG_DIR" "$SESSION"
