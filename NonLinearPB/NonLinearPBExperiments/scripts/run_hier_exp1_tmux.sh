#!/usr/bin/env bash
# Conda environment: mrl_env_cuda12
# Usage (from code/):
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_hier_exp1_tmux.sh
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_hier_exp1_tmux.sh --fast
# Arguments pass through to run_hier_exp1.py (one queue, both GPUs).
set -euo pipefail
PYTHON="/home/argha/miniconda3/envs/mrl_env_cuda12/bin/python"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
STAMP="$(date +%Y_%m_%d__%H_%M_%S)"
SESSION="hier-exp1-$STAMP"
LOG_DIR="$PROJECT_ROOT/files/results/launch_logs/$SESSION"
command -v tmux >/dev/null
test -x "$PYTHON"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'Session already exists: %s\n' "$SESSION" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
arguments=("$PYTHON" -u "$SCRIPT_DIR/run_hier_exp1.py" "$@")
printf -v command_text '%q ' "${arguments[@]}"
printf -v log_quoted '%q' "$LOG_DIR/workflow.log"
printf -v shell_command 'bash -c %q' "set -o pipefail; $command_text 2>&1 | tee $log_quoted"
printf '%s\n' "$command_text" > "$LOG_DIR/command.txt"
tmux new-session -d -s "$SESSION" -n workflow -c "$CODE_ROOT" "$shell_command"
tmux set-window-option -t "$SESSION:workflow" remain-on-exit on
printf 'Session: %s\nLog: %s/workflow.log\nAttach: tmux attach -t %s\n' "$SESSION" "$LOG_DIR" "$SESSION"
