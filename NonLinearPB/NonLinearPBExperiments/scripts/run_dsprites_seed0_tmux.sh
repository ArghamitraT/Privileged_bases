#!/usr/bin/env bash
# Detached one-seed pilot: 100 VAE epochs, 2000 metric-classifier epoch cap.
# Conda environment: mrl_env_cuda12 (use its Python directly; activation unnecessary).
# Usage (from code/):
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_seed0_tmux.sh
#   tmux attach -t dsprites-seed0-100ep
#   bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_seed0_tmux.sh --resume-vae /path/to/checkpoint_epoch_10.pt
# Detach with Ctrl-b, then d. Switch model windows with Ctrl-b, then n.
# Existing sessions/results are preserved; an existing pilot session is an error.

set -euo pipefail

SESSION="dsprites-seed0-100ep"
PYTHON="/home/argha/miniconda3/envs/mrl_env_cuda12/bin/python"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
ENTRY="$CODE_ROOT/NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py"
LAUNCH_STAMP="$(date +%Y_%m_%d__%H_%M_%S)"
LOG_DIR="$PROJECT_ROOT/files/results/launch_logs/dsprites_seed0_$LAUNCH_STAMP"
MODE="${1:-launch}"
RESUME_CHECKPOINT=""

command -v tmux >/dev/null
test -x "$PYTHON"
if [[ "$MODE" == "--resume-vae" ]]; then
    RESUME_CHECKPOINT="${2:?Supply a VAE checkpoint path}"
    test -f "$RESUME_CHECKPOINT"
    tmux has-session -t "$SESSION"
    if [[ "$(tmux display-message -p -t "$SESSION:vae" '#{pane_dead}')" != "1" ]]; then
        printf 'The original VAE pane is still alive; no duplicate was started.\n' >&2
        exit 1
    fi
    if tmux list-windows -t "$SESSION" -F '#{window_name}' | rg -q '^vae-resumed$'; then
        printf 'A vae-resumed window already exists; inspect it before resuming again.\n' >&2
        exit 1
    fi
elif [[ "$MODE" != "launch" ]]; then
    printf 'Unknown mode: %s\n' "$MODE" >&2
    exit 1
elif tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'Session %s already exists. Attach to it; no new training was launched.\n' "$SESSION" >&2
    exit 1
fi
"$PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"'
mkdir -p "$LOG_DIR"

launch_window() {
    local name="$1" beta="$2" device="$3" kind="$4"
    local command_text log_quoted shell_command
    local -a arguments=("$PYTHON" -u "$ENTRY" --epochs 100 --seeds 0
                        --classifier-epochs 2000 --betas "$beta" --device "$device")
    # The random initialization is independent of beta: evaluate it once on GPU 0.
    if [[ "$beta" == "4" ]]; then
        arguments+=(--skip-untrained)
    fi
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        arguments+=(--skip-untrained --resume-checkpoint "$RESUME_CHECKPOINT")
    fi
    printf -v command_text '%q ' "${arguments[@]}"
    printf -v log_quoted '%q' "$LOG_DIR/$name.log"
    printf -v shell_command 'bash -c %q' "set -o pipefail; $command_text 2>&1 | tee $log_quoted"
    printf '%s\n' "$command_text" > "$LOG_DIR/$name.command.txt"
    if [[ "$kind" == "session" ]]; then
        tmux new-session -d -s "$SESSION" -n "$name" -c "$CODE_ROOT" "$shell_command"
    else
        tmux new-window -d -t "$SESSION" -n "$name" -c "$CODE_ROOT" "$shell_command"
    fi
    # Retain the pane and its exit status after the command finishes.
    tmux set-window-option -t "$SESSION:$name" remain-on-exit on
}

if [[ "$MODE" == "--resume-vae" ]]; then
    launch_window vae-resumed 1 cuda:0 window
    printf 'Started VAE recovery in %s:vae-resumed\nLogs: %s\n' "$SESSION" "$LOG_DIR"
    tmux list-panes -s -t "$SESSION" -F '#{window_name} dead=#{pane_dead} exit_status=#{pane_dead_status}'
    exit 0
fi

launch_window vae 1 cuda:0 session
# Wait for the first process to create its timestamped output directory, then
# cross a second boundary to avoid simultaneous timestamp-name collisions.
for ((attempt = 0; attempt < 30; attempt++)); do
    if [[ -f "$LOG_DIR/vae.log" ]] && rg -q '^Run directory:' "$LOG_DIR/vae.log"; then
        break
    fi
    sleep 1
done
if ! [[ -f "$LOG_DIR/vae.log" ]] || ! rg -q '^Run directory:' "$LOG_DIR/vae.log"; then
    printf 'VAE startup was not confirmed. Inspect %s; beta-VAE was not launched.\n' "$LOG_DIR/vae.log" >&2
    exit 1
fi
sleep 2
launch_window beta-vae 4 cuda:1 window

printf 'Started detached session: %s\nLogs: %s\n' "$SESSION" "$LOG_DIR"
printf 'Attach: tmux attach -t %s\n' "$SESSION"
tmux list-panes -s -t "$SESSION" -F '#{session_name}:#{window_name} command=#{pane_current_command} dead=#{pane_dead}'
