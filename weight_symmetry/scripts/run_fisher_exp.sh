#!/usr/bin/env bash
# Run exp2_fisher_loss.py inside a tmux session.
#
# Usage:
#   bash weight_symmetry/scripts/run_fisher_exp.sh                          # full run, all models
#   bash weight_symmetry/scripts/run_fisher_exp.sh --fast                   # smoke test
#   bash weight_symmetry/scripts/run_fisher_exp.sh --models fp_fisher       # single model
#   bash weight_symmetry/scripts/run_fisher_exp.sh --models fisher fp_fisher prefix_l1_fisher
#
# The session is named "fisher". Attach with:  tmux attach -t fisher
# Tail logs without attaching:               tail -f /tmp/fisher_run.log

SESSION="fisher"
CODE_DIR="/home/argha/Mat_embedding_hyperbole/code"
LOG_FILE="/tmp/fisher_run.log"
SCRIPT="weight_symmetry/experiments/exp2_fisher_loss.py"
EXTRA_ARGS="$@"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null

# Build the command that will run inside tmux
RUN_CMD="cd $CODE_DIR"
RUN_CMD="$RUN_CMD && conda run -n mrl_env_cuda12 python $SCRIPT $EXTRA_ARGS 2>&1 | tee $LOG_FILE"
RUN_CMD="$RUN_CMD; echo '=== DONE ==='"

# Start detached tmux session
tmux new-session \
    -d \
    -s "$SESSION" \
    "$RUN_CMD"

echo "Session '$SESSION' started."
echo "Attach  : tmux attach -t $SESSION"
echo "Tail log: tail -f $LOG_FILE"
