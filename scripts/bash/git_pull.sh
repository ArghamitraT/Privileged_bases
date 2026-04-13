#!/bin/bash
# Git pull, then update mrl_env if the yml changed.
#
# Usage:
#   bash scripts/bash/git_pull.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
YML="${REPO_ROOT}/env/mrl_env.yml"

# snapshot yml hash before pull
yml_before=$(md5sum "$YML" 2>/dev/null | awk '{print $1}' || echo "none")

echo "============================================================"
echo "git_pull.sh"
echo "============================================================"
git -C "$REPO_ROOT" pull

# check if yml changed
yml_after=$(md5sum "$YML" 2>/dev/null | awk '{print $1}' || echo "none")

if [ "$yml_before" != "$yml_after" ]; then
    echo ""
    echo "mrl_env.yml changed — updating conda env..."
    conda env update -n mrl_env -f "$YML" --prune
    echo "Conda env updated."
else
    echo ""
    echo "mrl_env.yml unchanged — no env update needed."
fi

echo ""
echo "============================================================"
echo "Done."
echo "============================================================"
