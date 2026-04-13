#!/bin/bash
# Update mrl_env from env/mrl_env.yml after a git pull that changed the yml.
#
# Usage:
#   bash scripts/bash/update_env.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
YML="${REPO_ROOT}/env/mrl_env.yml"

echo "Updating mrl_env from: $YML"
conda env update -n mrl_env -f "$YML" --prune

echo ""
echo "Done. Activate with: conda activate mrl_env"
