#!/bin/bash
# Create mrl_env from scratch using env/mrl_env.yml.
#
# Usage:
#   bash scripts/bash/create_env.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
YML="${REPO_ROOT}/env/mrl_env.yml"

echo "Creating mrl_env from: $YML"
conda env create -f "$YML"

echo ""
echo "Done. Activate with: conda activate mrl_env"
