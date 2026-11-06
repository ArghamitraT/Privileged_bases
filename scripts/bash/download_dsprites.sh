#!/bin/bash
# Download the data used by NonLinearPB (dSprites + the Scream image) and verify checksums.
#
# Usage:
#   bash scripts/bash/download_dsprites.sh
#
# Data is saved to: <project root>/data/{dsprites,scream}/
# <project root> is the folder that contains code/ (/home/argha/Mat_embedding on aeolus,
# /mnt/home/at3836/Mat_embedding on Empire AI). The NonLinearPB code finds the archive there
# by default (vae_reproduction/runs.py: PROJECT_ROOT / "data/dsprites/...").
#
# See NonLinearPB/DATA_SETUP.md for the folder layout and archive contents.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$(dirname "$REPO_ROOT")/data"

DSPRITES_FILE="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_URL="https://github.com/deepmind/dsprites-dataset/raw/master/${DSPRITES_FILE}"
DSPRITES_SHA="857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2"

SCREAM_URL="https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg"
SCREAM_SHA="a9bd171fe0a8da87f90a467e30542be195fb2e2a46d650f5287421869e57d390"

echo "Data directory: $DATA"
mkdir -p "$DATA/dsprites" "$DATA/scream"

echo ""
echo "Downloading dSprites (~19 MB)..."
[ -f "$DATA/dsprites/$DSPRITES_FILE" ] || wget -O "$DATA/dsprites/$DSPRITES_FILE" "$DSPRITES_URL"

echo ""
echo "Downloading Scream image (~4 MB, only needed for scream_dsprites in disentanglement_lib)..."
[ -f "$DATA/scream/scream.jpg" ] || wget -O "$DATA/scream/scream.jpg" "$SCREAM_URL"

echo ""
echo "Verifying checksums..."
echo "${DSPRITES_SHA}  $DATA/dsprites/$DSPRITES_FILE" | sha256sum -c -
echo "${SCREAM_SHA}  $DATA/scream/scream.jpg" | sha256sum -c - \
    || echo "WARNING: scream.jpg differs from the aeolus copy (Wikimedia may have changed it). dSprites is unaffected."

echo ""
echo "Done."
