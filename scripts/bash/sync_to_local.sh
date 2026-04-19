#!/usr/bin/env bash
# Pull results from aeolus cluster to local Mac.
# Skips files that already exist on the destination.
# Run this FROM your Mac: bash sync_to_local.sh

# Name of the cluster as it appears in your Mac's SSH config (~/.ssh/config)
CLUSTER="aeolus"

# Full path to the folder/file you want to copy — on the cluster
REMOTE_SRC="/home/argha/Mat_embedding_hyperbole/files/results/ICMLWorkshop_weightSymmetry2026/figures"

# Full path to where it should land — on your Mac
LOCAL_DST="/Users/arghamitratalukder/Google Drive/My Drive/technical_work/Mat_embedding_hyperbole/files/results/ICMLWorkshop_weightSymmetry2026/figures"

rsync -avzP \
    --ignore-existing \
    "${CLUSTER}:${REMOTE_SRC}/" \
    "${LOCAL_DST}/"
