#!/bin/bash
# Download PBMC 10k Multiome data from 10x Genomics
#
# Usage:
#   bash scripts/bash/download_pbmc.sh
#
# Data is saved to: $HOME/Mat_embedding_hyperbole/data/pbmc_10k_multiome/
# This path is consistent across clusters — only $HOME changes per cluster.
#
# NOTE: Verify download URLs from the 10x Genomics dataset page before running:
#   https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0
#
# After download, extract the analysis archive:
#   tar -xzf $HOME/Mat_embedding_hyperbole/data/pbmc_10k_multiome/pbmc_granulocyte_sorted_10k_analysis.tar.gz \
#       -C $HOME/Mat_embedding_hyperbole/data/pbmc_10k_multiome/

OUT="${HOME}/Mat_embedding_hyperbole/data/pbmc_10k_multiome"

echo "Home directory:  $HOME"
echo "Output directory: $OUT"

mkdir -p "$OUT"

BASE="https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k"

echo ""
echo "Downloading filtered feature barcode matrix (HDF5) (~192 MB)..."
wget -c -P "$OUT" "${BASE}/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"

echo ""
echo "Downloading secondary analysis outputs (~485 MB)..."
wget -c -P "$OUT" "${BASE}/pbmc_granulocyte_sorted_10k_analysis.tar.gz"

echo ""
echo "Done. Files in $OUT"
echo ""
echo "Next step — extract analysis outputs:"
echo "  tar -xzf ${OUT}/pbmc_granulocyte_sorted_10k_analysis.tar.gz -C ${OUT}/"
