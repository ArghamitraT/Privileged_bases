#!/bin/bash
# Create the seacells conda environment and install SEACells from the local clone.
#
# Usage:
#   bash scripts/bash/create_seacells_env.sh
#
# Requires: SEACells repo cloned at $HOME/Mat_embedding_hyperbole/model/SEACells
#
# After creation, activate with: conda activate seacells
#
# Lessons learned during setup:
#   - Python 3.9 required: jaxlib CPU wheels are not available for Python 3.8
#   - Skip requirements.txt: has overly strict old pins that break on Python 3.9+
#   - jax[cpu] must be installed BEFORE SEACells to satisfy jaxopt -> jaxlib chain
#   - ipython + ipywidgets required at runtime (SEACells imports IPython,
#     and tqdm uses notebook widgets even in terminal mode)

set -e

SEACELLS_DIR="${HOME}/Mat_embedding_hyperbole/model/SEACells"

if [ ! -d "$SEACELLS_DIR" ]; then
    echo "ERROR: SEACells repo not found at $SEACELLS_DIR"
    echo "Run: git clone https://github.com/dpeerlab/SEACells $SEACELLS_DIR"
    exit 1
fi

echo "============================================================"
echo "Creating seacells conda environment (Python 3.9)"
echo "============================================================"

conda create -y -n seacells python=3.9 -c conda-forge

eval "$(conda shell.bash hook)"
conda activate seacells

echo ""
echo "Installing conda-forge packages (umap-learn, numba need pre-built binaries)..."
conda install -y -c conda-forge umap-learn numba cmake cython

echo ""
echo "Pre-installing jax[cpu] (must come before SEACells to satisfy jaxopt -> jaxlib)..."
pip install "jax[cpu]"

echo ""
echo "Installing SEACells from local clone (skipping requirements.txt)..."
pip install "${SEACELLS_DIR}"

echo ""
echo "Installing runtime extras required by SEACells outside a notebook..."
pip install ipython ipywidgets

echo ""
echo "============================================================"
echo "Done. Activate with: conda activate seacells"
echo "Verify with: python -c \"import SEACells; print('SEACells OK')\""
echo "============================================================"
