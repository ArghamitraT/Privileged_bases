"""
Helper subprocess for test_extract_embeddings in run_tests_exp13.py.

Imports torch and runs embedding extraction on synthetic data.
Kept as a subprocess so torch never enters the test runner process.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import numpy as np
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from models.encoder import MLPEncoder
from experiments.exp13_mrl_cd34_supervised import extract_embeddings

n_cells   = 100
n_hvg     = 50
embed_dim = 8

rng    = np.random.default_rng(0)
X_hvg  = rng.standard_normal((n_cells, n_hvg)).astype(np.float32)
X_pca  = rng.standard_normal((n_cells, embed_dim)).astype(np.float32)

# Build a trained (random-init) encoder
encoder = MLPEncoder(input_dim=n_hvg, hidden_dim=64, embed_dim=embed_dim)
encoder.eval()

trained_encoders = {
    "ce":             encoder,
    "mrl":            encoder,
    "fixed_lp":       encoder,
    "learned_lp":     encoder,
    "learned_lp_vec": encoder,
}
models_to_run = ["pca", "ce", "mrl", "fixed_lp", "learned_lp", "learned_lp_vec"]

embeddings = extract_embeddings(
    models_to_run, trained_encoders, X_hvg, X_pca, embed_dim
)

# --- shape check ---
for tag, Z in embeddings.items():
    assert Z.shape == (n_cells, embed_dim), (
        f"[{tag}] shape {Z.shape} != ({n_cells}, {embed_dim})"
    )
print("shape: OK")

# --- fixed_lp reversal check ---
# Extract raw encoder output (no reversal)
with torch.no_grad():
    Z_raw = encoder(torch.tensor(X_hvg)).numpy()

Z_flp = embeddings["fixed_lp"]
# After reversal, first dim of Z_flp should equal last dim of Z_raw
assert np.allclose(Z_flp[:, 0], Z_raw[:, -1], atol=1e-5), (
    "fixed_lp dim 0 should equal raw dim -1 after reversal"
)
assert np.allclose(Z_flp[:, -1], Z_raw[:, 0], atol=1e-5), (
    "fixed_lp dim -1 should equal raw dim 0 after reversal"
)
print("reversal: OK")

print("All helper_exp13_embed checks passed.")
