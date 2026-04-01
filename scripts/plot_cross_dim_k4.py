"""
Cross-experiment comparison: Linear and 1-NN accuracy at prefix k=1..4
for three trained models (embed_dim=8, 16, 64), showing Standard, L1, MRL,
and PCA for each.

Draws a 2-panel figure (Linear top, 1-NN bottom). Line color = model type;
line style = final embedding dimension. Legend notes both.

Inputs:
    Three experiment folders (hardcoded below):
        - exprmnt_2026_03_23__19_00_40  (embed_dim=8)
        - exprmnt_2026_03_23__19_34_15  (embed_dim=16)
        - exprmnt_2026_03_20__22_15_23/seed_42  (embed_dim=64)

Outputs:
    cross_dim_k4_comparison.png — saved into the script directory's parent
    figure/ folder.

Usage:
    python scripts/plot_cross_dim_k4.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA as SklearnPCA

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from config import ExpConfig
from data.loader import load_data
from models.encoder import MLPEncoder
from experiments.exp7_mrl_vs_ff import get_embeddings_np, evaluate_1nn

# ------------------------------------------------------------------
# Experiment registry: (folder_path, embed_dim)
# ------------------------------------------------------------------
RESULTS_ROOT = (
    "/Users/arghamitratalukder/Library/CloudStorage/"
    "GoogleDrive-at3836@columbia.edu/My Drive/technical_work/"
    "Mat_embedding_hyperbole/files/results"
)

EXPERIMENTS = [
    (os.path.join(RESULTS_ROOT, "exprmnt_2026_03_23__19_00_40"),          8),
    (os.path.join(RESULTS_ROOT, "exprmnt_2026_03_23__19_34_15"),         16),
    (os.path.join(RESULTS_ROOT, "exprmnt_2026_03_20__22_15_23", "seed_42"), 64),
]

# Only evaluate these prefix sizes
EVAL_KS  = [1, 2, 3, 4]
MAX_DB   = 10_000   # cap for 1-NN database speed
HIDDEN   = 256      # hidden_dim matches all three experiments

# ------------------------------------------------------------------
# Load MNIST once (same data for all experiments)
# ------------------------------------------------------------------
print("[cross_dim] Loading MNIST ...")
cfg_base = ExpConfig(dataset="mnist", embed_dim=64, hidden_dim=HIDDEN,
                     head_mode="shared_head", seed=42, data_seed=42)
data = load_data(cfg_base)

y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_encoder(folder, tag, embed_dim):
    """Load a saved MLPEncoder from a result folder.

    Args:
        folder    (str): path to the experiment output folder.
        tag       (str): model tag — 'standard', 'l1', or 'mat'.
        embed_dim (int): final embedding dimension the model was trained with.

    Returns:
        MLPEncoder: loaded model in eval mode.
    """
    enc = MLPEncoder(data.input_dim, HIDDEN, embed_dim)
    path = os.path.join(folder, f"{tag}_encoder_best.pt")
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    return enc


def evaluate_linear(Z_tr, Z_te, y_tr, y_te, max_train=MAX_DB, seed=42):
    """Logistic regression accuracy on k-dim prefix embeddings.

    Args:
        Z_tr      (np.ndarray): train embeddings [N, k].
        Z_te      (np.ndarray): test  embeddings [M, k].
        y_tr      (np.ndarray): train labels [N].
        y_te      (np.ndarray): test  labels [M].
        max_train (int): subsample training set to this size for speed.
        seed      (int): random seed for subsampling.

    Returns:
        float: top-1 classification accuracy on the test set.
    """
    rng = np.random.default_rng(seed)
    if len(Z_tr) > max_train:
        idx = rng.choice(len(Z_tr), max_train, replace=False)
        Z_tr, y_tr = Z_tr[idx], y_tr[idx]
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(Z_tr, y_tr)
    return float(lr.score(Z_te, y_te))


# ------------------------------------------------------------------
# Sweep all experiments
# ------------------------------------------------------------------
# results[embed_dim][model_name][k] = accuracy
lin_results = {}
nn_results  = {}

for folder, embed_dim in EXPERIMENTS:
    print(f"\n[cross_dim] === embed_dim={embed_dim} | {os.path.basename(folder)} ===")
    lin_results[embed_dim] = {}
    nn_results[embed_dim]  = {}

    # --- Neural encoder embeddings ---
    for tag, display in [("standard", "Standard"), ("l1", "L1"), ("mat", "MRL")]:
        print(f"  Loading {display} encoder ...")
        enc = load_encoder(folder, tag, embed_dim)
        Z_tr = get_embeddings_np(enc, data.X_train)
        Z_te = get_embeddings_np(enc, data.X_test)
        lin_results[embed_dim][display] = {}
        nn_results[embed_dim][display]  = {}
        for k in EVAL_KS:
            la = evaluate_linear(Z_tr[:, :k], Z_te[:, :k], y_train_np, y_test_np)
            na = evaluate_1nn(Z_tr[:, :k], Z_te[:, :k], y_train_np, y_test_np,
                              max_db_samples=MAX_DB, seed=42)
            lin_results[embed_dim][display][k] = la
            nn_results[embed_dim][display][k]  = na
            print(f"    k={k}  lin={la:.4f}  1-NN={na:.4f}")

    # --- PCA baseline (fit on pixels, not on encoder output) ---
    print(f"  Computing PCA (n_components={max(EVAL_KS)}) ...")
    pca = SklearnPCA(n_components=max(EVAL_KS), random_state=42)
    Z_tr_pca = pca.fit_transform(X_train_np)
    Z_te_pca = pca.transform(X_test_np)
    lin_results[embed_dim]["PCA"] = {}
    nn_results[embed_dim]["PCA"]  = {}
    for k in EVAL_KS:
        la = evaluate_linear(Z_tr_pca[:, :k], Z_te_pca[:, :k], y_train_np, y_test_np)
        na = evaluate_1nn(Z_tr_pca[:, :k], Z_te_pca[:, :k], y_train_np, y_test_np,
                          max_db_samples=MAX_DB, seed=42)
        lin_results[embed_dim]["PCA"][k] = la
        nn_results[embed_dim]["PCA"][k]  = na
        print(f"    PCA k={k}  lin={la:.4f}  1-NN={na:.4f}")

# ------------------------------------------------------------------
# Plot — 3 rows × 2 cols
#   rows (top→bottom): MRL · Standard · L1
#   cols (left→right): Linear accuracy · 1-NN accuracy
#   lines inside each subplot: one per embed_dim (8, 16, 64)
#   Y-axis limits set tight from each subplot's own data.
# ------------------------------------------------------------------
ROW_MODELS  = ["MRL", "Standard", "L1"]   # top to bottom
DIMS        = sorted(lin_results.keys())   # [8, 16, 64]

DIM_COLORS  = {8: "royalblue", 16: "darkorange", 64: "seagreen"}
DIM_MARKERS = {8: "o",         16: "s",          64: "D"}

def tight_ylim(values, pad=0.08):
    """Return (ymin, ymax) tight around a list of accuracy values.

    Args:
        values (list[float]): accuracy values in this subplot.
        pad    (float): fractional padding applied to the data span.

    Returns:
        tuple: (ymin, ymax) for ax.set_ylim.
    """
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 0.05
    return max(0.0, lo - pad * span), min(1.02, hi + pad * span)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(11, 10))

fig.suptitle(
    "MRL vs Standard vs L1 — First 4 Dimensions\n"
    "(lines = final embed_dim: 8 / 16 / 64)",
    fontsize=13, fontweight="bold"
)

METRICS = [
    (lin_results, "Linear Accuracy",  "Linear Classification Accuracy"),
    (nn_results,  "1-NN Accuracy",    "1-NN Accuracy"),
]

for row, model_name in enumerate(ROW_MODELS):
    for col, (results_dict, ylabel, col_title) in enumerate(METRICS):
        ax = axes[row, col]

        # Collect values for this subplot to compute tight y limits
        subplot_vals = [
            results_dict[dim][model_name][k]
            for dim in DIMS
            for k in EVAL_KS
        ]
        ax.set_ylim(tight_ylim(subplot_vals))

        # One line per embed_dim
        for dim in DIMS:
            accs  = [results_dict[dim][model_name][k] for k in EVAL_KS]
            ax.plot(EVAL_KS, accs,
                    marker=DIM_MARKERS[dim],
                    color=DIM_COLORS[dim],
                    linewidth=2, markersize=6,
                    label=f"dim={dim}")

        # Row label (model name) on left column only
        ax.set_ylabel(
            f"{model_name}\n{ylabel}" if col == 0 else ylabel,
            fontsize=10
        )

        # Column title on top row only
        if row == 0:
            ax.set_title(col_title, fontsize=11)

        # x-axis label on bottom row only
        if row == len(ROW_MODELS) - 1:
            ax.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=10)

        ax.set_xticks(EVAL_KS)
        ax.legend(fontsize=9, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(CODE_DIR, "figure", "cross_dim_k4_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[cross_dim] Saved: {out_path}")
