"""
L1 model: evaluate accuracy using the best-k dimensions (k=1..4) selected
by three importance methods: mean-abs, variance, and per-dim linear probe.

Layout: 3 rows × 2 cols
  rows (top→bottom): best-k by mean_abs · variance · linear_probe
  cols (left→right): Linear (top-1) accuracy · 1-NN accuracy
  lines inside each subplot: one per final embed_dim (8, 16, 64)

Y-axis is set tight from each subplot's own data.

Inputs:
    Three experiment folders (L1 encoder weights only):
        - exprmnt_2026_03_23__19_00_40  (embed_dim=8)
        - exprmnt_2026_03_23__19_34_15  (embed_dim=16)
        - exprmnt_2026_03_20__22_15_23/seed_42  (embed_dim=64)

Output:
    code/figure/l1_best_dims_k4.png

Usage:
    python scripts/plot_l1_best_dims.py
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
    (os.path.join(RESULTS_ROOT, "exprmnt_2026_03_23__19_00_40"),             8),
    (os.path.join(RESULTS_ROOT, "exprmnt_2026_03_23__19_34_15"),            16),
    (os.path.join(RESULTS_ROOT, "exprmnt_2026_03_20__22_15_23", "seed_42"), 64),
]

EVAL_KS = [1, 2, 3, 4]
MAX_DB  = 10_000
HIDDEN  = 256

# ------------------------------------------------------------------
# Load MNIST once
# ------------------------------------------------------------------
print("[l1_best] Loading MNIST ...")
cfg_base = ExpConfig(dataset="mnist", embed_dim=64, hidden_dim=HIDDEN,
                     head_mode="shared_head", seed=42, data_seed=42)
data = load_data(cfg_base)

y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_l1_encoder(folder, embed_dim):
    """Load the L1 encoder from a result folder.

    Args:
        folder    (str): path to the experiment output folder.
        embed_dim (int): final embedding dimension.

    Returns:
        MLPEncoder: loaded model in eval mode.
    """
    enc = MLPEncoder(data.input_dim, HIDDEN, embed_dim)
    path = os.path.join(folder, "l1_encoder_best.pt")
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    return enc


def importance_mean_abs(Z_train):
    """Per-dim importance by mean absolute activation on training set.

    Args:
        Z_train (np.ndarray): train embeddings [N, D].

    Returns:
        np.ndarray: importance score per dimension, shape [D].
    """
    return np.mean(np.abs(Z_train), axis=0)


def importance_variance(Z_train):
    """Per-dim importance by variance on training set.

    Args:
        Z_train (np.ndarray): train embeddings [N, D].

    Returns:
        np.ndarray: importance score per dimension, shape [D].
    """
    return np.var(Z_train, axis=0)


def importance_linear_probe(Z_train, y_train, max_train=5_000, seed=42):
    """Per-dim importance by single-dimension logistic regression accuracy.

    Fits an independent LR probe on each dimension separately and returns
    the accuracy as the importance score.

    Args:
        Z_train   (np.ndarray): train embeddings [N, D].
        y_train   (np.ndarray): train labels [N].
        max_train (int): subsample cap for speed.
        seed      (int): random seed for subsampling.

    Returns:
        np.ndarray: per-dimension accuracy, shape [D].
    """
    rng = np.random.default_rng(seed)
    if len(Z_train) > max_train:
        idx = rng.choice(len(Z_train), max_train, replace=False)
        Z_train = Z_train[idx]
        y_train = y_train[idx]
    D = Z_train.shape[1]
    scores = np.zeros(D)
    for d in range(D):
        lr = LogisticRegression(max_iter=500, random_state=seed)
        lr.fit(Z_train[:, d:d+1], y_train)
        scores[d] = lr.score(Z_train[:, d:d+1], y_train)
    return scores


def evaluate_linear(Z_tr, Z_te, y_tr, y_te, max_train=MAX_DB, seed=42):
    """Logistic regression accuracy on selected columns.

    Args:
        Z_tr      (np.ndarray): train embeddings [N, k].
        Z_te      (np.ndarray): test  embeddings [M, k].
        y_tr      (np.ndarray): train labels [N].
        y_te      (np.ndarray): test  labels [M].
        max_train (int): subsample training set to this size.
        seed      (int): random seed for subsampling.

    Returns:
        float: top-1 accuracy on the test set.
    """
    rng = np.random.default_rng(seed)
    if len(Z_tr) > max_train:
        idx = rng.choice(len(Z_tr), max_train, replace=False)
        Z_tr, y_tr = Z_tr[idx], y_tr[idx]
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(Z_tr, y_tr)
    return float(lr.score(Z_te, y_te))


# ------------------------------------------------------------------
# Importance methods registry
# ------------------------------------------------------------------
METHODS = [
    ("mean_abs",     "Best-k by Mean |activation|"),
    ("variance",     "Best-k by Variance"),
    ("linear_probe", "Best-k by Linear Probe"),
]

# results[method_key][embed_dim][metric][k] = accuracy
results = {m: {} for m, _ in METHODS}

DIMS = [exp[1] for exp in EXPERIMENTS]

for folder, embed_dim in EXPERIMENTS:
    print(f"\n[l1_best] === embed_dim={embed_dim} ===")
    enc   = load_l1_encoder(folder, embed_dim)
    Z_tr  = get_embeddings_np(enc, data.X_train)
    Z_te  = get_embeddings_np(enc, data.X_test)

    # Compute all importance scores once
    print(f"  Computing importance scores ...")
    scores = {
        "mean_abs":     importance_mean_abs(Z_tr),
        "variance":     importance_variance(Z_tr),
        "linear_probe": importance_linear_probe(Z_tr, y_train_np),
    }

    for method_key, _ in METHODS:
        imp       = scores[method_key]
        # Sorted indices: most important first
        ranked    = np.argsort(imp)[::-1]
        results[method_key][embed_dim] = {"linear": {}, "1nn": {}}

        for k in EVAL_KS:
            top_k_idx = ranked[:k]              # best-k dimension indices
            Z_tr_k    = Z_tr[:, top_k_idx]
            Z_te_k    = Z_te[:, top_k_idx]

            la = evaluate_linear(Z_tr_k, Z_te_k, y_train_np, y_test_np)
            na = evaluate_1nn(Z_tr_k, Z_te_k, y_train_np, y_test_np,
                              max_db_samples=MAX_DB, seed=42)

            results[method_key][embed_dim]["linear"][k] = la
            results[method_key][embed_dim]["1nn"][k]    = na
            print(f"  [{method_key}] k={k}  lin={la:.4f}  1-NN={na:.4f}")

# ------------------------------------------------------------------
# Plot — 3 rows × 2 cols
# ------------------------------------------------------------------
DIM_COLORS  = {8: "royalblue", 16: "darkorange", 64: "seagreen"}
DIM_MARKERS = {8: "o",         16: "s",          64: "D"}

def tight_ylim(values, pad=0.08):
    """Return (ymin, ymax) tight around accuracy values with padding.

    Args:
        values (list[float]): all accuracy values in this subplot.
        pad    (float): fractional padding on the data span.

    Returns:
        tuple: (ymin, ymax).
    """
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 0.05
    return max(0.0, lo - pad * span), min(1.02, hi + pad * span)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(11, 10))

fig.suptitle(
    "L1 Model — Best-k Dimensions (k=1..4)\n"
    "Rows: selection method · Cols: metric · Lines: final embed_dim",
    fontsize=13, fontweight="bold"
)

METRIC_KEYS   = ["linear", "1nn"]
METRIC_LABELS = ["Linear Accuracy", "1-NN Accuracy"]
METRIC_TITLES = ["Linear Classification Accuracy", "1-NN Accuracy"]

for row, (method_key, method_label) in enumerate(METHODS):
    for col, (metric_key, ylabel, col_title) in enumerate(
            zip(METRIC_KEYS, METRIC_LABELS, METRIC_TITLES)):
        ax = axes[row, col]

        # Tight y limits from this subplot's data
        subplot_vals = [
            results[method_key][dim][metric_key][k]
            for dim in DIMS
            for k in EVAL_KS
        ]
        ax.set_ylim(tight_ylim(subplot_vals))

        # One line per embed_dim
        for dim in DIMS:
            accs = [results[method_key][dim][metric_key][k] for k in EVAL_KS]
            ax.plot(EVAL_KS, accs,
                    marker=DIM_MARKERS[dim],
                    color=DIM_COLORS[dim],
                    linewidth=2, markersize=6,
                    label=f"dim={dim}")

        # Row label on left column only
        ax.set_ylabel(
            f"{method_label}\n{ylabel}" if col == 0 else ylabel,
            fontsize=10
        )

        # Column title on top row only
        if row == 0:
            ax.set_title(col_title, fontsize=11)

        # x-label on bottom row only
        if row == len(METHODS) - 1:
            ax.set_xlabel("k  (number of dimensions used)", fontsize=10)

        ax.set_xticks(EVAL_KS)
        ax.legend(fontsize=9, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(CODE_DIR, "figure", "l1_best_dims_k4.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[l1_best] Saved: {out_path}")
