"""
One-off script: compute Linear and 1-NN accuracy vs prefix k=1..MAX_K for
Standard, L1, MRL, and PCA models saved in exp9 seed_42, and save a
two-panel combined figure into that same folder.

The models were trained with embed_dim=64; this script evaluates only the
first MAX_K dimensions and notes the full embed_dim in the figure title.

Usage:
    python scripts/plot_1nn_seed42.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA as SklearnPCA

CODE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED42_DIR = (
    "/Users/arghamitratalukder/Library/CloudStorage/"
    "GoogleDrive-at3836@columbia.edu/My Drive/technical_work/"
    "Mat_embedding_hyperbole/files/results/"
    "exprmnt_2026_03_20__22_15_23/seed_42"
)
sys.path.insert(0, CODE_DIR)

from config import ExpConfig
from data.loader import load_data
from models.encoder import MLPEncoder
from experiments.exp7_mrl_vs_ff import get_embeddings_np, evaluate_1nn

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
MAX_K      = 16   # evaluate prefix k=1..MAX_K
FULL_DIM   = 64   # actual embedding dim the models were trained with
MAX_DB     = 10_000  # cap database size for 1-NN speed

# ------------------------------------------------------------------
# Config — must match what exp9 used for seed_42
# ------------------------------------------------------------------
cfg = ExpConfig(
    dataset       = "mnist",
    embed_dim     = FULL_DIM,
    hidden_dim    = 256,
    head_mode     = "shared_head",
    eval_prefixes = list(range(1, MAX_K + 1)),
    epochs        = 20,
    seed          = 42,
    data_seed     = 42,
)

print("[plot_1nn] Loading MNIST (data_seed=42) ...")
data = load_data(cfg)

y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

# ------------------------------------------------------------------
# Load encoder weights
# ------------------------------------------------------------------
def load_encoder(tag):
    """Load a saved MLPEncoder from SEED42_DIR.

    Args:
        tag (str): model tag, e.g. 'standard', 'l1', 'mat'.

    Returns:
        MLPEncoder: loaded model in eval mode.
    """
    enc = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    enc.load_state_dict(torch.load(
        os.path.join(SEED42_DIR, f"{tag}_encoder_best.pt"), map_location="cpu"
    ))
    enc.eval()
    return enc

print("[plot_1nn] Loading model weights ...")
std_enc = load_encoder("standard")
l1_enc  = load_encoder("l1")
mat_enc = load_encoder("mat")

# ------------------------------------------------------------------
# Extract full embeddings once, then slice per k
# ------------------------------------------------------------------
print("[plot_1nn] Extracting embeddings ...")
Z_train = {
    "Standard": get_embeddings_np(std_enc, data.X_train),
    "L1":       get_embeddings_np(l1_enc,  data.X_train),
    "MRL":      get_embeddings_np(mat_enc, data.X_train),
}
Z_test = {
    "Standard": get_embeddings_np(std_enc, data.X_test),
    "L1":       get_embeddings_np(l1_enc,  data.X_test),
    "MRL":      get_embeddings_np(mat_enc, data.X_test),
}

# PCA: fit on raw pixels, keep MAX_K components
X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)
pca = SklearnPCA(n_components=MAX_K, random_state=42)
Z_train["PCA"] = pca.fit_transform(X_train_np)
Z_test["PCA"]  = pca.transform(X_test_np)

print(f"[plot_1nn] Embeddings ready. Shapes: "
      f"train={Z_train['Standard'].shape}, test={Z_test['Standard'].shape}")

# ------------------------------------------------------------------
# Helper: linear probe accuracy at prefix k (subsampled for speed)
# ------------------------------------------------------------------
def evaluate_linear(Z_tr, Z_te, y_tr, y_te, max_train=10_000, seed=42):
    """Fit a logistic regression probe on the first k dims and return accuracy.

    Args:
        Z_tr   (np.ndarray): train embeddings [N, k].
        Z_te   (np.ndarray): test  embeddings [M, k].
        y_tr   (np.ndarray): train labels [N].
        y_te   (np.ndarray): test  labels [M].
        max_train (int): cap on training samples for speed.
        seed   (int): random seed for subsampling.

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
# Sweep k=1..MAX_K for all models
# ------------------------------------------------------------------
MODEL_NAMES = ["Standard", "L1", "MRL", "PCA"]
lin_results = {m: {} for m in MODEL_NAMES}
nn_results  = {m: {} for m in MODEL_NAMES}

for model_name in MODEL_NAMES:
    print(f"[plot_1nn] Sweeping k=1..{MAX_K} for {model_name} ...")
    for k in range(1, MAX_K + 1):
        Ztr_k = Z_train[model_name][:, :k]
        Zte_k = Z_test[model_name][:, :k]

        lin_acc = evaluate_linear(Ztr_k, Zte_k, y_train_np, y_test_np,
                                  max_train=MAX_DB, seed=42)
        nn_acc  = evaluate_1nn(Ztr_k, Zte_k, y_train_np, y_test_np,
                               max_db_samples=MAX_DB, seed=42)

        lin_results[model_name][k] = lin_acc
        nn_results[model_name][k]  = nn_acc
        print(f"  k={k:>2}  lin={lin_acc:.4f}  1-NN={nn_acc:.4f}")

# ------------------------------------------------------------------
# Plot — two-panel figure matching the exp10 style
# ------------------------------------------------------------------
MODEL_STYLES = {
    "Standard": ("steelblue",   "o-",   "Standard"),
    "L1":       ("orchid",      "D-",   "L1 (lambda=0.05)"),
    "MRL":      ("darkorange",  "s-",   "MRL"),
    "PCA":      ("seagreen",    "x--",  "PCA"),
}

ks = list(range(1, MAX_K + 1))

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

# Super-title mentions the full training dimension
fig.suptitle(
    f"Standard vs L1 vs MRL vs PCA — Dense Prefix Sweep\n"
    f"(models trained with embed_dim={FULL_DIM}; showing first {MAX_K} dims)",
    fontsize=13, fontweight="bold"
)

# --- Top panel: Linear accuracy ---
ax_lin = axes[0]
for model_name in MODEL_NAMES:
    color, style, label = MODEL_STYLES[model_name]
    accs = [lin_results[model_name][k] for k in ks]
    ax_lin.plot(ks, accs, style, color=color, label=label,
                linewidth=2, markersize=5)
ax_lin.set_ylabel("Linear Accuracy", fontsize=12)
ax_lin.set_ylim(0, 1.05)
ax_lin.set_title("Linear Classification Accuracy vs Prefix k  (dense)", fontsize=11)
ax_lin.legend(fontsize=10, loc="lower right")

# --- Bottom panel: 1-NN accuracy ---
ax_nn = axes[1]
for model_name in MODEL_NAMES:
    color, style, label = MODEL_STYLES[model_name]
    accs = [nn_results[model_name][k] for k in ks]
    ax_nn.plot(ks, accs, style, color=color, label=label,
               linewidth=2, markersize=5)
ax_nn.set_ylabel("1-NN Accuracy", fontsize=12)
ax_nn.set_ylim(0, 1.05)
ax_nn.set_title("1-NN Accuracy vs Prefix k  (dense)", fontsize=11)
ax_nn.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
ax_nn.set_xticks(ks)
ax_nn.legend(fontsize=10, loc="lower right")

plt.tight_layout()

out_path = os.path.join(SEED42_DIR, f"combined_comparison_k{MAX_K}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[plot_1nn] Saved: {out_path}")
