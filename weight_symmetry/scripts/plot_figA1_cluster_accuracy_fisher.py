"""
Script: weight_symmetry/scripts/plot_figA1_cluster_accuracy_fisher.py
----------------------------------------------------------------------
Appendix Figure A1B — Cluster visualisation + classification accuracy for Fisher models.

For each of 4 Fisher models × 4 prefix sizes k ∈ {2, 4, 8, 16}:
  - t-SNE scatter of X_test embeddings (first k dims), coloured by class
  - Linear-probe accuracy (trained on X_train embeddings, scored on X_test)

Layout: 4 rows (k=2, 4, 8, 16) × 4 cols (Fisher, MRL Fisher, FP Fisher, PrefixL1 Fisher)

Data   : orderedBoth synthetic, seed=47 — X_test / y_test (2 000 points, 20 classes)
Weights: see WEIGHTS_MAP below (two separate experiment folders)
Note   : embed_dim=19 (= n_lda = C-1); model is LinearAE (no classifier heads)

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/plot_figA1_cluster_accuracy_fisher.py
    python weight_symmetry/scripts/plot_figA1_cluster_accuracy_fisher.py \\
        --results-root /abs/path/to/files/results
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT   = os.path.dirname(_HERE)
_CODE_ROOT = os.path.dirname(_WS_ROOT)
_PROJ_ROOT = os.path.dirname(_CODE_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.data.synthetic import load_synthetic
from weight_symmetry.models.linear_ae import LinearAE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_ROOT = os.path.join(_PROJ_ROOT, "files", "results")
FIGURES_DIR = os.path.join(
    DEFAULT_RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "figures"
)

# model tag → (experiment folder, weight filename)
WEIGHTS_MAP = {
    "fisher":           ("exprmnt_2026_04_20__01_31_36", "seed47_fisher_best.pt"),
    "fp_fisher":        ("exprmnt_2026_04_20__01_44_24", "seed47_fp_fisher_best.pt"),
    "std_mrl_fisher":   ("exprmnt_2026_04_20__11_33_48", "seed47_std_mrl_fisher_best.pt"),
    "prefix_l1_fisher": ("exprmnt_2026_04_20__11_33_48", "seed47_prefix_l1_fisher_best.pt"),
}

# ---------------------------------------------------------------------------
# Global style (matches ICMLWorkshop_figure_style_plan.md)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# ---------------------------------------------------------------------------
# Fisher model definitions (column order)
# ---------------------------------------------------------------------------
FISHER_MODELS = [
    dict(tag="fisher",           label="Fisher",           flip=False),
    dict(tag="std_mrl_fisher",   label="MRL Fisher",       flip=False),
    dict(tag="fp_fisher",        label="FP Fisher",        flip=False),
    dict(tag="prefix_l1_fisher", label="PrefixL1 Fisher",  flip=True),
]

PREFIX_SIZES = [2, 4, 8, 16]

CMAP = plt.get_cmap("tab20")
CLASS_COLORS = [CMAP(i) for i in range(20)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_embeddings(weights_path: str, X: np.ndarray,
                       input_dim: int, embed_dim: int) -> np.ndarray:
    """Load LinearAE weights, pass X through encoder, return numpy embeddings."""
    model = LinearAE(input_dim=input_dim, embed_dim=embed_dim)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        Z = model.encoder(torch.tensor(X, dtype=torch.float32))
    return Z.numpy()


def linear_probe_accuracy(Z_train: np.ndarray, y_train: np.ndarray,
                           Z_test: np.ndarray,  y_test: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    clf.fit(Z_train, y_train)
    return float(clf.score(Z_test, y_test))


def run_tsne(Z: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    return tsne.fit_transform(Z)


def save_fig(fig, base_name: str, fig_stamp: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{base_name}_{fig_stamp}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    print(f"[figA1B] Saved {stem}.{{pdf,png,svg}} → {out_dir}")
    return stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Fig A1B — Fisher cluster visualisation")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                        help="Root folder containing experiment subfolders")
    parser.add_argument("--out-dir", default=FIGURES_DIR,
                        help="Output directory for saved figures")
    args = parser.parse_args()

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")

    # -----------------------------------------------------------------------
    # Load data (seed=47, matches Fisher training seed)
    # -----------------------------------------------------------------------
    print("[figA1B] Loading orderedBoth synthetic data (seed=47) ...")
    raw = load_synthetic(seed=47, ordered_lda=True, noise_scale_decay=0.9)
    X_train = raw["X_train"]   # (7000, 69)
    y_train = raw["y_train"]
    X_test  = raw["X_test"]    # (2000, 69)
    y_test  = raw["y_test"]

    input_dim = X_train.shape[1]   # 69
    embed_dim = 19                 # = n_lda = C-1
    n_classes = int(y_train.max()) + 1   # 20

    # -----------------------------------------------------------------------
    # Extract embeddings for all models
    # -----------------------------------------------------------------------
    print("[figA1B] Extracting embeddings ...")
    embeddings_train = {}
    embeddings_test  = {}

    for mc in FISHER_MODELS:
        tag = mc["tag"]
        exp_folder, fname = WEIGHTS_MAP[tag]
        wpath = os.path.join(args.results_root, exp_folder, fname)
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"Weight file not found: {wpath}")
        print(f"  loading {tag} from {exp_folder} ...")
        Z_tr = extract_embeddings(wpath, X_train, input_dim, embed_dim)
        Z_te = extract_embeddings(wpath, X_test,  input_dim, embed_dim)
        if mc["flip"]:
            Z_tr = np.ascontiguousarray(Z_tr[:, ::-1])
            Z_te = np.ascontiguousarray(Z_te[:, ::-1])
        embeddings_train[tag] = Z_tr
        embeddings_test[tag]  = Z_te

    # -----------------------------------------------------------------------
    # Compute accuracies + t-SNE for each (model, k)
    # -----------------------------------------------------------------------
    print("[figA1B] Computing accuracies and running t-SNE ...")
    results = {}   # (tag, k) -> {"acc": float, "xy": (N,2)}

    for mc in FISHER_MODELS:
        tag = mc["tag"]
        for k in PREFIX_SIZES:
            Z_tr_k = embeddings_train[tag][:, :k]
            Z_te_k = embeddings_test[tag][:, :k]
            acc    = linear_probe_accuracy(Z_tr_k, y_train, Z_te_k, y_test)
            print(f"  {tag}  k={k}  acc={acc:.3f}  running t-SNE ...")
            xy = run_tsne(Z_te_k)
            results[(tag, k)] = {"acc": acc, "xy": xy}

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    n_rows = len(PREFIX_SIZES)     # 4
    n_cols = len(FISHER_MODELS)    # 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 8.5))

    for col_idx, mc in enumerate(FISHER_MODELS):
        tag = mc["tag"]
        for row_idx, k in enumerate(PREFIX_SIZES):
            ax = axes[row_idx][col_idx]
            xy  = results[(tag, k)]["xy"]
            acc = results[(tag, k)]["acc"]

            for c in range(n_classes):
                mask = y_test == c
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           color=CLASS_COLORS[c], s=2, alpha=0.5,
                           linewidths=0, rasterized=True)

            if row_idx == 0:
                ax.set_title(mc["label"], fontsize=9, pad=4)

            ax.set_xlabel(f"Lin. acc: {acc*100:.1f}%",
                          fontsize=7, labelpad=2, style="italic")

            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(f"$k = {k}$", fontsize=8, rotation=90, labelpad=4)

    fig.tight_layout(pad=0.8, h_pad=0.6, w_pad=0.5)
    save_fig(fig, "figA1B_cluster_fisher", fig_stamp, args.out_dir)
    plt.close(fig)
    print("[figA1B] Done.")


if __name__ == "__main__":
    main()
