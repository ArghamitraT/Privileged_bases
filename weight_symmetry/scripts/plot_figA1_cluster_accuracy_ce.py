"""
Script: weight_symmetry/scripts/plot_figA1_cluster_accuracy_ce.py
------------------------------------------------------------------
Appendix Figure A1 — Cluster visualisation + classification accuracy for CE models.

For each of 4 CE models × 3 prefix sizes k ∈ {2, 4, 8}:
  - t-SNE scatter of X_test embeddings (first k dims), coloured by class
  - Linear-probe accuracy (trained on X_train embeddings, scored on X_test)

Layout: 3 rows (k=2, 4, 8) × 4 cols (Normal CE, MRL CE, FP MRL CE, PrefixL1 CE)

Data : orderedBoth synthetic, seed=42 — X_test / y_test (2 000 points, 20 classes)
Weights: exprmnt_2026_04_19__16_00_49/seed42_<tag>_best.pt

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/plot_figA1_cluster_accuracy_ce.py
    python weight_symmetry/scripts/plot_figA1_cluster_accuracy_ce.py \\
        --weights-dir /abs/path/to/exprmnt_folder
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
from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_ROOT = os.path.join(_PROJ_ROOT, "files", "results")
DEFAULT_WEIGHTS_DIR  = os.path.join(
    DEFAULT_RESULTS_ROOT, "exprmnt_2026_04_19__16_00_49"
)
FIGURES_DIR = os.path.join(
    DEFAULT_RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "figures"
)

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
# CE model definitions
# ---------------------------------------------------------------------------
CE_MODELS = [
    dict(tag="normal_ce",    label="Normal CE",     flip=False),
    dict(tag="std_mrl_ce",   label="MRL (CE)",      flip=False),
    dict(tag="fp_mrl_ce",    label="FP MRL (CE)",   flip=False),
    dict(tag="prefix_l1_ce", label="PrefixL1 (CE)", flip=True),
]

PREFIX_SIZES = [2, 4, 8, 16]

# 20-class colormap — consistent across all panels
CMAP = plt.get_cmap("tab20")
CLASS_COLORS = [CMAP(i) for i in range(20)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_embeddings(weights_path: str, X: np.ndarray,
                       input_dim: int, embed_dim: int, n_classes: int) -> np.ndarray:
    """Load model weights, pass X through encoder, return numpy embeddings."""
    model = LinearAEWithHeads(input_dim=input_dim, embed_dim=embed_dim,
                              n_classes=n_classes)
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
    print(f"[figA1] Saved {stem}.{{pdf,png,svg}} → {out_dir}")
    return stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Fig A1 — CE cluster visualisation")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                        help="Folder containing seed42_<tag>_best.pt weight files")
    parser.add_argument("--out-dir", default=FIGURES_DIR,
                        help="Output directory for saved figures")
    args = parser.parse_args()

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("[figA1] Loading orderedBoth synthetic data (seed=42) ...")
    raw = load_synthetic(seed=42, ordered_lda=True, noise_scale_decay=0.9)
    X_train = raw["X_train"]   # (7000, 69)
    y_train = raw["y_train"]
    X_test  = raw["X_test"]    # (2000, 69)
    y_test  = raw["y_test"]

    input_dim = X_train.shape[1]   # 69
    embed_dim = 50
    n_classes = int(y_train.max()) + 1   # 20

    # -----------------------------------------------------------------------
    # Extract embeddings for all models
    # -----------------------------------------------------------------------
    print("[figA1] Extracting embeddings ...")
    embeddings_train = {}
    embeddings_test  = {}

    for mc in CE_MODELS:
        tag = mc["tag"]
        wpath = os.path.join(args.weights_dir, f"seed42_{tag}_best.pt")
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"Weight file not found: {wpath}")
        print(f"  loading {tag} ...")
        Z_tr = extract_embeddings(wpath, X_train, input_dim, embed_dim, n_classes)
        Z_te = extract_embeddings(wpath, X_test,  input_dim, embed_dim, n_classes)
        if mc["flip"]:
            Z_tr = np.ascontiguousarray(Z_tr[:, ::-1])
            Z_te = np.ascontiguousarray(Z_te[:, ::-1])
        embeddings_train[tag] = Z_tr
        embeddings_test[tag]  = Z_te

    # -----------------------------------------------------------------------
    # Compute accuracies + t-SNE for each (model, k)
    # -----------------------------------------------------------------------
    print("[figA1] Computing accuracies and running t-SNE ...")
    results = {}   # (tag, k) -> {"acc": float, "xy": (N,2)}

    for mc in CE_MODELS:
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
    n_rows = len(PREFIX_SIZES)     # 3
    n_cols = len(CE_MODELS)        # 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 8.5))

    for col_idx, mc in enumerate(CE_MODELS):
        tag = mc["tag"]
        for row_idx, k in enumerate(PREFIX_SIZES):
            ax = axes[row_idx][col_idx]
            xy  = results[(tag, k)]["xy"]
            acc = results[(tag, k)]["acc"]

            # scatter — one colour per class
            for c in range(n_classes):
                mask = y_test == c
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           color=CLASS_COLORS[c], s=2, alpha=0.5,
                           linewidths=0, rasterized=True)

            # column header (row 0 only)
            if row_idx == 0:
                ax.set_title(mc["label"], fontsize=9, pad=4)

            # accuracy subtitle
            ax.set_xlabel(f"Lin. acc: {acc*100:.1f}%",
                          fontsize=7, labelpad=2, style="italic")

            # no axis ticks (t-SNE axes not interpretable)
            ax.set_xticks([])
            ax.set_yticks([])

            # row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"$k = {k}$", fontsize=8, rotation=90, labelpad=4)

    fig.tight_layout(pad=0.8, h_pad=0.6, w_pad=0.5)
    save_fig(fig, "figA1_cluster_ce", fig_stamp, args.out_dir)
    plt.close(fig)
    print("[figA1] Done.")


if __name__ == "__main__":
    main()
