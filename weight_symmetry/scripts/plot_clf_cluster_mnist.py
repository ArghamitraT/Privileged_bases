"""
Script: weight_symmetry/scripts/plot_clf_cluster_mnist.py
---------------------------------------------------------
Cluster visualisation (UMAP) for the MNIST classification experiment
(`exprmnt_2026_04_22__15_40_00`).

For each of 2 models (FP-MRL-E, PrefixL1 α) × N prefix sizes k (default 2, 4, 6):
  - UMAP scatter of X_test embeddings (first k dims), coloured by class (0..9).
  - Linear-probe accuracy (fresh LR on train embeddings, scored on test).

Layout: N rows (one per k) × 2 cols (FP-MRL-E, PrefixL1).
Progress bar via tqdm over (model, k) pairs.

Encoder: weight_symmetry.experiments.classification.exp_clf.MLPEncoder
  MLP with BatchNorm + Dropout + L2-normalised output; input_dim=784, embed_dim=16.

Data: MNIST via weight_symmetry.data.loader.load_data("mnist", seed=42) — exact
same split used to train the models (seed=42 reproduces the 60k/10k → 60k×0.1 val).

Conda environment: mrl_env

Usage:
    python weight_symmetry/scripts/plot_clf_cluster_mnist.py
    python weight_symmetry/scripts/plot_clf_cluster_mnist.py \\
        --weights-dir /abs/path/to/exprmnt_2026_04_22__15_40_00 \\
        --l1-alpha 1.0 --prefixes 2 4 6
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT   = os.path.dirname(_HERE)
_CODE_ROOT = os.path.dirname(_WS_ROOT)
_PROJ_ROOT = os.path.dirname(_CODE_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT, os.path.dirname(_WS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.data.loader import load_data
from weight_symmetry.experiments.classification.exp_clf import MLPEncoder
from weight_symmetry.plotting.style import apply_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_ROOT = os.path.join(_PROJ_ROOT, "files", "results")
DEFAULT_WEIGHTS_DIR  = os.path.join(
    DEFAULT_RESULTS_ROOT, "exprmnt_2026_04_22__15_40_00"
)
FIGURES_DIR = os.path.join(
    DEFAULT_RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "figures"
)

# ---------------------------------------------------------------------------
# Global style — same rcParams as plot_fig_combined panel (b)
# ---------------------------------------------------------------------------
apply_style()

# MNIST: 10 classes → tab10
CMAP = plt.get_cmap("tab10")
CLASS_COLORS = [CMAP(i) for i in range(10)]


def _grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha_tag(alpha: float) -> str:
    """Match exp_clf.py naming convention: f'prefix_l1_a{str(alpha).replace(".","")}'."""
    return f"prefix_l1_a{str(alpha).replace('.', '')}"


def extract_embeddings(encoder_weights: str, X: torch.Tensor,
                        input_dim: int, hidden_dim: int, embed_dim: int) -> np.ndarray:
    """Load encoder state_dict, forward X, return numpy embeddings."""
    encoder = MLPEncoder(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)
    state = torch.load(encoder_weights, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    with torch.no_grad():
        chunks = [encoder(X[i:i+512]).cpu().numpy() for i in range(0, len(X), 512)]
    return np.concatenate(chunks, axis=0)


def run_umap(Z: np.ndarray, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(Z)


def save_fig(fig, base_name: str, fig_stamp: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{base_name}_{fig_stamp}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    print(f"[clf-cluster] Saved {stem}.{{pdf,png,svg}}  →  {out_dir}")
    return stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UMAP cluster visualisation for MNIST classification experiment"
    )
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                        help="Folder containing mrl_e_*.pt and prefix_l1_aXX_*.pt weight files")
    parser.add_argument("--out-dir", default=FIGURES_DIR,
                        help="Output directory for saved figures")
    parser.add_argument("--l1-alpha", type=float, default=1.0,
                        help="PrefixL1 alpha to visualise (default 1.0 = linear decay)")
    parser.add_argument("--prefixes", type=int, nargs="+", default=[2, 4, 6],
                        help="Prefix sizes to visualise (default 2 4 6)")
    parser.add_argument("--subsample", type=int, default=5000,
                        help="Cap on test points used for UMAP / scatter (default 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used to train models and for this plot's subsampling")
    parser.add_argument("--fast", action="store_true",
                        help="Debug mode: subsample=1000 (prefixes unchanged)")
    args = parser.parse_args()

    if args.fast:
        args.subsample = 1000
        print(f"[clf-cluster] FAST mode: subsample=1000 (prefixes={args.prefixes})",
              flush=True)

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")

    # -----------------------------------------------------------------------
    # Models to visualise — column order matters for the plot
    # -----------------------------------------------------------------------
    l1_tag = _alpha_tag(args.l1_alpha)
    models = [
        dict(tag="mrl_e", label="FP-MRL",          flip=False),
        dict(tag=l1_tag,  label=r"MD-$\ell_1$",    flip=True),
    ]

    # -----------------------------------------------------------------------
    # Load MNIST (same seed as training → same split)
    # -----------------------------------------------------------------------
    print(f"[clf-cluster] Loading MNIST (seed={args.seed}) ...")
    data = load_data("mnist", seed=args.seed)

    X_test = data.X_test
    y_test = np.array(data.y_test.tolist(), dtype=np.int64)

    input_dim  = data.input_dim
    hidden_dim = 256     # matches EXPERIMENT config
    embed_dim  = 16      # matches EXPERIMENT config
    n_classes  = int(y_test.max()) + 1

    max_k = max(args.prefixes)
    if max_k > embed_dim:
        raise ValueError(f"Requested prefix {max_k} > embed_dim {embed_dim}")

    # -----------------------------------------------------------------------
    # Subsample test points for plotting (UMAP on the full set is fine but slower)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_test = len(y_test)
    if n_test > args.subsample:
        te_idx = rng.choice(n_test, args.subsample, replace=False)
    else:
        te_idx = np.arange(n_test)
    y_test_sub = y_test[te_idx]
    print(f"[clf-cluster] Using {len(te_idx)} / {n_test} test points for scatter",
          flush=True)

    # -----------------------------------------------------------------------
    # Extract embeddings per model
    # -----------------------------------------------------------------------
    print("[clf-cluster] Extracting test embeddings ...", flush=True)
    embed_test = {}
    for mc in models:
        wpath = os.path.join(args.weights_dir, f"{mc['tag']}_encoder_best.pt")
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"Encoder weight not found: {wpath}")
        print(f"  loading {mc['tag']} from {wpath}", flush=True)
        Z_te = extract_embeddings(wpath, X_test, input_dim, hidden_dim, embed_dim)
        if mc["flip"]:
            Z_te = np.ascontiguousarray(Z_te[:, ::-1])
        embed_test[mc["tag"]] = Z_te

    # -----------------------------------------------------------------------
    # UMAP for each (model, k)
    # -----------------------------------------------------------------------
    print("[clf-cluster] Running UMAP ...", flush=True)
    results = {}
    jobs = [(mc, k) for mc in models for k in args.prefixes]
    pbar = tqdm(jobs, desc="UMAP", unit="panel")
    for mc, k in pbar:
        tag = mc["tag"]
        pbar.set_postfix_str(f"{tag} k={k}")
        Z_te_k_sub = embed_test[tag][:, :k][te_idx]
        xy = run_umap(Z_te_k_sub, args.seed)
        tqdm.write(f"  {tag:<24} k={k}  umap_xy={xy.shape}")
        results[(tag, k)] = {"xy": xy}

    # -----------------------------------------------------------------------
    # Plot — mirrors panel (b) of plot_fig_combined.py:
    #   - figsize / gridspec / spacing copied from panel (b) (≈6.35 × 3.69 in)
    #   - font sizes come from apply_style() (serif 9pt)
    #   - panel labels (i)..(vi) column-major, row labels via annotate,
    #     shared legend strip below, _grid on every axis.
    # -----------------------------------------------------------------------
    n_rows = len(models)          # 2 (FP-MRL, MD-l1)
    n_cols = len(args.prefixes)   # default 3 (k=2,4,6)

    fig = plt.figure(figsize=(9.0, 6.0))
    gs  = fig.add_gridspec(n_rows, n_cols,
                            left=0.13, right=0.98,
                            top=0.89, bottom=0.18,
                            wspace=0.32, hspace=0.38)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_rows)]

    row_labels = [mc["label"] for mc in models]

    for row_idx, mc in enumerate(models):
        tag = mc["tag"]
        for col_idx, k in enumerate(args.prefixes):
            ax = axes[row_idx][col_idx]
            xy = results[(tag, k)]["xy"]

            for c in range(n_classes):
                mask = y_test_sub == c
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           color=CLASS_COLORS[c], s=3, alpha=0.7,
                           linewidths=0, rasterized=True)

            if row_idx == 0:
                ax.set_title(f"$k = {k}$")
            # UMAP-1 on every row (user request)
            ax.set_xlabel("UMAP-1")
            if col_idx == 0:
                ax.set_ylabel("UMAP-2")
                # bold rotated row label, well clear of y-tick numbers
                ax.annotate(row_labels[row_idx],
                            xy=(-0.35, 0.5), xycoords="axes fraction",
                            fontsize=11, rotation=90, va="center", ha="center",
                            fontweight="bold")

            _grid(ax)

    # Shared 10-class legend — horizontal strip at bottom, matches panel (b) style
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=CLASS_COLORS[c], markeredgecolor="none",
               label=str(c))
        for c in range(n_classes)
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.54, 0.02),
               ncol=n_classes, frameon=True, handlelength=1.0,
               borderpad=0.4, labelspacing=0.3, columnspacing=1.0,
               fontsize=8, title="MNIST class", title_fontsize=8)

    fig.suptitle("UMAP of MNIST test embeddings — first $k$ prefix",
                 fontsize=9, y=0.97)

    base_name = f"clf_cluster_mnist_umap_alpha{str(args.l1_alpha).replace('.', '')}"
    save_fig(fig, base_name, fig_stamp, args.out_dir)
    plt.close(fig)
    print("[clf-cluster] Done.")


if __name__ == "__main__":
    main()
