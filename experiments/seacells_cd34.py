"""
Script: experiments/seacells_cd34.py
--------------------------------------
Run SEACells on the CD34 multiome RNA dataset (GSE200046) and reproduce Fig. 2d
from the SEACells paper (Persad et al., Nature Biotechnology 2023).

Dataset: 6,881 CD34+ HSPCs from 2 bone marrow donors (batch 0 + 1), merged.
         Already preprocessed — log-normalised, PCA (30 PCs), UMAP, celltype labels.
         h5ad also contains paper's original SEACell assignments for comparison.

Cell types: HSC, HMP, Mono, Ery, MEP, CLP, pDC, cDC

Conda environment: seacells  (NOT mrl_env)

Usage:
    conda activate seacells
    python experiments/seacells_cd34.py           # full run (~6881 cells, 195 metacells)
    python experiments/seacells_cd34.py --fast    # 500-cell subsample, quick check
    python experiments/seacells_cd34.py --no-gpu  # force CPU (default falls back auto)
    python experiments/seacells_cd34.py --load-checkpoint <path/to/adata_with_seacells.h5ad>

Outputs (in files/results/[test_runs/]exprmnt_{timestamp}/):
    umap_celltype_{stamp}.png              — UMAP coloured by celltype (ground truth)
    umap_paper_style_{stamp}.png           — black bg, cells by celltype, white metacell circles
    umap_seacell_coloured_{stamp}.png      — UMAP cells coloured by SEACell assignment
    umap_seacell_positions_{stamp}.png     — UMAP grey cells, metacell centroids
    convergence_{stamp}.png               — RSS per iteration
    metacell_sizes_{stamp}.png            — metacell size distribution
    purity_boxplot_{stamp}.png            — cell type purity per metacell
    compactness_boxplot_{stamp}.png       — compactness per metacell
    separation_boxplot_{stamp}.png        — separation per metacell
    purity.csv / compactness.csv / separation.csv
    adata_with_seacells.h5ad              — checkpoint with SEACell assignments
    results_summary.txt
    experiment_description.log
    runtime.txt
    code_snapshot/
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import create_run_dir, save_runtime, save_code_snapshot

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH        = os.path.join(os.environ["HOME"], "Mat_embedding_hyperbole",
                                "data", "cd34_multiome",
                                "GSE200046_cd34_multiome_rna.h5ad")
N_SEACELLS       = 100    # matches paper (Fig. 2d)
N_WAYPOINT_EIGS  = 10
BUILD_KERNEL_ON  = "X_pca"
RANDOM_STATE     = 42
USE_GPU          = True   # try GPU first; falls back to CPU automatically

# Cell type display order + colours extracted from Fig. 2d (Persad et al. 2023)
# Approximate hex values read from the paper legend
CELLTYPE_ORDER  = ["HSC", "HMP", "MEP", "Ery", "Mono", "cDC", "pDC", "CLP"]
CELLTYPE_COLORS = {
    "HSC":  "#B34B78",
    "HMP":  "#985275",
    "MEP":  "#3D611A",
    "Ery":  "#4A8C47",
    "Mono": "#E68A98",
    "cDC":  "#62B1EC",
    "pDC":  "#294D64",
    "CLP":  "#583E18",
}
# ==============================================================================


def load_data(h5ad_path, fast=False):
    """
    Load the pre-processed CD34 h5ad.

    The file already contains:
      - adata.X         : log-normalised counts
      - adata.obsm['X_pca']  : 30 PCs (computed on HVGs)
      - adata.obsm['X_umap'] : 2D UMAP
      - adata.obs['celltype']: HSC / HMP / MEP / Ery / Mono / cDC / pDC / CLP
      - adata.obs['batch']   : donor replicate (0 or 1)
      - adata.obs['SEACell'] : paper's original metacell assignment

    We recompute the neighbor graph so SEACells can build its kernel.
    """
    print(f"Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Cell types: {dict(adata.obs['celltype'].value_counts())}")
    print(f"  Batches: {dict(adata.obs['batch'].value_counts())}")
    print(f"  Paper SEACells: {adata.obs['SEACell'].nunique()}")

    if fast:
        sc.pp.subsample(adata, n_obs=500, random_state=RANDOM_STATE)
        print(f"  --fast: subsampled to {adata.n_obs} cells")

    # Recompute neighbor graph on the existing PCA (needed by SEACells kernel)
    print("  Recomputing neighbor graph on X_pca ...")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15,
                    random_state=RANDOM_STATE)

    return adata


def run_seacells(adata, n_seacells, use_gpu=True):
    """
    Fit SEACells model and return the fitted model.

    GPU/CPU compatibility: tries GPU first (if use_gpu=True), falls back to
    CPU automatically on any failure (missing CUDA, insufficient memory, etc.).
    Works identically on Mac (CPU), Linux without GPU, and multi-GPU clusters.
    """
    def _build_model(gpu):
        return SEACells.core.SEACells(
            adata,
            build_kernel_on=BUILD_KERNEL_ON,
            n_SEACells=n_seacells,
            n_waypoint_eigs=N_WAYPOINT_EIGS,
            convergence_epsilon=1e-5,
            use_gpu=gpu,
        )

    print(f"\nFitting SEACells (n_SEACells={n_seacells}, use_gpu={use_gpu}) ...")

    if use_gpu:
        try:
            model = _build_model(gpu=True)
            model.construct_kernel_matrix()
            model.initialize_archetypes()
            model.fit(min_iter=10, max_iter=50)
            print(f"  Converged after {len(model.RSS_iters)} iterations (GPU)")
            return model
        except Exception as e:
            print(f"  GPU init failed ({e}), falling back to CPU ...")

    # CPU path (either use_gpu=False, or GPU fallback)
    model = _build_model(gpu=False)
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)
    print(f"  Converged after {len(model.RSS_iters)} iterations (CPU)")
    return model


def save_checkpoint(adata, run_dir):
    """Save adata with SEACell assignments so the algo need not be re-run."""
    path = os.path.join(run_dir, "adata_with_seacells.h5ad")
    adata.write_h5ad(path)
    print(f"  Checkpoint saved: {path}")
    return path


def _metacell_centroids(adata):
    """
    Return a DataFrame with columns: SEACell, umap1, umap2, celltype (majority).
    """
    umap = pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs_names,
        columns=["umap1", "umap2"],
    )
    umap["SEACell"] = adata.obs["SEACell"].values
    umap["celltype"] = adata.obs["celltype"].values

    centroids = (
        umap.groupby("SEACell")[["umap1", "umap2"]]
        .mean()
        .reset_index()
    )
    majority_ct = (
        umap.groupby("SEACell")["celltype"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    centroids = centroids.merge(majority_ct, on="SEACell")
    return centroids


def plot_paper_style(adata, run_dir, fig_stamp):
    """
    Reproduce Fig. 2d style:
      - Black background
      - Small dots per cell, colored by cell type
      - Large white-bordered circles at metacell centroid positions,
        filled with the majority cell type color
    """
    centroids = _metacell_centroids(adata)
    umap_coords = adata.obsm["X_umap"]
    celltypes   = adata.obs["celltype"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # --- scatter: one layer per cell type (small dots) ---
    for ct in CELLTYPE_ORDER:
        mask = celltypes == ct
        if mask.sum() == 0:
            continue
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            s=3, alpha=0.6,
            color=CELLTYPE_COLORS.get(ct, "gray"),
            rasterized=True, label=ct,
        )

    # --- metacell circles: white border, majority-celltype fill ---
    for _, row in centroids.iterrows():
        color = CELLTYPE_COLORS.get(row["celltype"], "gray")
        circle = plt.Circle(
            (row["umap1"], row["umap2"]),
            radius=0.12,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            alpha=0.9,
        )
        ax.add_patch(circle)

    # --- legend ---
    legend_handles = [
        mpatches.Patch(color=CELLTYPE_COLORS.get(ct, "gray"), label=ct)
        for ct in CELLTYPE_ORDER
        if (celltypes == ct).sum() > 0
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=8,
        loc="lower right",
        framealpha=0.3,
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        markerscale=2,
    )

    ax.set_title("CD34 RNA — SEACells (paper style)", fontsize=12, color="white")
    ax.set_xlabel("UMAP 1", color="white"); ax.set_ylabel("UMAP 2", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(run_dir, f"umap_paper_style{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_umap_celltype(adata, run_dir, fig_stamp):
    """UMAP coloured by ground-truth cell type — matches Fig. 2d colour scheme."""
    fig, ax = plt.subplots(figsize=(6, 5))
    celltypes   = adata.obs["celltype"].values
    umap_coords = adata.obsm["X_umap"]
    for ct in CELLTYPE_ORDER:
        mask = celltypes == ct
        if mask.sum() == 0:
            continue
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   s=2, alpha=0.6,
                   color=CELLTYPE_COLORS.get(ct, "gray"), label=ct, rasterized=True)
    ax.legend(markerscale=4, fontsize=8, loc="best", framealpha=0.7)
    ax.set_title("CD34 RNA — cell types", fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(run_dir, f"umap_celltype{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metacell_clusters(adata, run_dir, fig_stamp):
    """
    UMAP where each cell is coloured by its metacell assignment, with a KDE
    contour boundary drawn around each metacell's cells — like Fig. 2b style.
    No centroid circles; just colored dots + contour outlines per metacell.
    """
    from scipy.stats import gaussian_kde
    import numpy as np

    umap_coords = adata.obsm["X_umap"]
    seacell_ids = adata.obs["SEACell"].values
    unique_ids  = sorted(set(seacell_ids))
    n           = len(unique_ids)

    # one colour per metacell, cycling through tab20
    cmap   = plt.get_cmap("tab20")
    colors = {sid: cmap(i % 20) for i, sid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # grid for KDE contours
    x_all, y_all = umap_coords[:, 0], umap_coords[:, 1]
    margin = 0.5
    xi = np.linspace(x_all.min() - margin, x_all.max() + margin, 200)
    yi = np.linspace(y_all.min() - margin, y_all.max() + margin, 200)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.vstack([Xi.ravel(), Yi.ravel()])

    for sid in unique_ids:
        mask   = seacell_ids == sid
        pts    = umap_coords[mask]
        color  = colors[sid]

        # scatter: cells coloured by metacell
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[color], s=2, alpha=0.6, rasterized=True)

        # KDE contour boundary (outermost contour only) — skip if too few cells
        if mask.sum() < 5:
            continue
        try:
            kde = gaussian_kde(pts.T, bw_method=0.4)
            Z   = kde(grid_pts).reshape(Xi.shape)
            # draw only the outermost contour at a low density threshold
            threshold = Z.max() * 0.05
            ax.contour(Xi, Yi, Z, levels=[threshold],
                       colors=[color], linewidths=0.8, alpha=0.85)
        except Exception:
            pass  # skip if KDE fails (e.g. degenerate group)

    ax.set_title(f"CD34 RNA — {n} metacell clusters", fontsize=12, color="white")
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(run_dir, f"umap_metacell_clusters{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_plots(model, adata, run_dir, fig_stamp):
    """Save all SEACells visualisations."""
    # 1. Cell type UMAP (ground truth)
    plot_umap_celltype(adata, run_dir, fig_stamp)

    # 2. Paper-style: black bg + metacell circles
    plot_paper_style(adata, run_dir, fig_stamp)

    # 3. Cells coloured by metacell cluster (shows cell→metacell grouping)
    plot_metacell_clusters(adata, run_dir, fig_stamp)

    # 3. UMAP cells coloured by SEACell assignment + centroids
    path = os.path.join(run_dir, f"umap_seacell_coloured{fig_stamp}.png")
    SEACells.plot.plot_2D(adata, key="X_umap", colour_metacells=True,
                          title="CD34 RNA — cells coloured by SEACell",
                          save_as=path, show=False)
    print(f"  Saved: {path}")

    # 4. UMAP grey cells + metacell centroid positions
    path = os.path.join(run_dir, f"umap_seacell_positions{fig_stamp}.png")
    SEACells.plot.plot_2D(adata, key="X_umap", colour_metacells=False,
                          title="CD34 RNA — metacell positions",
                          save_as=path, show=False)
    print(f"  Saved: {path}")

    # 5. Convergence (RSS per iteration)
    path = os.path.join(run_dir, f"convergence{fig_stamp}.png")
    model.plot_convergence(save_as=path, show=False)
    print(f"  Saved: {path}")

    # 6. Metacell size distribution
    path = os.path.join(run_dir, f"metacell_sizes{fig_stamp}.png")
    SEACells.plot.plot_SEACell_sizes(adata, save_as=path, show=False)
    print(f"  Saved: {path}")


def plot_metric_boxplot(df, metric, run_dir, fig_stamp):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=df, y=metric, ax=ax)
    ax.set_title(metric.replace("_", " ").title())
    sns.despine()
    path = os.path.join(run_dir, f"{metric}_boxplot{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="SEACells on CD34 multiome RNA")
    parser.add_argument("--fast", action="store_true",
                        help="Subsample 500 cells, 20 metacells — quick smoke test")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU even if GPU is available")
    parser.add_argument("--load-checkpoint", metavar="H5AD",
                        help="Skip SEACells fitting; load adata from this h5ad checkpoint")
    args = parser.parse_args()

    use_gpu = USE_GPU and not args.no_gpu

    t0        = time.time()
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

    # When loading a checkpoint, create new run dir *inside* the checkpoint folder
    # (same convention as --use-weights in other experiments)
    if args.load_checkpoint:
        weights_dir = os.path.dirname(os.path.abspath(args.load_checkpoint))
        sub_stamp   = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir     = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[utility] Run directory created (inside checkpoint folder): {run_dir}")
    else:
        run_dir = create_run_dir(fast=args.fast)

    n_seacells = 20 if args.fast else N_SEACELLS

    # --- experiment description ---
    desc = (
        "Experiment : seacells_cd34\n"
        "Goal       : Run SEACells on CD34 HSPCs (GSE200046) — reproduce Fig. 2d\n"
        "             of Persad et al. (Nature Biotechnology 2023)\n"
        "Dataset    : 6,881 CD34+ HSPCs, 2 bone marrow donors, RNA only\n"
        "Input      : GSE200046_cd34_multiome_rna.h5ad (pre-processed, log-norm)\n"
        "Output     : paper-style UMAP + purity/compactness/separation + h5ad checkpoint\n"
        "\nConfig:\n"
        f"  N_SEACELLS      = {n_seacells}\n"
        f"  N_WAYPOINT_EIGS = {N_WAYPOINT_EIGS}\n"
        f"  BUILD_KERNEL_ON = {BUILD_KERNEL_ON}\n"
        f"  RANDOM_STATE    = {RANDOM_STATE}\n"
        f"  use_gpu         = {use_gpu}\n"
        f"  --fast          = {args.fast}\n"
        f"  --load-checkpoint = {args.load_checkpoint}\n"
    )
    with open(os.path.join(run_dir, "experiment_description.log"), "w") as f:
        f.write(desc)
    print(desc)

    # --- load or checkpoint ---
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        adata = sc.read_h5ad(args.load_checkpoint)
        print(f"  Loaded: {adata.n_obs} cells, SEACells: {adata.obs['SEACell'].nunique()}")

        print("\nSaving plots ...")
        plot_umap_celltype(adata, run_dir, fig_stamp)
        plot_paper_style(adata, run_dir, fig_stamp)
        plot_metacell_clusters(adata, run_dir, fig_stamp)

        print("\nComputing evaluation metrics ...")
        purity      = SEACells.evaluate.compute_celltype_purity(adata, "celltype")
        compactness = SEACells.evaluate.compactness(adata, BUILD_KERNEL_ON)
        separation  = SEACells.evaluate.separation(adata, BUILD_KERNEL_ON, nth_nbr=1)

        plot_metric_boxplot(purity,      "celltype_purity", run_dir, fig_stamp)
        plot_metric_boxplot(compactness, "compactness",     run_dir, fig_stamp)
        plot_metric_boxplot(separation,  "separation",      run_dir, fig_stamp)

        purity.to_csv(     os.path.join(run_dir, "purity.csv"))
        compactness.to_csv(os.path.join(run_dir, "compactness.csv"))
        separation.to_csv( os.path.join(run_dir, "separation.csv"))

        summary = (
            f"Dataset:             CD34 HSPCs (GSE200046)\n"
            f"Cells:               {adata.n_obs}\n"
            f"Metacells:           {adata.obs['SEACell'].nunique()}\n"
            f"Loaded from:         {args.load_checkpoint}\n"
            f"Median purity:       {purity['celltype_purity'].median():.4f}\n"
            f"Median compactness:  {compactness['compactness'].median():.4f}\n"
            f"Median separation:   {separation['separation'].median():.4f}\n"
        )
        print("\n" + summary)
        with open(os.path.join(run_dir, "results_summary.txt"), "w") as f:
            f.write(summary)

        save_runtime(run_dir, time.time() - t0)
        save_code_snapshot(run_dir)
        print(f"\n[seacells_cd34] Output: {run_dir}")
        return

    # --- normal path: load + fit ---
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found:\n  {DATA_PATH}")
        sys.exit(1)

    adata = load_data(DATA_PATH, fast=args.fast)

    # --- fit SEACells (GPU → CPU fallback) ---
    model = run_seacells(adata, n_seacells, use_gpu=use_gpu)

    # --- checkpoint ---
    print("\nSaving checkpoint ...")
    save_checkpoint(adata, run_dir)

    # --- plots ---
    print("\nSaving plots ...")
    save_plots(model, adata, run_dir, fig_stamp)

    # --- evaluation metrics ---
    print("\nComputing evaluation metrics ...")
    purity      = SEACells.evaluate.compute_celltype_purity(adata, "celltype")
    compactness = SEACells.evaluate.compactness(adata, BUILD_KERNEL_ON)
    separation  = SEACells.evaluate.separation(adata, BUILD_KERNEL_ON, nth_nbr=1)

    plot_metric_boxplot(purity,      "celltype_purity", run_dir, fig_stamp)
    plot_metric_boxplot(compactness, "compactness",     run_dir, fig_stamp)
    plot_metric_boxplot(separation,  "separation",      run_dir, fig_stamp)

    purity.to_csv(     os.path.join(run_dir, "purity.csv"))
    compactness.to_csv(os.path.join(run_dir, "compactness.csv"))
    separation.to_csv( os.path.join(run_dir, "separation.csv"))

    # --- results summary ---
    summary = (
        f"Dataset:             CD34 HSPCs (GSE200046)\n"
        f"Cells:               {adata.n_obs}\n"
        f"Metacells:           {n_seacells}\n"
        f"SEACell iterations:  {len(model.RSS_iters)}\n"
        f"use_gpu:             {use_gpu}\n"
        f"Median purity:       {purity['celltype_purity'].median():.4f}\n"
        f"Median compactness:  {compactness['compactness'].median():.4f}\n"
        f"Median separation:   {separation['separation'].median():.4f}\n"
        f"\nCell type counts:\n"
    )
    for ct in CELLTYPE_ORDER:
        n = (adata.obs["celltype"] == ct).sum()
        summary += f"  {ct:<8} {n}\n"
    print("\n" + summary)
    with open(os.path.join(run_dir, "results_summary.txt"), "w") as f:
        f.write(summary)

    save_runtime(run_dir, time.time() - t0)
    save_code_snapshot(run_dir)
    print(f"\n[seacells_cd34] Output: {run_dir}")


if __name__ == "__main__":
    main()
