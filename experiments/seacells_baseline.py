"""
SEACells baseline — run SEACells on PBMC 10k Multiome (GEX).

Establishes the metacell baseline that MRL experiments will be compared against.
Replicates SEACells paper Fig. 4a: UMAP with metacell circles, cell type purity,
compactness, and separation metrics.

Conda environment: seacells  (NOT mrl_env)

Usage:
    conda activate seacells
    python experiments/seacells_baseline.py           # full run
    python experiments/seacells_baseline.py --fast    # 500-cell subsample, quick check

Outputs (in files/results/[test_runs/]exprmnt_{timestamp}/):
    umap_metacells_{stamp}.png        — UMAP with metacell circles
    purity_boxplot_{stamp}.png        — cell type purity per metacell
    compactness_boxplot_{stamp}.png   — compactness per metacell
    separation_boxplot_{stamp}.png    — separation per metacell
    results_summary.txt               — median purity / compactness / separation
    experiment_description.log        — config dump
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
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import create_run_dir, save_runtime, save_code_snapshot

# ==============================================================================
# CONFIG
# ==============================================================================
N_HVG            = 1500
N_PCS            = 50
N_NEIGHBORS      = 15
RANDOM_STATE     = 42
N_SEACELLS       = 195    # number of metacells (~cells/100 for 11k cells)
N_WAYPOINT_EIGS  = 10     # eigenvalues for archetype initialisation
BUILD_KERNEL_ON  = "X_pca"

DATA_DIR     = os.path.join(os.environ["HOME"], "Mat_embedding_hyperbole", "data", "pbmc_10k_multiome")
H5_FILE      = os.path.join(DATA_DIR, "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
UMAP_CSV     = os.path.join(ANALYSIS_DIR, "dimensionality_reduction", "gex", "umap_projection.csv")
CLUSTER_CSV  = os.path.join(ANALYSIS_DIR, "clustering", "gex", "graphclust", "clusters.csv")
# ==============================================================================


def load_and_preprocess(h5_path, umap_csv, cluster_csv, fast=False):
    """Load GEX, preprocess, attach 10x UMAP + cluster labels."""
    print("Loading data...")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    if "feature_types" in adata.var.columns:
        adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    if fast:
        sc.pp.subsample(adata, n_obs=500, random_state=RANDOM_STATE)
        print(f"  --fast: subsampled to {adata.n_obs} cells")

    # raw counts needed by SEACells summarize step
    adata.raw = adata

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    sc.tl.pca(adata, n_comps=N_PCS, mask_var="highly_variable", random_state=RANDOM_STATE)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=RANDOM_STATE)
    print(f"  After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

    # attach 10x pre-computed UMAP + graph cluster labels
    umap_df    = pd.read_csv(umap_csv, index_col=0)
    cluster_df = pd.read_csv(cluster_csv, index_col=0)
    common     = adata.obs_names.intersection(umap_df.index)
    adata      = adata[common].copy()
    adata.obsm["X_umap"] = umap_df.loc[common, ["UMAP-1", "UMAP-2"]].values
    adata.obs["cluster"] = cluster_df.loc[common, "Cluster"].astype(str).values
    print(f"  Attached 10x UMAP + {adata.obs['cluster'].nunique()} graph clusters")

    return adata


def run_seacells(adata, n_seacells):
    """Fit SEACells model and assign metacells."""
    print(f"\nFitting SEACells (n={n_seacells})...")
    model = SEACells.core.SEACells(
        adata,
        build_kernel_on=BUILD_KERNEL_ON,
        n_SEACells=n_seacells,
        n_waypoint_eigs=N_WAYPOINT_EIGS,
        convergence_epsilon=1e-5,
    )
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)
    print(f"  Converged after {len(model.RSS_iters)} iterations")
    return model


def save_plots(model, adata, run_dir, fig_stamp):
    """Save all SEACells visualisations using SEACells' own plot functions."""

    # 1. UMAP: cells colored by SEACell assignment, centroids overlaid (paper style)
    path = os.path.join(run_dir, f"umap_coloured_by_seacell{fig_stamp}.png")
    SEACells.plot.plot_2D(adata, key="X_umap", colour_metacells=True,
                          title="PBMC RNA — cells coloured by SEACell",
                          save_as=path, show=False)
    print(f"  Saved: {path}")

    # 2. UMAP: cells grey, metacell centroids in red (clean overview)
    path = os.path.join(run_dir, f"umap_metacell_positions{fig_stamp}.png")
    SEACells.plot.plot_2D(adata, key="X_umap", colour_metacells=False,
                          title="PBMC RNA — metacell positions",
                          save_as=path, show=False)
    print(f"  Saved: {path}")

    # 3. Convergence (RSS per iteration)
    path = os.path.join(run_dir, f"convergence{fig_stamp}.png")
    model.plot_convergence(save_as=path, show=False)
    print(f"  Saved: {path}")

    # 4. Metacell size distribution
    path = os.path.join(run_dir, f"metacell_sizes{fig_stamp}.png")
    SEACells.plot.plot_SEACell_sizes(adata, save_as=path, show=False)
    print(f"  Saved: {path}")


def plot_metric_boxplot(df, metric, run_dir, fig_stamp):
    """Boxplot for a single SEACells evaluation metric."""
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=df, y=metric, ax=ax)
    ax.set_title(metric.replace("_", " ").title())
    sns.despine()
    path = os.path.join(run_dir, f"{metric}_boxplot{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Subsample 500 cells, fewer metacells — quick smoke test")
    args = parser.parse_args()

    t0        = time.time()
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

    n_seacells = 20 if args.fast else N_SEACELLS

    # --- experiment description ---
    desc = (
        "Experiment : seacells_baseline\n"
        "Goal       : Run SEACells on PBMC 10k Multiome GEX — establish metacell baseline\n"
        "             for comparison against MRL embeddings\n"
        "Input      : filtered_feature_bc_matrix.h5 + 10x analysis UMAP + graph clusters\n"
        "Output     : UMAP with metacell circles, purity/compactness/separation metrics\n"
        "\nConfig:\n"
        f"  N_HVG           = {N_HVG}\n"
        f"  N_PCS           = {N_PCS}\n"
        f"  N_SEACELLS      = {n_seacells}\n"
        f"  N_WAYPOINT_EIGS = {N_WAYPOINT_EIGS}\n"
        f"  BUILD_KERNEL_ON = {BUILD_KERNEL_ON}\n"
        f"  RANDOM_STATE    = {RANDOM_STATE}\n"
        f"  --fast          = {args.fast}\n"
    )
    with open(os.path.join(run_dir, "experiment_description.log"), "w") as f:
        f.write(desc)
    print(desc)

    for path, label in [(H5_FILE, "H5 data"), (UMAP_CSV, "UMAP csv"), (CLUSTER_CSV, "cluster csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found:\n  {path}")
            sys.exit(1)

    # --- load + preprocess ---
    adata = load_and_preprocess(H5_FILE, UMAP_CSV, CLUSTER_CSV, fast=args.fast)

    # --- fit SEACells ---
    model = run_seacells(adata, n_seacells)

    # --- plots ---
    save_plots(model, adata, run_dir, fig_stamp)

    # --- evaluation metrics ---
    print("\nComputing evaluation metrics...")
    purity      = SEACells.evaluate.compute_celltype_purity(adata, "cluster")
    compactness = SEACells.evaluate.compactness(adata, BUILD_KERNEL_ON)
    separation  = SEACells.evaluate.separation(adata, BUILD_KERNEL_ON, nth_nbr=1)

    plot_metric_boxplot(purity,      "cluster_purity",  run_dir, fig_stamp)
    plot_metric_boxplot(compactness, "compactness",     run_dir, fig_stamp)
    plot_metric_boxplot(separation,  "separation",      run_dir, fig_stamp)

    # save raw metric CSVs
    purity.to_csv(os.path.join(run_dir, "purity.csv"))
    compactness.to_csv(os.path.join(run_dir, "compactness.csv"))
    separation.to_csv(os.path.join(run_dir, "separation.csv"))

    # --- results summary ---
    summary = (
        f"Cells:               {adata.n_obs}\n"
        f"Metacells:           {n_seacells}\n"
        f"SEACell iterations:  {len(model.RSS_iters)}\n"
        f"Median purity:       {purity['cluster_purity'].median():.4f}\n"
        f"Median compactness:  {compactness['compactness'].median():.4f}\n"
        f"Median separation:   {separation['separation'].median():.4f}\n"
    )
    print("\n" + summary)
    with open(os.path.join(run_dir, "results_summary.txt"), "w") as f:
        f.write(summary)

    save_runtime(run_dir, time.time() - t0)
    save_code_snapshot(run_dir)
    print(f"\nOutput: {run_dir}")


if __name__ == "__main__":
    main()
