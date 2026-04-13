"""
SEACells UMAP verification check.

Loads the 10x PBMC 10k Multiome filtered feature barcode matrix and the
pre-computed UMAP + graph cluster assignments from the 10x secondary analysis
outputs. Plots the official 10x UMAP colored by graph clusters.

Goal: confirm we have the right dataset and replicate SEACells paper Fig. 4a(i).
Using 10x pre-computed UMAP ensures the topology matches the paper exactly.

Conda environment: seacells  (NOT mrl_env)

Usage:
    conda activate seacells
    python experiments/seacells_umap_check.py           # full run
    python experiments/seacells_umap_check.py --fast    # 2000-cell subsample, quick check
"""

import os
import sys
import time
import argparse

import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import create_run_dir, save_runtime, save_code_snapshot

# ==============================================================================
# CONFIG
# ==============================================================================
N_HVG        = 1500     # highly variable genes
N_PCS        = 50       # PCA components (for preprocessing only)
RANDOM_STATE = 42

DATA_DIR     = os.path.join(os.environ["HOME"], "Mat_embedding_hyperbole", "data", "pbmc_10k_multiome")
H5_FILE      = os.path.join(DATA_DIR, "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
UMAP_CSV     = os.path.join(ANALYSIS_DIR, "dimensionality_reduction", "gex", "umap_projection.csv")
CLUSTER_CSV  = os.path.join(ANALYSIS_DIR, "clustering", "gex", "graphclust", "clusters.csv")
# ==============================================================================


def load_gex(h5_path):
    """Load 10x multiome h5, keep only Gene Expression features."""
    print(f"Loading: {h5_path}")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    if "feature_types" in adata.var.columns:
        adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()

    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def preprocess(adata):
    """Normalize and compute PCA (needed for downstream MRL; not used for UMAP here)."""
    adata.raw = adata

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    print(f"  HVGs selected: {adata.var['highly_variable'].sum()}")

    sc.tl.pca(adata, n_comps=N_PCS, mask_var="highly_variable", random_state=RANDOM_STATE)
    return adata


def attach_10x_umap_and_clusters(adata, umap_csv, cluster_csv):
    """
    Load 10x pre-computed UMAP coordinates and graph cluster labels,
    align by barcode, and attach to adata.
    """
    umap_df    = pd.read_csv(umap_csv, index_col=0)
    cluster_df = pd.read_csv(cluster_csv, index_col=0)

    # align to filtered adata barcodes
    common = adata.obs_names.intersection(umap_df.index)
    adata  = adata[common].copy()

    adata.obsm["X_umap"] = umap_df.loc[common, ["UMAP-1", "UMAP-2"]].values
    adata.obs["cluster"] = cluster_df.loc[common, "Cluster"].astype(str).values

    print(f"  Attached 10x UMAP + clusters for {len(common)} cells")
    print(f"  Graph clusters: {adata.obs['cluster'].nunique()}")
    return adata


def plot_umap(adata, run_dir, fig_stamp):
    """Save UMAP colored by 10x graph clusters, matching SEACells paper Fig. 4a style."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(
        adata,
        color="cluster",
        frameon=False,
        title="PBMC RNA — 10x graph clusters",
        ax=ax,
        show=False,
    )
    path = os.path.join(run_dir, f"umap_10x_clusters{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Subsample 2000 cells for quick check")
    args = parser.parse_args()

    t0        = time.time()
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

    # --- experiment description log ---
    desc = (
        "Experiment : seacells_umap_check\n"
        "Goal       : Verify PBMC 10k Multiome dataset and replicate SEACells Fig. 4a(i) UMAP\n"
        "Input      : filtered_feature_bc_matrix.h5 (GEX only)\n"
        "            + 10x pre-computed UMAP + graph clusters from analysis/\n"
        "Pipeline   : normalize_total → log1p → HVG → PCA → attach 10x UMAP → plot\n"
        "Output     : umap_10x_clusters.png colored by 10x graph clusters\n"
        "\nConfig:\n"
        f"  N_HVG        = {N_HVG}\n"
        f"  N_PCS        = {N_PCS}\n"
        f"  RANDOM_STATE = {RANDOM_STATE}\n"
        f"  --fast       = {args.fast}\n"
    )
    with open(os.path.join(run_dir, "experiment_description.log"), "w") as f:
        f.write(desc)
    print(desc)

    for path, label in [(H5_FILE, "H5 data"), (UMAP_CSV, "UMAP csv"), (CLUSTER_CSV, "cluster csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found:\n  {path}")
            sys.exit(1)

    adata = load_gex(H5_FILE)

    if args.fast:
        sc.pp.subsample(adata, n_obs=2000, random_state=RANDOM_STATE)
        print(f"  --fast: subsampled to {adata.n_obs} cells")

    adata = preprocess(adata)
    adata = attach_10x_umap_and_clusters(adata, UMAP_CSV, CLUSTER_CSV)
    plot_umap(adata, run_dir, fig_stamp)

    # --- results summary ---
    summary = (
        f"Cells:    {adata.n_obs}\n"
        f"Genes:    {adata.n_vars}\n"
        f"HVGs:     {adata.var['highly_variable'].sum()}\n"
        f"PCs:      {N_PCS}\n"
        f"Clusters: {adata.obs['cluster'].nunique()}\n"
    )
    print("\n" + summary)
    with open(os.path.join(run_dir, "results_summary.txt"), "w") as f:
        f.write(summary)

    save_runtime(run_dir, time.time() - t0)
    save_code_snapshot(run_dir)
    print(f"\nOutput: {run_dir}")


if __name__ == "__main__":
    main()
