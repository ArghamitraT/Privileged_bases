"""
Script: experiments/exp5_seed_stability.py
-------------------------------------------
Experiment 5 — Seed Stability.

Tests whether the learned coordinate ordering is reproducible across
different random seeds. The data split is held fixed (single data_seed)
so that only model initialisation and training randomness vary.

For each of N model seeds:
    1. Train a Standard model and a Matryoshka model.
    2. Compute the prefix accuracy curve for each.
    3. Store the test-set embeddings for cross-seed correlation analysis.

After all seeds are done:
    - Compute mean ± std of accuracy at each prefix k  (stability of accuracy).
    - Compute pairwise dimension-correlation matrices between seed pairs
      (stability of what each dimension encodes).

Expected results:
    - Mat prefix curves have lower variance than Standard (stable ordering).
    - Mat diagonal correlation is higher than Standard (dimensions encode
      consistent information across runs).

Usage:
    python experiments/exp5_seed_stability.py
    python experiments/exp5_seed_stability.py --fast
    python experiments/exp5_seed_stability.py --low-dim          # 10-dim mode
    python experiments/exp5_seed_stability.py --fast --low-dim   # quick 10-dim smoke

Inputs:  ExpConfig (defaults from config.py, overridden below)
Outputs: Per-run folder containing —
           experiment_description.log        : what/why/expected + config
           seed_{s}/standard_train.log       : per-seed training logs
           seed_{s}/mat_train.log
           seed_{s}/*_best.pt                : per-seed checkpoints
           prefix_stability.png              : mean ± std prefix curves
           variance_comparison.png           : std bar chart per prefix k
           correlation_heatmaps.png          : avg cross-seed corr matrices
           correlation_summary.txt           : mean diagonal correlation
           training_curves.png               : loss-vs-epoch for every seed
           results_summary.txt               : full accuracy table

         When embed_dim <= 16 (e.g. --low-dim), additionally:
           cosine_similarity.png             : per-dim cosine sim bar chart
           spearman_correlation.png          : per-dim Spearman rank corr bar chart
           per_dim_correlation.png           : per-dim Pearson corr bar chart
           cka_summary.png                   : CKA bar chart (Std vs Mat)
           cka_summary.txt                   : CKA numeric summary
           tsne_dim_coloring.png             : t-SNE colored by dim value
"""

import os
import sys
import time
import json
import dataclasses
from itertools import combinations

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from tqdm import tqdm

# Allow imports from the project root regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss
from training.trainer import train
from evaluation.prefix_eval import evaluate_prefix_sweep, evaluate_pca_baseline


# ==============================================================================
# Helpers (reused from exp1 where possible)
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for reproducibility across numpy, torch.

    Args:
        seed (int): Master random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[main] Random seeds set to {seed}")


def save_experiment_description(cfg: ExpConfig, run_dir: str,
                                model_seeds: list, data_seed: int):
    """
    Write a human-readable experiment log for Experiment 5.

    Args:
        cfg         (ExpConfig): The experiment configuration.
        run_dir     (str)      : Path to the run output directory.
        model_seeds (list)     : List of model training seeds.
        data_seed   (int)      : Seed used for the fixed data split.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 5 — Seed Stability\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Trains Standard and Matryoshka models {len(model_seeds)} times each,\n"
            f"varying only the model initialisation seed while keeping the data\n"
            f"split fixed (data_seed={data_seed}).\n\n"
            f"For each run, computes the prefix accuracy curve and stores the\n"
            f"test-set embeddings. After all runs:\n"
            f"  - Measures mean and variance of accuracy at each prefix k.\n"
            f"  - Computes pairwise cross-seed dimension correlation matrices\n"
            f"    to test whether individual dimensions encode consistent info.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Interpretability requires stability. If dimension 1 means\n"
            "something different in every training run, the coordinate\n"
            "ordering is not truly meaningful. Matryoshka training forces\n"
            "early dimensions to be useful, which should make the ordering\n"
            "more reproducible across random seeds.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "- Mat prefix curves: LOW variance (stable ordering).\n"
            "- Standard prefix curves: HIGH variance (arbitrary ordering).\n"
            "- Mat dimension correlation: HIGH diagonal (consistent meaning).\n"
            "- Standard dimension correlation: LOW diagonal (shuffled meaning).\n\n"
        )

        f.write("SEEDS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  data_seed   : {data_seed}\n")
        f.write(f"  model_seeds : {model_seeds}\n\n")

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[main] Experiment description saved to {log_path}")


# ==============================================================================
# Correlation analysis
# ==============================================================================

def compute_cross_seed_correlation(embeddings_by_seed: dict) -> dict:
    """
    Compute pairwise dimension-correlation matrices across seed pairs.

    For seeds i and j, the correlation matrix C has shape (embed_dim, embed_dim)
    where C[d1, d2] = pearson_corr(embed_i[:, d1], embed_j[:, d2]).

    If coordinates are stable, the diagonal of C should be large (dimension d
    in run i correlates with dimension d in run j).

    Args:
        embeddings_by_seed (dict): {seed: np.ndarray of shape (N_test, embed_dim)}

    Returns:
        dict: {
            'mean_corr_matrix' : np.ndarray (embed_dim, embed_dim) — avg |corr| over pairs,
            'mean_diag_corr'   : float — mean of |diagonal| of the avg corr matrix,
            'per_pair'         : list of (seed_i, seed_j, diag_corr) tuples,
        }
    """
    seeds = sorted(embeddings_by_seed.keys())
    embed_dim = next(iter(embeddings_by_seed.values())).shape[1]

    # Collect absolute correlation matrices for all pairs
    corr_matrices = []
    per_pair = []

    for si, sj in combinations(seeds, 2):
        Ei = embeddings_by_seed[si]  # (N, D)
        Ej = embeddings_by_seed[sj]  # (N, D)

        # Pearson correlation between all dim pairs
        # np.corrcoef on stacked columns: shape (2D, 2D), take top-right block
        # More efficient: compute manually
        # Center each dimension
        Ei_c = Ei - Ei.mean(axis=0, keepdims=True)
        Ej_c = Ej - Ej.mean(axis=0, keepdims=True)

        # Normalise
        Ei_n = Ei_c / (np.linalg.norm(Ei_c, axis=0, keepdims=True) + 1e-12)
        Ej_n = Ej_c / (np.linalg.norm(Ej_c, axis=0, keepdims=True) + 1e-12)

        # Correlation matrix: (D, D) — after unit-normalising each column,
        # the dot product gives Pearson r directly (no division by N needed).
        C = Ei_n.T @ Ej_n

        abs_C = np.abs(C)
        corr_matrices.append(abs_C)

        diag_corr = np.mean(np.diag(abs_C))
        per_pair.append((si, sj, diag_corr))

        print(f"  Seeds ({si}, {sj}): mean |diag corr| = {diag_corr:.4f}")

    # Average correlation matrix across all pairs
    mean_corr = np.mean(corr_matrices, axis=0)
    mean_diag = np.mean(np.diag(mean_corr))

    return {
        "mean_corr_matrix": mean_corr,
        "mean_diag_corr": mean_diag,
        "per_pair": per_pair,
    }


# ==============================================================================
# Low-dim stability metrics (gated on embed_dim <= 16)
# ==============================================================================

def compute_cosine_similarity(embeddings_by_seed: dict) -> np.ndarray:
    """
    Per-dimension cosine similarity across seed pairs.

    For each dimension d, treat embeddings[:, d] as a vector over test samples.
    Compute cosine similarity between seed_i's dim d and seed_j's dim d,
    then average over all seed pairs.

    Args:
        embeddings_by_seed (dict): {seed: np.ndarray (N_test, embed_dim)}

    Returns:
        np.ndarray: shape (embed_dim,) — mean cosine similarity per dimension.
    """
    seeds = sorted(embeddings_by_seed.keys())
    embed_dim = next(iter(embeddings_by_seed.values())).shape[1]
    pair_sims = []

    for si, sj in combinations(seeds, 2):
        Ei = embeddings_by_seed[si]
        Ej = embeddings_by_seed[sj]
        # Per-dim cosine similarity: dot(a, b) / (||a|| * ||b||)
        sims = np.array([
            np.dot(Ei[:, d], Ej[:, d]) /
            (np.linalg.norm(Ei[:, d]) * np.linalg.norm(Ej[:, d]) + 1e-12)
            for d in range(embed_dim)
        ])
        pair_sims.append(sims)

    return np.mean(pair_sims, axis=0)


def compute_spearman_correlation(embeddings_by_seed: dict) -> np.ndarray:
    """
    Per-dimension Spearman rank correlation across seed pairs.

    For each dimension d, rank test samples by their value on dim d,
    compute Spearman rank correlation across seed pairs, then average.

    Args:
        embeddings_by_seed (dict): {seed: np.ndarray (N_test, embed_dim)}

    Returns:
        np.ndarray: shape (embed_dim,) — mean |Spearman corr| per dimension.
    """
    seeds = sorted(embeddings_by_seed.keys())
    embed_dim = next(iter(embeddings_by_seed.values())).shape[1]
    pair_corrs = []

    for si, sj in combinations(seeds, 2):
        Ei = embeddings_by_seed[si]
        Ej = embeddings_by_seed[sj]
        corrs = np.array([
            abs(spearmanr(Ei[:, d], Ej[:, d]).statistic)
            for d in range(embed_dim)
        ])
        pair_corrs.append(corrs)

    return np.mean(pair_corrs, axis=0)


def compute_cka(embeddings_by_seed: dict) -> dict:
    """
    Compute linear Centered Kernel Alignment (CKA) across seed pairs.

    CKA is rotation-invariant — high CKA alone doesn't prove privileged basis.
    Key diagnostic: high CKA + low per-dim correlation = info present but
    scrambled (no privileged basis). High CKA + high per-dim corr = stable
    privileged basis.

    Args:
        embeddings_by_seed (dict): {seed: np.ndarray (N_test, embed_dim)}

    Returns:
        dict: {
            'mean_cka': float — average CKA over all seed pairs,
            'std_cka' : float — std of CKA over pairs,
            'per_pair': list of (seed_i, seed_j, cka_value) tuples,
        }
    """
    seeds = sorted(embeddings_by_seed.keys())
    per_pair = []

    for si, sj in combinations(seeds, 2):
        X = embeddings_by_seed[si]  # (N, D)
        Y = embeddings_by_seed[sj]  # (N, D)

        # Center the matrices
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # Linear CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        XtY = X.T @ Y
        XtX = X.T @ X
        YtY = Y.T @ Y

        hsic_xy = np.sum(XtY ** 2)
        hsic_xx = np.sum(XtX ** 2)
        hsic_yy = np.sum(YtY ** 2)

        cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)
        per_pair.append((si, sj, cka))

    cka_values = [v for _, _, v in per_pair]
    return {
        "mean_cka": np.mean(cka_values),
        "std_cka": np.std(cka_values),
        "per_pair": per_pair,
    }


# ==============================================================================
# Low-dim plotting functions (gated on embed_dim <= 16)
# ==============================================================================

def plot_per_dim_cosine_similarity(
    std_cos: np.ndarray,
    mat_cos: np.ndarray,
    cfg,
    run_dir: str,
):
    """
    Bar chart of per-dimension cosine similarity (Std vs Mat).

    Args:
        std_cos (np.ndarray): Per-dim cosine sim for Standard, shape (embed_dim,).
        mat_cos (np.ndarray): Per-dim cosine sim for Matryoshka, shape (embed_dim,).
        cfg     (ExpConfig) : For title info.
        run_dir (str)       : Save path.
    """
    D = len(std_cos)
    x = np.arange(D)
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width / 2, std_cos, width, label="Standard", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, mat_cos, width, label="Matryoshka", color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(D)])
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity (across seed pairs)", fontsize=12)
    ax.set_title(
        f"Per-Dimension Cosine Similarity — {cfg.dataset} (embed_dim={cfg.embed_dim})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "cosine_similarity.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] Cosine similarity plot saved to {plot_path}")


def plot_per_dim_spearman(
    std_spearman: np.ndarray,
    mat_spearman: np.ndarray,
    cfg,
    run_dir: str,
):
    """
    Bar chart of per-dimension Spearman rank correlation (Std vs Mat).

    Args:
        std_spearman (np.ndarray): Per-dim |Spearman| for Standard, shape (embed_dim,).
        mat_spearman (np.ndarray): Per-dim |Spearman| for Matryoshka, shape (embed_dim,).
        cfg          (ExpConfig) : For title info.
        run_dir      (str)       : Save path.
    """
    D = len(std_spearman)
    x = np.arange(D)
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width / 2, std_spearman, width, label="Standard", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, mat_spearman, width, label="Matryoshka", color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(D)])
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("Mean |Spearman Rank Corr| (across seed pairs)", fontsize=12)
    ax.set_title(
        f"Per-Dimension Spearman Rank Correlation — {cfg.dataset} (embed_dim={cfg.embed_dim})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "spearman_correlation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] Spearman correlation plot saved to {plot_path}")


def plot_per_dim_correlation(
    std_corr: dict,
    mat_corr: dict,
    cfg,
    run_dir: str,
):
    """
    Bar chart of per-dimension Pearson correlation from diagonal of mean corr matrix.

    Decomposes the scalar "mean diag corr" into per-dimension detail.

    Args:
        std_corr (dict): Output of compute_cross_seed_correlation for Standard.
        mat_corr (dict): Output of compute_cross_seed_correlation for Matryoshka.
        cfg      (ExpConfig): For title info.
        run_dir  (str) : Save path.
    """
    std_diag = np.diag(std_corr["mean_corr_matrix"])
    mat_diag = np.diag(mat_corr["mean_corr_matrix"])
    D = len(std_diag)
    x = np.arange(D)
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width / 2, std_diag, width, label="Standard", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, mat_diag, width, label="Matryoshka", color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(D)])
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("Mean |Pearson Corr| (diagonal of cross-seed matrix)", fontsize=12)
    ax.set_title(
        f"Per-Dimension Pearson Correlation — {cfg.dataset} (embed_dim={cfg.embed_dim})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "per_dim_correlation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] Per-dim Pearson correlation plot saved to {plot_path}")


def plot_cka_summary(
    std_cka: dict,
    mat_cka: dict,
    run_dir: str,
):
    """
    Bar chart comparing CKA values for Standard vs Matryoshka (mean ± std).
    Also saves a text summary.

    Args:
        std_cka (dict): Output of compute_cka for Standard.
        mat_cka (dict): Output of compute_cka for Matryoshka.
        run_dir (str) : Save path.
    """
    # --- Bar chart ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))

    means = [std_cka["mean_cka"], mat_cka["mean_cka"]]
    stds = [std_cka["std_cka"], mat_cka["std_cka"]]
    labels = ["Standard", "Matryoshka"]
    colors = ["steelblue", "darkorange"]

    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Linear CKA (mean ± std across seed pairs)", fontsize=12)
    ax.set_title("Centered Kernel Alignment — Cross-Seed", fontsize=13)
    ax.set_ylim(0, 1.1)

    # Annotate bars with numeric values
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "cka_summary.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] CKA summary plot saved to {plot_path}")

    # --- Text summary ---
    txt_path = os.path.join(run_dir, "cka_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Centered Kernel Alignment (CKA) — Cross-Seed Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Standard  — CKA: {std_cka['mean_cka']:.4f} ± {std_cka['std_cka']:.4f}\n")
        f.write(f"Matryoshka — CKA: {mat_cka['mean_cka']:.4f} ± {mat_cka['std_cka']:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("  High CKA + high per-dim corr → stable privileged basis\n")
        f.write("  High CKA + low per-dim corr  → info present but scrambled\n\n")
        f.write("Per-pair breakdown:\n")
        f.write("-" * 40 + "\n")
        f.write("\nStandard:\n")
        for si, sj, v in std_cka["per_pair"]:
            f.write(f"  seeds ({si}, {sj}): CKA = {v:.4f}\n")
        f.write("\nMatryoshka:\n")
        for si, sj, v in mat_cka["per_pair"]:
            f.write(f"  seeds ({si}, {sj}): CKA = {v:.4f}\n")
    print(f"[main] CKA summary text saved to {txt_path}")


def plot_tsne_dim_coloring(
    std_embeddings: dict,
    mat_embeddings: dict,
    data,
    cfg,
    run_dir: str,
    n_seeds_to_show: int = 3,
):
    """
    t-SNE visualization colored by each dimension's value.

    Embed test set in 2D with t-SNE (computed once), then create a grid of
    panels: rows = seeds, columns = dimensions. Each panel colors points by
    the value of that dimension. Produces two grids: one for Standard, one
    for Matryoshka.

    If Mat dimension d has consistent meaning across seeds, the color pattern
    should look similar across rows.

    Args:
        std_embeddings  (dict): {seed: np.ndarray (N, D)} Standard embeddings.
        mat_embeddings  (dict): {seed: np.ndarray (N, D)} Matryoshka embeddings.
        data                  : Data object with X_test.
        cfg             (ExpConfig): For embed_dim, dataset info.
        run_dir         (str) : Save path.
        n_seeds_to_show (int) : Max number of seed rows to display.
    """
    seeds = sorted(std_embeddings.keys())[:n_seeds_to_show]
    D = cfg.embed_dim

    # Compute t-SNE once on the test data (use first seed's mat embedding as basis)
    print("[main] Computing t-SNE for dimension coloring...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data.X_test) - 1))

    # Use the raw test features for t-SNE layout (shared across all panels)
    X_test_np = data.X_test.numpy() if hasattr(data.X_test, 'numpy') else np.array(data.X_test)
    coords = tsne.fit_transform(X_test_np)

    for model_label, embeddings in [("Standard", std_embeddings),
                                     ("Matryoshka", mat_embeddings)]:
        n_rows = len(seeds)
        n_cols = D
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows),
                                  squeeze=False)

        for row_idx, seed in enumerate(seeds):
            emb = embeddings[seed]
            for col_idx in range(D):
                ax = axes[row_idx, col_idx]
                vals = emb[:, col_idx]
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="viridis",
                                s=5, alpha=0.7, rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f"dim {col_idx}", fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"seed {seed}", fontsize=9)

        fig.suptitle(
            f"t-SNE Colored by Dimension Value — {model_label}\n"
            f"({cfg.dataset}, embed_dim={cfg.embed_dim})",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        plot_path = os.path.join(run_dir, f"tsne_dim_coloring_{model_label.lower()}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[main] t-SNE dim coloring ({model_label}) saved to {plot_path}")


# ==============================================================================
# Plotting
# ==============================================================================

def plot_prefix_stability(
    std_curves: dict,
    mat_curves: dict,
    pca_curves: dict,
    cfg: ExpConfig,
    run_dir: str,
):
    """
    Plot mean ± std prefix accuracy curves for Standard, Mat, and PCA.

    Args:
        std_curves (dict): {seed: {k: accuracy}} for the standard model.
        mat_curves (dict): {seed: {k: accuracy}} for the Matryoshka model.
        pca_curves (dict): {seed: {k: accuracy}} for PCA (only 1 run, but
                           included for reference).
        cfg        (ExpConfig): Used for title/label info.
        run_dir    (str): Where to save the PNG.
    """
    prefixes = sorted(cfg.eval_prefixes)

    # Build arrays: shape (n_seeds, n_prefixes)
    std_arr = np.array([[std_curves[s][k] for k in prefixes]
                        for s in sorted(std_curves.keys())])
    mat_arr = np.array([[mat_curves[s][k] for k in prefixes]
                        for s in sorted(mat_curves.keys())])

    std_mean, std_std = std_arr.mean(axis=0), std_arr.std(axis=0)
    mat_mean, mat_std = mat_arr.mean(axis=0), mat_arr.std(axis=0)

    # PCA: only one run (deterministic given fixed data), so no band
    pca_accs = [list(pca_curves.values())[0][k] for k in prefixes]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    # Standard: mean line + shaded band
    ax.plot(prefixes, std_mean, "o-", color="steelblue", label="Standard (mean)",
            linewidth=2, markersize=7)
    ax.fill_between(prefixes, std_mean - std_std, std_mean + std_std,
                    color="steelblue", alpha=0.2, label="Standard (±1 std)")

    # Matryoshka: mean line + shaded band
    ax.plot(prefixes, mat_mean, "s-", color="darkorange", label="Matryoshka (mean)",
            linewidth=2, markersize=7)
    ax.fill_between(prefixes, mat_mean - mat_std, mat_mean + mat_std,
                    color="darkorange", alpha=0.2, label="Matryoshka (±1 std)")

    # PCA: single line, no band
    ax.plot(prefixes, pca_accs, "^--", color="seagreen", label="PCA baseline",
            linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(prefixes)
    ax.set_xticklabels([str(k) for k in prefixes])
    ax.set_xlabel("Prefix size k  (number of embedding dimensions used)", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(
        f"Seed Stability — Prefix Performance Curve (mean ± std)\n"
        f"Dataset: {cfg.dataset}  |  embed_dim={cfg.embed_dim}  |  "
        f"n_seeds={len(std_curves)}",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "prefix_stability.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] Prefix stability plot saved to {plot_path}")


def plot_variance_comparison(
    std_curves: dict,
    mat_curves: dict,
    cfg: ExpConfig,
    run_dir: str,
):
    """
    Bar chart comparing the std of Standard vs Mat at each prefix k.

    Args:
        std_curves (dict): {seed: {k: accuracy}} for Standard.
        mat_curves (dict): {seed: {k: accuracy}} for Matryoshka.
        cfg        (ExpConfig): For prefix list.
        run_dir    (str): Save path.
    """
    prefixes = sorted(cfg.eval_prefixes)

    std_arr = np.array([[std_curves[s][k] for k in prefixes]
                        for s in sorted(std_curves.keys())])
    mat_arr = np.array([[mat_curves[s][k] for k in prefixes]
                        for s in sorted(mat_curves.keys())])

    std_std = std_arr.std(axis=0)
    mat_std = mat_arr.std(axis=0)

    x = np.arange(len(prefixes))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, std_std, width, label="Standard", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, mat_std, width, label="Matryoshka", color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in prefixes])
    ax.set_xlabel("Prefix size k", fontsize=12)
    ax.set_ylabel("Std of accuracy across seeds", fontsize=12)
    ax.set_title("Variance Comparison — Standard vs Matryoshka", fontsize=13)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "variance_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] Variance comparison plot saved to {plot_path}")


def plot_correlation_heatmaps(
    std_corr: dict,
    mat_corr: dict,
    run_dir: str,
    annotate: bool = False,
):
    """
    Side-by-side heatmaps of the mean |correlation| matrices.

    Strong diagonal = dimensions have consistent meaning across seeds.

    Args:
        std_corr (dict)  : Output of compute_cross_seed_correlation for Standard.
        mat_corr (dict)  : Output of compute_cross_seed_correlation for Mat.
        run_dir  (str)   : Save path.
        annotate (bool)  : If True, annotate each cell with its numeric value
                           (2 decimal places). Best for small matrices (<=16 dims).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, corr, title, cmap in [
        (axes[0], std_corr["mean_corr_matrix"], "Standard", "Blues"),
        (axes[1], mat_corr["mean_corr_matrix"], "Matryoshka", "Oranges"),
    ]:
        im = ax.imshow(corr, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        ax.set_title(
            f"{title}\nmean |diag corr| = "
            f"{np.mean(np.diag(corr)):.4f}",
            fontsize=12,
        )
        ax.set_xlabel("Dimension (run j)", fontsize=10)
        ax.set_ylabel("Dimension (run i)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells with numeric values when matrix is small
        if annotate:
            D = corr.shape[0]
            for i in range(D):
                for j in range(D):
                    # Use white text on dark cells, black on light
                    text_color = "white" if corr[i, j] > 0.6 else "black"
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)

    fig.suptitle(
        "Cross-Seed Dimension Correlation (avg over all seed pairs)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    plot_path = os.path.join(run_dir, "correlation_heatmaps.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Correlation heatmaps saved to {plot_path}")


def plot_all_training_curves(
    all_std_histories: dict,
    all_mat_histories: dict,
    run_dir: str,
):
    """
    Plot loss-vs-epoch for every seed, one subplot row per seed.

    Each row has two subplots: Standard (left) and Matryoshka (right).

    Args:
        all_std_histories (dict): {seed: history_dict} for Standard.
        all_mat_histories (dict): {seed: history_dict} for Matryoshka.
        run_dir           (str) : Save path.
    """
    seeds = sorted(all_std_histories.keys())
    n_seeds = len(seeds)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_seeds, 2, figsize=(13, 4 * n_seeds), squeeze=False)

    for row, seed in enumerate(seeds):
        for col, (history, title, color) in enumerate([
            (all_std_histories[seed], f"Standard (seed={seed})", "steelblue"),
            (all_mat_histories[seed], f"Matryoshka (seed={seed})", "darkorange"),
        ]):
            ax = axes[row, col]
            epochs = range(1, len(history["train_losses"]) + 1)
            best_epoch = history["best_epoch"] + 1

            ax.plot(epochs, history["train_losses"], "-", color=color,
                    label="Train loss", linewidth=2)
            ax.plot(epochs, history["val_losses"], "--", color=color,
                    label="Val loss", alpha=0.7, linewidth=2)
            ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
                       label=f"Best epoch ({best_epoch})")

            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle("Training Curves — All Seeds", fontsize=14, y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Training curves saved to {plot_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():

    # ------------------------------------------------------------------
    # Step 0: Parse arguments
    # ------------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 5: Seed Stability")
    parser.add_argument(
        "--fast", action="store_true",
        help=(
            "Smoke-test mode: use digits dataset, 2 seeds, 3 epochs. "
            "Skips MNIST download and runs in seconds."
        ),
    )
    parser.add_argument(
        "--low-dim", action="store_true",
        help=(
            "Use 10-dim embedding for interpretable visualizations. "
            "Overrides embed_dim=10 and eval_prefixes=[1..10]. "
            "Set dataset='digits' in config.py for best results."
        ),
    )
    args = parser.parse_args()
    run_start = time.time()   # record wall-clock start for runtime.txt

    # ------------------------------------------------------------------
    # Step 1: Configure
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset="digits",
            embed_dim=16,
            eval_prefixes=[1, 2, 4, 8, 16],
            epochs=3,
            patience=2,
            data_seed=42,
            model_seeds=[100, 200],
            experiment_name="exp5_seed_stability",
        )
        print("[main] --fast mode: digits dataset, 2 seeds, 3 epochs")
    else:
        cfg = ExpConfig(experiment_name="exp5_seed_stability")

    # Apply --low-dim overrides (compatible with both --fast and default)
    if args.low_dim:
        cfg.embed_dim = 10
        cfg.eval_prefixes = list(range(1, 11))  # [1, 2, 3, ..., 10]
        print(f"[main] --low-dim mode: embed_dim={cfg.embed_dim}, "
              f"eval_prefixes={cfg.eval_prefixes}")

    # ------------------------------------------------------------------
    # Step 2: Setup — run directory, description log
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[main] Outputs will be saved to: {run_dir}\n")

    save_experiment_description(cfg, run_dir, cfg.model_seeds, cfg.data_seed)

    # ------------------------------------------------------------------
    # Step 3: Load data ONCE with the fixed data_seed
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Loading data (fixed data_seed)")
    print("=" * 60)
    cfg.seed = cfg.data_seed
    set_seeds(cfg.data_seed)
    data = load_data(cfg)

    # PCA is deterministic given fixed data — compute once
    pca_results = evaluate_pca_baseline(data, cfg)

    # ------------------------------------------------------------------
    # Step 4: Train loop over model seeds
    # ------------------------------------------------------------------
    # Storage for results across seeds
    std_prefix_curves = {}   # {seed: {k: acc}}
    mat_prefix_curves = {}
    pca_prefix_curves = {}   # single entry, but dict for consistency
    std_embeddings = {}      # {seed: np.ndarray (N_test, embed_dim)}
    mat_embeddings = {}
    all_std_histories = {}   # {seed: history_dict}
    all_mat_histories = {}

    seed_bar = tqdm(enumerate(cfg.model_seeds), total=len(cfg.model_seeds),
                    desc="Seeds", unit="seed", position=0)

    for i, mseed in seed_bar:
        seed_bar.set_description(f"Seed {mseed}")
        print("\n" + "=" * 60)
        print(f"STEP 4.{i+1}: Training with model_seed={mseed}  "
              f"({i+1}/{len(cfg.model_seeds)})")
        print("=" * 60)

        # Create per-seed subfolder for checkpoints and logs
        seed_dir = os.path.join(run_dir, f"seed_{mseed}")
        os.makedirs(seed_dir, exist_ok=True)

        # Set the model seed (data is already loaded and fixed)
        cfg.seed = mseed
        set_seeds(mseed)

        # ---- Train Standard model ----
        print(f"\n--- Training Standard model (seed={mseed}) ---")
        std_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        std_head    = build_head(cfg, data.n_classes)
        std_loss    = build_loss(cfg, "standard")
        std_opt     = torch.optim.Adam(
            list(std_encoder.parameters()) + list(std_head.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        std_history = train(
            std_encoder, std_head, std_loss, std_opt,
            data, cfg, seed_dir, model_tag="standard",
        )
        all_std_histories[mseed] = std_history

        # ---- Train Matryoshka model ----
        print(f"\n--- Training Matryoshka model (seed={mseed}) ---")
        set_seeds(mseed)   # re-seed so Mat init is determined by mseed alone
        mat_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        mat_head    = build_head(cfg, data.n_classes)
        mat_loss    = build_loss(cfg, "matryoshka")
        mat_opt     = torch.optim.Adam(
            list(mat_encoder.parameters()) + list(mat_head.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        mat_history = train(
            mat_encoder, mat_head, mat_loss, mat_opt,
            data, cfg, seed_dir, model_tag="mat",
        )
        all_mat_histories[mseed] = mat_history

        # ---- Prefix sweep ----
        std_prefix_curves[mseed] = evaluate_prefix_sweep(
            std_encoder, std_head, data, cfg, f"standard_seed{mseed}")
        mat_prefix_curves[mseed] = evaluate_prefix_sweep(
            mat_encoder, mat_head, data, cfg, f"mat_seed{mseed}")

        # ---- Store embeddings for correlation analysis ----
        std_encoder.eval()
        mat_encoder.eval()
        with torch.no_grad():
            std_emb = std_encoder(data.X_test).numpy()
            mat_emb = mat_encoder(data.X_test).numpy()
        std_embeddings[mseed] = std_emb
        mat_embeddings[mseed] = mat_emb

        print(f"[main] Seed {mseed} complete.")

    # Use the single PCA result for all "seeds" (it's deterministic)
    pca_prefix_curves[cfg.data_seed] = pca_results

    # ------------------------------------------------------------------
    # Step 5: Aggregate and print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Results summary")
    print("=" * 60)

    prefixes = sorted(cfg.eval_prefixes)

    std_arr = np.array([[std_prefix_curves[s][k] for k in prefixes]
                        for s in sorted(std_prefix_curves.keys())])
    mat_arr = np.array([[mat_prefix_curves[s][k] for k in prefixes]
                        for s in sorted(mat_prefix_curves.keys())])

    pca_accs = [pca_results[k] for k in prefixes]

    # Print table
    header = (f"{'k':>6}  {'Std mean':>10}  {'Std std':>10}  "
              f"{'Mat mean':>10}  {'Mat std':>10}  {'PCA':>8}")
    print(f"\n{header}")
    print("-" * len(header))
    for j, k in enumerate(prefixes):
        print(
            f"{k:>6}  "
            f"{std_arr[:, j].mean():>10.4f}  {std_arr[:, j].std():>10.4f}  "
            f"{mat_arr[:, j].mean():>10.4f}  {mat_arr[:, j].std():>10.4f}  "
            f"{pca_accs[j]:>8.4f}"
        )

    # Save results table
    results_path = os.path.join(run_dir, "results_summary.txt")
    with open(results_path, "w") as f:
        f.write(f"{header}\n")
        f.write("-" * len(header) + "\n")
        for j, k in enumerate(prefixes):
            f.write(
                f"{k:>6}  "
                f"{std_arr[:, j].mean():>10.4f}  {std_arr[:, j].std():>10.4f}  "
                f"{mat_arr[:, j].mean():>10.4f}  {mat_arr[:, j].std():>10.4f}  "
                f"{pca_accs[j]:>8.4f}\n"
            )

        # Also write per-seed raw values
        f.write("\n\nPer-seed raw accuracy values\n")
        f.write("=" * 60 + "\n")
        for label, curves in [("Standard", std_prefix_curves),
                               ("Matryoshka", mat_prefix_curves)]:
            f.write(f"\n{label}:\n")
            f.write(f"{'seed':>8}  " + "  ".join(f"k={k:>3}" for k in prefixes) + "\n")
            for s in sorted(curves.keys()):
                vals = "  ".join(f"{curves[s][k]:.4f}" for k in prefixes)
                f.write(f"{s:>8}  {vals}\n")

    print(f"\n[main] Results table saved to {results_path}")

    # ------------------------------------------------------------------
    # Step 6: Cross-seed dimension correlation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Cross-seed dimension correlation")
    print("=" * 60)

    print("\n--- Standard model ---")
    std_corr = compute_cross_seed_correlation(std_embeddings)
    print(f"  Overall mean |diag corr|: {std_corr['mean_diag_corr']:.4f}")

    print("\n--- Matryoshka model ---")
    mat_corr = compute_cross_seed_correlation(mat_embeddings)
    print(f"  Overall mean |diag corr|: {mat_corr['mean_diag_corr']:.4f}")

    # Save correlation summary
    corr_path = os.path.join(run_dir, "correlation_summary.txt")
    with open(corr_path, "w") as f:
        f.write("Cross-Seed Dimension Correlation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Standard  — mean |diag corr|: {std_corr['mean_diag_corr']:.4f}\n")
        f.write(f"Matryoshka — mean |diag corr|: {mat_corr['mean_diag_corr']:.4f}\n\n")

        f.write("Per-pair breakdown:\n")
        f.write("-" * 50 + "\n")
        f.write("\nStandard:\n")
        for si, sj, dc in std_corr["per_pair"]:
            f.write(f"  seeds ({si}, {sj}): {dc:.4f}\n")
        f.write("\nMatryoshka:\n")
        for si, sj, dc in mat_corr["per_pair"]:
            f.write(f"  seeds ({si}, {sj}): {dc:.4f}\n")

    print(f"[main] Correlation summary saved to {corr_path}")

    # ------------------------------------------------------------------
    # Step 7: Plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Plotting")
    print("=" * 60)

    plot_prefix_stability(
        std_prefix_curves, mat_prefix_curves, pca_prefix_curves, cfg, run_dir)

    plot_variance_comparison(
        std_prefix_curves, mat_prefix_curves, cfg, run_dir)

    plot_correlation_heatmaps(std_corr, mat_corr, run_dir,
                              annotate=(cfg.embed_dim <= 16))

    plot_all_training_curves(all_std_histories, all_mat_histories, run_dir)

    # ------------------------------------------------------------------
    # Step 7b: Low-dim bonus analyses (only when embed_dim <= 16)
    # ------------------------------------------------------------------
    if cfg.embed_dim <= 16:
        print("\n--- Low-dim stability analyses (embed_dim <= 16) ---")

        # Compute metrics
        print("[main] Computing per-dim cosine similarity...")
        std_cos = compute_cosine_similarity(std_embeddings)
        mat_cos = compute_cosine_similarity(mat_embeddings)

        print("[main] Computing per-dim Spearman rank correlation...")
        std_spearman = compute_spearman_correlation(std_embeddings)
        mat_spearman = compute_spearman_correlation(mat_embeddings)

        print("[main] Computing CKA...")
        std_cka = compute_cka(std_embeddings)
        mat_cka = compute_cka(mat_embeddings)

        # Plot
        plot_per_dim_cosine_similarity(std_cos, mat_cos, cfg, run_dir)
        plot_per_dim_spearman(std_spearman, mat_spearman, cfg, run_dir)
        plot_per_dim_correlation(std_corr, mat_corr, cfg, run_dir)
        plot_cka_summary(std_cka, mat_cka, run_dir)
        # plot_tsne_dim_coloring(std_embeddings, mat_embeddings, data, cfg, run_dir)

    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print("\n[main] Experiment 5 complete.")
    print(f"\n[main] Runtime is {time.time() - run_start:.2f} seconds.")
    print(f"[main] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
