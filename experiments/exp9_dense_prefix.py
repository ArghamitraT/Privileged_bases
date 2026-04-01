"""
Script: experiments/exp9_dense_prefix.py
-----------------------------------------
Experiment 9 — Dense Prefix Evaluation with Multiple Data Seeds.

Trains Standard, L1, MRL, and PCA models and evaluates them at every
single embedding dimension k = 1, 2, ..., embed_dim (dense). The key
question: does MRL's first-k accuracy closely match the oracle best-k
accuracy at all k, while Standard and L1 show a persistent gap?

Run over two data seeds (42, 123) to confirm robustness across data splits.

Models:
  Standard — plain CE loss on full embed_dim embedding
  L1       — CE + λ·‖z‖₁ (sparsity, no ordering)
  MRL      — CE summed at every prefix scale (ordering enforced)
  PCA      — analytical baseline (variance-ordered components)

Analyses per seed (all inherited from exp8, imported directly):
  1. Prefix accuracy curve: accuracy vs k (1..embed_dim, smooth dense curve)
  2. Best-k vs First-k: oracle top-k dims vs standard prefix for each model
  3. Method agreement: Spearman rho between 3 importance methods

Cross-seed output:
  gap_comparison.png — gap(best_k - first_k) per model for both seeds overlaid

Inputs:
  --fast : smoke test — digits, embed_dim=16, 5 epochs, 1 seed (42)

Outputs (all in a new timestamped run folder):
  training_curves.png          : loss vs epoch for all models x both seeds
  seed_42/
    standard_encoder_best.pt   : saved weights
    l1_encoder_best.pt
    mat_encoder_best.pt
    prefix_accuracy.png        : accuracy vs k (1..64) for all 4 models
    best_vs_first_k.png        : first-k vs best-k per model (3 importance methods)
    method_agreement.png       : Spearman rho scatter plots
  seed_123/ (same structure)
  gap_comparison.png           : best_k - first_k gap for both seeds per model
  results_summary.txt
  experiment_description.log
  runtime.txt
  code_snapshot/

Usage:
    python experiments/exp9_dense_prefix.py --fast   # smoke test (digits, 5 epochs, 1 seed)
    python experiments/exp9_dense_prefix.py          # full run (MNIST, 20 epochs, 2 seeds)
    python tests/run_tests_exp9.py --fast            # unit tests only
    python tests/run_tests_exp9.py                   # unit tests + e2e smoke
"""

import os

# Cap BLAS thread count BEFORE importing numpy/scipy/sklearn to prevent
# deadlocks on macOS (scipy/sklearn can conflict with OpenBLAS thread pool).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse
import dataclasses

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

# Add code/ to path so all project imports resolve regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot
from data.loader import load_data

# Reuse training + embedding helpers from exp7 (no duplication)
from experiments.exp7_mrl_vs_ff import train_single_model, get_embeddings_np

# Reuse all analysis + one plotting function from exp8 (no duplication)
from experiments.exp8_dim_importance import (
    compute_importance_scores,
    compute_best_vs_first_k,
    compute_method_agreement,
    get_pca_embeddings_np,
    plot_importance_scores,
    plot_method_agreement,
    IMPORTANCE_METHODS,
    METHOD_PAIRS,
    METHOD_LABELS,
    PAIR_LABELS,
)


# ==============================================================================
# Module-level constants
# ==============================================================================

# The two data seeds to run.  Seed 42 matches all prior experiments.
DATA_SEEDS = [42, 123]

MODEL_NAMES = ["Standard", "L1", "MRL", "PCA"]

MODEL_COLORS = {
    "Standard": "steelblue",
    "L1":       "orchid",
    "MRL":      "darkorange",
    "PCA":      "seagreen",
}

# Tags used when saving weight files and training logs
MODEL_TAGS = {
    "Standard": "standard",
    "L1":       "l1",
    "MRL":      "mat",
}


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for numpy, torch, and python random.

    Args:
        seed (int): Master random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[exp9] Random seeds set to {seed}")


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(cfg: ExpConfig, run_dir: str,
                                  fast: bool, data_seeds: list):
    """
    Write a human-readable log describing this experiment run.

    Args:
        cfg        (ExpConfig): Experiment configuration.
        run_dir    (str)      : Root output directory.
        fast       (bool)     : Whether fast/smoke mode is active.
        data_seeds (list)     : Data seeds used in this run.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 9 — Dense Prefix Evaluation + Multi-Seed Best-k vs First-k\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains Standard, L1, MRL, and PCA models and evaluates each at\n"
            "EVERY dimension k = 1..embed_dim (dense prefix sweep).\n"
            "Run over two data seeds to test robustness across data splits.\n\n"
            "Three analyses per seed:\n"
            "  1. Prefix accuracy curve: accuracy vs k (smooth, dense)\n"
            "  2. Best-k vs First-k: oracle top-k vs standard prefix (3 methods)\n"
            "  3. Method agreement: Spearman rho between importance methods\n\n"
            "Cross-seed: gap_comparison.png overlays both seeds per model.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Exp7/8 used sparse prefixes (powers of 2). Dense eval shows the\n"
            "ordering story at every dimension — the gap between best-k and\n"
            "first-k is visible continuously, not just at 7 checkpoints.\n"
            "Multiple data seeds confirm the findings are not split artifacts.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "  MRL:      gap(best_k - first_k) ≈ 0 at all k  (ordering enforced)\n"
            "  Standard: gap > 0 at small k, converges at large k\n"
            "  L1:       gap between Standard and MRL\n"
            "  PCA:      gap ≈ 0 (variance ordering = best variance ordering)\n"
            "  Pattern stable across both data seeds.\n\n"
        )

        f.write("CONFIG\n")
        f.write("-" * 40 + "\n")
        f.write(f"  fast mode     : {fast}\n")
        f.write(f"  data_seeds    : {data_seeds}\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[exp9] Experiment description saved to {log_path}")


# ==============================================================================
# Training curves (mandatory) — combined for all seeds
# ==============================================================================

def _parse_train_log(log_path: str):
    """
    Parse a trainer log file and extract per-epoch train/val loss values.

    Args:
        log_path (str): Path to a {tag}_train.log file.

    Returns:
        Tuple (train_losses, val_losses): both list[float], empty if not found.
    """
    train_losses, val_losses = [], []
    if not os.path.isfile(log_path):
        return train_losses, val_losses
    with open(log_path) as fh:
        for line in fh:
            if "train_loss=" in line and "val_loss=" in line:
                try:
                    tl = float(line.split("train_loss=")[1].split()[0])
                    vl = float(line.split("val_loss=")[1].split()[0])
                    train_losses.append(tl)
                    val_losses.append(vl)
                except (IndexError, ValueError):
                    pass
    return train_losses, val_losses


def plot_training_curves(run_dir: str, seed_dirs: dict, data_seeds: list):
    """
    Plot combined training curves for all models across all seeds.

    Layout: rows = seeds, columns = models (Standard / L1 / MRL).
    Each cell shows train + val loss vs epoch.

    Args:
        run_dir    (str)  : Root output directory (where the PNG is saved).
        seed_dirs  (dict) : {seed (int): seed_dir (str)} mapping.
        data_seeds (list) : Ordered list of seeds (determines row order).
    """
    model_tags  = list(MODEL_TAGS.values())   # ["standard", "l1", "mat"]
    n_rows = len(data_seeds)
    n_cols = len(model_tags)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    any_data = False
    for row_idx, seed in enumerate(data_seeds):
        seed_dir = seed_dirs.get(seed)
        for col_idx, tag in enumerate(model_tags):
            ax = axes[row_idx, col_idx]
            if seed_dir is None:
                ax.axis("off")
                continue

            train_l, val_l = _parse_train_log(
                os.path.join(seed_dir, f"{tag}_train.log")
            )
            if not train_l:
                ax.text(0.5, 0.5, "no log", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.axis("off")
                continue

            any_data = True
            epochs = range(1, len(train_l) + 1)
            ax.plot(epochs, train_l, label="Train", linewidth=2)
            ax.plot(epochs, val_l,   label="Val",   linewidth=2, linestyle="--")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Loss",  fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            if row_idx == 0:
                ax.set_title(tag.upper(), fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"Seed {seed}\nLoss", fontsize=9)

    if not any_data:
        # Fallback placeholder
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No training logs found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle("Training Curves — Standard / L1 / MRL  (rows = data seeds)",
                 fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp9] Saved training_curves.png")


# ==============================================================================
# Per-seed plots
# ==============================================================================

def plot_prefix_accuracy(gap_results: dict, seed_dir: str,
                          eval_prefixes: list, seed: int):
    """
    Plot accuracy vs k (dense) for all 4 models for a single seed.

    X-axis is linear (not log) since we evaluate at every k.
    Y-axis is accuracy [0, 1].

    Args:
        gap_results   (dict)     : {model_name: {"first_k": {k: acc}, ...}}.
        seed_dir      (str)      : Where to save the PNG.
        eval_prefixes (list[int]): Dense k values (1..embed_dim).
        seed          (int)      : Data seed (for title).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name in MODEL_NAMES:
        accs = [gap_results[model_name]["first_k"].get(k, float("nan"))
                for k in eval_prefixes]
        ax.plot(eval_prefixes, accs,
                color=MODEL_COLORS[model_name],
                label=model_name, linewidth=2)

    # Mark every 8th tick to keep x-axis readable
    tick_step = max(1, len(eval_prefixes) // 8)
    shown_ticks = [k for k in eval_prefixes if (k - 1) % tick_step == 0]
    shown_ticks.append(eval_prefixes[-1])
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels([str(k) for k in shown_ticks], fontsize=9)

    ax.set_xlabel("Prefix size k (number of embedding dims used)", fontsize=11)
    ax.set_ylabel("Logistic Regression Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Prefix Accuracy Curve — Dense Eval  (data seed {seed})",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(seed_dir, "prefix_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp9] Saved seed_{seed}/prefix_accuracy.png")


def plot_best_vs_first_k_dense(gap_results: dict, seed_dir: str,
                                 eval_prefixes: list, seed: int):
    """
    Plot best-k vs first-k accuracy for all 4 models (dense x-axis).

    Layout: 1 row × 4 columns (one per model).
    Each panel shows 4 lines:
      - first_k (black solid): standard prefix eval
      - best_k_mean_abs  (blue dashed)
      - best_k_variance  (red dashed)
      - best_k_probe_acc (green dashed)

    Uses a linear x-axis (appropriate for dense eval).

    Args:
        gap_results   (dict)     : {model_name: {curve_key: {k: acc}}}.
        seed_dir      (str)      : Where to save the PNG.
        eval_prefixes (list[int]): Dense k values (1..embed_dim).
        seed          (int)      : Data seed (for title).
    """
    CURVE_STYLES = {
        "first_k":          ("black",     "-",  "First-k (prefix eval)",  2.5),
        "best_k_mean_abs":  ("steelblue", "--", "Best-k (mean |z|)",      1.8),
        "best_k_variance":  ("firebrick", "--", "Best-k (variance)",      1.8),
        "best_k_probe_acc": ("seagreen",  "--", "Best-k (probe acc)",     1.8),
    }

    n_models = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(6 * n_models, 5),
                              sharey=True, squeeze=False)

    # Readable x-tick subset
    tick_step  = max(1, len(eval_prefixes) // 8)
    shown_ticks = [k for k in eval_prefixes if (k - 1) % tick_step == 0]
    shown_ticks.append(eval_prefixes[-1])

    for col, model_name in enumerate(MODEL_NAMES):
        ax    = axes[0, col]
        color = MODEL_COLORS[model_name]
        data  = gap_results[model_name]

        for curve_key, (c, ls, label, lw) in CURVE_STYLES.items():
            accs = [data[curve_key].get(k, float("nan")) for k in eval_prefixes]
            ax.plot(eval_prefixes, accs, linestyle=ls, color=c,
                    label=label, linewidth=lw)

        ax.set_xticks(shown_ticks)
        ax.set_xticklabels([str(k) for k in shown_ticks], fontsize=8)
        ax.set_xlabel("Prefix size k", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(model_name, fontsize=12, color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("Logistic Regression Accuracy", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"Best-k vs First-k Accuracy — Dense Eval  (data seed {seed})\n"
        f"gap ≈ 0 for MRL → ordering enforced",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = os.path.join(seed_dir, "best_vs_first_k.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp9] Saved seed_{seed}/best_vs_first_k.png")


# ==============================================================================
# Cross-seed gap comparison
# ==============================================================================

def plot_gap_comparison(all_seed_results: dict, run_dir: str,
                         eval_prefixes: list, data_seeds: list):
    """
    Plot best_k - first_k gap for both seeds overlaid on the same axes.

    Layout: 1 row × 4 models.
    Each panel shows 2 lines (one per seed), where the gap is the mean
    of the 3 importance methods' gaps. A shaded band spans [min_gap, max_gap]
    across the 3 methods to show method uncertainty.

    Args:
        all_seed_results (dict) : {seed: {"gap_results": {model: {...}}}}.
        run_dir          (str)  : Root output dir.
        eval_prefixes    (list) : Dense k values.
        data_seeds       (list) : Seeds in the run.
    """
    n_models   = len(MODEL_NAMES)
    seed_styles = {
        data_seeds[0]: ("-",  "solid"),
        data_seeds[1]: ("--", "dashed"),
    } if len(data_seeds) > 1 else {data_seeds[0]: ("-", "solid")}

    fig, axes = plt.subplots(1, n_models,
                              figsize=(6 * n_models, 5),
                              sharey=True, squeeze=False)

    tick_step   = max(1, len(eval_prefixes) // 8)
    shown_ticks = [k for k in eval_prefixes if (k - 1) % tick_step == 0]
    shown_ticks.append(eval_prefixes[-1])

    for col, model_name in enumerate(MODEL_NAMES):
        ax    = axes[0, col]
        color = MODEL_COLORS[model_name]

        for seed in data_seeds:
            if seed not in all_seed_results:
                continue
            gap_res = all_seed_results[seed]["gap_results"][model_name]
            first_k_vals = np.array([gap_res["first_k"][k] for k in eval_prefixes])

            # Compute gap for each method
            method_gaps = []
            for method in IMPORTANCE_METHODS:
                best_vals = np.array([gap_res[f"best_k_{method}"][k]
                                       for k in eval_prefixes])
                method_gaps.append(best_vals - first_k_vals)

            method_gaps = np.array(method_gaps)  # (3, n_k)
            mean_gap    = method_gaps.mean(axis=0)
            min_gap     = method_gaps.min(axis=0)
            max_gap     = method_gaps.max(axis=0)

            ls = seed_styles[seed][0]
            ax.plot(eval_prefixes, mean_gap,
                    linestyle=ls, color=color, linewidth=2,
                    label=f"seed {seed}")
            # Shaded band = spread across the 3 methods
            ax.fill_between(eval_prefixes, min_gap, max_gap,
                             color=color, alpha=0.12)

        # Zero-gap reference line
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

        ax.set_xticks(shown_ticks)
        ax.set_xticklabels([str(k) for k in shown_ticks], fontsize=8)
        ax.set_xlabel("Prefix size k", fontsize=10)
        ax.set_title(model_name, fontsize=12, color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("Gap  =  best_k_acc − first_k_acc", fontsize=10)
            ax.legend(fontsize=9)

    fig.suptitle(
        "Gap (best_k − first_k) Across Data Seeds  "
        "(mean of 3 methods, shading = method spread)\n"
        "Gap ≈ 0 for MRL at all k; large gap for Standard/L1 at small k",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = os.path.join(run_dir, "gap_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp9] Saved gap_comparison.png")


# ==============================================================================
# Results summary table
# ==============================================================================

def save_results_summary(all_seed_results: dict, eval_prefixes: list,
                          run_dir: str, data_seeds: list):
    """
    Write results_summary.txt with accuracy and gap tables for all seeds.

    Table 1 (per seed): first-k accuracy for all 4 models at each k.
    Table 2 (per seed): best_k - first_k gap per model per method at each k.
    Table 3 (per seed): Spearman method agreement per model.

    Args:
        all_seed_results (dict) : {seed: {"gap_results": ..., "agreement": ...}}.
        eval_prefixes    (list) : Dense k values.
        run_dir          (str)  : Root output directory.
        data_seeds       (list) : Seeds in the run.
    """
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 9 — Dense Prefix + Multi-Seed Results\n")
        f.write("=" * 72 + "\n\n")

        for seed in data_seeds:
            if seed not in all_seed_results:
                continue
            res      = all_seed_results[seed]
            gap_res  = res["gap_results"]
            agreement = res["agreement"]

            f.write(f"=" * 72 + "\n")
            f.write(f"DATA SEED {seed}\n")
            f.write(f"=" * 72 + "\n\n")

            # --- Table 1: first-k accuracy per model ---
            f.write("TABLE 1: First-k (Prefix) Accuracy\n")
            f.write("-" * 60 + "\n")
            hdr = f"{'k':>4}  " + "  ".join(f"{m:<12}" for m in MODEL_NAMES)
            f.write(hdr + "\n")
            f.write("-" * len(hdr) + "\n")
            # Sample every 4th k to keep the table readable
            sample_ks = [k for k in eval_prefixes if (k - 1) % 4 == 0]
            sample_ks.append(eval_prefixes[-1])
            for k in sample_ks:
                row = f"{k:>4}  "
                row += "  ".join(
                    f"{gap_res[m]['first_k'].get(k, float('nan')):>12.4f}"
                    for m in MODEL_NAMES
                )
                f.write(row + "\n")
            f.write("\n")

            # --- Table 2: gap per model per method (sampled) ---
            f.write("TABLE 2: Gap = best_k_acc − first_k_acc  "
                    "(mean across 3 methods)\n")
            f.write("-" * 60 + "\n")
            hdr2 = f"{'k':>4}  " + "  ".join(f"{m:<12}" for m in MODEL_NAMES)
            f.write(hdr2 + "\n")
            f.write("-" * len(hdr2) + "\n")
            for k in sample_ks:
                row = f"{k:>4}  "
                for m in MODEL_NAMES:
                    first_v = gap_res[m]["first_k"].get(k, float("nan"))
                    gaps_at_k = [
                        gap_res[m][f"best_k_{mth}"].get(k, float("nan")) - first_v
                        for mth in IMPORTANCE_METHODS
                    ]
                    mean_gap = float(np.mean(gaps_at_k))
                    row += f"  {mean_gap:>+12.4f}"
                f.write(row + "\n")
            f.write("\n")

            # --- Table 3: Spearman agreement ---
            f.write("TABLE 3: Spearman rho Between Importance Methods\n")
            f.write("-" * 60 + "\n")
            pair_labels = [PAIR_LABELS[p] for p in METHOD_PAIRS]
            hdr3 = f"{'Model':<12}  " + "  ".join(f"{pl:>20}" for pl in pair_labels)
            f.write(hdr3 + "\n")
            f.write("-" * len(hdr3) + "\n")
            for model_name in MODEL_NAMES:
                rhos = [agreement[model_name].get(p, float("nan"))
                        for p in METHOD_PAIRS]
                rho_strs = "  ".join(f"rho={r:>+6.3f}" for r in rhos)
                f.write(f"{model_name:<12}  {rho_strs}\n")
            f.write("\n\n")

    print(f"[exp9] Results summary saved to {path}")


# ==============================================================================
# Single-seed pipeline
# ==============================================================================

def run_one_seed(seed: int, cfg: ExpConfig, run_dir: str,
                  fast: bool, max_probe_samples: int,
                  max_lr_samples: int) -> dict:
    """
    Run the full training + analysis pipeline for a single data seed.

    Creates a subdirectory seed_{seed}/ inside run_dir, trains all three
    neural models there, then runs importance scoring and best-k vs first-k
    analysis. All per-seed outputs are saved in the subdirectory.

    Args:
        seed             (int)      : Data seed (controls train/test split).
        cfg              (ExpConfig): Base experiment config.
        run_dir          (str)      : Root output directory.
        fast             (bool)     : Whether fast/smoke mode is active.
        max_probe_samples(int)      : Cap on samples for per-dim LR probe.
        max_lr_samples   (int)      : Cap on training samples for best-k LR fits.

    Returns:
        dict with keys:
            "seed_dir"   (str)  : Path to the seed subdirectory.
            "gap_results"(dict) : {model: {curve_key: {k: acc}}}.
            "scores"     (dict) : {model: {method: np.ndarray}}.
            "agreement"  (dict) : {model: {(ma, mb): rho}}.
    """
    # ------------------------------------------------------------------
    # Create per-seed output directory
    # ------------------------------------------------------------------
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    print(f"\n[exp9] ===== DATA SEED {seed}  →  {seed_dir} =====\n")

    # ------------------------------------------------------------------
    # Build a per-seed config: only the data seed changes
    # ------------------------------------------------------------------
    seed_cfg = ExpConfig(
        dataset          = cfg.dataset,
        embed_dim        = cfg.embed_dim,
        hidden_dim       = cfg.hidden_dim,
        head_mode        = cfg.head_mode,
        eval_prefixes    = cfg.eval_prefixes,
        epochs           = cfg.epochs,
        patience         = cfg.patience,
        lr               = cfg.lr,
        weight_decay     = cfg.weight_decay,
        batch_size       = cfg.batch_size,
        val_size         = cfg.val_size,
        seed             = seed,           # model init seed = data seed here
        l1_lambda        = cfg.l1_lambda,
        experiment_name  = cfg.experiment_name,
    )

    set_seeds(seed)

    # ------------------------------------------------------------------
    # Load data with this seed's split
    # ------------------------------------------------------------------
    print(f"[exp9] Loading data  (seed={seed}) ...")
    data = load_data(seed_cfg)
    print(f"[exp9] Data loaded: train={data.X_train.shape}, "
          f"val={data.X_val.shape}, test={data.X_test.shape}")

    # ------------------------------------------------------------------
    # Train Standard, L1, MRL (saves weights + logs to seed_dir)
    # ------------------------------------------------------------------
    print(f"\n[exp9] Training Standard model  (seed={seed}) ...")
    std_enc, std_hd = train_single_model(
        seed_cfg, data, seed_dir, model_type="standard", model_tag="standard"
    )

    print(f"\n[exp9] Training L1 model  (lambda={seed_cfg.l1_lambda}, seed={seed}) ...")
    l1_enc,  l1_hd  = train_single_model(
        seed_cfg, data, seed_dir, model_type="l1", model_tag="l1"
    )

    print(f"\n[exp9] Training MRL model  (seed={seed}) ...")
    mat_enc, mat_hd = train_single_model(
        seed_cfg, data, seed_dir, model_type="matryoshka", model_tag="mat"
    )

    # ------------------------------------------------------------------
    # Extract embeddings for all 4 models
    # ------------------------------------------------------------------
    print(f"\n[exp9] Extracting embeddings  (seed={seed}) ...")
    Z_train_std = get_embeddings_np(std_enc, data.X_train)
    Z_test_std  = get_embeddings_np(std_enc, data.X_test)

    Z_train_l1  = get_embeddings_np(l1_enc,  data.X_train)
    Z_test_l1   = get_embeddings_np(l1_enc,  data.X_test)

    Z_train_mrl = get_embeddings_np(mat_enc, data.X_train)
    Z_test_mrl  = get_embeddings_np(mat_enc, data.X_test)

    Z_train_pca, Z_test_pca = get_pca_embeddings_np(data, seed_cfg)

    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    print(f"[exp9] Embedding shapes: std={Z_test_std.shape}, "
          f"l1={Z_test_l1.shape}, mrl={Z_test_mrl.shape}, pca={Z_test_pca.shape}")

    all_embeddings = {
        "Standard": (Z_train_std, Z_test_std),
        "L1":       (Z_train_l1,  Z_test_l1),
        "MRL":      (Z_train_mrl, Z_test_mrl),
        "PCA":      (Z_train_pca, Z_test_pca),
    }

    # ------------------------------------------------------------------
    # Subsample training data for best-k LR fits (speed-up)
    # Dense eval = 64 k values × 4 curve types = 256 LR fits per model.
    # Capping training samples keeps each fit under ~1 second.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    if len(y_train_np) > max_lr_samples:
        lr_idx   = rng.choice(len(y_train_np), max_lr_samples, replace=False)
        print(f"[exp9] LR fits: subsampling train to {max_lr_samples} "
              f"(from {len(y_train_np)}) for speed.")
    else:
        lr_idx   = np.arange(len(y_train_np))

    # ------------------------------------------------------------------
    # Analysis 1: Importance scoring (3 methods per model)
    # ------------------------------------------------------------------
    print(f"\n[exp9] Analysis 1: Importance scoring  (seed={seed}) ...")
    all_scores = {}
    for model_name, (Z_train, Z_test) in all_embeddings.items():
        all_scores[model_name] = compute_importance_scores(
            Z_test=Z_test, Z_train=Z_train,
            y_train=y_train_np, y_test=y_test_np,
            max_probe_samples=max_probe_samples,
            seed=seed, model_tag=model_name,
        )

    # ------------------------------------------------------------------
    # Analysis 2: Best-k vs First-k  (uses subsampled train)
    # ------------------------------------------------------------------
    print(f"\n[exp9] Analysis 2: Best-k vs First-k  (seed={seed}) ...")
    all_gap_results = {}
    for model_name, (Z_train, Z_test) in all_embeddings.items():
        Z_train_sub = Z_train[lr_idx]
        y_train_sub = y_train_np[lr_idx]
        all_gap_results[model_name] = compute_best_vs_first_k(
            Z_train=Z_train_sub, Z_test=Z_test,
            y_train=y_train_sub, y_test=y_test_np,
            importance_scores=all_scores[model_name],
            eval_prefixes=seed_cfg.eval_prefixes,
            seed=seed, model_tag=model_name,
        )

    # ------------------------------------------------------------------
    # Analysis 3: Method agreement (Spearman rho)
    # ------------------------------------------------------------------
    print(f"\n[exp9] Analysis 3: Method agreement  (seed={seed}) ...")
    all_agreement = {}
    for model_name in MODEL_NAMES:
        all_agreement[model_name] = compute_method_agreement(
            all_scores[model_name], model_tag=model_name
        )

    # ------------------------------------------------------------------
    # Per-seed plots
    # ------------------------------------------------------------------
    print(f"\n[exp9] Saving per-seed plots  (seed={seed}) ...")

    plot_prefix_accuracy(all_gap_results, seed_dir,
                          seed_cfg.eval_prefixes, seed)

    plot_best_vs_first_k_dense(all_gap_results, seed_dir,
                                 seed_cfg.eval_prefixes, seed)

    # Reuse exp8's method_agreement plot (saves method_agreement.png to seed_dir)
    plot_method_agreement(all_scores, all_agreement, seed_dir, seed_cfg)

    return {
        "seed_dir":    seed_dir,
        "gap_results": all_gap_results,
        "scores":      all_scores,
        "agreement":   all_agreement,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Orchestrate Experiment 9:
      1.  Parse args, build config, create run_dir
      2.  Save experiment_description.log
      3.  For each data seed:
            a. Train Standard, L1, MRL on that data split
            b. Run all 3 analyses
            c. Save per-seed plots
      4.  Plot combined training_curves.png
      5.  Plot gap_comparison.png (cross-seed)
      6.  Save results_summary.txt
      7.  Save runtime.txt and code_snapshot/
    """
    run_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Experiment 9 — Dense Prefix + Multi-Seed Best-k vs First-k"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: digits, embed_dim=16, 5 epochs, 1 seed (42).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2: Config
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset       = "digits",
            embed_dim     = 16,
            hidden_dim    = 128,
            head_mode     = "shared_head",
            eval_prefixes = list(range(1, 17)),   # dense: 1..16
            epochs        = 5,
            patience      = 3,
            seed          = 42,
            l1_lambda     = 0.05,
            experiment_name = "exp9_dense_prefix",
        )
        data_seeds        = [42]          # single seed for smoke test
        max_probe_samples = 500
        max_lr_samples    = 1000
    else:
        cfg = ExpConfig(
            dataset       = "mnist",
            embed_dim     = 64,
            hidden_dim    = 256,
            head_mode     = "shared_head",
            eval_prefixes = list(range(1, 65)),   # dense: 1..64
            epochs        = 20,
            patience      = 5,
            seed          = 42,
            l1_lambda     = 0.05,
            experiment_name = "exp9_dense_prefix",
        )
        data_seeds        = DATA_SEEDS    # [42, 123]
        max_probe_samples = 2000
        max_lr_samples    = 10000

    # ------------------------------------------------------------------
    # Step 3: Create output directory + save description
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[exp9] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, args.fast, data_seeds)

    # ------------------------------------------------------------------
    # Step 4: Run pipeline for each data seed
    # ------------------------------------------------------------------
    all_seed_results = {}
    seed_dirs        = {}

    for seed in data_seeds:
        result = run_one_seed(
            seed             = seed,
            cfg              = cfg,
            run_dir          = run_dir,
            fast             = args.fast,
            max_probe_samples = max_probe_samples,
            max_lr_samples   = max_lr_samples,
        )
        all_seed_results[seed] = result
        seed_dirs[seed]        = result["seed_dir"]

    # ------------------------------------------------------------------
    # Step 5: Combined training curves (all seeds × all models)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Plotting combined training curves")
    print("=" * 60)
    plot_training_curves(run_dir, seed_dirs, data_seeds)

    # ------------------------------------------------------------------
    # Step 6: Cross-seed gap comparison
    # ------------------------------------------------------------------
    if len(data_seeds) > 1:
        print("=" * 60)
        print("STEP 6: Cross-seed gap comparison")
        print("=" * 60)
        plot_gap_comparison(
            all_seed_results, run_dir, cfg.eval_prefixes, data_seeds
        )
    else:
        print("[exp9] Only one seed — skipping gap_comparison.png")

    # ------------------------------------------------------------------
    # Step 7: Results summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Saving results summary")
    print("=" * 60)
    save_results_summary(all_seed_results, cfg.eval_prefixes, run_dir, data_seeds)

    # Quick accuracy table to stdout
    print(f"\n{'Seed':>6}  {'k':>4}  " +
          "  ".join(f"{m:<12}" for m in MODEL_NAMES))
    print("-" * 60)
    sample_ks = [1, 4, 8, 16, 32, cfg.embed_dim] if not args.fast else [1, 4, 8, 16]
    sample_ks = [k for k in sample_ks if k in cfg.eval_prefixes]
    for seed in data_seeds:
        gap_res = all_seed_results[seed]["gap_results"]
        for k in sample_ks:
            row = f"{seed:>6}  {k:>4}  "
            row += "  ".join(
                f"{gap_res[m]['first_k'].get(k, float('nan')):>12.4f}"
                for m in MODEL_NAMES
            )
            print(row)
        print()

    # ------------------------------------------------------------------
    # Step 8: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp9] Experiment 9 complete.")
    print(f"[exp9] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
