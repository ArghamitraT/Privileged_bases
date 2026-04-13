"""
Script: experiments/exp1_prefix_curve.py
-----------------------------------------
Experiment 1 — Prefix Performance Curve.

Trains three models on the configured dataset and evaluates each by sweeping
over prefix sizes k ∈ cfg.eval_prefixes:

    1. Standard model  : MLP encoder + classifier, trained with plain CE loss.
    2. Matryoshka model: same architecture, trained with MRL loss (CE summed
                         at every prefix scale).
    3. PCA baseline    : PCA on training data + linear probe at each k.

The expected result: Matryoshka accuracy stays high at small k; Standard drops
off; PCA falls somewhere in between.

All outputs (logs, weights, config snapshot, plot) are saved to a timestamped
folder under:
    Mat_embedding_hyperbole/files/results/exp1_prefix_curve_{timestamp}/

Conda environment: mrl_env

Usage:
    python experiments/exp1_prefix_curve.py

To change settings, edit config.py — that file is the single source of truth
for all hyperparameters. The experiment reads defaults from there via ExpConfig().

Inputs:  ExpConfig (defaults loaded from config.py)
Outputs: Per-run folder containing —
           standard_train.log, mat_train.log  : training logs
           standard_encoder_best.pt           : best standard encoder weights
           standard_head_best.pt              : best standard head weights
           mat_encoder_best.pt                : best Mat encoder weights
           mat_head_best.pt                   : best Mat head weights
           config.txt                         : snapshot of the config used
           prefix_curve.png                   : the main result plot
           training_curves.png                : train/val loss per epoch for both models
"""

import os
import sys
import time
import dataclasses

import torch
import matplotlib.pyplot as plt

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
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET       = "mnist"
EMBED_DIM     = 64
HIDDEN_DIM    = 256
HEAD_MODE     = "shared_head"
EVAL_PREFIXES = [1, 2, 4, 8, 16, 32, 64]
EPOCHS        = 20
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
# ==============================================================================


# ==============================================================================
# Helpers
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for reproducibility across numpy, torch.

    Args:
        seed (int): Master random seed from cfg.seed.
    """
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[main] Random seeds set to {seed}")


def save_experiment_description(cfg: ExpConfig, run_dir: str):
    """
    Write a human-readable experiment log describing what this run is doing,
    why it is being run, and what the expected outcome is. Also appends the
    full config so the folder is entirely self-contained.

    This file is the first thing to read when revisiting an old run.

    Args:
        cfg     (ExpConfig): The experiment configuration.
        run_dir (str)      : Path to the run output directory.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 1 — Prefix Performance Curve\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains two MLP models (Standard and Matryoshka) that both produce a\n"
            f"{cfg.embed_dim}-dimensional embedding from {cfg.dataset} input features.\n"
            "At test time, evaluates classification accuracy using only the first k\n"
            "embedding dimensions (zeroing out the rest) for each k in:\n"
            f"  {cfg.eval_prefixes}\n\n"
            "A PCA baseline is also included: PCA components are naturally ordered\n"
            "by explained variance, giving it a built-in dimension ordering.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Matryoshka training explicitly optimises the model so that early\n"
            "embedding dimensions are already informative on their own. Standard\n"
            "training has no such constraint — information may be spread arbitrarily\n"
            "across all dimensions. This experiment tests whether MRL successfully\n"
            "induces a meaningful ordering (a 'privileged basis'), which would be\n"
            "evidence of improved interpretability.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "- Matryoshka: accuracy stays high even at small k (early dims informative).\n"
            "- Standard:   accuracy drops sharply at small k (information spread randomly).\n"
            "- PCA:        somewhere in between (variance ordering, but no task-awareness).\n"
            "- At k = embed_dim (all dims): all three models perform similarly.\n\n"
        )

        f.write("MODELS\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Architecture : MLP  input={cfg.dataset} -> hidden={cfg.hidden_dim} -> embed={cfg.embed_dim}\n"
            f"  Head mode    : {cfg.head_mode}\n"
            f"  Standard loss: CrossEntropy on full embedding\n"
            f"  Mat loss     : Sum of CrossEntropy at each prefix scale\n\n"
        )

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[main] Experiment description saved to {log_path}")


def plot_prefix_curve(
    standard_results: dict,
    mat_results: dict,
    pca_results: dict,
    cfg: ExpConfig,
    run_dir: str,
):
    """
    Plot the prefix performance curve and save to the run directory.

    X-axis: prefix size k (log scale).
    Y-axis: classification accuracy.
    Three lines: Standard, Matryoshka, PCA.

    Args:
        standard_results (dict): {k: accuracy} for the standard model.
        mat_results      (dict): {k: accuracy} for the Matryoshka model.
        pca_results      (dict): {k: accuracy} for the PCA baseline.
        cfg              (ExpConfig): Used for title/label info.
        run_dir          (str): Where to save the PNG.
    """
    prefixes = sorted(standard_results.keys())

    std_accs = [standard_results[k] for k in prefixes]
    mat_accs = [mat_results[k]      for k in prefixes]
    pca_accs = [pca_results[k]      for k in prefixes]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(prefixes, std_accs, "o-",  color="steelblue",   label="Standard",     linewidth=2, markersize=7)
    ax.plot(prefixes, mat_accs, "s-",  color="darkorange",  label="Matryoshka",   linewidth=2, markersize=7)
    ax.plot(prefixes, pca_accs, "^--", color="seagreen",    label="PCA baseline", linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(prefixes)
    ax.set_xticklabels([str(k) for k in prefixes])
    ax.set_xlabel("Prefix size k  (number of embedding dimensions used)", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(
        f"Prefix Performance Curve\n"
        f"Dataset: {cfg.dataset}  |  embed_dim={cfg.embed_dim}  |  "
        f"head_mode={cfg.head_mode}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "prefix_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[main] Plot saved to {plot_path}")


def plot_training_curves(
    std_history: dict,
    mat_history: dict,
    run_dir: str,
):
    """
    Plot train and validation loss per epoch for both models and save to disk.

    Gives a quick visual check that both models converged and early stopping
    fired at a sensible point. The best-epoch marker shows which checkpoint
    was saved and loaded for evaluation.

    Args:
        std_history (dict): Return value of train() for the standard model.
                            Keys: 'train_losses', 'val_losses', 'best_epoch'.
        mat_history (dict): Same structure for the Matryoshka model.
        run_dir     (str) : Directory where the PNG is saved.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, history, title, color in [
        (axes[0], std_history, "Standard model",    "steelblue"),
        (axes[1], mat_history, "Matryoshka model",  "darkorange"),
    ]:
        epochs     = range(1, len(history["train_losses"]) + 1)
        best_epoch = history["best_epoch"] + 1   # 1-indexed for display

        ax.plot(epochs, history["train_losses"], "-",  color=color,  label="Train loss",      linewidth=2)
        ax.plot(epochs, history["val_losses"],   "--", color=color,  label="Val loss",  alpha=0.7, linewidth=2)

        # Mark the checkpoint epoch
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Best epoch ({best_epoch})")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)

    plt.suptitle("Training curves — train and validation loss", fontsize=14, y=1.02)
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
    # Step 1: Configure the experiment
    # All settings live in the CONFIG block above — edit there, not here.
    # ------------------------------------------------------------------
    cfg = ExpConfig(
        dataset       = DATASET,
        embed_dim     = EMBED_DIM,
        hidden_dim    = HIDDEN_DIM,
        head_mode     = HEAD_MODE,
        eval_prefixes = EVAL_PREFIXES,
        lr            = LR,
        epochs        = EPOCHS,
        batch_size    = BATCH_SIZE,
        patience      = PATIENCE,
        weight_decay  = WEIGHT_DECAY,
        seed          = SEED,
        experiment_name = "exp1_prefix_curve",
    )
    run_start = time.time()   # record wall-clock start for runtime.txt

    # ------------------------------------------------------------------
    # Step 2: Setup — seeds, run directory, config snapshot
    # ------------------------------------------------------------------
    set_seeds(cfg.seed)

    run_dir = create_run_dir()
    print(f"[main] Outputs will be saved to: {run_dir}\n")

    save_experiment_description(cfg, run_dir)

    # ------------------------------------------------------------------
    # Step 3: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # ------------------------------------------------------------------
    # Step 4: Train the standard model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Training standard model")
    print("=" * 60)

    std_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    std_head    = build_head(cfg, data.n_classes)
    std_loss    = build_loss(cfg, "standard")
    std_opt     = torch.optim.Adam(
        list(std_encoder.parameters()) + list(std_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    std_history = train(
        std_encoder, std_head, std_loss, std_opt,
        data, cfg, run_dir, model_tag="standard",
    )
    print(f"[main] Standard model best epoch: {std_history['best_epoch'] + 1}")

    # ------------------------------------------------------------------
    # Step 5: Train the Matryoshka model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Training Matryoshka model")
    print("=" * 60)

    mat_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    mat_head    = build_head(cfg, data.n_classes)
    mat_loss    = build_loss(cfg, "matryoshka")
    mat_opt     = torch.optim.Adam(
        list(mat_encoder.parameters()) + list(mat_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    mat_history = train(
        mat_encoder, mat_head, mat_loss, mat_opt,
        data, cfg, run_dir, model_tag="mat",
    )
    print(f"[main] Matryoshka model best epoch: {mat_history['best_epoch'] + 1}")

    # ------------------------------------------------------------------
    # Step 6: Evaluate — prefix sweep for all three models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Prefix sweep evaluation")
    print("=" * 60)

    std_results = evaluate_prefix_sweep(std_encoder, std_head, data, cfg, "standard")
    mat_results = evaluate_prefix_sweep(mat_encoder, mat_head, data, cfg, "mat")
    pca_results = evaluate_pca_baseline(data, cfg)

    # ------------------------------------------------------------------
    # Step 7: Print summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Results summary")
    print("=" * 60)
    print(f"\n{'k':>6}  {'Standard':>10}  {'Matryoshka':>12}  {'PCA':>8}")
    print("-" * 44)
    for k in sorted(std_results.keys()):
        print(
            f"{k:>6}  "
            f"{std_results[k]:>10.4f}  "
            f"{mat_results[k]:>12.4f}  "
            f"{pca_results[k]:>8.4f}"
        )

    # Save results as a text file too
    results_path = os.path.join(run_dir, "results_summary.txt")
    with open(results_path, "w") as f:
        f.write(f"{'k':>6}  {'Standard':>10}  {'Matryoshka':>12}  {'PCA':>8}\n")
        f.write("-" * 44 + "\n")
        for k in sorted(std_results.keys()):
            f.write(
                f"{k:>6}  "
                f"{std_results[k]:>10.4f}  "
                f"{mat_results[k]:>12.4f}  "
                f"{pca_results[k]:>8.4f}\n"
            )
    print(f"\n[main] Results table saved to {results_path}")

    # ------------------------------------------------------------------
    # Step 8: Plot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Plotting prefix performance curve")
    print("=" * 60)

    plot_prefix_curve(std_results, mat_results, pca_results, cfg, run_dir)
    plot_training_curves(std_history, mat_history, run_dir)

    # ------------------------------------------------------------------
    # Step 9: Save runtime and full code snapshot for reproducibility
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print("\n[main] Experiment 1 complete.")
    print(f"\n[main] Runtime is {time.time() - run_start:.2f} seconds.")
    print(f"[main] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
