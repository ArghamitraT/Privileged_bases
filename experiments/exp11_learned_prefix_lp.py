"""
Script: experiments/exp11_learned_prefix_lp.py
------------------------------------------------
Experiment 11 — LearnedPrefixLp: comparing MRL, PrefixLp (fixed p), and
LearnedPrefixLp (p is a learned scalar nn.Parameter).

Three loss families are compared on the dense prefix sweep task:
  MRL             — Matryoshka loss: CE summed at every prefix scale k=1..embed_dim
  PrefixLp        — CE + front-loaded weighted Lp penalty, p fixed as hyperparameter
  LearnedPrefixLp — same penalty as PrefixLp but p is jointly optimised with the
                    encoder/head via gradient descent

Scientific question: does gradient descent converge to a stable p, and does the
learned p produce better prefix ordering than any fixed p?

Standard and PCA baselines are excluded — this experiment isolates the loss
family comparison.

PrefixLp / LearnedPrefixLp note:
  Dimensions are *reversed* before the prefix sweep (option A — "flip").
  dim 0 receives the highest penalty → least information → put last after flip.
  Legend labels: "PrefixLp (rev)", "LearnedPrefixLp (rev)".

New outputs (exp11-specific):
  p_trajectory_{stamp}.png   : learned p vs epoch with reference lines
  p_and_val_acc_{stamp}.png  : dual-axis p (left) and val accuracy (right) vs epoch

Conda environment: mrl_env

Usage:
    python experiments/exp11_learned_prefix_lp.py           # full run (MNIST, embed_dim=8)
    python experiments/exp11_learned_prefix_lp.py --fast    # smoke test (digits, 3 epochs)
    python tests/run_tests_exp11.py --fast                  # unit tests only
"""

import os

# Cap BLAS thread count before numpy/scipy imports to prevent deadlocks on macOS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import logging
import argparse
import dataclasses

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot, get_path
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss, LearnedPrefixLpLoss

# Reuse helpers from exp7 (no duplication)
from experiments.exp7_mrl_vs_ff import (
    train_single_model,
    get_embeddings_np,
    evaluate_1nn,
    evaluate_prefix_1nn,
    plot_training_curves,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET       = "mnist"
EMBED_DIM     = 8
HIDDEN_DIM    = 256
HEAD_MODE     = "shared_head"
EVAL_PREFIXES = list(range(1, EMBED_DIM + 1))   # always derived from EMBED_DIM
EPOCHS        = 10
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
L1_LAMBDA     = 0.05
P_FIXED       = 5       # exponent for the fixed PrefixLp baseline
P_INIT        = 0.0     # p_raw init → effective p ≈ 1.69 at epoch 0
P_MAX         = 10.0    # p clamped to (1, 1+P_MAX]; safety cap — see LearnedPrefixLpLoss
MAX_1NN_DB    = 10_000  # cap on 1-NN training-set size for speed
# ==============================================================================


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[exp11] Random seeds set to {seed}")


# ==============================================================================
# Custom training loop for LearnedPrefixLp (tracks p and val_acc per epoch)
# ==============================================================================

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _setup_logger(run_dir: str, model_tag: str) -> logging.Logger:
    """Logger that writes to console and {model_tag}_train.log (same format as trainer.py)."""
    logger = logging.getLogger(f"exp11_{model_tag}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if _tqdm is not None:
        class TqdmHandler(logging.StreamHandler):
            def emit(self, record):
                _tqdm.write(self.format(record))
        ch = TqdmHandler()
    else:
        ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(run_dir, f"{model_tag}_train.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def train_learned_p(encoder, head, loss_fn: LearnedPrefixLpLoss,
                    optimiser, data, cfg, run_dir: str, model_tag: str):
    """
    Training loop for LearnedPrefixLp — structurally identical to trainer.train()
    with two additions:
      1. p_trajectory: learned p value logged after every epoch.
      2. val_accs: full-embedding val accuracy logged after every epoch.

    The optimiser MUST include loss_fn.parameters() so p_raw receives gradients.
    Saves best encoder + head checkpoints (by val loss) to run_dir.

    Args:
        encoder   (MLPEncoder)         : Encoder network.
        head      (nn.Module)          : Classifier head.
        loss_fn   (LearnedPrefixLpLoss): Loss with learnable p.
        optimiser (Optimizer)          : Should include loss_fn.parameters().
        data      (DataSplit)          : Train/val/test splits.
        cfg       (ExpConfig)          : Experiment config.
        run_dir   (str)                : Output directory.
        model_tag (str)                : Filename prefix for checkpoints and logs.

    Returns:
        Tuple:
          history      (dict)       : {'train_losses', 'val_losses', 'best_epoch'}
          p_trajectory (List[float]): effective p value at end of each epoch
          val_accs     (List[float]): full-embedding val accuracy at end of each epoch
    """
    logger = _setup_logger(run_dir, model_tag)
    logger.info(f"=== Starting training: {model_tag} (LearnedPrefixLp) ===")
    logger.info(f"  epochs={cfg.epochs}, lr={cfg.lr}, batch={cfg.batch_size}, "
                f"patience={cfg.patience}, p_init(eff)={loss_fn.p.item():.4f}")

    train_ds = TensorDataset(data.X_train, data.y_train)
    val_ds   = TensorDataset(data.X_val,   data.y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    torch.manual_seed(cfg.seed)

    train_losses: list = []
    val_losses:   list = []
    p_trajectory: list = []
    val_accs:     list = []
    best_val_loss      = float("inf")
    best_epoch         = 0
    epochs_no_improve  = 0

    enc_ckpt  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_ckpt = os.path.join(run_dir, f"{model_tag}_head_best.pt")

    epoch_iter = (
        _tqdm(range(cfg.epochs), desc=f"[{model_tag}]", unit="epoch", leave=True)
        if _tqdm is not None else range(cfg.epochs)
    )

    for epoch in epoch_iter:

        # ---- Train phase ----
        encoder.train()
        head.train()
        epoch_train_loss = 0.0
        n_batches = len(train_loader)

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimiser.zero_grad()
            embedding = encoder(X_batch)
            loss      = loss_fn(embedding, y_batch, head)
            loss.backward()
            optimiser.step()
            epoch_train_loss += loss.item() * len(X_batch)

            if _tqdm is not None and (batch_idx + 1) % 10 == 0:
                epoch_iter.set_postfix(
                    batch=f"{batch_idx+1}/{n_batches}",
                    loss=f"{loss.item():.4f}",
                    p=f"{loss_fn.p.item():.3f}",
                )

        epoch_train_loss /= len(data.X_train)
        train_losses.append(epoch_train_loss)

        # ---- Validation phase ----
        encoder.eval()
        head.eval()
        epoch_val_loss = 0.0
        val_logits_all = []
        val_labels_all = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                embedding = encoder(X_batch)
                logits    = head(embedding)
                val_logits_all.append(logits)
                val_labels_all.append(y_batch)

        # val loss: re-run with grad disabled but loss_fn needs p (no grad needed here)
        with torch.no_grad():
            epoch_val_loss = 0.0
            for X_batch, y_batch in val_loader:
                embedding = encoder(X_batch)
                loss      = loss_fn(embedding, y_batch, head)
                epoch_val_loss += loss.item() * len(X_batch)
        epoch_val_loss /= len(data.X_val)
        val_losses.append(epoch_val_loss)

        all_logits = torch.cat(val_logits_all)
        all_labels = torch.cat(val_labels_all)
        val_acc    = _accuracy(all_logits, all_labels)

        # ---- Track p and val_acc ----
        current_p = loss_fn.p.item()
        p_trajectory.append(current_p)
        val_accs.append(val_acc)

        logger.info(
            f"Epoch {epoch+1:>3}/{cfg.epochs}  "
            f"train_loss={epoch_train_loss:.4f}  "
            f"val_loss={epoch_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"p={current_p:.4f}"
        )
        if _tqdm is not None:
            epoch_iter.set_postfix(
                train_loss=f"{epoch_train_loss:.4f}",
                val_loss=f"{epoch_val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
                p=f"{current_p:.3f}",
            )

        # ---- Best checkpoint ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), enc_ckpt)
            torch.save(head.state_dict(),    head_ckpt)
            logger.info(f"  -> New best val_loss={best_val_loss:.4f}  checkpoint saved.")
        else:
            epochs_no_improve += 1

        # ---- Early stopping ----
        if cfg.patience is not None and epochs_no_improve >= cfg.patience:
            logger.info(
                f"Early stopping at epoch {epoch+1} "
                f"(no improvement for {cfg.patience} epochs)."
            )
            break

    # Load best weights back
    encoder.load_state_dict(torch.load(enc_ckpt, weights_only=True))
    head.load_state_dict(   torch.load(head_ckpt, weights_only=True))
    logger.info(f"Best checkpoint loaded from epoch {best_epoch+1}.")
    logger.info(f"Final learned p = {loss_fn.p.item():.4f}")
    logger.info(f"=== Training complete: {model_tag} ===\n")

    history = {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
    }
    return history, p_trajectory, val_accs


# ==============================================================================
# Evaluation helpers
# ==============================================================================

def evaluate_prefix_linear(Z_train, Z_test, y_train, y_test,
                            eval_prefixes, model_tag, seed=42):
    """
    Logistic regression accuracy at each prefix k (dense sweep).

    Args:
        Z_train       (np.ndarray): Train embeddings, shape (n_train, embed_dim).
        Z_test        (np.ndarray): Test embeddings,  shape (n_test,  embed_dim).
        y_train       (np.ndarray): Train labels.
        y_test        (np.ndarray): Test labels.
        eval_prefixes (list[int]) : Prefix sizes to evaluate.
        model_tag     (str)       : Label for print output.
        seed          (int)       : For LogisticRegression reproducibility.

    Returns:
        dict[int, float]: {k: accuracy}
    """
    print(f"\n[exp11] Linear accuracy sweep for '{model_tag}' ...")
    rng = np.random.default_rng(seed)
    max_train = 10_000
    if len(Z_train) > max_train:
        idx      = rng.choice(len(Z_train), max_train, replace=False)
        Ztr_sub  = Z_train[idx]
        ytr_sub  = y_train[idx]
    else:
        Ztr_sub, ytr_sub = Z_train, y_train

    results = {}
    for k in eval_prefixes:
        lr = LogisticRegression(solver="saga", max_iter=1000,
                                random_state=seed, n_jobs=1)
        lr.fit(Ztr_sub[:, :k], ytr_sub)
        acc = float(lr.score(Z_test[:, :k], y_test))
        results[k] = acc
        print(f"  k={k:>3}  linear={acc:.4f}")
    return results


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(cfg, run_dir, fast, p_fixed, p_init, p_max):
    """Write a human-readable log describing this experiment run."""
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 11 — LearnedPrefixLp: MRL vs PrefixLp vs LearnedPrefixLp\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains three loss families and evaluates each at EVERY prefix k\n"
            "from 1 to embed_dim (dense sweep):\n"
            "  MRL            : CE summed at every prefix scale (Matryoshka loss)\n"
            "  PrefixLp       : CE + front-loaded weighted Lp penalty (fixed p)\n"
            "  LearnedPrefixLp: same penalty as PrefixLp but p is a learned\n"
            "                   scalar nn.Parameter optimised with the encoder/head\n\n"
            "New outputs vs exp10:\n"
            "  p_trajectory.png   : learned p vs epoch with reference lines\n"
            "  p_and_val_acc.png  : dual-axis p (left) + val accuracy (right) vs epoch\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Exp10 showed that PrefixLp with different fixed p values produces\n"
            "different ordering quality. If p can be learned, we ask:\n"
            "  - Does gradient descent converge to a stable p?\n"
            "  - Does the learned p match our best hand-tuned value?\n"
            "  - Does the p trajectory reveal how ordering pressure evolves?\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "  MRL            : high accuracy even at small k; smooth monotone curve\n"
            "  PrefixLp       : good ordering at small k after dim reversal\n"
            "  LearnedPrefixLp: p converges to a stable value; ordering quality\n"
            "                   at least as good as fixed p baseline\n\n"
        )

        f.write("LEARNED P CONFIG\n")
        f.write("-" * 40 + "\n")
        f.write(f"  P_FIXED  = {p_fixed}   (exponent for fixed PrefixLp baseline)\n")
        f.write(f"  P_INIT   = {p_init}   (p_raw init; effective p_init ≈ 1.69)\n")
        f.write(f"  P_MAX    = {p_max}  (clamp: effective p ∈ (1, {1+p_max}])\n\n")

        f.write(f"  Fast mode: {fast}\n\n")

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[exp11] Experiment description saved to {log_path}")


# ==============================================================================
# Plots — p trajectory and joint p+val_acc
# ==============================================================================

def plot_p_trajectory(p_trajectory, p_fixed, run_dir, fig_stamp=""):
    """
    Plot the learned p value vs epoch with reference lines.

    Args:
        p_trajectory (List[float]): p value at end of each training epoch.
        p_fixed      (int/float)  : Fixed p of the PrefixLp baseline (reference line).
        run_dir      (str)        : Output directory.
        fig_stamp    (str)        : Timestamp suffix for the filename.
    """
    epochs = list(range(1, len(p_trajectory) + 1))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, p_trajectory, "b-o", linewidth=2, markersize=4, label="Learned p")

    # Reference lines for common fixed-p values
    for ref_p, style, color in [(1, "--", "gray"), (2, ":", "gray"),
                                 (3, "-.", "gray")]:
        ax.axhline(y=ref_p, linestyle=style, color=color, linewidth=1,
                   label=f"p={ref_p} (reference)", alpha=0.6)
    # Highlight the fixed-p baseline used in this experiment
    if p_fixed not in (1, 2, 3):
        ax.axhline(y=p_fixed, linestyle="--", color="crimson", linewidth=1.5,
                   label=f"p={p_fixed} (PrefixLp baseline)", alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learned p", fontsize=12)
    ax.set_title("LearnedPrefixLp — p value vs epoch", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = f"p_trajectory{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp11] Saved {fname}")


def plot_p_and_val_acc(p_trajectory, val_accs, run_dir, fig_stamp=""):
    """
    Dual-axis plot: learned p (left y-axis, blue) and full-embedding val accuracy
    (right y-axis, red) vs epoch.

    Shows whether accuracy improvements correlate with p rising or settling.

    Args:
        p_trajectory (List[float]): p value at end of each epoch.
        val_accs     (List[float]): Full-embedding val accuracy at end of each epoch.
        run_dir      (str)        : Output directory.
        fig_stamp    (str)        : Timestamp suffix for the filename.
    """
    epochs = list(range(1, len(p_trajectory) + 1))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 4))

    color_p   = "steelblue"
    color_acc = "crimson"

    ax1.plot(epochs, p_trajectory, color=color_p, marker="o", linewidth=2,
             markersize=4, label="Learned p")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Learned p", fontsize=12, color=color_p)
    ax1.tick_params(axis="y", labelcolor=color_p)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, color=color_acc, marker="s", linewidth=2,
             markersize=4, linestyle="--", label="Val accuracy (full embed)")
    ax2.set_ylabel("Val accuracy (full embedding)", fontsize=12, color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")

    fig.suptitle("LearnedPrefixLp — p and val accuracy vs epoch", fontsize=13)
    plt.tight_layout()

    fname = f"p_and_val_acc{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp11] Saved {fname}")


# ==============================================================================
# Accuracy curves (reused from exp10 pattern, adapted for 3 models)
# ==============================================================================

_MODEL_STYLES = {
    "MRL":                 ("darkorange", "s-",  "MRL"),
    "PrefixLp (rev)":      ("crimson",    "^-",  "PrefixLp (rev)"),
    "LearnedPrefixLp (rev)": ("mediumpurple", "D-", "LearnedPrefixLp (rev)"),
}


def _single_accuracy_plot(results_dict, eval_prefixes, ylabel, title,
                           out_path, fig_stamp=""):
    if fig_stamp:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}{fig_stamp}{ext}"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, acc_dict in results_dict.items():
        color, style, label = _MODEL_STYLES.get(
            model_name, ("gray", "x-", model_name))
        accs = [acc_dict.get(k, float("nan")) for k in eval_prefixes]
        ax.plot(eval_prefixes, accs, style, color=color,
                label=label, linewidth=2, markersize=4)
    ax.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, eval_prefixes[-1])
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp11] Saved {os.path.basename(out_path)}")


def plot_all_curves(linear_results, nn1_results, eval_prefixes, run_dir, fig_stamp=""):
    """Save linear accuracy, 1-NN accuracy, and combined 2-panel plots."""
    _single_accuracy_plot(
        linear_results, eval_prefixes,
        ylabel="Linear Classification Accuracy",
        title="Linear Accuracy vs Prefix k  (MRL vs PrefixLp vs LearnedPrefixLp — dense)",
        out_path=os.path.join(run_dir, "linear_accuracy_curve.png"),
        fig_stamp=fig_stamp,
    )
    _single_accuracy_plot(
        nn1_results, eval_prefixes,
        ylabel="1-NN Accuracy",
        title="1-NN Accuracy vs Prefix k  (MRL vs PrefixLp vs LearnedPrefixLp — dense)",
        out_path=os.path.join(run_dir, "1nn_accuracy_curve.png"),
        fig_stamp=fig_stamp,
    )

    # Combined 2-panel
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for model_name in linear_results:
        color, style, label = _MODEL_STYLES.get(
            model_name, ("gray", "x-", model_name))
        lin_accs = [linear_results[model_name].get(k, float("nan")) for k in eval_prefixes]
        nn1_accs = [nn1_results[model_name].get(k,    float("nan")) for k in eval_prefixes]
        ax_top.plot(eval_prefixes, lin_accs, style, color=color,
                    label=label, linewidth=2, markersize=4)
        ax_bot.plot(eval_prefixes, nn1_accs, style, color=color,
                    label=label, linewidth=2, markersize=4)
    for ax in (ax_top, ax_bot):
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1, eval_prefixes[-1])
        ax.legend(fontsize=10)
    ax_top.set_ylabel("Linear Accuracy", fontsize=12)
    ax_top.set_title("Linear Classification Accuracy vs Prefix k  (dense)", fontsize=12)
    ax_bot.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax_bot.set_ylabel("1-NN Accuracy", fontsize=12)
    ax_bot.set_title("1-NN Accuracy vs Prefix k  (dense)", fontsize=12)
    fig.suptitle("MRL vs PrefixLp vs LearnedPrefixLp — Dense Prefix Sweep",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    combined_name = f"combined_comparison{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, combined_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp11] Saved {combined_name}")


# ==============================================================================
# Results table
# ==============================================================================

def save_results_summary(linear_results, nn1_results, eval_prefixes, run_dir,
                          p_trajectory, p_fixed):
    """
    Write plain-text table of linear + 1-NN accuracy per (model, k),
    plus a LearnedP summary section.
    """
    model_names = list(linear_results.keys())
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 11 — LearnedPrefixLp Results\n")
        f.write("=" * 60 + "\n\n")

        # Learned p summary
        f.write("LEARNED P SUMMARY\n")
        f.write("-" * 40 + "\n")
        if p_trajectory:
            f.write(f"  p_init (epoch 0) = {p_trajectory[0]:.4f}\n")
            f.write(f"  p_final          = {p_trajectory[-1]:.4f}\n")
            f.write(f"  p_fixed baseline = {p_fixed}\n")
            f.write(f"  p_trajectory     = {[round(v, 4) for v in p_trajectory]}\n")
        else:
            f.write("  (no trajectory — training skipped)\n")
        f.write("\n")

        # Accuracy table
        header = f"{'k':>4}  {'Model':<22}  {'Linear Acc':>12}  {'1-NN Acc':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for k in eval_prefixes:
            for model_name in model_names:
                lin = linear_results[model_name].get(k, float("nan"))
                nn1 = nn1_results[model_name].get(k,    float("nan"))
                f.write(f"{k:>4}  {model_name:<22}  {lin:>12.4f}  {nn1:>10.4f}\n")
            f.write("\n")

    print(f"[exp11] Results summary saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Experiment 11 — LearnedPrefixLp.

    Steps:
      1-3  : Parse args, build config, setup output dir
      4    : Load data
      5    : Train MRL
      6    : Train PrefixLp (fixed p = P_FIXED)
      7    : Train LearnedPrefixLp (p is learned)
      8    : Plot training curves
      9    : Plot p trajectory + joint p/val_acc
      10   : Extract embeddings (PrefixLp + LearnedPrefixLp reversed)
      11   : Linear accuracy sweep
      12   : 1-NN accuracy sweep
      13   : Plot accuracy curves + save results summary
      14   : Runtime + code snapshot
    """
    run_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Parse args
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Exp11 — LearnedPrefixLp")
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test: digits dataset, 3 epochs, small 1-NN DB")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2: Build config
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset="digits", embed_dim=8, hidden_dim=128,
            head_mode="shared_head", eval_prefixes=list(range(1, 9)),
            lr=LR, epochs=3, batch_size=BATCH_SIZE, patience=3,
            weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
            experiment_name="exp11_learned_prefix_lp",
        )
        p_fixed   = P_FIXED
        p_init    = P_INIT
        p_max     = P_MAX
        max_1nn_db = 500
    else:
        cfg = ExpConfig(
            dataset=DATASET, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
            head_mode=HEAD_MODE, eval_prefixes=EVAL_PREFIXES,
            lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
            weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
            experiment_name="exp11_learned_prefix_lp",
        )
        p_fixed   = P_FIXED
        p_init    = P_INIT
        p_max     = P_MAX
        max_1nn_db = MAX_1NN_DB

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Step 3: Setup output directory
    # ------------------------------------------------------------------
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"[exp11] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, args.fast, p_fixed, p_init, p_max)

    # ------------------------------------------------------------------
    # Step 4: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # ------------------------------------------------------------------
    # Step 5: Train MRL
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Training MRL model")
    print("=" * 60)
    mrl_encoder, mrl_head = train_single_model(
        cfg, data, run_dir, model_type="matryoshka", model_tag="mat"
    )

    # ------------------------------------------------------------------
    # Step 6: Train PrefixLp (fixed p)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 6: Training PrefixLp model  (p={p_fixed}, lambda={cfg.l1_lambda})")
    print("=" * 60)
    pl_encoder, pl_head = train_single_model(
        cfg, data, run_dir,
        model_type=f"prefix_l{p_fixed}", model_tag=f"pl{p_fixed}",
    )

    # ------------------------------------------------------------------
    # Step 7: Train LearnedPrefixLp (p is learned)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 7: Training LearnedPrefixLp  "
          f"(p_init(eff)≈1.69, p_max={p_max}, lambda={cfg.l1_lambda})")
    print("=" * 60)
    lp_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    lp_head    = build_head(cfg, data.n_classes)
    lp_loss    = LearnedPrefixLpLoss(
        embed_dim=cfg.embed_dim, lambda_l1=cfg.l1_lambda,
        p_init=p_init, p_max=p_max,
    )
    lp_opt = torch.optim.Adam(
        list(lp_encoder.parameters()) +
        list(lp_head.parameters()) +
        list(lp_loss.parameters()),    # includes p_raw
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    lp_history, p_trajectory, val_accs = train_learned_p(
        lp_encoder, lp_head, lp_loss, lp_opt, data, cfg, run_dir, "lp_learned"
    )
    lp_encoder.eval()
    lp_head.eval()
    print(f"[exp11] LearnedPrefixLp final p = {lp_loss.p.item():.4f}")

    # ------------------------------------------------------------------
    # Step 8: Training curves (MANDATORY)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Plotting training curves")
    print("=" * 60)
    plot_training_curves(
        run_dir,
        model_tags=["mat", f"pl{p_fixed}", "lp_learned"],
        fig_stamp=fig_stamp,
    )

    # ------------------------------------------------------------------
    # Step 9: p trajectory plots (exp11-specific)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Plotting p trajectory")
    print("=" * 60)
    plot_p_trajectory(p_trajectory, p_fixed, run_dir, fig_stamp=fig_stamp)
    plot_p_and_val_acc(p_trajectory, val_accs, run_dir, fig_stamp=fig_stamp)

    # ------------------------------------------------------------------
    # Step 10: Extract embeddings (PrefixLp + LearnedPrefixLp reversed)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 10: Extracting embeddings")
    print("=" * 60)
    Z_train_mrl = get_embeddings_np(mrl_encoder, data.X_train)
    Z_test_mrl  = get_embeddings_np(mrl_encoder, data.X_test)
    print(f"[exp11] MRL: train={Z_train_mrl.shape}, test={Z_test_mrl.shape}")

    # PrefixLp — reverse: lightest-penalty dim (last) becomes dim 0
    Z_train_pl = get_embeddings_np(pl_encoder, data.X_train)
    Z_test_pl  = get_embeddings_np(pl_encoder, data.X_test)
    Z_train_pl = np.ascontiguousarray(Z_train_pl[:, ::-1])
    Z_test_pl  = np.ascontiguousarray(Z_test_pl[:,  ::-1])
    print(f"[exp11] PrefixLp: train={Z_train_pl.shape}, reversed (most informative first)")

    # LearnedPrefixLp — same reversal convention
    Z_train_lp = get_embeddings_np(lp_encoder, data.X_train)
    Z_test_lp  = get_embeddings_np(lp_encoder, data.X_test)
    Z_train_lp = np.ascontiguousarray(Z_train_lp[:, ::-1])
    Z_test_lp  = np.ascontiguousarray(Z_test_lp[:,  ::-1])
    print(f"[exp11] LearnedPrefixLp: train={Z_train_lp.shape}, reversed")

    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 11: Linear accuracy (dense prefix sweep)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 11: Evaluating linear accuracy  (k=1..{cfg.embed_dim})")
    print("=" * 60)
    mrl_lin = evaluate_prefix_linear(
        Z_train_mrl, Z_test_mrl, y_train_np, y_test_np,
        cfg.eval_prefixes, "MRL", seed=cfg.seed,
    )
    pl_lin = evaluate_prefix_linear(
        Z_train_pl, Z_test_pl, y_train_np, y_test_np,
        cfg.eval_prefixes, f"PrefixL{p_fixed}", seed=cfg.seed,
    )
    lp_lin = evaluate_prefix_linear(
        Z_train_lp, Z_test_lp, y_train_np, y_test_np,
        cfg.eval_prefixes, "LearnedPrefixLp", seed=cfg.seed,
    )
    linear_results = {
        "MRL":                   mrl_lin,
        "PrefixLp (rev)":        pl_lin,
        "LearnedPrefixLp (rev)": lp_lin,
    }

    # ------------------------------------------------------------------
    # Step 12: 1-NN accuracy (dense prefix sweep)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 12: Evaluating 1-NN accuracy  (k=1..{cfg.embed_dim})")
    print("=" * 60)
    mrl_1nn = evaluate_prefix_1nn(
        mrl_encoder, data, cfg.eval_prefixes, "MRL",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )

    # PrefixLp 1-NN: use pre-flipped embeddings
    print(f"\n[exp11] 1-NN sweep for 'PrefixLp (rev)' ...")
    pl_1nn = {}
    for k in cfg.eval_prefixes:
        acc = evaluate_1nn(
            Z_train_pl[:, :k], Z_test_pl[:, :k],
            y_train_np, y_test_np,
            max_db_samples=max_1nn_db, seed=cfg.seed,
        )
        pl_1nn[k] = acc
        print(f"  k={k:>3}  1-NN={acc:.4f}")

    print(f"\n[exp11] 1-NN sweep for 'LearnedPrefixLp (rev)' ...")
    lp_1nn = {}
    for k in cfg.eval_prefixes:
        acc = evaluate_1nn(
            Z_train_lp[:, :k], Z_test_lp[:, :k],
            y_train_np, y_test_np,
            max_db_samples=max_1nn_db, seed=cfg.seed,
        )
        lp_1nn[k] = acc
        print(f"  k={k:>3}  1-NN={acc:.4f}")

    nn1_results = {
        "MRL":                   mrl_1nn,
        "PrefixLp (rev)":        pl_1nn,
        "LearnedPrefixLp (rev)": lp_1nn,
    }

    # ------------------------------------------------------------------
    # Step 13: Plots + results table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 13: Saving plots and results")
    print("=" * 60)
    plot_all_curves(linear_results, nn1_results, cfg.eval_prefixes, run_dir,
                    fig_stamp=fig_stamp)
    save_results_summary(linear_results, nn1_results, cfg.eval_prefixes, run_dir,
                         p_trajectory, p_fixed)

    # Compact stdout table
    sample_ks   = [k for k in cfg.eval_prefixes if k % 4 == 0 or k == 1]
    model_names = list(linear_results.keys())
    print(f"\n{'k':>4}  {'Model':<24}  {'Linear':>8}  {'1-NN':>8}")
    print("-" * 50)
    for k in sample_ks:
        for model_name in model_names:
            lin = linear_results[model_name].get(k, float("nan"))
            nn1 = nn1_results[model_name].get(k,    float("nan"))
            print(f"{k:>4}  {model_name:<24}  {lin:>8.4f}  {nn1:>8.4f}")
        print()

    # ------------------------------------------------------------------
    # Step 14: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 14: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp11] Experiment 11 complete.")
    print(f"[exp11] All outputs saved to: {run_dir}")
    print(f"[exp11] Final learned p = {lp_loss.p.item():.4f}")


if __name__ == "__main__":
    main()
