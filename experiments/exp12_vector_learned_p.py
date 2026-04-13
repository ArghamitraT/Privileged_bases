"""
Script: experiments/exp12_vector_learned_p.py
----------------------------------------------
Experiment 12 — VectorLearnedPrefixLp: MRL vs ScalarLearnedPrefixLp vs VectorLearnedPrefixLp.

Three loss families compared on dense prefix sweep + per-dim importance analysis:
  MRL                  — Matryoshka loss: CE summed at every prefix scale
  ScalarLearnedPrefixLp — same penalty as PrefixLp; one shared p learned for all dims
  VectorLearnedPrefixLp — same penalty; one p learned independently per dim

Scientific questions:
  - Do different dims converge to different p values?
  - Do high-penalty dims (large dim_weight) learn higher or lower p?
  - Does per-dim p improve prefix ordering over scalar p?
  - Do the three models agree on which dims are most important?

Dimension reversal:
  Both Lp models reverse before eval — same convention as exp10/exp11.
  Legend labels: "ScalarLearnedPrefixLp (rev)", "VectorLearnedPrefixLp (rev)".

New outputs vs exp11:
  scalar_p_trajectory_{stamp}.png     : single p line vs epoch
  scalar_p_and_val_acc_{stamp}.png    : dual-axis p + val acc
  vector_p_trajectory_{stamp}.png     : embed_dim lines (one per dim) vs epoch
  vector_p_and_val_acc_{stamp}.png    : mean p across dims + val acc dual-axis
  importance_scores_{stamp}.png       : per-dim importance (3 methods x 3 models)
  method_agreement_{stamp}.png        : Spearman rho scatter per model

Conda environment: mrl_env

Usage:
    python experiments/exp12_vector_learned_p.py           # full run (MNIST, embed_dim=8)
    python experiments/exp12_vector_learned_p.py --fast    # smoke test (digits, 3 epochs)
    python tests/run_tests_exp12.py --fast                 # unit tests only
"""

import os

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
from utility import create_run_dir, save_runtime, save_code_snapshot
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import LearnedPrefixLpLoss, VectorLearnedPrefixLpLoss

# Shared helpers from exp7 (no duplication)
from experiments.exp7_mrl_vs_ff import (
    train_single_model,
    get_embeddings_np,
    evaluate_1nn,
    evaluate_prefix_1nn,
    plot_training_curves,
)

# Importance scoring + method agreement from exp8
from experiments.exp8_dim_importance import (
    compute_importance_scores,
    compute_method_agreement,
    plot_importance_scores,
    plot_method_agreement,
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
EVAL_PREFIXES = list(range(1, EMBED_DIM + 1))
EPOCHS        = 10
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
L1_LAMBDA     = 0.05
P_INIT        = 15.0     # p_raw init for both scalar and vector models → eff p ≈ 1.69
P_MAX         = 10.0    # clamp: p ∈ (1, 1+P_MAX]; safety cap
MAX_1NN_DB    = 10_000
MAX_PROBE     = 10_000  # cap on samples for per-dim logistic probe (exp8)
# ==============================================================================

# Colours for exp12's three models — used in importance plots and accuracy curves
_MODEL_COLORS = {
    "MRL":                           "darkorange",
    "ScalarLearnedPrefixLp (rev)":   "mediumpurple",
    "VectorLearnedPrefixLp (rev)":   "teal",
}
_MODEL_STYLES = {
    "MRL":                           ("darkorange",    "s-"),
    "ScalarLearnedPrefixLp (rev)":   ("mediumpurple",  "D-"),
    "VectorLearnedPrefixLp (rev)":   ("teal",          "^-"),
}


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
    print(f"[exp12] Random seeds set to {seed}")


# ==============================================================================
# Shared training loop for both scalar and vector LearnedPrefixLp
# ==============================================================================

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def _setup_logger(run_dir: str, model_tag: str) -> logging.Logger:
    logger = logging.getLogger(f"exp12_{model_tag}")
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


def train_learned_p(encoder, head, loss_fn, optimiser, data, cfg,
                    run_dir: str, model_tag: str):
    """
    Training loop for scalar or vector LearnedPrefixLp models.

    Works for both LearnedPrefixLpLoss (p scalar) and VectorLearnedPrefixLpLoss
    (p vector of shape embed_dim) — detects shape at runtime.

    Tracks p_trajectory (list of numpy arrays, one per epoch) and
    val_accs (full-embedding val accuracy per epoch).

    Returns:
        history      (dict)            : train_losses, val_losses, best_epoch
        p_trajectory (list[np.ndarray]): p value(s) per epoch
        val_accs     (list[float])     : full-embedding val accuracy per epoch
    """
    logger = _setup_logger(run_dir, model_tag)
    logger.info(f"=== Starting training: {model_tag} ===")
    logger.info(f"  epochs={cfg.epochs}, lr={cfg.lr}, batch={cfg.batch_size}, "
                f"patience={cfg.patience}")

    train_ds = TensorDataset(data.X_train, data.y_train)
    val_ds   = TensorDataset(data.X_val,   data.y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    torch.manual_seed(cfg.seed)

    train_losses: list = []
    val_losses:   list = []
    p_trajectory: list = []
    val_accs:     list = []
    best_val_loss     = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0

    enc_ckpt  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_ckpt = os.path.join(run_dir, f"{model_tag}_head_best.pt")

    epoch_iter = (
        _tqdm(range(cfg.epochs), desc=f"[{model_tag}]", unit="epoch", leave=True)
        if _tqdm is not None else range(cfg.epochs)
    )

    for epoch in epoch_iter:
        # ---- Train ----
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
                p_summary = _p_summary(loss_fn)
                epoch_iter.set_postfix(
                    batch=f"{batch_idx+1}/{n_batches}",
                    loss=f"{loss.item():.4f}",
                    p=p_summary,
                )

        epoch_train_loss /= len(data.X_train)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        encoder.eval()
        head.eval()
        val_logits_all = []
        val_labels_all = []
        epoch_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                embedding = encoder(X_batch)
                val_logits_all.append(head(embedding))
                val_labels_all.append(y_batch)

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                loss = loss_fn(encoder(X_batch), y_batch, head)
                epoch_val_loss += loss.item() * len(X_batch)
        epoch_val_loss /= len(data.X_val)
        val_losses.append(epoch_val_loss)

        val_acc = _accuracy(torch.cat(val_logits_all), torch.cat(val_labels_all))
        val_accs.append(val_acc)

        # ---- Track p (scalar → 0-d array, vector → 1-d array) ----
        p_np = loss_fn.p.detach().cpu().numpy()
        p_trajectory.append(p_np.copy())

        p_summary = _p_summary(loss_fn)
        logger.info(
            f"Epoch {epoch+1:>3}/{cfg.epochs}  "
            f"train_loss={epoch_train_loss:.4f}  "
            f"val_loss={epoch_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"p={p_summary}"
        )
        if _tqdm is not None:
            epoch_iter.set_postfix(
                train_loss=f"{epoch_train_loss:.4f}",
                val_loss=f"{epoch_val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
                p=p_summary,
            )

        # ---- Checkpoint ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), enc_ckpt)
            torch.save(head.state_dict(),    head_ckpt)
            logger.info(f"  -> New best val_loss={best_val_loss:.4f}  checkpoint saved.")
        else:
            epochs_no_improve += 1

        if cfg.patience is not None and epochs_no_improve >= cfg.patience:
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

    encoder.load_state_dict(torch.load(enc_ckpt, weights_only=True))
    head.load_state_dict(   torch.load(head_ckpt, weights_only=True))
    logger.info(f"Best checkpoint loaded from epoch {best_epoch+1}.")
    logger.info(f"Final p: {_p_summary(loss_fn)}")
    logger.info(f"=== Training complete: {model_tag} ===\n")

    history = {"train_losses": train_losses, "val_losses": val_losses,
               "best_epoch": best_epoch}
    return history, p_trajectory, val_accs


def _p_summary(loss_fn) -> str:
    """Compact string: scalar → '1.69', vector → 'mean=1.72 min=1.60 max=1.85'."""
    p = loss_fn.p.detach().cpu()
    if p.ndim == 0:
        return f"{p.item():.3f}"
    return f"mean={p.mean():.3f} min={p.min():.3f} max={p.max():.3f}"


# ==============================================================================
# Linear accuracy evaluation
# ==============================================================================

def evaluate_prefix_linear(Z_train, Z_test, y_train, y_test,
                            eval_prefixes, model_tag, seed=42):
    """Logistic regression accuracy at each prefix k (dense sweep)."""
    print(f"\n[exp12] Linear accuracy sweep for '{model_tag}' ...")
    rng = np.random.default_rng(seed)
    max_train = 10_000
    if len(Z_train) > max_train:
        idx = rng.choice(len(Z_train), max_train, replace=False)
        Ztr_sub, ytr_sub = Z_train[idx], y_train[idx]
    else:
        Ztr_sub, ytr_sub = Z_train, y_train

    results = {}
    for k in eval_prefixes:
        lr = LogisticRegression(solver="saga", max_iter=1000,
                                random_state=seed, n_jobs=1)
        lr.fit(Ztr_sub[:, :k], ytr_sub)
        results[k] = float(lr.score(Z_test[:, :k], y_test))
        print(f"  k={k:>3}  linear={results[k]:.4f}")
    return results


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(cfg, run_dir, fast, p_init, p_max):
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 12 — VectorLearnedPrefixLp\n")
        f.write("=" * 70 + "\n\n")
        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains three loss families and evaluates each at EVERY prefix k\n"
            "from 1 to embed_dim (dense sweep):\n"
            "  MRL                   : CE summed at every prefix scale\n"
            "  ScalarLearnedPrefixLp : CE + weighted Lp, one shared p learned\n"
            "  VectorLearnedPrefixLp : CE + weighted Lp, one p per dim learned\n\n"
            "Additional analysis from exp8:\n"
            "  importance_scores.png : per-dim importance (mean|z|, var, probe_acc)\n"
            "  method_agreement.png  : Spearman rho between importance methods\n\n"
        )
        f.write("LEARNED P CONFIG\n")
        f.write("-" * 40 + "\n")
        f.write(f"  P_INIT = {p_init}  (raw init; all dims start at eff p ≈ 1.69)\n")
        f.write(f"  P_MAX  = {p_max} (clamp: p ∈ (1, {1+p_max}])\n\n")
        f.write(f"  Fast mode: {fast}\n\n")
        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")
    print(f"[exp12] Experiment description saved to {log_path}")


# ==============================================================================
# p trajectory plots
# ==============================================================================

def plot_scalar_p_trajectory(p_trajectory, run_dir, fig_stamp=""):
    """Single p line vs epoch for ScalarLearnedPrefixLp."""
    p_vals = [arr.item() for arr in p_trajectory]
    epochs = list(range(1, len(p_vals) + 1))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, p_vals, color="mediumpurple", marker="o",
            linewidth=2, markersize=4, label="Scalar learned p")
    for ref_p, style, alpha in [(1, "--", 0.5), (2, ":", 0.5), (3, "-.", 0.5)]:
        ax.axhline(y=ref_p, linestyle=style, color="gray",
                   linewidth=1, alpha=alpha, label=f"p={ref_p}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learned p", fontsize=12)
    ax.set_title("ScalarLearnedPrefixLp — p vs epoch", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"scalar_p_trajectory{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp12] Saved {fname}")


def plot_scalar_p_and_val_acc(p_trajectory, val_accs, run_dir, fig_stamp=""):
    """Dual-axis: scalar p (left) + val accuracy (right) vs epoch."""
    p_vals = [arr.item() for arr in p_trajectory]
    epochs = list(range(1, len(p_vals) + 1))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, p_vals, color="mediumpurple", marker="o",
             linewidth=2, markersize=4, label="Scalar p")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Learned p", fontsize=12, color="mediumpurple")
    ax1.tick_params(axis="y", labelcolor="mediumpurple")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, color="crimson", marker="s",
             linewidth=2, markersize=4, linestyle="--", label="Val accuracy")
    ax2.set_ylabel("Val accuracy (full embedding)", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")
    fig.suptitle("ScalarLearnedPrefixLp — p and val accuracy vs epoch", fontsize=13)
    plt.tight_layout()
    fname = f"scalar_p_and_val_acc{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp12] Saved {fname}")


def plot_vector_p_trajectory(p_trajectory, embed_dim, run_dir, fig_stamp=""):
    """
    One line per embedding dimension, colored by dim index.

    p_trajectory: list of (embed_dim,) arrays, one per epoch.
    Uses tab10 colormap for up to 10 dims; falls back to viridis for more.
    """
    epochs = list(range(1, len(p_trajectory) + 1))
    p_arr  = np.stack(p_trajectory, axis=0)   # (n_epochs, embed_dim)

    cmap  = plt.cm.get_cmap("tab10" if embed_dim <= 10 else "viridis", embed_dim)
    colors = [cmap(d) for d in range(embed_dim)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in range(embed_dim):
        ax.plot(epochs, p_arr[:, d], color=colors[d], linewidth=1.5,
                marker="o", markersize=3, label=f"dim {d}")

    for ref_p, style, alpha in [(1, "--", 0.4), (2, ":", 0.4), (3, "-.", 0.4)]:
        ax.axhline(y=ref_p, linestyle=style, color="gray",
                   linewidth=1, alpha=alpha)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learned p (per dim)", fontsize=12)
    ax.set_title("VectorLearnedPrefixLp — p per dim vs epoch", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    fname = f"vector_p_trajectory{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp12] Saved {fname}")


def plot_vector_p_and_val_acc(p_trajectory, val_accs, run_dir, fig_stamp=""):
    """Dual-axis: mean p across dims (left) + val accuracy (right) vs epoch."""
    p_arr  = np.stack(p_trajectory, axis=0)   # (n_epochs, embed_dim)
    p_mean = p_arr.mean(axis=1)
    p_min  = p_arr.min(axis=1)
    p_max  = p_arr.max(axis=1)
    epochs = list(range(1, len(p_mean) + 1))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, p_mean, color="teal", marker="o",
             linewidth=2, markersize=4, label="Mean p across dims")
    ax1.fill_between(epochs, p_min, p_max, color="teal", alpha=0.15,
                     label="Min/max p range")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Learned p (mean across dims)", fontsize=12, color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, color="crimson", marker="s",
             linewidth=2, markersize=4, linestyle="--", label="Val accuracy")
    ax2.set_ylabel("Val accuracy (full embedding)", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")
    fig.suptitle("VectorLearnedPrefixLp — mean p and val accuracy vs epoch", fontsize=13)
    plt.tight_layout()
    fname = f"vector_p_and_val_acc{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp12] Saved {fname}")


# ==============================================================================
# Accuracy curves
# ==============================================================================

def _single_accuracy_plot(results_dict, eval_prefixes, ylabel, title,
                           out_path, fig_stamp=""):
    if fig_stamp:
        base, ext = os.path.splitext(out_path)
        out_path  = f"{base}{fig_stamp}{ext}"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, acc_dict in results_dict.items():
        color, style = _MODEL_STYLES.get(model_name, ("gray", "x-"))
        accs = [acc_dict.get(k, float("nan")) for k in eval_prefixes]
        ax.plot(eval_prefixes, accs, style, color=color,
                label=model_name, linewidth=2, markersize=4)
    ax.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, eval_prefixes[-1])
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp12] Saved {os.path.basename(out_path)}")


def plot_all_curves(linear_results, nn1_results, eval_prefixes, run_dir, fig_stamp=""):
    """Save linear, 1-NN, and combined 2-panel accuracy plots."""
    _single_accuracy_plot(
        linear_results, eval_prefixes,
        ylabel="Linear Classification Accuracy",
        title="Linear Accuracy vs Prefix k  (MRL vs Scalar vs Vector LearnedP — dense)",
        out_path=os.path.join(run_dir, "linear_accuracy_curve.png"),
        fig_stamp=fig_stamp,
    )
    _single_accuracy_plot(
        nn1_results, eval_prefixes,
        ylabel="1-NN Accuracy",
        title="1-NN Accuracy vs Prefix k  (MRL vs Scalar vs Vector LearnedP — dense)",
        out_path=os.path.join(run_dir, "1nn_accuracy_curve.png"),
        fig_stamp=fig_stamp,
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for model_name in linear_results:
        color, style = _MODEL_STYLES.get(model_name, ("gray", "x-"))
        ax_top.plot(eval_prefixes,
                    [linear_results[model_name].get(k, float("nan")) for k in eval_prefixes],
                    style, color=color, label=model_name, linewidth=2, markersize=4)
        ax_bot.plot(eval_prefixes,
                    [nn1_results[model_name].get(k, float("nan")) for k in eval_prefixes],
                    style, color=color, label=model_name, linewidth=2, markersize=4)
    for ax in (ax_top, ax_bot):
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1, eval_prefixes[-1])
        ax.legend(fontsize=10)
    ax_top.set_ylabel("Linear Accuracy", fontsize=12)
    ax_top.set_title("Linear Accuracy vs Prefix k  (dense)", fontsize=12)
    ax_bot.set_xlabel("Prefix size k", fontsize=12)
    ax_bot.set_ylabel("1-NN Accuracy", fontsize=12)
    ax_bot.set_title("1-NN Accuracy vs Prefix k  (dense)", fontsize=12)
    fig.suptitle("MRL vs ScalarLearnedPrefixLp vs VectorLearnedPrefixLp",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fname = f"combined_comparison{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp12] Saved {fname}")


# ==============================================================================
# Importance score plots (wraps exp8 functions with exp12 colour override)
# ==============================================================================

def run_importance_analysis(all_embeddings, y_train_np, y_test_np,
                             cfg, run_dir, max_probe, fig_stamp=""):
    """
    Compute importance scores + method agreement for all models, then plot.

    all_embeddings: {model_name: (Z_train, Z_test)}
    """
    print("\n[exp12] Running importance score analysis ...")

    # Monkey-patch MODEL_COLORS in exp8 so our model names get the right colours
    import experiments.exp8_dim_importance as _exp8
    _orig_colors = dict(_exp8.MODEL_COLORS)
    _exp8.MODEL_COLORS.update(_MODEL_COLORS)

    all_scores    = {}
    all_agreement = {}
    model_names   = list(all_embeddings.keys())

    for model_name, (Z_train, Z_test) in all_embeddings.items():
        scores    = compute_importance_scores(
            Z_test, Z_train, y_train_np, y_test_np,
            max_probe_samples=max_probe, seed=cfg.seed, model_tag=model_name,
        )
        agreement = compute_method_agreement(scores, model_tag=model_name)
        all_scores[model_name]    = scores
        all_agreement[model_name] = agreement

    # Save with fig_stamp inserted before .png
    def _stamped(fname):
        base, ext = os.path.splitext(fname)
        return os.path.join(run_dir, f"{base}{fig_stamp}{ext}")

    # Temporarily redirect save path by monkeypatching plt.savefig would be messy;
    # instead save normally and rename if needed (fig_stamp is in filename directly)
    _orig_savefig = plt.savefig

    def _patched_savefig(path, **kwargs):
        # Insert fig_stamp before .png in any path produced by exp8 plot functions
        base, ext = os.path.splitext(path)
        if fig_stamp and not base.endswith(fig_stamp):
            path = f"{base}{fig_stamp}{ext}"
        _orig_savefig(path, **kwargs)

    plt.savefig = _patched_savefig
    try:
        plot_importance_scores(all_scores, run_dir, cfg, model_names)
        plot_method_agreement(all_scores, all_agreement, run_dir, cfg, model_names)
    finally:
        plt.savefig = _orig_savefig
        _exp8.MODEL_COLORS.clear()
        _exp8.MODEL_COLORS.update(_orig_colors)

    return all_scores, all_agreement


# ==============================================================================
# Results summary
# ==============================================================================

def save_results_summary(linear_results, nn1_results, eval_prefixes,
                          scalar_p_traj, vector_p_traj,
                          all_agreement, run_dir):
    model_names = list(linear_results.keys())
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 12 — VectorLearnedPrefixLp Results\n")
        f.write("=" * 60 + "\n\n")

        # Scalar p summary
        f.write("SCALAR LEARNED P SUMMARY\n")
        f.write("-" * 40 + "\n")
        if scalar_p_traj:
            f.write(f"  p_init = {scalar_p_traj[0].item():.4f}\n")
            f.write(f"  p_final= {scalar_p_traj[-1].item():.4f}\n")
            f.write(f"  trajectory = {[round(float(v), 4) for v in scalar_p_traj]}\n")
        f.write("\n")

        # Vector p summary
        f.write("VECTOR LEARNED P SUMMARY\n")
        f.write("-" * 40 + "\n")
        if vector_p_traj:
            p_final = vector_p_traj[-1]
            f.write(f"  p_init (all dims) = {vector_p_traj[0].mean():.4f}\n")
            f.write(f"  p_final per dim   = {[round(float(v), 4) for v in p_final]}\n")
            f.write(f"  p_final mean      = {p_final.mean():.4f}\n")
            f.write(f"  p_final min/max   = {p_final.min():.4f} / {p_final.max():.4f}\n")
        f.write("\n")

        # Method agreement
        f.write("METHOD AGREEMENT (Spearman rho)\n")
        f.write("-" * 40 + "\n")
        for model_name, agree in all_agreement.items():
            f.write(f"  {model_name}:\n")
            for (ma, mb), rho in agree.items():
                f.write(f"    {ma} vs {mb}: rho={rho:.4f}\n")
        f.write("\n")

        # Accuracy table
        header = f"{'k':>4}  {'Model':<28}  {'Linear':>10}  {'1-NN':>8}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for k in eval_prefixes:
            for model_name in model_names:
                lin = linear_results[model_name].get(k, float("nan"))
                nn1 = nn1_results[model_name].get(k,    float("nan"))
                f.write(f"{k:>4}  {model_name:<28}  {lin:>10.4f}  {nn1:>8.4f}\n")
            f.write("\n")
    print(f"[exp12] Results summary saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Experiment 12 — VectorLearnedPrefixLp.

    Steps:
      1-3  : Parse args, build config, setup output dir
      4    : Load data
      5    : Train MRL
      6    : Train ScalarLearnedPrefixLp
      7    : Train VectorLearnedPrefixLp
      8    : Training curves
      9    : p trajectory plots (scalar + vector)
      10   : Extract embeddings (both Lp models reversed)
      11   : Linear accuracy sweep
      12   : 1-NN accuracy sweep
      13   : Importance scores + method agreement (exp8)
      14   : Accuracy plots + results summary
      15   : Runtime + code snapshot
    """
    run_start = time.time()

    parser = argparse.ArgumentParser(description="Exp12 — VectorLearnedPrefixLp")
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test: digits dataset, 3 epochs, small probes")
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
            experiment_name="exp12_vector_learned_p",
        )
        p_init    = P_INIT
        p_max     = P_MAX
        max_1nn_db = 500
        max_probe  = 500
    else:
        cfg = ExpConfig(
            dataset=DATASET, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
            head_mode=HEAD_MODE, eval_prefixes=EVAL_PREFIXES,
            lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
            weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
            experiment_name="exp12_vector_learned_p",
        )
        p_init    = P_INIT
        p_max     = P_MAX
        max_1nn_db = MAX_1NN_DB
        max_probe  = MAX_PROBE

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Step 3: Setup output directory
    # ------------------------------------------------------------------
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"[exp12] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, args.fast, p_init, p_max)

    # ------------------------------------------------------------------
    # Step 4: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Loading data")
    print("=" * 60)
    data = load_data(cfg)
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

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
    # Step 6: Train ScalarLearnedPrefixLp
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 6: Training ScalarLearnedPrefixLp  (p_init(eff)≈1.69)")
    print("=" * 60)
    sc_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    sc_head    = build_head(cfg, data.n_classes)
    sc_loss    = LearnedPrefixLpLoss(
        embed_dim=cfg.embed_dim, lambda_l1=cfg.l1_lambda,
        p_init=p_init, p_max=p_max,
    )
    sc_opt = torch.optim.Adam(
        list(sc_encoder.parameters()) +
        list(sc_head.parameters()) +
        list(sc_loss.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sc_history, scalar_p_traj, scalar_val_accs = train_learned_p(
        sc_encoder, sc_head, sc_loss, sc_opt, data, cfg, run_dir, "sc_learned"
    )
    sc_encoder.eval(); sc_head.eval()
    print(f"[exp12] Scalar final p = {sc_loss.p.item():.4f}")

    # ------------------------------------------------------------------
    # Step 7: Train VectorLearnedPrefixLp
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 7: Training VectorLearnedPrefixLp  (embed_dim={cfg.embed_dim})")
    print("=" * 60)
    vec_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    vec_head    = build_head(cfg, data.n_classes)
    vec_loss    = VectorLearnedPrefixLpLoss(
        embed_dim=cfg.embed_dim, lambda_l1=cfg.l1_lambda,
        p_init=p_init, p_max=p_max,
    )
    vec_opt = torch.optim.Adam(
        list(vec_encoder.parameters()) +
        list(vec_head.parameters()) +
        list(vec_loss.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    vec_history, vector_p_traj, vector_val_accs = train_learned_p(
        vec_encoder, vec_head, vec_loss, vec_opt, data, cfg, run_dir, "vec_learned"
    )
    vec_encoder.eval(); vec_head.eval()
    p_final_vec = vec_loss.p.detach().cpu().numpy()
    print(f"[exp12] Vector final p: {p_final_vec.round(4).tolist()}")

    # ------------------------------------------------------------------
    # Step 8: Training curves (MANDATORY)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Plotting training curves")
    print("=" * 60)
    plot_training_curves(
        run_dir,
        model_tags=["mat", "sc_learned", "vec_learned"],
        fig_stamp=fig_stamp,
    )

    # ------------------------------------------------------------------
    # Step 9: p trajectory plots
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Plotting p trajectories")
    print("=" * 60)
    plot_scalar_p_trajectory(scalar_p_traj, run_dir, fig_stamp=fig_stamp)
    plot_scalar_p_and_val_acc(scalar_p_traj, scalar_val_accs, run_dir, fig_stamp=fig_stamp)
    plot_vector_p_trajectory(vector_p_traj, cfg.embed_dim, run_dir, fig_stamp=fig_stamp)
    plot_vector_p_and_val_acc(vector_p_traj, vector_val_accs, run_dir, fig_stamp=fig_stamp)

    # ------------------------------------------------------------------
    # Step 10: Extract embeddings (both Lp models reversed)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 10: Extracting embeddings")
    print("=" * 60)
    Z_train_mrl = get_embeddings_np(mrl_encoder, data.X_train)
    Z_test_mrl  = get_embeddings_np(mrl_encoder, data.X_test)
    print(f"[exp12] MRL: train={Z_train_mrl.shape}")

    Z_train_sc = get_embeddings_np(sc_encoder, data.X_train)
    Z_test_sc  = get_embeddings_np(sc_encoder, data.X_test)
    Z_train_sc = np.ascontiguousarray(Z_train_sc[:, ::-1])
    Z_test_sc  = np.ascontiguousarray(Z_test_sc[:,  ::-1])
    print(f"[exp12] ScalarLearnedPrefixLp: train={Z_train_sc.shape}, reversed")

    Z_train_vec = get_embeddings_np(vec_encoder, data.X_train)
    Z_test_vec  = get_embeddings_np(vec_encoder, data.X_test)
    Z_train_vec = np.ascontiguousarray(Z_train_vec[:, ::-1])
    Z_test_vec  = np.ascontiguousarray(Z_test_vec[:,  ::-1])
    print(f"[exp12] VectorLearnedPrefixLp: train={Z_train_vec.shape}, reversed")

    # ------------------------------------------------------------------
    # Step 11: Linear accuracy (dense prefix sweep)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 11: Linear accuracy  (k=1..{cfg.embed_dim})")
    print("=" * 60)
    linear_results = {
        "MRL": evaluate_prefix_linear(
            Z_train_mrl, Z_test_mrl, y_train_np, y_test_np,
            cfg.eval_prefixes, "MRL", seed=cfg.seed),
        "ScalarLearnedPrefixLp (rev)": evaluate_prefix_linear(
            Z_train_sc, Z_test_sc, y_train_np, y_test_np,
            cfg.eval_prefixes, "ScalarLearnedPrefixLp", seed=cfg.seed),
        "VectorLearnedPrefixLp (rev)": evaluate_prefix_linear(
            Z_train_vec, Z_test_vec, y_train_np, y_test_np,
            cfg.eval_prefixes, "VectorLearnedPrefixLp", seed=cfg.seed),
    }

    # ------------------------------------------------------------------
    # Step 12: 1-NN accuracy (dense prefix sweep)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 12: 1-NN accuracy  (k=1..{cfg.embed_dim})")
    print("=" * 60)
    mrl_1nn = evaluate_prefix_1nn(
        mrl_encoder, data, cfg.eval_prefixes, "MRL",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )

    print("\n[exp12] 1-NN sweep for 'ScalarLearnedPrefixLp (rev)' ...")
    sc_1nn = {}
    for k in cfg.eval_prefixes:
        sc_1nn[k] = evaluate_1nn(
            Z_train_sc[:, :k], Z_test_sc[:, :k], y_train_np, y_test_np,
            max_db_samples=max_1nn_db, seed=cfg.seed,
        )
        print(f"  k={k:>3}  1-NN={sc_1nn[k]:.4f}")

    print("\n[exp12] 1-NN sweep for 'VectorLearnedPrefixLp (rev)' ...")
    vec_1nn = {}
    for k in cfg.eval_prefixes:
        vec_1nn[k] = evaluate_1nn(
            Z_train_vec[:, :k], Z_test_vec[:, :k], y_train_np, y_test_np,
            max_db_samples=max_1nn_db, seed=cfg.seed,
        )
        print(f"  k={k:>3}  1-NN={vec_1nn[k]:.4f}")

    nn1_results = {
        "MRL":                          mrl_1nn,
        "ScalarLearnedPrefixLp (rev)":  sc_1nn,
        "VectorLearnedPrefixLp (rev)":  vec_1nn,
    }

    # ------------------------------------------------------------------
    # Step 13: Importance scores + method agreement (exp8)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 13: Importance scores + method agreement")
    print("=" * 60)
    all_embeddings = {
        "MRL":                          (Z_train_mrl, Z_test_mrl),
        "ScalarLearnedPrefixLp (rev)":  (Z_train_sc,  Z_test_sc),
        "VectorLearnedPrefixLp (rev)":  (Z_train_vec, Z_test_vec),
    }
    all_scores, all_agreement = run_importance_analysis(
        all_embeddings, y_train_np, y_test_np,
        cfg, run_dir, max_probe, fig_stamp=fig_stamp,
    )

    # ------------------------------------------------------------------
    # Step 14: Accuracy plots + results summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 14: Saving plots and results")
    print("=" * 60)
    plot_all_curves(linear_results, nn1_results, cfg.eval_prefixes, run_dir,
                    fig_stamp=fig_stamp)
    save_results_summary(linear_results, nn1_results, cfg.eval_prefixes,
                         scalar_p_traj, vector_p_traj, all_agreement, run_dir)

    # Compact stdout table
    sample_ks = [k for k in cfg.eval_prefixes if k % 4 == 0 or k == 1]
    print(f"\n{'k':>4}  {'Model':<30}  {'Linear':>8}  {'1-NN':>8}")
    print("-" * 55)
    for k in sample_ks:
        for model_name in linear_results:
            lin = linear_results[model_name].get(k, float("nan"))
            nn1 = nn1_results[model_name].get(k,    float("nan"))
            print(f"{k:>4}  {model_name:<30}  {lin:>8.4f}  {nn1:>8.4f}")
        print()

    # ------------------------------------------------------------------
    # Step 15: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 15: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp12] Experiment 12 complete.")
    print(f"[exp12] All outputs saved to: {run_dir}")
    print(f"[exp12] Scalar final p   = {sc_loss.p.item():.4f}")
    print(f"[exp12] Vector final p   = {p_final_vec.round(4).tolist()}")


if __name__ == "__main__":
    main()
