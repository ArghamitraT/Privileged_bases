"""
Script: experiments/exp14_two_eval_compare.py
---------------------------------------------
Experiment 14 — Two-Evaluation Comparison on Dense MRL vs PrefixL1.

Training is identical to exp9/10 (SharedClassifier, Dense MRL + PrefixL1).
After training, the SAME models are evaluated two ways:

  Evaluation 1 — exactly as exp9/10 (unchanged):
    - Linear accuracy : fresh logistic regression fitted on ≤10 000 subsampled
                        z_train[:, :k], evaluated on z_test[:, :k]
    - 1-NN accuracy   : fixed ≤10 000 random subsample of z_train as database;
                        every z_test point queried against it
    - probe_acc[j]    : fresh 1D logistic regression fitted on ≤2 000 subsampled
                        z_train[:, j], evaluated on z_test[:, j]

  Evaluation 2 — no new fitting (except probe_acc):
    - Linear accuracy : trained W used directly; no new LR is fitted.
                        logits_k = z_test[:, :k] @ W[:, :k].T + b → argmax
    - 1-NN accuracy   : full z_train as database (all ~56 000 training points);
                        every z_test point queried against all of them
    - probe_acc[j]    : same as Eval 1 (fresh 1D LR per dim; W was trained
                        jointly and cannot isolate dim j's contribution)

PrefixL1 note:
  Dimensions are reversed before all evaluation (Z = Z[:, ::-1]) and W columns
  are reversed accordingly for Eval 2 (W_rev = W[:, ::-1]). Legend everywhere:
  "PrefixL1 (rev)". PrefixL1 Eval 2 known limitation: W[:, :k] was not trained
  for prefix k — accuracy at small k will be lower than Eval 1 (which refits LR).

Conda environment: mrl_env

Usage:
    python experiments/exp14_two_eval_compare.py --fast                         # smoke test (digits, 5 epochs)
    python experiments/exp14_two_eval_compare.py                                # full run (MNIST, embed_dim from CONFIG)
    python experiments/exp14_two_eval_compare.py --embed-dim 16                 # full run, embed_dim=16
    python experiments/exp14_two_eval_compare.py --embed-dim 32                 # full run, embed_dim=32
    python experiments/exp14_two_eval_compare.py --embed-dim 64                 # full run, embed_dim=64
    python experiments/exp14_two_eval_compare.py --dataset cd34                 # CD34 multiome, sweep [8, 16]
    python experiments/exp14_two_eval_compare.py --dataset cd34 --embed-dim 8   # CD34, single dim
    python experiments/exp14_two_eval_compare.py --dataset cd34 --fast          # CD34 smoke test
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import PrefixLpLoss
from training.trainer import train
from experiments.exp7_mrl_vs_ff import train_single_model, get_embeddings_np, evaluate_1nn
from experiments.exp8_dim_importance import compute_importance_scores, compute_method_agreement


# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# CURRENT RUN: CD34 multiome, input=2000 HVGs → hidden=256 → embed ∈ {8, 16}.
# Switch DATASET to "mnist" (or pass --dataset mnist) to use MNIST_* overrides.
# ==============================================================================
DATASET           = "cd34"       # "cd34" or "mnist"  (CLI --dataset overrides)
EMBED_DIMS        = [8, 16]      # sweep; --embed-dim overrides
HIDDEN_DIM        = 256
HEAD_MODE         = "shared_head"
EPOCHS            = 30           # 15 is sized for CD34; MNIST override below
PATIENCE          = 5
LR                = 1e-3
BATCH_SIZE        = 128
WEIGHT_DECAY      = 1e-4
SEED              = 42
L1_LAMBDA         = 0.05
MAX_LR_SAMPLES    = 10_000       # Eval 1: subsample cap for fresh LR fits
MAX_1NN_DB_E1     = 10_000       # Eval 1: subsample cap for 1-NN database
MAX_PROBE_SAMPLES = 2_000        # per-dim probe accuracy subsample (both evals)
# Eval 2: 1-NN uses full z_train (max_db_samples=None); linear uses trained W directly

# --- CD34-specific data loading (used when DATASET == "cd34") ----------------
# CD34 is small (~6.9k cells, 8 cell types). With ~4.8k train cells both Eval 1
# and Eval 2 1-NN databases collapse to the same full set — the linear-accuracy
# gap is the meaningful signal on CD34.
CD34_DATA_PATH     = os.path.join(os.environ["HOME"], "Mat_embedding_hyperbole",
                                  "data", "cd34_multiome",
                                  "GSE200046_cd34_multiome_rna.h5ad")
CD34_N_HVG         = 2000        # highly variable genes used as input features
CD34_RECOMPUTE_HVG = False       # False: use h5ad precomputed HVGs; True: recompute via scanpy

# --- MNIST overrides (used when DATASET == "mnist") --------------------------
MNIST_EMBED_DIMS   = [16]
MNIST_EPOCHS       = 20
# ==============================================================================


MODEL_COLORS = {
    "Dense MRL":              "darkorange",
    "MRL-E":                  "mediumseagreen",
    "PrefixL1 (rev)":         "crimson",
    "PrefixL1 α=0.75 (rev)":  "orchid",
    "PrefixL1 α=0.5 (rev)":   "steelblue",
}

# Alpha values for the two additional PrefixL1 variants
L1_ALPHAS = [0.75, 0.5]


# ==============================================================================
# Training helper for PrefixL1 with custom weight_alpha
# ==============================================================================

def train_prefix_l1_alpha(cfg: ExpConfig, data, run_dir: str,
                           model_tag: str, alpha: float):
    """Train PrefixL1 with dim_weights = (d - j)^alpha instead of (d - j)."""
    encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    head    = build_head(cfg, data.n_classes)
    loss_fn = PrefixLpLoss(embed_dim=cfg.embed_dim, lambda_l1=cfg.l1_lambda, p=1)
    # Override the registered buffer with alpha-scaled weights
    w = torch.arange(cfg.embed_dim, 0, -1, dtype=torch.float32) ** alpha
    loss_fn.dim_weights = w
    print(f"[exp14] PrefixL1 alpha={alpha}  dim_weights (first 8): {w[:8].tolist()}")
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    train(encoder, head, loss_fn, opt, data, cfg, run_dir, model_tag=model_tag)
    enc_path  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_path = os.path.join(run_dir, f"{model_tag}_head_best.pt")
    encoder.load_state_dict(torch.load(enc_path,  map_location="cpu"))
    head.load_state_dict(   torch.load(head_path, map_location="cpu"))
    encoder.eval()
    head.eval()
    return encoder, head


# ==============================================================================
# MRL-E — direct weight slicing, no bias  (MRL paper Algorithm 2, efficient)
# ==============================================================================

class MRLEHead(nn.Module):
    """Single linear classifier with no bias — weight accessed directly for MRL-E."""
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes, bias=False)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


class MRLELoss(nn.Module):
    """MRL-E loss: (1/d) * sum_k CE(z[:,:k] @ W[:,:k].T, y) — no bias term."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.embed_dim = embed_dim

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor,
                head: MRLEHead) -> torch.Tensor:
        W = head.head.weight  # (n_classes, embed_dim)
        total = torch.tensor(0.0, device=embedding.device, requires_grad=True)
        for k in range(1, self.embed_dim + 1):
            logits = torch.matmul(embedding[:, :k], W[:, :k].T)
            total = total + self.ce(logits, labels)
        return total / self.embed_dim


def train_mrl_e(cfg: ExpConfig, data, run_dir: str, model_tag: str = "mrl_e"):
    """Train MRL-E: same encoder, no-bias head, direct weight-slice loss."""
    encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    head    = MRLEHead(cfg.embed_dim, data.n_classes)
    loss_fn = MRLELoss(cfg.embed_dim)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    train(encoder, head, loss_fn, opt, data, cfg, run_dir, model_tag=model_tag)
    enc_path  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_path = os.path.join(run_dir, f"{model_tag}_head_best.pt")
    encoder.load_state_dict(torch.load(enc_path,  map_location="cpu"))
    head.load_state_dict(   torch.load(head_path, map_location="cpu"))
    encoder.eval()
    head.eval()
    return encoder, head


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
    print(f"[exp14] Seeds set to {seed}")


# ==============================================================================
# Experiment description
# ==============================================================================

def save_experiment_description(cfg: ExpConfig, run_dir: str, fast: bool):
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 14 — Two-Evaluation Comparison: Dense MRL vs PrefixL1\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Training is identical to exp9/10. After training, the same two models\n"
            "(Dense MRL, PrefixL1) are evaluated two ways at every prefix k=1..d:\n\n"
            "  Eval 1 (unchanged from exp9/10):\n"
            "    - Linear accuracy: fresh LR on <= 10k subsampled z_train[:, :k]\n"
            "    - 1-NN: <= 10k random train subsample as database\n"
            "    - probe_acc[j]: fresh 1D LR on <= 2k z_train[:, j]\n\n"
            "  Eval 2 (no new fitting, except probe_acc):\n"
            "    - Linear accuracy: trained W directly; logits = z_test[:,:k]@W[:,:k].T+b\n"
            "    - 1-NN: full z_train as database (all ~56k training points)\n"
            "    - probe_acc[j]: same as Eval 1 (W[:,j] trained jointly; must refit)\n\n"
            "PrefixL1 dims reversed before all eval. For Eval 2, W columns also reversed.\n"
            "PrefixL1 Eval 2 limitation: W[:,: k] was not trained for prefix k < d.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Eval 1 is the standard metric. Eval 2 removes all re-fitting to see\n"
            "whether the trained W alone encodes the prefix ordering property.\n"
            "Gap between Eval 1 and Eval 2 for Dense MRL (should be small) vs\n"
            "PrefixL1 (expected larger at small k) quantifies how much ordering\n"
            "is encoded in W vs. being recovered by the fresh LR.\n\n"
        )

        f.write("FAST MODE\n" if fast else "FULL RUN\n")
        f.write("-" * 40 + "\n")
        f.write(f"  fast={fast}\n\n")

        f.write("CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[exp14] Saved experiment_description.log")


# ==============================================================================
# Training curves (mandatory)
# ==============================================================================

def plot_training_curves(run_dir: str, model_tags: list, fig_stamp: str):
    histories = {}
    for tag in model_tags:
        log_path = os.path.join(run_dir, f"{tag}_train.log")
        if not os.path.isfile(log_path):
            continue
        train_losses, val_losses = [], []
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
        if train_losses:
            histories[tag] = {"train": train_losses, "val": val_losses}

    if not histories:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No training logs found.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
    else:
        n = len(histories)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for col, (tag, hist) in enumerate(histories.items()):
            ax = axes[0, col]
            epochs = range(1, len(hist["train"]) + 1)
            ax.plot(epochs, hist["train"], label="Train", linewidth=2)
            ax.plot(epochs, hist["val"],   label="Val",   linewidth=2, linestyle="--")
            ax.set_title(tag, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp14] Saved training_curves{fig_stamp}.png")


# ==============================================================================
# Evaluation 1 — fresh LR + subsampled 1-NN  (exact exp9/10)
# ==============================================================================

def eval1_linear_sweep(Z_train_sub: np.ndarray, Z_test: np.ndarray,
                        y_train_sub: np.ndarray, y_test: np.ndarray,
                        eval_prefixes: list, seed: int, model_tag: str) -> dict:
    """Fresh logistic regression at each prefix k on subsampled training set."""
    print(f"\n[exp14] Eval 1 linear sweep — {model_tag}")
    results = {}
    for k in eval_prefixes:
        lr = LogisticRegression(solver="saga", max_iter=1000, random_state=seed, n_jobs=1)
        lr.fit(Z_train_sub[:, :k], y_train_sub)
        acc = float(lr.score(Z_test[:, :k], y_test))
        results[k] = acc
        if k in {1, 4, 8, 16, 32, 64} or k == eval_prefixes[-1]:
            print(f"  k={k:>3}  linear_acc={acc:.4f}")
    return results


def eval_1nn_sweep(Z_train: np.ndarray, Z_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray,
                    eval_prefixes: list, max_db: int, seed: int,
                    model_tag: str) -> dict:
    """1-NN at each prefix k with a subsampled database (shared across both evals)."""
    print(f"\n[exp14] 1-NN sweep — {model_tag}  (max_db={max_db})")
    results = {}
    for k in eval_prefixes:
        acc = evaluate_1nn(
            Z_train[:, :k], Z_test[:, :k], y_train, y_test,
            max_db_samples=max_db, seed=seed,
        )
        results[k] = acc
        if k in {1, 4, 8, 16, 32, 64} or k == eval_prefixes[-1]:
            print(f"  k={k:>3}  1nn_acc={acc:.4f}")
    return results


# ==============================================================================
# Evaluation 2 — trained W  (no fitting except probe_acc)
# ==============================================================================

def eval2_linear_sweep(Z_test: np.ndarray, y_test: np.ndarray,
                        W: np.ndarray, b: np.ndarray,
                        eval_prefixes: list, model_tag: str) -> dict:
    """Use trained W directly — no new LR fitted.

    logits_k = z_test[:, :k] @ W[:, :k].T + b  →  argmax  →  accuracy.
    For PrefixL1, pass W with columns already reversed to match reversed Z.
    """
    print(f"\n[exp14] Eval 2 linear sweep — {model_tag}  (trained W, no fitting)")
    results = {}
    for k in eval_prefixes:
        logits = Z_test[:, :k] @ W[:, :k].T + b
        preds  = np.argmax(logits, axis=1)
        acc    = float((preds == y_test).mean())
        results[k] = acc
        if k in {1, 4, 8, 16, 32, 64} or k == eval_prefixes[-1]:
            print(f"  k={k:>3}  linear_acc(W)={acc:.4f}")
    return results


# ==============================================================================
# Plotting
# ==============================================================================

def plot_combined_comparison(linear_results: dict, nn_results: dict,
                              eval_prefixes: list, model_names: list,
                              run_dir: str, title_suffix: str,
                              fname_tag: str, fig_stamp: str):
    """2-panel figure: linear accuracy (top) + 1-NN accuracy (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for model_name in model_names:
        color = MODEL_COLORS[model_name]
        lin_accs = [linear_results[model_name].get(k, float("nan")) for k in eval_prefixes]
        nn_accs  = [nn_results[model_name].get(k,  float("nan")) for k in eval_prefixes]
        ax1.plot(eval_prefixes, lin_accs, color=color, label=model_name, linewidth=2)
        ax2.plot(eval_prefixes, nn_accs,  color=color, label=model_name, linewidth=2)

    tick_step   = max(1, len(eval_prefixes) // 8)
    shown_ticks = [k for k in eval_prefixes if (k - 1) % tick_step == 0]
    shown_ticks.append(eval_prefixes[-1])
    ax2.set_xticks(shown_ticks)
    ax2.set_xticklabels([str(k) for k in shown_ticks], fontsize=9)

    ax1.set_ylabel("Linear Accuracy",   fontsize=11)
    ax2.set_ylabel("1-NN Accuracy",     fontsize=11)
    ax2.set_xlabel("Prefix size k",     fontsize=11)
    for ax in (ax1, ax2):
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Dense Prefix Comparison — {title_suffix}", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(run_dir, f"combined_comparison_{fname_tag}{fig_stamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp14] Saved {os.path.basename(out_path)}")


def plot_importance_scores(all_scores: dict, model_names: list,
                            embed_dim: int, run_dir: str, fig_stamp: str):
    """3-method × N-model grid of per-dim bar charts."""
    METHODS       = ["mean_abs", "variance", "probe_acc"]
    METHOD_LABELS = {"mean_abs": "Mean |z|", "variance": "Variance", "probe_acc": "1D Probe Acc"}
    n_methods     = len(METHODS)
    n_models      = len(model_names)
    tick_step     = 4 if embed_dim > 16 else 1

    fig, axes = plt.subplots(
        n_methods, n_models,
        figsize=(5 * n_models, max(3, embed_dim * 0.22) * n_methods),
        squeeze=False,
    )

    for row, method in enumerate(METHODS):
        for col, model_name in enumerate(model_names):
            ax     = axes[row, col]
            scores = all_scores[model_name][method]
            dims   = np.arange(embed_dim)
            color  = MODEL_COLORS.get(model_name, "gray")
            ax.barh(dims, scores, color=color, alpha=0.8)
            ax.set_xlim(left=0)
            ax.invert_yaxis()
            ax.set_xlabel(METHOD_LABELS[method], fontsize=9)
            visible = [d for d in dims if d % tick_step == 0]
            ax.set_yticks(visible)
            ax.set_yticklabels([f"d{d}" for d in visible], fontsize=7)
            if row == 0:
                ax.set_title(model_name, fontsize=11,
                             color=color, fontweight="bold")
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=10)

    fig.suptitle("Per-Dimension Importance Scores — Dense MRL vs PrefixL1 (rev)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(run_dir, f"importance_scores{fig_stamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp14] Saved importance_scores{fig_stamp}.png")


def plot_method_agreement(all_scores: dict, all_agreement: dict,
                           model_names: list, embed_dim: int,
                           run_dir: str, fig_stamp: str):
    """Scatter plots + Spearman ρ per model × method pair."""
    METHOD_LABELS = {"mean_abs": "Mean |z|", "variance": "Variance", "probe_acc": "1D Probe Acc"}
    PAIRS = [("mean_abs", "variance"), ("mean_abs", "probe_acc"), ("variance", "probe_acc")]
    PAIR_LABELS = {
        ("mean_abs",  "variance"):  "mean|z| vs Var",
        ("mean_abs",  "probe_acc"): "mean|z| vs Probe",
        ("variance",  "probe_acc"): "Var vs Probe",
    }
    n_models  = len(model_names)
    n_pairs   = len(PAIRS)
    annotate  = embed_dim <= 16

    fig, axes = plt.subplots(n_models, n_pairs,
                              figsize=(5 * n_pairs, 4 * n_models),
                              squeeze=False)

    for row, model_name in enumerate(model_names):
        scores = all_scores[model_name]
        color  = MODEL_COLORS.get(model_name, "gray")
        for col, (ma, mb) in enumerate(PAIRS):
            ax  = axes[row, col]
            x   = scores[ma]
            y   = scores[mb]
            rho = all_agreement[model_name].get((ma, mb), float("nan"))
            ax.scatter(x, y, color=color, alpha=0.7, s=40)
            if annotate:
                for d, (xi, yi) in enumerate(zip(x, y)):
                    ax.annotate(str(d), (xi, yi), fontsize=6,
                                ha="left", va="bottom", alpha=0.8)
            ax.text(0.05, 0.92, f"ρ = {rho:.3f}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            ax.set_xlabel(METHOD_LABELS[ma], fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(PAIR_LABELS[(ma, mb)], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{model_name}\n{METHOD_LABELS[mb]}",
                              fontsize=9, color=color, fontweight="bold")
            else:
                ax.set_ylabel(METHOD_LABELS[mb], fontsize=9)

    fig.suptitle("Importance Method Agreement (Spearman ρ per model)", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(run_dir, f"method_agreement{fig_stamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp14] Saved method_agreement{fig_stamp}.png")


# ==============================================================================
# Results summary
# ==============================================================================

def save_results_summary(eval1_linear: dict, eval2_linear: dict,
                          nn_results: dict,
                          eval_prefixes: list, model_names: list,
                          all_scores: dict, all_agreement: dict,
                          run_dir: str):
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 14 — Two-Evaluation Comparison Results\n")
        f.write("=" * 72 + "\n\n")

        # Sample every 4th k to keep tables readable
        sample_ks = [k for k in eval_prefixes if (k - 1) % 4 == 0]
        sample_ks.append(eval_prefixes[-1])
        sample_ks = sorted(set(sample_ks))

        for eval_label, lin_res in [
            ("EVAL 1 — Fresh LR (subsampled z_train)", eval1_linear),
            ("EVAL 2 — Trained W (no fitting)",        eval2_linear),
        ]:
            nn_res = nn_results
            f.write(f"{'=' * 72}\n{eval_label}\n{'=' * 72}\n\n")

            # Linear accuracy table
            f.write("Linear Accuracy\n")
            f.write("-" * 60 + "\n")
            hdr = f"{'k':>4}  " + "  ".join(f"{m:<20}" for m in model_names)
            f.write(hdr + "\n")
            f.write("-" * len(hdr) + "\n")
            for k in sample_ks:
                row = f"{k:>4}  "
                row += "  ".join(f"{lin_res[m].get(k, float('nan')):>20.4f}" for m in model_names)
                f.write(row + "\n")
            f.write("\n")

            # 1-NN accuracy table
            f.write("1-NN Accuracy\n")
            f.write("-" * 60 + "\n")
            f.write(hdr + "\n")
            f.write("-" * len(hdr) + "\n")
            for k in sample_ks:
                row = f"{k:>4}  "
                row += "  ".join(f"{nn_res[m].get(k, float('nan')):>20.4f}" for m in model_names)
                f.write(row + "\n")
            f.write("\n\n")

        # Eval 2 − Eval 1 gap (linear accuracy)
        f.write("EVAL 2 − EVAL 1 LINEAR ACCURACY GAP  (positive = Eval2 > Eval1)\n")
        f.write("-" * 60 + "\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for k in sample_ks:
            row = f"{k:>4}  "
            for m in model_names:
                gap = eval2_linear[m].get(k, float("nan")) - eval1_linear[m].get(k, float("nan"))
                row += f"  {gap:>+19.4f}"
            f.write(row + "\n")
        f.write("\n\n")

        # Importance scores summary
        f.write("IMPORTANCE SCORES — Top-5 dims per model per method\n")
        f.write("-" * 60 + "\n")
        for model_name in model_names:
            f.write(f"  {model_name}\n")
            for method in ["mean_abs", "variance", "probe_acc"]:
                top5 = np.argsort(all_scores[model_name][method])[::-1][:5].tolist()
                f.write(f"    {method:<12}: {top5}\n")
            f.write("\n")

        # Method agreement
        f.write("METHOD AGREEMENT — Spearman ρ\n")
        f.write("-" * 60 + "\n")
        for model_name in model_names:
            rhos = [
                all_agreement[model_name].get(p, float("nan"))
                for p in [("mean_abs", "variance"), ("mean_abs", "probe_acc"), ("variance", "probe_acc")]
            ]
            f.write(f"  {model_name}: "
                    f"mean|z|×Var={rhos[0]:+.3f}  "
                    f"mean|z|×Probe={rhos[1]:+.3f}  "
                    f"Var×Probe={rhos[2]:+.3f}\n")
        f.write("\n")

    print(f"[exp14] Saved results_summary.txt")


# ==============================================================================
# Main
# ==============================================================================

def run_one_dim(embed_dim: int, data, y_train_np: np.ndarray, y_test_np: np.ndarray,
                dim_dir: str, fig_stamp: str, fast: bool,
                max_lr_samples: int, max_1nn_e1: int, max_probe_samples: int,
                dataset_name: str, epochs: int, include_dense_mrl: bool):
    """Train + evaluate models for a single embed_dim. Save all outputs to dim_dir.

    include_dense_mrl=False drops Dense MRL (controlled by --no-dense-mrl).
    """

    cfg = ExpConfig(
        dataset=dataset_name,
        embed_dim=embed_dim,
        hidden_dim=128 if fast else HIDDEN_DIM,
        head_mode=HEAD_MODE,
        eval_prefixes=list(range(1, embed_dim + 1)),
        lr=LR, epochs=epochs,
        batch_size=BATCH_SIZE, patience=3 if fast else PATIENCE,
        weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
        experiment_name="exp14_two_eval_compare",
    )

    print(f"\n{'=' * 60}")
    print(f"embed_dim={embed_dim}  →  {dim_dir}")
    print(f"{'=' * 60}")

    # Optionally train Dense MRL
    mrl_enc = mrl_head = None
    if include_dense_mrl:
        mrl_enc, mrl_head = train_single_model(
            cfg, data, dim_dir, model_type="matryoshka", model_tag="dense_mrl"
        )
    # Train MRL-E (same encoder arch, no-bias head, direct weight-slice loss)
    mrle_enc, mrle_head = train_mrl_e(cfg, data, dim_dir, model_tag="mrl_e")
    # Train PrefixL1 variants (alpha=1.0 standard + alpha sweep)
    pl1_models = {}
    for alpha in [1.0] + L1_ALPHAS:
        tag   = "prefix_l1" if alpha == 1.0 else f"prefix_l1_a{str(alpha).replace('.', '')}"
        enc, head = train_single_model(cfg, data, dim_dir,
                                       model_type="prefix_l1", model_tag=tag) \
                    if alpha == 1.0 else \
                    train_prefix_l1_alpha(cfg, data, dim_dir, model_tag=tag, alpha=alpha)
        pl1_models[alpha] = (enc, head)

    all_tags = (["dense_mrl"] if include_dense_mrl else []) + ["mrl_e"] + [
        "prefix_l1" if a == 1.0 else f"prefix_l1_a{str(a).replace('.', '')}"
        for a in [1.0] + L1_ALPHAS
    ]
    plot_training_curves(dim_dir, all_tags, fig_stamp)

    # Extract embeddings; reverse all PrefixL1 variants
    models_data = {}
    if include_dense_mrl:
        Z_train_mrl = get_embeddings_np(mrl_enc, data.X_train)
        Z_test_mrl  = get_embeddings_np(mrl_enc, data.X_test)
        W_mrl = mrl_head.head.weight.detach().cpu().numpy()
        b_mrl = mrl_head.head.bias.detach().cpu().numpy()
        models_data["Dense MRL"] = (Z_train_mrl, Z_test_mrl, W_mrl, b_mrl)

    Z_train_mrle = get_embeddings_np(mrle_enc, data.X_train)
    Z_test_mrle  = get_embeddings_np(mrle_enc, data.X_test)
    W_mrle = mrle_head.head.weight.detach().cpu().numpy()
    b_mrle = np.zeros(W_mrle.shape[0])  # MRL-E has no bias
    models_data["MRL-E"] = (Z_train_mrle, Z_test_mrle, W_mrle, b_mrle)

    alpha_to_name = {1.0: "PrefixL1 (rev)", 0.75: "PrefixL1 α=0.75 (rev)", 0.5: "PrefixL1 α=0.5 (rev)"}
    for alpha, (enc, head) in pl1_models.items():
        Ztr = np.ascontiguousarray(get_embeddings_np(enc, data.X_train)[:, ::-1])
        Zte = np.ascontiguousarray(get_embeddings_np(enc, data.X_test)[:,  ::-1])
        W_rev = np.ascontiguousarray(head.head.weight.detach().cpu().numpy()[:, ::-1])
        b     = head.head.bias.detach().cpu().numpy()
        models_data[alpha_to_name[alpha]] = (Ztr, Zte, W_rev, b)
    model_names = list(models_data.keys())

    # Subsample for Eval 1 LR
    rng = np.random.default_rng(cfg.seed)
    lr_idx = (rng.choice(len(y_train_np), max_lr_samples, replace=False)
              if len(y_train_np) > max_lr_samples else np.arange(len(y_train_np)))

    # Eval 1 — fresh LR
    eval1_linear = {
        m: eval1_linear_sweep(Z_train[lr_idx], Z_test,
                               y_train_np[lr_idx], y_test_np,
                               cfg.eval_prefixes, cfg.seed, m)
        for m, (Z_train, Z_test, W, b) in models_data.items()
    }

    # Eval 2 — trained W
    eval2_linear = {
        m: eval2_linear_sweep(Z_test, y_test_np, W, b, cfg.eval_prefixes, m)
        for m, (Z_train, Z_test, W, b) in models_data.items()
    }

    # 1-NN — shared
    nn_results = {
        m: eval_1nn_sweep(Z_train, Z_test, y_train_np, y_test_np,
                           cfg.eval_prefixes, max_1nn_e1, cfg.seed, m)
        for m, (Z_train, Z_test, W, b) in models_data.items()
    }

    # Importance scores + agreement
    all_scores    = {}
    all_agreement = {}
    for m, (Z_train, Z_test, W, b) in models_data.items():
        all_scores[m]    = compute_importance_scores(
            Z_test=Z_test, Z_train=Z_train,
            y_train=y_train_np, y_test=y_test_np,
            max_probe_samples=max_probe_samples, seed=cfg.seed, model_tag=m,
        )
        all_agreement[m] = compute_method_agreement(all_scores[m], model_tag=m)

    # Plots
    plot_combined_comparison(eval1_linear, nn_results, cfg.eval_prefixes, model_names,
                              dim_dir, "Eval 1 — Fresh LR", "eval1", fig_stamp)
    plot_combined_comparison(eval2_linear, nn_results, cfg.eval_prefixes, model_names,
                              dim_dir, "Eval 2 — Trained W", "eval2", fig_stamp)
    plot_importance_scores(all_scores, model_names, cfg.embed_dim, dim_dir, fig_stamp)
    plot_method_agreement(all_scores, all_agreement, model_names,
                          cfg.embed_dim, dim_dir, fig_stamp)
    save_results_summary(eval1_linear, eval2_linear, nn_results,
                          cfg.eval_prefixes, model_names, all_scores, all_agreement, dim_dir)

    # Quick stdout table
    sample_ks = [k for k in cfg.eval_prefixes if k in {1, 2, 4, 8, 16, 32, 64, embed_dim}]
    print(f"\n  embed_dim={embed_dim}  linear accuracy (E1=fresh LR, E2=trained W)")
    print(f"  {'k':>4}  " + "  ".join(f"{'E1-'+m[:8]:<14}  {'E2-'+m[:8]:<14}" for m in model_names))
    for k in sample_ks:
        row = f"  {k:>4}  "
        for m in model_names:
            row += f"{eval1_linear[m].get(k, float('nan')):>14.4f}  {eval2_linear[m].get(k, float('nan')):>14.4f}  "
        print(row)


def main():
    """
    Loops over EMBED_DIMS (or MNIST_EMBED_DIMS when --dataset mnist) — or a
    single --embed-dim if given. Data is loaded once; each dim gets its own
    subdirectory under the root run_dir.
    """
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="Exp14 — Two-Evaluation Comparison: Dense MRL vs PrefixL1"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test: embed_dim=8, 5 epochs (MNIST→digits; CD34→500 cells).")
    parser.add_argument("--embed-dim", type=int, default=None, metavar="N",
                        help="Run a single embed_dim instead of the full sweep.")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["mnist", "cd34"],
                        help="Dataset override: 'mnist' or 'cd34'. "
                             "Defaults to the DATASET constant in CONFIG.")
    parser.add_argument("--no-dense-mrl", action="store_true",
                        help="Skip Dense MRL training (keep only MRL-E + PrefixL1 variants).")
    args = parser.parse_args()

    include_dense_mrl = not args.no_dense_mrl

    # Resolve dataset: CLI flag overrides CONFIG constant
    dataset = args.dataset if args.dataset is not None else DATASET

    # Dataset-conditional config
    if dataset == "cd34":
        dataset_name       = "cd34"
        epochs             = 5 if args.fast else EPOCHS
        embed_dims_default = EMBED_DIMS
    else:
        dataset_name       = "digits" if args.fast else "mnist"
        epochs             = 5 if args.fast else MNIST_EPOCHS
        embed_dims_default = MNIST_EMBED_DIMS

    embed_dims = [args.embed_dim] if args.embed_dim is not None else (
        [8] if args.fast else embed_dims_default
    )

    if args.fast:
        max_lr_samples    = 1_000
        max_1nn_e1        = 1_000
        max_probe_samples = 500
    else:
        max_lr_samples    = MAX_LR_SAMPLES
        max_1nn_e1        = MAX_1NN_DB_E1
        max_probe_samples = MAX_PROBE_SAMPLES

    set_seeds(SEED)

    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"[exp14] Root output → {run_dir}")
    print(f"[exp14] Dataset     → {dataset}  (include_dense_mrl={include_dense_mrl})")
    print(f"[exp14] Embed dims  → {embed_dims}\n")

    # Load data once (shared across all dims)
    if dataset == "cd34":
        from experiments.exp13_mrl_cd34_supervised import (
            load_cd34_data, make_data_split,
        )
        if not os.path.exists(CD34_DATA_PATH):
            print(f"[exp14] ERROR: CD34 data file not found: {CD34_DATA_PATH}")
            sys.exit(1)
        X_hvg, y_int, _, _, _, _, _ = load_cd34_data(
            CD34_DATA_PATH, CD34_N_HVG, CD34_RECOMPUTE_HVG, args.fast, SEED,
        )
        data = make_data_split(X_hvg, y_int, test_size=0.2, val_size=0.1, seed=SEED)
    else:
        base_cfg = ExpConfig(
            dataset=dataset_name,
            embed_dim=embed_dims[0], hidden_dim=HIDDEN_DIM,
            head_mode=HEAD_MODE, eval_prefixes=list(range(1, embed_dims[0] + 1)),
            lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
            weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
            experiment_name="exp14_two_eval_compare",
        )
        data = load_data(base_cfg)

    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)
    print(f"[exp14] Data: train={data.X_train.shape}  test={data.X_test.shape}"
          f"  n_classes={data.n_classes}\n")

    for embed_dim in embed_dims:
        dim_dir = os.path.join(run_dir, f"embed_{embed_dim}")
        os.makedirs(dim_dir, exist_ok=True)
        run_one_dim(embed_dim, data, y_train_np, y_test_np,
                    dim_dir, fig_stamp, args.fast,
                    max_lr_samples, max_1nn_e1, max_probe_samples,
                    dataset_name=dataset_name, epochs=epochs,
                    include_dense_mrl=include_dense_mrl)

    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)
    print(f"\n[exp14] Done. All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
