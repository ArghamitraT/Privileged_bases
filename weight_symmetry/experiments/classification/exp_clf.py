"""
Script: weight_symmetry/experiments/classification/exp_clf.py
-------------------------------------------------------------
Final classification experiment for the Weight Symmetry paper.

Two model families trained on MNIST, evaluated at every prefix k = 1..embed_dim:

  Full Prefix MRL-E
      MRL-E (efficient Matryoshka) — single W ∈ R^{C×d}, no bias.
      Loss: (1/d) Σ_k CE(z_{1:k} W_{:,1:k}^T, y)
      Every prefix explicitly trained; no bias term.

  Full Prefix L1 α (rev)  — one model per alpha in L1_ALPHAS
      Plain CE + non-uniform L1 penalty with power-alpha decay:
        Loss = CE(Wz, y) + λ Σ_j (d−j)^α |z_j|
      α=1.0 → linear decay; smaller α → softer ordering pressure.
      Dimensions reversed before all evaluation (most informative first).
      W columns also reversed for Eval 2.

Two evaluations at every prefix k:
  Eval 1 — Refitted  : fresh LR on ≤MAX_LR_SAMPLES subsampled z_train[:,:k]
  Eval 2 — Trained W : logits = z_test[:,:k] @ W[:,:k].T + b  (no new fitting)

1-NN: single ≤MAX_1NN_DB subsampled database, shared across both evals.

Only imports from weight_symmetry.* — no code/ imports.

Conda environment: mrl_env

Usage:
    python weight_symmetry/experiments/classification/exp_clf.py --fast
    python weight_symmetry/experiments/classification/exp_clf.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import TensorDataset, DataLoader

# ws-only imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from weight_symmetry.data.loader import load_data
from weight_symmetry.utility import create_run_dir, save_runtime, save_code_snapshot, save_config


# ==============================================================================
# CONFIG — edit here to change the experiment; use --fast for a smoke test
# ==============================================================================
EXPERIMENT_NOTE = (
    "Classificatio for weight symmetry but wide range of L1 α values, "
)
DATASET           = "mnist"
EMBED_DIM         = 16
HIDDEN_DIM        = 256
EPOCHS            = 20
PATIENCE          = 5
LR                = 1e-3
BATCH_SIZE        = 128
WEIGHT_DECAY      = 1e-4
SEED              = 42
L1_LAMBDA         = 0.05
L1_ALPHAS         = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # one PrefixL1 model per alpha; edit freely
MAX_LR_SAMPLES    = 10_000   # Eval 1: subsample cap for fresh LR fits
MAX_1NN_DB        = 10_000   # 1-NN database cap (shared across both evals)
MAX_PROBE_SAMPLES = 2_000    # per-dim probe LR subsample
# ==============================================================================

# Color palette — extended automatically if more alphas are added
_MRL_E_COLOR   = "mediumseagreen"
_L1_COLORS     = ["crimson", "orchid", "steelblue", "darkorange", "mediumpurple"]

def _build_color_map(l1_alphas):
    cmap = {"Full Prefix MRL-E": _MRL_E_COLOR}
    for i, a in enumerate(l1_alphas):
        cmap[_l1_name(a)] = _L1_COLORS[i % len(_L1_COLORS)]
    return cmap

def _l1_name(alpha):
    return f"Full Prefix L1 α={alpha} (rev)"


# ==============================================================================
# Architecture
# ==============================================================================

class MLPEncoder(nn.Module):
    """input → Linear → BN → ReLU → Drop → Linear → BN → ReLU → Drop → Linear → L2-norm"""
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)


class MRLEHead(nn.Module):
    """Single linear classifier, no bias — weight accessed directly for MRL-E."""
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes, bias=False)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


# ==============================================================================
# Loss functions
# ==============================================================================

class MRLELoss(nn.Module):
    """(1/d) Σ_k CE(z[:,:k] @ W[:,:k].T, y) — no bias, direct weight slicing."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss()
        self.embed_dim = embed_dim

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor,
                head: MRLEHead) -> torch.Tensor:
        W = head.head.weight  # (n_classes, embed_dim)
        losses = [
            self.ce(torch.matmul(embedding[:, :k], W[:, :k].T), labels)
            for k in range(1, self.embed_dim + 1)
        ]
        return torch.stack(losses).mean()


class PrefixL1Loss(nn.Module):
    """CE(head(z), y) + λ Σ_j (d−j)^alpha |z_j| — non-uniform L1 with power-alpha decay."""
    def __init__(self, embed_dim: int, lambda_l1: float = 0.05, alpha: float = 1.0):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss()
        self.lambda_l1 = lambda_l1
        j = torch.arange(embed_dim, dtype=torch.float32)
        w = (embed_dim - j) ** alpha
        self.register_buffer("dim_weights", w)
        print(f"[PrefixL1Loss] embed_dim={embed_dim}, lambda_l1={lambda_l1}, alpha={alpha}")
        print(f"  dim_weights (first 8): {w[:8].tolist()}")

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor,
                head: nn.Module) -> torch.Tensor:
        ce_loss     = self.ce(head(embedding), labels)
        weighted_l1 = (self.dim_weights * embedding.abs()).mean(dim=0).sum()
        return ce_loss + self.lambda_l1 * weighted_l1


# ==============================================================================
# Classification training loop (self-contained — no ws/training/trainer.py)
# ==============================================================================

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _clf_logger(run_dir: str, model_tag: str) -> logging.Logger:
    logger = logging.getLogger(f"clf_{model_tag}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if _tqdm is not None:
        class TqdmHandler(logging.StreamHandler):
            def emit(self, record):
                _tqdm.write(self.format(record))
        ch = TqdmHandler()
    else:
        ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(run_dir, f"{model_tag}_train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def train_clf(encoder, head, loss_fn, opt, data,
              epochs: int, patience: int, batch_size: int, seed: int,
              run_dir: str, model_tag: str) -> dict:
    """Classification training loop with early stopping and best-checkpoint saving."""
    logger    = _clf_logger(run_dir, model_tag)
    enc_ckpt  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_ckpt = os.path.join(run_dir, f"{model_tag}_head_best.pt")

    torch.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(data.X_train, data.y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(data.X_val,   data.y_val),
                              batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val, best_epoch, no_improve = float("inf"), 0, 0

    epoch_iter = (
        _tqdm(range(epochs), desc=f"[{model_tag}]", unit="epoch", leave=True)
        if _tqdm is not None else range(epochs)
    )

    for epoch in epoch_iter:
        encoder.train(); head.train()
        epoch_train = 0.0
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = loss_fn(encoder(X_b), y_b, head)
            loss.backward()
            opt.step()
            epoch_train += loss.item() * len(X_b)
        epoch_train /= len(data.X_train)
        train_losses.append(epoch_train)

        encoder.eval(); head.eval()
        epoch_val, n_correct, n_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                z      = encoder(X_b)
                loss   = loss_fn(z, y_b, head)
                logits = head(z)
                epoch_val  += loss.item() * len(X_b)
                n_correct  += (logits.argmax(1) == y_b).sum().item()
                n_total    += len(y_b)
        epoch_val /= len(data.X_val)
        val_losses.append(epoch_val)
        val_acc = n_correct / n_total

        logger.info(f"Epoch {epoch+1:>3}/{epochs}  "
                    f"train_loss={epoch_train:.4f}  "
                    f"val_loss={epoch_val:.4f}  val_acc={val_acc:.4f}")
        if _tqdm is not None:
            epoch_iter.set_postfix(train_loss=f"{epoch_train:.4f}",
                                   val_loss=f"{epoch_val:.4f}", val_acc=f"{val_acc:.4f}")

        if epoch_val < best_val:
            best_val, best_epoch, no_improve = epoch_val, epoch, 0
            torch.save(encoder.state_dict(), enc_ckpt)
            torch.save(head.state_dict(),    head_ckpt)
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

    encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu", weights_only=True))
    head.load_state_dict(   torch.load(head_ckpt, map_location="cpu", weights_only=True))
    encoder.eval(); head.eval()
    logger.info(f"Best checkpoint: epoch {best_epoch+1}, val_loss={best_val:.4f}")
    return {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}


# ==============================================================================
# Model training helpers
# ==============================================================================

def train_mrl_e(data, embed_dim, hidden_dim, epochs, patience, batch_size,
                lr, weight_decay, seed, run_dir) -> tuple:
    encoder = MLPEncoder(data.input_dim, hidden_dim, embed_dim)
    head    = MRLEHead(embed_dim, data.n_classes)
    loss_fn = MRLELoss(embed_dim)
    opt     = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                               lr=lr, weight_decay=weight_decay)
    train_clf(encoder, head, loss_fn, opt, data,
              epochs, patience, batch_size, seed, run_dir, "mrl_e")
    return encoder, head


def train_prefix_l1(data, embed_dim, hidden_dim, l1_lambda, alpha,
                    epochs, patience, batch_size, lr, weight_decay, seed,
                    run_dir, model_tag) -> tuple:
    encoder = MLPEncoder(data.input_dim, hidden_dim, embed_dim)
    head    = nn.Linear(embed_dim, data.n_classes)
    loss_fn = PrefixL1Loss(embed_dim, lambda_l1=l1_lambda, alpha=alpha)
    opt     = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                               lr=lr, weight_decay=weight_decay)
    train_clf(encoder, head, loss_fn, opt, data,
              epochs, patience, batch_size, seed, run_dir, model_tag)
    return encoder, head


# ==============================================================================
# Embedding extraction
# ==============================================================================

def get_embeddings(encoder, X: torch.Tensor) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        chunks = [encoder(X[i:i+512]).cpu().numpy() for i in range(0, len(X), 512)]
    return np.concatenate(chunks, axis=0)


# ==============================================================================
# Evaluation
# ==============================================================================

def eval1_linear_sweep(Z_train_sub, Z_test, y_train_sub, y_test,
                       eval_prefixes, seed, model_tag) -> dict:
    """Eval 1 — fresh LR at each prefix k."""
    print(f"\n[clf] Eval 1 linear sweep — {model_tag}")
    results = {}
    for k in eval_prefixes:
        lr = LogisticRegression(solver="saga", max_iter=1000,
                                random_state=seed, n_jobs=1)
        lr.fit(Z_train_sub[:, :k], y_train_sub)
        results[k] = float(lr.score(Z_test[:, :k], y_test))
    return results


def eval2_linear_sweep(Z_test, y_test, W, b, eval_prefixes, model_tag) -> dict:
    """Eval 2 — trained W directly, no new fitting."""
    print(f"\n[clf] Eval 2 linear sweep — {model_tag}")
    results = {}
    for k in eval_prefixes:
        logits = Z_test[:, :k] @ W[:, :k].T + b
        results[k] = float((np.argmax(logits, axis=1) == y_test).mean())
    return results


def eval_1nn_sweep(Z_train_db, Z_test, y_train_db, y_test,
                   eval_prefixes, model_tag) -> dict:
    """1-NN at each prefix k using a fixed database."""
    print(f"\n[clf] 1-NN sweep — {model_tag}")
    results = {}
    for k in eval_prefixes:
        nn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
        nn.fit(Z_train_db[:, :k], y_train_db)
        results[k] = float(nn.score(Z_test[:, :k], y_test))
    return results


def compute_importance_scores(Z_test, Z_train, y_train, y_test,
                               max_probe_samples, seed, model_tag) -> dict:
    """Per-dim importance: mean_abs, variance, probe_acc."""
    print(f"[clf] Importance scores — {model_tag}")
    mean_abs = np.mean(np.abs(Z_test), axis=0)
    variance = np.var(Z_test, axis=0)

    rng = np.random.default_rng(seed)
    n   = len(y_train)
    idx = (rng.choice(n, min(n, max_probe_samples), replace=False)
           if n > max_probe_samples else np.arange(n))
    d   = Z_test.shape[1]
    probe_acc = np.zeros(d)
    for j in range(d):
        lr = LogisticRegression(solver="saga", max_iter=500, random_state=seed, n_jobs=1)
        lr.fit(Z_train[idx, j:j+1], y_train[idx])
        probe_acc[j] = float(lr.score(Z_test[:, j:j+1], y_test))
    top3 = np.argsort(probe_acc)[::-1][:3].tolist()
    print(f"  top-3 dims by probe_acc: {top3}")
    return {"mean_abs": mean_abs, "variance": variance, "probe_acc": probe_acc}


def compute_method_agreement(scores: dict, model_tag: str) -> dict:
    """Spearman ρ between each pair of importance methods."""
    methods = ["mean_abs", "variance", "probe_acc"]
    pairs   = [("mean_abs", "variance"), ("mean_abs", "probe_acc"), ("variance", "probe_acc")]
    rhos    = {}
    for ma, mb in pairs:
        rho, _ = spearmanr(scores[ma], scores[mb])
        rhos[(ma, mb)] = float(rho)
    return rhos


# ==============================================================================
# Plotting
# ==============================================================================

def plot_training_curves(run_dir: str, model_tags: list, fig_stamp: str):
    histories = {}
    for tag in model_tags:
        log_path = os.path.join(run_dir, f"{tag}_train.log")
        if not os.path.isfile(log_path):
            continue
        tl, vl = [], []
        with open(log_path) as fh:
            for line in fh:
                if "train_loss=" in line and "val_loss=" in line:
                    try:
                        tl.append(float(line.split("train_loss=")[1].split()[0]))
                        vl.append(float(line.split("val_loss=")[1].split()[0]))
                    except (IndexError, ValueError):
                        pass
        if tl:
            histories[tag] = {"train": tl, "val": vl}

    n = max(1, len(histories))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for col, (tag, hist) in enumerate(histories.items()):
        ax = axes[0, col]
        ep = range(1, len(hist["train"]) + 1)
        ax.plot(ep, hist["train"], label="Train", linewidth=2)
        ax.plot(ep, hist["val"],   label="Val",   linewidth=2, linestyle="--")
        ax.set_title(tag, fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[clf] Saved training_curves{fig_stamp}.png")


def plot_combined_comparison(eval1_linear, eval2_linear, nn_results,
                              eval_prefixes, model_names, color_map,
                              run_dir: str, embed_dim: int, fig_stamp: str):
    """2×2 grid: (Eval1 linear, Eval2 linear) × (1-NN shared)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = [
        (axes[0, 0], eval1_linear, "Linear Accuracy — Eval 1 (Fresh LR)"),
        (axes[0, 1], eval2_linear, "Linear Accuracy — Eval 2 (Trained W)"),
        (axes[1, 0], nn_results,   "1-NN Accuracy (shared database)"),
        (axes[1, 1], nn_results,   "1-NN Accuracy (same, right panel)"),
    ]
    for ax, results, title in panels:
        for name in model_names:
            color = color_map.get(name, "gray")
            ls    = "--" if "L1" in name else "-"
            accs  = [results[name].get(k, float("nan")) for k in eval_prefixes]
            ax.plot(eval_prefixes, accs, color=color, label=name,
                    linewidth=2, linestyle=ls)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Prefix size k", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        shown = list(range(1, embed_dim + 1, max(1, embed_dim // 8)))
        if embed_dim not in shown:
            shown.append(embed_dim)
        ax.set_xticks(shown)

    axes[0, 0].set_ylabel("Accuracy", fontsize=10)
    axes[1, 0].set_ylabel("Accuracy", fontsize=10)
    fig.suptitle("Full Prefix MRL-E vs PrefixL1 — Prefix Sweep", fontsize=13)
    plt.tight_layout()
    out = os.path.join(run_dir, f"combined_comparison{fig_stamp}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[clf] Saved combined_comparison{fig_stamp}.png")


def plot_importance_scores(all_scores, model_names, color_map,
                            embed_dim, run_dir, fig_stamp):
    METHODS = ["mean_abs", "variance", "probe_acc"]
    MLABELS = {"mean_abs": "Mean |z|", "variance": "Variance", "probe_acc": "1D Probe Acc"}
    n_m, n_mod = len(METHODS), len(model_names)
    tick_step  = max(1, embed_dim // 8)

    fig, axes = plt.subplots(n_m, n_mod,
                              figsize=(5 * n_mod, max(3, embed_dim * 0.22) * n_m),
                              squeeze=False)
    for row, method in enumerate(METHODS):
        for col, name in enumerate(model_names):
            ax     = axes[row, col]
            scores = all_scores[name][method]
            color  = color_map.get(name, "gray")
            ax.barh(np.arange(embed_dim), scores, color=color, alpha=0.8)
            ax.set_xlim(left=0); ax.invert_yaxis()
            visible = [d for d in range(embed_dim) if d % tick_step == 0]
            ax.set_yticks(visible)
            ax.set_yticklabels([f"d{d}" for d in visible], fontsize=7)
            ax.set_xlabel(MLABELS[method], fontsize=9)
            if row == 0:
                ax.set_title(name, fontsize=10, color=color, fontweight="bold")
            if col == 0:
                ax.set_ylabel(MLABELS[method], fontsize=10)
    fig.suptitle("Per-Dimension Importance Scores", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(run_dir, f"importance_scores{fig_stamp}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[clf] Saved importance_scores{fig_stamp}.png")


def plot_method_agreement(all_scores, all_agreement, model_names, color_map,
                           embed_dim, run_dir, fig_stamp):
    MLABELS = {"mean_abs": "Mean |z|", "variance": "Variance", "probe_acc": "1D Probe Acc"}
    PAIRS   = [("mean_abs", "variance"), ("mean_abs", "probe_acc"), ("variance", "probe_acc")]
    PLABELS = {p: f"{MLABELS[p[0]]} vs {MLABELS[p[1]]}" for p in PAIRS}
    annotate = embed_dim <= 16
    n_mod, n_p = len(model_names), len(PAIRS)

    fig, axes = plt.subplots(n_mod, n_p, figsize=(5 * n_p, 4 * n_mod), squeeze=False)
    for row, name in enumerate(model_names):
        color = color_map.get(name, "gray")
        for col, (ma, mb) in enumerate(PAIRS):
            ax  = axes[row, col]
            x, y = all_scores[name][ma], all_scores[name][mb]
            rho  = all_agreement[name].get((ma, mb), float("nan"))
            ax.scatter(x, y, color=color, alpha=0.7, s=40)
            if annotate:
                for d, (xi, yi) in enumerate(zip(x, y)):
                    ax.annotate(str(d), (xi, yi), fontsize=6, ha="left", va="bottom")
            ax.text(0.05, 0.92, f"ρ = {rho:.3f}", transform=ax.transAxes,
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="white", alpha=0.8))
            ax.set_xlabel(MLABELS[ma], fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(PLABELS[(ma, mb)], fontsize=10)
            ax.set_ylabel(
                (f"{name}\n" if col == 0 else "") + MLABELS[mb],
                fontsize=9, **({"color": color, "fontweight": "bold"} if col == 0 else {})
            )
    fig.suptitle("Importance Method Agreement (Spearman ρ)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(run_dir, f"method_agreement{fig_stamp}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[clf] Saved method_agreement{fig_stamp}.png")


# ==============================================================================
# Results summary — every k
# ==============================================================================

def save_results_summary(eval1_linear, eval2_linear, nn_results,
                          eval_prefixes, model_names, all_scores, all_agreement,
                          embed_dim, run_dir):
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("Full Prefix MRL-E vs PrefixL1 — Classification Results\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"embed_dim={embed_dim}  eval_prefixes=1..{embed_dim}\n\n")

        for eval_label, lin_res in [
            ("EVAL 1 — Refitted LR", eval1_linear),
            ("EVAL 2 — Trained W (no refitting)", eval2_linear),
        ]:
            f.write(f"{'=' * 72}\n{eval_label}\n{'=' * 72}\n\n")

            # Header
            col_w = 22
            hdr   = f"{'k':>4}  " + "  ".join(f"{m[:col_w]:<{col_w}}" for m in model_names)

            for metric_label, res in [("Linear Accuracy", lin_res),
                                       ("1-NN Accuracy",   nn_results)]:
                f.write(f"{metric_label}\n{'-' * 60}\n{hdr}\n{'-' * len(hdr)}\n")
                for k in eval_prefixes:
                    row = f"{k:>4}  "
                    row += "  ".join(f"{res[m].get(k, float('nan')):>{col_w}.4f}"
                                     for m in model_names)
                    f.write(row + "\n")
                f.write("\n")

        # Eval2 − Eval1 gap
        f.write("EVAL 2 − EVAL 1 LINEAR ACCURACY GAP  (+ = Eval2 > Eval1)\n")
        f.write("-" * 60 + "\n")
        col_w = 22
        hdr   = f"{'k':>4}  " + "  ".join(f"{m[:col_w]:<{col_w}}" for m in model_names)
        f.write(hdr + "\n" + "-" * len(hdr) + "\n")
        for k in eval_prefixes:
            row = f"{k:>4}  "
            for m in model_names:
                gap = eval2_linear[m].get(k, float("nan")) - eval1_linear[m].get(k, float("nan"))
                row += f"  {gap:>+{col_w-2}.4f}"
            f.write(row + "\n")
        f.write("\n\n")

        # Importance scores
        f.write("IMPORTANCE SCORES — Top-5 dims per model per method\n")
        f.write("-" * 60 + "\n")
        for name in model_names:
            f.write(f"  {name}\n")
            for method in ["mean_abs", "variance", "probe_acc"]:
                top5 = np.argsort(all_scores[name][method])[::-1][:5].tolist()
                f.write(f"    {method:<12}: {top5}\n")
            f.write("\n")

        # Method agreement
        f.write("METHOD AGREEMENT — Spearman ρ\n")
        f.write("-" * 60 + "\n")
        for name in model_names:
            rhos = [all_agreement[name].get(p, float("nan"))
                    for p in [("mean_abs", "variance"),
                               ("mean_abs", "probe_acc"),
                               ("variance", "probe_acc")]]
            f.write(f"  {name}: "
                    f"mean|z|×Var={rhos[0]:+.3f}  "
                    f"mean|z|×Probe={rhos[1]:+.3f}  "
                    f"Var×Probe={rhos[2]:+.3f}\n")
        f.write("\n")

    print(f"[clf] Saved results_summary.txt")


# ==============================================================================
# Experiment description
# ==============================================================================

def save_experiment_description(cfg: dict, run_dir: str):
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Full Prefix MRL-E vs PrefixL1 — Weight Symmetry Classification\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            "Models:\n"
            "  Full Prefix MRL-E : single W (no bias), trained at every prefix k=1..d\n"
            "                      via (1/d) Σ_k CE(z[:,:k] @ W[:,:k].T, y)\n"
            "  Full Prefix L1 α  : CE(Wz,y) + λ Σ_j (d-j)^α |z_j|, one per alpha\n"
            "                      dims reversed before all evaluation\n\n"
            "Evaluations (both at every k=1..d):\n"
            "  Eval 1 — Refitted  : fresh LR on ≤MAX_LR_SAMPLES subsampled z_train[:,:k]\n"
            "  Eval 2 — Trained W : logits = z_test[:,:k] @ W[:,:k].T + b (no fitting)\n"
            "  1-NN               : ≤MAX_1NN_DB subsampled database, shared both evals\n\n"
        )
        f.write("CONFIG\n" + "-" * 40 + "\n")
        for k, v in cfg.items():
            f.write(f"  {k:<22} = {v}\n")
        f.write("\n")
    print(f"[clf] Saved experiment_description.log")


# ==============================================================================
# Main experiment
# ==============================================================================

def run_experiment(fast: bool = False):
    run_start = time.time()

    # Fast-mode overrides
    if fast:
        dataset           = "digits"
        embed_dim         = 8
        hidden_dim        = 64
        epochs            = 5
        patience          = 3
        max_lr_samples    = 500
        max_1nn_db        = 500
        max_probe_samples = 200
    else:
        dataset           = DATASET
        embed_dim         = EMBED_DIM
        hidden_dim        = HIDDEN_DIM
        epochs            = EPOCHS
        patience          = PATIENCE
        max_lr_samples    = MAX_LR_SAMPLES
        max_1nn_db        = MAX_1NN_DB
        max_probe_samples = MAX_PROBE_SAMPLES

    cfg = {
        "dataset": dataset, "embed_dim": embed_dim, "hidden_dim": hidden_dim,
        "epochs": epochs, "patience": patience, "lr": LR,
        "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY, "seed": SEED,
        "l1_lambda": L1_LAMBDA, "l1_alphas": L1_ALPHAS,
        "max_lr_samples": max_lr_samples, "max_1nn_db": max_1nn_db,
        "max_probe_samples": max_probe_samples, "fast": fast,
    }

    # Seed
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    run_dir   = create_run_dir(fast=fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    eval_prefixes = list(range(1, embed_dim + 1))
    color_map = _build_color_map(L1_ALPHAS)

    print(f"[clf] Output → {run_dir}")
    print(f"[clf] embed_dim={embed_dim}  L1_ALPHAS={L1_ALPHAS}\n")

    save_experiment_description(cfg, run_dir)
    save_config(cfg, run_dir)

    # Load data
    data = load_data(dataset, seed=SEED)
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)
    print(f"[clf] Data: train={data.X_train.shape}  test={data.X_test.shape}\n")

    # ---- Train all models ----
    print("[clf] Training Full Prefix MRL-E ...")
    mrle_enc, mrle_head = train_mrl_e(
        data, embed_dim, hidden_dim, epochs, patience, BATCH_SIZE,
        LR, WEIGHT_DECAY, SEED, run_dir,
    )

    pl1_models = {}
    for alpha in L1_ALPHAS:
        tag = f"prefix_l1_a{str(alpha).replace('.', '')}"
        print(f"\n[clf] Training Full Prefix L1 α={alpha} ...")
        enc, head = train_prefix_l1(
            data, embed_dim, hidden_dim, L1_LAMBDA, alpha,
            epochs, patience, BATCH_SIZE, LR, WEIGHT_DECAY, SEED,
            run_dir, tag,
        )
        pl1_models[alpha] = (enc, head)

    # ---- Training curves ----
    all_tags = ["mrl_e"] + [f"prefix_l1_a{str(a).replace('.', '')}" for a in L1_ALPHAS]
    plot_training_curves(run_dir, all_tags, fig_stamp)

    # ---- Extract embeddings ----
    Z_tr_mrle = get_embeddings(mrle_enc, data.X_train)
    Z_te_mrle = get_embeddings(mrle_enc, data.X_test)
    W_mrle    = mrle_head.head.weight.detach().cpu().numpy()
    b_mrle    = np.zeros(W_mrle.shape[0])

    models_data = {"Full Prefix MRL-E": (Z_tr_mrle, Z_te_mrle, W_mrle, b_mrle)}

    for alpha, (enc, head) in pl1_models.items():
        # Reverse dims (most informative first) and mirror W columns accordingly
        Ztr   = np.ascontiguousarray(get_embeddings(enc, data.X_train)[:, ::-1])
        Zte   = np.ascontiguousarray(get_embeddings(enc, data.X_test)[:,  ::-1])
        W_rev = np.ascontiguousarray(head.weight.detach().cpu().numpy()[:, ::-1])
        b     = head.bias.detach().cpu().numpy()
        models_data[_l1_name(alpha)] = (Ztr, Zte, W_rev, b)

    model_names = list(models_data.keys())

    # ---- Subsampling for Eval 1 LR and 1-NN database ----
    rng    = np.random.default_rng(SEED)
    n_tr   = len(y_train_np)
    lr_idx = (rng.choice(n_tr, max_lr_samples, replace=False)
              if n_tr > max_lr_samples else np.arange(n_tr))

    # Build per-model 1-NN database (same seed, same subsample fraction)
    nn_db = {}
    for name, (Ztr, Zte, W, b) in models_data.items():
        rng2   = np.random.default_rng(SEED)
        db_idx = (rng2.choice(n_tr, max_1nn_db, replace=False)
                  if n_tr > max_1nn_db else np.arange(n_tr))
        nn_db[name] = (Ztr[db_idx], y_train_np[db_idx])

    # ---- Eval 1 — fresh LR ----
    eval1_linear = {}
    for name, (Ztr, Zte, W, b) in models_data.items():
        eval1_linear[name] = eval1_linear_sweep(
            Ztr[lr_idx], Zte, y_train_np[lr_idx], y_test_np,
            eval_prefixes, SEED, name,
        )

    # ---- Eval 2 — trained W ----
    eval2_linear = {}
    for name, (Ztr, Zte, W, b) in models_data.items():
        eval2_linear[name] = eval2_linear_sweep(Zte, y_test_np, W, b, eval_prefixes, name)

    # ---- 1-NN — shared database per model ----
    nn_results = {}
    for name, (Ztr, Zte, W, b) in models_data.items():
        db_Z, db_y = nn_db[name]
        nn_results[name] = eval_1nn_sweep(db_Z, Zte, db_y, y_test_np,
                                           eval_prefixes, name)

    # ---- Importance scores + method agreement ----
    all_scores, all_agreement = {}, {}
    for name, (Ztr, Zte, W, b) in models_data.items():
        all_scores[name]    = compute_importance_scores(
            Zte, Ztr, y_train_np, y_test_np, max_probe_samples, SEED, name)
        all_agreement[name] = compute_method_agreement(all_scores[name], name)

    # ---- Plots ----
    plot_combined_comparison(eval1_linear, eval2_linear, nn_results,
                              eval_prefixes, model_names, color_map,
                              run_dir, embed_dim, fig_stamp)
    plot_importance_scores(all_scores, model_names, color_map,
                           embed_dim, run_dir, fig_stamp)
    plot_method_agreement(all_scores, all_agreement, model_names, color_map,
                          embed_dim, run_dir, fig_stamp)

    # ---- Results summary (every k) ----
    save_results_summary(eval1_linear, eval2_linear, nn_results,
                          eval_prefixes, model_names, all_scores, all_agreement,
                          embed_dim, run_dir)

    # ---- Quick stdout table ----
    sample_ks = [k for k in eval_prefixes if k in {1, 2, 4, 8, 16, 32, embed_dim}]
    print(f"\n  Eval 1 linear accuracy at selected k:")
    print(f"  {'k':>3}  " + "  ".join(f"{n[:18]:<18}" for n in model_names))
    for k in sample_ks:
        row = f"  {k:>3}  "
        row += "  ".join(f"{eval1_linear[n].get(k, float('nan')):>18.4f}" for n in model_names)
        print(row)

    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)
    print(f"\n[clf] Done → {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Full Prefix MRL-E vs PrefixL1 — Weight Symmetry Classification"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test: digits, embed_dim=8, 5 epochs.")
    args = parser.parse_args()
    run_experiment(fast=args.fast)


if __name__ == "__main__":
    main()
