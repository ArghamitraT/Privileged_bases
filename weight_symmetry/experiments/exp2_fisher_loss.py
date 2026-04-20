"""
Experiment 2 (Fisher): Fisher / LDA Loss — Does ordering require a prefix loss?
---------------------------------------------------------------------------------
Hypothesis: CE loss does not recover the LDA *ordering* because it is rotation-
invariant within the optimal subspace. Fisher loss (LDA criterion) should also
recover the LDA span with no ordering under standard training, but when wrapped
in a full-prefix sum it should force ordered LDA recovery — dim 1 ≈ top LDA
direction, dim 2 ≈ second, etc.

Models trained (embed_dim = n_lda = 19):
    1. fisher          — FisherLoss on full d-dim embedding (span only)
    2. fp_fisher       — FullPrefixFisherLoss: sum of Fisher over k=1..d
                         → expected to recover ordered LDA
    3. prefix_l1_fisher— FisherLoss(full) + front-loaded L1 penalty; dims reversed
                         → alternative ordering mechanism without prefix Fisher
    4. std_mrl_fisher  — Fisher at fixed M prefix sizes only (not all k=1..d)
                         → recovers LDA ordering at M sizes, gaps between

Baseline:
    5. LDA projection — sklearn LDA directions; angle = 0 by construction.

Primary metric: mean principal angle between B^T[:,1:k] and top-k LDA directions
Secondary     : cosine similarity to LDA, prefix classification accuracy

Usage:
    Conda environment: mrl_env

    python weight_symmetry/experiments/exp2_fisher_loss.py --fast
    python weight_symmetry/experiments/exp2_fisher_loss.py
    python weight_symmetry/experiments/exp2_fisher_loss.py --sgd                    # plain SGD instead of default full-batch Adam
    python weight_symmetry/experiments/exp2_fisher_loss.py --models fp_fisher       # train only fp_fisher
    python weight_symmetry/experiments/exp2_fisher_loss.py --use-weights FOLDER
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_WS_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.utility import (
    create_run_dir, save_runtime, save_code_snapshot, save_config
)
from weight_symmetry.data.loader import load_data
from weight_symmetry.data.synthetic import load_synthetic, SYNTHETIC_VARIANTS
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.losses.losses import (
    FisherLoss, FullPrefixFisherLoss,
    PrefixL1FisherLoss, StandardMRLFisherLoss,
)
from weight_symmetry.training.trainer import train_ae
from weight_symmetry.evaluation.metrics import (
    compute_encoder_subspace_metrics, compute_prefix_accuracy
)

# ==============================================================================
# CONFIG — edit here to change the full run
# ==============================================================================
EXPERIMENT_NOTE = ("Fisher running for l1 prefix and standard MRL losses; 10k epoch")
DATASET           = "synthetic"
SYNTHETIC_VARIANT = "orderedBoth"   # 20 classes → n_lda = 19; ordered signal dims
EMBED_DIM         = 19              # = n_lda = C-1  (matches LDA dimension exactly)
EPOCHS            = 10000
PATIENCE          = 2000
LR                = 1e-3            # Adam default; SGD needs ~1e-2
BATCH_SIZE        = None            # None = full training set (exact scatter matrices)
OPTIMIZER         = "adam"          # "adam" (adaptive LR) | "sgd" (plain GD)
WEIGHT_DECAY      = 1e-4            # only used when OPTIMIZER="adam"
SEEDS             = [47]
FISHER_EPS        = 1e-4            # S_W regularisation in Fisher loss
L1_LAMBDA         = 0.01            # L1 penalty strength for PrefixL1FisherLoss
STANDARD_MRL_M    = [5, 10, 15, 19] # prefix sizes for StandardMRLFisherLoss
# ==============================================================================

MODEL_CONFIGS = [
    dict(tag="fisher",           loss="fisher",           label="Fisher (standard)",      flip_dims=False),
    dict(tag="fp_fisher",        loss="fp_fisher",        label="Full-prefix Fisher",     flip_dims=False),
    dict(tag="prefix_l1_fisher", loss="prefix_l1_fisher", label="PrefixL1 Fisher (rev)",  flip_dims=True),
    dict(tag="std_mrl_fisher",   loss="std_mrl_fisher",   label="Standard MRL Fisher",    flip_dims=False),
]

PLOT_STYLES = {
    "Fisher (standard)":     dict(color="#0072B2", linestyle="--",  marker="o", markersize=3),
    "Full-prefix Fisher":    dict(color="#009E73", linestyle="-",   marker=""),
    "PrefixL1 Fisher (rev)": dict(color="#E69F00", linestyle="-.",  marker="s", markersize=3),
    "Standard MRL Fisher":   dict(color="#CC79A7", linestyle="--",  marker="^", markersize=3),
    "LDA baseline":          dict(color="black",   linestyle=":",   marker=""),
}


# ==============================================================================
# Log parsing (fallback when histories_raw.npz is missing)
# ==============================================================================

def parse_train_log(log_path: str) -> dict:
    """Extract train/val loss history from a trainer .log file."""
    import re
    train_losses, val_losses, best_epoch = [], [], 0
    pat_epoch = re.compile(
        r'\S+\s+Epoch\s+\d+/\d+\s+train=([\d.\-eE+]+)\s+val=([\d.\-eE+]+)')
    pat_best  = re.compile(r'Best checkpoint loaded \(epoch (\d+)')
    with open(log_path) as fh:
        for line in fh:
            m = pat_epoch.search(line)
            if m:
                train_losses.append(float(m.group(1)))
                val_losses.append(float(m.group(2)))
            m2 = pat_best.search(line)
            if m2:
                best_epoch = int(m2.group(1))
    return {"train_losses": train_losses, "val_losses": val_losses,
            "best_epoch": best_epoch}


# ==============================================================================
# Build loss
# ==============================================================================

def build_loss_fn(loss_type: str, eps: float, l1_lambda: float = 0.01,
                  prefix_sizes: list = None):
    if loss_type == "fisher":
        return FisherLoss(eps=eps)
    elif loss_type == "fp_fisher":
        return FullPrefixFisherLoss(eps=eps)
    elif loss_type == "prefix_l1_fisher":
        return PrefixL1FisherLoss(l1_lambda=l1_lambda, eps=eps)
    elif loss_type == "std_mrl_fisher":
        if not prefix_sizes:
            raise ValueError("std_mrl_fisher requires prefix_sizes in cfg")
        return StandardMRLFisherLoss(prefix_sizes=prefix_sizes, eps=eps)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==============================================================================
# LDA eigenvalues + paired cosine
# ==============================================================================

def compute_lda_eigenvalues(data) -> np.ndarray:
    """
    Fit sklearn LDA on training data and return explained_variance_ratio_ as λ_j weights.
    These are the normalised LDA eigenvalues in descending order (sum = 1).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(data.X_train.numpy(), data.y_train.numpy())
    return lda.explained_variance_ratio_.astype(np.float64)


def compute_paired_cosines_lda(model, lda_dirs: np.ndarray,
                               flip_dims: bool = False) -> np.ndarray:
    """
    Per-dimension paired cosine |cos(b_k, lda_k)| for k = 1..n_lda.
    b_k = k-th encoder row (in R^p), lda_k = k-th LDA direction.
    Tests whether encoder dim k aligns with exactly LDA direction k.
    For PrefixL1 models pass flip_dims=True to reverse encoder columns before pairing.
    """
    from weight_symmetry.evaluation.metrics import paired_cosine
    n_lda  = lda_dirs.shape[1]
    B_T    = model.get_encoder_matrix().cpu().numpy().T.astype(np.float64)  # (p, d)
    if flip_dims:
        B_T = np.ascontiguousarray(B_T[:, ::-1])
    n_dims = min(n_lda, B_T.shape[1])   # can only pair up to embed_dim
    return np.array([
        paired_cosine(B_T[:, k], lda_dirs[:, k]) for k in range(n_dims)
    ])


# ==============================================================================
# LDA baseline
# ==============================================================================

def compute_lda_baseline_metrics(lda_dirs, data, embed_dim):
    """
    LDA projection baseline: encoder = LDA directions directly.
    Angle to LDA = 0 at every k by definition.
    Accuracy = logistic regression on LDA-projected data.
    """
    from sklearn.linear_model import LogisticRegression
    from weight_symmetry.evaluation.metrics import column_alignment

    d     = embed_dim
    n_lda = lda_dirs.shape[1]

    X_tr = data.X_train.numpy().astype(np.float64)
    X_te = data.X_test.numpy().astype(np.float64)
    y_tr = data.y_train.numpy()
    y_te = data.y_test.numpy()

    lda_angles  = []
    lda_cosine  = []
    accuracies  = []

    for k in range(1, d + 1):
        lda_angles.append(0.0)
        L_k = lda_dirs[:, :k]
        lda_cosine.append(column_alignment(L_k, L_k))   # = 1.0 by definition

        Z_tr = X_tr @ lda_dirs[:, :k]
        Z_te = X_te @ lda_dirs[:, :k]
        clf  = LogisticRegression(max_iter=300, n_jobs=1)
        clf.fit(Z_tr, y_tr)
        accuracies.append(float(clf.score(Z_te, y_te)))

    return {
        "prefix_sizes": list(range(1, d + 1)),
        "lda_angles":   lda_angles,
        "lda_cosine":   lda_cosine,
        "accuracies":   accuracies,
        "n_lda":        n_lda,
    }


# ==============================================================================
# Training
# ==============================================================================

def train_all_models(data, cfg, run_dir, seed, device):
    # Full-batch GD: override batch_size to full training set size
    n_train    = len(data.X_train)
    batch_size = cfg.get("batch_size") or n_train   # None → full batch
    use_adam   = cfg.get("optimizer", "sgd") == "adam"
    active     = cfg.get("models") or [mc["tag"] for mc in MODEL_CONFIGS]

    models = {}
    for mc in [mc for mc in MODEL_CONFIGS if mc["tag"] in active]:
        tag = mc["tag"]
        model = LinearAE(data.input_dim, cfg["embed_dim"]).to(device)
        torch.manual_seed(seed)
        loss_fn = build_loss_fn(
            mc["loss"],
            eps=cfg.get("fisher_eps", FISHER_EPS),
            l1_lambda=cfg.get("l1_lambda", L1_LAMBDA),
            prefix_sizes=cfg.get("standard_mrl_m", STANDARD_MRL_M),
        )
        if use_adam:
            opt = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0)
            )
        else:
            opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
        seed_cfg              = dict(cfg)
        seed_cfg["seed"]      = seed
        seed_cfg["batch_size"] = batch_size
        history = train_ae(
            model, loss_fn, opt, data, seed_cfg,
            run_dir, f"seed{seed}_{tag}",
            orthogonalize=False, supervised=True,
        )
        models[tag] = (model, history)
    return models


# ==============================================================================
# Plots
# ==============================================================================

def plot_training_curves(all_histories, run_dir, fig_stamp):
    fig, ax = plt.subplots(figsize=(8, 4))
    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        style = PLOT_STYLES[label]
        rows    = [h["train_losses"] for h in all_histories[tag]]
        max_len = max(len(r) for r in rows)
        mat     = np.array([
            np.pad(r, (0, max_len - len(r)), constant_values=r[-1]) for r in rows
        ])
        epochs = np.arange(1, max_len + 1)
        mean   = mat.mean(0)
        std    = mat.std(0)
        ax.plot(epochs, mean, label=label, **style)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=style["color"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fisher loss (negative Tr criterion)")
    ax.set_title("Training curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2_fisher] Saved {path}")


def plot_metrics(all_metrics, lda_baseline, all_accuracies, run_dir, fig_stamp, embed_dim):
    """Three-panel figure: LDA angle | LDA cosine | Prefix accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n_lda = min(lda_baseline["n_lda"], embed_dim)
    xs    = list(range(1, n_lda + 1))

    # ── Panel 1: Angle to LDA ────────────────────────────────────────────────
    ax = axes[0]
    for mc in MODEL_CONFIGS:
        tag, label = mc["tag"], mc["label"]
        style   = PLOT_STYLES[label]
        n_seeds = len(all_metrics[tag])
        mat     = np.array([[all_metrics[tag][s]["lda_angles"][k] for k in range(n_lda)]
                            for s in range(n_seeds)])
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(xs, mean, label=label, **style)
        if n_seeds > 1:
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=style["color"])
    ax.plot(xs, lda_baseline["lda_angles"][:n_lda], label="LDA baseline",
            **PLOT_STYLES["LDA baseline"])
    ax.set_xlabel("Prefix size k")
    ax.set_ylabel("Mean principal angle (°)")
    ax.set_title("Subspace angle to LDA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Cosine similarity to LDA ───────────────────────────────────
    ax = axes[1]
    for mc in MODEL_CONFIGS:
        tag, label = mc["tag"], mc["label"]
        style   = PLOT_STYLES[label]
        n_seeds = len(all_metrics[tag])
        mat     = np.array([[all_metrics[tag][s]["lda_cosine"][k] for k in range(n_lda)]
                            for s in range(n_seeds)])
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(xs, mean, label=label, **style)
        if n_seeds > 1:
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=style["color"])
    ax.plot(xs, lda_baseline["lda_cosine"][:n_lda], label="LDA baseline",
            **PLOT_STYLES["LDA baseline"])
    ax.set_xlabel("Prefix size k")
    ax.set_ylabel("Mean max cosine similarity")
    ax.set_title("Cosine similarity to LDA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Prefix accuracy ─────────────────────────────────────────────
    ax = axes[2]
    all_xs = list(range(1, embed_dim + 1))
    for mc in MODEL_CONFIGS:
        tag, label = mc["tag"], mc["label"]
        style   = PLOT_STYLES[label]
        n_seeds = len(all_accuracies[tag])
        mat     = np.array(all_accuracies[tag])   # (n_seeds, embed_dim)
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(all_xs, mean, label=label, **style)
        if n_seeds > 1:
            ax.fill_between(all_xs, mean - std, mean + std, alpha=0.15, color=style["color"])
    ax.plot(all_xs, lda_baseline["accuracies"], label="LDA baseline",
            **PLOT_STYLES["LDA baseline"])
    ax.set_xlabel("Prefix size k")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Prefix classification accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(run_dir, f"fisher_metrics{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2_fisher] Saved {path}")


def plot_ordered_lda_recovery(all_paired_cosines, lda_eigenvalues,
                              run_dir, fig_stamp, top_k=5):
    """
    Two-panel figure testing whether encoder dims recover LDA directions in order.

    Left  (Option 1): |cos(b_k, lda_k)| for k = 1..top_k only.
        Shows alignment with the most important LDA directions individually.

    Right (Option 2): Eigenvalue-weighted score Σ_j λ_j |cos(b_j, lda_j)|
        Single scalar per model; λ_j = LDA explained_variance_ratio (sums to 1).
        Score = 1.0 only if every encoder dim k aligns perfectly with lda_k,
        weighted by how discriminative each direction is.
    """
    n_lda  = lda_eigenvalues.shape[0]
    avail  = min(top_k, np.array(list(all_paired_cosines.values())[0]).shape[1])
    top_k  = min(top_k, n_lda, avail)
    xs     = list(range(1, top_k + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: per-dimension paired cosine (top_k only) ───────────────────────
    ax = axes[0]
    for mc in MODEL_CONFIGS:
        tag, label = mc["tag"], mc["label"]
        style  = PLOT_STYLES[label]
        mat    = np.array(all_paired_cosines[tag])[:, :top_k]   # (n_seeds, top_k)
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(xs, mean, label=label, **style)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=style["color"])

    # LDA baseline = 1.0 by definition
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="LDA baseline")
    ax.set_xlabel("Dimension k")
    ax.set_ylabel("|cos(b_k, lda_k)|")
    ax.set_title(
        f"Option 1 — Paired cosine for top-{top_k} LDA directions\n"
        "|cos(b_k, lda_k)|: encoder dim k vs LDA direction k\n"
        "1.0 = perfect ordered recovery"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: eigenvalue-weighted scalar ────────────────────────────────────
    ax = axes[1]
    model_labels = [mc["label"] for mc in MODEL_CONFIGS]
    colors       = [PLOT_STYLES[l]["color"] for l in model_labels]
    scores       = []
    for mc in MODEL_CONFIGS:
        tag    = mc["tag"]
        mat    = np.array(all_paired_cosines[tag])             # (n_seeds, n_dims)
        n_dims = mat.shape[1]                                  # may be < n_lda
        eigs   = lda_eigenvalues[:n_dims] / lda_eigenvalues[:n_dims].sum()
        seed_scores = (mat * eigs[np.newaxis, :]).sum(axis=1)
        scores.append((seed_scores.mean(), seed_scores.std()))

    xs_bar = np.arange(len(MODEL_CONFIGS))
    means  = [s[0] for s in scores]
    stds   = [s[1] for s in scores]
    ax.bar(xs_bar, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(xs_bar, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="Perfect (LDA baseline)")

    ax.set_xticks(xs_bar)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Σ λ_j |cos(b_j, lda_j)|")
    ax.set_title(
        "Option 2 — Eigenvalue-weighted ordered LDA recovery\n"
        "Σ_j λ_j |cos(b_j, lda_j)|,  λ_j = LDA explained variance ratio\n"
        "1.0 = perfect ordered recovery weighted by discriminability"
    )
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(run_dir, f"ordered_lda_recovery{fig_stamp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[exp2_fisher] Saved {path}")


# ==============================================================================
# Save data + summary
# ==============================================================================

def save_raw_data(all_metrics, all_histories, all_accuracies, lda_baseline, run_dir):
    md = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        md[f"{tag}_lda_angles"] = np.array([m["lda_angles"] for m in all_metrics[tag]])
        md[f"{tag}_lda_cosine"] = np.array([m["lda_cosine"] for m in all_metrics[tag]])
        md[f"{tag}_pca_angles"] = np.array([m["pca_angles"] for m in all_metrics[tag]])
        md[f"{tag}_pca_cosine"] = np.array([m["pca_cosine"] for m in all_metrics[tag]])
        md[f"{tag}_prefix_sizes"] = np.array(all_metrics[tag][0]["prefix_sizes"])
        md[f"{tag}_n_lda"]        = np.array(all_metrics[tag][0]["n_lda"])
    md["lda_baseline_lda_angles"]  = np.array(lda_baseline["lda_angles"])
    md["lda_baseline_lda_cosine"]  = np.array(lda_baseline["lda_cosine"])
    md["lda_baseline_accuracies"]  = np.array(lda_baseline["accuracies"])
    np.savez(os.path.join(run_dir, "metrics_raw.npz"), **md)

    hd = {}
    for mc in MODEL_CONFIGS:
        tag        = mc["tag"]
        train_list = [np.array(h["train_losses"]) for h in all_histories[tag]]
        val_list   = [np.array(h["val_losses"])   for h in all_histories[tag]]
        best_list  = [h["best_epoch"]             for h in all_histories[tag]]
        max_len    = max(len(t) for t in train_list)
        def pad(rows, length):
            return np.array([
                np.pad(r, (0, length - len(r)), constant_values=r[-1]) for r in rows
            ])
        hd[f"{tag}_train_losses"] = pad(train_list, max_len)
        hd[f"{tag}_val_losses"]   = pad(val_list,   max_len)
        hd[f"{tag}_best_epochs"]  = np.array(best_list)
    np.savez(os.path.join(run_dir, "histories_raw.npz"), **hd)

    ad = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        ad[f"{tag}_accuracies"] = np.array(all_accuracies[tag])
    np.savez(os.path.join(run_dir, "accuracies_raw.npz"), **ad)
    print(f"[exp2_fisher] Saved metrics_raw.npz, histories_raw.npz, accuracies_raw.npz")


def save_results_summary(all_metrics, all_accuracies, lda_baseline, run_dir, embed_dim):
    n_lda  = lda_baseline["n_lda"]
    key_ks = sorted(set([1, n_lda // 4, n_lda // 2, n_lda]))

    # ── Part 1: compact summary table at key prefix sizes ────────────────────
    lines = [
        "=" * 80,
        "Experiment 2 (Fisher): Fisher Loss — ordered LDA recovery",
        "=" * 80, "",
        "SUMMARY TABLE (key prefix sizes)",
        "-" * 80,
        f"{'Model':<30}  " +
        "  ".join([f"LDAang@{k:2d}  LDAcos@{k:2d}  acc@{k:2d}" for k in key_ks]),
        "-" * 100,
    ]

    def summary_row(label, lda_angs, lda_cos, accs):
        r = f"{label:<30}"
        for k in key_ks:
            idx = k - 1
            la  = f"{lda_angs[idx]:.1f}°" if idx < len(lda_angs) else "  —"
            lc  = f"{lda_cos[idx]:.3f}"   if idx < len(lda_cos)  else "  —"
            ac  = f"{accs[idx]:.3f}"      if idx < len(accs)      else "  —"
            r  += f"  {la:>10}  {lc:>10}  {ac:>8}"
        return r

    for mc in MODEL_CONFIGS:
        tag     = mc["tag"]
        label   = mc["label"]
        n_seeds = len(all_metrics[tag])
        la_mean = np.array([all_metrics[tag][s]["lda_angles"] for s in range(n_seeds)]).mean(0)
        lc_mean = np.array([all_metrics[tag][s]["lda_cosine"] for s in range(n_seeds)]).mean(0)
        ac_mean = np.array(all_accuracies[tag]).mean(0)
        lines.append(summary_row(label, la_mean, lc_mean, ac_mean))

    lines.append(summary_row("LDA baseline",
                              lda_baseline["lda_angles"],
                              lda_baseline["lda_cosine"],
                              lda_baseline["accuracies"]))

    # ── Part 2: full per-dimension tables (all k = 1..n_lda) ─────────────────
    col_w  = 15
    header = (f"  {'k':>3}  {'PrincipalAngle(°)':>{col_w}}  "
              f"{'MaxCosine':>{col_w}}  {'PairedCosine':>{col_w}}")
    sep    = "  " + "-" * 3 + "  " + "-" * col_w + "  " + "-" * col_w + "  " + "-" * col_w

    lines += ["", "=" * 80, "FULL PER-DIMENSION TABLES (k = 1 .. embed_dim)", "=" * 80]

    for mc in MODEL_CONFIGS:
        tag     = mc["tag"]
        label   = mc["label"]
        n_seeds = len(all_metrics[tag])
        la_arr  = np.array([all_metrics[tag][s]["lda_angles"] for s in range(n_seeds)])
        lc_arr  = np.array([all_metrics[tag][s]["lda_cosine"] for s in range(n_seeds)])
        lp_arr  = np.array([all_metrics[tag][s]["lda_paired"] for s in range(n_seeds)])
        la_mean, la_std = la_arr.mean(0), la_arr.std(0)
        lc_mean, lc_std = lc_arr.mean(0), lc_arr.std(0)
        lp_mean, lp_std = lp_arr.mean(0), lp_arr.std(0)
        n_rows = la_mean.shape[0]   # = embed_dim (may be < n_lda)

        lines += ["", f"--- {label} (n_seeds={n_seeds}) ---", header, sep]
        for k in range(1, n_rows + 1):
            idx = k - 1
            la_s = f"{la_mean[idx]:.2f}" + (f"±{la_std[idx]:.2f}" if n_seeds > 1 else "")
            lc_s = f"{lc_mean[idx]:.4f}" + (f"±{lc_std[idx]:.4f}" if n_seeds > 1 else "")
            lp_s = f"{lp_mean[idx]:.4f}" + (f"±{lp_std[idx]:.4f}" if n_seeds > 1 else "")
            lines.append(f"  {k:>3}  {la_s:>{col_w}}  {lc_s:>{col_w}}  {lp_s:>{col_w}}")

    # LDA baseline (angles = 0 by definition, cosine = 1); rows = embed_dim
    lines += ["", f"--- LDA baseline ---", header, sep]
    for k in range(1, embed_dim + 1):
        idx = k - 1
        la_s = f"{lda_baseline['lda_angles'][idx]:.2f}"
        lc_s = f"{lda_baseline['lda_cosine'][idx]:.4f}"
        lines.append(f"  {k:>3}  {la_s:>{col_w}}  {lc_s:>{col_w}}  {'1.0000':>{col_w}}")

    lines += [
        "",
        "Columns:",
        "  PrincipalAngle(°) : mean principal angle between span(B^T[:,1:k]) and top-k LDA subspace",
        "                      0° = perfect ordered subspace recovery",
        "  MaxCosine         : mean max cosine similarity — each encoder col vs nearest LDA dir",
        "  PairedCosine      : |cos(b_k, lda_k)| — encoder dim k vs exactly LDA direction k",
        "                      1.0 = perfect ordered recovery of individual directions",
        f"  k = 1..{n_lda} (= C-1 = number of LDA directions for this dataset)",
        "",
        "Key prediction:",
        "  Fisher (standard)     : angle near 0° ONLY at k=n_lda (span OK, ordering bad)",
        "  Full-prefix Fisher    : angle near 0° at small k too (ordered LDA recovery)",
        "  PrefixL1 Fisher (rev) : intermediate ordering via L1 pressure",
        "  Standard MRL Fisher   : ordering only near evaluated M prefix sizes",
    ]

    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2_fisher] Saved {path}")


def save_experiment_description(cfg, run_dir):
    lines = [
        "=" * 60,
        "Experiment 2 (Fisher): Fisher Loss — Ordered LDA Recovery",
        "=" * 60, "",
        "Hypothesis:",
        "  Fisher (LDA) criterion directly optimises encoder directions to be",
        "  LDA-optimal, but imposes no ordering.  Wrapping it in a full-prefix",
        "  sum forces prefix k to recover the top-k LDA directions, analogous",
        "  to how full-prefix MRL + MSE recovers ordered PCA.",
        "",
        "Models:",
        "  1. Fisher (standard)    — FisherLoss on full d-dim embedding",
        "     Expected: angle → 0° at k=d (span correct), NOT at small k",
        "  2. Full-prefix Fisher   — sum of Fisher over all prefixes k=1..d",
        "     Expected: angle → 0° at small k too (ordered LDA recovery)",
        "  3. PrefixL1 Fisher (rev)— Fisher(full) + front-loaded L1 penalty; dims reversed",
        "     Expected: partial ordering via L1; weaker than prefix Fisher",
        "  4. Standard MRL Fisher  — Fisher at fixed M prefix sizes only",
        "     Expected: ordering at M sizes, gaps between them",
        "",
        "Baseline:",
        "  5. LDA baseline — sklearn LDA directions used directly as encoder",
        "     Angle to LDA = 0° by construction at every k",
        "",
        "Metric: mean principal angle between B^T[:,1:k] and top-k LDA directions",
        "(rotation-invariant; correct measure for subspace recovery)",
        "",
        "Config:",
    ] + [f"  {k}: {v}" for k, v in cfg.items()]

    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2_fisher] Saved {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 2 (Fisher): Fisher loss LDA recovery")
    parser.add_argument("--fast",        action="store_true",
                        help="Smoke test: d=8, 1 seed, 10 epochs")
    parser.add_argument("--sgd",         action="store_true",
                        help="Use full-batch SGD instead of default full-batch Adam")
    parser.add_argument("--models",      type=str, nargs="+", default=None,
                        metavar="TAG",
                        help="Subset of models to train, e.g. --models fp_fisher")
    parser.add_argument("--use-weights", type=str, default=None, metavar="FOLDER",
                        help="Reload .pt files, recompute metrics, regenerate plots")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, metavar="SEED",
                        help="Override seeds to process in --use-weights mode (e.g. --seeds 47)")
    args = parser.parse_args()

    t_start = time.time()

    # Filter MODEL_CONFIGS globally so all plot/eval loops see only active models
    global MODEL_CONFIGS
    if args.models:
        MODEL_CONFIGS = [mc for mc in MODEL_CONFIGS if mc["tag"] in args.models]
        print(f"[exp2_fisher] --models: running only {[mc['tag'] for mc in MODEL_CONFIGS]}")

    embed_dim      = EMBED_DIM
    epochs         = EPOCHS
    patience       = PATIENCE
    seeds          = SEEDS
    l1_lambda      = L1_LAMBDA
    standard_mrl_m = STANDARD_MRL_M

    if args.fast:
        embed_dim      = 8
        epochs         = 10
        patience       = 5
        seeds          = [42]
        standard_mrl_m = [2, 4, 6, 8]
        print(f"[exp2_fisher] --fast mode: d={embed_dim}, 1 seed, {epochs} epochs")

    cfg = dict(
        experiment_name   = "exp2_fisher_loss",
        dataset           = DATASET,
        synthetic_variant = SYNTHETIC_VARIANT,
        embed_dim         = embed_dim,
        epochs            = epochs,
        patience          = patience,
        lr                = LR,
        batch_size        = BATCH_SIZE,
        optimizer         = "sgd" if args.sgd else OPTIMIZER,
        models            = args.models,
        weight_decay      = WEIGHT_DECAY,
        seeds             = seeds,
        fisher_eps        = FISHER_EPS,
        l1_lambda         = l1_lambda,
        standard_mrl_m    = standard_mrl_m,
        fast              = args.fast,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp2_fisher] Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    cfg["device"] = str(device)

    # ------------------------------------------------------------------
    # --use-weights: reload .pt files, recompute metrics, regenerate plots
    # ------------------------------------------------------------------
    if args.use_weights:
        import json
        weights_dir = args.use_weights
        if not os.path.isabs(weights_dir):
            from weight_symmetry.utility import get_path
            weights_dir = os.path.join(get_path("files/results"), weights_dir)
        print(f"[exp2_fisher] --use-weights: loading from {weights_dir}")

        with open(os.path.join(weights_dir, "config.json")) as f:
            saved_cfg = json.load(f)
        embed_dim = saved_cfg["embed_dim"]
        seeds     = args.seeds if args.seeds else saved_cfg["seeds"]

        vparams  = SYNTHETIC_VARIANTS[SYNTHETIC_VARIANT]
        data     = load_data(DATASET, seed=seeds[0], synthetic_variant=SYNTHETIC_VARIANT)
        raw      = load_synthetic(seed=seeds[0], **vparams)
        pca_dirs = raw["pca_dirs"].astype(np.float64)
        lda_dirs = raw["lda_dirs"].astype(np.float64)

        # Clamp embed_dim to n_lda
        n_lda     = lda_dirs.shape[1]
        embed_dim = min(embed_dim, n_lda)

        hist_npz_path = os.path.join(weights_dir, "histories_raw.npz")
        all_histories = {}
        if os.path.exists(hist_npz_path):
            histories_npz = np.load(hist_npz_path)
            for mc in MODEL_CONFIGS:
                tag = mc["tag"]
                tm  = histories_npz[f"{tag}_train_losses"]
                vm  = histories_npz[f"{tag}_val_losses"]
                bm  = histories_npz[f"{tag}_best_epochs"]
                all_histories[tag] = [
                    {"train_losses": tm[s].tolist(), "val_losses": vm[s].tolist(),
                     "best_epoch": int(bm[s])}
                    for s in range(len(seeds))
                ]
        else:
            print("[exp2_fisher] histories_raw.npz not found — parsing .log files")
            for mc in MODEL_CONFIGS:
                tag = mc["tag"]
                all_histories[tag] = []
                for seed in seeds:
                    log_path = os.path.join(weights_dir, f"seed{seed}_{tag}_train.log")
                    if os.path.exists(log_path):
                        all_histories[tag].append(parse_train_log(log_path))
                    else:
                        print(f"  [warn] log not found: {log_path} — using empty history")
                        all_histories[tag].append(
                            {"train_losses": [], "val_losses": [], "best_epoch": 0})

        print("[exp2_fisher] Recomputing metrics from saved weights ...")
        all_metrics        = {mc["tag"]: [] for mc in MODEL_CONFIGS}
        all_accuracies     = {mc["tag"]: [] for mc in MODEL_CONFIGS}
        all_paired_cosines = {mc["tag"]: [] for mc in MODEL_CONFIGS}
        for seed in seeds:
            seed_raw  = load_synthetic(seed=seed, **vparams)
            seed_pca  = seed_raw["pca_dirs"].astype(np.float64)
            seed_lda  = seed_raw["lda_dirs"].astype(np.float64)
            seed_data = load_data(DATASET, seed=seed, synthetic_variant=SYNTHETIC_VARIANT)
            for mc in MODEL_CONFIGS:
                tag  = mc["tag"]
                flip = mc["flip_dims"]
                ckpt = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
                model = LinearAE(data.input_dim, embed_dim)
                model.load_state_dict(
                    torch.load(ckpt, weights_only=True, map_location="cpu"))
                model.eval()
                m = compute_encoder_subspace_metrics(
                    model, seed_pca, seed_lda, flip_dims=flip, model_type="lae_fisher")
                all_metrics[tag].append(m)
                a = compute_prefix_accuracy(model, seed_data, device, "lae", flip_dims=flip)
                all_accuracies[tag].append(a)
                all_paired_cosines[tag].append(
                    compute_paired_cosines_lda(model, seed_lda, flip_dims=flip))

        lda_baseline    = compute_lda_baseline_metrics(lda_dirs, data, embed_dim)
        lda_eigenvalues = compute_lda_eigenvalues(data)

        sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

        save_raw_data(all_metrics, all_histories, all_accuracies, lda_baseline, run_dir)
        save_experiment_description(saved_cfg, run_dir)
        plot_training_curves(all_histories, run_dir, fig_stamp)
        plot_metrics(all_metrics, lda_baseline, all_accuracies, run_dir, fig_stamp, embed_dim)
        plot_ordered_lda_recovery(all_paired_cosines, lda_eigenvalues, run_dir, fig_stamp)
        save_results_summary(all_metrics, all_accuracies, lda_baseline, run_dir, embed_dim)
        elapsed = time.time() - t_start
        save_runtime(run_dir, elapsed)
        save_code_snapshot(run_dir)
        print(f"\n[exp2_fisher] Done. Results in: {run_dir}")
        return

    # ------------------------------------------------------------------
    # Setup run directory
    # ------------------------------------------------------------------
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    save_config(cfg, run_dir)
    save_experiment_description(cfg, run_dir)

    # ------------------------------------------------------------------
    # Load data + ground-truth directions
    # ------------------------------------------------------------------
    vparams = SYNTHETIC_VARIANTS[SYNTHETIC_VARIANT]
    print(f"\n[exp2_fisher] Loading {DATASET} (variant={SYNTHETIC_VARIANT}) ...")
    data     = load_data(DATASET, seed=seeds[0], synthetic_variant=SYNTHETIC_VARIANT)
    raw_data = load_synthetic(seed=seeds[0], **vparams)
    pca_dirs = raw_data["pca_dirs"].astype(np.float64)
    lda_dirs = raw_data["lda_dirs"].astype(np.float64)

    n_lda     = lda_dirs.shape[1]
    embed_dim = min(embed_dim, n_lda)   # safety: never exceed number of LDA directions
    cfg["embed_dim"] = embed_dim
    print(f"[exp2_fisher] input_dim={data.input_dim}  embed_dim={embed_dim}  n_lda={n_lda}")

    np.savez(os.path.join(run_dir, "directions.npz"),
             pca_dirs=pca_dirs, lda_dirs=lda_dirs)

    # ------------------------------------------------------------------
    # Train + evaluate over seeds
    # ------------------------------------------------------------------
    all_histories      = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_metrics        = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_accuracies     = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_paired_cosines = {mc["tag"]: [] for mc in MODEL_CONFIGS}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[exp2_fisher] === Seed {seed} ({seed_idx+1}/{len(seeds)}) ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        seed_data = load_data(DATASET, seed=seed, synthetic_variant=SYNTHETIC_VARIANT)
        seed_raw  = load_synthetic(seed=seed, **vparams)
        seed_pca  = seed_raw["pca_dirs"].astype(np.float64)
        seed_lda  = seed_raw["lda_dirs"].astype(np.float64)

        trained = train_all_models(seed_data, cfg, run_dir, seed, device)

        for mc in MODEL_CONFIGS:
            tag  = mc["tag"]
            flip = mc["flip_dims"]
            model, history = trained[tag]
            all_histories[tag].append(history)

            metrics = compute_encoder_subspace_metrics(
                model, seed_pca, seed_lda, flip_dims=flip, model_type="lae")
            all_metrics[tag].append(metrics)

            accs = compute_prefix_accuracy(model, seed_data, device, "lae", flip_dims=flip)
            all_accuracies[tag].append(accs)
            all_paired_cosines[tag].append(
                compute_paired_cosines_lda(model, seed_lda, flip_dims=flip))

            lda_str = f"{metrics['lda_angles'][embed_dim-1]:.1f}°"
            print(f"  [{tag}] LDA angle@k={embed_dim}: {lda_str}  "
                  f"LDA cosine@k={embed_dim}: {metrics['lda_cosine'][embed_dim-1]:.3f}  "
                  f"acc@k={embed_dim}: {accs[-1]:.3f}")

    # ------------------------------------------------------------------
    # LDA baseline + eigenvalues
    # ------------------------------------------------------------------
    print("\n[exp2_fisher] Computing LDA baseline ...")
    lda_baseline    = compute_lda_baseline_metrics(lda_dirs, data, embed_dim)
    lda_eigenvalues = compute_lda_eigenvalues(data)

    # ------------------------------------------------------------------
    # Save + plot
    # ------------------------------------------------------------------
    save_raw_data(all_metrics, all_histories, all_accuracies, lda_baseline, run_dir)
    print("\n[exp2_fisher] Generating plots ...")
    plot_training_curves(all_histories, run_dir, fig_stamp)
    plot_metrics(all_metrics, lda_baseline, all_accuracies, run_dir, fig_stamp, embed_dim)
    plot_ordered_lda_recovery(all_paired_cosines, lda_eigenvalues, run_dir, fig_stamp)
    save_results_summary(all_metrics, all_accuracies, lda_baseline, run_dir, embed_dim)

    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"\n[exp2_fisher] Done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
