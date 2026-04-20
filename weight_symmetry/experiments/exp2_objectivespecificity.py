"""
Experiment 2: Objective Specificity — Privileged Bases Across Loss Families
-----------------------------------------------------------------------------
Core claim: same architecture, same training, only the loss changes →
privileged directions change.

    FP MRL + MSE    →  encoder aligns with PCA (variance-optimal)
    FP MRL + CE     →  encoder aligns with LDA (classification-optimal)
    FP MRL + Fisher →  encoder aligns with LDA (directly optimises LDA criterion)

Merges exp2_divergence.py (MSE + CE families) and exp2_fisher_loss.py (Fisher family)
into a single script controlled by --loss-family flag.

Models per family:
    MSE    : Unordered LAE, FP MRL+ortho, MRL, PrefixL1 (rev), NonUniform L2
    CE     : Unordered CE, FP MRL (CE), MRL (CE), PrefixL1 (CE, rev)
    Fisher : Unordered Fisher, FP Fisher, MRL Fisher, PrefixL1 Fisher (rev)

Baselines:
    MSE / CE : PCA + linear probe
    Fisher   : LDA projection

Usage:
    Conda environment: mrl_env_cuda12  (GPU)
                       mrl_env         (CPU fallback)

    python weight_symmetry/experiments/exp2_objectivespecificity.py --fast
    python weight_symmetry/experiments/exp2_objectivespecificity.py
    python weight_symmetry/experiments/exp2_objectivespecificity.py --loss-family mse ce
    python weight_symmetry/experiments/exp2_objectivespecificity.py --loss-family fisher
    python weight_symmetry/experiments/exp2_objectivespecificity.py --loss-family fisher --sgd
    python weight_symmetry/experiments/exp2_objectivespecificity.py --use-weights FOLDER
    python weight_symmetry/experiments/exp2_objectivespecificity.py --use-weights FOLDER --loss-family fisher
"""

import os
import sys
import time
import argparse
import json
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
from weight_symmetry.data.loader import load_data, load_data_with_directions
from weight_symmetry.data.synthetic import load_synthetic, SYNTHETIC_VARIANTS
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads
from weight_symmetry.losses.losses import (
    MSELoss, FullPrefixMRLLoss, StandardMRLLoss, NonUniformL2Loss, PrefixL1MSELoss,
    CELoss, FullPrefixCELoss, StandardMRLCELoss, PrefixL1CELoss,
    FisherLoss, FullPrefixFisherLoss, PrefixL1FisherLoss, StandardMRLFisherLoss,
)
from weight_symmetry.training.trainer import train_ae
from weight_symmetry.evaluation.metrics import (
    compute_encoder_subspace_metrics, compute_prefix_accuracy,
    paired_cosine, column_alignment,
)

# ==============================================================================
# CONFIG
# ==============================================================================
EXPERIMENT_NOTE = (
    "Merged exp2_divergence + exp2_fisher_loss into unified script. "
    "All three loss families (MSE, CE, Fisher) controlled by --loss-family flag. "
    "Single metrics_raw.npz per run covers all active families."
)

DATASET           = "synthetic"
SYNTHETIC_VARIANT = "orderedBoth"
SEEDS             = [47]
LOSS_FAMILIES     = ["mse", "ce", "fisher"]   # run all by default

# Per-family training settings — edit here, not in individual flags
FAMILY_CFG = {
    "mse": dict(
        embed_dim      = 50,
        epochs         = 500,
        patience       = 50,
        batch_size     = 256,
        optimizer      = "adam",
        standard_mrl_m = [5, 10, 25, 50],
    ),
    "ce": dict(
        embed_dim      = 50,
        epochs         = 500,
        patience       = 50,
        batch_size     = 256,
        optimizer      = "adam",
        standard_mrl_m = [5, 10, 25, 50],
    ),
    "fisher": dict(
        embed_dim      = 19,       # = C-1; clamped to n_lda at runtime
        epochs         = 10000,
        patience       = 2000,
        batch_size     = None,     # None → full training set
        optimizer      = "adam",
        standard_mrl_m = [5, 10, 15, 19],
    ),
}

LR                = 1e-3
WEIGHT_DECAY      = 1e-4
L1_LAMBDA         = 0.01
NONUNIFORM_L2_LAM = 1.0
FISHER_EPS        = 1e-4
# ==============================================================================

ALL_MODEL_CONFIGS = [
    # ── MSE family ─────────────────────────────────────────────────────────────
    dict(tag="mse_lae",          family="mse",    loss="mse",             model_type="lae",
         supervised=False, orthogonalize=False, label="Unordered",      flip_dims=False),
    dict(tag="fp_mrl_mse_ortho", family="mse",    loss="fullprefix",      model_type="lae",
         supervised=False, orthogonalize=True,  label="FP MRL",         flip_dims=False),
    dict(tag="std_mrl_mse",      family="mse",    loss="standard_mse",    model_type="lae",
         supervised=False, orthogonalize=False, label="MRL",            flip_dims=False),
    dict(tag="prefix_l1_mse",    family="mse",    loss="prefix_l1_mse",   model_type="lae",
         supervised=False, orthogonalize=False, label="PrefixL1 (rev)", flip_dims=True),
    dict(tag="nonuniform_l2",    family="mse",    loss="nonuniform_l2",   model_type="lae",
         supervised=False, orthogonalize=False, label="NonUniform L2",  flip_dims=False),
    # ── CE family ──────────────────────────────────────────────────────────────
    dict(tag="normal_ce",        family="ce",     loss="ce",              model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="Unordered",      flip_dims=False),
    dict(tag="fp_mrl_ce",        family="ce",     loss="fullprefix_ce",   model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="FP MRL",         flip_dims=False),
    dict(tag="std_mrl_ce",       family="ce",     loss="standard_ce",     model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="MRL",            flip_dims=False),
    dict(tag="prefix_l1_ce",     family="ce",     loss="prefix_l1_ce",    model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="PrefixL1 (rev)", flip_dims=True),
    # ── Fisher family ──────────────────────────────────────────────────────────
    dict(tag="fisher",           family="fisher", loss="fisher",           model_type="lae_fisher",
         supervised=True,  orthogonalize=False, label="Unordered",      flip_dims=False),
    dict(tag="fp_fisher",        family="fisher", loss="fp_fisher",        model_type="lae_fisher",
         supervised=True,  orthogonalize=False, label="FP MRL",         flip_dims=False),
    dict(tag="std_mrl_fisher",   family="fisher", loss="std_mrl_fisher",   model_type="lae_fisher",
         supervised=True,  orthogonalize=False, label="MRL",            flip_dims=False),
    dict(tag="prefix_l1_fisher", family="fisher", loss="prefix_l1_fisher", model_type="lae_fisher",
         supervised=True,  orthogonalize=False, label="PrefixL1 (rev)", flip_dims=True),
]

PLOT_STYLES = {
    "Unordered":      dict(color="#888888", linestyle="--", lw=1.0,  marker=""),
    "FP MRL":         dict(color="#009E73", linestyle="-",  lw=1.8,  marker=""),
    "MRL":            dict(color="#E07B00", linestyle="-",  lw=1.8,  marker="o", markersize=3),
    "PrefixL1 (rev)": dict(color="#CC79A7", linestyle="-",  lw=1.8,  marker="^", markersize=3),
    "NonUniform L2":  dict(color="#56B4E9", linestyle="-",  lw=1.8,  marker=""),
    "PCA + probe":    dict(color="black",   linestyle=":",  lw=1.0,  marker=""),
    "LDA baseline":   dict(color="black",   linestyle=":",  lw=1.0,  marker=""),
}


def _style(label):
    s = PLOT_STYLES.get(label, dict(color="gray", linestyle="-", lw=1.0, marker=""))
    return dict(color=s["color"], linestyle=s["linestyle"],
                lw=s.get("lw", 1.0),
                marker=s.get("marker", ""), markersize=s.get("markersize", 4))


# ==============================================================================
# Build loss + model
# ==============================================================================

def build_loss_fn(loss_type, standard_mrl_m, l1_lambda, nonuniform_l2_lam, fisher_eps):
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "fullprefix":
        return FullPrefixMRLLoss()
    elif loss_type == "standard_mse":
        return StandardMRLLoss(prefix_sizes=standard_mrl_m)
    elif loss_type == "nonuniform_l2":
        return NonUniformL2Loss(lam=nonuniform_l2_lam)
    elif loss_type == "prefix_l1_mse":
        return PrefixL1MSELoss(l1_lambda=l1_lambda)
    elif loss_type == "ce":
        return CELoss()
    elif loss_type == "fullprefix_ce":
        return FullPrefixCELoss()
    elif loss_type == "standard_ce":
        return StandardMRLCELoss(prefix_sizes=standard_mrl_m)
    elif loss_type == "prefix_l1_ce":
        return PrefixL1CELoss(l1_lambda=l1_lambda)
    elif loss_type == "fisher":
        return FisherLoss(eps=fisher_eps)
    elif loss_type == "fp_fisher":
        return FullPrefixFisherLoss(eps=fisher_eps)
    elif loss_type == "std_mrl_fisher":
        return StandardMRLFisherLoss(prefix_sizes=standard_mrl_m, eps=fisher_eps)
    elif loss_type == "prefix_l1_fisher":
        return PrefixL1FisherLoss(l1_lambda=l1_lambda, eps=fisher_eps)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_model(mc, input_dim, embed_dim, n_classes, device):
    if mc["model_type"] == "lae_heads":
        return LinearAEWithHeads(input_dim, embed_dim, n_classes).to(device)
    return LinearAE(input_dim, embed_dim).to(device)


# ==============================================================================
# Training
# ==============================================================================

def train_family(family_mcs, data, global_cfg, fam_cfg, run_dir, seed, device):
    """Train all models for one family + one seed. Returns dict tag -> (model, history)."""
    embed_dim      = fam_cfg["embed_dim"]
    standard_mrl_m = fam_cfg["standard_mrl_m"]
    n_train        = len(data.X_train)
    batch_size     = fam_cfg.get("batch_size") or n_train
    use_adam       = fam_cfg.get("optimizer", "adam") == "adam"

    seed_cfg = dict(
        embed_dim  = embed_dim,
        epochs     = fam_cfg["epochs"],
        patience   = fam_cfg["patience"],
        batch_size = batch_size,
        seed       = seed,
        lr         = global_cfg["lr"],
        weight_decay = global_cfg["weight_decay"],
    )

    models = {}
    for mc in family_mcs:
        tag = mc["tag"]
        model = build_model(mc, data.input_dim, embed_dim, data.n_classes, device)
        torch.manual_seed(seed)
        loss_fn = build_loss_fn(
            mc["loss"], standard_mrl_m,
            global_cfg["l1_lambda"], global_cfg["nonuniform_l2_lam"], global_cfg["fisher_eps"],
        )
        if use_adam:
            opt = torch.optim.Adam(
                model.parameters(), lr=global_cfg["lr"],
                weight_decay=global_cfg["weight_decay"],
            )
        else:
            opt = torch.optim.SGD(model.parameters(), lr=global_cfg["lr"])

        history = train_ae(
            model, loss_fn, opt, data, seed_cfg,
            run_dir, f"seed{seed}_{tag}",
            orthogonalize=mc["orthogonalize"], supervised=mc["supervised"],
        )
        models[tag] = (model, history)
    return models


# ==============================================================================
# Baselines
# ==============================================================================

def compute_pca_probe_metrics(pca_dirs, lda_dirs, data, embed_dim):
    from sklearn.linear_model import LogisticRegression
    from weight_symmetry.evaluation.metrics import subspace_angle

    d, n_lda = embed_dim, lda_dirs.shape[1]
    X_tr = data.X_train.numpy()
    X_te = data.X_test.numpy()
    y_tr = data.y_train.numpy()
    y_te = data.y_test.numpy()

    pca_angles, lda_angles = [], []
    pca_cosine, lda_cosine = [], []
    pca_paired, lda_paired = [], []
    accuracies = []

    for k in range(1, d + 1):
        pca_angles.append(0.0)
        pca_cosine.append(1.0)
        pca_paired.append(1.0)
        U_k = pca_dirs[:, :k].astype(np.float64)
        if k <= n_lda:
            L_k = lda_dirs[:, :k].astype(np.float64)
            lda_angles.append(subspace_angle(U_k, L_k))
            lda_cosine.append(column_alignment(U_k, L_k))
            lda_paired.append(paired_cosine(pca_dirs[:, k-1], lda_dirs[:, k-1]))
        else:
            lda_angles.append(float("nan"))
            lda_cosine.append(float("nan"))
            lda_paired.append(float("nan"))
        Z_tr = X_tr @ pca_dirs[:, :k]
        Z_te = X_te @ pca_dirs[:, :k]
        clf  = LogisticRegression(max_iter=300, n_jobs=1)
        clf.fit(Z_tr, y_tr)
        accuracies.append(float(clf.score(Z_te, y_te)))

    return dict(prefix_sizes=list(range(1, d+1)), n_lda=n_lda,
                pca_angles=pca_angles, lda_angles=lda_angles,
                pca_cosine=pca_cosine, lda_cosine=lda_cosine,
                pca_paired=pca_paired, lda_paired=lda_paired,
                accuracies=accuracies)


def compute_lda_baseline_metrics(lda_dirs, data, embed_dim):
    from sklearn.linear_model import LogisticRegression

    d, n_lda = embed_dim, lda_dirs.shape[1]
    X_tr = data.X_train.numpy().astype(np.float64)
    X_te = data.X_test.numpy().astype(np.float64)
    y_tr = data.y_train.numpy()
    y_te = data.y_test.numpy()

    lda_angles, lda_cosine, accuracies = [], [], []
    for k in range(1, d + 1):
        lda_angles.append(0.0)
        L_k = lda_dirs[:, :k]
        lda_cosine.append(column_alignment(L_k, L_k))
        Z_tr = X_tr @ lda_dirs[:, :k]
        Z_te = X_te @ lda_dirs[:, :k]
        clf  = LogisticRegression(max_iter=300, n_jobs=1)
        clf.fit(Z_tr, y_tr)
        accuracies.append(float(clf.score(Z_te, y_te)))

    return dict(prefix_sizes=list(range(1, d+1)), n_lda=n_lda,
                lda_angles=lda_angles, lda_cosine=lda_cosine, accuracies=accuracies)


def compute_lda_eigenvalues(data):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(data.X_train.numpy(), data.y_train.numpy())
    return lda.explained_variance_ratio_.astype(np.float64)


def compute_paired_cosines_lda(model, lda_dirs, flip_dims=False):
    n_lda = lda_dirs.shape[1]
    B_T   = model.get_encoder_matrix().cpu().numpy().T.astype(np.float64)
    if flip_dims:
        B_T = np.ascontiguousarray(B_T[:, ::-1])
    n_dims = min(n_lda, B_T.shape[1])
    return np.array([paired_cosine(B_T[:, k], lda_dirs[:, k]) for k in range(n_dims)])


# ==============================================================================
# Plots
# ==============================================================================

def plot_training_curves(all_histories, active_mcs, run_dir, fig_stamp):
    families = list(dict.fromkeys(mc["family"] for mc in active_mcs))
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4))
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fam_mcs  = [mc for mc in active_mcs if mc["family"] == fam]
        mse_tags = {mc["tag"] for mc in fam_mcs if mc["model_type"] == "lae"}
        ax2      = ax.twinx() if fam == "mse" else None

        for mc in fam_mcs:
            tag, label = mc["tag"], mc["label"]
            if tag not in all_histories or not all_histories[tag]:
                continue
            rows    = [h["train_losses"] for h in all_histories[tag]]
            max_len = max(len(r) for r in rows)
            mat     = np.array([
                np.pad(r, (0, max_len - len(r)), constant_values=r[-1]) for r in rows
            ])
            epochs  = np.arange(1, max_len + 1)
            mean, std = mat.mean(0), mat.std(0)
            s = _style(label)
            target = ax2 if (fam == "mse" and tag in mse_tags and ax2) else ax
            target.plot(epochs, mean, label=label, **s)
            target.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=s["color"])

        ax.set_xlabel("Epoch")
        ax.set_title(f"{fam.upper()} — training loss")
        ax.grid(True, alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        if ax2:
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7)
            ax2.set_ylabel("MSE loss", color="steelblue")
            ax2.tick_params(axis="y", labelcolor="steelblue")
        else:
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2] Saved {path}")


def plot_subspace_metrics(all_metrics, baselines, active_mcs, run_dir, fig_stamp,
                          family_embed_dim):
    """One 2×2 figure per family: (angle / cosine) × (PCA / LDA)."""
    families = list(dict.fromkeys(mc["family"] for mc in active_mcs))

    for fam in families:
        fam_mcs   = [mc for mc in active_mcs if mc["family"] == fam]
        embed_dim = family_embed_dim[fam]
        baseline  = baselines.get(fam, {})
        n_lda     = int(baseline.get("n_lda", embed_dim))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        refs = [("pca", 0), ("lda", 1)]

        for ref, col in refs:
            n_pts = embed_dim if ref == "pca" else n_lda
            xs    = list(range(1, n_pts + 1))
            bkey  = "PCA + probe" if ref == "pca" else "LDA baseline"

            for row, metric in enumerate(["angles", "cosine"]):
                ax  = axes[row, col]
                key = f"{ref}_{metric}"

                for mc in fam_mcs:
                    tag, label = mc["tag"], mc["label"]
                    if tag not in all_metrics or not all_metrics[tag]:
                        continue
                    n_seeds = len(all_metrics[tag])
                    try:
                        if ref == "pca":
                            mat = np.array([all_metrics[tag][s][key]
                                            for s in range(n_seeds)])
                            mat = mat[:, :n_pts]
                        else:
                            mat = np.array([[all_metrics[tag][s][key][k]
                                             for k in range(n_lda)]
                                            for s in range(n_seeds)])
                            mat = np.where(mat < 0, np.nan, mat)
                    except (KeyError, IndexError):
                        continue
                    mean = np.nanmean(mat, axis=0)
                    std  = np.nanstd(mat,  axis=0)
                    s    = _style(label)
                    ax.plot(xs, mean, label=label, **s)
                    ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=s["color"])

                # Baseline line
                if key in baseline:
                    bvals = np.array(baseline[key])
                    bvals = np.where(bvals < 0, np.nan, bvals)
                    bvals = bvals[:n_pts]
                    if not np.all(np.isnan(bvals)):
                        ax.plot(xs, bvals, label=bkey, **_style(bkey))

                ylabel = "Mean principal angle (°)" if metric == "angles" else "Max cosine similarity"
                ax.set_xlabel("Prefix size k")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{fam.upper()} — {'Angle' if metric == 'angles' else 'Cosine'} to {ref.upper()}")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(run_dir, f"subspace_metrics_{fam}{fig_stamp}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[exp2] Saved {path}")


def plot_prefix_accuracy(all_accuracies, baselines, active_mcs, run_dir, fig_stamp,
                         family_embed_dim):
    families = list(dict.fromkeys(mc["family"] for mc in active_mcs))
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4))
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fam_mcs   = [mc for mc in active_mcs if mc["family"] == fam]
        embed_dim = family_embed_dim[fam]
        xs        = list(range(1, embed_dim + 1))
        baseline  = baselines.get(fam, {})
        bkey      = "PCA + probe" if fam in ("mse", "ce") else "LDA baseline"

        for mc in fam_mcs:
            tag, label = mc["tag"], mc["label"]
            if tag not in all_accuracies or not all_accuracies[tag]:
                continue
            mat = np.array(all_accuracies[tag])
            mean, std = mat.mean(0), mat.std(0)
            s = _style(label)
            ax.plot(xs, mean, label=label, **s)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=s["color"])

        if "accuracies" in baseline:
            ax.plot(xs, baseline["accuracies"][:embed_dim], label=bkey, **_style(bkey))

        ax.set_xlabel("Prefix size k")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{fam.upper()} — Prefix accuracy")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(run_dir, f"prefix_accuracy{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2] Saved {path}")


def plot_paired_cosines_pca(all_metrics, pca_probe, active_mcs, run_dir, fig_stamp, embed_dim):
    mse_mcs = [mc for mc in active_mcs if mc["family"] == "mse"]
    if not mse_mcs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(1, embed_dim + 1))
    for mc in mse_mcs:
        tag, label = mc["tag"], mc["label"]
        if tag not in all_metrics or not all_metrics[tag]:
            continue
        n_seeds = len(all_metrics[tag])
        mat     = np.array([all_metrics[tag][s]["pca_paired"] for s in range(n_seeds)])
        mean, std = mat.mean(0), mat.std(0)
        s = _style(label)
        ax.plot(xs, mean, label=label, **s)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=s["color"])
    if "pca_paired" in pca_probe:
        ax.plot(xs, pca_probe["pca_paired"], label="PCA + probe", **_style("PCA + probe"))
    ax.set_xlabel("Dimension k")
    ax.set_ylabel("|cos(b_k, u_k)|")
    ax.set_title("Ordered PCA recovery — paired cosine [MSE models]")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"ordered_pca_recovery_paired_cosine{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2] Saved {path}")


def plot_ordered_lda_recovery(all_paired_cosines, lda_eigenvalues, active_mcs,
                              run_dir, fig_stamp, top_k=5):
    fisher_mcs = [mc for mc in active_mcs if mc["family"] == "fisher"]
    if not fisher_mcs or not all_paired_cosines:
        return

    valid_tags = [mc["tag"] for mc in fisher_mcs
                  if mc["tag"] in all_paired_cosines and all_paired_cosines[mc["tag"]]]
    if not valid_tags:
        return

    n_lda = lda_eigenvalues.shape[0]
    avail = min(np.array(all_paired_cosines[valid_tags[0]]).shape[1], n_lda, top_k)
    top_k = max(1, avail)
    xs    = list(range(1, top_k + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for mc in fisher_mcs:
        tag, label = mc["tag"], mc["label"]
        if tag not in all_paired_cosines or not all_paired_cosines[tag]:
            continue
        mat = np.array(all_paired_cosines[tag])[:, :top_k]
        mean, std = mat.mean(0), mat.std(0)
        s = _style(label)
        ax.plot(xs, mean, label=label, **s)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=s["color"])
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="LDA baseline")
    ax.set_xlabel("Dimension k")
    ax.set_ylabel("|cos(b_k, lda_k)|")
    ax.set_title(f"Paired cosine: top-{top_k} LDA directions [Fisher models]")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels_list = [mc["label"] for mc in fisher_mcs]
    colors      = [_style(l)["color"] for l in labels_list]
    scores = []
    for mc in fisher_mcs:
        tag = mc["tag"]
        if tag not in all_paired_cosines or not all_paired_cosines[tag]:
            scores.append((0.0, 0.0))
            continue
        mat    = np.array(all_paired_cosines[tag])
        n_dims = mat.shape[1]
        eigs   = lda_eigenvalues[:n_dims] / lda_eigenvalues[:n_dims].sum()
        ss     = (mat * eigs[np.newaxis, :]).sum(axis=1)
        scores.append((ss.mean(), ss.std()))
    xs_bar = np.arange(len(fisher_mcs))
    means  = [s[0] for s in scores]
    stds   = [s[1] for s in scores]
    ax.bar(xs_bar, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(xs_bar, means, yerr=stds, fmt="none", color="black", capsize=4)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="Perfect (LDA baseline)")
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(labels_list, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Σ λ_j |cos(b_j, lda_j)|")
    ax.set_title("Eigenvalue-weighted ordered LDA recovery [Fisher models]")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(run_dir, f"ordered_lda_recovery{fig_stamp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[exp2] Saved {path}")


# ==============================================================================
# Save data
# ==============================================================================

def _safe_arr(vals):
    """Convert list to array, replacing NaN with -1."""
    arr = []
    for v in vals:
        try:
            arr.append(-1.0 if np.isnan(float(v)) else float(v))
        except (TypeError, ValueError):
            arr.append(float(v))
    return np.array(arr)


def save_raw_data(all_metrics, all_histories, all_accuracies, baselines, run_dir, active_mcs):
    md = {}
    for mc in active_mcs:
        tag = mc["tag"]
        if tag not in all_metrics or not all_metrics[tag]:
            continue
        md[f"{tag}_pca_angles"]   = np.array([m["pca_angles"] for m in all_metrics[tag]])
        md[f"{tag}_lda_angles"]   = np.array([_safe_arr(m["lda_angles"]) for m in all_metrics[tag]])
        md[f"{tag}_pca_cosine"]   = np.array([m["pca_cosine"] for m in all_metrics[tag]])
        md[f"{tag}_lda_cosine"]   = np.array([_safe_arr(m["lda_cosine"]) for m in all_metrics[tag]])
        md[f"{tag}_pca_paired"]   = np.array([m["pca_paired"] for m in all_metrics[tag]])
        md[f"{tag}_lda_paired"]   = np.array([_safe_arr(m["lda_paired"]) for m in all_metrics[tag]])
        md[f"{tag}_prefix_sizes"] = np.array(all_metrics[tag][0]["prefix_sizes"])
        md[f"{tag}_n_lda"]        = np.array(all_metrics[tag][0]["n_lda"])

    for fam, bl in baselines.items():
        pfx = f"baseline_{fam}"
        for k, v in bl.items():
            if isinstance(v, list):
                md[f"{pfx}_{k}"] = _safe_arr(v)
            elif isinstance(v, int):
                md[f"{pfx}_{k}"] = np.array(v)

    np.savez(os.path.join(run_dir, "metrics_raw.npz"), **md)

    hd = {}
    for mc in active_mcs:
        tag = mc["tag"]
        if tag not in all_histories or not all_histories[tag]:
            continue
        train_list = [np.array(h["train_losses"]) for h in all_histories[tag]]
        val_list   = [np.array(h["val_losses"])   for h in all_histories[tag]]
        best_list  = [h["best_epoch"]             for h in all_histories[tag]]
        max_len    = max(len(t) for t in train_list)
        def pad(rows, length):
            return np.array([np.pad(r, (0, length-len(r)), constant_values=r[-1]) for r in rows])
        hd[f"{tag}_train_losses"] = pad(train_list, max_len)
        hd[f"{tag}_val_losses"]   = pad(val_list, max_len)
        hd[f"{tag}_best_epochs"]  = np.array(best_list)
    np.savez(os.path.join(run_dir, "histories_raw.npz"), **hd)

    ad = {}
    for mc in active_mcs:
        tag = mc["tag"]
        if tag not in all_accuracies or not all_accuracies[tag]:
            continue
        ad[f"{tag}_accuracies"] = np.array(all_accuracies[tag])
    np.savez(os.path.join(run_dir, "accuracies_raw.npz"), **ad)
    print(f"[exp2] Saved metrics_raw.npz, histories_raw.npz, accuracies_raw.npz")


def save_results_summary(all_metrics, all_accuracies, baselines, run_dir,
                         active_mcs, family_embed_dim):
    lines = ["=" * 80,
             "Experiment 2: Objective Specificity — Privileged Bases Across Loss Families",
             "=" * 80, ""]
    families = list(dict.fromkeys(mc["family"] for mc in active_mcs))
    for fam in families:
        fam_mcs   = [mc for mc in active_mcs if mc["family"] == fam]
        embed_dim = family_embed_dim[fam]
        bl        = baselines.get(fam, {})
        n_lda     = int(bl.get("n_lda", embed_dim))
        key_k     = sorted(set([1, n_lda // 2, n_lda, embed_dim // 2, embed_dim]))
        lines += [f"\n{'─'*60}", f"  {fam.upper()} FAMILY  (embed_dim={embed_dim}, n_lda={n_lda})",
                  f"{'─'*60}"]
        header = f"  {'Model':<30}" + "".join(f"  LDAcos@{k:2d}" for k in key_k) \
                                    + "".join(f"   acc@{k:2d}" for k in key_k)
        lines.append(header)
        for mc in fam_mcs:
            tag, label = mc["tag"], mc["label"]
            if tag not in all_metrics or not all_metrics[tag]:
                continue
            n_seeds  = len(all_metrics[tag])
            lc_arr   = np.array([_safe_arr(all_metrics[tag][s]["lda_cosine"])
                                  for s in range(n_seeds)])
            lc_mean  = np.where(lc_arr < 0, np.nan, lc_arr).mean(0)
            ac_mean  = np.array(all_accuracies.get(tag, [])).mean(0) \
                       if tag in all_accuracies and all_accuracies[tag] else []
            row = f"  {label:<30}"
            for k in key_k:
                idx = k - 1
                lc  = f"{lc_mean[idx]:.3f}" if idx < len(lc_mean) and not np.isnan(lc_mean[idx]) else "   —"
                row += f"  {lc:>10}"
            for k in key_k:
                idx = k - 1
                ac  = f"{ac_mean[idx]:.3f}" if idx < len(ac_mean) else "   —"
                row += f"  {ac:>8}"
            lines.append(row)
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2] Saved {path}")


def save_experiment_description(cfg, run_dir, active_families, fam_cfg, dataset_info):
    lines = [
        "=" * 60,
        "Experiment 2: Objective Specificity — Privileged Bases",
        "=" * 60, "",
        "Core claim:",
        "  FP MRL + MSE  → PCA alignment",
        "  FP MRL + CE   → LDA alignment",
        "  FP Fisher     → LDA alignment (direct Fisher criterion)",
        "",
        f"Active families: {active_families}", "",
    ]
    for fam in active_families:
        fc = fam_cfg[fam]
        lines.append(
            f"  {fam.upper()}: embed_dim={fc['embed_dim']}  epochs={fc['epochs']}  "
            f"patience={fc['patience']}  batch={fc.get('batch_size')}  "
            f"opt={fc.get('optimizer')}  M={fc['standard_mrl_m']}"
        )
    lines += ["", "Dataset info:"] + [f"  {k}: {v}" for k, v in dataset_info.items()]
    lines += ["", "Global config:"] + [f"  {k}: {v}" for k, v in cfg.items()]
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2] Saved {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 2: Objective Specificity")
    parser.add_argument("--fast",        action="store_true",
                        help="Smoke test: 1 seed, 5 epochs, small embed_dim")
    parser.add_argument("--loss-family", nargs="+", default=None,
                        choices=["mse", "ce", "fisher"], metavar="FAMILY",
                        help="Which families to run (default: all)")
    parser.add_argument("--use-weights", type=str, default=None, metavar="FOLDER",
                        help="Reload .pt files, recompute metrics, regenerate plots")
    parser.add_argument("--sgd",         action="store_true",
                        help="Use SGD instead of Adam for Fisher family")
    args = parser.parse_args()

    t_start = time.time()

    active_families = args.loss_family or LOSS_FAMILIES
    active_mcs      = [mc for mc in ALL_MODEL_CONFIGS if mc["family"] in active_families]
    fam_cfg         = {fam: dict(FAMILY_CFG[fam]) for fam in active_families}
    if args.sgd and "fisher" in fam_cfg:
        fam_cfg["fisher"]["optimizer"] = "sgd"

    if args.fast:
        seeds = [42]
        for fam in active_families:
            fam_cfg[fam]["epochs"]    = 5
            fam_cfg[fam]["patience"]  = 3
            fam_cfg[fam]["embed_dim"] = 8 if fam == "fisher" else 16
            fam_cfg[fam]["standard_mrl_m"] = [
                m for m in fam_cfg[fam]["standard_mrl_m"]
                if m <= fam_cfg[fam]["embed_dim"]
            ]
        print(f"[exp2] --fast: families={active_families}, 1 seed, 5 epochs")
    else:
        seeds = SEEDS

    global_cfg = dict(
        experiment_name   = "exp2_objectivespecificity",
        dataset           = DATASET,
        synthetic_variant = SYNTHETIC_VARIANT,
        seeds             = seeds,
        active_families   = active_families,
        lr                = LR,
        weight_decay      = WEIGHT_DECAY,
        l1_lambda         = L1_LAMBDA,
        nonuniform_l2_lam = NONUNIFORM_L2_LAM,
        fisher_eps        = FISHER_EPS,
        fast              = args.fast,
        experiment_note   = EXPERIMENT_NOTE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp2] Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    global_cfg["device"] = str(device)

    # ------------------------------------------------------------------
    # --use-weights path
    # ------------------------------------------------------------------
    if args.use_weights:
        weights_dir = args.use_weights
        if not os.path.isabs(weights_dir):
            from weight_symmetry.utility import get_path
            weights_dir = os.path.join(get_path("files/results"), weights_dir)
        print(f"[exp2] --use-weights: loading from {weights_dir}")

        with open(os.path.join(weights_dir, "config.json")) as f:
            saved_cfg = json.load(f)
        seeds           = saved_cfg.get("seeds", seeds)
        active_families = args.loss_family or saved_cfg.get("active_families", active_families)
        active_mcs      = [mc for mc in ALL_MODEL_CONFIGS if mc["family"] in active_families]
        fam_cfg         = {fam: dict(FAMILY_CFG[fam]) for fam in active_families}

        vparams  = SYNTHETIC_VARIANTS[SYNTHETIC_VARIANT]
        data     = load_data(DATASET, seed=seeds[0], synthetic_variant=SYNTHETIC_VARIANT)
        raw      = load_synthetic(seed=seeds[0], **vparams)
        pca_dirs = raw["pca_dirs"].astype(np.float64)
        lda_dirs = raw["lda_dirs"].astype(np.float64)
        n_lda    = lda_dirs.shape[1]
        if "fisher" in fam_cfg:
            fam_cfg["fisher"]["embed_dim"] = min(fam_cfg["fisher"]["embed_dim"], n_lda)

        family_embed_dim = {fam: fam_cfg[fam]["embed_dim"] for fam in active_families}

        hist_path = os.path.join(weights_dir, "histories_raw.npz")
        all_histories = {mc["tag"]: [] for mc in active_mcs}
        if os.path.exists(hist_path):
            hnpz = np.load(hist_path)
            for mc in active_mcs:
                tag = mc["tag"]
                if f"{tag}_train_losses" not in hnpz:
                    continue
                tm, vm, bm = hnpz[f"{tag}_train_losses"], hnpz[f"{tag}_val_losses"], hnpz[f"{tag}_best_epochs"]
                all_histories[tag] = [
                    {"train_losses": tm[s].tolist(), "val_losses": vm[s].tolist(),
                     "best_epoch": int(bm[s])}
                    for s in range(len(seeds))
                ]

        all_metrics        = {mc["tag"]: [] for mc in active_mcs}
        all_accuracies     = {mc["tag"]: [] for mc in active_mcs}
        all_paired_cosines = {mc["tag"]: [] for mc in active_mcs if mc["family"] == "fisher"}

        print("[exp2] Recomputing metrics from saved weights ...")
        for seed in seeds:
            seed_raw  = load_synthetic(seed=seed, **vparams)
            seed_pca  = seed_raw["pca_dirs"].astype(np.float64)
            seed_lda  = seed_raw["lda_dirs"].astype(np.float64)
            seed_data = load_data(DATASET, seed=seed, synthetic_variant=SYNTHETIC_VARIANT)
            for mc in active_mcs:
                tag       = mc["tag"]
                fam       = mc["family"]
                flip      = mc["flip_dims"]
                embed_dim = fam_cfg[fam]["embed_dim"]
                ckpt      = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
                if not os.path.exists(ckpt):
                    print(f"  [warn] missing checkpoint: {ckpt}")
                    continue
                model = build_model(mc, data.input_dim, embed_dim, data.n_classes, device)
                model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
                model.eval()
                m = compute_encoder_subspace_metrics(
                    model, seed_pca, seed_lda, flip_dims=flip, model_type=mc["model_type"])
                all_metrics[tag].append(m)
                a = compute_prefix_accuracy(
                    model, seed_data, device, mc["model_type"], flip_dims=flip)
                all_accuracies[tag].append(a)
                if fam == "fisher":
                    all_paired_cosines[tag].append(
                        compute_paired_cosines_lda(model, seed_lda, flip_dims=flip))

        baselines = _compute_baselines(active_families, fam_cfg, pca_dirs, lda_dirs, data)
        lda_eigenvalues = compute_lda_eigenvalues(data) if "fisher" in active_families else None

        sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

        _run_plots_and_save(all_metrics, all_histories, all_accuracies, all_paired_cosines,
                            baselines, lda_eigenvalues, active_mcs, active_families,
                            fam_cfg, family_embed_dim, run_dir, fig_stamp, saved_cfg,
                            dataset_info={})
        elapsed = time.time() - t_start
        save_runtime(run_dir, elapsed)
        save_code_snapshot(run_dir)
        print(f"\n[exp2] Done. Results in: {run_dir}")
        return

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    save_config(global_cfg, run_dir)

    vparams  = SYNTHETIC_VARIANTS[SYNTHETIC_VARIANT]
    print(f"\n[exp2] Loading {DATASET} (variant={SYNTHETIC_VARIANT}) ...")
    data     = load_data(DATASET, seed=seeds[0], synthetic_variant=SYNTHETIC_VARIANT)
    raw_data = load_synthetic(seed=seeds[0], **vparams)
    pca_dirs = raw_data["pca_dirs"].astype(np.float64)
    lda_dirs = raw_data["lda_dirs"].astype(np.float64)
    dataset_info = dict(raw_data["params"])

    n_lda = lda_dirs.shape[1]
    if "fisher" in fam_cfg:
        fam_cfg["fisher"]["embed_dim"] = min(fam_cfg["fisher"]["embed_dim"], n_lda)
    family_embed_dim = {fam: fam_cfg[fam]["embed_dim"] for fam in active_families}

    np.savez(os.path.join(run_dir, "directions.npz"), pca_dirs=pca_dirs, lda_dirs=lda_dirs)
    save_experiment_description(global_cfg, run_dir, active_families, fam_cfg, dataset_info)

    all_histories      = {mc["tag"]: [] for mc in active_mcs}
    all_metrics        = {mc["tag"]: [] for mc in active_mcs}
    all_accuracies     = {mc["tag"]: [] for mc in active_mcs}
    all_paired_cosines = {mc["tag"]: [] for mc in active_mcs if mc["family"] == "fisher"}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[exp2] === Seed {seed} ({seed_idx+1}/{len(seeds)}) ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        seed_data = load_data(DATASET, seed=seed, synthetic_variant=SYNTHETIC_VARIANT)
        seed_raw  = load_synthetic(seed=seed, **vparams)
        seed_pca  = seed_raw["pca_dirs"].astype(np.float64)
        seed_lda  = seed_raw["lda_dirs"].astype(np.float64)

        for fam in active_families:
            fam_mcs = [mc for mc in active_mcs if mc["family"] == fam]
            print(f"  [exp2] Training {fam.upper()} family ({len(fam_mcs)} models) ...")
            trained = train_family(fam_mcs, seed_data, global_cfg, fam_cfg[fam],
                                   run_dir, seed, device)
            for mc in fam_mcs:
                tag  = mc["tag"]
                flip = mc["flip_dims"]
                model, history = trained[tag]
                all_histories[tag].append(history)
                m = compute_encoder_subspace_metrics(
                    model, seed_pca, seed_lda, flip_dims=flip, model_type=mc["model_type"])
                all_metrics[tag].append(m)
                a = compute_prefix_accuracy(
                    model, seed_data, device, mc["model_type"], flip_dims=flip)
                all_accuracies[tag].append(a)
                if fam == "fisher":
                    all_paired_cosines[tag].append(
                        compute_paired_cosines_lda(model, seed_lda, flip_dims=flip))
                ed = fam_cfg[fam]["embed_dim"]
                n_av = min(m["n_lda"], len(m["lda_cosine"]))
                print(f"    [{tag}] PCA cos@{ed}: {m['pca_cosine'][-1]:.3f}  "
                      f"LDA cos@{n_av}: {m['lda_cosine'][n_av-1]:.3f}  "
                      f"acc@{ed}: {a[-1]:.3f}")

    baselines       = _compute_baselines(active_families, fam_cfg, pca_dirs, lda_dirs, data)
    lda_eigenvalues = compute_lda_eigenvalues(data) if "fisher" in active_families else None

    _run_plots_and_save(all_metrics, all_histories, all_accuracies, all_paired_cosines,
                        baselines, lda_eigenvalues, active_mcs, active_families,
                        fam_cfg, family_embed_dim, run_dir, fig_stamp, global_cfg,
                        dataset_info)
    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"\n[exp2] Done. Results in: {run_dir}")


def _compute_baselines(active_families, fam_cfg, pca_dirs, lda_dirs, data):
    baselines = {}
    if "mse" in active_families or "ce" in active_families:
        ed = fam_cfg.get("mse", fam_cfg.get("ce"))["embed_dim"]
        bl = compute_pca_probe_metrics(pca_dirs, lda_dirs, data, ed)
        for fam in ("mse", "ce"):
            if fam in active_families:
                baselines[fam] = bl
    if "fisher" in active_families:
        baselines["fisher"] = compute_lda_baseline_metrics(
            lda_dirs, data, fam_cfg["fisher"]["embed_dim"])
    return baselines


def _run_plots_and_save(all_metrics, all_histories, all_accuracies, all_paired_cosines,
                        baselines, lda_eigenvalues, active_mcs, active_families,
                        fam_cfg, family_embed_dim, run_dir, fig_stamp, cfg, dataset_info):
    save_raw_data(all_metrics, all_histories, all_accuracies, baselines, run_dir, active_mcs)
    print("\n[exp2] Generating plots ...")
    plot_training_curves(all_histories, active_mcs, run_dir, fig_stamp)
    plot_subspace_metrics(all_metrics, baselines, active_mcs, run_dir, fig_stamp, family_embed_dim)
    plot_prefix_accuracy(all_accuracies, baselines, active_mcs, run_dir, fig_stamp, family_embed_dim)
    if "mse" in active_families and "mse" in baselines:
        plot_paired_cosines_pca(all_metrics, baselines["mse"], active_mcs, run_dir, fig_stamp,
                                fam_cfg["mse"]["embed_dim"])
    if "fisher" in active_families and lda_eigenvalues is not None:
        plot_ordered_lda_recovery(all_paired_cosines, lda_eigenvalues, active_mcs,
                                  run_dir, fig_stamp)
    save_results_summary(all_metrics, all_accuracies, baselines, run_dir, active_mcs, family_embed_dim)
    if dataset_info:
        save_experiment_description(cfg, run_dir, active_families, fam_cfg, dataset_info)


if __name__ == "__main__":
    main()
