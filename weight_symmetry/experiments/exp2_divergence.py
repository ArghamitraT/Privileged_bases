"""
Experiment 2: Divergence — Objective-Specific Privileged Bases
---------------------------------------------------------------
Validates the core claim: full-prefix MRL privilege is determined by the
objective, not the method.

Same architecture, same training procedure, only the loss changes:
    Full-prefix MRL + MSE  →  encoder aligns with PCA subspace (variance-optimal)
    Full-prefix MRL + CE   →  encoder aligns with LDA subspace (classification-optimal)

Data: ordered LDA synthetic dataset (data/synthetic_data/orderedLDA/).
    Signal dims are ordered by discriminative power (dim 0 most discriminative).
    This lets us measure whether MRL + CE recovers the LDA *ordering*, not just
    the subspace — analogous to exp1 testing PCA ordering recovery.

Models trained:
    1. MSE LAE               — no ordering baseline (encoder directions arbitrary)
    2. Full-prefix MRL (MSE) — reconstruction task → expect PCA alignment
    3. Full-prefix MRL (CE)  — classification task → expect LDA alignment
    4. Standard MRL (CE)     — CE with M={...} prefixes → LDA but weaker

Baselines (no neural training):
    5. PCA + linear probe    — uses PCA subspace; angle to PCA = 0 by definition

Metrics (per prefix k, averaged over seeds):
    - Mean principal angle to PCA subspace: span(B^T[:,1:k]) vs top-k PCA eigenvectors
    - Mean principal angle to LDA subspace: span(B^T[:,1:k]) vs top-k LDA directions
      (only for k ≤ n_lda = C-1; NaN beyond)
    - Prefix test accuracy for each k

Outputs (in run_dir):
    - training_curves_{stamp}.png
    - subspace_angles_{stamp}.png
    - prefix_accuracy_{stamp}.png
    - results_summary.txt
    - experiment_description.log
    - metrics_raw.npz
    - histories_raw.npz
    - runtime.txt
    - code_snapshot/
    - config.json

Usage:
    Conda environment: mrl_env_cuda12  (GPU)
                       mrl_env         (CPU fallback)

    python weight_symmetry/experiments/exp2_divergence.py --fast
    python weight_symmetry/experiments/exp2_divergence.py
    python weight_symmetry/experiments/exp2_divergence.py --use-weights FOLDER
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
from weight_symmetry.data.loader import load_data, load_data_with_directions
from weight_symmetry.data.synthetic import load_synthetic
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads
from weight_symmetry.losses.losses import (
    MSELoss, FullPrefixMRLLoss, StandardMRLLoss, NonUniformL2Loss, PrefixL1MSELoss,
    CELoss, FullPrefixCELoss, StandardMRLCELoss, PrefixL1CELoss,
)
from weight_symmetry.training.trainer import train_ae
from weight_symmetry.evaluation.metrics import (
    compute_encoder_subspace_metrics, compute_prefix_accuracy, paired_cosine
)

# ==============================================================================
# CONFIG
# ==============================================================================
DATASET           = "synthetic"       # "synthetic"           → see SYNTHETIC_VARIANT below
                                     # "20newsgroups"        → TF-IDF + TruncatedSVD, p=P_SVD
                                     # "mnist_noise"         → MNIST + noise dims, p=P_PCA_PROJ+N_NOISE_DIMS
                                     # "fashion_mnist_noise" → same on Fashion-MNIST
SYNTHETIC_VARIANT = "orderedBoth"    # only used when DATASET == "synthetic":
                                     #   "nonOrderedLDA" → flat noise, random signal
                                     #   "orderedLDA"    → flat noise, ordered signal
                                     #   "orderedBoth"   → decaying noise, ordered signal (distinct eigenvalues)
EMBED_DIM      = 50
EPOCHS         = 500
PATIENCE       = 50
LR             = 1e-3
BATCH_SIZE     = 256
WEIGHT_DECAY   = 1e-4
SEEDS          = [47]
STANDARD_MRL_M = [5, 10, 25, 50]   # for d=50
L1_LAMBDA          = 0.01           # PrefixL1 regularisation strength (MSE + CE)
NONUNIFORM_L2_LAM  = 1.0            # NonUniformL2Loss λ (Kunin et al. 2019)

# Real-data parameters (used when DATASET != "synthetic")
P_SVD              = 100    # 20newsgroups: TruncatedSVD output dim = model input_dim
TFIDF_MAX_FEATURES = 10000  # 20newsgroups: TF-IDF vocabulary size
P_PCA_PROJ         = 50     # mnist_noise / fashion_mnist_noise: PCA projection dim
N_NOISE_DIMS       = 25     # mnist_noise / fashion_mnist_noise: prepended noise dims
SIGMA_NOISE        = 5.0    # noise std (high-variance, class-agnostic)

# EXPERIMENT NOTE — fill in manually to describe this run's motivation/changes
# Printed at the top of experiment_description.log
# ------------------------------------------------------------------------------
EXPERIMENT_NOTE = ("LDA comparison for CE models now uses row space of (W_k B_{1:k}) "
                   "— the effective linear classifier in input space — instead of "
                   "encoder rows B^T. This is the correct object to compare to LDA: "
                   "the classifier directions, not the encoder directions.")
# ==============================================================================

# MSE group: reconstruction objective, plain LinearAE
# CE group:  classification objective, LinearAEWithHeads
# orthogonalize=True  → call model.orthogonalize() after each optimizer step
MODEL_CONFIGS = [
    # --- MSE group ---
    dict(tag="mse_lae",          loss="mse",           model_type="lae",
         supervised=False, orthogonalize=False, label="LAE",                    flip_dims=False),
    dict(tag="fp_mrl_mse_ortho", loss="fullprefix",    model_type="lae",
         supervised=False, orthogonalize=True,  label="Full prefix MRL+ortho",  flip_dims=False),
    dict(tag="prefix_l1_mse",    loss="prefix_l1_mse", model_type="lae",
         supervised=False, orthogonalize=False, label="PrefixL1 MSE (rev)",     flip_dims=True),
    dict(tag="std_mrl_mse",      loss="standard_mse",  model_type="lae",
         supervised=False, orthogonalize=False, label="Standard MRL (MSE)",     flip_dims=False),
    dict(tag="nonuniform_l2",    loss="nonuniform_l2", model_type="lae",
         supervised=False, orthogonalize=False, label="NonUniform L2",          flip_dims=False),
    # --- CE group ---
    dict(tag="normal_ce",        loss="ce",            model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="Normal CE",              flip_dims=False),
    dict(tag="fp_mrl_ce",        loss="fullprefix_ce", model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="Full prefix MRL (CE)",   flip_dims=False),
    dict(tag="prefix_l1_ce",     loss="prefix_l1_ce",  model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="PrefixL1 (CE) (rev)",    flip_dims=True),
    dict(tag="std_mrl_ce",       loss="standard_ce",   model_type="lae_heads",
         supervised=True,  orthogonalize=False, label="Standard MRL (CE)",      flip_dims=False),
]

MSE_TAGS = {mc["tag"] for mc in MODEL_CONFIGS if mc["model_type"] == "lae"}
CE_TAGS  = {mc["tag"] for mc in MODEL_CONFIGS if mc["model_type"] == "lae_heads"}

# MSE-scale models go on secondary y-axis in training curves
_MSE_TAGS = MSE_TAGS

PLOT_STYLES = {
    # MSE group
    "LAE":                   dict(color="gray",       linestyle="--", marker=""),
    "Full prefix MRL+ortho": dict(color="steelblue",  linestyle="-",  marker=""),
    "PrefixL1 MSE (rev)":    dict(color="peru",       linestyle="-",  marker="^", markersize=3),
    "Standard MRL (MSE)":    dict(color="royalblue",  linestyle="-",  marker="o", markersize=3),
    "NonUniform L2":         dict(color="mediumpurple",linestyle="-", marker=""),
    # CE group
    "Normal CE":             dict(color="gray",       linestyle="--", marker=""),
    "Full prefix MRL (CE)":  dict(color="green",      linestyle="-",  marker=""),
    "PrefixL1 (CE) (rev)":   dict(color="crimson",    linestyle="-",  marker="^", markersize=3),
    "Standard MRL (CE)":     dict(color="orange",     linestyle="-",  marker="o", markersize=3),
    # Baseline
    "PCA + probe":           dict(color="black",      linestyle=":",  marker=""),
}


# ==============================================================================
# Build loss
# ==============================================================================

def build_loss_fn(loss_type: str, embed_dim: int, standard_mrl_m: list,
                  n_classes: int, l1_lambda: float = 0.01, nonuniform_l2_lam: float = 1.0):
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
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==============================================================================
# Training
# ==============================================================================

def train_all_models(data, cfg, run_dir, seed, standard_mrl_m, device):
    """Train all neural models for one seed. Returns dict tag -> (model, history)."""
    models = {}
    for mc in MODEL_CONFIGS:
        tag        = mc["tag"]
        supervised = mc["supervised"]
        ortho      = mc["orthogonalize"]
        n_classes  = data.n_classes

        if mc["model_type"] == "lae_heads":
            model = LinearAEWithHeads(
                data.input_dim, cfg["embed_dim"], n_classes
            ).to(device)
        else:
            model = LinearAE(data.input_dim, cfg["embed_dim"]).to(device)

        torch.manual_seed(seed)
        loss_fn = build_loss_fn(
            mc["loss"], cfg["embed_dim"], standard_mrl_m, n_classes,
            l1_lambda=cfg.get("l1_lambda", L1_LAMBDA),
            nonuniform_l2_lam=cfg.get("nonuniform_l2_lam", NONUNIFORM_L2_LAM),
        )
        opt = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        seed_cfg         = dict(cfg)
        seed_cfg["seed"] = seed

        history = train_ae(
            model, loss_fn, opt, data, seed_cfg,
            run_dir, f"seed{seed}_{tag}",
            orthogonalize=ortho, supervised=supervised,
        )
        models[tag] = (model, history)
    return models


# ==============================================================================
# PCA + probe baseline
# ==============================================================================

def compute_pca_probe_metrics(pca_dirs, lda_dirs, data, embed_dim):
    """
    PCA + linear probe baseline.
    Encoder = PCA projection matrix → angle to PCA = 0 by definition.
    Angle to LDA = inherent PCA vs LDA divergence in the data.
    Prefix accuracy = logistic regression on frozen PCA embeddings.
    """
    from sklearn.linear_model import LogisticRegression

    d     = embed_dim
    n_lda = lda_dirs.shape[1]

    pca_angles   = []
    lda_angles   = []
    pca_cosine   = []
    lda_cosine   = []
    pca_paired   = []
    lda_paired   = []
    accuracies   = []

    X_tr = data.X_train.numpy()
    X_te = data.X_test.numpy()
    y_tr = data.y_train.numpy()
    y_te = data.y_test.numpy()

    from weight_symmetry.evaluation.metrics import subspace_angle, column_alignment

    for k in range(1, d + 1):
        # PCA angle/cosine = 0 / 1 by definition (this IS the PCA subspace)
        pca_angles.append(0.0)
        pca_cosine.append(1.0)
        pca_paired.append(1.0)   # pca_k · pca_k = 1 by definition

        U_k = pca_dirs[:, :k].astype(np.float64)
        if k <= n_lda:
            L_k = lda_dirs[:, :k].astype(np.float64)
            lda_angles.append(subspace_angle(U_k, L_k))
            lda_cosine.append(column_alignment(U_k, L_k))
            lda_paired.append(paired_cosine(pca_dirs[:, k - 1], lda_dirs[:, k - 1]))
        else:
            lda_angles.append(float("nan"))
            lda_cosine.append(float("nan"))
            lda_paired.append(float("nan"))

        Z_tr = X_tr @ pca_dirs[:, :k]
        Z_te = X_te @ pca_dirs[:, :k]
        clf  = LogisticRegression(max_iter=300, n_jobs=1)
        clf.fit(Z_tr, y_tr)
        accuracies.append(float(clf.score(Z_te, y_te)))

    return {
        "prefix_sizes": list(range(1, d + 1)),
        "pca_angles":   pca_angles,
        "lda_angles":   lda_angles,
        "pca_cosine":   pca_cosine,
        "lda_cosine":   lda_cosine,
        "pca_paired":   pca_paired,
        "lda_paired":   lda_paired,
        "n_lda":        n_lda,
        "accuracies":   accuracies,
    }


# ==============================================================================
# Plots
# ==============================================================================

def plot_training_curves(all_histories, run_dir, fig_stamp):
    """
    Plot train/val loss curves.  MSE-based models (tiny loss scale) are drawn on
    a secondary right y-axis so they remain visible alongside CE models (scale ~1–3).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, loss_key, title in [
        (axes[0], "train_losses", "Train Loss"),
        (axes[1], "val_losses",   "Val Loss"),
    ]:
        ax2 = ax.twinx()   # secondary axis for MSE-scale models

        for mc in MODEL_CONFIGS:
            tag   = mc["tag"]
            label = mc["label"]
            style = PLOT_STYLES[label]

            rows    = [h[loss_key] for h in all_histories[tag]]
            max_len = max(len(r) for r in rows)

            mat    = np.array([
                np.pad(r, (0, max_len - len(r)), constant_values=r[-1])
                for r in rows
            ])
            epochs = np.arange(1, max_len + 1)
            mean   = mat.mean(0)
            std    = mat.std(0)

            target = ax2 if tag in _MSE_TAGS else ax
            target.plot(epochs, mean, label=label, **style)
            target.fill_between(epochs, mean - std, mean + std,
                                alpha=0.15, color=style["color"])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("CE / Combined Loss", color="black")
        ax2.set_ylabel("MSE Loss", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Merge legends from both axes
        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8)

    plt.tight_layout()
    path = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2] Saved {path}")


def _plot_metric_panel(ax, tag_set, all_metrics, pca_probe_metrics,
                       ref, metric, embed_dim, n_lda, show_probe):
    """
    Draw one panel for a subset of models.
    ref    : 'pca' | 'lda'
    metric : 'angle' | 'cosine'
    """
    n_lda  = min(n_lda, embed_dim)
    xs     = list(range(1, (embed_dim if ref == "pca" else n_lda) + 1))
    key    = f"{ref}_{'angles' if metric == 'angle' else 'cosine'}"
    p_key  = f"{ref}_{'angles' if metric == 'angle' else 'cosine'}"

    for mc in MODEL_CONFIGS:
        if mc["tag"] not in tag_set:
            continue
        tag, label = mc["tag"], mc["label"]
        style   = PLOT_STYLES[label]
        n_seeds = len(all_metrics[tag])
        if ref == "pca":
            mat = np.array([all_metrics[tag][s][key] for s in range(n_seeds)])
        else:
            mat = np.array([[all_metrics[tag][s][key][k] for k in range(n_lda)]
                            for s in range(n_seeds)])
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(xs, mean, label=label, **style)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=style["color"])

    if show_probe:
        probe_style = PLOT_STYLES["PCA + probe"]
        vals = pca_probe_metrics[p_key] if ref == "pca" \
               else pca_probe_metrics[p_key][:n_lda]
        ax.plot(xs, vals, label="PCA + probe", **probe_style)

    ref_label = "PCA" if ref == "pca" else f"LDA (k≤{n_lda})"
    ax.set_xlabel("Prefix size k")
    if metric == "angle":
        ax.set_ylabel("Mean principal angle (°)")
        ax.set_title(f"Angle to {ref_label}")
    else:
        ax.set_ylabel("Mean max cosine similarity")
        ax.set_title(f"Cosine similarity to {ref_label}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_subspace_angles(all_metrics, pca_probe_metrics, run_dir, fig_stamp,
                         embed_dim):
    """4×2 grid: (MSE / CE) × (PCA / LDA) × (angle / cosine)."""
    n_lda = pca_probe_metrics["n_lda"]

    _, axes = plt.subplots(4, 2, figsize=(14, 20))

    for row, (tag_set, group) in enumerate([(MSE_TAGS, "MSE"), (CE_TAGS, "CE")]):
        for col, (ref, ref_label) in enumerate([("pca", "PCA"), ("lda", "LDA")]):
            base_row = row * 2
            _plot_metric_panel(axes[base_row, col], tag_set, all_metrics,
                               pca_probe_metrics, ref, "angle",
                               embed_dim, n_lda, show_probe=True)
            axes[base_row, col].set_title(f"{group} models — Angle to {ref_label}")

            _plot_metric_panel(axes[base_row + 1, col], tag_set, all_metrics,
                               pca_probe_metrics, ref, "cosine",
                               embed_dim, n_lda, show_probe=True)
            axes[base_row + 1, col].set_title(f"{group} models — Cosine sim to {ref_label}")

    plt.tight_layout()
    path = os.path.join(run_dir, f"subspace_angles{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2] Saved {path}")


def plot_prefix_accuracy(all_accuracies, pca_probe_metrics, run_dir, fig_stamp,
                         embed_dim):
    fig, ax = plt.subplots(figsize=(8, 5))

    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        style = PLOT_STYLES[label]
        n_seeds = len(all_accuracies[tag])
        acc_mat = np.array(all_accuracies[tag])   # (n_seeds, d)
        mean, std = acc_mat.mean(0), acc_mat.std(0)
        xs = list(range(1, embed_dim + 1))
        ax.plot(xs, mean, label=label, **style)
        ax.fill_between(xs, mean - std, mean + std,
                        alpha=0.15, color=style["color"])

    probe_style = PLOT_STYLES["PCA + probe"]
    ax.plot(list(range(1, embed_dim + 1)),
            pca_probe_metrics["accuracies"],
            label="PCA + probe", **probe_style)

    ax.set_xlabel("Prefix size k")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Prefix classification accuracy vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"prefix_accuracy{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp2] Saved {path}")


def plot_paired_cosines(all_metrics, pca_probe_metrics, run_dir, fig_stamp, embed_dim):
    """
    Paired cosine |cos(b_k, u_k)| per dimension k for MSE models vs PCA eigenvectors.
    Tests ordering (dim k aligned with exactly eigenvector k), not span coverage.
    Unlike column_alignment which takes max over top-k set — this is strictly paired.
    """
    xs_pca      = list(range(1, embed_dim + 1))
    probe_style = PLOT_STYLES["PCA + probe"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for mc in MODEL_CONFIGS:
        if mc["tag"] not in MSE_TAGS:
            continue
        tag, label = mc["tag"], mc["label"]
        style   = PLOT_STYLES[label]
        n_seeds = len(all_metrics[tag])
        pca_mat = np.array([all_metrics[tag][s]["pca_paired"] for s in range(n_seeds)])
        mean, std = pca_mat.mean(0), pca_mat.std(0)
        ax.plot(xs_pca, mean, label=label, **style)
        ax.fill_between(xs_pca, mean - std, mean + std, alpha=0.15, color=style["color"])

    ax.plot(xs_pca, pca_probe_metrics["pca_paired"], label="PCA + probe", **probe_style)

    ax.set_xlabel("Dimension k")
    ax.set_ylabel("|cos(b_k, u_k)|")
    ax.set_title(
        "Ordered PCA recovery: paired cosine |cos(b_k, u_k)|  [MSE models]\n"
        "encoder dim k vs PCA eigenvector k  —  tests ordering, not span\n"
        "cf. column alignment which takes max over top-k set — this is strictly paired"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(run_dir, f"ordered_pca_recovery_paired_cosine{fig_stamp}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[exp2] Saved {path}")


# ==============================================================================
# Save / load raw data
# ==============================================================================

def save_raw_data(all_metrics, all_histories, all_accuracies,
                  pca_probe_metrics, run_dir):
    # metrics
    def _nan_to_neg1(vals):
        return [v if not np.isnan(v) else -1.0 for v in vals]

    md = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        md[f"{tag}_pca_angles"]  = np.array([m["pca_angles"] for m in all_metrics[tag]])
        md[f"{tag}_lda_angles"]  = np.array([_nan_to_neg1(m["lda_angles"]) for m in all_metrics[tag]])
        md[f"{tag}_pca_cosine"]  = np.array([m["pca_cosine"] for m in all_metrics[tag]])
        md[f"{tag}_lda_cosine"]  = np.array([_nan_to_neg1(m["lda_cosine"]) for m in all_metrics[tag]])
        md[f"{tag}_pca_paired"]  = np.array([m["pca_paired"] for m in all_metrics[tag]])
        md[f"{tag}_lda_paired"]  = np.array([_nan_to_neg1(m["lda_paired"]) for m in all_metrics[tag]])
        md[f"{tag}_prefix_sizes"] = np.array(all_metrics[tag][0]["prefix_sizes"])
        md[f"{tag}_n_lda"]        = np.array(all_metrics[tag][0]["n_lda"])
    md["pca_probe_pca_angles"] = np.array(pca_probe_metrics["pca_angles"])
    md["pca_probe_lda_angles"] = np.array(_nan_to_neg1(pca_probe_metrics["lda_angles"]))
    md["pca_probe_pca_cosine"] = np.array(pca_probe_metrics["pca_cosine"])
    md["pca_probe_lda_cosine"] = np.array(_nan_to_neg1(pca_probe_metrics["lda_cosine"]))
    md["pca_probe_pca_paired"] = np.array(pca_probe_metrics["pca_paired"])
    md["pca_probe_lda_paired"] = np.array(_nan_to_neg1(pca_probe_metrics["lda_paired"]))
    md["pca_probe_accuracies"] = np.array(pca_probe_metrics["accuracies"])
    md["pca_probe_n_lda"]      = np.array(pca_probe_metrics["n_lda"])
    np.savez(os.path.join(run_dir, "metrics_raw.npz"), **md)

    # histories
    hd = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
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

    # accuracies
    ad = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        ad[f"{tag}_accuracies"] = np.array(all_accuracies[tag])
    np.savez(os.path.join(run_dir, "accuracies_raw.npz"), **ad)

    print(f"[exp2] Saved metrics_raw.npz, histories_raw.npz, accuracies_raw.npz")


def save_results_summary(all_metrics, all_accuracies, pca_probe_metrics,
                         run_dir, embed_dim):
    n_lda = pca_probe_metrics["n_lda"]
    key_k = sorted(set([1, n_lda // 2, n_lda, embed_dim // 2, embed_dim]))

    lines = ["=" * 80,
             "Experiment 2: Divergence — Objective-Specific Privileged Bases",
             "=" * 80, ""]

    header = f"{'Model':<30}"
    for k in key_k:
        header += f"  PCAang@{k:2d}  LDAang@{k:2d}  PCAcos@{k:2d}  LDAcos@{k:2d}  acc@{k:2d}"
    lines.append(header)
    lines.append("-" * 100)

    def row_for(label, pca_angs, lda_angs, pca_cos, lda_cos, accs):
        r = f"{label:<30}"
        for k in key_k:
            idx    = k - 1
            pa     = f"{pca_angs[idx]:.1f}°"  if idx < len(pca_angs) else "  —"
            la_raw = lda_angs[idx] if idx < len(lda_angs) else float("nan")
            la     = f"{la_raw:.1f}°" if (not np.isnan(la_raw) and la_raw >= 0) else "  —"
            pc     = f"{pca_cos[idx]:.3f}"    if idx < len(pca_cos)  else "  —"
            lc_raw = lda_cos[idx] if idx < len(lda_cos) else float("nan")
            lc     = f"{lc_raw:.3f}" if (not np.isnan(lc_raw) and lc_raw >= 0) else "  —"
            ac     = f"{accs[idx]:.3f}"       if idx < len(accs)     else "  —"
            r     += f"  {pa:>10}  {la:>10}  {pc:>10}  {lc:>10}  {ac:>8}"
        return r

    for mc in MODEL_CONFIGS:
        tag     = mc["tag"]
        label   = mc["label"]
        n_seeds = len(all_metrics[tag])
        pca_mean = np.array([all_metrics[tag][s]["pca_angles"] for s in range(n_seeds)]).mean(0)
        lda_mean = np.array([
            [v if not np.isnan(v) else -1.0 for v in all_metrics[tag][s]["lda_angles"]]
            for s in range(n_seeds)
        ]).mean(0)
        pca_cos_mean = np.array([all_metrics[tag][s]["pca_cosine"] for s in range(n_seeds)]).mean(0)
        lda_cos_mean = np.array([
            [v if not np.isnan(v) else -1.0 for v in all_metrics[tag][s]["lda_cosine"]]
            for s in range(n_seeds)
        ]).mean(0)
        acc_mean = np.array(all_accuracies[tag]).mean(0)
        lines.append(row_for(label, pca_mean, lda_mean, pca_cos_mean, lda_cos_mean, acc_mean))

    lines.append(row_for("PCA + probe",
                         pca_probe_metrics["pca_angles"],
                         pca_probe_metrics["lda_angles"],
                         pca_probe_metrics["pca_cosine"],
                         pca_probe_metrics["lda_cosine"],
                         pca_probe_metrics["accuracies"]))

    lines += ["",
              "PCAang: mean principal angle to top-k PCA subspace (lower = PCA-like)",
              "LDAang: mean principal angle to top-k LDA subspace (lower = LDA-like)",
              "PCAcos: mean max cosine similarity to top-k PCA directions (higher = PCA-like)",
              "LDAcos: mean max cosine similarity to top-k LDA directions (higher = LDA-like)",
              f"LDA metrics only up to k={n_lda} (= C-1 = number of LDA directions)",
              "Accuracy: mean over seeds for neural models, single run for PCA+probe"]

    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2] Saved {path}")


def save_experiment_description(cfg, run_dir, standard_mrl_m, dataset_info):
    note = cfg.get("experiment_note", "").strip()
    note_block = (
        ["=" * 60, "EXPERIMENT NOTE:", f"  {note}", "=" * 60, ""]
        if note else []
    )
    lines = note_block + [
        "Experiment 2: Divergence — Objective-Specific Privileged Bases",
        "=" * 60, "",
        "Purpose:",
        "  Show full-prefix MRL privilege is determined by objective:",
        "  MSE loss → encoder aligns with PCA (variance-optimal subspace)",
        "  CE loss  → encoder aligns with LDA (classification-optimal subspace)",
        "",
        "Models (MSE group):",
        "  1. LAE                    — no ordering baseline",
        "  2. Full prefix MRL+ortho  — reconstruction + dec ortho → expect PCA alignment",
        "  3. PrefixL1 MSE (rev)     — MSE + front-loaded L1; dims reversed before eval",
        "  4. Standard MRL (MSE)     — MRL at M={} only".format(standard_mrl_m),
        "  5. NonUniform L2          — Kunin et al. (2019) sum L2 reg → PCA recovery",
        "",
        "Models (CE group):",
        "  6. Normal CE              — plain CE baseline, no ordering",
        "  7. Full prefix MRL (CE)   — classification → expect LDA alignment",
        "  8. PrefixL1 (CE) (rev)    — CE + front-loaded L1; dims reversed before eval",
        "  9. Standard MRL (CE)      — CE at M={} → LDA but weaker".format(standard_mrl_m),
        "",
        "Baseline:",
        " 10. PCA + probe            — angle to PCA = 0 by definition",
        "",
        "Metric: encoder rows B^T[:,1:k] compared to PCA/LDA subspaces",
        "",
        "Dataset info:",
    ] + [f"  {k}: {v}" for k, v in dataset_info.items()] + [
        "",
        "Config:",
    ] + [f"  {k}: {v}" for k, v in cfg.items()]

    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2] Saved {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 2: Divergence")
    parser.add_argument("--fast",        action="store_true",
                        help="Smoke test: synthetic, d=16, 1 seed, 5 epochs")
    parser.add_argument("--dataset",     type=str, default=None)
    parser.add_argument("--embed_dim",   type=int, default=None)
    parser.add_argument("--use-weights", type=str, default=None, metavar="FOLDER",
                        help="Reload .pt files, recompute metrics, regenerate plots")
    args = parser.parse_args()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    dataset           = args.dataset   or DATASET
    embed_dim         = args.embed_dim or EMBED_DIM
    synthetic_variant = SYNTHETIC_VARIANT
    epochs            = EPOCHS
    patience       = PATIENCE
    seeds          = SEEDS
    standard_mrl_m = STANDARD_MRL_M

    if args.fast:
        dataset        = args.dataset   or DATASET
        # For synthetic use a small d; keep configured dim for real datasets
        embed_dim      = args.embed_dim or (16 if dataset == "synthetic" else EMBED_DIM)
        epochs         = 5
        patience       = 3
        seeds          = [42]
        standard_mrl_m = [m for m in STANDARD_MRL_M if m <= embed_dim]
        print(f"[exp2] --fast mode: {dataset}, d={embed_dim}, 1 seed, {epochs} epochs")

    cfg = dict(
        experiment_name    = "exp2_divergence",
        dataset            = dataset,
        synthetic_variant  = synthetic_variant,
        embed_dim          = embed_dim,
        epochs             = epochs,
        patience           = patience,
        lr                 = LR,
        batch_size         = BATCH_SIZE,
        weight_decay       = WEIGHT_DECAY,
        seeds              = seeds,
        standard_mrl_m     = standard_mrl_m,
        l1_lambda          = L1_LAMBDA,
        nonuniform_l2_lam  = NONUNIFORM_L2_LAM,
        fast               = args.fast,
        experiment_note    = EXPERIMENT_NOTE,
        # real-data params (ignored when dataset=="synthetic")
        p_svd              = P_SVD,
        tfidf_max_features = TFIDF_MAX_FEATURES,
        p_pca_proj         = P_PCA_PROJ,
        n_noise_dims       = N_NOISE_DIMS,
        sigma_noise        = SIGMA_NOISE,
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp2] Device: {device}"
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
        print(f"[exp2] --use-weights: loading from {weights_dir}")

        with open(os.path.join(weights_dir, "config.json")) as f:
            saved_cfg = json.load(f)
        embed_dim      = saved_cfg["embed_dim"]
        standard_mrl_m = saved_cfg["standard_mrl_m"]
        seeds          = saved_cfg["seeds"]
        dataset        = saved_cfg["dataset"]
        synthetic_variant = saved_cfg.get("synthetic_variant", "nonOrderedLDA")

        if dataset == "synthetic":
            data     = load_data(dataset, seed=seeds[0],
                                 synthetic_variant=synthetic_variant)
            from weight_symmetry.data.synthetic import SYNTHETIC_VARIANTS
            vparams  = SYNTHETIC_VARIANTS[synthetic_variant]
            raw      = load_synthetic(seed=seeds[0], **vparams)
            pca_dirs = raw["pca_dirs"].astype(np.float64)
            lda_dirs = raw["lda_dirs"].astype(np.float64)
            dataset_info = {k: v for k, v in raw["params"].items()}
        else:
            # Load saved directions (avoids re-fetching large datasets)
            dir_path = os.path.join(weights_dir, "directions.npz")
            if os.path.exists(dir_path):
                d_npz    = np.load(dir_path)
                pca_dirs = d_npz["pca_dirs"].astype(np.float64)
                lda_dirs = d_npz["lda_dirs"].astype(np.float64)
                print(f"[exp2] Loaded saved directions from {dir_path}")
            else:
                print(f"[exp2] directions.npz not found — re-fetching dataset ...")
                p_svd_c   = saved_cfg.get("p_svd",              P_SVD)
                tfidf_c   = saved_cfg.get("tfidf_max_features",  TFIDF_MAX_FEATURES)
                p_pca_c   = saved_cfg.get("p_pca_proj",          P_PCA_PROJ)
                n_noise_c = saved_cfg.get("n_noise_dims",        N_NOISE_DIMS)
                sigma_c   = saved_cfg.get("sigma_noise",         SIGMA_NOISE)
                _, pca_dirs, lda_dirs, _ = load_data_with_directions(
                    dataset, seed=seeds[0],
                    p_svd=p_svd_c, tfidf_max_features=tfidf_c,
                    p_pca_proj=p_pca_c, n_noise_dims=n_noise_c, sigma_noise=sigma_c,
                )
            # Re-load first-seed data for PCA probe baseline
            p_svd_c   = saved_cfg.get("p_svd",              P_SVD)
            tfidf_c   = saved_cfg.get("tfidf_max_features",  TFIDF_MAX_FEATURES)
            p_pca_c   = saved_cfg.get("p_pca_proj",          P_PCA_PROJ)
            n_noise_c = saved_cfg.get("n_noise_dims",        N_NOISE_DIMS)
            sigma_c   = saved_cfg.get("sigma_noise",         SIGMA_NOISE)
            data, _, _, dataset_info = load_data_with_directions(
                dataset, seed=seeds[0],
                p_svd=p_svd_c, tfidf_max_features=tfidf_c,
                p_pca_proj=p_pca_c, n_noise_dims=n_noise_c, sigma_noise=sigma_c,
            )

        histories_npz = np.load(os.path.join(weights_dir, "histories_raw.npz"))
        all_histories = {}
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

        print("[exp2] Recomputing metrics from saved weights ...")
        all_metrics    = {mc["tag"]: [] for mc in MODEL_CONFIGS}
        all_accuracies = {mc["tag"]: [] for mc in MODEL_CONFIGS}
        for seed in seeds:
            for mc in MODEL_CONFIGS:
                tag  = mc["tag"]
                flip = mc["flip_dims"]
                ckpt = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
                if mc["model_type"] == "lae_heads":
                    model = LinearAEWithHeads(data.input_dim, embed_dim, data.n_classes)
                else:
                    model = LinearAE(data.input_dim, embed_dim)
                model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
                model.eval()
                m = compute_encoder_subspace_metrics(model, pca_dirs, lda_dirs,
                                                     flip_dims=flip,
                                                     model_type=mc["model_type"])
                all_metrics[tag].append(m)
                a = compute_prefix_accuracy(model, data, device, mc["model_type"],
                                            flip_dims=flip)
                all_accuracies[tag].append(a)

        pca_probe_metrics = compute_pca_probe_metrics(pca_dirs, lda_dirs, data, embed_dim)

        sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

        save_raw_data(all_metrics, all_histories, all_accuracies, pca_probe_metrics, run_dir)
        save_experiment_description(saved_cfg, run_dir, standard_mrl_m, dataset_info)

        print("\n[exp2] Generating plots ...")
        plot_training_curves(all_histories, run_dir, fig_stamp)
        plot_subspace_angles(all_metrics, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
        plot_prefix_accuracy(all_accuracies, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
        plot_paired_cosines(all_metrics, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
        save_results_summary(all_metrics, all_accuracies, pca_probe_metrics, run_dir, embed_dim)
        elapsed = time.time() - t_start
        save_runtime(run_dir, elapsed)
        save_code_snapshot(run_dir)
        print(f"\n[exp2] Done. Results in: {run_dir}")
        return

    # ------------------------------------------------------------------
    # Setup run directory
    # ------------------------------------------------------------------
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    save_config(cfg, run_dir)

    # ------------------------------------------------------------------
    # Load data + ground-truth directions
    # ------------------------------------------------------------------
    from weight_symmetry.data.synthetic import SYNTHETIC_VARIANTS
    print(f"\n[exp2] Loading {dataset} (variant={synthetic_variant}) ...")

    if dataset == "synthetic":
        vparams      = SYNTHETIC_VARIANTS[synthetic_variant]
        data         = load_data(dataset, seed=seeds[0], synthetic_variant=synthetic_variant)
        raw_data     = load_synthetic(seed=seeds[0], **vparams)
        pca_dirs     = raw_data["pca_dirs"].astype(np.float64)    # (p, p_noise)
        lda_dirs     = raw_data["lda_dirs"].astype(np.float64)    # (p, C-1)
        dataset_info = {k: v for k, v in raw_data["params"].items()}
    else:
        data, pca_dirs, lda_dirs, dataset_info = load_data_with_directions(
            dataset, seed=seeds[0],
            p_svd=P_SVD, tfidf_max_features=TFIDF_MAX_FEATURES,
            p_pca_proj=P_PCA_PROJ, n_noise_dims=N_NOISE_DIMS, sigma_noise=SIGMA_NOISE,
        )
        # Ensure embed_dim does not exceed input_dim
        if embed_dim > data.input_dim:
            raise ValueError(
                f"embed_dim={embed_dim} > input_dim={data.input_dim} for dataset '{dataset}'. "
                f"Reduce EMBED_DIM in the CONFIG block."
            )

    print(f"[exp2] pca_dirs: {pca_dirs.shape}  lda_dirs: {lda_dirs.shape}")

    # Save directions for --use-weights reloading (avoids re-fetching dataset)
    np.savez(os.path.join(run_dir, "directions.npz"),
             pca_dirs=pca_dirs, lda_dirs=lda_dirs)

    save_experiment_description(cfg, run_dir, standard_mrl_m, dataset_info)

    # ------------------------------------------------------------------
    # Train + evaluate over seeds
    # ------------------------------------------------------------------
    all_histories  = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_metrics    = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_accuracies = {mc["tag"]: [] for mc in MODEL_CONFIGS}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[exp2] === Seed {seed} ({seed_idx+1}/{len(seeds)}) ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        if dataset == "synthetic":
            seed_data = load_data(dataset, seed=seed, synthetic_variant=synthetic_variant)
            seed_raw  = load_synthetic(seed=seed, **vparams)
            seed_pca  = seed_raw["pca_dirs"].astype(np.float64)
            seed_lda  = seed_raw["lda_dirs"].astype(np.float64)
        else:
            seed_data, seed_pca, seed_lda, _ = load_data_with_directions(
                dataset, seed=seed,
                p_svd=P_SVD, tfidf_max_features=TFIDF_MAX_FEATURES,
                p_pca_proj=P_PCA_PROJ, n_noise_dims=N_NOISE_DIMS, sigma_noise=SIGMA_NOISE,
            )

        trained = train_all_models(seed_data, cfg, run_dir, seed, standard_mrl_m, device)

        for mc in MODEL_CONFIGS:
            tag       = mc["tag"]
            flip      = mc["flip_dims"]
            model, history = trained[tag]
            all_histories[tag].append(history)

            metrics = compute_encoder_subspace_metrics(model, seed_pca, seed_lda,
                                                       flip_dims=flip,
                                                       model_type=mc["model_type"])
            all_metrics[tag].append(metrics)

            accs = compute_prefix_accuracy(model, seed_data, device, mc["model_type"],
                                           flip_dims=flip)
            all_accuracies[tag].append(accs)

            n_lda_avail = min(metrics['n_lda'], len(metrics['lda_angles']))
            lda_str = f"{metrics['lda_angles'][n_lda_avail-1]:.1f}°" if n_lda_avail > 0 else "—"
            print(f"  [{tag}] PCA angle@k={embed_dim}: {metrics['pca_angles'][-1]:.1f}°  "
                  f"LDA angle@k={n_lda_avail}: {lda_str}  "
                  f"acc@k={embed_dim}: {accs[-1]:.3f}")

    # ------------------------------------------------------------------
    # PCA + probe baseline (use first seed's data/directions)
    # ------------------------------------------------------------------
    print("\n[exp2] Computing PCA + probe baseline ...")
    pca_probe_metrics = compute_pca_probe_metrics(pca_dirs, lda_dirs, data, embed_dim)

    # ------------------------------------------------------------------
    # Save raw data + plots + summary
    # ------------------------------------------------------------------
    save_raw_data(all_metrics, all_histories, all_accuracies, pca_probe_metrics, run_dir)

    print("\n[exp2] Generating plots ...")
    plot_training_curves(all_histories, run_dir, fig_stamp)
    plot_subspace_angles(all_metrics, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
    plot_prefix_accuracy(all_accuracies, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
    plot_paired_cosines(all_metrics, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
    save_results_summary(all_metrics, all_accuracies, pca_probe_metrics, run_dir, embed_dim)

    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"\n[exp2] Done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
