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
    MSELoss, FullPrefixMRLLoss, FullPrefixCELoss, StandardMRLCELoss
)
from weight_symmetry.training.trainer import train_ae
from weight_symmetry.evaluation.metrics import (
    compute_encoder_subspace_metrics, compute_prefix_accuracy
)

# ==============================================================================
# CONFIG
# ==============================================================================
DATASET        =  "synthetic"   # "synthetic"      → data/synthetic_data/{orderedLDA,nonOrderedLDA}/ (pre-generated)
                                 # "20newsgroups"   → fetched + TF-IDF + TruncatedSVD, p=P_SVD
                                 # "mnist_noise"    → MNIST PCA-projected + noise dims, p=P_PCA_PROJ+N_NOISE_DIMS
                                 # "fashion_mnist_noise" → same as mnist_noise on Fashion-MNIST
ORDERED_LDA    = False            # only used when DATASET == "synthetic":
                                 #   True  → data/synthetic_data/orderedLDA/   (LDA dims ordered by disc. power)
                                 #   False → data/synthetic_data/nonOrderedLDA/ (random class means)
EMBED_DIM      = 50
EPOCHS         = 500
PATIENCE       = 50
LR             = 1e-3
BATCH_SIZE     = 256
WEIGHT_DECAY   = 1e-4
SEEDS          = [42]
STANDARD_MRL_M = [5, 10, 25, 50]   # for d=50
L1_LAMBDA      = 0.01               # PrefixL1 CE regularisation strength

# Real-data parameters (used when DATASET != "synthetic")
P_SVD              = 100    # 20newsgroups: TruncatedSVD output dim = model input_dim
TFIDF_MAX_FEATURES = 10000  # 20newsgroups: TF-IDF vocabulary size
P_PCA_PROJ         = 50     # mnist_noise / fashion_mnist_noise: PCA projection dim
N_NOISE_DIMS       = 25     # mnist_noise / fashion_mnist_noise: prepended noise dims
SIGMA_NOISE        = 5.0    # noise std (high-variance, class-agnostic)
# ==============================================================================

MODEL_CONFIGS = [
    dict(tag="mse_lae",       loss="mse",           model_type="lae",
         supervised=False, label="MSE LAE",               flip_dims=False),
    dict(tag="fp_mrl_mse",    loss="fullprefix",    model_type="lae",
         supervised=False, label="Full-prefix MRL (MSE)", flip_dims=False),
    dict(tag="fp_mrl_ce",     loss="fullprefix_ce", model_type="lae_heads",
         supervised=True,  label="Full-prefix MRL (CE)",  flip_dims=False),
    dict(tag="std_mrl_ce",    loss="standard_ce",   model_type="lae_heads",
         supervised=True,  label="Standard MRL (CE)",     flip_dims=False),
    dict(tag="prefix_l1_ce",  loss="prefix_l1_ce",  model_type="lae_heads",
         supervised=True,  label="PrefixL1 (CE) (rev)",   flip_dims=True),
]

# Tags for reconstruction-loss models — plotted on a secondary y-axis in training
# curves because their MSE scale (~0.001–0.1) is much smaller than CE (~1–3).
_MSE_TAGS = {"mse_lae", "fp_mrl_mse"}

PLOT_STYLES = {
    "MSE LAE":               dict(color="gray",    linestyle="--", marker=""),
    "Full-prefix MRL (MSE)": dict(color="blue",    linestyle="-",  marker=""),
    "Full-prefix MRL (CE)":  dict(color="green",   linestyle="-",  marker=""),
    "Standard MRL (CE)":     dict(color="orange",  linestyle="-",  marker="o", markersize=3),
    "PCA + probe":           dict(color="black",   linestyle=":",  marker=""),
    "PrefixL1 (CE) (rev)":   dict(color="crimson", linestyle="-",  marker="^", markersize=3),
}


# ==============================================================================
# Build loss
# ==============================================================================

def build_loss_fn(loss_type: str, embed_dim: int, standard_mrl_m: list,
                  n_classes: int, l1_lambda: float = 0.01):
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "fullprefix":
        return FullPrefixMRLLoss()
    elif loss_type == "fullprefix_ce":
        return FullPrefixCELoss()
    elif loss_type == "standard_ce":
        return StandardMRLCELoss(prefix_sizes=standard_mrl_m)
    elif loss_type == "prefix_l1_ce":
        from weight_symmetry.losses.losses import PrefixL1CELoss
        return PrefixL1CELoss(l1_lambda=l1_lambda)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==============================================================================
# Training
# ==============================================================================

def train_all_models(data, cfg, run_dir, seed, standard_mrl_m, device):
    """Train all 4 neural models for one seed. Returns dict tag -> (model, history)."""
    models = {}
    for mc in MODEL_CONFIGS:
        tag        = mc["tag"]
        supervised = mc["supervised"]
        n_classes  = data.n_classes

        if mc["model_type"] == "lae_heads":
            model = LinearAEWithHeads(
                data.input_dim, cfg["embed_dim"], n_classes
            ).to(device)
        else:
            model = LinearAE(data.input_dim, cfg["embed_dim"]).to(device)

        torch.manual_seed(seed)
        loss_fn = build_loss_fn(mc["loss"], cfg["embed_dim"], standard_mrl_m, n_classes,
                                l1_lambda=cfg.get("l1_lambda", L1_LAMBDA))
        opt     = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        seed_cfg       = dict(cfg)
        seed_cfg["seed"] = seed

        history = train_ae(
            model, loss_fn, opt, data, seed_cfg,
            run_dir, f"seed{seed}_{tag}",
            orthogonalize=False, supervised=supervised
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
    accuracies   = []

    X_tr = data.X_train.numpy()
    X_te = data.X_test.numpy()
    y_tr = data.y_train.numpy()
    y_te = data.y_test.numpy()

    for k in range(1, d + 1):
        # PCA angle = 0 by definition (this IS the PCA subspace)
        pca_angles.append(0.0)

        # LDA angle = subspace_angle(top-k PCA, top-k LDA)
        from weight_symmetry.evaluation.metrics import subspace_angle
        if k <= n_lda:
            lda_angles.append(subspace_angle(
                pca_dirs[:, :k].astype(np.float64),
                lda_dirs[:, :k].astype(np.float64)
            ))
        else:
            lda_angles.append(float("nan"))

        # Prefix accuracy: logistic regression on k-dim PCA features
        Z_tr = X_tr @ pca_dirs[:, :k]   # (n_train, k)
        Z_te = X_te @ pca_dirs[:, :k]   # (n_test, k)
        clf  = LogisticRegression(max_iter=300, n_jobs=1)
        clf.fit(Z_tr, y_tr)
        accuracies.append(float(clf.score(Z_te, y_te)))

    return {
        "prefix_sizes": list(range(1, d + 1)),
        "pca_angles":   pca_angles,
        "lda_angles":   lda_angles,
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


def plot_subspace_angles(all_metrics, pca_probe_metrics, run_dir, fig_stamp,
                         embed_dim, standard_mrl_m):
    """Two-panel: angle to PCA (left) and angle to LDA (right) vs prefix k."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_lda = pca_probe_metrics["n_lda"]

    # Neural models
    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        style = PLOT_STYLES[label]
        n_seeds = len(all_metrics[tag])

        pca_mat = np.array([all_metrics[tag][s]["pca_angles"] for s in range(n_seeds)])
        lda_raw = [all_metrics[tag][s]["lda_angles"] for s in range(n_seeds)]

        # PCA panel
        mean, std = pca_mat.mean(0), pca_mat.std(0)
        xs = list(range(1, embed_dim + 1))
        axes[0].plot(xs, mean, label=label, **style)
        axes[0].fill_between(xs, mean - std, mean + std,
                             alpha=0.15, color=style["color"])

        # LDA panel — only up to n_lda, skip NaN
        xs_lda = list(range(1, n_lda + 1))
        lda_mat = np.array([[row[k] for k in range(n_lda)]
                             for row in lda_raw])
        mean_l, std_l = lda_mat.mean(0), lda_mat.std(0)
        axes[1].plot(xs_lda, mean_l, label=label, **style)
        axes[1].fill_between(xs_lda, mean_l - std_l, mean_l + std_l,
                             alpha=0.15, color=style["color"])

    # PCA + probe baseline
    probe_style = PLOT_STYLES["PCA + probe"]
    xs     = list(range(1, embed_dim + 1))
    xs_lda = list(range(1, n_lda + 1))
    axes[0].plot(xs, pca_probe_metrics["pca_angles"],
                 label="PCA + probe", **probe_style)
    axes[1].plot(xs_lda, pca_probe_metrics["lda_angles"][:n_lda],
                 label="PCA + probe", **probe_style)

    # Standard MRL markers
    for m in standard_mrl_m:
        if m <= embed_dim:
            axes[1].axvline(x=m, color="orange", linestyle=":", alpha=0.4, lw=0.8)

    axes[0].set_xlabel("Prefix size k")
    axes[0].set_ylabel("Mean principal angle (degrees)")
    axes[0].set_title("Angle to PCA subspace (lower = more PCA-like)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Prefix size k")
    axes[1].set_ylabel("Mean principal angle (degrees)")
    axes[1].set_title(f"Angle to LDA subspace (lower = more LDA-like, k ≤ {n_lda})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

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


# ==============================================================================
# Save / load raw data
# ==============================================================================

def save_raw_data(all_metrics, all_histories, all_accuracies,
                  pca_probe_metrics, run_dir):
    # metrics
    md = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        md[f"{tag}_pca_angles"]  = np.array([m["pca_angles"] for m in all_metrics[tag]])
        md[f"{tag}_lda_angles"]  = np.array([
            [v if not np.isnan(v) else -1.0 for v in m["lda_angles"]]
            for m in all_metrics[tag]
        ])
        md[f"{tag}_prefix_sizes"] = np.array(all_metrics[tag][0]["prefix_sizes"])
        md[f"{tag}_n_lda"]        = np.array(all_metrics[tag][0]["n_lda"])
    md["pca_probe_pca_angles"] = np.array(pca_probe_metrics["pca_angles"])
    md["pca_probe_lda_angles"] = np.array([
        v if not np.isnan(v) else -1.0 for v in pca_probe_metrics["lda_angles"]
    ])
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
        header += f"  PCA@k={k:2d}  LDA@k={k:2d}  acc@k={k:2d}"
    lines.append(header)
    lines.append("-" * 80)

    def row_for(label, pca_angs, lda_angs, accs):
        r = f"{label:<30}"
        for k in key_k:
            idx = k - 1
            pa  = f"{pca_angs[idx]:.1f}°" if idx < len(pca_angs) else "   —"
            la_raw = lda_angs[idx] if idx < len(lda_angs) else float("nan")
            la  = f"{la_raw:.1f}°" if (not np.isnan(la_raw) and la_raw >= 0) else "  —"
            ac  = f"{accs[idx]:.3f}"  if idx < len(accs)   else "  —"
            r  += f"  {pa:>8}  {la:>8}  {ac:>8}"
        return r

    # Neural models (mean over seeds)
    for mc in MODEL_CONFIGS:
        tag     = mc["tag"]
        label   = mc["label"]
        n_seeds = len(all_metrics[tag])
        pca_mean = np.array([all_metrics[tag][s]["pca_angles"] for s in range(n_seeds)]).mean(0)
        lda_mean = np.array([
            [v if not np.isnan(v) else -1.0 for v in all_metrics[tag][s]["lda_angles"]]
            for s in range(n_seeds)
        ]).mean(0)
        acc_mean = np.array(all_accuracies[tag]).mean(0)
        lines.append(row_for(label, pca_mean, lda_mean, acc_mean))

    # PCA + probe
    lines.append(row_for("PCA + probe",
                         pca_probe_metrics["pca_angles"],
                         pca_probe_metrics["lda_angles"],
                         pca_probe_metrics["accuracies"]))

    lines += ["",
              "PCA angle: mean principal angle to top-k PCA subspace (lower = PCA-like)",
              "LDA angle: mean principal angle to top-k LDA subspace (lower = LDA-like)",
              f"LDA angles only up to k={n_lda} (= C-1 = number of LDA directions)",
              "Accuracy: mean over seeds for neural models, single run for PCA+probe"]

    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2] Saved {path}")


def save_experiment_description(cfg, run_dir, standard_mrl_m, dataset_info):
    lines = [
        "Experiment 2: Divergence — Objective-Specific Privileged Bases",
        "=" * 60, "",
        "Purpose:",
        "  Show full-prefix MRL privilege is determined by objective:",
        "  MSE loss → encoder aligns with PCA (variance-optimal subspace)",
        "  CE loss  → encoder aligns with LDA (classification-optimal subspace)",
        "",
        "Models:",
        "  1. MSE LAE              — no ordering baseline",
        "  2. Full-prefix MRL (MSE)— reconstruction → expect PCA alignment",
        "  3. Full-prefix MRL (CE) — classification → expect LDA alignment",
        "  4. Standard MRL (CE)    — CE with M={} → LDA but weaker".format(standard_mrl_m),
        "  5. PrefixL1 (CE) (rev) — CE + front-loaded L1; dims reversed before eval",
        "  6. PCA + probe          — theoretical baseline (angle to PCA = 0)",
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
    dataset        = args.dataset   or DATASET
    embed_dim      = args.embed_dim or EMBED_DIM
    ordered_lda    = ORDERED_LDA
    epochs         = EPOCHS
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
        ordered_lda        = ordered_lda,
        embed_dim          = embed_dim,
        epochs             = epochs,
        patience           = patience,
        lr                 = LR,
        batch_size         = BATCH_SIZE,
        weight_decay       = WEIGHT_DECAY,
        seeds              = seeds,
        standard_mrl_m     = standard_mrl_m,
        l1_lambda          = L1_LAMBDA,
        fast               = args.fast,
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
        ordered_lda    = saved_cfg.get("ordered_lda", False)

        if dataset == "synthetic":
            data     = load_data(dataset, seed=seeds[0], ordered_lda=ordered_lda)
            raw      = load_synthetic(seed=seeds[0], ordered_lda=ordered_lda)
            pca_dirs = raw["pca_dirs"].astype(np.float64)
            lda_dirs = raw["lda_dirs"].astype(np.float64)
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
            data, _, _, _ = load_data_with_directions(
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
                model.load_state_dict(torch.load(ckpt, weights_only=True))
                model.eval()
                m = compute_encoder_subspace_metrics(model, pca_dirs, lda_dirs,
                                                     flip_dims=flip)
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
        print("\n[exp2] Generating plots ...")
        plot_training_curves(all_histories, run_dir, fig_stamp)
        plot_subspace_angles(all_metrics, pca_probe_metrics, run_dir, fig_stamp,
                             embed_dim, standard_mrl_m)
        plot_prefix_accuracy(all_accuracies, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
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
    print(f"\n[exp2] Loading {dataset} (ordered_lda={ordered_lda}) ...")

    if dataset == "synthetic":
        data         = load_data(dataset, seed=seeds[0], ordered_lda=ordered_lda)
        raw_data     = load_synthetic(seed=seeds[0], ordered_lda=ordered_lda)
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
            seed_data = load_data(dataset, seed=seed, ordered_lda=ordered_lda)
            seed_raw  = load_synthetic(seed=seed, ordered_lda=ordered_lda)
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
                                                       flip_dims=flip)
            all_metrics[tag].append(metrics)

            accs = compute_prefix_accuracy(model, seed_data, device, mc["model_type"],
                                           flip_dims=flip)
            all_accuracies[tag].append(accs)

            print(f"  [{tag}] PCA angle@k={embed_dim}: {metrics['pca_angles'][-1]:.1f}°  "
                  f"LDA angle@k={metrics['n_lda']}: "
                  f"{metrics['lda_angles'][metrics['n_lda']-1]:.1f}°  "
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
    plot_subspace_angles(all_metrics, pca_probe_metrics, run_dir, fig_stamp,
                         embed_dim, standard_mrl_m)
    plot_prefix_accuracy(all_accuracies, pca_probe_metrics, run_dir, fig_stamp, embed_dim)
    save_results_summary(all_metrics, all_accuracies, pca_probe_metrics, run_dir, embed_dim)

    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"\n[exp2] Done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
