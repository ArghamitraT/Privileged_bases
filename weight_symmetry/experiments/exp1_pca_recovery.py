"""
Experiment 1: PCA Subspace Recovery
-------------------------------------
Validates Theorem 1 of the paper (two-part PCA recovery result).

Trains five linear AE variants on raw MNIST or Fashion-MNIST and measures
how well each recovers the ground-truth PCA subspace at every prefix size m=1..d.

Models trained (all with MSE reconstruction loss):
    1. MSE LAE           — no constraint (baseline, expect no recovery)
    2. MSE LAE + ortho   — A^T A = I (known PCA recovery result, upper baseline)
    3. Standard MRL      — prefix losses at M only (expect gaps at non-M sizes)
    4. Full-prefix MRL   — prefix losses at m=1..d, no constraint (Theorem 1 Part 1)
    5. Full-prefix MRL + dec ortho  — m=1..d with A^T A = I (Theorem 1 Part 2)
    6. Full-prefix MRL + both ortho — m=1..d with A^T A = I and B B^T = I (enc+dec)

Metrics (per prefix m, averaged over seeds):
    - Mean principal angle (degrees) between A_{1:m} and top-m PCA subspace
    - Mean max cosine similarity between A columns and PCA eigenvectors

Outputs (in run_dir):
    - training_curves_{stamp}.png
    - subspace_angle_{stamp}.png
    - column_alignment_{stamp}.png
    - results_summary.txt
    - experiment_description.log
    - runtime.txt
    - code_snapshot/
    - config.json

Usage:
    Conda environment: mrl_env_cuda12  (GPU, CUDA 12.4, torch 2.5.1+cu124)
                       mrl_env         (CPU fallback — same code, no GPU)

    python weight_symmetry/experiments/exp1_pca_recovery.py --fast
    python weight_symmetry/experiments/exp1_pca_recovery.py
    python weight_symmetry/experiments/exp1_pca_recovery.py --dataset fashion_mnist --embed_dim 16
    python weight_symmetry/experiments/exp1_pca_recovery.py --use-weights exprmnt_2026_04_12__10_00_00
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

# Make weight_symmetry imports work regardless of cwd
_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.utility import (
    create_run_dir, save_runtime, save_code_snapshot, save_config
)
from weight_symmetry.data.loader import load_data
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.losses.losses import (
    MSELoss, StandardMRLLoss, FullPrefixMRLLoss, OftadehLoss
)
from weight_symmetry.training.trainer import train_ae
from weight_symmetry.evaluation.metrics import (
    compute_pca_directions, compute_all_prefix_metrics
)

# ==============================================================================
# CONFIG — edit here for full runs; --fast overrides below in main()
# ==============================================================================
DATASET       = "fashion_mnist"          # "mnist" or "fashion_mnist"
EMBED_DIM     = 32               # 16 or 32
EPOCHS        = 500
PATIENCE      = 50
# Non-ortho models (Adam)
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
# Ortho models (SGD + cosine annealing, weight_decay=0 to avoid fighting QR projection)
LR_ORTHO      = 0.05
BATCH_SIZE    = 256
SEEDS         = [42]              # [42, 43, 44, 45, 46]
# Standard MRL prefix set — adjusted to EMBED_DIM
STANDARD_MRL_M_32 = [4, 8, 16, 32]
STANDARD_MRL_M_16 = [2, 4, 8, 16]

# ------------------------------------------------------------------------------
# EXPERIMENT NOTE — fill in manually to describe this run's motivation/changes
# Printed at the top of experiment_description.log
# ------------------------------------------------------------------------------
EXPERIMENT_NOTE = "Trying to get 1 cos sim with PCA"
# ==============================================================================


MODEL_CONFIGS = [
    dict(tag="mse_lae",              loss="mse",          ortho=False, ortho_enc=False,
         label="MSE LAE"),
    dict(tag="mse_lae_ortho",        loss="mse",          ortho=True,  ortho_enc=False,
         label="MSE LAE + ortho"),
    dict(tag="standard_mrl",         loss="standard_mrl", ortho=False, ortho_enc=False,
         label="Standard MRL"),
    dict(tag="fullprefix_mrl",       loss="fullprefix",   ortho=False, ortho_enc=False,
         label="Full-prefix MRL"),
    dict(tag="fullprefix_mrl_ortho", loss="fullprefix",   ortho=True,  ortho_enc=False,
         label="Full-prefix MRL + dec ortho"),
    # dict(tag="fullprefix_mrl_both_ortho", loss="fullprefix", ortho=True, ortho_enc=True,
    #      label="Full-prefix MRL + both ortho"),
]

PLOT_STYLES = {
    "MSE LAE":                       dict(color="gray",   linestyle="--",  marker=""),
    "MSE LAE + ortho":               dict(color="black",  linestyle="-",   marker=""),
    "Standard MRL":                  dict(color="orange", linestyle="-",   marker="o", markersize=3),
    "Full-prefix MRL":               dict(color="blue",   linestyle="-",   marker=""),
    "Full-prefix MRL + dec ortho":   dict(color="green",  linestyle="-",   marker=""),
    "Full-prefix MRL + both ortho":  dict(color="red",    linestyle="-",   marker=""),
}


def build_loss_fn(loss_type: str, embed_dim: int, standard_mrl_m: list):
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "standard_mrl":
        return StandardMRLLoss(prefix_sizes=standard_mrl_m)
    elif loss_type == "fullprefix":
        return FullPrefixMRLLoss()
    elif loss_type == "oftadeh":
        return OftadehLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_all_models(data, cfg: dict, run_dir: str, seed: int, standard_mrl_m: list,
                     device: torch.device):
    """Train all models for one seed. Returns dict: tag -> (model, history)."""
    models = {}
    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        model = LinearAE(input_dim=data.input_dim, embed_dim=cfg["embed_dim"]).to(device)
        torch.manual_seed(seed)
        loss_fn = build_loss_fn(mc["loss"], cfg["embed_dim"], standard_mrl_m)

        is_ortho = mc["ortho"] or mc["ortho_enc"]
        if is_ortho:
            # SGD + momentum with no weight decay: avoids fighting the QR projection
            # that enforces A^T A = I (weight decay would shrink columns toward 0)
            opt = torch.optim.SGD(
                model.parameters(), lr=cfg["lr_ortho"], momentum=0.9, weight_decay=0.0
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg["epochs"], eta_min=1e-5
            )
        else:
            opt = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
            scheduler = None

        seed_cfg = dict(cfg)
        seed_cfg["seed"] = seed
        history = train_ae(
            model, loss_fn, opt, data, seed_cfg,
            run_dir, f"seed{seed}_{tag}",
            orthogonalize=mc["ortho"],
            orthogonalize_encoder=mc["ortho_enc"],
            scheduler=scheduler,
        )
        models[tag] = (model, history)
    return models


def plot_training_curves(all_histories, run_dir: str, fig_stamp: str):
    """Loss curves for all models across seeds (mean ± std)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        style = PLOT_STYLES[label]

        train_rows = [h["train_losses"] for h in all_histories[tag]]
        val_rows   = [h["val_losses"]   for h in all_histories[tag]]

        # Pad to same length (early stopping may differ across seeds)
        max_len = max(len(r) for r in train_rows)
        def pad(rows):
            return np.array([
                np.pad(r, (0, max_len - len(r)), constant_values=r[-1])
                for r in rows
            ])
        train_mat = pad(train_rows)
        val_mat   = pad(val_rows)

        epochs = np.arange(1, max_len + 1)
        for ax, mat, title in [(axes[0], train_mat, "Train Loss"),
                               (axes[1], val_mat,   "Val Loss")]:
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)
            ax.plot(epochs, mean, label=label, **style)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15,
                            color=style["color"])

    for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp1] Saved {path}")


def plot_prefix_metrics(all_metrics, run_dir: str, fig_stamp: str, embed_dim: int,
                        standard_mrl_m: list):
    """
    Two plots: subspace angle vs m, column alignment vs m.
    Mean ± std over seeds per model.
    Standard MRL has vertical markers at its evaluated prefix sizes.
    """
    prefix_sizes = list(range(1, embed_dim + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        style = PLOT_STYLES[label]

        ang_mat   = np.array([all_metrics[tag][s]["subspace_angles"]
                              for s in range(len(all_metrics[tag]))])
        align_mat = np.array([all_metrics[tag][s]["column_alignments"]
                              for s in range(len(all_metrics[tag]))])

        ang_mean, ang_std     = ang_mat.mean(0), ang_mat.std(0)
        align_mean, align_std = align_mat.mean(0), align_mat.std(0)

        for ax, mean, std in [(axes[0], ang_mean, ang_std),
                              (axes[1], align_mean, align_std)]:
            ax.plot(prefix_sizes, mean, label=label, **style)
            ax.fill_between(prefix_sizes, mean - std, mean + std,
                            alpha=0.15, color=style["color"])

    # Mark standard MRL evaluation points
    for m in standard_mrl_m:
        axes[0].axvline(x=m, color="orange", linestyle=":", alpha=0.5, linewidth=0.8)
        axes[1].axvline(x=m, color="orange", linestyle=":", alpha=0.5, linewidth=0.8)

    axes[0].set_xlabel("Prefix size m")
    axes[0].set_ylabel("Mean principal angle (degrees)")
    axes[0].set_title("Subspace Angle to PCA (lower = better recovery)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Prefix size m")
    axes[1].set_ylabel("Mean max cosine similarity")
    axes[1].set_title("Column Alignment to PCA eigenvectors (higher = better)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    ang_path   = os.path.join(run_dir, f"subspace_angle{fig_stamp}.png")
    align_path = os.path.join(run_dir, f"column_alignment{fig_stamp}.png")
    fig.savefig(ang_path, dpi=150)
    plt.close()

    # Column alignment separate
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        style = PLOT_STYLES[label]
        align_mat = np.array([all_metrics[tag][s]["column_alignments"]
                              for s in range(len(all_metrics[tag]))])
        mean = align_mat.mean(0)
        std  = align_mat.std(0)
        ax2.plot(prefix_sizes, mean, label=label, **style)
        ax2.fill_between(prefix_sizes, mean - std, mean + std,
                         alpha=0.15, color=style["color"])
    for m in standard_mrl_m:
        ax2.axvline(x=m, color="orange", linestyle=":", alpha=0.5, linewidth=0.8)
    ax2.set_xlabel("Prefix size m")
    ax2.set_ylabel("Mean max cosine similarity")
    ax2.set_title("Column Alignment to PCA eigenvectors")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(align_path, dpi=150)
    plt.close()

    print(f"[exp1] Saved {ang_path}")
    print(f"[exp1] Saved {align_path}")


def save_raw_data(all_metrics, all_histories, pca_dirs, run_dir: str):
    """
    Save all raw numerical data to disk so plots can be reproduced without re-training.

    Saves:
        metrics_raw.npz   — per-seed subspace_angles + column_alignments for every model
        histories_raw.npz — per-seed train_losses + val_losses for every model
        pca_directions.npy — ground-truth PCA eigenvectors (p, d)
    """
    import json

    # --- metrics ---
    metrics_dict = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        metrics_dict[f"{tag}_subspace_angles"]   = np.array(
            [m["subspace_angles"]   for m in all_metrics[tag]]
        )
        metrics_dict[f"{tag}_column_alignments"] = np.array(
            [m["column_alignments"] for m in all_metrics[tag]]
        )
        metrics_dict[f"{tag}_prefix_sizes"] = np.array(
            all_metrics[tag][0]["prefix_sizes"]
        )
    np.savez(os.path.join(run_dir, "metrics_raw.npz"), **metrics_dict)

    # --- histories (pad to same length per tag) ---
    histories_dict = {}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        train_list = [np.array(h["train_losses"]) for h in all_histories[tag]]
        val_list   = [np.array(h["val_losses"])   for h in all_histories[tag]]
        best_list  = [h["best_epoch"]             for h in all_histories[tag]]
        max_len    = max(len(t) for t in train_list)

        def pad_rows(rows, length):
            return np.array([
                np.pad(r, (0, length - len(r)), constant_values=r[-1]) for r in rows
            ])

        histories_dict[f"{tag}_train_losses"] = pad_rows(train_list, max_len)
        histories_dict[f"{tag}_val_losses"]   = pad_rows(val_list,   max_len)
        histories_dict[f"{tag}_best_epochs"]  = np.array(best_list)

    np.savez(os.path.join(run_dir, "histories_raw.npz"), **histories_dict)

    # --- PCA directions ---
    np.save(os.path.join(run_dir, "pca_directions.npy"), pca_dirs)

    print(f"[exp1] Saved metrics_raw.npz, histories_raw.npz, pca_directions.npy")


def save_results_summary(all_metrics, run_dir: str, embed_dim: int):
    """Write results_summary.txt with per-model mean metrics at key prefix sizes."""
    key_prefixes = sorted(set(
        [1, embed_dim // 4, embed_dim // 2, embed_dim]
    ))
    lines = ["=" * 70, "Experiment 1: PCA Subspace Recovery — Results Summary",
             "=" * 70, ""]

    lines.append(f"{'Model':<30}  " +
                 "  ".join(f"angle@m={m:2d}  align@m={m:2d}" for m in key_prefixes))
    lines.append("-" * 70)

    for mc in MODEL_CONFIGS:
        tag   = mc["tag"]
        label = mc["label"]
        n_seeds = len(all_metrics[tag])

        row = f"{label:<30}"
        for m in key_prefixes:
            idx = m - 1
            angs   = [all_metrics[tag][s]["subspace_angles"][idx]   for s in range(n_seeds)]
            aligns = [all_metrics[tag][s]["column_alignments"][idx] for s in range(n_seeds)]
            row += f"  {np.mean(angs):6.2f}°±{np.std(angs):.2f}  "
            row += f"{np.mean(aligns):.3f}±{np.std(aligns):.3f}"
        lines.append(row)

    lines += ["", "Columns: mean ± std over seeds",
              "angle = mean principal angle (degrees, lower = better recovery)",
              "align = mean max cosine similarity (higher = better alignment)"]

    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp1] Saved {path}")


def save_experiment_description(cfg: dict, run_dir: str, standard_mrl_m: list):
    note = cfg.get("experiment_note", "").strip()
    note_block = (
        ["=" * 60, "EXPERIMENT NOTE:", f"  {note}", "=" * 60, ""]
        if note else []
    )
    lines = (
        ["Experiment 1: PCA Subspace Recovery", "=" * 60, ""]
        + note_block
        + [
            "Purpose:",
            "  Validate Theorem 1 (two-part PCA recovery result).",
            "  Show full-prefix MRL recovers PCA subspaces at every m.",
            "  Show standard MRL has gaps at non-evaluated prefix sizes.",
            "  Show orthogonality additionally recovers individual eigenvectors.",
            "",
            "Optimizer:",
            "  Non-ortho models : Adam (lr={lr}, weight_decay={wd})".format(
                lr=cfg.get("lr"), wd=cfg.get("weight_decay")),
            "  Ortho models     : SGD+momentum (lr={lr}, weight_decay=0) + CosineAnnealingLR".format(
                lr=cfg.get("lr_ortho")),
            "",
            "Models:",
            "  1. MSE LAE                — baseline, no symmetry breaking",
            "  2. MSE LAE + ortho        — known PCA recovery (upper baseline)",
            "  3. Standard MRL           — prefix losses at M only",
            "  4. Full-prefix MRL        — prefix losses at m=1..d (Theorem 1 Part 1)",
            "  5. Full-prefix + dec ortho — same + A^T A = I (Theorem 1 Part 2)",
            "  6. Full-prefix + both ortho — same + A^T A = I and B B^T = I",
            "",
            f"Standard MRL M = {standard_mrl_m}",
            "",
            "Config:",
        ] + [f"  {k}: {v}" for k, v in cfg.items()]
    )

    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp1] Saved {path}")


# ==============================================================================
# Main
# ==============================================================================

def load_raw_data(weights_dir: str):
    """
    Reconstruct all_metrics, all_histories, pca_dirs from saved .npz files.
    Returns (all_metrics, all_histories, pca_dirs, embed_dim, standard_mrl_m, n_seeds).
    """
    import json

    metrics_npz   = np.load(os.path.join(weights_dir, "metrics_raw.npz"))
    histories_npz = np.load(os.path.join(weights_dir, "histories_raw.npz"))
    pca_dirs      = np.load(os.path.join(weights_dir, "pca_directions.npy"))

    # Read config to recover embed_dim and standard_mrl_m
    cfg_path = os.path.join(weights_dir, "config.json")
    with open(cfg_path) as f:
        saved_cfg = json.load(f)
    embed_dim      = saved_cfg["embed_dim"]
    standard_mrl_m = saved_cfg["standard_mrl_m"]

    # Infer n_seeds from first tag's array shape
    first_tag = MODEL_CONFIGS[0]["tag"]
    n_seeds = metrics_npz[f"{first_tag}_subspace_angles"].shape[0]

    all_metrics   = {}
    all_histories = {}

    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        ang_mat   = metrics_npz[f"{tag}_subspace_angles"]    # (n_seeds, d)
        align_mat = metrics_npz[f"{tag}_column_alignments"]  # (n_seeds, d)
        prefix_sizes = metrics_npz[f"{tag}_prefix_sizes"].tolist()

        all_metrics[tag] = [
            {
                "prefix_sizes":      prefix_sizes,
                "subspace_angles":   ang_mat[s].tolist(),
                "column_alignments": align_mat[s].tolist(),
            }
            for s in range(n_seeds)
        ]

        train_mat = histories_npz[f"{tag}_train_losses"]  # (n_seeds, max_epochs)
        val_mat   = histories_npz[f"{tag}_val_losses"]
        best_arr  = histories_npz[f"{tag}_best_epochs"]

        all_histories[tag] = [
            {
                "train_losses": train_mat[s].tolist(),
                "val_losses":   val_mat[s].tolist(),
                "best_epoch":   int(best_arr[s]),
            }
            for s in range(n_seeds)
        ]

    return all_metrics, all_histories, pca_dirs, embed_dim, standard_mrl_m


def main():
    parser = argparse.ArgumentParser(description="Exp 1: PCA Subspace Recovery")
    parser.add_argument("--fast",        action="store_true",
                        help="Smoke test: digits dataset, d=8, 1 seed, 10 epochs")
    parser.add_argument("--dataset",     type=str, default=None,
                        help="Override dataset (mnist, fashion_mnist, digits)")
    parser.add_argument("--embed_dim",   type=int, default=None,
                        help="Override embed_dim (16 or 32)")
    parser.add_argument("--use-weights", type=str, default=None, metavar="FOLDER",
                        help="Skip training; load metrics_raw.npz + histories_raw.npz "
                             "from FOLDER and regenerate plots only")
    args = parser.parse_args()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Apply config (full run defaults, then --fast overrides)
    # ------------------------------------------------------------------
    dataset   = args.dataset   or DATASET
    embed_dim = args.embed_dim or EMBED_DIM
    epochs    = EPOCHS
    patience  = PATIENCE
    seeds     = SEEDS

    if args.fast:
        dataset   = args.dataset or "digits"
        embed_dim = args.embed_dim or 8
        epochs    = 30
        patience  = 10
        seeds     = [42]
        print("[exp1] --fast mode: digits, d=8, 1 seed, 30 epochs")

    # Derive standard MRL prefix set after embed_dim is finalised
    # Use evenly spaced prefixes that don't exceed embed_dim
    if embed_dim == 16:
        standard_mrl_m = STANDARD_MRL_M_16
    elif embed_dim == 32:
        standard_mrl_m = STANDARD_MRL_M_32
    else:
        # Generic: 4 evenly spaced points up to embed_dim
        step = max(1, embed_dim // 4)
        standard_mrl_m = list(range(step, embed_dim + 1, step))
        if standard_mrl_m[-1] != embed_dim:
            standard_mrl_m.append(embed_dim)

    cfg = dict(
        experiment_name  = "exp1_pca_recovery",
        dataset          = dataset,
        embed_dim        = embed_dim,
        epochs           = epochs,
        patience         = patience,
        lr               = LR,
        lr_ortho         = LR_ORTHO,
        batch_size       = BATCH_SIZE,
        weight_decay     = WEIGHT_DECAY,
        seeds            = seeds,
        standard_mrl_m   = standard_mrl_m,
        fast             = args.fast,
        experiment_note  = EXPERIMENT_NOTE,
    )

    # ------------------------------------------------------------------
    # --use-weights: reload .pt files, recompute metrics, regenerate plots
    # ------------------------------------------------------------------
    if args.use_weights:
        import json
        weights_dir = args.use_weights
        if not os.path.isabs(weights_dir):
            from weight_symmetry.utility import get_path
            weights_dir = os.path.join(get_path("files/results"), weights_dir)

        print(f"[exp1] --use-weights: loading from {weights_dir}")

        # Read saved config
        with open(os.path.join(weights_dir, "config.json")) as f:
            saved_cfg = json.load(f)
        embed_dim      = saved_cfg["embed_dim"]
        standard_mrl_m = saved_cfg["standard_mrl_m"]
        seeds          = saved_cfg["seeds"]

        # Load PCA directions and histories (still correct)
        pca_dirs      = np.load(os.path.join(weights_dir, "pca_directions.npy"))
        histories_npz = np.load(os.path.join(weights_dir, "histories_raw.npz"))

        all_histories = {}
        for mc in MODEL_CONFIGS:
            tag = mc["tag"]
            train_mat = histories_npz[f"{tag}_train_losses"]
            val_mat   = histories_npz[f"{tag}_val_losses"]
            best_arr  = histories_npz[f"{tag}_best_epochs"]
            all_histories[tag] = [
                {"train_losses": train_mat[s].tolist(),
                 "val_losses":   val_mat[s].tolist(),
                 "best_epoch":   int(best_arr[s])}
                for s in range(len(seeds))
            ]

        # Reload .pt files and recompute metrics with fixed code
        print("[exp1] Recomputing metrics from saved weights ...")
        all_metrics = {mc["tag"]: [] for mc in MODEL_CONFIGS}
        for seed in seeds:
            for mc in MODEL_CONFIGS:
                tag   = mc["tag"]
                ckpt  = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
                model = LinearAE(input_dim=pca_dirs.shape[0], embed_dim=embed_dim)
                model.load_state_dict(torch.load(ckpt, weights_only=True))
                model.eval()
                metrics = compute_all_prefix_metrics(model, pca_dirs)
                all_metrics[tag].append(metrics)
                print(f"  [seed{seed}_{tag}] angle@m={embed_dim}: "
                      f"{metrics['subspace_angles'][-1]:.2f}°  "
                      f"align: {metrics['column_alignments'][-1]:.3f}")

        sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

        save_raw_data(all_metrics, all_histories, pca_dirs, run_dir)

        print("\n[exp1] Generating plots ...")
        plot_training_curves(all_histories, run_dir, fig_stamp)
        plot_prefix_metrics(all_metrics, run_dir, fig_stamp, embed_dim, standard_mrl_m)
        save_results_summary(all_metrics, run_dir, embed_dim)

        elapsed = time.time() - t_start
        save_runtime(run_dir, elapsed)
        save_code_snapshot(run_dir)
        print(f"\n[exp1] Done. Results in: {run_dir}")
        return

    # ------------------------------------------------------------------
    # Device — GPU if available, CPU fallback
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1] Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    cfg["device"] = str(device)

    # ------------------------------------------------------------------
    # Setup run directory
    # ------------------------------------------------------------------
    run_dir   = create_run_dir(fast=args.fast)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

    save_config(cfg, run_dir)
    save_experiment_description(cfg, run_dir, standard_mrl_m)

    # ------------------------------------------------------------------
    # Load data (standardised)
    # ------------------------------------------------------------------
    print(f"\n[exp1] Loading {dataset} ...")
    data = load_data(dataset, seed=seeds[0])

    # ------------------------------------------------------------------
    # Ground-truth PCA directions (from training data)
    # ------------------------------------------------------------------
    print(f"[exp1] Computing top-{embed_dim} PCA directions ...")
    pca_dirs = compute_pca_directions(data.X_train, embed_dim)
    print(f"[exp1] PCA directions shape: {pca_dirs.shape}")

    # ------------------------------------------------------------------
    # Train all models over all seeds
    # ------------------------------------------------------------------
    # all_histories[tag] = list of history dicts (one per seed)
    # all_metrics[tag]   = list of metric dicts (one per seed)
    all_histories = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_metrics   = {mc["tag"]: [] for mc in MODEL_CONFIGS}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[exp1] === Seed {seed} ({seed_idx+1}/{len(seeds)}) ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Re-load data with this seed's split
        seed_data = load_data(dataset, seed=seed)

        trained = train_all_models(seed_data, cfg, run_dir, seed, standard_mrl_m, device)

        for mc in MODEL_CONFIGS:
            tag = mc["tag"]
            model, history = trained[tag]
            all_histories[tag].append(history)

            metrics = compute_all_prefix_metrics(model, pca_dirs)
            all_metrics[tag].append(metrics)
            print(f"  [{tag}] mean subspace angle @ m={embed_dim}: "
                  f"{metrics['subspace_angles'][-1]:.2f} deg  "
                  f"col_align: {metrics['column_alignments'][-1]:.3f}")

    # ------------------------------------------------------------------
    # Save raw data (metrics + histories + PCA dirs) for re-plotting
    # ------------------------------------------------------------------
    save_raw_data(all_metrics, all_histories, pca_dirs, run_dir)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[exp1] Generating plots ...")
    plot_training_curves(all_histories, run_dir, fig_stamp)
    plot_prefix_metrics(all_metrics, run_dir, fig_stamp, embed_dim, standard_mrl_m)

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    save_results_summary(all_metrics, run_dir, embed_dim)

    # ------------------------------------------------------------------
    # Runtime + code snapshot
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)

    print(f"\n[exp1] Done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
