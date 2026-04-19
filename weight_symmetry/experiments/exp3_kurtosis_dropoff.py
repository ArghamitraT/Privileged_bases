"""
Experiment 3: Kurtosis + Drop-off Shape
-----------------------------------------
Uses saved checkpoints from Experiment 2. No new training.

3a — Kurtosis (latent geometry):
    For each model and each latent dimension k, compute absolute kurtosis
        kappa_k = E[(z_k - mu_k)^4] / E[(z_k - mu_k)^2]^2
    over the test set (Gaussian baseline = 3).
    Report mean kappa_bar = (1/d) sum_k kappa_k, std over seeds.

    Plots:
      - Bar chart: kappa_bar per model (mean ± std over seeds), dashed line at 3
      - Kurtosis profile: kappa_k vs dim index k, one line per model (mean ± std)

3b — Drop-off Shape (importance concentration):
    For each model, marginal prefix accuracy gain
        Delta_k = Acc(1:k) - Acc(1:k-1)   (k = 1..d, Acc(1:0) = 0)
    loaded from exp2's accuracies_raw.npz (no recomputation).
    Also plots "MSE LAE (sorted)" — MSE LAE's Delta_k sorted descending —
    as the post-hoc oracle, to test whether MRL's training-time ordering adds value.

    Plot: Delta_k vs rank k (log-scale y-axis), one line per model + sorted baseline.

Results go in a subfolder of the exp2 run directory:
    exprmnt_<exp2_stamp>/
    └── exp3_kurtosis_<timestamp>/
        ├── kurtosis_bar_{stamp}.png
        ├── kurtosis_profile_{stamp}.png
        ├── dropoff_{stamp}.png
        ├── results_summary.txt
        ├── experiment_description.log
        ├── runtime.txt
        └── code_snapshot/

Usage:
    Conda environment: mrl_env_cuda12  (GPU)
                       mrl_env         (CPU fallback)

    # Point at an exp2 run folder (relative name or full path)
    python weight_symmetry/experiments/exp3_kurtosis_dropoff.py --exp2-dir exprmnt_2026_04_16__13_01_30
    python weight_symmetry/experiments/exp3_kurtosis_dropoff.py --exp2-dir /full/path/to/exp2/run
    python weight_symmetry/experiments/exp3_kurtosis_dropoff.py --exp2-dir exprmnt_... --fast
"""

import os
import sys
import time
import json
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

from weight_symmetry.utility import save_runtime, save_code_snapshot
from weight_symmetry.data.loader import load_data, load_data_with_directions
from weight_symmetry.data.synthetic import load_synthetic
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads

# ==============================================================================
# CONFIG — override EXP2_DIR here or pass --exp2-dir on the command line
# ==============================================================================
EXP2_DIR = None   # e.g. "exprmnt_2026_04_16__13_01_30"  (relative or absolute)
# ==============================================================================

# Tags and display properties — must match exp2's MODEL_CONFIGS
MODEL_META = [
    dict(tag="mse_lae",    model_type="lae",       label="MSE LAE"),
    dict(tag="fp_mrl_mse", model_type="lae",       label="Full-prefix MRL (MSE)"),
    dict(tag="fp_mrl_ce",  model_type="lae_heads", label="Full-prefix MRL (CE)"),
    dict(tag="std_mrl_ce", model_type="lae_heads", label="Standard MRL (CE)"),
]

PLOT_STYLES = {
    "MSE LAE":               dict(color="gray",   linestyle="--", marker=""),
    "MSE LAE (sorted)":      dict(color="gray",   linestyle=":",  marker=""),
    "Full-prefix MRL (MSE)": dict(color="blue",   linestyle="-",  marker=""),
    "Full-prefix MRL (CE)":  dict(color="green",  linestyle="-",  marker=""),
    "Standard MRL (CE)":     dict(color="orange", linestyle="-",  marker="o", markersize=3),
}


# ==============================================================================
# Utility: resolve exp2 run dir path
# ==============================================================================

def resolve_exp2_dir(raw: str) -> str:
    if os.path.isabs(raw):
        return raw
    from weight_symmetry.utility import get_path
    candidate = os.path.join(get_path("files/results"), raw)
    if os.path.isdir(candidate):
        return candidate
    # Also try test_runs/
    candidate2 = os.path.join(get_path("files/results"), "test_runs", raw)
    if os.path.isdir(candidate2):
        return candidate2
    raise FileNotFoundError(
        f"Could not find exp2 run dir '{raw}'. "
        f"Tried:\n  {candidate}\n  {candidate2}"
    )


# ==============================================================================
# Data loading — mirrors exp2's --use-weights data loading
# ==============================================================================

def load_exp2_data(saved_cfg: dict):
    """
    Reload dataset using parameters from exp2's config.json.
    Returns DataSplit for the first seed (seed=saved_cfg['seeds'][0]).
    Kurtosis uses this single test split for all seeds' models.
    """
    dataset     = saved_cfg["dataset"]
    seed        = saved_cfg["seeds"][0]
    ordered_lda = saved_cfg.get("ordered_lda", False)

    if dataset == "synthetic":
        return load_data(dataset, seed=seed, ordered_lda=ordered_lda)
    else:
        p_svd   = saved_cfg.get("p_svd",              100)
        tfidf   = saved_cfg.get("tfidf_max_features",  10000)
        p_pca   = saved_cfg.get("p_pca_proj",          50)
        n_noise = saved_cfg.get("n_noise_dims",        25)
        sigma   = saved_cfg.get("sigma_noise",         5.0)
        data, _, _, _ = load_data_with_directions(
            dataset, seed=seed,
            p_svd=p_svd, tfidf_max_features=tfidf,
            p_pca_proj=p_pca, n_noise_dims=n_noise, sigma_noise=sigma,
        )
        return data


# ==============================================================================
# Load model checkpoint
# ==============================================================================

def load_model(ckpt_path: str, model_type: str, input_dim: int,
               embed_dim: int, n_classes: int) -> torch.nn.Module:
    if model_type == "lae_heads":
        model = LinearAEWithHeads(input_dim, embed_dim, n_classes)
    else:
        model = LinearAE(input_dim, embed_dim)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    return model


# ==============================================================================
# 3a — Kurtosis
# ==============================================================================

def compute_kurtosis(model, X_test_np: np.ndarray) -> np.ndarray:
    """
    Compute absolute kurtosis (Pearson, Gaussian=3) for each latent dimension.

    Args:
        model      : trained LinearAE / LinearAEWithHeads
        X_test_np  : (n_test, p) float32 numpy test data (centred)

    Returns:
        kappa : (d,) array of per-dimension kurtosis values
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    B = model.get_encoder_matrix().cpu().numpy().astype(np.float64)  # (d, p)
    Z = X_test_np.astype(np.float64) @ B.T  # (n_test, d)
    # fisher=False: absolute kurtosis (Gaussian = 3)
    kappa = np.array([scipy_kurtosis(Z[:, k], fisher=False) for k in range(Z.shape[1])])
    return kappa


def plot_kurtosis_bar(kappa_means: dict, kappa_stds: dict, run_dir: str, fig_stamp: str):
    """Bar chart of mean kappa_bar per model, ± std over seeds."""
    labels = [mm["label"] for mm in MODEL_META]
    means  = [kappa_means[mm["tag"]] for mm in MODEL_META]
    stds   = [kappa_stds[mm["tag"]]  for mm in MODEL_META]
    colors = [PLOT_STYLES[lbl]["color"] for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5, width=0.55)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.2, label="Gaussian baseline (κ=3)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean kurtosis κ̄  (absolute, Gaussian=3)")
    ax.set_title("Mean latent kurtosis by model")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"kurtosis_bar{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp3] Saved {path}")


def plot_kurtosis_profile(kappa_profiles: dict, kappa_profiles_std: dict,
                          run_dir: str, fig_stamp: str, embed_dim: int):
    """Per-dimension kurtosis κ_k vs dim index k."""
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(1, embed_dim + 1))
    for mm in MODEL_META:
        tag   = mm["tag"]
        label = mm["label"]
        style = PLOT_STYLES[label]
        mean  = kappa_profiles[tag]
        std   = kappa_profiles_std[tag]
        ax.plot(xs, mean, label=label, **style)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.12, color=style["color"])
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.0, label="Gaussian (κ=3)")
    ax.set_xlabel("Latent dimension index k")
    ax.set_ylabel("Kurtosis κ_k  (absolute)")
    ax.set_title("Per-dimension kurtosis profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"kurtosis_profile{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp3] Saved {path}")


# ==============================================================================
# 3b — Drop-off shape
# ==============================================================================

def compute_delta_k(acc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Delta_k = Acc(1:k) - Acc(1:k-1) for k=1..d.
    acc_matrix: (n_seeds, d)  — prefix accuracies already aligned 1-indexed
    Returns: (n_seeds, d)
    """
    acc_padded = np.concatenate([np.zeros((acc_matrix.shape[0], 1)), acc_matrix], axis=1)
    return np.diff(acc_padded, axis=1)  # (n_seeds, d)


def plot_dropoff(delta_means: dict, delta_stds: dict, delta_sorted_mse: np.ndarray,
                 run_dir: str, fig_stamp: str, embed_dim: int):
    """Delta_k vs rank k, log-scale y-axis."""
    xs  = list(range(1, embed_dim + 1))
    fig, ax = plt.subplots(figsize=(10, 5))

    for mm in MODEL_META:
        tag   = mm["tag"]
        label = mm["label"]
        style = PLOT_STYLES[label]
        mean  = delta_means[tag]
        std   = delta_stds[tag]
        # Clip to positive for log scale
        mean_pos = np.clip(mean, 1e-6, None)
        ax.plot(xs, mean_pos, label=label, **style)

    # Post-hoc sorted MSE LAE
    sorted_style = PLOT_STYLES["MSE LAE (sorted)"]
    ax.plot(xs, np.clip(delta_sorted_mse, 1e-6, None),
            label="MSE LAE (sorted)", **sorted_style)

    ax.set_yscale("log")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Marginal accuracy gain Δ_k  (log scale)")
    ax.set_title("Prefix accuracy drop-off: Δ_k = Acc(1:k) − Acc(1:k−1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(run_dir, f"dropoff{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[exp3] Saved {path}")


# ==============================================================================
# Results summary
# ==============================================================================

def save_results_summary(kappa_means: dict, kappa_stds: dict,
                         delta_means: dict, delta_stds: dict,
                         embed_dim: int, run_dir: str):
    lines = [
        "=" * 70,
        "Experiment 3: Kurtosis + Drop-off Shape",
        "=" * 70, "",
        "3a — Mean kurtosis κ̄ (absolute, Gaussian=3)",
        f"  {'Model':<30}  {'mean κ̄':>10}  {'std':>8}",
        "  " + "-" * 52,
    ]
    for mm in MODEL_META:
        tag = mm["tag"]
        lines.append(f"  {mm['label']:<30}  {kappa_means[tag]:>10.3f}  {kappa_stds[tag]:>8.3f}")

    key_ks = [1, 5, 10, 25, embed_dim // 2, embed_dim]
    key_ks = sorted(set(k for k in key_ks if 1 <= k <= embed_dim))

    lines += ["",
              "3b — Mean Delta_k at selected prefix sizes",
              f"  {'Model':<30}" + "".join(f"  k={k:>3}" for k in key_ks),
              "  " + "-" * (30 + 8 * len(key_ks))]
    for mm in MODEL_META:
        tag = mm["tag"]
        row = f"  {mm['label']:<30}"
        for k in key_ks:
            row += f"  {delta_means[tag][k-1]:>6.4f}"
        lines.append(row)

    lines += [
        "",
        "Kurtosis: mean over seeds of (1/d) sum_k kurtosis_k  (Pearson, Gaussian=3)",
        "Delta_k: mean over seeds of Acc(1:k) - Acc(1:k-1)",
    ]
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp3] Saved {path}")


def save_experiment_description(exp2_dir: str, saved_cfg: dict, run_dir: str):
    lines = [
        "Experiment 3: Kurtosis + Drop-off Shape",
        "=" * 60, "",
        "Source exp2 run:",
        f"  {exp2_dir}",
        "",
        "3a — Kurtosis",
        "  Absolute kurtosis (Pearson, Gaussian=3) per latent dim,",
        "  averaged over dim index k. Test set from first seed.",
        "",
        "3b — Drop-off",
        "  Delta_k = Acc(1:k) - Acc(1:k-1), loaded from accuracies_raw.npz.",
        "  'MSE LAE (sorted)' = MSE LAE Delta_k sorted descending (oracle ordering).",
        "",
        "Source exp2 config:",
    ] + [f"  {k}: {v}" for k, v in saved_cfg.items()]
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp3] Saved {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 3: Kurtosis + Drop-off")
    parser.add_argument("--exp2-dir", type=str, default=None,
                        help="Path (or folder name) of an exp2 run to analyse")
    parser.add_argument("--fast",     action="store_true",
                        help="Smoke test: skip data reload warning, minimal output")
    args = parser.parse_args()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Resolve exp2 run directory
    # ------------------------------------------------------------------
    raw_dir = args.exp2_dir or EXP2_DIR
    if raw_dir is None:
        parser.error("Provide --exp2-dir or set EXP2_DIR in the CONFIG block.")
    exp2_dir = resolve_exp2_dir(raw_dir)
    print(f"[exp3] Source exp2 dir: {exp2_dir}")

    # Load exp2 config
    with open(os.path.join(exp2_dir, "config.json")) as f:
        saved_cfg = json.load(f)

    embed_dim  = saved_cfg["embed_dim"]
    seeds      = saved_cfg["seeds"]
    dataset    = saved_cfg["dataset"]
    std_mrl_m  = saved_cfg["standard_mrl_m"]

    # ------------------------------------------------------------------
    # Create output subfolder inside exp2_dir
    # ------------------------------------------------------------------
    stamp     = time.strftime("exp3_kurtosis_%Y_%m_%d__%H_%M_%S")
    run_dir   = os.path.join(exp2_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"[exp3] Output dir: {run_dir}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp3] Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ------------------------------------------------------------------
    # 3a — Kurtosis: reload dataset (seed=seeds[0]) once for all models
    # ------------------------------------------------------------------
    print(f"\n[exp3] Reloading dataset '{dataset}' (seed={seeds[0]}) for kurtosis ...")
    data      = load_exp2_data(saved_cfg)
    X_test_np = data.X_test.numpy().astype(np.float32)   # (n_test, p)
    n_classes = data.n_classes
    input_dim = data.input_dim
    print(f"[exp3] X_test: {X_test_np.shape}  n_classes={n_classes}  embed_dim={embed_dim}")

    # kappa_all[tag] = list of (d,) arrays, one per seed
    kappa_all = {mm["tag"]: [] for mm in MODEL_META}

    for seed in seeds:
        print(f"[exp3] Kurtosis seed={seed} ...")
        for mm in MODEL_META:
            tag   = mm["tag"]
            ckpt  = os.path.join(exp2_dir, f"seed{seed}_{tag}_best.pt")
            if not os.path.exists(ckpt):
                print(f"  [WARNING] checkpoint not found: {ckpt}  — skipping")
                continue
            model = load_model(ckpt, mm["model_type"], input_dim, embed_dim, n_classes)
            kappa = compute_kurtosis(model, X_test_np)   # (d,)
            kappa_all[tag].append(kappa)

    # Aggregate: mean and std of kappa_bar (scalar) and per-dim profile (d,) across seeds
    kappa_means       = {}   # tag -> scalar (mean kappa_bar)
    kappa_stds        = {}   # tag -> scalar (std of kappa_bar)
    kappa_profiles    = {}   # tag -> (d,) mean per-dim
    kappa_profiles_std = {}  # tag -> (d,) std per-dim

    for mm in MODEL_META:
        tag = mm["tag"]
        if not kappa_all[tag]:
            kappa_means[tag]        = float("nan")
            kappa_stds[tag]         = float("nan")
            kappa_profiles[tag]     = np.full(embed_dim, float("nan"))
            kappa_profiles_std[tag] = np.full(embed_dim, float("nan"))
            continue
        mat = np.array(kappa_all[tag])           # (n_seeds, d)
        kappa_bar_per_seed = mat.mean(axis=1)    # (n_seeds,)
        kappa_means[tag]        = float(kappa_bar_per_seed.mean())
        kappa_stds[tag]         = float(kappa_bar_per_seed.std())
        kappa_profiles[tag]     = mat.mean(axis=0)
        kappa_profiles_std[tag] = mat.std(axis=0)
        print(f"  [{tag}]  κ̄ = {kappa_means[tag]:.3f} ± {kappa_stds[tag]:.3f}")

    # ------------------------------------------------------------------
    # 3b — Drop-off: load from accuracies_raw.npz
    # ------------------------------------------------------------------
    print("\n[exp3] Loading prefix accuracies from accuracies_raw.npz ...")
    acc_path = os.path.join(exp2_dir, "accuracies_raw.npz")
    if not os.path.exists(acc_path):
        print(f"  [WARNING] {acc_path} not found — skipping 3b")
        acc_npz = None
    else:
        acc_npz = np.load(acc_path)

    delta_means = {}
    delta_stds  = {}

    if acc_npz is not None:
        for mm in MODEL_META:
            tag = mm["tag"]
            key = f"{tag}_accuracies"
            if key not in acc_npz:
                print(f"  [WARNING] key '{key}' not in accuracies_raw.npz — skipping")
                delta_means[tag] = np.zeros(embed_dim)
                delta_stds[tag]  = np.zeros(embed_dim)
                continue
            acc_mat    = acc_npz[key]               # (n_seeds, d)
            delta_mat  = compute_delta_k(acc_mat)   # (n_seeds, d)
            delta_means[tag] = delta_mat.mean(axis=0)
            delta_stds[tag]  = delta_mat.std(axis=0)

        # Post-hoc sorted MSE LAE: sort mean Delta_k descending
        mse_delta_mean       = delta_means.get("mse_lae", np.zeros(embed_dim))
        delta_sorted_mse_lae = np.sort(mse_delta_mean)[::-1]
    else:
        for mm in MODEL_META:
            delta_means[mm["tag"]] = np.zeros(embed_dim)
            delta_stds[mm["tag"]]  = np.zeros(embed_dim)
        delta_sorted_mse_lae = np.zeros(embed_dim)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[exp3] Generating plots ...")
    plot_kurtosis_bar(kappa_means, kappa_stds, run_dir, fig_stamp)
    plot_kurtosis_profile(kappa_profiles, kappa_profiles_std, run_dir, fig_stamp, embed_dim)
    if acc_npz is not None:
        plot_dropoff(delta_means, delta_stds, delta_sorted_mse_lae, run_dir, fig_stamp, embed_dim)

    # ------------------------------------------------------------------
    # Save raw kurtosis data
    # ------------------------------------------------------------------
    kd = {}
    for mm in MODEL_META:
        tag = mm["tag"]
        if kappa_all[tag]:
            kd[f"{tag}_kappa_all"] = np.array(kappa_all[tag])  # (n_seeds, d)
    kd["embed_dim"] = np.array(embed_dim)
    np.savez(os.path.join(run_dir, "kurtosis_raw.npz"), **kd)
    print(f"[exp3] Saved kurtosis_raw.npz")

    # ------------------------------------------------------------------
    # Results summary + description
    # ------------------------------------------------------------------
    save_results_summary(kappa_means, kappa_stds, delta_means, delta_stds, embed_dim, run_dir)
    save_experiment_description(exp2_dir, saved_cfg, run_dir)

    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"\n[exp3] Done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
