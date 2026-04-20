"""
Kurtosis analysis script
-------------------------
Post-hoc kurtosis analysis of one or more saved experiment result folders.
No new training — reads checkpoints and dataset config from each folder.

For each folder:
  - Scans for seed*_best.pt checkpoints to discover which models are present
  - Reloads the dataset using config.json
  - Computes absolute (Pearson) kurtosis per latent dimension for each model/seed
  - Produces kurtosis bar chart and per-dimension profile plot

Results go in a kurtosis_<timestamp>/ subfolder inside each input folder:
    exprmnt_.../
    └── kurtosis_<timestamp>/
        ├── kurtosis_bar_{stamp}.png
        ├── kurtosis_profile_{stamp}.png
        ├── kurtosis_raw.npz
        ├── results_summary.txt
        ├── experiment_description.log
        ├── runtime.txt
        └── code_snapshot/

Usage:
    Conda environment: mrl_env_cuda12  (GPU)
                       mrl_env         (CPU fallback)

    # Single folder (relative name or full path)
    python weight_symmetry/scripts/kurtosis_dropoff.py --dirs exprmnt_2026_04_16__13_01_30

    # Multiple folders
    python weight_symmetry/scripts/kurtosis_dropoff.py --dirs exprmnt_AAA exprmnt_BBB

    # Subfolders (e.g. --use-weights reruns inside a parent)
    python weight_symmetry/scripts/kurtosis_dropoff.py --dirs exprmnt_AAA/exprmnt_BBB

    # Smoke test
    python weight_symmetry/scripts/kurtosis_dropoff.py --dirs exprmnt_... --fast
"""

import os
import sys
import re
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
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads

# Cycle of colours for auto-assigned plot styles (no hardcoded model list)
_COLOUR_CYCLE = [
    "#0072B2", "#009E73", "#E69F00", "#CC79A7",
    "#D55E00", "#56B4E9", "#F0E442", "gray", "black",
]


# ==============================================================================
# Resolve a folder name to an absolute path
# ==============================================================================

def resolve_dir(raw: str) -> str:
    if os.path.isabs(raw):
        if os.path.isdir(raw):
            return raw
        raise FileNotFoundError(f"Not a directory: {raw}")
    from weight_symmetry.utility import get_path
    results_root = get_path("files/results")
    for candidate in [
        os.path.join(results_root, raw),
        os.path.join(results_root, "test_runs", raw),
        raw,  # relative to cwd
    ]:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find directory '{raw}'. Tried under files/results/, "
        f"files/results/test_runs/, and cwd."
    )


# ==============================================================================
# Discover checkpoints in a folder → list of (seed, tag) pairs
# ==============================================================================

def discover_checkpoints(folder: str):
    """
    Scan folder for seed*_best.pt files.
    Returns sorted list of (seed: int, tag: str).
    Pattern: seed{N}_{tag}_best.pt
    """
    pattern = re.compile(r"^seed(\d+)_(.+)_best\.pt$")
    found = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            found.append((int(m.group(1)), m.group(2)))
    found.sort()
    return found


# ==============================================================================
# Data loading — from config.json
# ==============================================================================

def load_data_from_cfg(saved_cfg: dict):
    dataset  = saved_cfg["dataset"]
    seed     = saved_cfg["seeds"][0]

    if dataset == "synthetic":
        variant = saved_cfg.get("synthetic_variant", "nonOrderedLDA")
        return load_data(dataset, seed=seed, synthetic_variant=variant)
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
# Load model — try LinearAEWithHeads first, fall back to LinearAE
# ==============================================================================

def load_model(ckpt_path: str, input_dim: int, embed_dim: int,
               n_classes: int) -> torch.nn.Module:
    """
    Try to load as LinearAEWithHeads (has classification heads).
    Fall back to LinearAE if the state dict doesn't match.
    Both expose get_encoder_matrix().
    """
    for cls, kwargs in [
        (LinearAEWithHeads, dict(input_dim=input_dim, embed_dim=embed_dim, n_classes=n_classes)),
        (LinearAE,          dict(input_dim=input_dim, embed_dim=embed_dim)),
    ]:
        try:
            model = cls(**kwargs)
            model.load_state_dict(
                torch.load(ckpt_path, weights_only=True, map_location="cpu")
            )
            model.eval()
            return model
        except Exception:
            continue
    raise RuntimeError(f"Could not load checkpoint {ckpt_path} as any known model type.")


# ==============================================================================
# Kurtosis computation
# ==============================================================================

def compute_kurtosis(model, X_test_np: np.ndarray) -> np.ndarray:
    """
    Absolute (Pearson) kurtosis per latent dimension (Gaussian baseline = 3).
    Z = X_test @ B.T  where B = encoder weight matrix (d, p).
    Returns kappa: (d,)
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    B = model.get_encoder_matrix().cpu().numpy().astype(np.float64)  # (d, p)
    Z = X_test_np.astype(np.float64) @ B.T                           # (n_test, d)
    return np.array([scipy_kurtosis(Z[:, k], fisher=False) for k in range(Z.shape[1])])


# ==============================================================================
# Auto-assign plot styles from tag names
# ==============================================================================

def build_plot_styles(tags: list) -> dict:
    styles = {}
    for i, tag in enumerate(tags):
        styles[tag] = dict(
            color=_COLOUR_CYCLE[i % len(_COLOUR_CYCLE)],
            linestyle="-",
            marker="",
        )
    return styles


# ==============================================================================
# Plots
# ==============================================================================

def plot_kurtosis_bar(tags, labels, kappa_means, kappa_stds,
                      plot_styles, run_dir, fig_stamp):
    means  = [kappa_means[t] for t in tags]
    stds   = [kappa_stds[t]  for t in tags]
    colors = [plot_styles[t]["color"] for t in tags]

    fig, ax = plt.subplots(figsize=(max(6, len(tags) * 1.2), 5))
    x = np.arange(len(tags))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5, width=0.55)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.2,
               label="Gaussian baseline (κ=3)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean kurtosis κ̄  (absolute, Gaussian=3)")
    ax.set_title("Mean latent kurtosis by model")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"kurtosis_bar{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[kurtosis] Saved {path}")


def plot_kurtosis_profile(tags, labels, kappa_profiles, kappa_profiles_std,
                          plot_styles, embed_dim, run_dir, fig_stamp):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(1, embed_dim + 1))
    for tag, label in zip(tags, labels):
        style = plot_styles[tag]
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
    print(f"[kurtosis] Saved {path}")


# ==============================================================================
# Results summary
# ==============================================================================

def save_results_summary(tags, labels, kappa_means, kappa_stds, embed_dim, run_dir):
    lines = [
        "=" * 70,
        "Kurtosis Analysis",
        "=" * 70, "",
        "Mean kurtosis κ̄ (absolute, Gaussian=3)",
        f"  {'Model':<35}  {'mean κ̄':>10}  {'std':>8}",
        "  " + "-" * 57,
    ]
    for tag, label in zip(tags, labels):
        lines.append(
            f"  {label:<35}  {kappa_means[tag]:>10.3f}  {kappa_stds[tag]:>8.3f}"
        )
    lines += [
        "",
        "κ̄ = mean over seeds of (1/d) sum_k kurtosis_k  (Pearson, Gaussian=3)",
    ]
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[kurtosis] Saved {path}")


def save_experiment_description(source_dir, saved_cfg, tags, run_dir):
    lines = [
        "Kurtosis Analysis",
        "=" * 60, "",
        f"Source folder: {source_dir}",
        f"Models found:  {tags}",
        "",
        "Method:",
        "  Absolute kurtosis (Pearson, Gaussian=3) per latent dim.",
        "  Z = X_test @ B.T  (B = encoder weight matrix).",
        "  Averaged over seeds.",
        "",
        "Source config:",
    ] + [f"  {k}: {v}" for k, v in saved_cfg.items()]
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[kurtosis] Saved {path}")


# ==============================================================================
# Per-folder analysis
# ==============================================================================

def run_kurtosis_for_folder(source_dir: str, fast: bool):
    t_start = time.time()

    cfg_path = os.path.join(source_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"[kurtosis] WARNING: no config.json in {source_dir} — skipping")
        return

    with open(cfg_path) as f:
        saved_cfg = json.load(f)

    embed_dim = saved_cfg["embed_dim"]
    seeds     = saved_cfg["seeds"]

    # Discover which (seed, tag) checkpoints exist
    ckpts = discover_checkpoints(source_dir)
    if not ckpts:
        print(f"[kurtosis] WARNING: no seed*_best.pt found in {source_dir} — skipping")
        return

    tags_ordered = []
    seen = set()
    for _, tag in ckpts:
        if tag not in seen:
            tags_ordered.append(tag)
            seen.add(tag)

    # Only process seeds listed in config (skip stray checkpoints)
    seeds_set    = set(seeds)
    ckpts_active = [(s, t) for s, t in ckpts if s in seeds_set]

    print(f"\n[kurtosis] Folder : {source_dir}")
    print(f"[kurtosis] Models : {tags_ordered}")
    print(f"[kurtosis] Seeds  : {seeds}")

    # Output subfolder
    stamp     = time.strftime("kurtosis_%Y_%m_%d__%H_%M_%S")
    run_dir   = os.path.join(source_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"[kurtosis] Output : {run_dir}")

    # Load dataset once (test set used for all seeds/models)
    print(f"[kurtosis] Loading dataset '{saved_cfg['dataset']}' ...")
    data      = load_data_from_cfg(saved_cfg)
    X_test_np = data.X_test.numpy().astype(np.float32)
    n_classes = data.n_classes
    input_dim = data.input_dim
    print(f"[kurtosis] X_test: {X_test_np.shape}  n_classes={n_classes}  embed_dim={embed_dim}")

    # Compute kurtosis per (seed, tag)
    kappa_all = {tag: [] for tag in tags_ordered}

    for seed, tag in ckpts_active:
        ckpt = os.path.join(source_dir, f"seed{seed}_{tag}_best.pt")
        print(f"[kurtosis]   seed={seed}  tag={tag}")
        try:
            model = load_model(ckpt, input_dim, embed_dim, n_classes)
            kappa = compute_kurtosis(model, X_test_np)
            kappa_all[tag].append(kappa)
        except Exception as e:
            print(f"  [WARNING] failed ({e}) — skipping")

    # Aggregate
    kappa_means        = {}
    kappa_stds         = {}
    kappa_profiles     = {}
    kappa_profiles_std = {}

    for tag in tags_ordered:
        if not kappa_all[tag]:
            kappa_means[tag]        = float("nan")
            kappa_stds[tag]         = float("nan")
            kappa_profiles[tag]     = np.full(embed_dim, float("nan"))
            kappa_profiles_std[tag] = np.full(embed_dim, float("nan"))
            continue
        mat = np.array(kappa_all[tag])        # (n_seeds, d)
        bar = mat.mean(axis=1)                # (n_seeds,)
        kappa_means[tag]        = float(bar.mean())
        kappa_stds[tag]         = float(bar.std())
        kappa_profiles[tag]     = mat.mean(axis=0)
        kappa_profiles_std[tag] = mat.std(axis=0)
        print(f"  [{tag}]  κ̄ = {kappa_means[tag]:.3f} ± {kappa_stds[tag]:.3f}")

    # Use tag names as labels (no separate label mapping needed)
    labels      = tags_ordered
    plot_styles = build_plot_styles(tags_ordered)

    # Plots
    print("[kurtosis] Generating plots ...")
    plot_kurtosis_bar(tags_ordered, labels, kappa_means, kappa_stds,
                      plot_styles, run_dir, fig_stamp)
    plot_kurtosis_profile(tags_ordered, labels, kappa_profiles, kappa_profiles_std,
                          plot_styles, embed_dim, run_dir, fig_stamp)

    # Save raw data
    kd = {"embed_dim": np.array(embed_dim)}
    for tag in tags_ordered:
        if kappa_all[tag]:
            kd[f"{tag}_kappa_all"] = np.array(kappa_all[tag])
    np.savez(os.path.join(run_dir, "kurtosis_raw.npz"), **kd)
    print(f"[kurtosis] Saved kurtosis_raw.npz")

    save_results_summary(tags_ordered, labels, kappa_means, kappa_stds, embed_dim, run_dir)
    save_experiment_description(source_dir, saved_cfg, tags_ordered, run_dir)

    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"[kurtosis] Done. Results in: {run_dir}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kurtosis analysis for experiment folders")
    parser.add_argument(
        "--dirs", nargs="+", required=True, metavar="FOLDER",
        help="One or more experiment result folders (relative names or full paths)",
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test mode (no effect on computation, skips code snapshot)")
    args = parser.parse_args()

    resolved = []
    for raw in args.dirs:
        try:
            resolved.append(resolve_dir(raw))
        except FileNotFoundError as e:
            print(f"[kurtosis] ERROR: {e}")

    if not resolved:
        print("[kurtosis] No valid directories found. Exiting.")
        return

    for folder in resolved:
        run_kurtosis_for_folder(folder, fast=args.fast)


if __name__ == "__main__":
    main()
