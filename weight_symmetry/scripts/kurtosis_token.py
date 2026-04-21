"""
Token-wise kurtosis analysis script
-------------------------------------
Post-hoc kurtosis analysis of one or more saved experiment result folders.
No new training — reads checkpoints and dataset config from each folder.

Theoretical basis:
  If a model has NO privileged basis, features are represented in random directions.
  A random direction in high-d space is (up to rescaling) a vector of i.i.d. Gaussian
  samples. So for a single token's activation vector z ∈ R^d, treating the d values as
  i.i.d. samples, we expect kurtosis ≈ 3 (Gaussian).
  If the model HAS a privileged basis (sparse features align with axes), most dimensions
  will be near-zero for any given token and a few will be large → heavy tails → kurtosis > 3.

Computation:
  For each token i: kurtosis_i = Pearson kurtosis of Z[i, :] over dimensions (d values).
  Report: mean and std of kurtosis_i over all test tokens, averaged over seeds.

Contrast with kurtosis_dropoff.py:
  That script computes kurtosis of Z[:, k] over tokens for each dimension k.
  This script computes kurtosis of Z[i, :] over dimensions for each token i.

For each folder:
  - Scans for seed*_best.pt checkpoints
  - Reloads the dataset using config.json
  - Computes per-token kurtosis for each model/seed
  - Produces bar chart and distribution (KDE) plot

Results go in a kurtosis_token_<timestamp>/ subfolder inside each input folder:
    exprmnt_.../
    └── kurtosis_token_<timestamp>/
        ├── kurtosis_token_bar_{stamp}.png
        ├── kurtosis_token_dist_{stamp}.png
        ├── kurtosis_token_raw.npz
        ├── results_summary.txt
        ├── experiment_description.log
        ├── runtime.txt
        └── code_snapshot/

Usage:
    Conda environment: mrl_env_cuda12  (GPU)
                       mrl_env         (CPU fallback)

    # Single folder (relative name or full path)
    python weight_symmetry/scripts/kurtosis_token.py --dirs exprmnt_2026_04_16__13_01_30

    # Multiple folders
    python weight_symmetry/scripts/kurtosis_token.py --dirs exprmnt_AAA exprmnt_BBB

    # Subfolders (e.g. --use-weights reruns inside a parent)
    python weight_symmetry/scripts/kurtosis_token.py --dirs exprmnt_AAA/exprmnt_BBB

    # Smoke test
    python weight_symmetry/scripts/kurtosis_token.py --dirs exprmnt_... --fast
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
        raw,
    ]:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find directory '{raw}'. Tried under files/results/, "
        f"files/results/test_runs/, and cwd."
    )


# ==============================================================================
# Discover checkpoints
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
    dataset = saved_cfg["dataset"]
    seed    = saved_cfg["seeds"][0]

    if dataset in ("synthetic", "mnist", "fashion_mnist", "digits"):
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
# Load model
# ==============================================================================

def load_model(ckpt_path: str, input_dim: int, embed_dim: int,
               n_classes: int) -> torch.nn.Module:
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
# Kurtosis computation — per token, over dimensions
# ==============================================================================

def compute_kurtosis_token(model, X_test_np: np.ndarray) -> np.ndarray:
    """
    For each test token i, compute Pearson kurtosis of Z[i, :] over the d dimensions.
    Returns per_token_kurtosis: (n_test,)

    Theory: if no privileged basis, features are in random (Gaussian-like) directions,
    so Z[i, :] should look like i.i.d. Gaussian samples → kurtosis ≈ 3.
    Privileged basis + sparse features → kurtosis > 3.
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    B = model.get_encoder_matrix().cpu().numpy().astype(np.float64)  # (d, p)
    Z = X_test_np.astype(np.float64) @ B.T                           # (n_test, d)
    return np.array([scipy_kurtosis(Z[i, :], fisher=False) for i in range(Z.shape[0])])


# ==============================================================================
# Plot styles
# ==============================================================================

def build_plot_styles(tags: list) -> dict:
    styles = {}
    for i, tag in enumerate(tags):
        styles[tag] = dict(color=_COLOUR_CYCLE[i % len(_COLOUR_CYCLE)])
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
    ax.set_ylabel("Mean token kurtosis κ̄  (absolute, Gaussian=3)")
    ax.set_title("Mean per-token kurtosis by model\n"
                 "(kurtosis of each token's activation vector over dimensions)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"kurtosis_token_bar{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[kurtosis_token] Saved {path}")


def plot_kurtosis_distribution(tags, labels, kappa_distributions,
                               plot_styles, run_dir, fig_stamp):
    """
    KDE plot showing the distribution of per-token kurtosis values across the test set.
    Each model gets one curve (averaged over seeds).
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("[kurtosis_token] scipy not available for KDE — skipping distribution plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for tag, label in zip(tags, labels):
        vals = kappa_distributions[tag]
        if vals is None or len(vals) == 0:
            continue
        color = plot_styles[tag]["color"]
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            x_min = max(0, np.percentile(vals, 1))
            x_max = np.percentile(vals, 99)
            xs = np.linspace(x_min, x_max, 400)
            ax.plot(xs, kde(xs), label=label, color=color, linewidth=1.8)
            ax.axvline(np.mean(vals), color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        except Exception as e:
            print(f"[kurtosis_token] KDE failed for {tag}: {e}")

    ax.axvline(3.0, color="red", linestyle="--", linewidth=1.2, label="Gaussian (κ=3)")
    ax.set_xlabel("Per-token kurtosis κ  (absolute, Gaussian=3)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of per-token kurtosis across test set\n"
                 "(kurtosis of each token's activation vector over dimensions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, f"kurtosis_token_dist{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[kurtosis_token] Saved {path}")


# ==============================================================================
# Results summary
# ==============================================================================

def save_results_summary(tags, labels, kappa_means, kappa_stds, run_dir):
    lines = [
        "=" * 70,
        "Token-wise Kurtosis Analysis",
        "=" * 70, "",
        "Mean per-token kurtosis κ̄ (absolute, Gaussian=3)",
        "  Kurtosis computed over dimensions for each token, then averaged over",
        "  all test tokens and seeds.",
        "",
        f"  {'Model':<35}  {'mean κ̄':>10}  {'std':>8}",
        "  " + "-" * 57,
    ]
    for tag, label in zip(tags, labels):
        lines.append(
            f"  {label:<35}  {kappa_means[tag]:>10.3f}  {kappa_stds[tag]:>8.3f}"
        )
    lines += [
        "",
        "κ̄ = mean over seeds of (mean per-token kurtosis over test set)",
        "std = std over seeds of that per-seed mean",
        "",
        "Interpretation:",
        "  κ̄ ≈ 3  → activations look Gaussian across dims (consistent with no privileged basis)",
        "  κ̄ > 3  → heavy-tailed activation distribution (consistent with privileged basis)",
    ]
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[kurtosis_token] Saved {path}")


def save_experiment_description(source_dir, saved_cfg, tags, run_dir):
    lines = [
        "Token-wise Kurtosis Analysis",
        "=" * 60, "",
        f"Source folder: {source_dir}",
        f"Models found:  {tags}",
        "",
        "Method:",
        "  For each test token i: kurtosis of Z[i, :] over d dimensions.",
        "  Z = X_test @ B.T  (B = encoder weight matrix, shape d x p).",
        "  Mean and std computed over n_test tokens, then averaged over seeds.",
        "",
        "  Contrast with kurtosis_dropoff.py which computes kurtosis of Z[:, k]",
        "  (over tokens) for each dimension k.",
        "",
        "Theoretical basis:",
        "  No privileged basis → features in random Gaussian-like directions",
        "  → Z[i, :] ≈ i.i.d. Gaussian samples → kurtosis ≈ 3.",
        "  Privileged basis + sparse features → kurtosis > 3.",
        "",
        "Source config:",
    ] + [f"  {k}: {v}" for k, v in saved_cfg.items()]
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[kurtosis_token] Saved {path}")


# ==============================================================================
# Per-folder analysis
# ==============================================================================

def run_kurtosis_token_for_folder(source_dir: str, fast: bool, split: str = "test"):
    t_start = time.time()

    cfg_path = os.path.join(source_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"[kurtosis_token] WARNING: no config.json in {source_dir} — skipping")
        return

    with open(cfg_path) as f:
        saved_cfg = json.load(f)

    seeds = saved_cfg["seeds"]

    ckpts = discover_checkpoints(source_dir)
    if not ckpts:
        print(f"[kurtosis_token] WARNING: no seed*_best.pt found in {source_dir} — skipping")
        return

    tags_ordered = []
    seen = set()
    for _, tag in ckpts:
        if tag not in seen:
            tags_ordered.append(tag)
            seen.add(tag)

    if "embed_dim" in saved_cfg:
        embed_dim = saved_cfg["embed_dim"]
    else:
        first_ckpt = os.path.join(source_dir, f"seed{ckpts[0][0]}_{ckpts[0][1]}_best.pt")
        sd = torch.load(first_ckpt, weights_only=True, map_location="cpu")
        embed_dim = sd["encoder.weight"].shape[0]
        print(f"[kurtosis_token] embed_dim inferred from checkpoint: {embed_dim}")

    seeds_set    = set(seeds)
    ckpts_active = [(s, t) for s, t in ckpts if s in seeds_set]

    print(f"\n[kurtosis_token] Folder : {source_dir}")
    print(f"[kurtosis_token] Models : {tags_ordered}")
    print(f"[kurtosis_token] Seeds  : {seeds}")

    split_tag = f"kurtosis_token_{split}_" if split != "test" else "kurtosis_token_"
    stamp     = time.strftime(f"{split_tag}%Y_%m_%d__%H_%M_%S")
    run_dir   = os.path.join(source_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"[kurtosis_token] Output : {run_dir}")

    print(f"[kurtosis_token] Loading dataset '{saved_cfg['dataset']}' ...")
    data      = load_data_from_cfg(saved_cfg)
    if split == "train":
        X_test_np = data.X_train.numpy().astype(np.float32)
    else:
        X_test_np = data.X_test.numpy().astype(np.float32)
    n_classes = data.n_classes
    input_dim = data.input_dim
    print(f"[kurtosis_token] X_{split}: {X_test_np.shape}  n_classes={n_classes}  embed_dim={embed_dim}  split={split}")

    # kappa_per_seed[tag] = list of (n_test,) arrays, one per seed
    kappa_per_seed = {tag: [] for tag in tags_ordered}

    for seed, tag in ckpts_active:
        ckpt = os.path.join(source_dir, f"seed{seed}_{tag}_best.pt")
        print(f"[kurtosis_token]   seed={seed}  tag={tag}")
        try:
            model = load_model(ckpt, input_dim, embed_dim, n_classes)
            kappa = compute_kurtosis_token(model, X_test_np)  # (n_test,)
            kappa_per_seed[tag].append(kappa)
        except Exception as e:
            print(f"  [WARNING] failed ({e}) — skipping")

    # Aggregate: mean/std over seeds of per-token mean kurtosis
    kappa_means         = {}
    kappa_stds          = {}
    kappa_distributions = {}  # for KDE: concatenate per-token values across seeds

    for tag in tags_ordered:
        seed_arrays = kappa_per_seed[tag]
        if not seed_arrays:
            kappa_means[tag]         = float("nan")
            kappa_stds[tag]          = float("nan")
            kappa_distributions[tag] = None
            continue

        # mean kurtosis per seed (scalar per seed)
        per_seed_means = np.array([arr.mean() for arr in seed_arrays])
        kappa_means[tag] = float(per_seed_means.mean())
        kappa_stds[tag]  = float(per_seed_means.std())

        # for KDE: use first seed's full distribution (representative)
        kappa_distributions[tag] = seed_arrays[0]

        print(f"  [{tag}]  κ̄ = {kappa_means[tag]:.3f} ± {kappa_stds[tag]:.3f}")

    labels      = tags_ordered
    plot_styles = build_plot_styles(tags_ordered)

    print("[kurtosis_token] Generating plots ...")
    plot_kurtosis_bar(tags_ordered, labels, kappa_means, kappa_stds,
                      plot_styles, run_dir, fig_stamp)
    plot_kurtosis_distribution(tags_ordered, labels, kappa_distributions,
                               plot_styles, run_dir, fig_stamp)

    # Save raw data
    kd = {}
    for tag in tags_ordered:
        if kappa_per_seed[tag]:
            kd[f"{tag}_kappa_per_seed"] = np.array(kappa_per_seed[tag])  # (n_seeds, n_test)
    np.savez(os.path.join(run_dir, "kurtosis_token_raw.npz"), **kd)
    print(f"[kurtosis_token] Saved kurtosis_token_raw.npz")

    save_results_summary(tags_ordered, labels, kappa_means, kappa_stds, run_dir)
    save_experiment_description(source_dir, saved_cfg, tags_ordered, run_dir)

    elapsed = time.time() - t_start
    save_runtime(run_dir, elapsed)
    save_code_snapshot(run_dir)
    print(f"[kurtosis_token] Done. Results in: {run_dir}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Token-wise kurtosis analysis for experiment folders"
    )
    parser.add_argument(
        "--dirs", nargs="+", required=True, metavar="FOLDER",
        help="One or more experiment result folders (relative names or full paths)",
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test mode (no effect on computation, skips code snapshot)")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Which data split to compute kurtosis on (default: test)")
    args = parser.parse_args()

    resolved = []
    for raw in args.dirs:
        try:
            resolved.append(resolve_dir(raw))
        except FileNotFoundError as e:
            print(f"[kurtosis_token] ERROR: {e}")

    if not resolved:
        print("[kurtosis_token] No valid directories found. Exiting.")
        return

    for folder in resolved:
        run_kurtosis_token_for_folder(folder, fast=args.fast, split=args.split)


if __name__ == "__main__":
    main()
