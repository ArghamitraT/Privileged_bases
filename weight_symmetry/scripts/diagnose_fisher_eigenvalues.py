"""
Script: weight_symmetry/scripts/diagnose_fisher_eigenvalues.py
--------------------------------------------------------------
Diagnostic for exp2_fisher_loss: checks whether FisherLoss collapses
discriminative variance into the top eigenvalue (trivial solution problem,
DeepLDA paper Eq. 8 warning) or spreads it across all C-1 dimensions.

For each saved model (fisher, fp_fisher) across seeds:
  1. Load encoder B from saved checkpoint
  2. Compute z = Bx on training set
  3. Compute eigenvalues of (S_W + εI)^{-1} S_B on z
  4. Report % variance explained by top-1, top-3, top-5 dims
  5. Compare to LDA baseline eigenvalues on raw x

Plots:
  - Eigenvalue spectra per model (all seeds overlaid)
  - Cumulative variance curves
  - Bar chart: % variance in top-1 / top-3 / top-5 per model

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/diagnose_fisher_eigenvalues.py \\
        --weights-dir exprmnt_2026_04_19__20_02_18

    python weight_symmetry/scripts/diagnose_fisher_eigenvalues.py \\
        --weights-dir /absolute/path/to/exprmnt_...
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_WS_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.data.loader import load_data
from weight_symmetry.data.synthetic import SYNTHETIC_VARIANTS
from weight_symmetry.utility import get_path

MODEL_CONFIGS = [
    dict(tag="fisher",    label="FisherLoss (standard)", color="#0072B2", ls="--"),
    dict(tag="fp_fisher", label="FullPrefixFisher",      color="#009E73", ls="-"),
]

EPS = 1e-4


# ==============================================================================
# Core computation
# ==============================================================================

def compute_fisher_eigenvalues(z: np.ndarray, y: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Eigenvalues of (S_W + εI)^{-1} S_B computed on embeddings z.
    Returns eigenvalues in descending order.
    """
    z      = z.astype(np.float64)
    n      = z.shape[0]
    d      = z.shape[1]
    classes   = np.unique(y)
    mean_all  = z.mean(0)

    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))
    for c in classes:
        mask = (y == c)
        n_c  = mask.sum()
        z_c  = z[mask]
        mu_c = z_c.mean(0)
        diff = (mu_c - mean_all).reshape(-1, 1)
        S_B += (n_c / n) * (diff @ diff.T)
        z_cc = z_c - mu_c
        S_W += (z_cc.T @ z_cc) / n

    reg     = eps * np.eye(d)
    eigvals = np.linalg.eigvalsh(np.linalg.solve(S_W + reg, S_B))
    return np.sort(eigvals)[::-1]   # descending


def compute_lda_baseline_eigenvalues(data) -> np.ndarray:
    """LDA explained_variance_ratio on raw x (sklearn)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(data.X_train.numpy(), data.y_train.numpy())
    return lda.explained_variance_ratio_.astype(np.float64)


def summarise(eigvals: np.ndarray, label: str):
    ev_pos = np.maximum(eigvals, 0.0)
    total  = ev_pos.sum() + 1e-12
    cumvar = np.cumsum(ev_pos) / total
    top1   = 100 * ev_pos[0] / total
    top3   = 100 * ev_pos[:3].sum() / total
    top5   = 100 * ev_pos[:5].sum() / total
    k90    = int(np.searchsorted(cumvar, 0.90)) + 1

    print(f"\n  {label}")
    top10_str = "  ".join(f"{v:.3f}" for v in eigvals[:10])
    rest_str  = "  ".join(f"{v:.3f}" for v in eigvals[10:])
    print(f"    Eigenvalues [1-10]:  {top10_str}")
    if len(eigvals) > 10:
        print(f"    Eigenvalues [11-19]: {rest_str}")
    print(f"    Top-1  : {top1:.1f}%  |  Top-3: {top3:.1f}%  |  Top-5: {top5:.1f}%")
    print(f"    Dims needed for 90% cumulative variance: {k90}")
    return dict(top1=top1, top3=top3, top5=top5, k90=k90, cumvar=cumvar, eigvals=eigvals)


# ==============================================================================
# Plots
# ==============================================================================

def plot_diagnostics(all_eigvals, lda_evr, embed_dim, out_dir, fig_stamp):
    """
    Three-panel figure:
      Left  — eigenvalue spectra (all seeds per model)
      Middle — cumulative variance curves
      Right  — bar chart: % in top-1 / top-3 / top-5
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    xs = np.arange(1, embed_dim + 1)

    # ── Panel 1: eigenvalue spectra ─────────────────────────────────────────
    ax = axes[0]
    for mc in MODEL_CONFIGS:
        tag, label, color, ls = mc["tag"], mc["label"], mc["color"], mc["ls"]
        mat = np.array(all_eigvals[tag])             # (n_seeds, embed_dim)
        mat_pos = np.maximum(mat, 0.0)
        # Normalise each seed by its total so spectra are comparable
        norm = mat_pos / (mat_pos.sum(axis=1, keepdims=True) + 1e-12)
        mean, std = norm.mean(0), norm.std(0)
        ax.plot(xs, mean, label=label, color=color, linestyle=ls, linewidth=1.5)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)

    # LDA baseline
    ax.plot(xs, lda_evr[:embed_dim], label="LDA baseline (on raw x)",
            color="black", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Eigenvalue rank k")
    ax.set_ylabel("Normalised eigenvalue (fraction of total)")
    ax.set_title("Fisher eigenvalue spectrum\n(S_W + εI)^{-1} S_B on learned z")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: cumulative variance ─────────────────────────────────────────
    ax = axes[1]
    for mc in MODEL_CONFIGS:
        tag, label, color, ls = mc["tag"], mc["label"], mc["color"], mc["ls"]
        mat = np.array(all_eigvals[tag])
        mat_pos = np.maximum(mat, 0.0)
        norm = mat_pos / (mat_pos.sum(axis=1, keepdims=True) + 1e-12)
        cumv = np.cumsum(norm, axis=1)
        mean, std = cumv.mean(0), cumv.std(0)
        ax.plot(xs, mean, label=label, color=color, linestyle=ls, linewidth=1.5)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)

    lda_cumv = np.cumsum(lda_evr[:embed_dim])
    ax.plot(xs, lda_cumv, label="LDA baseline", color="black", linestyle=":", linewidth=1.5)
    ax.axhline(0.90, color="grey", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Number of dimensions k")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Cumulative Fisher variance\n(ideal: diagonal, each dim adds equal share)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: top-1 / top-3 / top-5 bar chart ────────────────────────────
    ax = axes[2]
    model_labels = [mc["label"] for mc in MODEL_CONFIGS] + ["LDA baseline"]
    colors       = [mc["color"] for mc in MODEL_CONFIGS] + ["black"]
    width        = 0.22
    x_pos        = np.arange(len(model_labels))

    # Compute per-model stats
    pct_vals = {k: [] for k in ["top1", "top3", "top5"]}
    for mc in MODEL_CONFIGS:
        tag = mc["tag"]
        mat = np.array(all_eigvals[tag])
        mat_pos = np.maximum(mat, 0.0)
        total   = mat_pos.sum(axis=1, keepdims=True) + 1e-12
        pct_vals["top1"].append(100 * mat_pos[:, 0] / total.squeeze())
        pct_vals["top3"].append(100 * mat_pos[:, :3].sum(1) / total.squeeze())
        pct_vals["top5"].append(100 * mat_pos[:, :5].sum(1) / total.squeeze())

    # LDA baseline
    lda_pos   = np.maximum(lda_evr[:embed_dim], 0.0)
    lda_total = lda_pos.sum() + 1e-12
    pct_vals["top1"].append(np.array([100 * lda_pos[0] / lda_total]))
    pct_vals["top3"].append(np.array([100 * lda_pos[:3].sum() / lda_total]))
    pct_vals["top5"].append(np.array([100 * lda_pos[:5].sum() / lda_total]))

    bar_styles = [
        ("top1", "Top-1",  0.0,    "///"),
        ("top3", "Top-3",  width,   ""),
        ("top5", "Top-5",  2*width, ".."),
    ]
    for key, blabel, offset, hatch in bar_styles:
        means = [v.mean() for v in pct_vals[key]]
        stds  = [v.std()  for v in pct_vals[key]]
        ax.bar(x_pos + offset, means, width=width, yerr=stds,
               label=blabel, hatch=hatch, alpha=0.75, capsize=4,
               color=[c for c in colors], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("% of total Fisher variance")
    ax.set_title("Variance collapse check\n(high top-1 = trivial solution problem)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, f"fisher_eigenvalue_diagnosis{fig_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[diagnose_fisher] Saved {path}")


def save_text_summary(all_stats, lda_stats, out_dir):
    lines = [
        "=" * 70,
        "Fisher eigenvalue spread diagnostic",
        "DeepLDA trivial-solution check: does top-1 dominate?",
        "=" * 70, "",
        f"{'Model':<28}  {'Seed':>5}  {'Top-1%':>8}  {'Top-3%':>8}  {'Top-5%':>8}  {'k@90%':>6}",
        "-" * 70,
    ]
    for mc in MODEL_CONFIGS:
        tag, label = mc["tag"], mc["label"]
        for i, s in enumerate(all_stats[tag]):
            lines.append(
                f"{label:<28}  {s['seed']:>5}  "
                f"{s['top1']:>7.1f}%  {s['top3']:>7.1f}%  "
                f"{s['top5']:>7.1f}%  {s['k90']:>6}"
            )
    lines += [
        "",
        f"{'LDA baseline (raw x)':<28}  {'—':>5}  "
        f"{lda_stats['top1']:>7.1f}%  {lda_stats['top3']:>7.1f}%  "
        f"{lda_stats['top5']:>7.1f}%  {lda_stats['k90']:>6}",
        "",
        "Interpretation:",
        "  If Top-1% >> LDA baseline Top-1%  →  trivial solution (DeepLDA Eq. 8 problem)",
        "  If Top-1% ≈ LDA baseline            →  variance spread evenly, no collapse",
        "  k@90%: fewer dims = more collapsed; more dims = better spread",
    ]
    path = os.path.join(out_dir, "fisher_eigenvalue_diagnosis.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[diagnose_fisher] Saved {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fisher eigenvalue spread diagnostic (exp2)")
    parser.add_argument("--weights-dir", required=True, metavar="FOLDER",
                        help="Saved exp2_fisher_loss run folder")
    args = parser.parse_args()

    weights_dir = args.weights_dir
    if not os.path.isabs(weights_dir):
        weights_dir = os.path.join(get_path("files/results"), weights_dir)

    print(f"[diagnose_fisher] Loading from: {weights_dir}")
    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg = json.load(f)

    embed_dim = cfg["embed_dim"]
    seeds     = cfg["seeds"]
    variant   = cfg.get("synthetic_variant", "orderedBoth")
    dataset   = cfg["dataset"]

    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC: Fisher eigenvalue spread  —  (S_W + εI)^{{-1}} S_B on z=Bx")
    print(f"{'='*70}")

    all_eigvals = {mc["tag"]: [] for mc in MODEL_CONFIGS}
    all_stats   = {mc["tag"]: [] for mc in MODEL_CONFIGS}

    for seed in seeds:
        data    = load_data(dataset, seed=seed, synthetic_variant=variant)
        X_train = data.X_train.numpy()
        y_train = data.y_train.numpy()

        print(f"\nSeed {seed}")
        for mc in MODEL_CONFIGS:
            tag, label = mc["tag"], mc["label"]
            ckpt  = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
            model = LinearAE(data.input_dim, embed_dim)
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
            model.eval()
            with torch.no_grad():
                z = model.encode(data.X_train).numpy()

            eigvals = compute_fisher_eigenvalues(z, y_train)
            all_eigvals[tag].append(eigvals)
            stats = summarise(eigvals, label)
            stats["seed"] = seed
            all_stats[tag].append(stats)

    # LDA baseline on raw x
    print(f"\n{'='*70}")
    print("BASELINE: LDA eigenvalues on raw input x (sklearn explained_variance_ratio_)")
    data0   = load_data(dataset, seed=seeds[0], synthetic_variant=variant)
    lda_evr = compute_lda_baseline_eigenvalues(data0)
    lda_pos = np.maximum(lda_evr, 0.0)
    lda_total = lda_pos.sum() + 1e-12
    lda_cumv  = np.cumsum(lda_pos) / lda_total
    lda_stats = dict(
        top1=100 * lda_pos[0] / lda_total,
        top3=100 * lda_pos[:3].sum() / lda_total,
        top5=100 * lda_pos[:5].sum() / lda_total,
        k90 =int(np.searchsorted(lda_cumv, 0.90)) + 1,
    )
    print(f"  LDA baseline — Top-1: {lda_stats['top1']:.1f}%  "
          f"Top-3: {lda_stats['top3']:.1f}%  Top-5: {lda_stats['top5']:.1f}%  "
          f"k@90%: {lda_stats['k90']}")

    plot_diagnostics(all_eigvals, lda_evr, embed_dim, weights_dir, fig_stamp)
    save_text_summary(all_stats, lda_stats, weights_dir)
    print(f"\n[diagnose_fisher] Done. Outputs in: {weights_dir}")


if __name__ == "__main__":
    main()
