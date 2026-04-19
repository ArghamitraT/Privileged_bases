"""
Fig 2 — PCA Recovery: Cosine Similarity + Subspace Angle (Exp 1)

Two-panel figure per plans/figure_design_spec.md:
  Panel 1 (left):  Mean max cosine similarity (left axis) + eigenvalue spectrum bars (right axis)
  Panel 2 (right): Mean principal angle (left axis) + eigenvalue gap bars (right axis)

Three models: LAE, MRL, Full prefix MRL.

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/plot_fig2_col_alignment.py \\
        --run exprmnt_2026_04_16__20_54_49
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from weight_symmetry.plotting.style import apply_style, save_fig

DEFAULT_RESULTS_ROOT = os.path.abspath(
    os.path.join(_CODE_ROOT, "..", "files", "results")
)

# ---------------------------------------------------------------------------
# Model selection — 3 models per spec
# data key, legend label, line color, linestyle
# ---------------------------------------------------------------------------
MODELS = [
    ("mse_lae",              "LAE",             "#888888", "--"),
    ("standard_mrl",         "MRL",             "#E07B00", "-"),
    ("fullprefix_mrl_ortho", "Full prefix MRL", "#009E73", "-"),
]

# Bar colors
EIG_SPECTRUM_COLOR = "#89C4E1"   # faded blue for eigenvalue spectrum
EIG_GAP_COLOR      = "#9B59B6"   # purple for eigenvalue gap (distinct from all line colors)


def _parse_eigenvalue_file(path):
    """Parse eigenvalue_gaps.txt → (lambdas, gaps_norm) arrays of length d."""
    lambdas, gaps_norm = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[0].isdigit():
                lambdas.append(float(parts[1]))
                gn = parts[3]
                gaps_norm.append(float(gn) if gn != "nan" else 0.0)
    return np.array(lambdas), np.array(gaps_norm)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 2: PCA recovery two-panel")
    parser.add_argument("--run", required=True,
                        help="Run folder name, e.g. exprmnt_2026_04_16__20_54_49")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    results_root = args.results_root or DEFAULT_RESULTS_ROOT
    run_dir  = os.path.join(results_root, args.run)
    npz_path = os.path.join(run_dir, "metrics_raw.npz")
    gap_path = os.path.join(run_dir, "eigenvalue_gaps.txt")

    for p in (npz_path, gap_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")

    data = np.load(npz_path)
    lambdas, gaps_norm = _parse_eigenvalue_file(gap_path)
    d = len(lambdas)
    prefix_sizes = np.arange(1, d + 1)

    lam_norm = lambdas / lambdas[0]          # normalised spectrum

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")

    apply_style()
    mpl.rcParams.update({
        "font.size":       9,
        "axes.titlesize":  9,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    # One-column figure: ~3.5 in wide, two panels side by side
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # ------------------------------------------------------------------
    # Panel 1 (left) — Cosine similarity + eigenvalue spectrum
    # ------------------------------------------------------------------
    axLr = axL.twinx()

    axLr.bar(prefix_sizes, lam_norm, color=EIG_SPECTRUM_COLOR, alpha=0.35,
             zorder=0, width=0.8)
    axLr.set_ylabel("Normalised eigenvalue", color=EIG_SPECTRUM_COLOR)
    axLr.tick_params(axis="y", labelcolor=EIG_SPECTRUM_COLOR)
    axLr.set_ylim(0, 1.6)

    for key, label, color, ls in MODELS:
        aligns = data[f"{key}_column_alignments"]
        mean = aligns.mean(axis=0)
        std  = aligns.std(axis=0)
        axL.plot(prefix_sizes, mean, label=label, color=color, ls=ls, lw=1.8, zorder=3)
        if aligns.shape[0] > 1:
            axL.fill_between(prefix_sizes, mean - std, mean + std,
                             alpha=0.15, color=color, zorder=2)

    axL.set_xlabel("Prefix size $m$")
    axL.set_ylabel("Cosine similarity")
    axL.set_xlim(1, d)
    axL.set_ylim(0, 1.05)
    axL.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray", zorder=1)
    axL.set_zorder(axLr.get_zorder() + 1)
    axL.patch.set_visible(False)

    # ------------------------------------------------------------------
    # Panel 2 (right) — Principal angle + eigenvalue gap
    # ------------------------------------------------------------------
    axRr = axR.twinx()

    # Cap y-axis at 0.12 so small-gap detail at higher dims is visible.
    # Dims 2–3 have large gaps (0.35, 0.40) that are real but would
    # otherwise squash all other variation — bars are clipped, not wrong.
    GAP_YMAX = 0.12
    axRr.bar(prefix_sizes, np.clip(gaps_norm, 0, GAP_YMAX),
             color=EIG_GAP_COLOR, alpha=0.30, zorder=0, width=0.8)
    axRr.set_ylabel("Normalised eigenvalue gap", color=EIG_GAP_COLOR)
    axRr.tick_params(axis="y", labelcolor=EIG_GAP_COLOR)
    axRr.set_ylim(0, GAP_YMAX)

    for key, label, color, ls in MODELS:
        angles = data[f"{key}_subspace_angles"]
        mean = angles.mean(axis=0)
        std  = angles.std(axis=0)
        axR.plot(prefix_sizes, mean, label=label, color=color, ls=ls, lw=1.8, zorder=3)
        if angles.shape[0] > 1:
            axR.fill_between(prefix_sizes, mean - std, mean + std,
                             alpha=0.15, color=color, zorder=2)

    axR.set_xlabel("Prefix size $m$")
    axR.set_ylabel("Principal angle (°)")
    axR.set_xlim(1, d)
    axR.set_ylim(bottom=0)
    axR.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray", zorder=1)
    axR.set_zorder(axRr.get_zorder() + 1)
    axR.patch.set_visible(False)
    axR.legend(loc="upper right", frameon=True)


    fig.tight_layout()

    stem = save_fig(fig, "exp1_pca_recovery", fig_stamp)
    plt.close(fig)

    print(
        f"\nReminder: update figure registry in plans/ICMLWorkshop_figure_style_plan.md\n"
        f"  Generated file stem : {stem}\n"
        f"  Source run          : {args.run}\n"
        f"  Date                : {time.strftime('%Y-%m-%d')}"
    )


if __name__ == "__main__":
    main()
