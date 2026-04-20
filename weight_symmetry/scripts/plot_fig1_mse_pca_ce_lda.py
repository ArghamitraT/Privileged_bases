"""
Fig 1 — Loss-Family Comparison: Cosine Similarity to PCA and LDA
-----------------------------------------------------------------
2 rows × 3 columns:
  Row 0: cosine similarity to PCA  (x up to N_PLOT_PCA = 20)
  Row 1: cosine similarity to LDA  (x up to N_PLOT_LDA =  5)
  Col 0: LAE / MSE-loss models  + eigenvalue bars (dual y-axis)
  Col 1: CE-loss models
  Col 2: Fisher / DeepLDA models  (pending models shown as placeholder)

Eigenvalue bars (col 0 only) loaded from:
  ICMLWorkshop_weightSymmetry2026/eigenvalues/
  → run compute_eigenvalues.py first if those files do not exist.

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/plot_fig1_mse_pca_ce_lda.py \\
        --run exprmnt_2026_04_19__16_00_49/exprmnt_2026_04_20__13_46_30

    python weight_symmetry/scripts/plot_fig1_mse_pca_ce_lda.py \\
        --run exprmnt_2026_04_19__16_00_49/exprmnt_2026_04_20__13_46_30 \\
        --fisher-run exprmnt_2026_04_20__01_31_36
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_PROJ_ROOT  = os.path.dirname(_CODE_ROOT)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from weight_symmetry.plotting.style import apply_style, save_fig

DEFAULT_RESULTS_ROOT    = os.path.join(_PROJ_ROOT, "files", "results")
DEFAULT_FISHER_RUN      = "exprmnt_2026_04_20__01_31_36"   # fisher baseline
DEFAULT_FP_FISHER_RUN   = "exprmnt_2026_04_20__01_44_24"   # fp_fisher (better run)
DEFAULT_EXTRA_FISHER_RUN= "exprmnt_2026_04_20__11_33_48"   # std_mrl_fisher + prefix_l1_fisher
DEFAULT_EIG_DIR         = os.path.join(
    DEFAULT_RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "eigenvalues"
)

# ---------------------------------------------------------------------------
# x-axis limits
# ---------------------------------------------------------------------------
N_PLOT_PCA = 20
N_PLOT_LDA = 5

# ---------------------------------------------------------------------------
# Bar overlay colors (separate for PCA vs LDA rows)
# ---------------------------------------------------------------------------
PCA_BAR_COLOR = "#89C4E1"   # faded blue  — PCA eigenvalue bars (row 0)
LDA_BAR_COLOR = "#F0A500"   # amber/gold  — LDA eigenvalue bars (row 1)

# ---------------------------------------------------------------------------
# Model configs: (tag, label, color, linestyle, lw, pending)
# Shared colors across all three columns so one legend covers all panels.
# Baseline = grey dashed; MRL = orange; FP MRL = green; PrefixL1 = mauve solid
# ---------------------------------------------------------------------------
MSE_MODELS = [
    ("mse_lae",          "Unordered",     "#888888", "--",  1.0, False),
    ("std_mrl_mse",      "MRL",           "#E07B00", "-",   1.8, False),
    ("fp_mrl_mse_ortho", "FP MRL",        "#009E73", "-",   1.8, False),
    ("prefix_l1_mse",    "PrefixL1",      "#CC79A7", "-",   1.8, False),
    ("nonuniform_l2",    "NonUniform L2", "#56B4E9", "-",   1.8, False),
]

CE_MODELS = [
    ("normal_ce",    "Unordered",    "#888888", "--",  1.0, False),
    ("std_mrl_ce",   "MRL",          "#E07B00", "-",   1.8, False),
    ("fp_mrl_ce",    "FP MRL",       "#009E73", "-",   1.8, False),
    ("prefix_l1_ce", "PrefixL1",     "#CC79A7", "-",   1.8, False),
]

FISHER_MODELS = [
    ("fisher",           "Unordered",    "#888888", "--",  1.0, False),
    ("fp_fisher",        "FP MRL",       "#009E73", "-",   1.8, False),
    ("std_mrl_fisher",   "MRL",          "#E07B00", "-",   1.8, False),
    ("prefix_l1_fisher", "PrefixL1",     "#CC79A7", "-",   1.8, False),
]

# Shared legend — placed once in axes[0, 2] (top-right panel)
LEGEND_HANDLES = [
    Line2D([0], [0], color="#888888", ls="--", lw=1.0, label="Unordered"),
    Line2D([0], [0], color="#E07B00", ls="-",  lw=1.8, label="MRL"),
    Line2D([0], [0], color="#009E73", ls="-",  lw=1.8, label="FP MRL"),
    Line2D([0], [0], color="#CC79A7", ls="-",  lw=1.8, label="PrefixL1"),
    Line2D([0], [0], color="#56B4E9", ls="-",  lw=1.8, label="NonUniform L2"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_eig(eig_dir):
    """Load pre-computed eigenvalue arrays."""
    pca_norm = np.load(os.path.join(eig_dir, "pca_eigenvalues_norm.npy"))
    lda_evr  = np.load(os.path.join(eig_dir, "lda_eigenvalues.npy"))
    return pca_norm, lda_evr


def _plot_panel(ax, axr, models, data, metric_suffix, n_plot,
                bg_vals=None, bar_color=None, bar_ylabel=None):
    """
    Draw one panel.  Returns list of (Line2D handle, label) for pending models.

    axr        : twin y-axis for eigenvalue bars; None for cols 1 and 2.
    bg_vals    : normalised bar heights (col 0 only), clipped to n_plot.
    bar_color  : color for eigenvalue bars.
    bar_ylabel : right-axis label for eigenvalue bars.
    """
    xs = np.arange(1, n_plot + 1)
    pending_handles = []

    # Background eigenvalue bars (col 0 only)
    if axr is not None and bg_vals is not None:
        bc = bar_color or PCA_BAR_COLOR
        bg = bg_vals[:n_plot]
        axr.bar(xs, bg, color=bc, alpha=0.30, zorder=0, width=0.8)
        axr.set_ylabel(bar_ylabel or "Norm. eigenvalue", color=bc, fontsize=7)
        axr.tick_params(axis="y", labelcolor=bc, labelsize=6)
        axr.set_ylim(0, max(bg) * 1.5)

    for tag, label, color, ls, lw, pending in models:
        key = f"{tag}_{metric_suffix}"
        if key not in data:
            if pending:
                h = Line2D([0], [0], color="#BBBBBB", ls=":", lw=1.0)
                pending_handles.append((h, f"{label} (pending)"))
            else:
                print(f"  [warn] missing key: {key}")
            continue

        mat = data[key].astype(float)
        mat = np.where(mat < 0, np.nan, mat)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        actual = min(n_plot, mat.shape[1])   # Fisher embed_dim=19 < N_PLOT_PCA=20
        vals   = mat[:, :actual]
        xs_m   = np.arange(1, actual + 1)
        mean   = np.nanmean(vals, axis=0)
        std    = np.nanstd(vals,  axis=0)

        ax.plot(xs_m, mean, label=label, color=color, ls=ls, lw=lw, zorder=3)
        if mat.shape[0] > 1:
            ax.fill_between(xs_m, mean - std, mean + std,
                            alpha=0.15, color=color, zorder=2)

    ax.set_xlim(0, n_plot + 0.5)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray", zorder=1)
    if axr is not None:
        ax.set_zorder(axr.get_zorder() + 1)
        ax.patch.set_visible(False)

    return pending_handles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Fig 1: 2×3 cosine similarity grid")
    parser.add_argument("--run", required=True,
                        help="Sub-run folder for LAE/CE data (contains metrics_raw.npz), "
                             "e.g. exprmnt_.../exprmnt_...")
    parser.add_argument("--fisher-run",       default=DEFAULT_FISHER_RUN,
                        help="Folder with 'fisher' baseline metrics")
    parser.add_argument("--fp-fisher-run",    default=DEFAULT_FP_FISHER_RUN,
                        help="Folder with 'fp_fisher' metrics")
    parser.add_argument("--extra-fisher-run", default=DEFAULT_EXTRA_FISHER_RUN,
                        help="Folder with std_mrl_fisher + prefix_l1_fisher metrics")
    parser.add_argument("--eig-dir",          default=None,
                        help="Directory containing pca/lda eigenvalue .npy files")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    results_root = args.results_root or DEFAULT_RESULTS_ROOT
    eig_dir      = args.eig_dir or DEFAULT_EIG_DIR

    run_dir  = os.path.join(results_root, args.run)
    npz_path = os.path.join(run_dir, "metrics_raw.npz")

    fisher_npzs = [
        os.path.join(results_root, args.fisher_run,       "metrics_raw.npz"),
        os.path.join(results_root, args.fp_fisher_run,    "metrics_raw.npz"),
        os.path.join(results_root, args.extra_fisher_run, "metrics_raw.npz"),
    ]

    for p in [npz_path] + fisher_npzs + [
        os.path.join(eig_dir, "pca_eigenvalues_norm.npy"),
        os.path.join(eig_dir, "lda_eigenvalues.npy"),
    ]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found: {p}\n"
                     "       Run compute_eigenvalues.py first if eigenvalue files are missing.")

    data = np.load(npz_path)

    # Merge all three Fisher npz files into one dict (later files override on key clash)
    fisher_data = {}
    for p in fisher_npzs:
        fisher_data.update(dict(np.load(p)))
        print(f"[fig1] Fisher data : {p}")

    pca_norm, lda_evr = _load_eig(eig_dir)
    print(f"[fig1] LAE/CE data : {npz_path}")
    print(f"[fig1] PCA eigs    : {pca_norm.shape}  LDA eigs: {lda_evr.shape}")

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    apply_style()
    mpl.rcParams.update({
        "axes.titlesize":  8,
        "axes.labelsize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))

    # (row, col, models, data_dict, metric_suffix, n_plot, bg_vals, bar_color, bar_ylabel)
    panels = [
        (0, 0, MSE_MODELS,    data,        "pca_cosine", N_PLOT_PCA, pca_norm, PCA_BAR_COLOR, "PCA eigenvectors"),
        (0, 1, CE_MODELS,     data,        "pca_cosine", N_PLOT_PCA, None,     None,          None),
        (0, 2, FISHER_MODELS, fisher_data, "pca_cosine", N_PLOT_PCA, None,     None,          None),
        (1, 0, MSE_MODELS,    data,        "lda_cosine", N_PLOT_LDA, lda_evr,  LDA_BAR_COLOR, "LDA eigenvectors"),
        (1, 1, CE_MODELS,     data,        "lda_cosine", N_PLOT_LDA, None,     None,          None),
        (1, 2, FISHER_MODELS, fisher_data, "lda_cosine", N_PLOT_LDA, None,     None,          None),
    ]

    col_titles  = ["MSE models", "CE models", "Fisher models"]
    row_ylabels = ["Cosine sim. to PCA", "Cosine sim. to LDA"]

    for row, col, models, d, msuffix, nplot, bg, bc, bylabel in panels:
        ax  = axes[row, col]
        axr = ax.twinx() if bg is not None else None

        _plot_panel(ax, axr, models, d, msuffix, nplot, bg, bc, bylabel)

        if row == 1:
            ax.set_xlabel("Prefix size $k$", fontsize=8)
        if col == 0:
            ax.set_ylabel(row_ylabels[row], fontsize=8)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8)

    # Single shared legend in top-right panel (row 0, col 2)
    axes[0, 2].legend(handles=LEGEND_HANDLES, loc="upper right",
                      frameon=True, fontsize=6, handlelength=1.8, borderpad=0.6)

    fig.tight_layout(h_pad=1.0, w_pad=0.6)

    stem = save_fig(fig, "fig1_cosine_pca_lda_grid", fig_stamp)
    plt.close(fig)

    print(
        f"\nReminder: update figure registry in plans/ICMLWorkshop_figure_style_plan.md\n"
        f"  Stem            : {stem}\n"
        f"  LAE/CE run      : {args.run}\n"
        f"  Fisher run      : {args.fisher_run}\n"
        f"  Date            : {time.strftime('%Y-%m-%d')}"
    )


if __name__ == "__main__":
    main()
