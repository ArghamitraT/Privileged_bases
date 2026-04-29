"""
Script: weight_symmetry/scripts/plot_fig_losses_combined.py
-----------------------------------------------------------
Companion figure to plot_fig_combined.py.

  (a) Left  — 2x2 training-loss curves for the PCALDA models.
  (b) Right — 2x2 pairwise (dim-to-dim) cosine similarity to PCA/LDA,
              mirroring the combined figure's panel (a) but with the
              paired-cosine metric in place of max-cosine.

Panels in both (a) and (b):
    (1,1) Synthetic + MSE/PCA   — mse_lae, std_mrl_mse, fp_mrl_mse_ortho,
                                  prefix_l1_mse, nonuniform_l2
    (1,2) Synthetic + Fisher/LDA — fisher, fp_fisher, std_mrl_fisher, prefix_l1_fisher
    (2,1) Fashion-MNIST + MSE/PCA — same 5 as (1,1)
    (2,2) Fashion-MNIST + Fisher/LDA — same 4 as (1,2)

Synthetic-Fisher `_lda_paired` is not in the original `metrics_raw.npz`;
it is loaded from `lda_paired_offline.npz` in each folder (computed by
`scratch_compute_synth_fisher_paired.py`).

Conda environment: mrl_env

Usage:
    python weight_symmetry/scripts/plot_fig_losses_combined.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_PROJ_ROOT  = os.path.dirname(_CODE_ROOT)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from weight_symmetry.plotting.style import apply_style, save_fig

RESULTS_ROOT = os.path.join(_PROJ_ROOT, "files", "results")
EIG_DIR      = os.path.join(RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "eigenvalues")

# ── Folders (same as plot_fig_combined.py) ────────────────────────────────────
DEFAULT_SYNTH_PCA_RUN     = "exprmnt_2026_04_23__20_18_55"
DEFAULT_SYNTH_FISHER_A    = "exprmnt_2026_04_20__01_31_36"   # fisher (Unordered)
DEFAULT_SYNTH_FISHER_B    = "exprmnt_2026_04_20__01_44_24"   # fp_fisher
DEFAULT_SYNTH_FISHER_C    = "exprmnt_2026_04_20__11_33_48"   # std_mrl_fisher, prefix_l1_fisher
DEFAULT_FMNIST_PCA_RUN    = "exprmnt_2026_04_23__15_29_53"
DEFAULT_FMNIST_FISHER_RUN = "exprmnt_2026_04_21__07_47_27"

# ── Per-panel epoch caps for fig (a) ──────────────────────────────────────────
EPOCH_CAP = {
    "synth_pca":    450,
    "synth_fisher": 5000,
    "fmnist_pca":   200,
    "fmnist_fisher":1500,
}

# ── Shared style (identical to plot_fig_combined.py) ──────────────────────────
FP_MRL_COLOR    = "#009E73"
PREFIX_L1_COLOR = "#CC79A7"
MRL_COLOR       = "#E07B00"
LAE_COLOR       = "#888888"
L2_COLOR        = "#56B4E9"

PCA_BAR_COLOR   = "#C9AEED"
PCA_TEXT_COLOR  = "#7B5AB8"
LDA_BAR_COLOR   = "#F0A500"

FP_MRL_LABEL    = "FP-MRL"
PREFIX_L1_LABEL = r"MD-$\ell_1$"

# fig (a) loss-panel model lists: (tag, label, color, ls, lw, folder_key)
SYNTH_PCA_LOSS = [
    ("mse_lae",          "Unordered",     LAE_COLOR,       "--", 1.2, "synth_pca"),
    ("std_mrl_mse",      "S-MRL",         MRL_COLOR,       "-",  1.5, "synth_pca"),
    ("fp_mrl_mse_ortho", "FP-MRL",        FP_MRL_COLOR,    "-",  1.5, "synth_pca"),
    ("prefix_l1_mse",    PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.5, "synth_pca"),
    ("nonuniform_l2",    r"NU-$\ell_2$",  L2_COLOR,        "-",  1.5, "synth_pca"),
]
SYNTH_FISHER_LOSS = [
    ("fisher",           "Unordered",     LAE_COLOR,       "--", 1.2, "synth_fisher_a"),
    ("fp_fisher",        "FP-MRL",        FP_MRL_COLOR,    "-",  1.5, "synth_fisher_b"),
    ("std_mrl_fisher",   "S-MRL",         MRL_COLOR,       "-",  1.5, "synth_fisher_c"),
    ("prefix_l1_fisher", PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.5, "synth_fisher_c"),
]
FMNIST_PCA_LOSS = [
    ("mse_lae",          "Unordered",     LAE_COLOR,       "--", 1.2, "fmnist_pca"),
    ("std_mrl_mse",      "S-MRL",         MRL_COLOR,       "-",  1.5, "fmnist_pca"),
    ("fp_mrl_mse_ortho", "FP-MRL",        FP_MRL_COLOR,    "-",  1.5, "fmnist_pca"),
    ("prefix_l1_mse",    PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.5, "fmnist_pca"),
    ("nonuniform_l2",    r"NU-$\ell_2$",  L2_COLOR,        "-",  1.5, "fmnist_pca"),
]
FMNIST_FISHER_LOSS = [
    ("fisher",           "Unordered",     LAE_COLOR,       "--", 1.2, "fmnist_fisher"),
    ("fp_fisher",        "FP-MRL",        FP_MRL_COLOR,    "-",  1.5, "fmnist_fisher"),
    ("std_mrl_fisher",   "S-MRL",         MRL_COLOR,       "-",  1.5, "fmnist_fisher"),
    ("prefix_l1_fisher", PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.5, "fmnist_fisher"),
]

# fig (b) paired-cosine model lists (mirrors combined fig (a))
SYNTH_PCA_MODELS = [
    ("mse_lae",          "Unordered",     LAE_COLOR,       "--", 1.0),
    ("std_mrl_mse",      "S-MRL",         MRL_COLOR,       "-",  1.8),
    ("fp_mrl_mse_ortho", "FP-MRL",        FP_MRL_COLOR,    "-",  1.8),
    ("prefix_l1_mse",    PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.8),
    ("nonuniform_l2",    r"NU-$\ell_2$",  L2_COLOR,        "-",  1.8),
]
# fig (b) angle-to-PCA: FP-MRL and S-MRL only
ANGLE_PCA_MODELS = [
    ("std_mrl_mse",      "S-MRL", MRL_COLOR,    "-", 1.8),
    ("fp_mrl_mse_ortho", "FP-MRL", FP_MRL_COLOR, "-", 1.8),
]
SYNTH_FISHER_MODELS = [
    ("fisher",           "Unordered",     LAE_COLOR,       "--", 1.0),
    ("fp_fisher",        "FP-MRL",        FP_MRL_COLOR,    "-",  1.8),
    ("std_mrl_fisher",   "S-MRL",         MRL_COLOR,       "-",  1.8),
    ("prefix_l1_fisher", PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.8),
]
FMNIST_PCA_MODELS = [
    ("mse_lae",          "Unordered",     LAE_COLOR,       "--", 1.0),
    ("std_mrl_mse",      "S-MRL",         MRL_COLOR,       "-",  1.8),
    ("fp_mrl_mse_ortho", "FP-MRL",        FP_MRL_COLOR,    "-",  1.8),
    ("prefix_l1_mse",    PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.8),
    ("nonuniform_l2",    r"NU-$\ell_2$",  L2_COLOR,        "-",  1.8),
]
FMNIST_FISHER_MODELS = [
    ("fisher",           "Unordered",     LAE_COLOR,       "--", 1.0),
    ("fp_fisher",        "FP-MRL",        FP_MRL_COLOR,    "-",  1.8),
    ("std_mrl_fisher",   "S-MRL",         MRL_COLOR,       "-",  1.8),
    ("prefix_l1_fisher", PREFIX_L1_LABEL, PREFIX_L1_COLOR, "-",  1.8),
]

SHARED_LEGEND_HANDLES = [
    Line2D([0], [0], color=LAE_COLOR,       ls="--", lw=1.2, label="Unordered"),
    Line2D([0], [0], color=MRL_COLOR,       ls="-",  lw=1.5, label="S-MRL"),
    Line2D([0], [0], color=FP_MRL_COLOR,    ls="-",  lw=1.5, label="FP-MRL"),
    Line2D([0], [0], color=PREFIX_L1_COLOR, ls="-",  lw=1.5, label=PREFIX_L1_LABEL),
    Line2D([0], [0], color=L2_COLOR,        ls="-",  lw=1.5, label=r"NU-$\ell_2$"),
]

# Legend for fig (b): same as (a) since paired-cosine panels use all 5 models;
# the angle column uses only S-MRL and FP-MRL — both are already in the list.
SHARED_LEGEND_B = SHARED_LEGEND_HANDLES


# ==============================================================================
# Loaders + helpers
# ==============================================================================

def _load_hist(folder):
    return dict(np.load(os.path.join(folder, "histories_raw.npz"),
                        allow_pickle=True))


def _load_metrics(folder):
    return dict(np.load(os.path.join(folder, "metrics_raw.npz"),
                        allow_pickle=True))


def _load_if_exists(path):
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    print(f"  [warn] missing file: {path}")
    return {}


def _norm_by_max(v):
    m = v.max()
    return v / m if m > 0 else v


def _load_synth_eigs():
    pca = np.load(os.path.join(EIG_DIR, "pca_eigenvalues.npy"),
                  allow_pickle=True).astype(float)
    lda = np.load(os.path.join(EIG_DIR, "lda_eigenvalues.npy"),
                  allow_pickle=True).astype(float)
    return _norm_by_max(pca), _norm_by_max(lda)


def _load_fmnist_eigs(fmnist_run_dir, fmnist_fisher_dir):
    pca = np.load(os.path.join(fmnist_run_dir, "pca_eigenvalues.npy"),
                  allow_pickle=True).astype(float)
    lda = np.load(os.path.join(fmnist_fisher_dir, "lda_eigenvalues_norm.npy"),
                  allow_pickle=True).astype(float)
    return _norm_by_max(pca), _norm_by_max(lda)


def _compact_log_minor(val, pos):
    """Format log-axis minor ticks as '6e-1' instead of '6×10⁻¹'."""
    if val <= 0:
        return ""
    exp  = int(np.floor(np.log10(val)))
    mant = int(round(val / (10 ** exp)))
    if mant in (2, 3, 4, 6):
        return f"{mant}e{exp}"
    return ""


# ==============================================================================
# Panel drawers
# ==============================================================================

def _panel_losses(ax, models, histories, epoch_cap):
    all_positive = True
    for tag, label, color, ls, lw, folder_key in models:
        key  = f"{tag}_train_losses"
        hist = histories.get(folder_key)
        if hist is None or key not in hist:
            print(f"  [warn] missing {key} in {folder_key}")
            continue
        arr = hist[key].astype(float)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        n    = min(len(arr), epoch_cap)
        xs   = np.arange(1, n + 1)
        vals = arr[:n]
        if np.any(vals <= 0):
            all_positive = False
        ax.plot(xs, vals, color=color, ls=ls, lw=lw, label=label)
    if all_positive:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        if ymin > 0 and np.log10(ymax / ymin) < 1.0:
            ax.yaxis.set_minor_formatter(FuncFormatter(_compact_log_minor))
            ax.tick_params(axis="y", which="minor", labelsize=6)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")


def _panel_paired(ax, models, data, metric_suffix, n_plot):
    for tag, label, color, ls, lw in models:
        key = f"{tag}_{metric_suffix}"
        if key not in data:
            print(f"  [warn] missing key: {key}")
            continue
        mat = data[key].astype(float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        mat    = np.where(mat < 0, np.nan, mat)
        actual = min(n_plot, mat.shape[1])
        xs_m   = np.arange(1, actual + 1)
        mean   = np.nanmean(mat[:, :actual], axis=0)
        std    = np.nanstd(mat[:, :actual],  axis=0)
        # Thinner strokes to de-clutter the busy paired panels.
        ax.plot(xs_m, mean, label=label, color=color, ls=ls, lw=lw * 0.65, zorder=3)
        if mat.shape[0] > 1:
            ax.fill_between(xs_m, mean - std, mean + std,
                            alpha=0.15, color=color, zorder=2)

    ax.set_xlim(0, n_plot + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray", zorder=1)


def _panel_angle(ax, models, data, n_plot):
    for tag, label, color, ls, lw in models:
        key = f"{tag}_pca_angles"
        if key not in data:
            print(f"  [warn] missing key: {key}")
            continue
        mat = data[key].astype(float)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        actual = min(n_plot, mat.shape[1])
        xs_m   = np.arange(1, actual + 1)
        mean   = np.nanmean(mat[:, :actual], axis=0)
        std    = np.nanstd(mat[:, :actual],  axis=0)
        ax.plot(xs_m, mean, label=label, color=color, ls=ls, lw=lw * 0.85, zorder=3)
        if mat.shape[0] > 1:
            ax.fill_between(xs_m, mean - std, mean + std,
                            alpha=0.15, color=color, zorder=2)

    ax.set_xlim(0, n_plot + 0.5)
    ax.set_ylim(0, 90)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray", zorder=1)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Losses + paired-cosine combined figure")
    parser.add_argument("--synth-pca-run",      default=DEFAULT_SYNTH_PCA_RUN)
    parser.add_argument("--synth-fisher-a",     default=DEFAULT_SYNTH_FISHER_A)
    parser.add_argument("--synth-fisher-b",     default=DEFAULT_SYNTH_FISHER_B)
    parser.add_argument("--synth-fisher-c",     default=DEFAULT_SYNTH_FISHER_C)
    parser.add_argument("--fmnist-pca-run",     default=DEFAULT_FMNIST_PCA_RUN)
    parser.add_argument("--fmnist-fisher-run",  default=DEFAULT_FMNIST_FISHER_RUN)
    args = parser.parse_args()

    t0 = time.time()

    def _abs(rel):
        return os.path.join(RESULTS_ROOT, rel) if not os.path.isabs(rel) else rel

    synth_pca_dir      = _abs(args.synth_pca_run)
    synth_fisher_a_dir = _abs(args.synth_fisher_a)
    synth_fisher_b_dir = _abs(args.synth_fisher_b)
    synth_fisher_c_dir = _abs(args.synth_fisher_c)
    fmnist_pca_dir     = _abs(args.fmnist_pca_run)
    fmnist_fisher_dir  = _abs(args.fmnist_fisher_run)

    print("[losses] Loading histories …")
    histories = {
        "synth_pca":       _load_hist(synth_pca_dir),
        "synth_fisher_a":  _load_hist(synth_fisher_a_dir),
        "synth_fisher_b":  _load_hist(synth_fisher_b_dir),
        "synth_fisher_c":  _load_hist(synth_fisher_c_dir),
        "fmnist_pca":      _load_hist(fmnist_pca_dir),
        "fmnist_fisher":   _load_hist(fmnist_fisher_dir),
    }

    print("[paired] Loading metrics + offline synth-Fisher paired …")
    synth_data    = _load_metrics(synth_pca_dir)
    fmnist_data   = _load_metrics(fmnist_pca_dir)
    fmnist_fisher = _load_metrics(fmnist_fisher_dir)

    synth_fisher = {}
    synth_fisher.update(_load_metrics(synth_fisher_a_dir))
    synth_fisher.update(_load_metrics(synth_fisher_b_dir))
    synth_fisher.update(_load_metrics(synth_fisher_c_dir))
    # Offline-computed `_lda_paired` lives in a sibling npz in each folder.
    synth_fisher.update(_load_if_exists(os.path.join(synth_fisher_a_dir, "lda_paired_offline.npz")))
    synth_fisher.update(_load_if_exists(os.path.join(synth_fisher_b_dir, "lda_paired_offline.npz")))
    synth_fisher.update(_load_if_exists(os.path.join(synth_fisher_c_dir, "lda_paired_offline.npz")))

    synth_pca_eig,  synth_lda_eig  = _load_synth_eigs()
    fmnist_pca_eig, fmnist_lda_eig = _load_fmnist_eigs(fmnist_pca_dir, fmnist_fisher_dir)

    print("[fig] Rendering …")
    apply_style()

    fig = plt.figure(figsize=(13.5, 5.2))
    # Match plot_fig_combined.py; (a) shifted slightly right so the dataset
    # row label has clearance from the y-axis title.
    gs_a = fig.add_gridspec(2, 2, left=0.07, right=0.43,
                             wspace=0.34, hspace=0.52, top=0.91, bottom=0.20)
    gs_b = fig.add_gridspec(2, 3, left=0.52, right=0.99,
                             wspace=0.38, hspace=0.52, top=0.91, bottom=0.20)

    axes_a = [[fig.add_subplot(gs_a[r, c]) for c in range(2)] for r in range(2)]
    axes_b = [[fig.add_subplot(gs_b[r, c]) for c in range(3)] for r in range(2)]

    # ── fig (a) losses ──────────────────────────────────────────────────────
    loss_panels = [
        (0, 0, SYNTH_PCA_LOSS,     EPOCH_CAP["synth_pca"]),
        (0, 1, SYNTH_FISHER_LOSS,  EPOCH_CAP["synth_fisher"]),
        (1, 0, FMNIST_PCA_LOSS,    EPOCH_CAP["fmnist_pca"]),
        (1, 1, FMNIST_FISHER_LOSS, EPOCH_CAP["fmnist_fisher"]),
    ]
    col_titles_a = ["MSE / PCA models", "Fisher / LDA models"]
    row_labels   = ["Synthetic Data", "Fashion-MNIST"]
    panel_labels_a = [["(i)", "(iii)"], ["(ii)", "(iv)"]]
    panel_labels_b = [["(i)", "(iii)", "(v)"], ["(ii)", "(iv)", "(vi)"]]

    for row, col, models, cap in loss_panels:
        ax = axes_a[row][col]
        _panel_losses(ax, models, histories, cap)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss"
                      + (" (log)" if ax.get_yscale() == "log" else ""))
        if row == 0:
            ax.set_title(col_titles_a[col])
        if col == 0:
            ax.annotate(row_labels[row], xy=(-0.32, 0.5),
                        xycoords="axes fraction",
                        fontsize=9, rotation=90, va="center", ha="center",
                        fontweight="bold")
        ax.text(-0.08, 1.02, panel_labels_a[row][col],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left", zorder=5)
        ax.set_xlim(1, cap)

    # ── fig (b) paired cosine (col 0) + angle to PCA (col 1) + LDA (col 2) ──
    # col 0: PCA paired (synth row + fmnist row)
    paired_pca_panels = [
        (0, SYNTH_PCA_MODELS,  synth_data,  "pca_paired", 50),
        (1, FMNIST_PCA_MODELS, fmnist_data, "pca_paired", 32),
    ]
    # col 2: LDA paired
    paired_lda_panels = [
        (0, SYNTH_FISHER_MODELS,  synth_fisher,  "lda_paired", 19),
        (1, FMNIST_FISHER_MODELS, fmnist_fisher, "lda_paired",  9),
    ]
    # col 1: angle to PCA, FP-MRL and S-MRL only
    angle_panels = [
        (0, synth_data,  50),
        (1, fmnist_data, 32),
    ]
    col_titles_b = ["Angle to PCA", "Paired cos. to PCA", "Paired cos. to LDA"]

    for row, data_src, n_plot in angle_panels:
        ax = axes_b[row][0]
        _panel_angle(ax, ANGLE_PCA_MODELS, data_src, n_plot)
        ax.set_xlabel("Prefix size $k$")
        ax.set_ylabel("Mean principal angle (°)")
        if row == 0:
            ax.set_title(col_titles_b[0])
        ax.annotate(row_labels[row], xy=(-0.34, 0.5),
                    xycoords="axes fraction",
                    fontsize=9, rotation=90, va="center", ha="center",
                    fontweight="bold")
        ax.text(-0.08, 1.02, panel_labels_b[row][0],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left", zorder=5)

    for row, models, data, metric, n_plot in paired_pca_panels:
        ax = axes_b[row][1]
        _panel_paired(ax, models, data, metric, n_plot)
        ax.set_xlabel("Prefix size $k$")
        ax.set_ylabel("Paired cos. sim.")
        if row == 0:
            ax.set_title(col_titles_b[1])
        ax.text(-0.08, 1.02, panel_labels_b[row][1],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left", zorder=5)

    for row, models, data, metric, n_plot in paired_lda_panels:
        ax = axes_b[row][2]
        _panel_paired(ax, models, data, metric, n_plot)
        ax.set_xlabel("Prefix size $k$")
        ax.set_ylabel("Paired cos. sim.")
        if row == 0:
            ax.set_title(col_titles_b[2])
        ax.text(-0.08, 1.02, panel_labels_b[row][2],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left", zorder=5)

    # Separate legends matching plot_fig_combined.py.
    fig.legend(handles=SHARED_LEGEND_HANDLES,
               loc="lower center", bbox_to_anchor=(0.24, 0.06),
               ncol=5, frameon=True, handlelength=1.4,
               borderpad=0.4, labelspacing=0.3, columnspacing=1.0,
               fontsize=8)
    fig.legend(handles=SHARED_LEGEND_B,
               loc="lower center", bbox_to_anchor=(0.755, 0.06),
               ncol=5, frameon=True, handlelength=1.4,
               borderpad=0.4, labelspacing=0.3, columnspacing=1.0,
               fontsize=8)

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    stem = save_fig(fig, "fig_losses_combined", fig_stamp)
    plt.close(fig)
    print(f"[fig] Done in {time.time()-t0:.1f}s  →  {stem}")


if __name__ == "__main__":
    main()
