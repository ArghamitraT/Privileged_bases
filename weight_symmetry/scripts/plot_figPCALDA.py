"""
Script: weight_symmetry/scripts/plot_figPCALDA.py
-----------------------------------------------------------------
2×2 figure comparing cosine similarity to PCA and LDA eigenvectors
across synthetic and Fashion-MNIST datasets.

Layout
------
  Row 0 — Synthetic data (orderedBoth, embed_dim=50 PCA / 19 LDA)
    (0,0)  PCA cosine sim — LAE/MSE + CE family models  (all 50 dims)
    (0,1)  LDA cosine sim — Fisher family models         (all 19 dims)

  Row 1 — Fashion-MNIST data (embed_dim=32 PCA / 9 LDA)
    (1,0)  PCA cosine sim — MSE family models            (all 32 dims)
    (1,1)  LDA cosine sim — Fisher family models         (all  9 dims)

Eigenvalue bars are drawn behind the cosine curves in every panel
(PCA bars for PCA panels, LDA bars for LDA panels), using a twin
y-axis exactly as in plot_fig1_mse_pca_ce_lda.py.

Default run folders (pass CLI args to override):
  --synth-run     exprmnt_2026_04_19__16_00_49/exprmnt_2026_04_20__13_46_30
  --fp-fisher-run exprmnt_2026_04_20__01_44_24
  --extra-fisher-run  exprmnt_2026_04_20__11_33_48
  --fmnist-run    exprmnt_2026_04_20__22_41_23
  --fmnist-fisher-run exprmnt_2026_04_21__07_47_27

Eigenvalue sources:
  Synthetic PCA — global eigenvalues dir (pca_eigenvalues_norm.npy)
  Synthetic LDA — global eigenvalues dir (lda_eigenvalues_norm.npy)
  Fashion-MNIST PCA — fmnist-run folder (pca_eigenvalues.npy, raw)
  Fashion-MNIST LDA — fmnist-fisher-run folder (lda_eigenvalues_norm.npy)

Conda environment: mrl_env

Usage:
    python weight_symmetry/scripts/plot_figPCALDA.py
    python weight_symmetry/scripts/plot_figPCALDA.py \\
        --synth-run    <folder> \\
        --fmnist-run   <folder> \\
        --fmnist-fisher-run <folder>
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

RESULTS_ROOT = os.path.join(_PROJ_ROOT, "files", "results")
EIG_DIR      = os.path.join(
    RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "eigenvalues"
)

# Default run folders (relative to RESULTS_ROOT)
DEFAULT_SYNTH_RUN         = "exprmnt_2026_04_19__16_00_49/exprmnt_2026_04_20__13_46_30"
DEFAULT_NORMAL_FISHER_RUN = "exprmnt_2026_04_20__01_31_36"   # Normal Fisher baseline
DEFAULT_FP_FISHER_RUN     = "exprmnt_2026_04_20__01_44_24"   # FP Fisher (better run)
DEFAULT_EXTRA_FISHER_RUN  = "exprmnt_2026_04_20__11_33_48"   # MRL Fisher + PrefixL1 Fisher
DEFAULT_FMNIST_RUN        = "exprmnt_2026_04_20__22_41_23"
DEFAULT_FMNIST_FISHER     = "exprmnt_2026_04_21__07_47_27"

# ---------------------------------------------------------------------------
# Bar overlay colors
# ---------------------------------------------------------------------------
PCA_BAR_COLOR = "#89C4E1"   # faded blue
LDA_BAR_COLOR = "#F0A500"   # amber / gold

# ---------------------------------------------------------------------------
# Model configs  (tag, legend label, color, linestyle, linewidth, pending)
# ---------------------------------------------------------------------------

# Panel (0,0) — Synthetic PCA, MSE/LAE family only
SYNTH_PCA_MODELS = [
    ("mse_lae",          "LAE",          "#888888", "--", 1.0, False),
    ("std_mrl_mse",      "MRL",          "#E07B00", "-",  1.8, False),
    ("fp_mrl_mse_ortho", "FP MRL",       "#009E73", "-",  1.8, False),
    ("prefix_l1_mse",    "PrefixL1",     "#CC79A7", "-",  1.8, False),
    ("nonuniform_l2",    "NonUnif. L2",  "#56B4E9", "-",  1.8, False),
]

# Panel (0,1) — Synthetic LDA, Fisher family
# fisher key comes from normal_fisher_run (exprmnt_2026_04_20__01_31_36)
SYNTH_FISHER_MODELS = [
    ("fisher",           "Unordered",  "#888888", "--", 1.0, False),
    ("fp_fisher",        "FP MRL",     "#009E73", "-",  1.8, False),
    ("std_mrl_fisher",   "MRL",        "#E07B00", "-",  1.8, False),
    ("prefix_l1_fisher", "PrefixL1",   "#CC79A7", "-",  1.8, False),
]

# Panel (1,0) — Fashion-MNIST PCA, MSE family
FMNIST_PCA_MODELS = [
    ("mse_lae",          "LAE",          "#888888", "--", 1.0, False),
    ("std_mrl_mse",      "MRL",          "#E07B00", "-",  1.8, False),
    ("fp_mrl_mse_ortho", "FP MRL",       "#009E73", "-",  1.8, False),
    ("prefix_l1_mse",    "PrefixL1",     "#CC79A7", "-",  1.8, False),
    ("nonuniform_l2",    "NonUnif. L2",  "#56B4E9", "-",  1.8, False),
]

# Panel (1,1) — Fashion-MNIST LDA, Fisher family
FMNIST_FISHER_MODELS = [
    ("fisher",           "Unordered",  "#888888", "--", 1.0, False),
    ("fp_fisher",        "FP MRL",     "#009E73", "-",  1.8, False),
    ("std_mrl_fisher",   "MRL",        "#E07B00", "-",  1.8, False),
    ("prefix_l1_fisher", "PrefixL1",   "#CC79A7", "-",  1.8, False),
]


# ---------------------------------------------------------------------------
# Eigenvalue loading
# ---------------------------------------------------------------------------

def _norm_by_max(v: np.ndarray) -> np.ndarray:
    m = v.max()
    return v / m if m > 0 else v


def _load_synth_eigs():
    pca_raw = np.load(os.path.join(EIG_DIR, "pca_eigenvalues.npy"),
                      allow_pickle=True).astype(float)
    lda_raw = np.load(os.path.join(EIG_DIR, "lda_eigenvalues.npy"),
                      allow_pickle=True).astype(float)
    return _norm_by_max(pca_raw), _norm_by_max(lda_raw)


def _load_fmnist_eigs(fmnist_run_dir: str, fmnist_fisher_dir: str):
    pca_raw = np.load(os.path.join(fmnist_run_dir, "pca_eigenvalues.npy"),
                      allow_pickle=True).astype(float)
    lda_raw = np.load(os.path.join(fmnist_fisher_dir, "lda_eigenvalues_norm.npy"),
                      allow_pickle=True).astype(float)
    return _norm_by_max(pca_raw), _norm_by_max(lda_raw)


# ---------------------------------------------------------------------------
# Panel drawing  (identical logic to plot_fig1; kept self-contained here)
# ---------------------------------------------------------------------------

def _plot_panel(ax, axr, models, data, metric_suffix, n_plot,
                bg_vals=None, bar_color=None, bar_ylabel=None):
    xs = np.arange(1, n_plot + 1)

    if axr is not None and bg_vals is not None:
        bc  = bar_color or PCA_BAR_COLOR
        bg  = bg_vals[:n_plot]
        axr.bar(xs, bg, color=bc, alpha=0.30, zorder=0, width=0.8)
        axr.set_ylabel(bar_ylabel or "Norm. eigenvalue", color=bc, fontsize=7)
        axr.tick_params(axis="y", labelcolor=bc, labelsize=6)
        axr.set_ylim(0, min(1.01, max(bg) * 1.3))

    for tag, label, color, ls, lw, pending in models:
        key = f"{tag}_{metric_suffix}"
        if key not in data:
            if pending:
                pass   # placeholder; not drawn
            else:
                print(f"  [warn] missing key: {key}")
            continue

        mat = data[key].astype(float)
        mat = np.where(mat < 0, np.nan, mat)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        actual = min(n_plot, mat.shape[1])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2×2 PCALDA figure: PCA/LDA cosine sim across two datasets"
    )
    parser.add_argument("--synth-run",          default=DEFAULT_SYNTH_RUN,
                        help="Folder with synthetic MSE metrics_raw.npz")
    parser.add_argument("--normal-fisher-run",  default=DEFAULT_NORMAL_FISHER_RUN,
                        help="Folder with Normal Fisher baseline metrics_raw.npz")
    parser.add_argument("--fp-fisher-run",      default=DEFAULT_FP_FISHER_RUN,
                        help="Folder with FP Fisher (better run) metrics_raw.npz")
    parser.add_argument("--extra-fisher-run",   default=DEFAULT_EXTRA_FISHER_RUN,
                        help="Folder with MRL Fisher + PrefixL1 Fisher metrics_raw.npz")
    parser.add_argument("--fmnist-run",         default=DEFAULT_FMNIST_RUN,
                        help="Folder with Fashion-MNIST MSE metrics_raw.npz")
    parser.add_argument("--fmnist-fisher-run",  default=DEFAULT_FMNIST_FISHER,
                        help="Folder with Fashion-MNIST Fisher metrics_raw.npz")
    parser.add_argument("--results-root",       default=RESULTS_ROOT)
    args = parser.parse_args()

    root = args.results_root

    # Resolve absolute paths
    def _abs(rel):
        return os.path.join(root, rel) if not os.path.isabs(rel) else rel

    synth_dir          = _abs(args.synth_run)
    normal_fisher_dir  = _abs(args.normal_fisher_run)
    fp_fisher_dir      = _abs(args.fp_fisher_run)
    extra_fisher_dir   = _abs(args.extra_fisher_run)
    fmnist_dir         = _abs(args.fmnist_run)
    fmnist_fisher_dir  = _abs(args.fmnist_fisher_run)

    # Verify all npz files exist
    npz_paths = {
        "synth":         os.path.join(synth_dir,         "metrics_raw.npz"),
        "normal_fisher": os.path.join(normal_fisher_dir, "metrics_raw.npz"),
        "fp_fisher":     os.path.join(fp_fisher_dir,     "metrics_raw.npz"),
        "extra_fisher":  os.path.join(extra_fisher_dir,  "metrics_raw.npz"),
        "fmnist":        os.path.join(fmnist_dir,        "metrics_raw.npz"),
        "fmnist_fisher": os.path.join(fmnist_fisher_dir, "metrics_raw.npz"),
    }
    for name, p in npz_paths.items():
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found ({name}): {p}")

    # Load all data
    # Merge order: normal_fisher first (provides fisher_lda_cosine), then fp_fisher
    # overrides fp_fisher_lda_cosine with the better dedicated run, then extra_fisher
    # adds std_mrl_fisher and prefix_l1_fisher.
    synth_data   = dict(np.load(npz_paths["synth"]))
    synth_fisher = {}
    synth_fisher.update(dict(np.load(npz_paths["normal_fisher"])))
    synth_fisher.update(dict(np.load(npz_paths["fp_fisher"])))
    synth_fisher.update(dict(np.load(npz_paths["extra_fisher"])))
    fmnist_data  = dict(np.load(npz_paths["fmnist"]))
    fmnist_fisher= dict(np.load(npz_paths["fmnist_fisher"]))

    for name, p in npz_paths.items():
        print(f"[PCALDA] {name:15s}: {p}")

    # Load eigenvalues
    synth_pca_eig, synth_lda_eig = _load_synth_eigs()
    fmnist_pca_eig, fmnist_lda_eig = _load_fmnist_eigs(fmnist_dir, fmnist_fisher_dir)

    print(f"[PCALDA] Synth PCA eig:   {synth_pca_eig.shape}  "
          f"LDA eig: {synth_lda_eig.shape}")
    print(f"[PCALDA] FMnist PCA eig:  {fmnist_pca_eig.shape}  "
          f"LDA eig: {fmnist_lda_eig.shape}")

    # Embed dims per panel
    N_SYNTH_PCA    = 50
    N_SYNTH_LDA    = 19
    N_FMNIST_PCA   = 32
    N_FMNIST_LDA   = 9

    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    apply_style()
    mpl.rcParams.update({
        "axes.titlesize":  8,
        "axes.labelsize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # (row, col, models, data, metric_suffix, n_plot, eig_vals, bar_color, bar_ylabel)
    panels = [
        (0, 0, SYNTH_PCA_MODELS,    synth_data,   "pca_cosine",
         N_SYNTH_PCA,  synth_pca_eig,  PCA_BAR_COLOR, "Norm. PCA eigenvalue"),
        (0, 1, SYNTH_FISHER_MODELS,  synth_fisher, "lda_cosine",
         N_SYNTH_LDA,  synth_lda_eig,  LDA_BAR_COLOR, "Norm. LDA eigenvalue"),
        (1, 0, FMNIST_PCA_MODELS,    fmnist_data,  "pca_cosine",
         N_FMNIST_PCA, fmnist_pca_eig, PCA_BAR_COLOR, "Norm. PCA eigenvalue"),
        (1, 1, FMNIST_FISHER_MODELS, fmnist_fisher,"lda_cosine",
         N_FMNIST_LDA, fmnist_lda_eig, LDA_BAR_COLOR, "Norm. LDA eigenvalue"),
    ]

    col_titles = ["Cosine sim. to PCA", "Cosine sim. to LDA"]
    row_labels = ["Synthetic", "Fashion-MNIST"]

    for row, col, models, data, metric, n_plot, eig, bc, byl in panels:
        ax  = axes[row, col]
        axr = ax.twinx()

        _plot_panel(ax, axr, models, data, metric, n_plot,
                    bg_vals=eig, bar_color=bc, bar_ylabel=byl)

        ax.set_xlabel("Prefix size $k$", fontsize=8)
        if col == 0:
            ax.set_ylabel("Max cosine sim.", fontsize=8)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8)

        # Row label in left margin (col 0 only)
        if col == 0:
            ax.annotate(row_labels[row], xy=(-0.22, 0.5),
                        xycoords="axes fraction",
                        fontsize=8, rotation=90, va="center", ha="center",
                        fontweight="bold")

    # Single shared legend in (1,1) — bottom right
    shared_handles = [
        Line2D([0], [0], color="#888888", ls="--", lw=1.0, label="LAE / Unordered"),
        Line2D([0], [0], color="#E07B00", ls="-",  lw=1.8, label="MRL"),
        Line2D([0], [0], color="#009E73", ls="-",  lw=1.8, label="FP MRL"),
        Line2D([0], [0], color="#CC79A7", ls="-",  lw=1.8, label="PrefixL1"),
        Line2D([0], [0], color="#56B4E9", ls="-",  lw=1.8, label="NonUnif. L2"),
    ]
    axes[0, 1].legend(handles=shared_handles, loc="lower right",
                      frameon=True, fontsize=6, handlelength=1.6,
                      borderpad=0.5, labelspacing=0.3)

    fig.tight_layout(h_pad=1.2, w_pad=0.8)

    stem = save_fig(fig, "figPCALDA_cosine_grid", fig_stamp)
    plt.close(fig)

    print(
        f"\n[PCALDA] Done.\n"
        f"  Stem            : {stem}\n"
        f"  Synth run       : {args.synth_run}\n"
        f"  FP Fisher run   : {args.fp_fisher_run}\n"
        f"  Extra Fisher run: {args.extra_fisher_run}\n"
        f"  FMnist run      : {args.fmnist_run}\n"
        f"  FMnist Fisher   : {args.fmnist_fisher_run}\n"
        "\nRemember to update the figure registry in "
        "plans/ICMLWorkshop_figure_style_plan.md"
    )


if __name__ == "__main__":
    main()
