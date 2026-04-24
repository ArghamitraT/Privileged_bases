"""
Script: weight_symmetry/scripts/plot_fig_combined.py
------------------------------------------------------
Combined publication figure joining the PCALDA and CLF figures side-by-side.

  (a) Left  — 2×2 PCALDA: cosine similarity to PCA/LDA × synthetic/Fashion-MNIST
  (b) Right — 2×3 CLF:    per-dim importance, linear accuracy, method agreement

Identical font, size, and color palette throughout.

Conda environment: mrl_env

Usage:
    python weight_symmetry/scripts/plot_fig_combined.py \\
        --clf-run-dir /path/to/exprmnt_2026_04_22__15_40_00
    python weight_symmetry/scripts/plot_fig_combined.py \\
        --clf-run-dir /path/to/exprmnt_2026_04_22__15_40_00 --clf-alpha 0.5 --fast
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_PROJ_ROOT  = os.path.dirname(_CODE_ROOT)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from weight_symmetry.plotting.style import apply_style, save_fig
from weight_symmetry.data.loader import load_data

RESULTS_ROOT = os.path.join(_PROJ_ROOT, "files", "results")
EIG_DIR      = os.path.join(
    RESULTS_ROOT, "ICMLWorkshop_weightSymmetry2026", "eigenvalues"
)

# Default CLF run dir
DEFAULT_CLF_RUN = os.path.join(RESULTS_ROOT, "exprmnt_2026_04_22__15_40_00")

# Default PCALDA run folders
DEFAULT_SYNTH_RUN         = "exprmnt_2026_04_23__20_18_55"
DEFAULT_NORMAL_FISHER_RUN = "exprmnt_2026_04_20__01_31_36"
DEFAULT_FP_FISHER_RUN     = "exprmnt_2026_04_20__01_44_24"
DEFAULT_EXTRA_FISHER_RUN  = "exprmnt_2026_04_20__11_33_48"
DEFAULT_FMNIST_RUN        = "exprmnt_2026_04_23__15_29_53"
DEFAULT_FMNIST_FISHER     = "exprmnt_2026_04_21__07_47_27"

# ==============================================================================
# Shared style
# ==============================================================================

FP_MRL_COLOR    = "#009E73"
PREFIX_L1_COLOR = "#CC79A7"
MRL_COLOR       = "#E07B00"
LAE_COLOR       = "#888888"
L2_COLOR        = "#56B4E9"

PCA_BAR_COLOR  = "#C9AEED"
PCA_TEXT_COLOR = "#7B5AB8"   # darker purple — used only for PCA axis label/ticks
LDA_BAR_COLOR  = "#F0A500"

FP_MRL_LABEL    = "FP-MRL"
PREFIX_L1_LABEL = r"MD-$\ell_1$"

# PCALDA model configs (tag, label, color, ls, lw, pending)
SYNTH_PCA_MODELS = [
    ("mse_lae",          "Unordered",    LAE_COLOR,       "--", 1.0, False),
    ("std_mrl_mse",      "S-MRL",   MRL_COLOR,       "-",  1.8, False),
    ("fp_mrl_mse_ortho", "FP-MRL",       FP_MRL_COLOR,    "-",  1.8, False),
    ("prefix_l1_mse",    r"MD-$\ell_1$",     PREFIX_L1_COLOR, "-",  1.8, False),
    ("nonuniform_l2",    r"NU-$\ell_2$",  L2_COLOR,        "-",  1.8, False),
]
SYNTH_FISHER_MODELS = [
    ("fisher",           "Unordered",    LAE_COLOR,       "--", 1.0, False),
    ("fp_fisher",        "FP-MRL",       FP_MRL_COLOR,    "-",  1.8, False),
    ("std_mrl_fisher",   "S-MRL",   MRL_COLOR,       "-",  1.8, False),
    ("prefix_l1_fisher", r"MD-$\ell_1$",     PREFIX_L1_COLOR, "-",  1.8, False),
]
FMNIST_PCA_MODELS = [
    ("mse_lae",          "Unordered",    LAE_COLOR,       "--", 1.0, False),
    ("std_mrl_mse",      "S-MRL",   MRL_COLOR,       "-",  1.8, False),
    ("fp_mrl_mse_ortho", "FP-MRL",       FP_MRL_COLOR,    "-",  1.8, False),
    ("prefix_l1_mse",    r"MD-$\ell_1$",     PREFIX_L1_COLOR, "-",  1.8, False),
    ("nonuniform_l2",    r"NU-$\ell_2$",  L2_COLOR,        "-",  1.8, False),
]
FMNIST_FISHER_MODELS = [
    ("fisher",           "Unordered",    LAE_COLOR,       "--", 1.0, False),
    ("fp_fisher",        "FP-MRL",       FP_MRL_COLOR,    "-",  1.8, False),
    ("std_mrl_fisher",   "S-MRL",   MRL_COLOR,       "-",  1.8, False),
    ("prefix_l1_fisher", r"MD-$\ell_1$",     PREFIX_L1_COLOR, "-",  1.8, False),
]

SHARED_LEGEND_HANDLES = [
    Line2D([0], [0], color=LAE_COLOR,       ls="--", lw=1.0, label="Unordered"),
    Line2D([0], [0], color=MRL_COLOR,       ls="-",  lw=1.8, label="S-MRL"),
    Line2D([0], [0], color=FP_MRL_COLOR,    ls="-",  lw=1.8, label="FP-MRL"),
    Line2D([0], [0], color=PREFIX_L1_COLOR, ls="-",  lw=1.8, label=r"MD-$\ell_1$"),
    Line2D([0], [0], color=L2_COLOR,        ls="-",  lw=1.8, label=r"NU-$\ell_2$"),
]

SHARED_LEGEND_B = [
    Line2D([0], [0], color=FP_MRL_COLOR,    ls="-", lw=1.8, label=FP_MRL_LABEL),
    Line2D([0], [0], color=PREFIX_L1_COLOR, ls="-", lw=1.8, label=PREFIX_L1_LABEL),
]

# ==============================================================================
# CLF: architecture (mirrors exp_clf.py / plot_clf_figure.py)
# ==============================================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

class MRLEHead(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes, bias=False)
    def forward(self, x):
        return self.head(x)

# ==============================================================================
# PCALDA: data helpers
# ==============================================================================

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
    pca_raw = np.load(os.path.join(fmnist_run_dir, "pca_eigenvalues.npy"),
                      allow_pickle=True).astype(float)
    lda_raw = np.load(os.path.join(fmnist_fisher_dir, "lda_eigenvalues_norm.npy"),
                      allow_pickle=True).astype(float)
    return _norm_by_max(pca_raw), _norm_by_max(lda_raw)

# ==============================================================================
# PCALDA: panel drawing
# ==============================================================================

def _pcalda_panel(ax, axr, models, data, metric_suffix, n_plot,
                  bg_vals=None, bar_color=None, bar_ylabel=None):
    if axr is not None and bg_vals is not None:
        bc  = bar_color or PCA_BAR_COLOR
        bg  = bg_vals[:n_plot]
        xs  = np.arange(1, n_plot + 1)
        bar_alpha  = 0.55 if bc == PCA_BAR_COLOR else 0.30
        text_color = PCA_TEXT_COLOR if bc == PCA_BAR_COLOR else bc
        axr.bar(xs, bg, color=bc, alpha=bar_alpha, zorder=0, width=0.8)
        axr.set_ylabel(bar_ylabel or "Norm. eigenvalue", color=text_color)
        axr.tick_params(axis="y", labelcolor=text_color, labelsize=7)
        axr.set_ylim(0, min(1.01, max(bg) * 1.3))

    for tag, label, color, ls, lw, pending in models:
        key = f"{tag}_{metric_suffix}"
        if key not in data:
            if not pending:
                print(f"  [warn] missing key: {key}")
            continue
        mat = data[key].astype(float)
        mat = np.where(mat < 0, np.nan, mat)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        actual = min(n_plot, mat.shape[1])
        xs_m   = np.arange(1, actual + 1)
        mean   = np.nanmean(mat[:, :actual], axis=0)
        std    = np.nanstd(mat[:, :actual],  axis=0)
        ax.plot(xs_m, mean, label=label, color=color, ls=ls, lw=lw, zorder=3)
        if mat.shape[0] > 1:
            ax.fill_between(xs_m, mean - std, mean + std,
                            alpha=0.15, color=color, zorder=2)

    ax.set_xlim(0, n_plot + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray", zorder=1)
    if axr is not None:
        ax.set_zorder(axr.get_zorder() + 1)
        ax.patch.set_visible(False)

# ==============================================================================
# CLF: data helpers
# ==============================================================================

def _alpha_tag(alpha):
    return f"prefix_l1_a{str(alpha).replace('.', '')}"

def _get_embeddings(encoder, X):
    encoder.eval()
    with torch.no_grad():
        parts = [encoder(X[i:i+512]).cpu().numpy() for i in range(0, len(X), 512)]
    return np.concatenate(parts, axis=0)

def _load_encoder(run_dir, tag, input_dim, hidden_dim, embed_dim):
    enc = MLPEncoder(input_dim, hidden_dim, embed_dim)
    enc.load_state_dict(torch.load(os.path.join(run_dir, f"{tag}_encoder_best.pt"),
                                   map_location="cpu", weights_only=True))
    enc.eval()
    return enc

def _load_head_mrle(run_dir, embed_dim, n_classes):
    head = MRLEHead(embed_dim, n_classes)
    head.load_state_dict(torch.load(os.path.join(run_dir, "mrl_e_head_best.pt"),
                                    map_location="cpu", weights_only=True))
    head.eval()
    return head

def _load_head_l1(run_dir, tag, embed_dim, n_classes):
    head = nn.Linear(embed_dim, n_classes)
    head.load_state_dict(torch.load(os.path.join(run_dir, f"{tag}_head_best.pt"),
                                    map_location="cpu", weights_only=True))
    head.eval()
    return head

def _probe_acc(Z_train, Z_test, y_train, y_test, max_samples, seed):
    rng = np.random.default_rng(seed)
    n   = len(y_train)
    idx = rng.choice(n, min(n, max_samples), replace=False) if n > max_samples else np.arange(n)
    acc = np.zeros(Z_test.shape[1])
    for j in range(Z_test.shape[1]):
        lr = LogisticRegression(solver="saga", max_iter=500, random_state=seed, n_jobs=1)
        lr.fit(Z_train[idx, j:j+1], y_train[idx])
        acc[j] = float(lr.score(Z_test[:, j:j+1], y_test))
    return acc

def _eval2_sweep(Z_test, y_test, W, b, eval_prefixes):
    return {k: float((np.argmax(Z_test[:, :k] @ W[:, :k].T + b, axis=1) == y_test).mean())
            for k in eval_prefixes}

# ==============================================================================
# CLF: panel drawing
# ==============================================================================

def _grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")

def _dim_ticks(ax, embed_dim):
    # Even-spaced ticks (step, 2*step, ..., embed_dim); snap last to
    # embed_dim if close to avoid crowding.
    step  = max(1, embed_dim // 8)
    ticks = list(range(step, embed_dim + 1, step))
    if ticks[-1] != embed_dim:
        if embed_dim - ticks[-1] < step:
            ticks[-1] = embed_dim
        else:
            ticks.append(embed_dim)
    ax.set_xticks(ticks)

def _panel_bars(ax, vals_mrl, vals_l1, ylabel, title, embed_dim):
    x = np.arange(1, embed_dim + 1)
    w = 0.38
    ax.bar(x - w/2, vals_mrl, width=w, color=FP_MRL_COLOR,    label=FP_MRL_LABEL,    alpha=0.85)
    ax.bar(x + w/2, vals_l1,  width=w, color=PREFIX_L1_COLOR, label=PREFIX_L1_LABEL, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Prefix size $k$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, embed_dim + 0.5)
    _dim_ticks(ax, embed_dim)
    _grid(ax)

def _panel_linear_acc(ax, res_mrl, res_l1, eval_prefixes):
    ks = eval_prefixes
    ax.plot(ks, [res_mrl[k] for k in ks], color=FP_MRL_COLOR,    ls="-", lw=1.8, label=FP_MRL_LABEL)
    ax.plot(ks, [res_l1[k]  for k in ks], color=PREFIX_L1_COLOR, ls="-", lw=1.8, label=PREFIX_L1_LABEL)
    ax.set_title(r"Linear acc. on $z_{1:k}$")
    ax.set_xlabel("Prefix size $k$")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, ks[-1] + 0.5)
    _dim_ticks(ax, ks[-1])
    _grid(ax)

def _panel_agreement(ax, scores, model_color, title):
    x, y = scores["mean_abs"], scores["probe_acc"]
    rho, _ = spearmanr(x, y)
    ax.plot(x, y, color=model_color, lw=0.7, alpha=0.35, zorder=2)
    ax.scatter(x, y, color=model_color, alpha=0.85, s=14, zorder=3)
    for d, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(str(d), (xi, yi), fontsize=6, ha="left", va="bottom")
    ax.text(0.05, 0.93, f"$\\rho$ = {rho:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    ax.set_title(title)
    ax.set_xlabel(r"Mean $|z_k|$")
    ax.set_ylabel("1D Probe Acc")
    _grid(ax)

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Combined PCALDA + CLF figure")
    parser.add_argument("--clf-run-dir",        default=DEFAULT_CLF_RUN)
    parser.add_argument("--clf-alpha",          type=float, default=1.0)
    parser.add_argument("--fast",               action="store_true",
                        help="Limit probe to 200 samples for a quick check")
    parser.add_argument("--synth-run",          default=DEFAULT_SYNTH_RUN)
    parser.add_argument("--normal-fisher-run",  default=DEFAULT_NORMAL_FISHER_RUN)
    parser.add_argument("--fp-fisher-run",      default=DEFAULT_FP_FISHER_RUN)
    parser.add_argument("--extra-fisher-run",   default=DEFAULT_EXTRA_FISHER_RUN)
    parser.add_argument("--fmnist-run",         default=DEFAULT_FMNIST_RUN)
    parser.add_argument("--fmnist-fisher-run",  default=DEFAULT_FMNIST_FISHER)
    args = parser.parse_args()

    t0 = time.time()

    def _abs(rel):
        return os.path.join(RESULTS_ROOT, rel) if not os.path.isabs(rel) else rel

    # ── Load PCALDA data ────────────────────────────────────────────────────
    print("[combined] Loading PCALDA data …")
    synth_data   = dict(np.load(os.path.join(_abs(args.synth_run),         "metrics_raw.npz")))
    synth_fisher = {}
    synth_fisher.update(dict(np.load(os.path.join(_abs(args.normal_fisher_run), "metrics_raw.npz"))))
    synth_fisher.update(dict(np.load(os.path.join(_abs(args.fp_fisher_run),     "metrics_raw.npz"))))
    synth_fisher.update(dict(np.load(os.path.join(_abs(args.extra_fisher_run),  "metrics_raw.npz"))))
    fmnist_data  = dict(np.load(os.path.join(_abs(args.fmnist_run),        "metrics_raw.npz")))
    fmnist_fisher= dict(np.load(os.path.join(_abs(args.fmnist_fisher_run), "metrics_raw.npz")))

    synth_pca_eig,  synth_lda_eig  = _load_synth_eigs()
    fmnist_pca_eig, fmnist_lda_eig = _load_fmnist_eigs(
        _abs(args.fmnist_run), _abs(args.fmnist_fisher_run))

    # ── Load CLF data ────────────────────────────────────────────────────────
    print("[combined] Loading CLF models …")
    clf_dir   = os.path.abspath(args.clf_run_dir)
    alpha_tag = _alpha_tag(args.clf_alpha)
    with open(os.path.join(clf_dir, "config.json")) as fh:
        cfg = json.load(fh)

    dataset   = cfg["dataset"]
    embed_dim = cfg["embed_dim"]
    hidden_dim= cfg["hidden_dim"]
    seed      = cfg["seed"]
    max_probe = 200 if args.fast else cfg.get("max_probe_samples", 2000)

    data      = load_data(dataset, seed=seed)
    y_train   = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test    = np.array(data.y_test.tolist(),  dtype=np.int64)
    n_classes = data.n_classes

    enc_mrl  = _load_encoder(clf_dir, "mrl_e", data.input_dim, hidden_dim, embed_dim)
    head_mrl = _load_head_mrle(clf_dir, embed_dim, n_classes)
    Z_tr_mrl = _get_embeddings(enc_mrl, data.X_train)
    Z_te_mrl = _get_embeddings(enc_mrl, data.X_test)
    W_mrl    = head_mrl.head.weight.detach().cpu().numpy()
    b_mrl    = np.zeros(n_classes)

    enc_l1  = _load_encoder(clf_dir, alpha_tag, data.input_dim, hidden_dim, embed_dim)
    head_l1 = _load_head_l1(clf_dir, alpha_tag, embed_dim, n_classes)
    Z_tr_l1 = np.ascontiguousarray(_get_embeddings(enc_l1, data.X_train)[:, ::-1])
    Z_te_l1 = np.ascontiguousarray(_get_embeddings(enc_l1, data.X_test)[:,  ::-1])
    W_l1    = np.ascontiguousarray(head_l1.weight.detach().cpu().numpy()[:, ::-1])
    b_l1    = head_l1.bias.detach().cpu().numpy()

    eval_prefixes = list(range(1, embed_dim + 1))

    print("[combined] Computing CLF metrics …")
    scores_mrl = {
        "mean_abs":  np.mean(np.abs(Z_te_mrl), axis=0),
        "variance":  np.var(Z_te_mrl, axis=0),
        "probe_acc": _probe_acc(Z_tr_mrl, Z_te_mrl, y_train, y_test, max_probe, seed),
    }
    scores_l1 = {
        "mean_abs":  np.mean(np.abs(Z_te_l1), axis=0),
        "variance":  np.var(Z_te_l1, axis=0),
        "probe_acc": _probe_acc(Z_tr_l1, Z_te_l1, y_train, y_test, max_probe, seed),
    }
    eval2_mrl = _eval2_sweep(Z_te_mrl, y_test, W_mrl, b_mrl, eval_prefixes)
    eval2_l1  = _eval2_sweep(Z_te_l1,  y_test, W_l1,  b_l1,  eval_prefixes)

    # ── Build figure ─────────────────────────────────────────────────────────
    print("[combined] Rendering figure …")
    apply_style()

    fig = plt.figure(figsize=(13.5, 5.2))
    # Left section (a): PCALDA 2×2 — bottom raised to leave room for horizontal legend
    gs_a = fig.add_gridspec(2, 2, left=0.05, right=0.43,
                             wspace=0.34, hspace=0.52, top=0.91, bottom=0.20)
    # Right section (b): CLF 2×3 — bottom matches panel (a) for consistent row heights
    gs_b = fig.add_gridspec(2, 3, left=0.52, right=0.99,
                             wspace=0.38, hspace=0.52, top=0.91, bottom=0.20)

    axes_a = [[fig.add_subplot(gs_a[r, c]) for c in range(2)] for r in range(2)]
    axes_b = [[fig.add_subplot(gs_b[r, c]) for c in range(3)] for r in range(2)]

    # (a) section label
    fig.text(0.05, 0.97, "(a)", fontsize=9, fontweight="bold", va="top")
    # (b) section label
    fig.text(0.52, 0.97, "(b)", fontsize=9, fontweight="bold", va="top")

    # ── PCALDA panels ────────────────────────────────────────────────────────
    pcalda_panels = [
        (0, 0, SYNTH_PCA_MODELS,    synth_data,    "pca_cosine", 50,
         synth_pca_eig,  PCA_BAR_COLOR, "PCA eig."),
        (0, 1, SYNTH_FISHER_MODELS,  synth_fisher,  "lda_cosine", 19,
         synth_lda_eig,  LDA_BAR_COLOR, "LDA eig."),
        (1, 0, FMNIST_PCA_MODELS,    fmnist_data,   "pca_cosine", 32,
         fmnist_pca_eig, PCA_BAR_COLOR, "PCA eig."),
        (1, 1, FMNIST_FISHER_MODELS, fmnist_fisher, "lda_cosine",  9,
         fmnist_lda_eig, LDA_BAR_COLOR, "LDA eig."),
    ]
    col_titles  = ["Cosine sim. to PCA", "Cosine sim. to LDA"]
    row_labels  = ["Synthetic Data", "Fashion-MNIST"]

    # Column-major panel labels: (1,1)=i top-left, (2,1)=ii bottom-left,
    # (1,2)=iii top-right, (2,2)=iv bottom-right
    panel_labels = [["(i)", "(iii)"], ["(ii)", "(iv)"]]

    for row, col, models, d, metric, n_plot, eig, bc, byl in pcalda_panels:
        ax  = axes_a[row][col]
        axr = ax.twinx()
        _pcalda_panel(ax, axr, models, d, metric, n_plot, eig, bc, byl)
        ax.set_xlabel("Prefix size $k$")
        if col == 0:
            ax.set_ylabel("Max cosine sim.")
        if row == 0:
            ax.set_title(col_titles[col])
        if col == 0:
            ax.annotate(row_labels[row], xy=(-0.24, 0.5),
                        xycoords="axes fraction",
                        fontsize=9, rotation=90, va="center", ha="center",
                        fontweight="bold")
        ax.text(-0.08, 1.02, panel_labels[row][col],
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left", zorder=5)

    # Shared legend — single horizontal strip below panel (a)
    fig.legend(handles=SHARED_LEGEND_HANDLES,
               loc="lower center", bbox_to_anchor=(0.24, 0.06),
               ncol=5, frameon=True, handlelength=1.4,
               borderpad=0.4, labelspacing=0.3, columnspacing=1.0,
               fontsize=8)

    # ── CLF panels ───────────────────────────────────────────────────────────
    # Row 0
    _panel_bars(axes_b[0][0], scores_mrl["mean_abs"], scores_l1["mean_abs"],
                r"Mean $|z_k|$", "Per-dim magnitude", embed_dim)
    _panel_bars(axes_b[0][1], scores_mrl["probe_acc"], scores_l1["probe_acc"],
                "Classification accuracy", "1D Probe Acc", embed_dim)
    _panel_agreement(axes_b[0][2], scores_mrl,
                     FP_MRL_COLOR, "Method agreement")
    # Row 1
    _panel_bars(axes_b[1][0], scores_mrl["variance"], scores_l1["variance"],
                "Variance", "Per-dim variance", embed_dim)
    _panel_linear_acc(axes_b[1][1], eval2_mrl, eval2_l1, eval_prefixes)
    _panel_agreement(axes_b[1][2], scores_l1,
                     PREFIX_L1_COLOR, "Method agreement")

    # Shared legend for panel (b) — horizontal strip below, mirroring panel (a)
    fig.legend(handles=SHARED_LEGEND_B,
               loc="lower center", bbox_to_anchor=(0.755, 0.06),
               ncol=2, frameon=True, handlelength=1.4,
               borderpad=0.4, labelspacing=0.3, columnspacing=1.0,
               fontsize=8)

    # Column-major panel labels for (b): (1,1)=i, (2,1)=ii, (1,2)=iii,
    # (2,2)=iv, (1,3)=v, (2,3)=vi
    panel_labels_b = [["(i)", "(iii)", "(v)"], ["(ii)", "(iv)", "(vi)"]]
    for r in range(2):
        for c in range(3):
            axes_b[r][c].text(-0.08, 1.02, panel_labels_b[r][c],
                              transform=axes_b[r][c].transAxes,
                              fontsize=9, fontweight="bold",
                              va="bottom", ha="left", zorder=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    stem = save_fig(fig, "fig_combined_pcalda_clf", fig_stamp)
    plt.close(fig)
    print(f"[combined] Done in {time.time()-t0:.1f}s  →  {stem}")


if __name__ == "__main__":
    main()
