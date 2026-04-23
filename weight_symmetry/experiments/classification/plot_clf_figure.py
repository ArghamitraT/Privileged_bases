"""
Script: weight_symmetry/experiments/classification/plot_clf_figure.py
-----------------------------------------------------------------------
Publication-quality 2×3 classification figure for the Weight Symmetry paper.

Loads model checkpoints from a completed exp_clf.py run, re-extracts
embeddings, and produces a 2×3 panel grid:

  Row 0, Col 0  Mean |z| per dimension          grouped bars (FP MRL vs PrefixL1)
  Row 1, Col 0  Variance per dimension           grouped bars
  Row 0, Col 1  Probe accuracy per dimension     grouped bars
  Row 1, Col 1  Linear accuracy — trained W      line plot over prefix k (Eval 2, no refitting)
  Row 0, Col 2  Method agreement — FP MRL        3 normalised score lines + ρ(probe, mean)
  Row 1, Col 2  Method agreement — PrefixL1      same

Colors follow ICMLWorkshop_figure_style_plan.md:
  FP MRL   #009E73  solid   lw=1.8
  PrefixL1 #D55E00  -.      lw=1.8

Output is saved in a timestamped subfolder inside --run-dir AND in
  files/results/ICMLWorkshop_weightSymmetry2026/figures/

Conda environment: mrl_env

Usage:
    python weight_symmetry/experiments/classification/plot_clf_figure.py \\
        --run-dir /path/to/exprmnt_2026_04_22__15_40_00
    python weight_symmetry/experiments/classification/plot_clf_figure.py \\
        --run-dir /path/to/exprmnt_2026_04_22__15_40_00 --alpha 0.5
    python weight_symmetry/experiments/classification/plot_clf_figure.py \\
        --run-dir /path/to/exprmnt_2026_04_22__15_40_00 --fast
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from weight_symmetry.data.loader import load_data
from weight_symmetry.utility import save_runtime, save_code_snapshot

# ==============================================================================
# ICML Workshop 2026 style
# ==============================================================================
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

FIGURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "files", "results", "ICMLWorkshop_weightSymmetry2026", "figures",
)

FP_MRL_COLOR    = "#009E73"
PREFIX_L1_COLOR = "#CC79A7"
FP_MRL_LABEL    = "FP MRL"
PREFIX_L1_LABEL = "PrefixL1"

# ==============================================================================
# Architecture (mirrors exp_clf.py — self-contained, no import from there)
# ==============================================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class MRLEHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ==============================================================================
# Helpers
# ==============================================================================

def _alpha_tag(alpha: float) -> str:
    return f"prefix_l1_a{str(alpha).replace('.', '')}"


def _get_embeddings(encoder, X: torch.Tensor) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        parts = [encoder(X[i:i+512]).cpu().numpy() for i in range(0, len(X), 512)]
    return np.concatenate(parts, axis=0)


def _load_encoder(run_dir: str, tag: str, input_dim: int,
                  hidden_dim: int, embed_dim: int) -> MLPEncoder:
    enc = MLPEncoder(input_dim, hidden_dim, embed_dim)
    path = os.path.join(run_dir, f"{tag}_encoder_best.pt")
    enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    enc.eval()
    return enc


def _load_head_mrle(run_dir: str, embed_dim: int, n_classes: int) -> MRLEHead:
    head = MRLEHead(embed_dim, n_classes)
    path = os.path.join(run_dir, "mrl_e_head_best.pt")
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    head.eval()
    return head


def _load_head_l1(run_dir: str, tag: str, embed_dim: int,
                  n_classes: int) -> nn.Linear:
    head = nn.Linear(embed_dim, n_classes)
    path = os.path.join(run_dir, f"{tag}_head_best.pt")
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    head.eval()
    return head


def _probe_acc(Z_train: np.ndarray, Z_test: np.ndarray,
               y_train: np.ndarray, y_test: np.ndarray,
               max_samples: int, seed: int) -> np.ndarray:
    """Fit a 1-D logistic regression for each dimension; return accuracy array."""
    rng = np.random.default_rng(seed)
    n   = len(y_train)
    idx = rng.choice(n, min(n, max_samples), replace=False) if n > max_samples else np.arange(n)
    d   = Z_test.shape[1]
    acc = np.zeros(d)
    for j in range(d):
        lr = LogisticRegression(solver="saga", max_iter=500, random_state=seed, n_jobs=1)
        lr.fit(Z_train[idx, j:j+1], y_train[idx])
        acc[j] = float(lr.score(Z_test[:, j:j+1], y_test))
    return acc


def _eval2_sweep(Z_test: np.ndarray, y_test: np.ndarray,
                 W: np.ndarray, b: np.ndarray,
                 eval_prefixes: list) -> dict:
    """Accuracy at each prefix k using the trained classifier head (no refitting)."""
    results = {}
    for k in eval_prefixes:
        logits = Z_test[:, :k] @ W[:, :k].T + b
        results[k] = float((np.argmax(logits, axis=1) == y_test).mean())
    return results


# ==============================================================================
# Save helper (pdf + png + svg)
# ==============================================================================

def _save_fig(fig, base_name: str, fig_stamp: str, primary_dir: str) -> None:
    stem = f"{base_name}_{fig_stamp}"
    for out_dir in [primary_dir, os.path.abspath(FIGURES_DIR)]:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    print(f"[clf_fig] Saved {stem}.{{pdf,png,svg}}")


# ==============================================================================
# Axis helpers
# ==============================================================================

def _grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")


def _dim_ticks(ax, embed_dim: int):
    step = max(1, embed_dim // 8)
    ticks = list(range(0, embed_dim, step))
    if embed_dim - 1 not in ticks:
        ticks.append(embed_dim - 1)
    ax.set_xticks(ticks)


# ==============================================================================
# Panel drawing
# ==============================================================================

def _panel_bars(ax, vals_mrl: np.ndarray, vals_l1: np.ndarray,
                ylabel: str, title: str, embed_dim: int,
                show_legend: bool = False):
    """Grouped bar chart: two bars per dimension."""
    x   = np.arange(embed_dim)
    w   = 0.38
    ax.bar(x - w / 2, vals_mrl, width=w, color=FP_MRL_COLOR,
           label=FP_MRL_LABEL, alpha=0.85)
    ax.bar(x + w / 2, vals_l1,  width=w, color=PREFIX_L1_COLOR,
           label=PREFIX_L1_LABEL, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)
    _dim_ticks(ax, embed_dim)
    _grid(ax)
    if show_legend:
        ax.legend(loc="upper right")


def _panel_linear_acc(ax, res_mrl: dict, res_l1: dict, eval_prefixes: list):
    """Line plot of Eval-2 linear accuracy over prefix k."""
    ks   = eval_prefixes
    acc_mrl = [res_mrl[k] for k in ks]
    acc_l1  = [res_l1[k]  for k in ks]
    ax.plot(ks, acc_mrl, color=FP_MRL_COLOR,    ls="-",  lw=1.8, label=FP_MRL_LABEL)
    ax.plot(ks, acc_l1,  color=PREFIX_L1_COLOR, ls="-.", lw=1.8, label=PREFIX_L1_LABEL)
    ax.set_title("Linear acc. (trained $W$)")
    ax.set_xlabel("Prefix size $k$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    step = max(1, len(ks) // 8)
    shown = ks[::step]
    if ks[-1] not in shown:
        shown = shown + [ks[-1]]
    ax.set_xticks(shown)
    _grid(ax)
    ax.legend(loc="lower right")


def _panel_agreement(ax, scores: dict, model_color: str, title: str):
    """
    Scatter of Mean |z| vs 1D Probe Acc, one point per dimension.
    Dimension index annotated on each point; Spearman ρ in upper-left box.
    """
    x = scores["mean_abs"]
    y = scores["probe_acc"]
    rho, _ = spearmanr(x, y)

    # thin line connecting dims in order (0→1→…) to show trajectory
    ax.plot(x, y, color=model_color, lw=0.7, alpha=0.35, zorder=2)
    ax.scatter(x, y, color=model_color, alpha=0.85, s=14, zorder=3)
    for d, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(str(d), (xi, yi), fontsize=6, ha="left", va="bottom")

    ax.text(0.05, 0.93, f"$\\rho$ = {rho:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    ax.set_title(title)
    ax.set_xlabel("Mean $|z|$")
    ax.set_ylabel("1D Probe Acc")
    _grid(ax)


# ==============================================================================
# Main figure
# ==============================================================================

def make_figure(scores_mrl: dict, scores_l1: dict,
                eval2_mrl: dict, eval2_l1: dict,
                eval_prefixes: list, embed_dim: int,
                out_dir: str, fig_stamp: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    fig.subplots_adjust(hspace=0.52, wspace=0.42)

    # (0,0) Mean |z|
    _panel_bars(axes[0, 0],
                scores_mrl["mean_abs"], scores_l1["mean_abs"],
                ylabel="Mean $|z|$", title="Mean $|z|$",
                embed_dim=embed_dim, show_legend=True)

    # (1,0) Variance
    _panel_bars(axes[1, 0],
                scores_mrl["variance"], scores_l1["variance"],
                ylabel="Variance", title="Variance",
                embed_dim=embed_dim)

    # (0,1) Probe accuracy
    _panel_bars(axes[0, 1],
                scores_mrl["probe_acc"], scores_l1["probe_acc"],
                ylabel="Probe acc.", title="Probe accuracy",
                embed_dim=embed_dim)

    # (1,1) Linear accuracy — trained W (Eval 2)
    _panel_linear_acc(axes[1, 1], eval2_mrl, eval2_l1, eval_prefixes)

    # (0,2) Method agreement — FP MRL
    _panel_agreement(axes[0, 2], scores_mrl,
                     model_color=FP_MRL_COLOR,
                     title="Method agreement — FP MRL")

    # (1,2) Method agreement — PrefixL1
    _panel_agreement(axes[1, 2], scores_l1,
                     model_color=PREFIX_L1_COLOR,
                     title="Method agreement — PrefixL1")

    _save_fig(fig, "clf_figure", fig_stamp, out_dir)
    plt.close(fig)


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2×3 classification figure for the Weight Symmetry paper"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to a completed exp_clf.py output folder.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="PrefixL1 alpha to use (default 1.0).")
    parser.add_argument("--fast", action="store_true",
                        help="Limit probe samples to 200 for a quick check.")
    args = parser.parse_args()

    run_start = time.time()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    # Create timestamped subfolder inside run_dir (--use-weights convention)
    sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
    out_dir   = os.path.join(run_dir, sub_stamp)
    os.makedirs(out_dir, exist_ok=True)
    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    print(f"[clf_fig] Output → {out_dir}")

    # Load config from the source run
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    dataset    = cfg["dataset"]
    embed_dim  = cfg["embed_dim"]
    hidden_dim = cfg["hidden_dim"]
    seed       = cfg["seed"]

    max_probe = 200 if args.fast else cfg.get("max_probe_samples", 2000)

    alpha_tag = _alpha_tag(args.alpha)
    print(f"[clf_fig] dataset={dataset}  embed_dim={embed_dim}  "
          f"alpha={args.alpha} (tag={alpha_tag})")

    # Verify checkpoints exist
    for fname in [
        "mrl_e_encoder_best.pt", "mrl_e_head_best.pt",
        f"{alpha_tag}_encoder_best.pt", f"{alpha_tag}_head_best.pt",
    ]:
        p = os.path.join(run_dir, fname)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    # Load data
    data = load_data(dataset, seed=seed)
    y_train = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test  = np.array(data.y_test.tolist(),  dtype=np.int64)
    n_classes = data.n_classes
    print(f"[clf_fig] Data loaded: train={data.X_train.shape}  test={data.X_test.shape}")

    # ---- FP MRL-E ----
    print("[clf_fig] Loading FP MRL-E …")
    enc_mrl  = _load_encoder(run_dir, "mrl_e", data.input_dim, hidden_dim, embed_dim)
    head_mrl = _load_head_mrle(run_dir, embed_dim, n_classes)
    Z_tr_mrl = _get_embeddings(enc_mrl, data.X_train)
    Z_te_mrl = _get_embeddings(enc_mrl, data.X_test)
    W_mrl    = head_mrl.head.weight.detach().cpu().numpy()   # (n_classes, embed_dim)
    b_mrl    = np.zeros(n_classes)

    # ---- PrefixL1 ----
    print(f"[clf_fig] Loading PrefixL1 α={args.alpha} …")
    enc_l1  = _load_encoder(run_dir, alpha_tag, data.input_dim, hidden_dim, embed_dim)
    head_l1 = _load_head_l1(run_dir, alpha_tag, embed_dim, n_classes)
    Z_tr_l1 = np.ascontiguousarray(_get_embeddings(enc_l1, data.X_train)[:, ::-1])
    Z_te_l1 = np.ascontiguousarray(_get_embeddings(enc_l1, data.X_test)[:,  ::-1])
    W_l1    = np.ascontiguousarray(head_l1.weight.detach().cpu().numpy()[:, ::-1])
    b_l1    = head_l1.bias.detach().cpu().numpy()

    eval_prefixes = list(range(1, embed_dim + 1))

    # ---- Importance scores ----
    print("[clf_fig] Computing importance scores (mean, var, probe) …")
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

    # ---- Eval 2 — trained W ----
    print("[clf_fig] Computing Eval-2 linear accuracy …")
    eval2_mrl = _eval2_sweep(Z_te_mrl, y_test, W_mrl, b_mrl, eval_prefixes)
    eval2_l1  = _eval2_sweep(Z_te_l1,  y_test, W_l1,  b_l1,  eval_prefixes)

    # ---- Figure ----
    print("[clf_fig] Rendering figure …")
    make_figure(scores_mrl, scores_l1,
                eval2_mrl, eval2_l1,
                eval_prefixes, embed_dim,
                out_dir, fig_stamp)

    save_runtime(out_dir, time.time() - run_start)
    save_code_snapshot(out_dir)
    print(f"[clf_fig] Done → {out_dir}")


if __name__ == "__main__":
    main()
