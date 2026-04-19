"""
Script: weight_symmetry/scripts/diagnose_eigenvalue_gaps.py
------------------------------------------------------------
Diagnose whether near-degenerate PCA eigenvalues explain imperfect column
alignment in exp1 results — without re-running training.

Loads:
  - config.json        from the saved run folder (dataset, embed_dim)
  - metrics_raw.npz    from the saved run folder (column alignments per seed)

Computes:
  - Full SVD of the (standardised) training data → eigenvalues λ_k = σ_k² / n
  - Consecutive eigenvalue gaps  Δ_k = λ_k − λ_{k+1}

Plots (saved to --weights-dir):
  Panel 1 — Eigenvalue spectrum (top-d dims)
  Panel 2 — Consecutive eigenvalue gaps Δ_k (log scale)
  Panel 3 — Column alignment (from saved metrics) overlaid with gap markers
             so you can see exactly where gaps are small and alignment drops

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/diagnose_eigenvalue_gaps.py \\
        --weights-dir exprmnt_2026_04_15__12_00_00

    python weight_symmetry/scripts/diagnose_eigenvalue_gaps.py \\
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

_WS_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.data.loader import load_data
from weight_symmetry.utility import get_path


# Models whose column_alignment we plot (must match tags in metrics_raw.npz)
MODELS_TO_PLOT = [
    ("mse_lae_ortho",        "MSE LAE + ortho",         "black"),
    ("fullprefix_mrl",       "Full-prefix MRL",          "blue"),
    ("fullprefix_mrl_ortho", "Full-prefix MRL + dec ortho", "green"),
]


def load_run(weights_dir: str):
    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg = json.load(f)
    metrics = np.load(os.path.join(weights_dir, "metrics_raw.npz"))
    return cfg, metrics


def compute_eigenvalues(X_train, embed_dim: int):
    """
    Full SVD of centred data matrix.
    Returns:
        lambdas : (embed_dim,) eigenvalues in descending order (σ_k² / n)
        gaps    : (embed_dim-1,) consecutive gaps λ_k − λ_{k+1}
    """
    X = X_train.numpy().astype(np.float64)
    n = X.shape[0]
    # Use only as many singular values as needed (full SVD is expensive on MNIST)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    lambdas = (s[:embed_dim] ** 2) / n
    gaps    = lambdas[:-1] - lambdas[1:]   # Δ_k = λ_k - λ_{k+1}
    return lambdas, gaps


def plot_diagnostics(lambdas, gaps, metrics, embed_dim: int, out_dir: str):
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    dims      = np.arange(1, embed_dim + 1)
    gap_dims  = np.arange(1, embed_dim)      # gap_k is between dim k and k+1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ------------------------------------------------------------------
    # Panel 1 — Eigenvalue spectrum
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.bar(dims, lambdas, color="steelblue", alpha=0.8)
    ax.set_xlabel("PCA dimension k")
    ax.set_ylabel("Eigenvalue λ_k  (σ²/n)")
    ax.set_title("Eigenvalue Spectrum (top d dims)")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2 — Consecutive eigenvalue gaps (log scale)
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.bar(gap_dims, gaps, color="darkorange", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Dimension k  (gap = λ_k − λ_{k+1})")
    ax.set_ylabel("Eigenvalue gap  (log scale)")
    ax.set_title("Consecutive Eigenvalue Gaps\n(small gap = hard to recover individual eigenvector)")
    ax.grid(True, alpha=0.3, which="both")

    # Mark the five smallest gaps
    smallest = np.argsort(gaps)[:5]
    for idx in smallest:
        ax.axvline(x=gap_dims[idx], color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    # ------------------------------------------------------------------
    # Panel 3 — Column alignment vs gap (overlaid)
    # ------------------------------------------------------------------
    ax      = axes[2]
    ax2     = ax.twinx()

    for tag, label, color in MODELS_TO_PLOT:
        key = f"{tag}_column_alignments"
        if key not in metrics:
            continue
        mat   = metrics[key]      # (n_seeds, embed_dim)
        mean  = mat.mean(axis=0)
        std   = mat.std(axis=0)
        ax.plot(dims, mean, label=label, color=color)
        ax.fill_between(dims, mean - std, mean + std, alpha=0.12, color=color)

    # Normalised gap on secondary axis
    gap_padded        = np.append(gaps, np.nan)   # pad to length embed_dim for same x
    gap_norm          = gap_padded / (lambdas[0] + 1e-12)   # normalise by largest eigenvalue
    ax2.bar(dims, gap_norm, alpha=0.12, color="darkorange", label="norm. gap")
    ax2.set_ylabel("Normalised eigenvalue gap  (λ_k − λ_{k+1}) / λ_1", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax.set_xlabel("Prefix size m")
    ax.set_ylabel("Mean max cosine similarity")
    ax.set_title("Column Alignment vs Eigenvalue Gaps\n(orange bars: small gap → hard to recover)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"eigenvalue_diagnosis{fig_stamp}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[diagnose] Saved {out_path}")
    return out_path


def save_summary(lambdas, gaps, out_dir: str):
    lines = [
        "Eigenvalue Gap Diagnosis",
        "=" * 50,
        "",
        f"{'dim k':>6}  {'λ_k':>12}  {'gap (λ_k - λ_{k+1})':>22}  {'gap / λ_1':>10}",
        "-" * 58,
    ]
    for k in range(len(lambdas)):
        gap     = gaps[k - 1] if k > 0 else float("nan")
        gap_rel = gap / (lambdas[0] + 1e-12) if k > 0 else float("nan")
        lines.append(
            f"{k+1:>6}  {lambdas[k]:>12.4f}  {gap:>22.6f}  {gap_rel:>10.6f}"
        )
    out_path = os.path.join(out_dir, "eigenvalue_gaps.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[diagnose] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose eigenvalue gaps for exp1")
    parser.add_argument("--weights-dir", required=True, metavar="FOLDER",
                        help="Saved exp1 run folder (contains config.json + metrics_raw.npz)")
    args = parser.parse_args()

    weights_dir = args.weights_dir
    if not os.path.isabs(weights_dir):
        weights_dir = os.path.join(get_path("files/results"), weights_dir)

    print(f"[diagnose] Loading run from: {weights_dir}")
    cfg, metrics = load_run(weights_dir)

    dataset   = cfg["dataset"]
    embed_dim = cfg["embed_dim"]
    seed      = cfg["seeds"][0]

    print(f"[diagnose] Dataset: {dataset}  embed_dim: {embed_dim}")
    print(f"[diagnose] Loading data (seed={seed}) ...")
    data = load_data(dataset, seed=seed)

    print(f"[diagnose] Computing eigenvalues via SVD ...")
    lambdas, gaps = compute_eigenvalues(data.X_train, embed_dim)

    print(f"\n[diagnose] Top-{embed_dim} eigenvalues:")
    for k, (lam, gap) in enumerate(zip(lambdas, list(gaps) + [float("nan")])):
        print(f"  dim {k+1:>2}: λ={lam:8.3f}   gap_to_next={gap:8.4f}   rel={gap/(lambdas[0]+1e-12):.4f}")

    # Save outputs into the same run folder
    plot_diagnostics(lambdas, gaps, metrics, embed_dim, weights_dir)
    save_summary(lambdas, gaps, weights_dir)

    print(f"\n[diagnose] Done. Outputs written to: {weights_dir}")


if __name__ == "__main__":
    main()
