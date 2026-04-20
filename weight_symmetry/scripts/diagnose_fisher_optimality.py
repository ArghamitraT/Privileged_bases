"""
Script: weight_symmetry/scripts/diagnose_fisher_optimality.py
-------------------------------------------------------------
Checks whether the trained FisherLoss model has converged to the true LDA
optimum by comparing its loss value against the theoretical optimal.

Procedure:
  1. Load saved encoder B from checkpoint
  2. Compute Fisher loss on full training set with learned B  → L_learned
  3. Set B = sklearn LDA directions (true optimum)
  4. Compute Fisher loss with B = LDA directions             → L_optimal
  5. Report gap: if L_learned ≈ L_optimal the subspace IS correct
     and the angle metric is measuring only rotation, not subspace quality.
     If L_learned >> L_optimal the optimization genuinely failed.

Also reports:
  - Subspace angle between learned B and LDA directions (confirmation)
  - Eigenvalue sum of S_W^{-1} S_B in raw x space (absolute upper bound)

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/diagnose_fisher_optimality.py \\
        --weights-dir exprmnt_2026_04_19__20_02_18

    python weight_symmetry/scripts/diagnose_fisher_optimality.py \\
        --weights-dir exprmnt_2026_04_19__20_02_18 --seeds 47
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

_WS_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.data.loader import load_data
from weight_symmetry.utility import get_path

MODEL_TAGS = [
    ("fisher",    "FisherLoss (standard)"),
    ("fp_fisher", "FullPrefixFisherLoss"),
]
EPS = 1e-4


# ==============================================================================
# Core computations
# ==============================================================================

def fisher_loss_value(B: np.ndarray, X: np.ndarray, y: np.ndarray,
                      eps: float = EPS) -> float:
    """
    Compute -Tr((S_W + εI)^{-1} S_B) for embeddings z = X @ B.T.
    B: (d, p), X: (n, p), y: (n,)
    """
    z        = X @ B.T                      # (n, d)
    n, d     = z.shape
    classes  = np.unique(y)
    mean_all = z.mean(0)
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
    reg = eps * np.eye(d)
    return -float(np.trace(np.linalg.solve(S_W + reg, S_B)))


def fullprefix_fisher_loss_value(B: np.ndarray, X: np.ndarray, y: np.ndarray,
                                  eps: float = EPS) -> float:
    """
    Compute (1/d) * sum_{k=1}^{d} FisherLoss(B_{1:k}, X, y)
    """
    d     = B.shape[0]
    total = 0.0
    for k in range(1, d + 1):
        total += fisher_loss_value(B[:k], X, y, eps)
    return total / d


def get_lda_directions(X: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """
    Fit sklearn LDA, return top-d eigenvectors as rows of B_lda (d, p).
    These are the directions that maximize the Fisher criterion.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    # lda.scalings_: (p, n_components), columns are LDA directions
    A = lda.scalings_[:, :d]          # (p, d)
    # Normalise columns so each direction is a unit vector
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return A.T                         # (d, p) — rows are directions, like encoder B


def mean_principal_angle(B1: np.ndarray, B2: np.ndarray) -> float:
    """
    Mean principal angle (degrees) between row-spaces of B1 (d1, p) and B2 (d2, p).
    """
    Q1, _ = np.linalg.qr(B1.T)    # (p, d1)
    Q2, _ = np.linalg.qr(B2.T)    # (p, d2)
    M      = Q1.T @ Q2             # (d1, d2)
    svals  = np.linalg.svd(M, compute_uv=False)
    svals  = np.clip(svals, -1.0, 1.0)
    angles = np.degrees(np.arccos(svals))
    return float(angles.mean())


def upper_bound_fisher(X: np.ndarray, y: np.ndarray, d: int,
                       eps: float = EPS) -> float:
    """
    Sum of top-d eigenvalues of (S_W(x) + εI)^{-1} S_B(x) in raw input space.
    This is the theoretical maximum of FisherLoss achievable by any B.
    """
    n        = X.shape[0]
    classes  = np.unique(y)
    mean_all = X.mean(0)
    p        = X.shape[1]
    S_B = np.zeros((p, p))
    S_W = np.zeros((p, p))
    for c in classes:
        mask = (y == c)
        n_c  = mask.sum()
        X_c  = X[mask]
        mu_c = X_c.mean(0)
        diff = (mu_c - mean_all).reshape(-1, 1)
        S_B += (n_c / n) * (diff @ diff.T)
        X_cc = X_c - mu_c
        S_W += (X_cc.T @ X_cc) / n
    reg     = eps * np.eye(p)
    eigvals = np.linalg.eigvalsh(np.linalg.solve(S_W + reg, S_B))
    eigvals = np.sort(eigvals)[::-1]
    return -float(eigvals[:d].sum())   # negative because loss = -Tr(...)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fisher optimality diagnostic")
    parser.add_argument("--weights-dir", required=True, metavar="FOLDER")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    weights_dir = args.weights_dir
    if not os.path.isabs(weights_dir):
        weights_dir = os.path.join(get_path("files/results"), weights_dir)

    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg = json.load(f)

    embed_dim = cfg["embed_dim"]
    seeds     = args.seeds or cfg["seeds"]
    variant   = cfg.get("synthetic_variant", "orderedBoth")
    dataset   = cfg["dataset"]
    eps       = cfg.get("fisher_eps", EPS)

    print("=" * 70)
    print("Fisher optimality diagnostic")
    print("Compares learned loss vs optimal (LDA directions) vs upper bound")
    print("=" * 70)

    for seed in seeds:
        data    = load_data(dataset, seed=seed, synthetic_variant=variant)
        X_train = data.X_train.numpy().astype(np.float64)
        y_train = data.y_train.numpy()

        # LDA directions and upper bound — computed once per seed
        B_lda      = get_lda_directions(X_train, y_train, embed_dim)
        L_lda      = fisher_loss_value(B_lda, X_train, y_train, eps)
        L_fp_lda   = fullprefix_fisher_loss_value(B_lda, X_train, y_train, eps)
        L_upper    = upper_bound_fisher(X_train, y_train, embed_dim, eps)

        print(f"\n{'─'*70}")
        print(f"Seed {seed}")
        print(f"  Upper bound  (top-{embed_dim} eigvals of S_W^{{-1}}S_B in x-space): "
              f"{L_upper:.4f}")
        print(f"  LDA optimal  (B = sklearn LDA directions, FisherLoss):        "
              f"{L_lda:.4f}")
        print(f"  LDA optimal  (B = sklearn LDA directions, FullPrefixFisher):  "
              f"{L_fp_lda:.4f}")

        for tag, label in MODEL_TAGS:
            ckpt  = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
            if not os.path.exists(ckpt):
                print(f"\n  [{tag}] checkpoint not found: {ckpt}")
                continue

            model = LinearAE(data.input_dim, embed_dim)
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
            model.eval()
            B_learned = model.get_encoder_matrix().numpy().astype(np.float64)  # (d, p)

            L_fisher   = fisher_loss_value(B_learned, X_train, y_train, eps)
            L_fp       = fullprefix_fisher_loss_value(B_learned, X_train, y_train, eps)
            angle      = mean_principal_angle(B_learned, B_lda)

            if tag == "fisher":
                L_ref, ref_name = L_lda,    "FisherLoss@LDA"
                L_model         = L_fisher
            else:
                L_ref, ref_name = L_fp_lda, "FPFisher@LDA"
                L_model         = L_fp

            gap_pct = 100 * (L_model - L_ref) / (abs(L_ref) + 1e-12)

            print(f"\n  [{label}]")
            print(f"    FisherLoss    (learned B):  {L_fisher:.4f}")
            print(f"    FPFisherLoss  (learned B):  {L_fp:.4f}")
            print(f"    Loss at LDA optimum ({ref_name}): {L_ref:.4f}")
            print(f"    Gap from LDA optimal: {L_model - L_ref:+.4f}  "
                  f"({gap_pct:+.1f}%)")
            print(f"    Subspace angle (learned vs LDA, k={embed_dim}): {angle:.1f}°")
            if abs(gap_pct) < 5.0:
                print(f"    → Loss ≈ optimal: subspace IS correct, "
                      f"angle reflects rotation within LDA subspace only")
            else:
                print(f"    → Loss gap is significant: optimization has NOT "
                      f"converged to the LDA subspace")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
