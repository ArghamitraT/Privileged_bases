"""
Script: weight_symmetry/evaluation/metrics.py
----------------------------------------------
Evaluation metrics for weight_symmetry experiments.

Functions:
    compute_pca_directions  : top-d PCA eigenvectors from training data
    subspace_angles         : mean principal angle between two column spans
    column_alignment        : mean max cosine similarity of columns to eigenvectors
    compute_all_prefix_metrics: evaluate both metrics for every prefix m=1..d

Usage:
    from weight_symmetry.evaluation.metrics import (
        compute_pca_directions, compute_all_prefix_metrics
    )
"""

import numpy as np
import torch
from scipy.linalg import subspace_angles as scipy_subspace_angles
from typing import Tuple, Dict


# ==============================================================================
# PCA directions
# ==============================================================================

def compute_pca_directions(X_train: torch.Tensor, d: int) -> np.ndarray:
    """
    Compute top-d PCA eigenvectors from training data.

    Uses SVD of the (centered) data matrix. Data should already be centred
    (StandardScaler was applied in loader).

    Args:
        X_train : (n, p) training data tensor
        d       : number of eigenvectors to return

    Returns:
        U : (p, d) numpy array — columns are top-d eigenvectors, ordered by
            descending eigenvalue (= descending singular value of X_train)
    """
    X = X_train.numpy().astype(np.float64)  # (n, p)
    # SVD: X = U S Vh, right singular vectors Vh rows = eigenvectors of X^T X
    # We want top-d right singular vectors = first d rows of Vh, transposed -> (p, d)
    _, _, Vh = np.linalg.svd(X, full_matrices=False)   # Vh: (min(n,p), p)
    U = Vh[:d, :].T                                     # (p, d)
    return U


# ==============================================================================
# Subspace angle
# ==============================================================================

def subspace_angle(A_cols: np.ndarray, U_cols: np.ndarray) -> float:
    """
    Mean principal angle (degrees) between span(A_cols) and span(U_cols).

    Args:
        A_cols : (p, m) — columns of recovered decoder prefix
        U_cols : (p, m) — columns of ground-truth PCA subspace

    Returns:
        float: mean principal angle in degrees
    """
    # scipy_subspace_angles returns angles in radians, sorted ascending
    angles_rad = scipy_subspace_angles(A_cols, U_cols)
    return float(np.degrees(angles_rad).mean())


# ==============================================================================
# Column alignment
# ==============================================================================

def column_alignment(A_cols: np.ndarray, U_cols: np.ndarray) -> float:
    """
    Mean max absolute cosine similarity between each column of A_cols and
    its nearest column in U_cols.

    Args:
        A_cols : (p, m) — columns of recovered decoder prefix
        U_cols : (p, m) — top-m PCA eigenvectors (same prefix size as A_cols)

    Returns:
        float: mean over columns of A_cols of max |cos similarity| to U_cols
    """
    # Normalise columns
    A_norm = A_cols / (np.linalg.norm(A_cols, axis=0, keepdims=True) + 1e-10)
    U_norm = U_cols / (np.linalg.norm(U_cols, axis=0, keepdims=True) + 1e-10)

    # Cosine similarity matrix: (m, d)
    cos_sim = np.abs(A_norm.T @ U_norm)

    # For each column of A, take max similarity to any PCA eigenvector
    return float(cos_sim.max(axis=1).mean())


# ==============================================================================
# Evaluate all prefix sizes for one model
# ==============================================================================

def compute_all_prefix_metrics(
    model,
    pca_dirs: np.ndarray,
) -> Dict[str, list]:
    """
    For each prefix size m = 1..d, compute subspace angle and column alignment
    between model's decoder A_{1:m} and ground-truth PCA subspace.

    Args:
        model    : trained LinearAE
        pca_dirs : (p, d) PCA eigenvectors from compute_pca_directions

    Returns:
        dict with keys:
            "prefix_sizes"       : [1, 2, ..., d]
            "subspace_angles"    : mean principal angle (degrees) per prefix
            "column_alignments"  : mean max cosine similarity per prefix
    """
    d = model.embed_dim
    A = model.get_decoder_matrix().cpu().numpy().astype(np.float64)  # (p, d)

    prefix_sizes      = []
    subspace_ang_list = []
    col_align_list    = []

    for m in range(1, d + 1):
        A_m = A[:, :m]                  # first m columns of A, shape (p, m)
        U_m = pca_dirs[:, :m]           # top-m PCA eigenvectors, shape (p, m)

        ang   = subspace_angle(A_m, U_m)
        align = column_alignment(A_m, U_m)        # align against top-m eigenvectors only

        prefix_sizes.append(m)
        subspace_ang_list.append(ang)
        col_align_list.append(align)

    return {
        "prefix_sizes":      prefix_sizes,
        "subspace_angles":   subspace_ang_list,
        "column_alignments": col_align_list,
    }


# ==============================================================================
# Encoder subspace metrics (Exp 2 — divergence experiment)
# ==============================================================================

def compute_encoder_subspace_metrics(
    model,
    pca_dirs: np.ndarray,
    lda_dirs: np.ndarray,
    flip_dims: bool = False,
) -> Dict[str, list]:
    """
    For each prefix size k = 1..d, compute:
      - Principal angle between encoder subspace span(B^T[:,1:k]) and top-k PCA dirs
      - Principal angle between encoder subspace span(B^T[:,1:k]) and top-k LDA dirs
        (only for k <= n_lda = lda_dirs.shape[1]; NaN beyond that)

    Uses encoder rows B^T (= encoder.weight.T) as the learned directions in R^p.
    This is consistent for both reconstruction and classification models.

    Args:
        model    : trained LinearAE or LinearAEWithHeads
        pca_dirs : (p, n_pca) PCA eigenvectors from training data
        lda_dirs : (p, n_lda) LDA discriminant directions from training data

    Returns:
        dict with keys:
            "prefix_sizes" : [1, 2, ..., d]
            "pca_angles"   : mean principal angle to PCA subspace per prefix
            "lda_angles"   : mean principal angle to LDA subspace per prefix (NaN if k > n_lda)
            "n_lda"        : number of LDA directions available
    """
    d     = model.embed_dim
    B_T   = model.get_encoder_matrix().cpu().numpy().T.astype(np.float64)  # (p, d)
    if flip_dims:
        # Reverse column order so the most informative direction (last row of B)
        # becomes column 0, matching PrefixL1's reversed-prefix convention.
        B_T = np.ascontiguousarray(B_T[:, ::-1])
    n_lda = lda_dirs.shape[1]

    prefix_sizes = []
    pca_angles   = []
    lda_angles   = []

    for k in range(1, d + 1):
        B_k = B_T[:, :k]                     # (p, k)
        U_k = pca_dirs[:, :k].astype(np.float64)   # (p, k)
        pca_angles.append(subspace_angle(B_k, U_k))

        if k <= n_lda:
            L_k = lda_dirs[:, :k].astype(np.float64)  # (p, k)
            lda_angles.append(subspace_angle(B_k, L_k))
        else:
            lda_angles.append(float("nan"))

        prefix_sizes.append(k)

    return {
        "prefix_sizes": prefix_sizes,
        "pca_angles":   pca_angles,
        "lda_angles":   lda_angles,
        "n_lda":        n_lda,
    }


def compute_prefix_accuracy(model, data, device, model_type: str = "lae",
                            flip_dims: bool = False) -> list:
    """
    For each prefix k = 1..d, compute test classification accuracy.

    For CE models (model_type="lae_heads"):
        Evaluate the trained k-th head directly on test data.

    For reconstruction models (model_type="lae"):
        Extract frozen prefix-k embeddings, fit sklearn LogisticRegression,
        evaluate on test set.

    Args:
        model      : trained model
        data       : DataSplit namedtuple
        device     : torch device
        model_type : "lae" or "lae_heads"

    Returns:
        list of length d: test accuracy at each prefix k=1..d
    """
    from sklearn.linear_model import LogisticRegression

    d          = model.embed_dim
    model.eval()
    accuracies = []

    with torch.no_grad():
        if model_type == "lae_heads" and not flip_dims:
            X_test = data.X_test.to(device)
            y_test = data.y_test.numpy()
            for k in range(1, d + 1):
                logits = model.classify_prefix(X_test, k).cpu().numpy()
                preds  = logits.argmax(axis=1)
                accuracies.append(float((preds == y_test).mean()))
        elif flip_dims:
            # PrefixL1 (rev): extract full embedding, flip dims, logistic regression.
            # Per-prefix heads are not trained under PrefixL1 so we skip them.
            Z_tr = model.encode_prefix(data.X_train.to(device), d).cpu().numpy()
            Z_te = model.encode_prefix(data.X_test.to(device), d).cpu().numpy()
            Z_tr = np.ascontiguousarray(Z_tr[:, ::-1])
            Z_te = np.ascontiguousarray(Z_te[:, ::-1])
            y_tr = data.y_train.numpy()
            y_te = data.y_test.numpy()
            for k in range(1, d + 1):
                clf = LogisticRegression(max_iter=200, n_jobs=1)
                clf.fit(Z_tr[:, :k], y_tr)
                accuracies.append(float(clf.score(Z_te[:, :k], y_te)))
        else:
            # Extract frozen embeddings for all prefix sizes at once
            X_tr = data.X_train.to(device)
            X_te = data.X_test.to(device)
            for k in range(1, d + 1):
                Z_tr = model.get_encoder_matrix()[:k] @ data.X_train.T.to(device)
                Z_tr = Z_tr.T.cpu().numpy()   # (n_train, k)
                Z_te = model.get_encoder_matrix()[:k] @ data.X_test.T.to(device)
                Z_te = Z_te.T.cpu().numpy()   # (n_test, k)
                y_tr = data.y_train.numpy()
                y_te = data.y_test.numpy()

                clf = LogisticRegression(max_iter=200, n_jobs=1)
                clf.fit(Z_tr, y_tr)
                accuracies.append(float(clf.score(Z_te, y_te)))

    return accuracies


# ==============================================================================
# Sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from weight_symmetry.data.loader import load_data
    from weight_symmetry.models.linear_ae import LinearAE

    data = load_data("digits", seed=42)
    d    = 8
    p    = data.input_dim

    print("--- compute_pca_directions ---")
    U = compute_pca_directions(data.X_train, d)
    assert U.shape == (p, d), f"Expected ({p},{d}), got {U.shape}"
    # Columns should be orthonormal
    err = np.abs(U.T @ U - np.eye(d)).max()
    assert err < 1e-6, f"Not orthonormal: {err}"
    print(f"  U shape: {U.shape}  orthonormality err: {err:.2e}  PASSED")

    print("\n--- subspace_angle (same subspace -> 0 degrees) ---")
    ang = subspace_angle(U[:, :4], U[:, :4])
    assert ang < 1e-6, f"Same subspace angle should be 0: {ang}"
    print(f"  Same subspace angle: {ang:.2e} degrees  PASSED")

    print("\n--- column_alignment (perfect alignment -> 1.0) ---")
    align = column_alignment(U[:, :4], U)
    assert align > 0.99, f"Perfect alignment should be ~1: {align}"
    print(f"  Perfect alignment: {align:.4f}  PASSED")

    print("\n--- compute_all_prefix_metrics ---")
    # Use a model whose decoder IS the PCA directions (expect small angles)
    model = LinearAE(input_dim=p, embed_dim=d)
    model.decoder.weight.data = torch.tensor(U.T.astype(np.float32))  # (d, p) ... wait
    # decoder.weight shape is (p, d), so we need U (p, d)
    model.decoder.weight.data = torch.tensor(U.astype(np.float32))    # (p, d)
    results = compute_all_prefix_metrics(model, U)
    avg_angle = np.mean(results["subspace_angles"])
    print(f"  PCA-initialized model: mean subspace angle = {avg_angle:.2f} deg")
    print(f"  prefix_sizes: {results['prefix_sizes'][:5]} ...")
    print("  PASSED")

    print("\nAll metrics checks passed.")
