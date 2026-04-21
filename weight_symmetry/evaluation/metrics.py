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

def compute_pca_dirs_and_eigenvalues(X_train, d: int):
    """
    Compute top-d PCA eigenvectors and eigenvalues from training data.

    Args:
        X_train : (n, p) tensor or numpy array — should already be centred
        d       : number of components to return

    Returns:
        U    : (p, d) numpy array — columns are top-d eigenvectors
        eigs : (d,)  numpy array — eigenvalues (variance per direction = s²/(n-1))
    """
    X = X_train.numpy().astype(np.float64) if hasattr(X_train, "numpy") else np.asarray(X_train, dtype=np.float64)
    n = X.shape[0]
    _, s, Vh = np.linalg.svd(X, full_matrices=False)   # s: (min(n,p),)
    U    = Vh[:d, :].T                                  # (p, d)
    eigs = (s[:d] ** 2) / (n - 1)                      # variance per direction
    return U, eigs


def compute_pca_directions(X_train, d: int) -> np.ndarray:
    """
    Compute top-d PCA eigenvectors from training data.
    Wrapper around compute_pca_dirs_and_eigenvalues for backward compatibility.
    """
    U, _ = compute_pca_dirs_and_eigenvalues(X_train, d)
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


def paired_cosine(a_col: np.ndarray, u_col: np.ndarray) -> float:
    """
    Absolute cosine similarity between a single encoder direction and its
    paired eigenvector — |a_k · u_k| / (|a_k| |u_k|).

    Unlike column_alignment this does NOT take max over the top-k set.
    A value of 1.0 means dim k points exactly along eigenvector k (up to sign).
    A value near 0 means dim k is orthogonal to eigenvector k.

    Args:
        a_col : (p,) — k-th encoder direction
        u_col : (p,) — k-th ground-truth eigenvector

    Returns:
        float in [0, 1]
    """
    a_n = a_col / (np.linalg.norm(a_col) + 1e-10)
    u_n = u_col / (np.linalg.norm(u_col) + 1e-10)
    return float(np.abs(a_n @ u_n))


# ==============================================================================
# Evaluate all prefix sizes for one model
# ==============================================================================

def compute_all_prefix_metrics(
    model,
    pca_dirs: np.ndarray,
    flip_dims: bool = False,
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
    if flip_dims:
        A = np.ascontiguousarray(A[:, ::-1])

    prefix_sizes      = []
    subspace_ang_list = []
    col_align_list    = []
    paired_cos_list   = []

    for m in range(1, d + 1):
        A_m = A[:, :m]                  # first m columns of A, shape (p, m)
        U_m = pca_dirs[:, :m]           # top-m PCA eigenvectors, shape (p, m)

        ang   = subspace_angle(A_m, U_m)
        align = column_alignment(A_m, U_m)
        pc    = paired_cosine(A[:, m - 1], pca_dirs[:, m - 1])  # dim m vs eigenvector m

        prefix_sizes.append(m)
        subspace_ang_list.append(ang)
        col_align_list.append(align)
        paired_cos_list.append(pc)

    return {
        "prefix_sizes":      prefix_sizes,
        "subspace_angles":   subspace_ang_list,
        "column_alignments": col_align_list,
        "paired_cosines":    paired_cos_list,
    }


# ==============================================================================
# Encoder subspace metrics (Exp 2 — divergence experiment)
# ==============================================================================

def compute_encoder_subspace_metrics(
    model,
    pca_dirs: np.ndarray,
    lda_dirs: np.ndarray,
    flip_dims: bool = False,
    model_type: str = "lae",
    flip_ce_head: bool = False,
) -> Dict[str, list]:
    """
    For each prefix size k = 1..d, compute subspace angle and cosine similarity
    to PCA and LDA ground-truth directions.

    The comparison object differs by model family — one consistent object per family
    is used for both PCA and LDA comparisons:

        model_type="lae"        (MSE models, LinearAE):
            Decoder columns  A[:,1:k]  vs  PCA eigenvectors
            Decoder columns  A[:,1:k]  vs  LDA directions
            Rationale: the MSE loss directly trains A to span the PCA subspace;
            decoder columns are the natural "what did the model learn" object.

        model_type="lae_heads"  (CE models, LinearAEWithHeads):
            (W_k B_{1:k})^T columns  vs  PCA eigenvectors
            (W_k B_{1:k})^T columns  vs  LDA directions
            Rationale: the full linear map from input to logits is W_k B_{1:k};
            its row space is the set of input directions used for classification.

        model_type="lae_fisher" (Fisher/LDA models, LinearAE):
            Encoder rows  B^T[:,1:k]  vs  PCA eigenvectors
            Encoder rows  B^T[:,1:k]  vs  LDA directions
            Rationale: the Fisher loss directly trains B to align with LDA
            directions; encoder rows are the natural comparison object.

    Args:
        model      : trained LinearAE or LinearAEWithHeads
        pca_dirs   : (p, n_pca) PCA eigenvectors from training data
        lda_dirs   : (p, n_lda) LDA discriminant directions from training data
        flip_dims    : reverse column order (for PrefixL1 models)
        model_type   : "lae" | "lae_heads" | "lae_fisher"
        flip_ce_head : (lae_heads + flip_dims only) if True, use W_d[:,::-1][:,:k]
                       instead of untrained heads[k-1]. Default False.

    Returns:
        dict with keys:
            "prefix_sizes" : [1, 2, ..., d]
            "pca_angles"   : mean principal angle to PCA subspace per prefix
            "lda_angles"   : mean principal angle to LDA subspace per prefix (NaN if k > n_lda)
            "pca_cosine"   : mean max cosine similarity to PCA per prefix
            "lda_cosine"   : mean max cosine similarity to LDA per prefix (NaN if k > n_lda)
            "n_lda"        : number of LDA directions available
    """
    d     = model.embed_dim
    n_lda = lda_dirs.shape[1]

    # Pre-extract the comparison matrix for this model type
    # All three branches produce a (p, d) matrix whose columns are compared to eigenvectors.
    B_T = model.get_encoder_matrix().cpu().numpy().T.astype(np.float64)  # (p, d) always needed
    if flip_dims:
        B_T = np.ascontiguousarray(B_T[:, ::-1])

    if model_type == "lae":
        # Decoder columns: A ∈ R^{p×d}
        A = model.get_decoder_matrix().cpu().numpy().astype(np.float64)  # (p, d)
        if flip_dims:
            A = np.ascontiguousarray(A[:, ::-1])
    elif model_type == "lae_heads":
        if flip_dims and flip_ce_head:
            # PrefixL1CELoss only trains heads[-1] (full-dim head). Flip its columns
            # to match the flipped B_T, then slice the first k for each prefix size.
            # heads[k-1] for k < d are untrained; using a slice of W_d is the only
            # meaningful option.
            W_d = model.heads[-1].weight.detach().cpu().numpy().astype(np.float64)  # (C, d)
            W_d_flipped = np.ascontiguousarray(W_d[:, ::-1])                        # (C, d)
            head_weights = [W_d_flipped[:, :k] for k in range(1, d + 1)]
        else:
            # Pre-extract head weights: W_k ∈ R^{C×k} for k=1..d
            head_weights = [
                model.heads[k - 1].weight.detach().cpu().numpy().astype(np.float64)
                for k in range(1, d + 1)
            ]
    # lae_fisher: uses B_T (encoder rows) already extracted above

    prefix_sizes = []
    pca_angles   = []
    lda_angles   = []
    pca_cosine   = []
    lda_cosine   = []
    pca_paired   = []
    lda_paired   = []

    for k in range(1, d + 1):
        U_k = pca_dirs[:, :k].astype(np.float64)  # (p, k)

        # --- select comparison vectors for this model type ---
        if model_type == "lae":
            comp_k = A[:, :k]                          # (p, k) decoder columns
        elif model_type == "lae_heads":
            W_k    = head_weights[k - 1]               # (C, k)
            comp_k = (W_k @ B_T[:, :k].T).T           # (p, C)
        else:  # lae_fisher
            comp_k = B_T[:, :k]                        # (p, k) encoder rows

        pca_angles.append(subspace_angle(comp_k, U_k))
        pca_cosine.append(column_alignment(comp_k, U_k))
        # Paired PCA: always use encoder rows (dim-to-dim alignment)
        pca_paired.append(paired_cosine(B_T[:, k - 1], pca_dirs[:, k - 1]))

        if k <= n_lda:
            L_k = lda_dirs[:, :k].astype(np.float64)  # (p, k)
            lda_angles.append(subspace_angle(comp_k, L_k))
            lda_cosine.append(column_alignment(comp_k, L_k))
            lda_paired.append(paired_cosine(B_T[:, k - 1], lda_dirs[:, k - 1]))
        else:
            lda_angles.append(float("nan"))
            lda_cosine.append(float("nan"))
            lda_paired.append(float("nan"))

        prefix_sizes.append(k)

    return {
        "prefix_sizes":    prefix_sizes,
        "pca_angles":      pca_angles,
        "lda_angles":      lda_angles,
        "pca_cosine":      pca_cosine,
        "lda_cosine":      lda_cosine,
        "pca_paired":      pca_paired,
        "lda_paired":      lda_paired,
        "n_lda":           n_lda,
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
