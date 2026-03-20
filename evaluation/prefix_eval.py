"""
Script: evaluation/prefix_eval.py
-----------------------------------
Prefix sweep evaluation for trained encoder + head models.

For each prefix size k in cfg.eval_prefixes:
    1. Compute embeddings for the full test set using the encoder.
    2. Truncate to the first k dimensions (zero-pad or slice depending on head_mode).
    3. Run the classifier head to get logits.
    4. Compute classification accuracy.

Also handles the PCA baseline:
    - Fit PCA on training data to produce embed_dim components.
    - At prefix k, use only the first k PCA components (they are already
      ordered by explained variance, so this is the natural PCA truncation).
    - Train a simple linear probe (LogisticRegression) on the first k
      PCA components of the training set, evaluate on test set.
      (We use a linear probe rather than the neural head because PCA has
      no associated trained head.)

Inputs:
    encoder      (nn.Module)          : trained MLPEncoder
    head         (nn.Module)          : trained SharedClassifier or MultiHeadClassifier
    data         (DataSplit)          : train/val/test tensors from loader.py
    cfg          (ExpConfig)          : experiment config
    model_tag    (str)                : 'standard' or 'mat' — used in result keys

Outputs:
    dict mapping prefix k -> accuracy (float in [0, 1])
    e.g. {1: 0.12, 2: 0.18, 4: 0.35, 8: 0.61, 16: 0.79, 32: 0.87, 64: 0.91}
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Neural model evaluation (standard + matryoshka)
# ==============================================================================

def evaluate_prefix_sweep(
    encoder: nn.Module,
    head: nn.Module,
    data,
    cfg,
    model_tag: str,
) -> Dict[int, float]:
    """
    Sweep over all prefix sizes and compute test accuracy for each.

    Args:
        encoder   (nn.Module) : Trained MLPEncoder.
        head      (nn.Module) : Trained classifier head.
        data      (DataSplit) : Contains X_test, y_test tensors.
        cfg       (ExpConfig) : Uses eval_prefixes and head_mode.
        model_tag (str)       : Label for print output (e.g. 'standard', 'mat').

    Returns:
        Dict[int, float]: Maps each prefix k to test accuracy.
    """
    print(f"\n[prefix_eval] Evaluating '{model_tag}' model ...")

    encoder.eval()
    head.eval()

    results: Dict[int, float] = {}

    with torch.no_grad():

        # --- Step 1: Compute full embeddings for the entire test set ---
        embeddings = encoder(data.X_test)   # shape: (N_test, embed_dim)
        labels     = data.y_test            # shape: (N_test,)

        print(f"  Full embedding shape: {embeddings.shape}")

        # --- Step 2: For each prefix k, truncate and classify ---
        for k in cfg.eval_prefixes:

            if cfg.head_mode == "shared_head":
                # Mode A: zero-pad beyond k, pass full vector to single head
                logits = head.forward_prefix(embeddings, k)

            else:
                # Mode B: slice first k dims, pass to dedicated head_k
                logits = head(embeddings, k)

            # Accuracy: fraction of correct predictions
            preds    = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
            results[k] = accuracy

            print(f"  k={k:>3}  accuracy={accuracy:.4f}")

    print(f"[prefix_eval] '{model_tag}' sweep complete.\n")
    return results


# ==============================================================================
# PCA baseline evaluation
# ==============================================================================

def evaluate_pca_baseline(
    data,
    cfg,
) -> Dict[int, float]:
    """
    PCA baseline: fit PCA on training data, then for each prefix k train a
    linear probe on the first k principal components and evaluate on test set.

    PCA naturally orders components by explained variance, so the first k
    components are always the most informative — this is the natural analogue
    of the Matryoshka prefix property.

    Args:
        data (DataSplit) : Contains X_train, y_train, X_test, y_test tensors.
        cfg  (ExpConfig) : Uses embed_dim and eval_prefixes.

    Returns:
        Dict[int, float]: Maps each prefix k to test accuracy.
    """
    print("[prefix_eval] Evaluating PCA baseline ...")

    # Convert tensors to numpy for sklearn.
    # Use .tolist() -> np.array() to avoid the PyTorch-NumPy bridge (which
    # fails when PyTorch was built without NumPy support).
    X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    # --- Step 1: Fit PCA with embed_dim components on the training set ---
    # This gives us the same number of components as the neural embeddings,
    # ensuring a fair comparison at every prefix level.
    n_components = min(cfg.embed_dim, X_train_np.shape[0], X_train_np.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.seed)
    Z_train = pca.fit_transform(X_train_np)   # shape: (N_train, embed_dim)
    Z_test  = pca.transform(X_test_np)        # shape: (N_test,  embed_dim)

    explained = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA fitted: {n_components} components")
    print(f"  Cumulative explained variance at each prefix:")
    for k in cfg.eval_prefixes:
        idx = min(k, n_components) - 1
        print(f"    k={k:>3}  cumulative_var={explained[idx]:.4f}")

    # --- Step 2: For each prefix k, train a linear probe and evaluate ---
    results: Dict[int, float] = {}

    for k in cfg.eval_prefixes:
        k_eff = min(k, n_components)   # guard: k cannot exceed actual PCA dims

        # Slice first k PCA components
        Z_train_k = Z_train[:, :k_eff]
        Z_test_k  = Z_test[:,  :k_eff]

        # Logistic regression as the linear probe.
        # saga solver scales well to large datasets (e.g. MNIST 48K samples);
        # lbfgs (the default) is very slow at that scale.
        probe = LogisticRegression(
            solver="saga", max_iter=1000, random_state=cfg.seed, n_jobs=-1
        )
        probe.fit(Z_train_k, y_train_np)
        acc = probe.score(Z_test_k, y_test_np)

        results[k] = acc
        print(f"  k={k:>3}  accuracy={acc:.4f}")

    print("[prefix_eval] PCA baseline sweep complete.\n")
    return results


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from config import ExpConfig
    from data.loader import load_data
    from models.encoder import MLPEncoder
    from models.heads import build_head

    # Small config for a fast test
    cfg = ExpConfig(
        dataset="digits",
        embed_dim=16,
        eval_prefixes=[1, 2, 4, 8, 16],
        epochs=1,
    )

    print("--- Loading data ---")
    data = load_data(cfg)

    # Use an untrained encoder — we just want to verify shapes and logic
    print("\n--- Checkpoint: neural prefix sweep (untrained model) ---")
    encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    head    = build_head(cfg, data.n_classes)

    results = evaluate_prefix_sweep(encoder, head, data, cfg, model_tag="test_model")

    assert set(results.keys()) == set(cfg.eval_prefixes), "Missing prefix keys"
    for k, acc in results.items():
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range for k={k}"
    print(f"  Results: {results}")
    print("  PASSED")

    print("\n--- Checkpoint: PCA baseline ---")
    pca_results = evaluate_pca_baseline(data, cfg)

    assert set(pca_results.keys()) == set(cfg.eval_prefixes), "Missing prefix keys"
    for k, acc in pca_results.items():
        assert 0.0 <= acc <= 1.0, f"PCA accuracy out of range for k={k}"
    print(f"  Results: {pca_results}")
    print("  PASSED")

    print("\nAll prefix_eval checks passed.")
