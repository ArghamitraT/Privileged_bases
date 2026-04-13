"""
Script: experiments/exp6_ortho_mat_ae.py
------------------------------------------
Experiment 6 — Orthogonal Matryoshka Autoencoder ≈ PCA.

Core hypothesis:
    A linear autoencoder with orthogonal encoder columns + Matryoshka
    reconstruction loss should recover PCA eigenvectors and their ordering.

    PCA = orthogonal directions ordered by variance explained.
    That is exactly what Ortho + Mat + Reconstruction gives:
        - Ortho constraint  → forces independent directions (like eigenvectors)
        - Mat loss          → forces ordering by importance (like eigenvalue order)
        - Reconstruction    → finds directions preserving max variance (like PCA)

2×2 ablation design:
    ┌─────────────┬─────────────────────┬──────────────────────┐
    │             │  Standard Recon     │  Matryoshka Recon    │
    ├─────────────┼─────────────────────┼──────────────────────┤
    │  No ortho   │  Vanilla AE         │  Mat AE              │
    │  Ortho      │  Ortho AE           │  Ortho Mat AE ≈ PCA  │
    └─────────────┴─────────────────────┴──────────────────────┘

Key prediction: Only Ortho + Mat recovers both eigenvectors AND ordering.
The others recover the subspace but not the individual basis directions.

Conda environment: mrl_env

Usage:
    python experiments/exp6_ortho_mat_ae.py                                         # full run (digits, embed_dim=10)
    python experiments/exp6_ortho_mat_ae.py --fast                                  # smoke test (fewer epochs)
    python experiments/exp6_ortho_mat_ae.py --use-existing PATH                     # load saved .pt weights, skip training
    python experiments/exp6_ortho_mat_ae.py --use-existing files/results/exprmnt_2026_03_21__21_58_12  # use last saved run

Inputs:  ExpConfig (defaults overridden locally for this experiment)
Outputs: Per-run folder containing —
           experiment_description.log  : what/why/expected + config
           column_alignment.png        : |cos(w_i, pc_i)| per dim, 4 models
           reconstruction_curve.png    : MSE vs prefix k, 5 lines
           explained_variance.png      : cumulative variance explained vs k
           subspace_angle.png          : principal angle between learned vs PCA subspace
           training_curves.png         : loss vs epoch for all 4 models (MANDATORY)
           results_summary.txt         : full numeric tables
           vanilla_ae.pt               : Vanilla AE encoder weights
           mat_ae.pt                   : Mat AE encoder weights
           ortho_ae.pt                 : Ortho AE encoder weights
           ortho_mat_ae.pt             : Ortho Mat AE encoder weights
           runtime.txt                 : total elapsed time
           code_snapshot/              : full copy of code/ at run time
"""

import os

# Must be set before numpy/torch are imported to prevent macOS OMP deadlocks
# that cause DataLoader and subprocess torch inits to hang indefinitely.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import dataclasses
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader

# Allow imports from project root regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot
from data.loader import load_data
from models.linear_ae import LinearAutoencoder


# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET       = "mnist"
EMBED_DIM     = 64
HIDDEN_DIM    = 256
HEAD_MODE     = "shared_head"
EVAL_PREFIXES = list(range(1, 65))   # dense: 1..64
EPOCHS        = 20
PATIENCE      = 10
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
DATA_SEED     = 42
# ==============================================================================


# ==============================================================================
# Helpers
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Master random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[main] Seeds set to {seed}")


def save_experiment_description(cfg: ExpConfig, run_dir: str):
    """
    Write a human-readable experiment log for Experiment 6.

    Args:
        cfg     (ExpConfig): The experiment configuration.
        run_dir (str)      : Path to the run output directory.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 6 — Orthogonal Matryoshka Autoencoder ≈ PCA\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains four linear autoencoders in a 2×2 design:\n"
            "  Vanilla AE    : standard MSE, no orthogonality constraint\n"
            "  Mat AE        : Matryoshka MSE (sum over prefix scales), no ortho\n"
            "  Ortho AE      : standard MSE + orthogonalize() after each step\n"
            "  Ortho Mat AE  : Matryoshka MSE + orthogonalize() after each step\n\n"
            "Also computes PCA baseline (sklearn) as ground truth.\n\n"
            "Evaluates each model with metrics that test how close the learned\n"
            "embedding is to PCA: column alignment, reconstruction error,\n"
            "explained variance, and subspace angle.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "PCA = orthogonal directions ordered by variance explained.\n"
            "Orthogonality alone (Ortho AE) recovers PCA directions but ordering\n"
            "is arbitrary. Matryoshka alone (Mat AE) orders by importance but\n"
            "directions can mix/rotate. Only the combination (Ortho Mat AE)\n"
            "should recover PCA exactly — both eigenvectors AND their ordering.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "- Ortho Mat AE: column alignment ≈ 1.0, reconstruction ≈ PCA\n"
            "- Ortho AE    : good alignment but ordering may differ\n"
            "- Mat AE      : good ordering but directions may be rotated\n"
            "- Vanilla AE  : recovers subspace only (small subspace angle)\n\n"
        )

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[main] Experiment description saved to {log_path}")


# ==============================================================================
# Loss functions
# ==============================================================================

def standard_recon_loss(model: LinearAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    Standard MSE reconstruction loss using the full embedding.

    Args:
        model (LinearAutoencoder): The autoencoder.
        x     (torch.Tensor)     : Input batch, shape (batch_size, input_dim).

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    x_hat = model(x)
    return F.mse_loss(x_hat, x)


def matryoshka_recon_loss(
    model: LinearAutoencoder,
    x: torch.Tensor,
    prefixes: list,
) -> torch.Tensor:
    """
    Matryoshka MSE reconstruction loss — sum of MSE at each prefix scale k.

    At each scale k, encode using only the first k dimensions, then decode
    back. This forces column 1 to be the single most useful reconstruction
    direction (→ PC1), column 2 to add the most on top of column 1 (→ PC2), etc.

    Args:
        model    (LinearAutoencoder): The autoencoder.
        x        (torch.Tensor)     : Input batch, shape (batch_size, input_dim).
        prefixes (list)             : List of prefix sizes k to sum over.

    Returns:
        torch.Tensor: Scalar total loss (sum over all prefix scales).
    """
    total = torch.tensor(0.0, device=x.device, requires_grad=True)
    for k in prefixes:
        zk = model.encode_prefix(x, k)       # (batch, k)
        xk = model.decode_prefix(zk, k)      # (batch, input_dim)
        total = total + F.mse_loss(xk, x)
    return total


# ==============================================================================
# Training loop
# ==============================================================================

def train_autoencoder(
    model: LinearAutoencoder,
    data,
    cfg: ExpConfig,
    use_ortho: bool,
    use_mat: bool,
    model_tag: str,
    run_dir: str,
) -> dict:
    """
    Train a LinearAutoencoder with the specified loss and orthogonality setting.

    Args:
        model     (LinearAutoencoder): Model to train.
        data                         : DataSplit from loader.py (X_train, y_train, etc.)
        cfg       (ExpConfig)        : Experiment config.
        use_ortho (bool)             : If True, call model.orthogonalize() after each step.
        use_mat   (bool)             : If True, use Matryoshka MSE loss; else standard MSE.
        model_tag (str)              : Label for logging (e.g. 'vanilla_ae').
        run_dir   (str)              : Path to run output directory (for log file).

    Returns:
        dict: {
            'train_losses': list of per-epoch train loss,
            'val_losses'  : list of per-epoch val loss,
            'best_epoch'  : int — epoch index (0-based) of best val loss,
        }
    """
    print(f"\n[train] === Training {model_tag} (ortho={use_ortho}, mat={use_mat}) ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    # Build DataLoader from training tensors
    train_dataset = TensorDataset(data.X_train)
    train_loader  = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_epoch   = 0
    best_weights = None
    patience_counter = 0

    log_path = os.path.join(run_dir, f"{model_tag}_train.log")

    print(f"[train] {len(train_loader)} batches/epoch  |  {cfg.epochs} epochs  |  starting ...")

    with open(log_path, "w") as log_f:
        log_f.write(f"=== {model_tag} training log ===\n\n")

        for epoch in range(cfg.epochs):
            # ---- Training phase ----
            model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for (X_batch,) in train_loader:
                optimizer.zero_grad()

                if use_mat:
                    loss = matryoshka_recon_loss(model, X_batch, cfg.eval_prefixes)
                else:
                    loss = standard_recon_loss(model, X_batch)

                loss.backward()
                optimizer.step()

                # Project back onto Stiefel manifold if using ortho constraint
                if use_ortho:
                    model.orthogonalize()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # ---- Validation phase ----
            model.eval()
            with torch.no_grad():
                if use_mat:
                    val_loss = matryoshka_recon_loss(
                        model, data.X_val, cfg.eval_prefixes).item()
                else:
                    val_loss = standard_recon_loss(model, data.X_val).item()
            val_losses.append(val_loss)

            # ---- Early stopping / checkpointing ----
            if val_loss < best_val:
                best_val     = val_loss
                best_epoch   = epoch
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            msg = (f"Epoch {epoch+1:>3}/{cfg.epochs}  "
                   f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}"
                   f"{'  *' if best_epoch == epoch else ''}")
            print(f"[train]   {msg}")
            log_f.write(msg + "\n")

            if cfg.patience and patience_counter >= cfg.patience:
                print(f"[train] Early stopping at epoch {epoch+1}")
                log_f.write(f"Early stopping at epoch {epoch+1}\n")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"[train] Best epoch: {best_epoch+1}  val_loss={best_val:.4f}")

    return {
        "train_losses": train_losses,
        "val_losses"  : val_losses,
        "best_epoch"  : best_epoch,
    }


# ==============================================================================
# Metrics
# ==============================================================================

def compute_column_alignment(model: LinearAutoencoder, pca: PCA) -> np.ndarray:
    """
    Compute absolute cosine similarity between each learned direction and PCA eigenvector.

    For dimension i: |cos(W[i], PC[i])| where W[i] is the i-th row of the
    encoder weight matrix and PC[i] is the i-th PCA component.

    Args:
        model (LinearAutoencoder): Trained model.
        pca   (PCA)              : Fitted sklearn PCA.

    Returns:
        np.ndarray: Shape (embed_dim,) — per-dimension absolute cosine similarity.
    """
    # Use .tolist() to escape the torch-numpy bridge entirely.
    # In this environment (numpy 2.x + torch 2.x), both .numpy() and
    # np.array(tensor, dtype=...) fail due to broken __array__ / ufunc internals
    # when the tensor was loaded from disk via torch.load() + .data= assignment.
    # .tolist() converts to plain Python floats — no bridge involved — and
    # np.array(python_list) then creates a fully clean numpy array.
    W  = np.array(model.encoder.weight.detach().cpu().tolist(), dtype=np.float64)  # (embed_dim, input_dim)
    PC = np.array(pca.components_[:model.embed_dim],            dtype=np.float64)  # (embed_dim, input_dim)

    # Normalise rows to unit length for cosine similarity
    W_norm  = W  / (np.linalg.norm(W,  axis=1, keepdims=True) + 1e-12)
    PC_norm = PC / (np.linalg.norm(PC, axis=1, keepdims=True) + 1e-12)

    # Per-dimension absolute cosine similarity
    alignments = np.abs(np.sum(W_norm * PC_norm, axis=1))  # (embed_dim,)
    return alignments


def compute_reconstruction_curve(
    model: LinearAutoencoder,
    X_test: torch.Tensor,
    prefixes: list,
) -> dict:
    """
    Compute MSE reconstruction error at each prefix k.

    Args:
        model    (LinearAutoencoder): Trained model.
        X_test   (torch.Tensor)     : Test data, shape (N_test, input_dim).
        prefixes (list)             : List of prefix sizes k.

    Returns:
        dict: {k: mse_float} for each k in prefixes.
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for k in prefixes:
            zk   = model.encode_prefix(X_test, k)
            xk   = model.decode_prefix(zk, k)
            mse  = F.mse_loss(xk, X_test).item()
            results[k] = mse
    return results


def compute_pca_reconstruction_curve(
    pca: PCA,
    X_test_np: np.ndarray,
    prefixes: list,
) -> dict:
    """
    Compute MSE reconstruction error at each prefix k for PCA.

    Args:
        pca       (PCA)        : Fitted sklearn PCA.
        X_test_np (np.ndarray) : Test data, shape (N_test, input_dim).
        prefixes  (list)       : List of prefix sizes k.

    Returns:
        dict: {k: mse_float} for each k in prefixes.
    """
    # Full projection into PCA space
    Z_test = pca.transform(X_test_np)   # (N_test, n_components)
    mean   = pca.mean_                  # (input_dim,)
    PC     = pca.components_            # (n_components, input_dim)

    results = {}
    for k in prefixes:
        k_eff = min(k, Z_test.shape[1])
        # Reconstruct using first k components
        X_rec = Z_test[:, :k_eff] @ PC[:k_eff] + mean
        mse   = np.mean((X_rec - X_test_np) ** 2)
        results[k] = float(mse)
    return results


def compute_explained_variance(
    recon_curve: dict,
    X_test: torch.Tensor,
) -> dict:
    """
    Compute explained variance ratio at each prefix k.

    explained_var(k) = 1 - MSE(k) / Var(X_test)

    Args:
        recon_curve (dict)         : {k: mse} from compute_reconstruction_curve.
        X_test      (torch.Tensor) : Test data.

    Returns:
        dict: {k: explained_variance_ratio} for each k.
    """
    total_var = X_test.var().item()
    return {k: 1.0 - mse / (total_var + 1e-12) for k, mse in recon_curve.items()}


def compute_subspace_angle(
    model: LinearAutoencoder,
    pca: PCA,
    prefixes: list,
) -> dict:
    """
    Compute the largest principal angle between the learned subspace and PCA subspace.

    For prefix k: angle between span(W[:k]) and span(PC[:k]).
    A small angle means the model learned the same subspace as PCA (even if
    individual directions differ).

    Args:
        model    (LinearAutoencoder): Trained model.
        pca      (PCA)              : Fitted sklearn PCA.
        prefixes (list)             : List of prefix sizes k.

    Returns:
        dict: {k: angle_in_degrees} for each k in prefixes.
    """
    W  = np.array(model.encoder.weight.detach().cpu().tolist(), dtype=np.float64)  # (embed_dim, input_dim)
    PC = np.array(pca.components_,                              dtype=np.float64)  # (n_components, input_dim)

    results = {}
    for k in prefixes:
        k_eff = min(k, PC.shape[0])

        # Orthonormalise each basis (rows) via QR
        Wk, _  = np.linalg.qr(W[:k_eff].T)    # Wk:  (input_dim, k_eff)
        PCk, _ = np.linalg.qr(PC[:k_eff].T)   # PCk: (input_dim, k_eff)

        # Singular values of Wk^T @ PCk give cosines of principal angles
        sv = np.linalg.svd(Wk.T @ PCk, compute_uv=False)
        sv = np.clip(sv, -1.0, 1.0)

        # Largest angle (worst case)
        max_angle = np.degrees(np.arccos(np.min(sv)))
        results[k] = float(max_angle)

    return results


# ==============================================================================
# Plotting
# ==============================================================================

def plot_column_alignment(alignments: dict, cfg: ExpConfig, run_dir: str):
    """
    Bar chart of per-dimension column alignment (|cos| with PCA eigenvectors).

    Args:
        alignments (dict)   : {model_tag: np.ndarray (embed_dim,)} alignment values.
        cfg        (ExpConfig): For embed_dim and dataset info.
        run_dir    (str)    : Save path.
    """
    D      = cfg.embed_dim
    x      = np.arange(D)
    tags   = ["vanilla_ae", "mat_ae", "ortho_ae", "ortho_mat_ae"]
    labels = ["Vanilla AE", "Mat AE", "Ortho AE", "Ortho Mat AE"]
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    n      = len(tags)
    width  = 0.8 / n

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(10, D), 5))

    for i, (tag, label, color) in enumerate(zip(tags, labels, colors)):
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, alignments[tag], width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(D)])
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("|cos(W[i], PC[i])|", fontsize=12)
    ax.set_title(
        f"Column Alignment with PCA — {cfg.dataset} (embed_dim={cfg.embed_dim})\n"
        "High = learned direction matches PCA eigenvector",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    path = os.path.join(run_dir, "column_alignment.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[main] Column alignment plot saved to {path}")


def plot_reconstruction_curve(recon_curves: dict, cfg: ExpConfig, run_dir: str):
    """
    MSE reconstruction error vs prefix k for all 4 models + PCA.

    Args:
        recon_curves (dict)   : {model_tag: {k: mse}}.
        cfg          (ExpConfig): For prefix list and dataset info.
        run_dir      (str)    : Save path.
    """
    prefixes = sorted(cfg.eval_prefixes)

    tag_label_color = [
        ("vanilla_ae",    "Vanilla AE",   "steelblue",  "o-"),
        ("mat_ae",        "Mat AE",        "darkorange", "s-"),
        ("ortho_ae",      "Ortho AE",      "seagreen",   "^-"),
        ("ortho_mat_ae",  "Ortho Mat AE",  "crimson",    "D-"),
        ("pca",           "PCA baseline",  "purple",     "x--"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    for tag, label, color, style in tag_label_color:
        vals = [recon_curves[tag][k] for k in prefixes]
        ax.plot(prefixes, vals, style, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Prefix size k", fontsize=12)
    ax.set_ylabel("MSE Reconstruction Error", fontsize=12)
    ax.set_title(
        f"Reconstruction Error vs Prefix k\n"
        f"Dataset: {cfg.dataset}  |  embed_dim={cfg.embed_dim}",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(run_dir, "reconstruction_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[main] Reconstruction curve saved to {path}")


def plot_explained_variance(exp_var_curves: dict, cfg: ExpConfig, run_dir: str):
    """
    Cumulative explained variance ratio vs prefix k for all 4 models + PCA.

    Args:
        exp_var_curves (dict)   : {model_tag: {k: explained_var_ratio}}.
        cfg            (ExpConfig): For prefix list and dataset info.
        run_dir        (str)    : Save path.
    """
    prefixes = sorted(cfg.eval_prefixes)

    tag_label_color = [
        ("vanilla_ae",    "Vanilla AE",   "steelblue",  "o-"),
        ("mat_ae",        "Mat AE",        "darkorange", "s-"),
        ("ortho_ae",      "Ortho AE",      "seagreen",   "^-"),
        ("ortho_mat_ae",  "Ortho Mat AE",  "crimson",    "D-"),
        ("pca",           "PCA baseline",  "purple",     "x--"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    for tag, label, color, style in tag_label_color:
        vals = [exp_var_curves[tag][k] for k in prefixes]
        ax.plot(prefixes, vals, style, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Prefix size k", fontsize=12)
    ax.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax.set_title(
        f"Explained Variance vs Prefix k\n"
        f"Dataset: {cfg.dataset}  |  embed_dim={cfg.embed_dim}",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(run_dir, "explained_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[main] Explained variance plot saved to {path}")


def plot_subspace_angle(subspace_angles: dict, cfg: ExpConfig, run_dir: str):
    """
    Principal angle between learned subspace and PCA subspace at each prefix k.

    Args:
        subspace_angles (dict)   : {model_tag: {k: angle_degrees}}.
        cfg             (ExpConfig): For prefix list and dataset info.
        run_dir         (str)    : Save path.
    """
    prefixes = sorted(cfg.eval_prefixes)

    tag_label_color = [
        ("vanilla_ae",    "Vanilla AE",   "steelblue",  "o-"),
        ("mat_ae",        "Mat AE",        "darkorange", "s-"),
        ("ortho_ae",      "Ortho AE",      "seagreen",   "^-"),
        ("ortho_mat_ae",  "Ortho Mat AE",  "crimson",    "D-"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    for tag, label, color, style in tag_label_color:
        vals = [subspace_angles[tag][k] for k in prefixes]
        ax.plot(prefixes, vals, style, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Prefix size k", fontsize=12)
    ax.set_ylabel("Largest Principal Angle (degrees)", fontsize=12)
    ax.set_title(
        f"Subspace Angle vs PCA — {cfg.dataset} (embed_dim={cfg.embed_dim})\n"
        "Small angle = learned subspace matches PCA subspace",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(run_dir, "subspace_angle.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[main] Subspace angle plot saved to {path}")


def plot_training_curves(histories: dict, run_dir: str):
    """
    Loss vs epoch for all 4 models (MANDATORY per project convention).

    Args:
        histories (dict)  : {model_tag: {'train_losses', 'val_losses', 'best_epoch'}}.
        run_dir   (str)   : Save path.
    """
    tags   = ["vanilla_ae", "mat_ae", "ortho_ae", "ortho_mat_ae"]
    labels = ["Vanilla AE", "Mat AE", "Ortho AE", "Ortho Mat AE"]
    colors = ["steelblue",  "darkorange", "seagreen", "crimson"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)

    for idx, (tag, label, color) in enumerate(zip(tags, labels, colors)):
        ax      = axes[idx // 2][idx % 2]
        history = histories[tag]
        epochs  = range(1, len(history["train_losses"]) + 1)
        best_ep = history["best_epoch"] + 1

        ax.plot(epochs, history["train_losses"], "-",  color=color, label="Train", linewidth=2)
        ax.plot(epochs, history["val_losses"],   "--", color=color, label="Val",   linewidth=2, alpha=0.7)
        ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Best ({best_ep})")

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Training Curves — All 4 Models", fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[main] Training curves saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():

    # ------------------------------------------------------------------
    # Step 0: Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Experiment 6: Ortho Mat AE ≈ PCA")
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke-test mode: fewer epochs, digits dataset, embed_dim=10.",
    )
    parser.add_argument(
        "--use-existing", metavar="RUN_DIR",
        help="Load saved weights from a previous run directory and skip training.",
    )
    args = parser.parse_args()
    run_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Configure
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset="digits", embed_dim=10, hidden_dim=128,
            head_mode="shared_head", eval_prefixes=list(range(1, 11)),
            lr=LR, epochs=5, batch_size=BATCH_SIZE, patience=3,
            weight_decay=WEIGHT_DECAY, seed=SEED, data_seed=42,
            experiment_name="exp6_ortho_mat_ae",
        )
        print("[main] --fast mode: digits, embed_dim=10, 5 epochs")
    else:
        cfg = ExpConfig(
            dataset       = DATASET,
            embed_dim     = EMBED_DIM,
            hidden_dim    = HIDDEN_DIM,
            head_mode     = HEAD_MODE,
            eval_prefixes = EVAL_PREFIXES,
            lr            = LR,
            epochs        = EPOCHS,
            batch_size    = BATCH_SIZE,
            patience      = PATIENCE,
            weight_decay  = WEIGHT_DECAY,
            seed          = SEED,
            data_seed     = DATA_SEED,
            experiment_name = "exp6_ortho_mat_ae",
        )

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Step 2: Setup — run directory, experiment description
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[main] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir)

    # ------------------------------------------------------------------
    # Step 3: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # Convert test/train data to numpy for sklearn metrics.
    # Use .tolist() → np.array() to fully escape the torch-numpy bridge.
    # .numpy().copy() is not sufficient in this environment (numpy 2.x +
    # torch 2.x): the bridge corrupts internal numpy metadata so ufuncs
    # (subtraction, mean, linalg, etc.) crash with TypeError.
    X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)
    X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 4: Fit PCA baseline
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Fitting PCA baseline")
    print("=" * 60)
    n_components = min(cfg.embed_dim, X_train_np.shape[0], X_train_np.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.seed)
    pca.fit(X_train_np)

    cumvar = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA fitted: {n_components} components")
    for k in cfg.eval_prefixes:
        idx = min(k, n_components) - 1
        print(f"    k={k:>3}  cumulative_var={cumvar[idx]:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Train 4 models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Training 4 models")
    print("=" * 60)

    model_configs = [
        # (tag,            use_ortho, use_mat)
        ("vanilla_ae",    False,     False),
        ("mat_ae",        False,     True),
        ("ortho_ae",      True,      False),
        ("ortho_mat_ae",  True,      True),
    ]

    models    = {}
    histories = {}

    if args.use_existing:
        # ------------------------------------------------------------------
        # Load weights from a previous run — skip training entirely
        # ------------------------------------------------------------------
        src = args.use_existing
        print(f"[main] Loading saved weights from: {src}")
        for tag, _, _ in model_configs:
            pt_path = os.path.join(src, f"{tag}.pt")
            assert os.path.isfile(pt_path), f"Missing weight file: {pt_path}"
            model = LinearAutoencoder(input_dim=data.input_dim, embed_dim=cfg.embed_dim)
            model.encoder.weight.data = torch.load(pt_path, map_location="cpu")
            models[tag]    = model
            histories[tag] = None   # no training history when loading
            print(f"[main]   Loaded {tag} from {pt_path}")
        print("[main] All weights loaded — skipping training.\n")
    else:
        for tag, use_ortho, use_mat in model_configs:
            set_seeds(cfg.seed)   # same init for all models — fair comparison
            model = LinearAutoencoder(input_dim=data.input_dim, embed_dim=cfg.embed_dim)
            history = train_autoencoder(
                model, data, cfg, use_ortho, use_mat, tag, run_dir
            )
            models[tag]    = model
            histories[tag] = history

            # Save model weights
            pt_path = os.path.join(run_dir, f"{tag}.pt")
            torch.save(model.encoder.weight.data, pt_path)
            print(f"[main] {tag} weights saved to {pt_path}")

    # ------------------------------------------------------------------
    # Step 6: Compute metrics
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Computing metrics")
    print("=" * 60)

    # Column alignment with PCA
    print("[main] Computing column alignment...")
    alignments = {tag: compute_column_alignment(models[tag], pca)
                  for tag, _, _ in model_configs}

    # Reconstruction curves
    print("[main] Computing reconstruction curves...")
    recon_curves = {}
    for tag, _, _ in model_configs:
        recon_curves[tag] = compute_reconstruction_curve(
            models[tag], data.X_test, cfg.eval_prefixes)
    recon_curves["pca"] = compute_pca_reconstruction_curve(
        pca, X_test_np, cfg.eval_prefixes)

    # Explained variance
    print("[main] Computing explained variance...")
    exp_var_curves = {}
    for tag in recon_curves:
        if tag == "pca":
            # Use the original tensor directly — torch.tensor(numpy_array) fails
            # in this environment when the array was created via .tolist() conversion
            # (numpy 2.x dtype inference breaks). compute_explained_variance only
            # needs .var().item() so the original data.X_test tensor is fine.
            exp_var_curves["pca"] = compute_explained_variance(
                recon_curves["pca"],
                data.X_test)
        else:
            exp_var_curves[tag] = compute_explained_variance(
                recon_curves[tag], data.X_test)

    # Subspace angles
    print("[main] Computing subspace angles...")
    subspace_angles = {tag: compute_subspace_angle(models[tag], pca, cfg.eval_prefixes)
                       for tag, _, _ in model_configs}

    # ------------------------------------------------------------------
    # Step 7: Print and save results summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Results summary")
    print("=" * 60)

    summary_path = os.path.join(run_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT 6 — Orthogonal Matryoshka AE ≈ PCA\n")
        f.write("=" * 60 + "\n\n")

        # Column alignment table
        f.write("Column Alignment |cos(W[i], PC[i])|\n")
        f.write("-" * 50 + "\n")
        header = f"{'dim':>4}  " + "  ".join(f"{t:>14}" for t, _, _ in model_configs)
        f.write(header + "\n")
        for d in range(cfg.embed_dim):
            row = f"{d:>4}  " + "  ".join(
                f"{alignments[t][d]:>14.4f}" for t, _, _ in model_configs)
            f.write(row + "\n")
        f.write(f"\nMean alignment:\n")
        for t, _, _ in model_configs:
            f.write(f"  {t:<16}: {alignments[t].mean():.4f}\n")

        # Reconstruction error table
        f.write("\n\nReconstruction MSE at each prefix k\n")
        f.write("-" * 60 + "\n")
        all_tags = [t for t, _, _ in model_configs] + ["pca"]
        header2 = f"{'k':>4}  " + "  ".join(f"{t:>14}" for t in all_tags)
        f.write(header2 + "\n")
        for k in sorted(cfg.eval_prefixes):
            row = f"{k:>4}  " + "  ".join(f"{recon_curves[t][k]:>14.4f}" for t in all_tags)
            f.write(row + "\n")

        # Subspace angle table
        f.write("\n\nSubspace Angle (degrees) at each prefix k\n")
        f.write("-" * 60 + "\n")
        header3 = f"{'k':>4}  " + "  ".join(f"{t:>14}" for t, _, _ in model_configs)
        f.write(header3 + "\n")
        for k in sorted(cfg.eval_prefixes):
            row = f"{k:>4}  " + "  ".join(
                f"{subspace_angles[t][k]:>14.2f}" for t, _, _ in model_configs)
            f.write(row + "\n")

    print(f"[main] Results summary saved to {summary_path}")

    # Also print key numbers
    print("\n--- Mean column alignment per model ---")
    for t, _, _ in model_configs:
        print(f"  {t:<16}: {alignments[t].mean():.4f}")

    # ------------------------------------------------------------------
    # Step 8: Plots
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Plotting")
    print("=" * 60)

    plot_column_alignment(alignments, cfg, run_dir)
    plot_reconstruction_curve(recon_curves, cfg, run_dir)
    plot_explained_variance(exp_var_curves, cfg, run_dir)
    plot_subspace_angle(subspace_angles, cfg, run_dir)
    if any(v is not None for v in histories.values()):
        plot_training_curves(histories, run_dir)
    else:
        print("[main] Skipping training_curves.png — weights loaded from existing run.")

    # ------------------------------------------------------------------
    # Step 9: Save runtime and code snapshot (MANDATORY)
    # ------------------------------------------------------------------
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print("\n[main] Experiment 6 complete.")
    print(f"[main] Runtime: {time.time() - run_start:.2f}s")
    print(f"[main] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
