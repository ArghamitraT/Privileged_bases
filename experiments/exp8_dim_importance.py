"""
Script: experiments/exp8_dim_importance.py
-------------------------------------------
Experiment 8 — Per-Dimension Importance Scoring.

Analyzes which embedding dimensions carry semantic information for
Standard, L1, MRL, and PCA models using three complementary methods.

Three analyses:
  1. Importance scoring (3 methods per model):
       mean_abs[d]  = mean(|z[:, d]|) over all test samples
       variance[d]  = var(z[:, d]) over all test samples
       probe_acc[d] = logistic regression accuracy using only dimension d

  2. Best-k vs First-k accuracy:
       For each k, compare LR on first-k dims (standard prefix eval) vs
       top-k dims ranked by each importance method (oracle selection).
       Gap ≈ 0 for MRL (ordering enforced); Gap > 0 for Standard/L1.

  3. Method agreement:
       Spearman rank correlation between each pair of the 3 methods.
       High rho → importance is robustly measurable.

Inputs:
  --fast            : smoke test — digits, embed_dim=16, 5 epochs, 500 probe samples
  --use-weights     : path to exp7 or exp10 run folder; loads Standard/L1/MRL weights
  --embed-dim N     : override embed_dim (8, 16, 32, 64); derives eval_prefixes automatically

Outputs (all in a new timestamped run folder):
  importance_scores.png       : per-dim bar charts, 3 methods x 4 models
  dim_importance_heatmap.png  : heatmap models x dims, one panel per method
  best_vs_first_k.png         : first-k vs best-k accuracy curves per model
  method_agreement.png        : scatter plots + Spearman rho per model x pair
  training_curves.png         : loss vs epoch (MANDATORY; placeholder if --use-weights)
  results_summary.txt
  experiment_description.log
  runtime.txt
  code_snapshot/

Usage:
    python experiments/exp8_dim_importance.py --fast                       # smoke test (digits, 5 epochs)
    python experiments/exp8_dim_importance.py                              # full run (MNIST, 20 epochs)
    python experiments/exp8_dim_importance.py --use-weights PATH           # load weights, no retrain
    python experiments/exp8_dim_importance.py --embed-dim 8 --use-weights PATH   # at dim=8
    python experiments/exp8_dim_importance.py --embed-dim 16 --use-weights PATH  # at dim=16
    python experiments/exp8_dim_importance.py --embed-dim 32 --use-weights PATH  # at dim=32
    python tests/run_tests_exp8.py --fast                                  # unit tests only
    python tests/run_tests_exp8.py                                         # unit tests + e2e smoke
"""

import os

# Cap BLAS thread count before numpy/scipy imports to prevent deadlocks on macOS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse
import dataclasses

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot, get_path, load_config_json
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head

# Reuse training helper and embedding extractor from exp7 (no duplication)
from experiments.exp7_mrl_vs_ff import train_single_model, get_embeddings_np


# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET           = "mnist"
EMBED_DIM         = 64
HIDDEN_DIM        = 256
HEAD_MODE         = "shared_head"
EVAL_PREFIXES     = list(range(1, EMBED_DIM + 1))  # always derived from EMBED_DIM
EPOCHS            = 20
PATIENCE          = 5
LR                = 1e-3
BATCH_SIZE        = 128
WEIGHT_DECAY      = 1e-4
SEED              = 42
L1_LAMBDA         = 0.05
MAX_PROBE_SAMPLES = 2000   # max samples used for logistic-regression importance probes
# Path to an exp7 or exp10 output folder to load weights from (skips training).
# Leave as "" to train from scratch. CLI --use-weights overrides this.
USE_WEIGHTS       = "exprmnt_2026_04_01__22_04_54"
# ==============================================================================


# ==============================================================================
# Module-level constants
# ==============================================================================

MODEL_NAMES = ["Standard", "L1", "MRL", "PrefixL1", "PCA"]

MODEL_COLORS = {
    "Standard": "steelblue",
    "L1":       "orchid",
    "MRL":      "darkorange",
    "PrefixL1": "crimson",
    "PCA":      "seagreen",
}

IMPORTANCE_METHODS = ["mean_abs", "variance", "probe_acc"]

METHOD_LABELS = {
    "mean_abs":  "Mean |z|",
    "variance":  "Variance",
    "probe_acc": "1D Probe Acc",
}

# All unique method pairs for agreement analysis
METHOD_PAIRS = [
    ("mean_abs",  "variance"),
    ("mean_abs",  "probe_acc"),
    ("variance",  "probe_acc"),
]

PAIR_LABELS = {
    ("mean_abs",  "variance"):  "mean|z| vs Var",
    ("mean_abs",  "probe_acc"): "mean|z| vs Probe",
    ("variance",  "probe_acc"): "Var vs Probe",
}


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for numpy, torch, and python random.

    Args:
        seed (int): Master random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[exp8] Random seeds set to {seed}")


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(cfg, run_dir, weights_dir, fast):
    """
    Write a human-readable log describing this experiment run.

    Args:
        cfg         (ExpConfig)  : Experiment configuration.
        run_dir     (str)        : Output directory for this run.
        weights_dir (str | None) : Path to loaded weights (exp7 or exp10), or None.
        fast        (bool)       : Whether fast/smoke mode is active.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 8 — Per-Dimension Importance Scoring\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Analyzes which embedding dimensions carry semantic information for\n"
            "Standard, L1, MRL, and PCA models using three complementary methods:\n\n"
            "  Analysis 1 — Importance scoring (3 methods):\n"
            "    mean_abs[d]  = mean(|z[:, d]|) -- activation magnitude\n"
            "    variance[d]  = var(z[:, d])    -- spread of activations\n"
            "    probe_acc[d] = logistic regression accuracy using only dim d\n\n"
            "  Analysis 2 — Best-k vs First-k accuracy:\n"
            "    For each k, compare LR on first-k dims vs top-k dims ranked by\n"
            "    each importance method. Gap = best_k_acc - first_k_acc.\n"
            "    Gap ~= 0 for MRL (ordering enforced); Gap > 0 for L1/Standard.\n\n"
            "  Analysis 3 — Method agreement:\n"
            "    Spearman rank correlation between each pair of the 3 methods.\n"
            "    High rho -> importance is robustly measurable.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Exp7 showed MRL beats L1 at prefix eval. This asks WHY:\n"
            "  - MRL enforces front-loading (dim 0 most informative, then dim 1, ...)\n"
            "  - L1 creates sparse embeddings but scatters important dims randomly.\n"
            "The best-k vs first-k gap directly quantifies the ordering property.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "  MRL:      gap ~= 0 at all k  (first-k IS the best-k)\n"
            "  L1:       gap > 0 at small k  (important dims not front-loaded)\n"
            "  Standard: gap > 0             (no ordering enforced)\n"
            "  PCA:      gap ~= 0            (variance-ordered by construction)\n\n"
            "  mean_abs and variance should agree strongly for L1 (L1 kills both).\n"
            "  probe_acc may capture different signal for Standard/MRL.\n\n"
        )

        f.write("WEIGHTS SOURCE\n")
        f.write("-" * 40 + "\n")
        if weights_dir:
            f.write(f"  Loaded from: {weights_dir}\n\n")
        else:
            f.write("  Trained from scratch in this run.\n\n")
        f.write(f"  Fast mode: {fast}\n\n")

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for fi in dataclasses.fields(cfg):
            f.write(f"  {fi.name:<20} = {getattr(cfg, fi.name)}\n")
        f.write("\n")

    print(f"[exp8] Experiment description saved to {log_path}")


# ==============================================================================
# PCA baseline embeddings
# ==============================================================================

def get_pca_embeddings_np(data, cfg):
    """
    Fit PCA on training data and return train and test embeddings as numpy arrays.

    Uses up to embed_dim principal components. Zero-pads if fewer components
    are available (e.g., n_features < embed_dim).

    Args:
        data (DataSplit) : Train/test splits with X_train, X_test tensors.
        cfg  (ExpConfig) : Uses embed_dim and seed.

    Returns:
        Tuple (Z_train, Z_test):
            Z_train (np.ndarray): shape (n_train, embed_dim)
            Z_test  (np.ndarray): shape (n_test,  embed_dim)
    """
    X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
    X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)

    n_comp = min(cfg.embed_dim, X_train_np.shape[0], X_train_np.shape[1])
    print(f"[exp8] Fitting PCA with n_components={n_comp}  (embed_dim={cfg.embed_dim})")

    pca     = SklearnPCA(n_components=n_comp, random_state=cfg.seed)
    Z_train = pca.fit_transform(X_train_np)
    Z_test  = pca.transform(X_test_np)

    # Zero-pad to embed_dim if PCA returned fewer components
    if n_comp < cfg.embed_dim:
        pad     = cfg.embed_dim - n_comp
        Z_train = np.pad(Z_train, ((0, 0), (0, pad)))
        Z_test  = np.pad(Z_test,  ((0, 0), (0, pad)))

    print(f"[exp8] PCA embeddings: train={Z_train.shape}, test={Z_test.shape}")
    return Z_train.astype(np.float32), Z_test.astype(np.float32)


# ==============================================================================
# Analysis 1 — Importance scoring (3 methods)
# ==============================================================================

def compute_importance_scores(Z_test, Z_train, y_train, y_test,
                               max_probe_samples, seed, model_tag):
    """
    Compute three per-dimension importance scores for one model's embeddings.

    Method 1 — mean_abs:  mean(|z[:, d]|). Free — one numpy call.
    Method 2 — variance:  var(z[:, d]).    Free — one numpy call.
    Method 3 — probe_acc: accuracy of a 1D logistic regression probe per dim.
                          Loops over all dims; subsamples data if needed.

    Args:
        Z_test            (np.ndarray): Test embeddings,  shape (n_test,  embed_dim).
        Z_train           (np.ndarray): Train embeddings, shape (n_train, embed_dim).
        y_train           (np.ndarray): Train labels, shape (n_train,).
        y_test            (np.ndarray): Test labels,  shape (n_test,).
        max_probe_samples (int)       : Cap on samples used for per-dim LR probe.
        seed              (int)       : For LogisticRegression reproducibility.
        model_tag         (str)       : Label for print output.

    Returns:
        dict with keys:
            "mean_abs"  (np.ndarray): shape (embed_dim,) — mean absolute activation
            "variance"  (np.ndarray): shape (embed_dim,) — activation variance
            "probe_acc" (np.ndarray): shape (embed_dim,) — single-dim LR test accuracy
    """
    embed_dim = Z_test.shape[1]
    print(f"[exp8] Importance scores for {model_tag}  (embed_dim={embed_dim})")

    # --- Method 1: Mean absolute activation (free) ---
    mean_abs = np.abs(Z_test).mean(axis=0)

    # --- Method 2: Variance (free) ---
    variance = Z_test.var(axis=0)

    # --- Method 3: Per-dim 1D logistic probe ---
    # Subsample to max_probe_samples for speed if dataset is large
    rng = np.random.default_rng(seed)

    if len(Z_train) > max_probe_samples:
        idx_tr  = rng.choice(len(Z_train), max_probe_samples, replace=False)
        Ztr_sub = Z_train[idx_tr]
        ytr_sub = y_train[idx_tr]
    else:
        Ztr_sub, ytr_sub = Z_train, y_train

    if len(Z_test) > max_probe_samples:
        idx_te  = rng.choice(len(Z_test), max_probe_samples, replace=False)
        Zte_sub = Z_test[idx_te]
        yte_sub = y_test[idx_te]
    else:
        Zte_sub, yte_sub = Z_test, y_test

    probe_acc = np.zeros(embed_dim, dtype=np.float32)
    for d in range(embed_dim):
        if d % 8 == 0:
            print(f"  [probe] {model_tag}  dim {d}/{embed_dim} ...")
        try:
            lr = LogisticRegression(
                solver="saga", max_iter=500, random_state=seed, n_jobs=1,
            )
            lr.fit(Ztr_sub[:, d:d+1], ytr_sub)
            probe_acc[d] = float(lr.score(Zte_sub[:, d:d+1], yte_sub))
        except Exception:
            # Degenerate dim (constant/all-zero): fallback to 0
            probe_acc[d] = 0.0

    top3 = np.argsort(probe_acc)[::-1][:3].tolist()
    print(f"  [probe] Done. top-3 dims by probe_acc: {top3}")

    return {
        "mean_abs":  mean_abs.astype(np.float32),
        "variance":  variance.astype(np.float32),
        "probe_acc": probe_acc,
    }


# ==============================================================================
# Analysis 2 — Best-k vs First-k
# ==============================================================================

def compute_best_vs_first_k(Z_train, Z_test, y_train, y_test,
                              importance_scores, eval_prefixes, seed, model_tag):
    """
    Compare logistic regression accuracy on first-k dims vs best-k dims.

    For each k in eval_prefixes and each importance method, rank dims by
    importance descending and take the top-k. Compare to the standard
    prefix evaluation (always first k columns).

    Gap = best_k_acc - first_k_acc.
    MRL should have gap ~= 0; L1 and Standard should have gap > 0 at small k.

    Args:
        Z_train           (np.ndarray) : Train embeddings (n_train, embed_dim).
        Z_test            (np.ndarray) : Test embeddings  (n_test,  embed_dim).
        y_train           (np.ndarray) : Train labels.
        y_test            (np.ndarray) : Test labels.
        importance_scores (dict)       : Output of compute_importance_scores().
        eval_prefixes     (list[int])  : Prefix sizes to evaluate.
        seed              (int)        : For LogisticRegression.
        model_tag         (str)        : Label for print output.

    Returns:
        dict with keys:
            "first_k"           (dict[int, float]): {k: acc using first-k dims}
            "best_k_mean_abs"   (dict[int, float]): {k: acc using top-k by mean_abs}
            "best_k_variance"   (dict[int, float]): {k: acc using top-k by variance}
            "best_k_probe_acc"  (dict[int, float]): {k: acc using top-k by probe_acc}
    """
    print(f"\n[exp8] Best-k vs First-k for {model_tag} ...")

    # Precompute sorted dim indices for each method (descending importance)
    sorted_dims = {
        m: np.argsort(importance_scores[m])[::-1]
        for m in IMPORTANCE_METHODS
    }

    results = {
        "first_k":          {},
        "best_k_mean_abs":  {},
        "best_k_variance":  {},
        "best_k_probe_acc": {},
    }

    for k in eval_prefixes:
        # --- First-k: always columns 0..k-1 ---
        lr_first = LogisticRegression(
            solver="saga", max_iter=1000, random_state=seed, n_jobs=1,
        )
        lr_first.fit(Z_train[:, :k], y_train)
        acc_first = float(lr_first.score(Z_test[:, :k], y_test))
        results["first_k"][k] = acc_first

        # --- Best-k: top-k dims ranked by each importance method ---
        for method in IMPORTANCE_METHODS:
            top_k = sorted_dims[method][:k]
            lr_best = LogisticRegression(
                solver="saga", max_iter=1000, random_state=seed, n_jobs=1,
            )
            lr_best.fit(Z_train[:, top_k], y_train)
            acc_best = float(lr_best.score(Z_test[:, top_k], y_test))
            results[f"best_k_{method}"][k] = acc_best

        gap_ma = results["best_k_mean_abs"][k]  - acc_first
        gap_pa = results["best_k_probe_acc"][k] - acc_first
        print(f"  k={k:>3}  first_k={acc_first:.4f}  "
              f"best(mean_abs)={results['best_k_mean_abs'][k]:.4f} gap={gap_ma:+.4f}  "
              f"best(probe)={results['best_k_probe_acc'][k]:.4f} gap={gap_pa:+.4f}")

    return results


# ==============================================================================
# Analysis 3 — Method agreement (Spearman correlation)
# ==============================================================================

def compute_method_agreement(importance_scores, model_tag):
    """
    Compute Spearman rank correlation between each pair of importance methods.

    Args:
        importance_scores (dict) : Output of compute_importance_scores().
        model_tag         (str)  : Label for print output.

    Returns:
        dict: {(method_a, method_b): spearman_rho} for all 3 unique pairs.
    """
    agreement = {}
    for (ma, mb) in METHOD_PAIRS:
        result = spearmanr(importance_scores[ma], importance_scores[mb])
        # scipy >= 1.7 uses .statistic; older scipy uses .correlation
        rho = result.statistic if hasattr(result, "statistic") else result.correlation
        agreement[(ma, mb)] = float(rho)
        print(f"  [agree] {model_tag}  {ma} vs {mb}  rho={rho:.4f}")
    return agreement


# ==============================================================================
# Weight loading from a saved exp7 run
# ==============================================================================

def detect_arch_from_weights(weights_dir):
    """
    Infer embed_dim and hidden_dim from saved encoder weights.

    Reads the standard encoder checkpoint and inspects fc_out.weight:
        shape = (embed_dim, hidden_dim)

    This lets exp8 auto-configure itself when --use-weights is given,
    without requiring the user to also pass --embed-dim.

    Args:
        weights_dir (str): Path to an exp7 or exp10 output folder.

    Returns:
        Tuple[int, int]: (embed_dim, hidden_dim) inferred from the weights.

    Raises:
        FileNotFoundError: If standard_encoder_best.pt is not found.
    """
    enc_path = os.path.join(weights_dir, "standard_encoder_best.pt")
    if not os.path.isfile(enc_path):
        raise FileNotFoundError(
            f"Cannot detect architecture — file not found: {enc_path}"
        )
    state_dict = torch.load(enc_path, map_location="cpu")
    # fc_out.weight has shape (embed_dim, hidden_dim)
    embed_dim  = state_dict["fc_out.weight"].shape[0]
    hidden_dim = state_dict["fc_out.weight"].shape[1]
    print(f"[exp8] Auto-detected from weights: embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    return embed_dim, hidden_dim


def load_models_from_dir(weights_dir, cfg, data):
    """
    Load Standard, L1, MRL, and (optionally) PrefixL1 encoder+head weights
    from a saved run folder.

    Accepts output directories from exp7 or exp10. Weight filenames:
        standard_encoder_best.pt / standard_head_best.pt
        l1_encoder_best.pt       / l1_head_best.pt
        mat_encoder_best.pt      / mat_head_best.pt
        pl1_encoder_best.pt      / pl1_head_best.pt  (exp10 runs only)

    PrefixL1 is optional — if not found (e.g. older exp7 runs), None is
    returned for those and PrefixL1 is omitted from downstream analysis.

    Args:
        weights_dir (str)       : Path to exp7 or exp10 output folder.
        cfg         (ExpConfig) : Used to build matching architecture.
        data        (DataSplit) : Used for input_dim and n_classes.

    Returns:
        Tuple: (std_encoder, std_head, l1_encoder, l1_head,
                mat_encoder, mat_head, pl1_encoder, pl1_head)
               pl1_encoder and pl1_head are None when weights not found.
               All others are in eval mode.

    Raises:
        FileNotFoundError: If Standard, L1, or MRL weight files are missing.
    """
    required = [
        "standard_encoder_best.pt", "standard_head_best.pt",
        "l1_encoder_best.pt",       "l1_head_best.pt",
        "mat_encoder_best.pt",      "mat_head_best.pt",
    ]
    for fname in required:
        fpath = os.path.join(weights_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Expected weight file not found: {fpath}\n"
                "Make sure --use-weights points to a valid exp7 or exp10 output folder."
            )

    def _load(enc_file, head_file):
        enc  = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        head = build_head(cfg, data.n_classes)
        enc.load_state_dict( torch.load(os.path.join(weights_dir, enc_file),  map_location="cpu"))
        head.load_state_dict(torch.load(os.path.join(weights_dir, head_file), map_location="cpu"))
        enc.eval()
        head.eval()
        return enc, head

    print(f"[exp8] Loading weights from {weights_dir}")
    std_enc, std_hd = _load("standard_encoder_best.pt", "standard_head_best.pt")
    l1_enc,  l1_hd  = _load("l1_encoder_best.pt",       "l1_head_best.pt")
    mat_enc, mat_hd = _load("mat_encoder_best.pt",       "mat_head_best.pt")

    # PrefixL1 weights are optional (present in exp10 runs, absent in exp7 runs)
    pl1_enc_path = os.path.join(weights_dir, "pl1_encoder_best.pt")
    if os.path.isfile(pl1_enc_path):
        pl1_enc, pl1_hd = _load("pl1_encoder_best.pt", "pl1_head_best.pt")
        print("[exp8] PrefixL1 weights found and loaded.")
    else:
        pl1_enc, pl1_hd = None, None
        print("[exp8] PrefixL1 weights not found in this folder — skipping PrefixL1.")

    print("[exp8] Weights loaded successfully.")
    return std_enc, std_hd, l1_enc, l1_hd, mat_enc, mat_hd, pl1_enc, pl1_hd


# ==============================================================================
# Plotting — training curves (MANDATORY)
# ==============================================================================

def plot_training_curves(run_dir, model_tags):
    """
    Parse training log files and plot train/val loss vs epoch.

    If no log files are found (e.g. --use-weights skips training), saves a
    placeholder PNG so the mandatory training_curves.png is always present.

    Args:
        run_dir    (str)       : Output directory containing log files.
        model_tags (list[str]) : Tags to attempt to read (e.g. ['standard','l1','mat']).
    """
    histories = {}
    for tag in model_tags:
        log_path = os.path.join(run_dir, f"{tag}_train.log")
        if not os.path.isfile(log_path):
            continue
        train_losses, val_losses = [], []
        with open(log_path) as fh:
            for line in fh:
                if "train_loss=" in line and "val_loss=" in line:
                    try:
                        tl = float(line.split("train_loss=")[1].split()[0])
                        vl = float(line.split("val_loss=")[1].split()[0])
                        train_losses.append(tl)
                        val_losses.append(vl)
                    except (IndexError, ValueError):
                        pass
        if train_losses:
            histories[tag] = {"train": train_losses, "val": val_losses}

    # Placeholder when no training logs exist (--use-weights mode)
    if not histories:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                "Weights loaded from existing run —\nno training curves for this run.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_curves.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("[exp8] Saved training_curves.png  (placeholder — no training logs)")
        return

    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for col, (tag, hist) in enumerate(histories.items()):
        ax     = axes[0, col]
        epochs = range(1, len(hist["train"]) + 1)
        ax.plot(epochs, hist["train"], label="Train", linewidth=2)
        ax.plot(epochs, hist["val"],   label="Val",   linewidth=2, linestyle="--")
        ax.set_title(tag, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("Training Curves — Standard / L1 / MRL", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp8] Saved training_curves.png")


# ==============================================================================
# Plotting — Figure 1: per-dim bar charts (3 methods x 4 models)
# ==============================================================================

def plot_importance_scores(all_scores, run_dir, cfg):
    """
    Save importance_scores.png: 3-method x 4-model grid of per-dim bar charts.

    Rows = importance methods (mean_abs, variance, probe_acc).
    Columns = models (Standard, L1, MRL, PCA).
    Each subplot is a horizontal bar chart with dims on the y-axis.

    Args:
        all_scores (dict): {model_name: {"mean_abs": arr, "variance": arr,
                            "probe_acc": arr}}.
        run_dir    (str) : Output directory.
        cfg        (ExpConfig): Uses embed_dim.
    """
    embed_dim = cfg.embed_dim
    n_methods = len(IMPORTANCE_METHODS)
    n_models  = len(MODEL_NAMES)
    tick_step = 4 if embed_dim > 16 else 1

    fig, axes = plt.subplots(
        n_methods, n_models,
        figsize=(5 * n_models, max(3, embed_dim * 0.22) * n_methods),
        squeeze=False,
    )

    dim_labels = [f"d{d}" for d in range(embed_dim)]

    for row, method in enumerate(IMPORTANCE_METHODS):
        for col, model_name in enumerate(MODEL_NAMES):
            ax     = axes[row, col]
            scores = all_scores[model_name][method]
            dims   = np.arange(embed_dim)
            color  = MODEL_COLORS[model_name]

            ax.barh(dims, scores, color=color, alpha=0.8)
            ax.set_xlim(left=0)
            ax.invert_yaxis()  # dim 0 at top
            ax.set_xlabel(METHOD_LABELS[method], fontsize=9)

            visible = [d for d in dims if d % tick_step == 0]
            ax.set_yticks(visible)
            ax.set_yticklabels([dim_labels[d] for d in visible], fontsize=7)

            if row == 0:
                ax.set_title(model_name, fontsize=11,
                             color=color, fontweight="bold")
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=10)

    fig.suptitle("Per-Dimension Importance Scores — All Models & Methods",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "importance_scores.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp8] Saved importance_scores.png")


# ==============================================================================
# Plotting — Figure 2: heatmap (3 methods, each: 4 models x embed_dim dims)
# ==============================================================================

def plot_dim_importance_heatmap(all_scores, run_dir, cfg):
    """
    Save dim_importance_heatmap.png: one heatmap panel per method.

    Each panel is a (n_models x embed_dim) matrix. Importance values are
    normalized per method to [0, 1] so models are comparable.
    Color map: RdYlGn (low=red, high=green).

    Args:
        all_scores (dict): {model_name: {method: np.ndarray}}.
        run_dir    (str) : Output directory.
        cfg        (ExpConfig): Uses embed_dim.
    """
    embed_dim = cfg.embed_dim
    n_methods = len(IMPORTANCE_METHODS)
    annotate  = embed_dim <= 16
    tick_step = 4 if embed_dim > 16 else 1

    fig, axes = plt.subplots(
        n_methods, 1,
        figsize=(max(10, embed_dim * 0.4), 4 * n_methods),
        squeeze=False,
    )

    for row, method in enumerate(IMPORTANCE_METHODS):
        ax = axes[row, 0]

        # Stack: shape (n_models, embed_dim)
        matrix = np.vstack([
            all_scores[model_name][method] for model_name in MODEL_NAMES
        ])

        # Normalize each row independently to [0, 1]
        row_min = matrix.min(axis=1, keepdims=True)
        row_max = matrix.max(axis=1, keepdims=True)
        norm    = (matrix - row_min) / (row_max - row_min + 1e-8)

        img = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(img, ax=ax, fraction=0.02, pad=0.04)

        ax.set_yticks(range(len(MODEL_NAMES)))
        ax.set_yticklabels(MODEL_NAMES, fontsize=9)

        visible_x = [d for d in range(embed_dim) if d % tick_step == 0]
        ax.set_xticks(visible_x)
        ax.set_xticklabels([str(d) for d in visible_x], fontsize=7)
        ax.set_xlabel("Embedding Dimension", fontsize=9)
        ax.set_title(METHOD_LABELS[method], fontsize=11)

        # Annotate cells with normalized value when embed_dim is small
        if annotate:
            for r in range(len(MODEL_NAMES)):
                for c in range(embed_dim):
                    val       = norm[r, c]
                    txt_color = "black" if 0.25 < val < 0.75 else "white"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=txt_color)

    fig.suptitle("Dimension Importance Heatmap (normalized per method x model)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim_importance_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp8] Saved dim_importance_heatmap.png")


# ==============================================================================
# Plotting — Figure 3: best-k vs first-k accuracy (1 row x 4 cols)
# ==============================================================================

def plot_best_vs_first_k(all_gap_results, run_dir, cfg):
    """
    Save best_vs_first_k.png: accuracy curves per model (4 panels side by side).

    Each panel shows 4 lines:
      - first_k (black solid): standard prefix eval
      - best_k_mean_abs  (blue dashed):  oracle top-k by mean |z|
      - best_k_variance  (red dashed):   oracle top-k by variance
      - best_k_probe_acc (green dashed): oracle top-k by 1D probe accuracy

    Small gap between first_k and best_k => strong dimension ordering.

    Args:
        all_gap_results (dict): {model_name: {curve_key: {k: acc}}}.
        run_dir         (str) : Output directory.
        cfg             (ExpConfig): Uses eval_prefixes.
    """
    eval_prefixes = cfg.eval_prefixes
    n_models      = len(MODEL_NAMES)

    CURVE_STYLES = {
        "first_k":          ("black",     "-",  "First-k (prefix eval)",  2.5),
        "best_k_mean_abs":  ("steelblue", "--", "Best-k (mean |z|)",      1.8),
        "best_k_variance":  ("firebrick", "--", "Best-k (variance)",      1.8),
        "best_k_probe_acc": ("seagreen",  "--", "Best-k (probe acc)",     1.8),
    }

    fig, axes = plt.subplots(
        1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False,
    )

    for col, model_name in enumerate(MODEL_NAMES):
        ax    = axes[0, col]
        color = MODEL_COLORS[model_name]
        data  = all_gap_results[model_name]

        for curve_key, (c, ls, label, lw) in CURVE_STYLES.items():
            accs = [data[curve_key].get(k, float("nan")) for k in eval_prefixes]
            ax.plot(eval_prefixes, accs, linestyle=ls, color=c,
                    label=label, linewidth=lw, marker="o", markersize=5)

        ax.set_xscale("log", base=2)
        ax.set_xticks(eval_prefixes)
        ax.set_xticklabels([str(k) for k in eval_prefixes])
        ax.set_xlabel("Prefix size k", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(model_name, fontsize=12, color=color, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("Logistic Regression Accuracy", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Best-k vs First-k Accuracy  (gap ≈ 0 for MRL → ordering enforced)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "best_vs_first_k.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp8] Saved best_vs_first_k.png")


# ==============================================================================
# Plotting — Figure 4: method agreement scatter (4 models x 3 method pairs)
# ==============================================================================

def plot_method_agreement(all_scores, all_agreement, run_dir, cfg):
    """
    Save method_agreement.png: scatter plots + Spearman rho annotations.

    Grid: rows = models, columns = method pairs.
    Each scatter has one point per embedding dimension.
    Spearman rho is annotated in the upper-left corner.

    Args:
        all_scores    (dict): {model_name: {method: np.ndarray}}.
        all_agreement (dict): {model_name: {(ma, mb): rho}}.
        run_dir       (str) : Output directory.
        cfg           (ExpConfig): Uses embed_dim.
    """
    n_models  = len(MODEL_NAMES)
    n_pairs   = len(METHOD_PAIRS)
    annotate  = cfg.embed_dim <= 16

    fig, axes = plt.subplots(
        n_models, n_pairs,
        figsize=(5 * n_pairs, 4 * n_models),
        squeeze=False,
    )

    for row, model_name in enumerate(MODEL_NAMES):
        scores = all_scores[model_name]
        color  = MODEL_COLORS[model_name]

        for col, (ma, mb) in enumerate(METHOD_PAIRS):
            ax  = axes[row, col]
            x   = scores[ma]
            y   = scores[mb]
            rho = all_agreement[model_name].get((ma, mb), float("nan"))

            ax.scatter(x, y, color=color, alpha=0.7, s=40)

            # Annotate dim indices for small embed_dim
            if annotate:
                for d, (xi, yi) in enumerate(zip(x, y)):
                    ax.annotate(str(d), (xi, yi), fontsize=6,
                                ha="left", va="bottom", alpha=0.8)

            # Spearman rho text box
            ax.text(0.05, 0.92, f"ρ = {rho:.3f}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            ax.set_xlabel(METHOD_LABELS[ma], fontsize=9)
            ax.grid(True, alpha=0.3)

            # Column titles on top row only
            if row == 0:
                ax.set_title(PAIR_LABELS[(ma, mb)], fontsize=10)

            # Row labels on leftmost column only (include model name)
            if col == 0:
                ax.set_ylabel(f"{model_name}\n{METHOD_LABELS[mb]}",
                              fontsize=9, color=color, fontweight="bold")
            else:
                ax.set_ylabel(METHOD_LABELS[mb], fontsize=9)

    fig.suptitle("Importance Method Agreement (Spearman ρ per model)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "method_agreement.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp8] Saved method_agreement.png")


# ==============================================================================
# Results summary table
# ==============================================================================

def save_results_summary(all_gap_results, all_agreement, all_scores,
                          eval_prefixes, run_dir):
    """
    Write results_summary.txt with three tables:
      Table 1: best-k vs first-k gap for all (model, method, k) combos.
      Table 2: Spearman method agreement per model.
      Table 3: Top-5 most important dims per model per method.

    Args:
        all_gap_results (dict): {model: {curve_key: {k: acc}}}.
        all_agreement   (dict): {model: {(ma, mb): rho}}.
        all_scores      (dict): {model: {method: np.ndarray}}.
        eval_prefixes   (list): Prefix sizes.
        run_dir         (str) : Output directory.
    """
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 8 — Dimension Importance Scoring Results\n")
        f.write("=" * 68 + "\n\n")

        # --- Table 1: best-k vs first-k gaps ---
        f.write("TABLE 1: Best-k vs First-k Accuracy  (gap = best_k - first_k)\n")
        f.write("-" * 68 + "\n")
        hdr = (f"{'k':>4}  {'Model':<10}  {'Method':<12}  "
               f"{'first_k':>8}  {'best_k':>8}  {'gap':>8}")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for k in eval_prefixes:
            for model_name in MODEL_NAMES:
                first_k = all_gap_results[model_name]["first_k"][k]
                for method in IMPORTANCE_METHODS:
                    best_k = all_gap_results[model_name][f"best_k_{method}"][k]
                    gap    = best_k - first_k
                    f.write(f"{k:>4}  {model_name:<10}  {method:<12}  "
                            f"{first_k:>8.4f}  {best_k:>8.4f}  {gap:>+8.4f}\n")
            f.write("\n")

        # --- Table 2: method agreement ---
        f.write("\nTABLE 2: Spearman Rank Correlation Between Importance Methods\n")
        f.write("-" * 68 + "\n")
        pair_labels_short = [PAIR_LABELS[p] for p in METHOD_PAIRS]
        hdr2 = f"{'Model':<12}  " + "  ".join(f"{pl:>18}" for pl in pair_labels_short)
        f.write(hdr2 + "\n")
        f.write("-" * len(hdr2) + "\n")
        for model_name in MODEL_NAMES:
            rhos     = [all_agreement[model_name].get(p, float("nan")) for p in METHOD_PAIRS]
            rho_strs = "  ".join(f"rho={r:>+6.3f}" for r in rhos)
            f.write(f"{model_name:<12}  {rho_strs}\n")

        # --- Table 3: top-5 dims per model per method ---
        f.write("\n\nTABLE 3: Top-5 Most Important Dims (descending importance)\n")
        f.write("-" * 68 + "\n")
        hdr3 = f"{'Model':<12}  {'Method':<12}  Top-5 dims"
        f.write(hdr3 + "\n")
        f.write("-" * len(hdr3) + "\n")
        for model_name in MODEL_NAMES:
            for method in IMPORTANCE_METHODS:
                top5 = np.argsort(all_scores[model_name][method])[::-1][:5].tolist()
                f.write(f"{model_name:<12}  {method:<12}  {top5}\n")
            f.write("\n")

    print(f"[exp8] Results summary saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Orchestrate Experiment 8 — Per-Dimension Importance Scoring.

    Steps:
      1.  Parse args, build config
      2.  Set seeds, create run_dir
      3.  Save experiment_description.log
      4.  Load data
      5.  Train Standard/L1/MRL OR load weights from --use-weights
      6.  Plot training_curves.png (MANDATORY)
      7.  Extract embeddings for all 4 models (Standard, L1, MRL, PCA)
      8.  Analysis 1: compute_importance_scores for each model
      9.  Analysis 2: compute_best_vs_first_k for each model
      10. Analysis 3: compute_method_agreement for each model
      11. Save all 4 plots
      12. Save results_summary.txt
      13. Save runtime.txt and code_snapshot/
    """
    run_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Experiment 8 — Dimension Importance")
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: digits, embed_dim=16, 5 epochs, 500 probe samples.",
    )
    parser.add_argument(
        "--use-weights", type=str, default=None, metavar="PATH",
        help="Path to exp7 or exp10 output folder; load Standard/L1/MRL weights (no retraining).",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None, metavar="N",
        help="Override embed_dim (8, 16, 32, 64); eval_prefixes derived as powers-of-2 up to N.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2: Resolve weights_dir (CLI wins over CONFIG)
    # If the path is not absolute, resolve it relative to files/results/
    # so the user can set just the folder name, e.g. "exprmnt_2026_04_01__22_04_54"
    # ------------------------------------------------------------------
    weights_dir = args.use_weights or (USE_WEIGHTS if USE_WEIGHTS else None)
    if weights_dir and not os.path.isabs(weights_dir):
        weights_dir = os.path.join(get_path("files/results"), weights_dir)
    if weights_dir:
        print(f"[exp8] Using saved weights from: {weights_dir}")

    # ------------------------------------------------------------------
    # Step 3: Config
    # Priority:
    #   --fast         → fixed smoke-test values (always)
    #   weights_dir    → load config.json from that folder (all fields adopted);
    #                    fall back to weight-file detection if config.json absent
    #   otherwise      → use this script's CONFIG block
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset="digits", embed_dim=16, hidden_dim=128,
            head_mode="shared_head", eval_prefixes=[1, 2, 4, 8, 16],
            lr=LR, epochs=5, batch_size=BATCH_SIZE, patience=3,
            weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
            experiment_name="exp8_dim_importance",
        )
        max_probe_samples = 500

    elif weights_dir:
        # Load every field from the saved run's config.json if available
        saved = load_config_json(weights_dir)
        if saved:
            print("[exp8] Adopting all config fields from saved run's config.json")
            # --embed-dim CLI flag still overrides embed_dim if explicitly given
            if args.embed_dim is not None:
                saved["embed_dim"]     = args.embed_dim
                saved["eval_prefixes"] = list(range(1, args.embed_dim + 1))
            else:
                # Ensure eval_prefixes is always dense [1..embed_dim]
                saved["eval_prefixes"] = list(range(1, saved["embed_dim"] + 1))
            saved["experiment_name"] = "exp8_dim_importance"
            cfg = ExpConfig(**saved)
        else:
            # config.json absent (old run) — fall back to weight-file detection
            print("[exp8] config.json not found; detecting architecture from weights.")
            detected_embed, detected_hidden = detect_arch_from_weights(weights_dir)
            embed_dim  = args.embed_dim if args.embed_dim is not None else detected_embed
            cfg = ExpConfig(
                dataset       = DATASET,
                embed_dim     = embed_dim,
                hidden_dim    = detected_hidden,
                head_mode     = HEAD_MODE,
                eval_prefixes = list(range(1, embed_dim + 1)),
                lr            = LR, epochs=EPOCHS, batch_size=BATCH_SIZE,
                patience      = PATIENCE, weight_decay=WEIGHT_DECAY,
                seed          = SEED, l1_lambda=L1_LAMBDA,
                experiment_name = "exp8_dim_importance",
            )
        max_probe_samples = MAX_PROBE_SAMPLES

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
            l1_lambda     = L1_LAMBDA,
            experiment_name = "exp8_dim_importance",
        )
        max_probe_samples = MAX_PROBE_SAMPLES

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Step 4: Setup output directory + save description
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[exp8] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, weights_dir, args.fast)

    # ------------------------------------------------------------------
    # Step 4: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # ------------------------------------------------------------------
    # Step 5: Get models (train from scratch OR load from --use-weights dir)
    # ------------------------------------------------------------------
    if weights_dir:
        print("=" * 60)
        print("STEP 5: Loading weights from existing run")
        print("=" * 60)
        std_encoder, std_head, l1_encoder, l1_head, mat_encoder, mat_head, \
            pl1_encoder, pl1_head = load_models_from_dir(weights_dir, cfg, data)
    else:
        print("=" * 60)
        print("STEP 5a: Training Standard model")
        print("=" * 60)
        std_encoder, std_head = train_single_model(
            cfg, data, run_dir, model_type="standard", model_tag="standard"
        )

        print("=" * 60)
        print(f"STEP 5b: Training L1 model  (lambda={cfg.l1_lambda})")
        print("=" * 60)
        l1_encoder, l1_head = train_single_model(
            cfg, data, run_dir, model_type="l1", model_tag="l1"
        )

        print("=" * 60)
        print("STEP 5c: Training MRL model")
        print("=" * 60)
        mat_encoder, mat_head = train_single_model(
            cfg, data, run_dir, model_type="matryoshka", model_tag="mat"
        )

        print("=" * 60)
        print(f"STEP 5d: Training PrefixL1 model  (lambda={cfg.l1_lambda})")
        print("=" * 60)
        pl1_encoder, pl1_head = train_single_model(
            cfg, data, run_dir, model_type="prefix_l1", model_tag="pl1"
        )

    # ------------------------------------------------------------------
    # Step 6: Training curves (MANDATORY)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Plotting training curves")
    print("=" * 60)
    plot_training_curves(run_dir, model_tags=["standard", "l1", "mat", "pl1"])

    # ------------------------------------------------------------------
    # Step 7: Extract embeddings for all 4 models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Extracting embeddings")
    print("=" * 60)

    Z_train_std = get_embeddings_np(std_encoder, data.X_train)
    Z_test_std  = get_embeddings_np(std_encoder, data.X_test)
    print(f"[exp8] Standard: train={Z_train_std.shape}, test={Z_test_std.shape}")

    Z_train_l1  = get_embeddings_np(l1_encoder, data.X_train)
    Z_test_l1   = get_embeddings_np(l1_encoder, data.X_test)
    print(f"[exp8] L1:       train={Z_train_l1.shape},  test={Z_test_l1.shape}")

    Z_train_mrl = get_embeddings_np(mat_encoder, data.X_train)
    Z_test_mrl  = get_embeddings_np(mat_encoder, data.X_test)
    print(f"[exp8] MRL:      train={Z_train_mrl.shape}, test={Z_test_mrl.shape}")

    Z_train_pca, Z_test_pca = get_pca_embeddings_np(data, cfg)

    # Convert labels to numpy once
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    embeddings = {
        "Standard": (Z_train_std, Z_test_std),
        "L1":       (Z_train_l1,  Z_test_l1),
        "MRL":      (Z_train_mrl, Z_test_mrl),
        "PCA":      (Z_train_pca, Z_test_pca),
    }

    # PrefixL1 is optional — present when trained in this run or loaded from exp10
    if pl1_encoder is not None:
        Z_train_pl1 = get_embeddings_np(pl1_encoder, data.X_train)
        Z_test_pl1  = get_embeddings_np(pl1_encoder, data.X_test)
        print(f"[exp8] PrefixL1: train={Z_train_pl1.shape}, test={Z_test_pl1.shape}")
        # Insert before PCA so ordering matches MODEL_NAMES
        embeddings = {
            "Standard": (Z_train_std, Z_test_std),
            "L1":       (Z_train_l1,  Z_test_l1),
            "MRL":      (Z_train_mrl, Z_test_mrl),
            "PrefixL1": (Z_train_pl1, Z_test_pl1),
            "PCA":      (Z_train_pca, Z_test_pca),
        }
    else:
        print("[exp8] PrefixL1 not available — analysis will cover Standard, L1, MRL, PCA only.")

    # ------------------------------------------------------------------
    # Step 8: Analysis 1 — compute importance scores
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Analysis 1 — Importance scoring (3 methods per model)")
    print("=" * 60)
    all_scores = {}
    for model_name, (Z_train, Z_test) in embeddings.items():
        all_scores[model_name] = compute_importance_scores(
            Z_test=Z_test, Z_train=Z_train,
            y_train=y_train_np, y_test=y_test_np,
            max_probe_samples=max_probe_samples,
            seed=cfg.seed, model_tag=model_name,
        )

    # ------------------------------------------------------------------
    # Step 9: Analysis 2 — best-k vs first-k
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Analysis 2 — Best-k vs First-k")
    print("=" * 60)
    all_gap_results = {}
    for model_name, (Z_train, Z_test) in embeddings.items():
        all_gap_results[model_name] = compute_best_vs_first_k(
            Z_train=Z_train, Z_test=Z_test,
            y_train=y_train_np, y_test=y_test_np,
            importance_scores=all_scores[model_name],
            eval_prefixes=cfg.eval_prefixes,
            seed=cfg.seed, model_tag=model_name,
        )

    # ------------------------------------------------------------------
    # Step 10: Analysis 3 — method agreement
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 10: Analysis 3 — Method agreement")
    print("=" * 60)
    all_agreement = {}
    for model_name in MODEL_NAMES:
        print(f"\n[exp8] Method agreement for {model_name} ...")
        all_agreement[model_name] = compute_method_agreement(
            importance_scores=all_scores[model_name],
            model_tag=model_name,
        )

    # ------------------------------------------------------------------
    # Step 11: Save all plots
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 11: Saving plots")
    print("=" * 60)
    plot_importance_scores(all_scores, run_dir, cfg)
    plot_dim_importance_heatmap(all_scores, run_dir, cfg)
    plot_best_vs_first_k(all_gap_results, run_dir, cfg)
    plot_method_agreement(all_scores, all_agreement, run_dir, cfg)

    # ------------------------------------------------------------------
    # Step 12: Results summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 12: Saving results summary")
    print("=" * 60)
    save_results_summary(all_gap_results, all_agreement, all_scores,
                         cfg.eval_prefixes, run_dir)

    # ------------------------------------------------------------------
    # Step 13: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 13: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp8] Experiment 8 complete.")
    print(f"[exp8] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
