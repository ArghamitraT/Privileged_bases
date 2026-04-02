"""
Script: experiments/exp2_cluster_viz.py
----------------------------------------
Experiment 2 — Cluster Visualization & Geometry-Performance Analysis.

For each prefix dimension k, this script:
  1. Extracts the k-dimensional prefix embeddings z_{1:k} from Standard,
     Matryoshka, and PCA models on the test set.
  2. Projects these k-dim embeddings to 2D via t-SNE (and UMAP if umap-learn
     is installed) for visual inspection of how class cluster structure
     evolves as k increases.
  3. Computes quantitative cluster metrics on the k-dim prefix space:
       - Silhouette score (sklearn)
       - Mean intra-class L2 distance
       - Mean inter-class centroid distance
       - Separation ratio = inter / intra
  4. Combines geometry metrics with classification accuracy to reveal the
     geometry <-> performance link: does better cluster structure at k
     correspond to higher accuracy at k?

Models can be loaded from a prior Experiment 1 run (--weights-dir or
--use-exp1) or trained from scratch if no weights flag is provided.

Inputs:
  --weights-dir PATH  : path to any exp1 run folder containing
                        {standard,mat}_{encoder,head}_best.pt weights.
  --use-exp1          : shortcut — automatically loads weights from the
                        canonical exp1 run (exprmnt_2026_03_08__16_36_30).
                        Equivalent to passing --weights-dir with that path.
  --fast              : quick smoke test — fewer viz points, fewer prefixes,
                        fewer t-SNE iterations.

Outputs (all saved in a new timestamped run folder):
  tsne_grid.png           : t-SNE 2D projections (rows=k, cols=model)
  umap_grid.png           : UMAP 2D projections  (rows=k, cols=model) [if umap-learn installed]
  cluster_metrics.png     : silhouette + intra/inter + separation ratio vs k
  combined_summary.png    : accuracy (top) + silhouette (bottom) vs k — geometry<->performance
  results_summary.txt     : full numeric table (k x model x all metrics)
  experiment_description.log
  runtime.txt
  code_snapshot/

Usage:
    python experiments/exp2_cluster_viz.py --use-exp1         # load exp1 weights, full MNIST
    python experiments/exp2_cluster_viz.py --use-exp1 --fast  # load exp1 weights, subsampled
    python experiments/exp2_cluster_viz.py --fast             # train from scratch on digits
    python tests/run_tests_exp2.py --fast                     # unit tests only
    python tests/run_tests_exp2.py                            # unit tests + e2e smoke
"""

import os
import sys
import time
import argparse
import dataclasses

# Cap BLAS thread count BEFORE importing numpy/scipy/sklearn.
# On macOS, importing scipy (cdist) + sklearn (TSNE, silhouette_score) can
# initialise OpenBLAS thread pools that later conflict with numpy's large-array
# operations (e.g. StandardScaler on 56k×784 MNIST), causing a segfault.
# Setting these to 1 forces single-threaded BLAS and eliminates the conflict.
# Must be done before ANY numpy/scipy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression

# Allow imports from the project root regardless of invocation directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss
from training.trainer import train
from evaluation.prefix_eval import evaluate_prefix_sweep, evaluate_pca_baseline


# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET            = "mnist"
EMBED_DIM          = 64
HIDDEN_DIM         = 256
HEAD_MODE          = "shared_head"
EVAL_PREFIXES      = [1, 2, 4, 8, 16, 32, 64]
EPOCHS             = 20
PATIENCE           = 5
LR                 = 1e-3
BATCH_SIZE         = 128
WEIGHT_DECAY       = 1e-4
SEED               = 42
# Visualization / metric settings (not part of ExpConfig)
VIZ_PREFIXES       = [1, 2, 4, 8]    # which prefix sizes to plot t-SNE/UMAP for
MAX_VIZ_SAMPLES    = 3000             # max samples for 2-D projections
MAX_METRIC_SAMPLES = 2000             # max samples for silhouette score
TSNE_N_ITER        = 1000             # t-SNE iterations
TSNE_PERPLEXITY    = 40               # t-SNE perplexity
# ==============================================================================


# umap-learn is imported lazily inside reduce_to_2d() to avoid initialising
# numba's thread pool at module load time. Numba conflicts with OpenBLAS when
# numpy allocates large arrays (e.g. MNIST 70k×784), causing a segfault.
# HAS_UMAP is set at first use, not at import time.
HAS_UMAP = None  # None = not yet checked; True/False after first call


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for numpy, torch, and python random for reproducibility.

    Args:
        seed (int): Master random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[exp2] Random seeds set to {seed}")


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(
    cfg: ExpConfig,
    run_dir: str,
    weights_dir,
    fast: bool,
):
    """
    Write a human-readable log describing this experiment run.

    Covers: what it does, why, expected outcome, weights source, and full config.

    Args:
        cfg         (ExpConfig): Experiment configuration.
        run_dir     (str)      : Output directory for this run.
        weights_dir (str|None) : Path to loaded weights, or None if trained here.
        fast        (bool)     : Whether fast/smoke mode is active.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 2 — Cluster Visualization & Geometry-Performance Analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "For each prefix dimension k, extracts k-dimensional prefix embeddings\n"
            "z_{1:k} from Standard, Matryoshka, and PCA models on the test set.\n"
            "Reduces them to 2D via t-SNE and UMAP for visual cluster inspection.\n"
            "Also computes silhouette score, intra/inter-class distances, and\n"
            "separation ratio — then links all of these to classification accuracy.\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Experiment 1 showed Mat embeddings keep high accuracy at small k.\n"
            "Here we ask: does this accuracy advantage come from better class\n"
            "cluster geometry at low k? If yes, Mat early dims truly encode\n"
            "class-discriminative structure, not just incidentally support\n"
            "the classifier. This is key evidence for 'privileged bases'.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "- Mat embeddings: well-separated clusters visible even at k=1,2,4.\n"
            "- Standard embeddings: poor cluster separation at small k.\n"
            "- PCA: intermediate — variance-ordered but not task-aware.\n"
            "- Silhouette and separation ratio should track accuracy closely.\n\n"
        )

        f.write("WEIGHTS SOURCE\n")
        f.write("-" * 40 + "\n")
        if weights_dir:
            f.write(f"  Loaded from prior run: {weights_dir}\n\n")
        else:
            f.write("  Trained from scratch in this run.\n\n")

        f.write(f"  Fast mode: {fast}\n\n")

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[exp2] Experiment description saved to {log_path}")


# ==============================================================================
# Model loading / training
# ==============================================================================

def load_models_from_dir(weights_dir: str, cfg: ExpConfig, data):
    """
    Load Standard and Matryoshka encoder + head weights from a prior exp1 folder.

    Expects to find in weights_dir:
        standard_encoder_best.pt, standard_head_best.pt,
        mat_encoder_best.pt,      mat_head_best.pt

    Args:
        weights_dir (str)      : Path to the exp1 run folder.
        cfg         (ExpConfig): Must match the architecture used when saving.
        data        (DataSplit): Used for input_dim and n_classes.

    Returns:
        Tuple: (std_encoder, std_head, mat_encoder, mat_head) — all in eval mode.
    """
    print(f"\n[exp2] Loading weights from: {weights_dir}")

    std_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    std_head    = build_head(cfg, data.n_classes)
    mat_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    mat_head    = build_head(cfg, data.n_classes)

    weight_files = [
        ("standard_encoder_best.pt", std_encoder),
        ("standard_head_best.pt",    std_head),
        ("mat_encoder_best.pt",      mat_encoder),
        ("mat_head_best.pt",         mat_head),
    ]

    for fname, model in weight_files:
        path = os.path.join(weights_dir, fname)
        assert os.path.isfile(path), f"Weight file not found: {path}"
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"  Loaded {fname}")

    print("[exp2] All weights loaded successfully.\n")
    return std_encoder, std_head, mat_encoder, mat_head


def train_from_scratch(cfg: ExpConfig, data, run_dir: str):
    """
    Train Standard and Matryoshka models from scratch, save weights to run_dir.

    Args:
        cfg     (ExpConfig): Training configuration.
        data    (DataSplit): Train/val/test splits.
        run_dir (str)      : Where to save .pt weights and training logs.

    Returns:
        Tuple: (std_encoder, std_head, mat_encoder, mat_head) — all in eval mode.
    """
    print("\n[exp2] No --weights-dir provided — training from scratch.\n")

    # --- Standard model ---
    print("[exp2] Training standard model ...")
    std_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    std_head    = build_head(cfg, data.n_classes)
    std_loss    = build_loss(cfg, "standard")
    std_opt     = torch.optim.Adam(
        list(std_encoder.parameters()) + list(std_head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    train(std_encoder, std_head, std_loss, std_opt, data, cfg, run_dir, model_tag="standard")

    # --- Matryoshka model ---
    print("[exp2] Training Matryoshka model ...")
    mat_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    mat_head    = build_head(cfg, data.n_classes)
    mat_loss    = build_loss(cfg, "matryoshka")
    mat_opt     = torch.optim.Adam(
        list(mat_encoder.parameters()) + list(mat_head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    train(mat_encoder, mat_head, mat_loss, mat_opt, data, cfg, run_dir, model_tag="mat")

    # Reload best checkpoints saved by the trainer
    for fname, model in [
        ("standard_encoder_best.pt", std_encoder),
        ("standard_head_best.pt",    std_head),
        ("mat_encoder_best.pt",      mat_encoder),
        ("mat_head_best.pt",         mat_head),
    ]:
        path = os.path.join(run_dir, fname)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

    return std_encoder, std_head, mat_encoder, mat_head


# ==============================================================================
# Embedding extraction
# ==============================================================================

def get_neural_embeddings(encoder: torch.nn.Module, data) -> np.ndarray:
    """
    Run the encoder forward pass on the full test set and return a numpy array.

    Args:
        encoder (nn.Module): Trained MLPEncoder in eval mode.
        data    (DataSplit): Contains X_test tensor.

    Returns:
        np.ndarray: Shape (n_test, embed_dim), L2-normalised embeddings.
    """
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(data.X_test)
    # Use .tolist() as the PyTorch-NumPy bridge (project convention — avoids
    # potential issues with tensor.__array__() across numpy/torch versions).
    return np.array(embeddings.cpu().tolist(), dtype=np.float32)


def get_pca_full_embeddings(data, cfg: ExpConfig) -> np.ndarray:
    """
    Fit PCA on training data and project the test set to embed_dim components.

    Components are ordered by explained variance (largest first), so slicing
    the first k columns gives the natural PCA prefix — directly analogous to
    the Matryoshka prefix.

    Args:
        data (DataSplit): X_train for fitting, X_test for projection.
        cfg  (ExpConfig): Uses embed_dim and seed.

    Returns:
        np.ndarray: Shape (n_test, n_components), PCA projection of test set.
                    n_components = min(embed_dim, input_dim).
    """
    X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
    X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)

    n_components = min(cfg.embed_dim, X_train_np.shape[0], X_train_np.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.seed)
    pca.fit(X_train_np)
    Z_test = pca.transform(X_test_np)   # (n_test, n_components)

    print(f"[exp2] PCA fitted: {n_components} components, test projection: {Z_test.shape}")
    return Z_test


def extract_prefix(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Return the first k columns of an embedding matrix.

    Args:
        embeddings (np.ndarray): Shape (n, embed_dim).
        k          (int)       : Number of prefix dimensions to keep.

    Returns:
        np.ndarray: Shape (n, k).
    """
    k_eff = min(k, embeddings.shape[1])
    return embeddings[:, :k_eff]


# ==============================================================================
# Subsampling helper
# ==============================================================================

def subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    """
    Generate random indices to subsample up to max_n items from n total.

    Using the same indices across all models ensures the scatter plots are
    directly comparable (they show exactly the same test points).

    Args:
        n     (int): Total number of items.
        max_n (int): Maximum number to keep.
        seed  (int): Random seed.

    Returns:
        np.ndarray: Integer indices array of length min(n, max_n).
    """
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, max_n, replace=False)


# ==============================================================================
# 2D dimensionality reduction
# ==============================================================================

def reduce_to_2d(
    prefix_emb: np.ndarray,
    method: str,
    seed: int,
    n_iter_tsne: int = 1000,
    perplexity: int = 40,
) -> np.ndarray:
    """
    Reduce (n, k) prefix embeddings to 2D coordinates for scatter plotting.

    Edge-case handling for small k:
      k=1: plot value on x-axis; add small uniform y-jitter for visibility.
      k=2: return directly — no reduction needed.
      k>2: apply t-SNE or UMAP.

    Args:
        prefix_emb  (np.ndarray): Shape (n, k) — the k-dim prefix embeddings.
        method      (str)       : 't-SNE' or 'UMAP'.
        seed        (int)       : Random seed for reproducibility.
        n_iter_tsne (int)       : Number of t-SNE gradient descent iterations.
        perplexity  (int)       : t-SNE perplexity (should be < n/4).

    Returns:
        np.ndarray: Shape (n, 2) — 2D coordinates ready for scatter plotting.
    """
    n, k = prefix_emb.shape

    if k == 1:
        # 1D case: use the embedding value as x, add jitter on y for visibility
        rng = np.random.default_rng(seed)
        y_jitter = rng.uniform(-0.3, 0.3, size=(n, 1))
        return np.hstack([prefix_emb, y_jitter])

    if k == 2:
        # Already 2D — plot the two dimensions directly
        return prefix_emb.copy()

    # k > 2: reduce to 2D with the chosen method
    if method == "UMAP":
        # Lazy import: defer umap/numba initialisation until first actual UMAP call
        global HAS_UMAP
        if HAS_UMAP is None:
            try:
                import umap as umap_lib_local
                HAS_UMAP = True
                print("[exp2] umap-learn found — UMAP reduction active.")
            except ImportError:
                HAS_UMAP = False
                print("[exp2] umap-learn not installed — UMAP plots will be skipped.")
        if not HAS_UMAP:
            raise RuntimeError("umap-learn is not installed. Run: conda install -c conda-forge umap-learn")
        import umap as umap_lib_local
        reducer = umap_lib_local.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=seed,
            verbose=False,
        )
        return reducer.fit_transform(prefix_emb)

    else:
        # t-SNE — use PCA initialisation for stability, especially at higher dims
        safe_perplexity = min(perplexity, max(5, n // 4 - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            max_iter=n_iter_tsne,
            init="pca",
            random_state=seed,
            learning_rate="auto",
        )
        return reducer.fit_transform(prefix_emb)


# ==============================================================================
# Cluster quality metrics
# ==============================================================================

def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Compute cluster quality metrics for a set of embeddings and class labels.

    Metrics:
      silhouette  — sklearn silhouette score [-1, 1], higher = better.
      intra       — mean within-class pairwise L2 distance (lower = tighter).
      inter       — mean distance between class centroids (higher = more spread).
      separation  — inter / intra ratio (higher = better separated clusters).

    Note: subsamples to max_samples for computational efficiency. Intra-class
    distance additionally caps each class at 200 points to avoid O(n^2) blowup.

    Args:
        embeddings  (np.ndarray): Shape (n, d).
        labels      (np.ndarray): Shape (n,) — integer class labels.
        max_samples (int)       : Subsample to at most this many points.
        seed        (int)       : Random seed for subsampling.

    Returns:
        dict: Keys 'silhouette', 'intra', 'inter', 'separation' (all floats).
    """
    # Subsample for efficiency — keeps fast runtime for MNIST-scale data
    idx = subsample_indices(len(embeddings), max_samples, seed)
    emb  = embeddings[idx]
    labs = labels[idx]

    unique_classes = np.unique(labs)
    n_cls = len(unique_classes)

    # Need at least 2 classes with at least 2 samples each for meaningful metrics
    if n_cls < 2 or len(emb) < 4:
        return {"silhouette": 0.0, "intra": 0.0, "inter": 0.0, "separation": 0.0}

    # --- Silhouette score (measures both cohesion and separation together) ---
    try:
        sil = float(silhouette_score(emb, labs, metric="euclidean"))
    except Exception:
        sil = 0.0

    # --- Intra-class mean pairwise distance ---
    # Subsample each class to at most 200 points to avoid O(n^2) memory
    intra_per_class = []
    centroids = []
    for c in unique_classes:
        mask = labs == c
        class_emb = emb[mask]
        centroids.append(class_emb.mean(axis=0))   # centroid for inter-class dist

        n_c = len(class_emb)
        if n_c < 2:
            continue

        max_per_class = 200
        if n_c > max_per_class:
            rng = np.random.default_rng(seed + int(c))
            sub_idx = rng.choice(n_c, max_per_class, replace=False)
            class_emb = class_emb[sub_idx]

        dists = cdist(class_emb, class_emb, metric="euclidean")
        upper = dists[np.triu_indices(len(class_emb), k=1)]
        if len(upper) > 0:
            intra_per_class.append(float(upper.mean()))

    intra = float(np.mean(intra_per_class)) if intra_per_class else 0.0

    # --- Inter-class distance: mean pairwise L2 between class centroids ---
    centroid_arr = np.array(centroids)
    inter_dists  = cdist(centroid_arr, centroid_arr, metric="euclidean")
    upper_inter  = inter_dists[np.triu_indices(n_cls, k=1)]
    inter = float(upper_inter.mean()) if len(upper_inter) > 0 else 0.0

    # --- Separation ratio ---
    separation = inter / intra if intra > 1e-10 else 0.0

    return {
        "silhouette": sil,
        "intra":      intra,
        "inter":      inter,
        "separation": separation,
    }


# ==============================================================================
# Plotting
# ==============================================================================

def _get_class_colors(n_cls: int):
    """Return a list of n_cls distinct, fully-saturated colours from Set1/tab10."""
    # Use Set1 (high-saturation) for ≤9 classes, tab10 for ≤10, tab20 for more
    if n_cls <= 9:
        cmap_name = "Set1"
    elif n_cls <= 10:
        cmap_name = "tab10"
    else:
        cmap_name = "tab20"
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)
    # Set1 has 9 fixed colours; tab10/tab20 are indexed 0–1
    if cmap_name == "Set1":
        return [cmap(i) for i in range(n_cls)]
    return [cmap(i / max(1, n_cls - 1)) for i in range(n_cls)]


def plot_viz_grid(
    all_2d: dict,
    labels: np.ndarray,
    model_names: list,
    viz_prefixes: list,
    method_name: str,
    run_dir: str,
    accuracy_dict: dict = None,
    metrics: dict = None,
):
    """
    Plot the 2D cluster visualization grid: rows = model, cols = prefix k.

    Layout: 3 rows (Standard / Matryoshka / PCA) × 4 cols (k=1, 2, 4, 8).
    Each cell title shows classification accuracy and silhouette score for
    that (model, k) combination so performance is readable at a glance.

    Args:
        all_2d        (dict) : Keys (model_name, k) → np.ndarray (n_viz, 2).
        labels        (np.ndarray): Shape (n_viz,) — class labels for viz points.
        model_names   (list) : ['Standard', 'Matryoshka', 'PCA'].
        viz_prefixes  (list) : k values shown as columns, e.g. [1, 2, 4, 8].
        method_name   (str)  : 't-SNE' or 'UMAP' — used in title and filename.
        run_dir       (str)  : Where to save the PNG.
        accuracy_dict (dict) : Keys (model_name, k) → accuracy float (optional).
        metrics       (dict) : Keys (model_name, k) → metric dict (optional).
    """
    # rows = models, cols = prefix k  (3 × 4 for standard layout)
    n_rows = len(model_names)
    n_cols = len(viz_prefixes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))

    # Normalise axes to always be a 2D array regardless of shape
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    unique_labels = np.unique(labels)
    n_cls = len(unique_labels)
    colors = _get_class_colors(n_cls)

    for row_idx, model_name in enumerate(model_names):
        for col_idx, k in enumerate(viz_prefixes):
            ax  = axes[row_idx, col_idx]
            key = (model_name, k)

            if key not in all_2d:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
                ax.set_axis_off()
                continue

            coords = all_2d[key]  # (n_viz, 2)

            # Scatter one series per class — deeper alpha and larger points
            for cls_idx, cls in enumerate(unique_labels):
                mask = labels == cls
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=8, alpha=0.85, color=colors[cls_idx],
                    label=str(int(cls)), linewidths=0,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f5f5f5")  # light grey background for contrast

            # Row label (model name) on the left of the first column
            if col_idx == 0:
                ax.set_ylabel(model_name, fontsize=12, fontweight="bold")

            # Column header (k value) on top of the first row
            col_header = f"k = {k}" if k > 1 else "k = 1  (1-D strip)"
            if row_idx == 0:
                ax.set_title(col_header, fontsize=12, fontweight="bold", pad=6)

            # Per-cell annotation: accuracy + silhouette score
            info_lines = []
            if accuracy_dict is not None and key in accuracy_dict:
                info_lines.append(f"acc={accuracy_dict[key]:.3f}")
            if metrics is not None and key in metrics:
                sil = metrics[key].get("silhouette", float("nan"))
                info_lines.append(f"sil={sil:+.3f}")
            if info_lines:
                ax.text(
                    0.5, 1.0, "  ".join(info_lines),
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=9, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
                )

    # Shared class legend at the bottom
    if n_cls <= 20:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[i], markersize=9,
                       label=str(int(c)))
            for i, c in enumerate(unique_labels)
        ]
        fig.legend(
            handles=handles, title="Class",
            loc="lower center", ncol=min(n_cls, 10),
            fontsize=9, bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        f"{method_name} Cluster Evolution  —  rows = model,  cols = prefix k",
        fontsize=15, y=1.01,
    )
    plt.tight_layout()

    fname = "tsne_grid.png" if method_name == "t-SNE" else "umap_grid.png"
    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved {fname}")


def plot_cluster_metrics(
    metrics: dict,
    eval_prefixes: list,
    model_names: list,
    run_dir: str,
):
    """
    Three-panel figure showing cluster quality metrics vs prefix k.

    Panel 1 — Silhouette score  (higher = better, range [-1, 1])
    Panel 2 — Intra-class mean distance (lower = tighter clusters)
    Panel 3 — Separation ratio = inter/intra (higher = better)

    Args:
        metrics      (dict) : Keys (model_name, k) → metric dict.
        eval_prefixes(list) : All prefix sizes on the x-axis.
        model_names  (list) : Model names (determines line colours).
        run_dir      (str)  : Where to save cluster_metrics.png.
    """
    model_styles = {
        "Standard":   ("steelblue",  "o-"),
        "Matryoshka": ("darkorange", "s-"),
        "PCA":        ("seagreen",   "^--"),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("silhouette",  "Silhouette Score",              "higher = better"),
        ("intra",       "Mean Intra-class Distance",      "lower = tighter"),
        ("separation",  "Separation Ratio (inter/intra)", "higher = better"),
    ]

    for ax, (metric_key, ylabel, note) in zip(axes, panels):
        for model_name in model_names:
            vals   = [metrics[(model_name, k)][metric_key] for k in eval_prefixes]
            color, style = model_styles.get(model_name, ("gray", "x-"))
            ax.plot(
                eval_prefixes, vals, style,
                color=color, label=model_name, linewidth=2, markersize=7,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(eval_prefixes)
        ax.set_xticklabels([str(k) for k in eval_prefixes])
        ax.set_xlabel("Prefix size k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}\n({note})", fontsize=11)
        ax.legend(fontsize=10)

    fig.suptitle("Cluster Quality Metrics vs Prefix Size k", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(run_dir, "cluster_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp2] Saved cluster_metrics.png")


def plot_combined_summary(
    accuracy_dict: dict,
    metrics: dict,
    eval_prefixes: list,
    model_names: list,
    run_dir: str,
):
    """
    Combined 2-panel figure explicitly linking accuracy to cluster geometry.

    Top panel    : Classification accuracy vs k  (same shape as exp1 prefix curve)
    Bottom panel : Silhouette score vs k

    The shared x-axis makes it visually obvious whether accuracy improvements
    at larger k coincide with better cluster separation.

    Args:
        accuracy_dict (dict) : Keys (model_name, k) → accuracy float.
        metrics       (dict) : Keys (model_name, k) → metric dict.
        eval_prefixes (list) : Sorted prefix list.
        model_names   (list) : Model names.
        run_dir       (str)  : Where to save combined_summary.png.
    """
    model_styles = {
        "Standard":   ("steelblue",  "o-"),
        "Matryoshka": ("darkorange", "s-"),
        "PCA":        ("seagreen",   "^--"),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for model_name in model_names:
        color, style = model_styles.get(model_name, ("gray", "x-"))
        accs = [accuracy_dict.get((model_name, k), float("nan")) for k in eval_prefixes]
        sils = [metrics[(model_name, k)]["silhouette"]            for k in eval_prefixes]

        ax_top.plot(eval_prefixes, accs, style, color=color,
                    label=model_name, linewidth=2, markersize=7)
        ax_bot.plot(eval_prefixes, sils, style, color=color,
                    label=model_name, linewidth=2, markersize=7)

    for ax in (ax_top, ax_bot):
        ax.set_xscale("log", base=2)
        ax.set_xticks(eval_prefixes)
        ax.set_xticklabels([str(k) for k in eval_prefixes])
        ax.legend(fontsize=10)

    ax_top.set_ylabel("Classification Accuracy", fontsize=12)
    ax_top.set_title("Classification Accuracy vs Prefix k", fontsize=13)
    ax_top.set_ylim(0, 1.05)

    ax_bot.set_xlabel("Prefix size k  (number of embedding dimensions used)", fontsize=12)
    ax_bot.set_ylabel("Silhouette Score", fontsize=12)
    ax_bot.set_title("Cluster Silhouette Score vs Prefix k", fontsize=13)

    fig.suptitle(
        "Geometry \u2194 Performance:  Accuracy and Cluster Quality vs k",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    out_path = os.path.join(run_dir, "combined_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp2] Saved combined_summary.png")


# ==============================================================================
# 4D Visualization: Animated GIF + Interactive HTML
# ==============================================================================

def plot_4d_animation(
    model_embeddings: dict,
    y_viz: np.ndarray,
    viz_idx: np.ndarray,
    run_dir: str,
    dims: list = None,
    fps: int = 1,
):
    """
    Create an animated GIF cycling through all 6 pairwise 2D scatter plots of
    4 embedding dimensions for Standard, Matryoshka, and PCA models.

    Each of the C(4,2)=6 frames shows one dim pair as a scatter plot. All three
    models are displayed side-by-side per frame so class separability can be
    compared across dim pairs and across models at the same time.

    Args:
        model_embeddings (dict)      : Keys 'Standard', 'Matryoshka', 'PCA' ->
                                       np.ndarray (n_test, embed_dim).
        y_viz            (np.ndarray): Shape (n_viz,) — class labels for viz points.
        viz_idx          (np.ndarray): Indices used to subsample full embeddings.
        run_dir          (str)       : Where to save dim4_animation.gif.
        dims             (list)      : 4 dimension indices to animate. Default [0,1,2,3].
        fps              (int)       : Frames per second in the output GIF.

    Returns:
        None. Saves dim4_animation.gif to run_dir.
    """
    from itertools import combinations
    import matplotlib.animation as animation

    if dims is None:
        dims = [0, 1, 2, 3]

    dim_pairs   = list(combinations(dims, 2))   # C(4,2) = 6 pairs
    model_names = ["Standard", "Matryoshka", "PCA"]

    # Extract subsampled embeddings for each model
    emb_viz = {name: model_embeddings[name][viz_idx] for name in model_names}

    unique_labels = np.unique(y_viz)
    n_cls         = len(unique_labels)
    colors        = _get_class_colors(n_cls)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    def _draw_frame(frame_idx):
        """Render one frame: scatter of dim-pair (di, dj) for all 3 models."""
        di, dj = dim_pairs[frame_idx]
        for ax, model_name in zip(axes, model_names):
            ax.cla()
            emb = emb_viz[model_name]
            for cls_idx, cls in enumerate(unique_labels):
                mask = y_viz == cls
                ax.scatter(
                    emb[mask, di], emb[mask, dj],
                    s=6, alpha=0.75, color=colors[cls_idx], linewidths=0,
                )
            ax.set_xlabel(f"Dim {di}", fontsize=11)
            ax.set_ylabel(f"Dim {dj}", fontsize=11)
            ax.set_title(model_name, fontsize=12, fontweight="bold")
            ax.set_facecolor("#f5f5f5")
            ax.set_xticks([])
            ax.set_yticks([])

        # Shared class legend at the bottom
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[i], markersize=8,
                       label=str(int(c)))
            for i, c in enumerate(unique_labels)
        ]
        fig.legend(handles=handles, title="Class",
                   loc="lower center", ncol=min(n_cls, 10),
                   fontsize=8, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(
            f"4D Embedding  (dims {dims})   —   "
            f"Frame {frame_idx + 1}/{len(dim_pairs)}: Dim {di} vs Dim {dj}",
            fontsize=13, y=1.02,
        )
        plt.tight_layout(rect=[0, 0.07, 1, 1])

    ani = animation.FuncAnimation(
        fig, _draw_frame, frames=len(dim_pairs),
        interval=int(1000 / fps), repeat=True,
    )

    out_path = os.path.join(run_dir, "dim4_animation.gif")
    try:
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer, dpi=100)
        print(f"[exp2] Saved dim4_animation.gif  ({len(dim_pairs)} frames, dims={dims})")
    except Exception as exc:
        print(f"[exp2] Warning: could not save GIF — {exc}. Is Pillow installed?")
    finally:
        plt.close()


def plot_4d_interactive(
    model_embeddings: dict,
    y_viz: np.ndarray,
    viz_idx: np.ndarray,
    run_dir: str,
    dims: list = None,
):
    """
    Create an interactive HTML file with 3D scatter subplots for Standard,
    Matryoshka, and PCA embeddings using 4 embedding dimensions.

    Each subplot shows 3 of the 4 chosen dimensions as 3D axes (x, y, z),
    colored by class label. A dropdown menu cycles through all C(4,3)=4
    combinations so the user can inspect every 3D projection. All subplots
    are fully rotatable in the browser — no extra software required.

    Requires plotly (already in mrl_env). Skipped gracefully if not installed.

    Args:
        model_embeddings (dict)      : Keys 'Standard', 'Matryoshka', 'PCA' ->
                                       np.ndarray (n_test, embed_dim).
        y_viz            (np.ndarray): Shape (n_viz,) — class labels for viz points.
        viz_idx          (np.ndarray): Indices used to subsample full embeddings.
        run_dir          (str)       : Where to save dim4_interactive.html.
        dims             (list)      : 4 dimension indices to use. Default [0,1,2,3].

    Returns:
        None. Saves dim4_interactive.html to run_dir (or logs a warning if plotly missing).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(
            "[exp2] plotly not installed — skipping dim4_interactive.html.\n"
            "       Install with: pip install plotly"
        )
        return

    from itertools import combinations

    if dims is None:
        dims = [0, 1, 2, 3]

    # C(4,3) = 4 combinations: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    dim_combos  = list(combinations(dims, 3))
    model_names = ["Standard", "Matryoshka", "PCA"]
    n_models    = len(model_names)

    unique_labels = np.unique(y_viz)
    n_cls         = len(unique_labels)
    mpl_colors    = _get_class_colors(n_cls)

    # Convert matplotlib RGBA tuples → plotly hex colour strings
    plotly_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in mpl_colors
    ]

    # Subsampled embeddings
    emb_viz = {name: model_embeddings[name][viz_idx] for name in model_names}

    # ------------------------------------------------------------------
    # Build subplot figure: 1 row × 3 cols, all 3D scatter scenes
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}] * 3],
        subplot_titles=model_names,
        horizontal_spacing=0.02,
    )

    # Initial dim combo drives the first render
    di0, dj0, dk0 = dim_combos[0]

    # Add all traces using the initial combo
    for col_idx, model_name in enumerate(model_names):
        emb = emb_viz[model_name]
        for cls_idx, cls in enumerate(unique_labels):
            mask = y_viz == cls
            fig.add_trace(
                go.Scatter3d(
                    x=emb[mask, di0].tolist(),
                    y=emb[mask, dj0].tolist(),
                    z=emb[mask, dk0].tolist(),
                    mode="markers",
                    marker=dict(size=2, color=plotly_colors[cls_idx], opacity=0.7),
                    name=f"Class {int(cls)}",
                    legendgroup=f"class_{int(cls)}",
                    showlegend=(col_idx == 0),  # legend entries only from first subplot
                ),
                row=1, col=col_idx + 1,
            )

    # ------------------------------------------------------------------
    # Build dropdown buttons — each button restyles all traces with new
    # x, y, z data for the chosen 3-dim combo
    # ------------------------------------------------------------------
    buttons = []
    for di, dj, dk in dim_combos:
        # Collect new x,y,z for every trace in the same order they were added
        x_data, y_data, z_data = [], [], []
        for model_name in model_names:
            emb = emb_viz[model_name]
            for cls in unique_labels:
                mask = y_viz == cls
                x_data.append(emb[mask, di].tolist())
                y_data.append(emb[mask, dj].tolist())
                z_data.append(emb[mask, dk].tolist())

        buttons.append(dict(
            label=f"Dims  {di}, {dj}, {dk}",
            method="restyle",
            args=[{"x": x_data, "y": y_data, "z": z_data}],
        ))

    # ------------------------------------------------------------------
    # Layout: dropdown, title, initial axis labels
    # ------------------------------------------------------------------
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            showactive=True,
            x=0.0, xanchor="left",
            y=1.18, yanchor="top",
            pad={"r": 10, "t": 10},
            bgcolor="white",
            bordercolor="#cccccc",
        )],
        title=dict(
            text=(
                "4D Embedding — First 4 Dims  "
                "<span style='font-size:13px;color:#666'>"
                "(rotate subplots freely · use dropdown to change dimension view)"
                "</span>"
            ),
            font=dict(size=16),
            x=0.5,
        ),
        legend=dict(title="Class", itemsizing="constant"),
        height=650,
        margin=dict(t=130, b=20, l=10, r=10),
    )

    # Set initial axis labels on all 3 scenes
    fig.update_scenes(
        xaxis_title=f"Dim {di0}",
        yaxis_title=f"Dim {dj0}",
        zaxis_title=f"Dim {dk0}",
    )

    out_path = os.path.join(run_dir, "dim4_interactive.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print("[exp2] Saved dim4_interactive.html  —  open in browser to rotate 3D plots")


def plot_4d_slideshow(
    model_embeddings: dict,
    y_viz: np.ndarray,
    viz_idx: np.ndarray,
    run_dir: str,
    dims: list = None,
):
    """
    Create a self-contained HTML slideshow of the same 6 pairwise 2D scatter
    frames as dim4_animation.gif, but navigated manually via Prev / Next buttons
    instead of auto-playing.  A Play button with an adjustable speed slider is
    also provided for convenience.

    Each frame is rendered by matplotlib, encoded as a base64 PNG, and embedded
    directly in the HTML — no external dependencies, works offline.

    Args:
        model_embeddings (dict)      : Keys 'Standard', 'Matryoshka', 'PCA' ->
                                       np.ndarray (n_test, embed_dim).
        y_viz            (np.ndarray): Shape (n_viz,) — class labels for viz points.
        viz_idx          (np.ndarray): Indices used to subsample full embeddings.
        run_dir          (str)       : Where to save dim4_slideshow.html.
        dims             (list)      : 4 dimension indices to use. Default [0,1,2,3].

    Returns:
        None. Saves dim4_slideshow.html to run_dir.
    """
    import io
    import base64
    import json
    from itertools import combinations

    if dims is None:
        dims = [0, 1, 2, 3]

    dim_pairs   = list(combinations(dims, 2))   # C(4,2) = 6 frames
    model_names = ["Standard", "Matryoshka", "PCA"]
    emb_viz     = {name: model_embeddings[name][viz_idx] for name in model_names}

    unique_labels = np.unique(y_viz)
    n_cls         = len(unique_labels)
    colors        = _get_class_colors(n_cls)

    # ------------------------------------------------------------------
    # Render each frame to a base64-encoded PNG
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    frame_b64    = []
    frame_titles = []

    for frame_idx, (di, dj) in enumerate(dim_pairs):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("white")

        for ax, model_name in zip(axes, model_names):
            emb = emb_viz[model_name]
            for cls_idx, cls in enumerate(unique_labels):
                mask = y_viz == cls
                ax.scatter(
                    emb[mask, di], emb[mask, dj],
                    s=6, alpha=0.75, color=colors[cls_idx], linewidths=0,
                )
            ax.set_xlabel(f"Dim {di}", fontsize=11)
            ax.set_ylabel(f"Dim {dj}", fontsize=11)
            ax.set_title(model_name, fontsize=12, fontweight="bold")
            ax.set_facecolor("#f5f5f5")
            ax.set_xticks([])
            ax.set_yticks([])

        # Shared class legend at the bottom
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[i], markersize=8,
                       label=str(int(c)))
            for i, c in enumerate(unique_labels)
        ]
        fig.legend(handles=handles, title="Class",
                   loc="lower center", ncol=min(n_cls, 10),
                   fontsize=8, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(
            f"4D Embedding  (dims {dims})   —   "
            f"Frame {frame_idx + 1}/{len(dim_pairs)}: Dim {di} vs Dim {dj}",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0.07, 1, 1])

        # Encode to base64 PNG in memory
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frame_b64.append(base64.b64encode(buf.read()).decode("ascii"))
        frame_titles.append(
            f"Frame {frame_idx + 1}/{len(dim_pairs)}: Dim {di} vs Dim {dj}"
        )

    # ------------------------------------------------------------------
    # Build self-contained HTML with Prev / Next / Play controls
    # ------------------------------------------------------------------
    n_frames   = len(dim_pairs)
    frames_js  = json.dumps(frame_b64)
    titles_js  = json.dumps(frame_titles)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>4D Embedding Slideshow</title>
  <style>
    body        {{ font-family: Arial, sans-serif; background: #f8f8f8;
                  text-align: center; padding: 20px; }}
    h2          {{ color: #333; margin-bottom: 4px; }}
    #frame-title{{ font-size: 1.1em; color: #444; margin: 6px 0 2px; }}
    #counter    {{ font-size: 0.9em;  color: #888; margin-bottom: 10px; }}
    #slide-img  {{ max-width: 100%; border: 1px solid #ddd;
                  border-radius: 4px; background: white; }}
    .btn        {{ font-size: 1.05em; padding: 8px 24px; margin: 10px 5px 0;
                  border: none; border-radius: 4px; cursor: pointer;
                  background: #4a90d9; color: white; }}
    .btn:hover  {{ background: #357abd; }}
    .btn:disabled{{ background: #aaa; cursor: default; }}
    #play-btn   {{ background: #5cb85c; }}
    #play-btn:hover{{ background: #449d44; }}
    #speed-row  {{ margin-top: 10px; font-size: 0.9em; color: #555; }}
  </style>
</head>
<body>
  <h2>4D Embedding Pairwise Scatter — Slideshow</h2>
  <p id="frame-title">Loading…</p>
  <p id="counter"></p>
  <img id="slide-img" src="" alt="scatter frame">
  <br>
  <button class="btn" id="prev-btn" onclick="step(-1)">&#9664; Prev</button>
  <button class="btn" id="play-btn" onclick="togglePlay()">&#9654; Play</button>
  <button class="btn" id="next-btn" onclick="step(1)">Next &#9654;</button>
  <div id="speed-row">
    Speed:&nbsp;
    <input type="range" id="speed-slider"
           min="500" max="5000" step="500" value="1500"
           oninput="updateSpeed()">
    &nbsp;<span id="speed-val">1.5 s / frame</span>
  </div>

  <script>
    const frames   = {frames_js};
    const titles   = {titles_js};
    const N        = {n_frames};
    let   current  = 0;
    let   playing  = false;
    let   timer    = null;
    let   interval = 1500;   // milliseconds per frame in Play mode

    function show(idx) {{
      // Wrap around so Prev on frame 0 goes to the last frame
      current = ((idx % N) + N) % N;
      document.getElementById("slide-img").src =
        "data:image/png;base64," + frames[current];
      document.getElementById("frame-title").textContent = titles[current];
      document.getElementById("counter").textContent =
        (current + 1) + " / " + N;
    }}

    function step(delta) {{
      if (playing) stopPlay();
      show(current + delta);
    }}

    function togglePlay() {{
      playing ? stopPlay() : startPlay();
    }}

    function startPlay() {{
      playing = true;
      document.getElementById("play-btn").textContent = "⏸ Pause";
      // Advance immediately then keep ticking
      show(current + 1);
      timer = setInterval(() => show(current + 1), interval);
    }}

    function stopPlay() {{
      playing = false;
      document.getElementById("play-btn").textContent = "▶ Play";
      clearInterval(timer);
      timer = null;
    }}

    function updateSpeed() {{
      interval = parseInt(document.getElementById("speed-slider").value);
      document.getElementById("speed-val").textContent =
        (interval / 1000).toFixed(1) + " s / frame";
      if (playing) {{ stopPlay(); startPlay(); }}   // restart with new interval
    }}

    // Show first frame on load
    show(0);
  </script>
</body>
</html>"""

    out_path = os.path.join(run_dir, "dim4_slideshow.html")
    with open(out_path, "w") as fh:
        fh.write(html)
    print(
        f"[exp2] Saved dim4_slideshow.html  ({n_frames} frames, dims={dims})  "
        "— open in browser, use Prev/Next or Play"
    )


# ==============================================================================
# Results table
# ==============================================================================

def save_results_summary(
    accuracy_dict: dict,
    metrics: dict,
    eval_prefixes: list,
    model_names: list,
    run_dir: str,
):
    """
    Write a text table of accuracy + all cluster metrics for every (model, k).

    Args:
        accuracy_dict (dict) : Keys (model_name, k) → accuracy.
        metrics       (dict) : Keys (model_name, k) → metric dict.
        eval_prefixes (list) : Sorted prefix list (rows).
        model_names   (list) : Model names (columns).
        run_dir       (str)  : Where to save results_summary.txt.
    """
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 2 — Cluster Visualization Results\n")
        f.write("=" * 76 + "\n\n")
        header = (
            f"{'k':>4}  {'Model':<12}  {'Accuracy':>10}  "
            f"{'Silhouette':>12}  {'Intra':>8}  {'Inter':>8}  {'Sep.Ratio':>10}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for k in eval_prefixes:
            for model_name in model_names:
                acc = accuracy_dict.get((model_name, k), float("nan"))
                m   = metrics.get((model_name, k), {})
                f.write(
                    f"{k:>4}  {model_name:<12}  "
                    f"{acc:>10.4f}  "
                    f"{m.get('silhouette', 0):>12.4f}  "
                    f"{m.get('intra',      0):>8.4f}  "
                    f"{m.get('inter',      0):>8.4f}  "
                    f"{m.get('separation', 0):>10.4f}\n"
                )
            f.write("\n")

    print(f"[exp2] Results summary saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Orchestrate Experiment 2:
      - Parse args, set config, create run dir
      - Load data and models (from weights or train from scratch)
      - Compute embeddings for Standard, Matryoshka, PCA
      - Evaluate prefix accuracy (for all eval_prefixes)
      - Compute cluster metrics (for all eval_prefixes)
      - Run t-SNE and UMAP reductions (for viz_prefixes subset)
      - Generate and save all plots
      - Save results table, runtime, and code snapshot
    """
    run_start = time.time()

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Experiment 2 — Cluster Visualization")
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help=(
            "Path to an exp1 run folder containing saved .pt weight files. "
            "If omitted (and --use-exp1 not set), models are trained from scratch."
        ),
    )
    parser.add_argument(
        "--use-exp1", action="store_true",
        help=(
            "Shortcut: automatically load weights from the canonical exp1 run "
            "(exprmnt_2026_03_08__16_36_30). "
            "Equivalent to --weights-dir with that folder path."
        ),
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick smoke test: fewer viz samples, fewer prefixes, fewer t-SNE iterations.",
    )
    args = parser.parse_args()

    # --use-exp1 is a convenience alias for --weights-dir pointing at the known exp1 folder
    if args.use_exp1:
        if args.weights_dir:
            print("[exp2] Warning: --use-exp1 and --weights-dir both set; --weights-dir takes precedence.")
        else:
            # Resolve the exp1 run folder relative to the project root via utility.get_path
            from utility import get_path
            results_base   = get_path("files/results")
            exp1_folder    = "exprmnt_2026_03_08__16_36_30"
            args.weights_dir = os.path.join(results_base, exp1_folder)
            print(f"[exp2] --use-exp1: resolved weights dir to {args.weights_dir}")

    # ------------------------------------------------------------------
    # Config — build from CONFIG block above; --fast overrides to smoke-test
    # values; --weights-dir must match the architecture that was saved.
    # ------------------------------------------------------------------
    if args.fast:
        # Smoke-test overrides: tiny dataset, quick epochs
        cfg = ExpConfig(
            dataset="digits", embed_dim=16, hidden_dim=128,
            head_mode="shared_head", eval_prefixes=[1, 2, 4, 8, 16],
            lr=LR, epochs=5, batch_size=BATCH_SIZE, patience=3,
            weight_decay=WEIGHT_DECAY, seed=SEED,
            experiment_name="exp2_cluster_viz",
        )
        viz_prefixes       = [1, 2, 4, 8]
        max_viz_samples    = 500
        max_metric_samples = 500
        tsne_n_iter        = 300
        tsne_perplexity    = 20
    else:
        # Full run (also used when loading weights from --weights-dir)
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
            experiment_name = "exp2_cluster_viz",
        )
        viz_prefixes       = VIZ_PREFIXES
        max_viz_samples    = MAX_VIZ_SAMPLES
        max_metric_samples = MAX_METRIC_SAMPLES
        tsne_n_iter        = TSNE_N_ITER
        tsne_perplexity    = TSNE_PERPLEXITY

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Setup output directory
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[exp2] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, args.weights_dir, args.fast)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    data = load_data(cfg)
    y_test_np = np.array(data.y_test.tolist(), dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 2: Load or train models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Setting up models")
    print("=" * 60)

    if args.weights_dir:
        std_encoder, std_head, mat_encoder, mat_head = load_models_from_dir(
            args.weights_dir, cfg, data
        )
    else:
        std_encoder, std_head, mat_encoder, mat_head = train_from_scratch(
            cfg, data, run_dir
        )

    # ------------------------------------------------------------------
    # Step 3: Compute full embeddings for all 3 models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Computing embeddings")
    print("=" * 60)

    std_emb = get_neural_embeddings(std_encoder, data)  # (n_test, embed_dim)
    mat_emb = get_neural_embeddings(mat_encoder, data)  # (n_test, embed_dim)
    pca_emb = get_pca_full_embeddings(data, cfg)        # (n_test, n_pca_components)

    print(f"[exp2] Standard   embeddings: {std_emb.shape}")
    print(f"[exp2] Matryoshka embeddings: {mat_emb.shape}")
    print(f"[exp2] PCA        embeddings: {pca_emb.shape}")

    model_embeddings = {
        "Standard":   std_emb,
        "Matryoshka": mat_emb,
        "PCA":        pca_emb,
    }
    model_names = list(model_embeddings.keys())

    # Fixed subsample indices — identical across all models so scatter plots
    # show exactly the same test points (essential for fair visual comparison)
    n_test  = len(y_test_np)
    viz_idx = subsample_indices(n_test, max_viz_samples, cfg.seed)
    y_viz   = y_test_np[viz_idx]

    print(
        f"[exp2] Visualizing {len(viz_idx)} / {n_test} test samples "
        f"(same subsample across all models)."
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluate prefix accuracy (reuse exp1 evaluation helpers)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Evaluating prefix accuracy")
    print("=" * 60)

    std_acc = evaluate_prefix_sweep(std_encoder, std_head, data, cfg, "standard")
    mat_acc = evaluate_prefix_sweep(mat_encoder, mat_head, data, cfg, "mat")
    pca_acc = evaluate_pca_baseline(data, cfg)

    accuracy_dict = {}
    for k in cfg.eval_prefixes:
        accuracy_dict[("Standard",   k)] = std_acc[k]
        accuracy_dict[("Matryoshka", k)] = mat_acc[k]
        accuracy_dict[("PCA",        k)] = pca_acc[k]

    # ------------------------------------------------------------------
    # Step 5: Compute cluster metrics for ALL eval_prefixes
    # (metrics use the full test set, not the viz subsample, for accuracy)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Computing cluster metrics")
    print("=" * 60)

    metrics = {}
    for k in cfg.eval_prefixes:
        print(f"\n  k={k}:")
        for model_name, full_emb in model_embeddings.items():
            prefix_emb = extract_prefix(full_emb, k)
            m = compute_cluster_metrics(
                prefix_emb, y_test_np,
                max_samples=max_metric_samples,
                seed=cfg.seed,
            )
            metrics[(model_name, k)] = m
            print(
                f"    {model_name:<12}: sil={m['silhouette']:+.4f}  "
                f"intra={m['intra']:.4f}  inter={m['inter']:.4f}  "
                f"sep={m['separation']:.3f}"
            )

    # ------------------------------------------------------------------
    # Step 6: t-SNE reductions for viz_prefixes
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Running t-SNE reductions")
    print("=" * 60)

    tsne_2d = {}
    for k in viz_prefixes:
        print(f"\n  t-SNE for k={k} ...")
        for model_name, full_emb in model_embeddings.items():
            prefix_viz = extract_prefix(full_emb, k)[viz_idx]
            coords = reduce_to_2d(
                prefix_viz,
                method="t-SNE",
                seed=cfg.seed,
                n_iter_tsne=tsne_n_iter,
                perplexity=tsne_perplexity,
            )
            tsne_2d[(model_name, k)] = coords
            print(f"    {model_name}: done  shape={coords.shape}")

    # ------------------------------------------------------------------
    # Step 7: UMAP reductions for viz_prefixes (if umap-learn is installed)
    # ------------------------------------------------------------------
    # Check umap availability now (lazy — avoids numba init before data loading)
    umap_2d = {}
    global HAS_UMAP
    if HAS_UMAP is None:
        try:
            import umap as _umap_check  # noqa: F401
            HAS_UMAP = True
            print("[exp2] umap-learn found — UMAP plots will be generated.")
        except ImportError:
            HAS_UMAP = False
            print("[exp2] umap-learn not installed — UMAP plots will be skipped.")
    if HAS_UMAP:
        print("=" * 60)
        print("STEP 7: Running UMAP reductions")
        print("=" * 60)

        for k in viz_prefixes:
            print(f"\n  UMAP for k={k} ...")
            for model_name, full_emb in model_embeddings.items():
                prefix_viz = extract_prefix(full_emb, k)[viz_idx]
                coords = reduce_to_2d(
                    prefix_viz,
                    method="UMAP",
                    seed=cfg.seed,
                )
                umap_2d[(model_name, k)] = coords
                print(f"    {model_name}: done  shape={coords.shape}")
    else:
        print("\n[exp2] STEP 7: Skipping UMAP (umap-learn not installed).")

    # ------------------------------------------------------------------
    # Step 8: Generate all plots
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Generating plots")
    print("=" * 60)

    # t-SNE scatter grid (always generated)
    plot_viz_grid(tsne_2d, y_viz, model_names, viz_prefixes, "t-SNE", run_dir,
                  accuracy_dict=accuracy_dict, metrics=metrics)

    # UMAP scatter grid (generated if umap-learn is available)
    if HAS_UMAP and umap_2d:
        plot_viz_grid(umap_2d, y_viz, model_names, viz_prefixes, "UMAP", run_dir,
                      accuracy_dict=accuracy_dict, metrics=metrics)

    # Cluster quality metrics vs k (silhouette, intra/inter, separation ratio)
    plot_cluster_metrics(metrics, cfg.eval_prefixes, model_names, run_dir)

    # Combined accuracy + silhouette vs k (geometry <-> performance link)
    plot_combined_summary(accuracy_dict, metrics, cfg.eval_prefixes, model_names, run_dir)

    # 4D animated GIF: cycle through 6 pairwise 2D scatters of dims 0-3.
    # fps=1 → 1 second/frame in Pillow metadata; some GIF viewers ignore delays
    # and replay faster.  Use dim4_slideshow.html for reliable frame pacing.
    dims_4 = [0, 1, 2, 3]
    plot_4d_animation(model_embeddings, y_viz, viz_idx, run_dir, dims=dims_4, fps=1)

    # 4D interactive HTML: 3D scatter with dropdown for all C(4,3)=4 views (requires plotly)
    plot_4d_interactive(model_embeddings, y_viz, viz_idx, run_dir, dims=dims_4)

    # 4D slideshow HTML: same 6 frames as the GIF, stepped manually with Prev/Next
    # buttons or played at an adjustable speed — fixes the "GIF too fast" problem.
    plot_4d_slideshow(model_embeddings, y_viz, viz_idx, run_dir, dims=dims_4)

    # ------------------------------------------------------------------
    # Step 9: Results summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Saving results summary")
    print("=" * 60)

    save_results_summary(accuracy_dict, metrics, cfg.eval_prefixes, model_names, run_dir)

    # Print compact table to stdout
    print(f"\n{'k':>4}  {'Model':<12}  {'Accuracy':>10}  {'Silhouette':>12}  {'Sep.Ratio':>10}")
    print("-" * 54)
    for k in cfg.eval_prefixes:
        for model_name in model_names:
            acc = accuracy_dict.get((model_name, k), float("nan"))
            m   = metrics.get((model_name, k), {})
            print(
                f"{k:>4}  {model_name:<12}  {acc:>10.4f}  "
                f"{m.get('silhouette', 0):>12.4f}  {m.get('separation', 0):>10.4f}"
            )
        print()

    # ------------------------------------------------------------------
    # Step 10: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 10: Saving runtime and code snapshot")
    print("=" * 60)

    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp2] Experiment 2 complete.")
    print(f"[exp2] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
