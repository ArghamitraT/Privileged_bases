"""
Script: experiments/exp13_mrl_cd34_supervised.py
-------------------------------------------------
Exp13 — Supervised MRL on CD34 vs SEACells Baseline.

Train multiple loss variants on CD34 gene expression data using known cell type
labels (HSC, HMP, MEP, Ery, Mono, cDC, pDC, CLP).  Evaluate each model with a
dense prefix sweep (k=1..EMBED_DIM) using k-means clustering and three metrics —
cell type purity, compactness, separation — and compare against the SEACells
metacell baseline stored in the h5ad file.

This is the supervised upper-bound experiment in the MRL-SEACells roadmap.
See plans/mrl_seacells.md for the full paradigm design.

Loss variants (controlled by MODELS_TO_RUN in CONFIG):
    pca           — sklearn PCA, no training, first-k dims = max variance
    ce            — StandardLoss (plain cross-entropy on full embedding)
    mrl           — MatryoshkaLoss (CE summed at each prefix in MRL_TRAIN_PREFIXES)
    fixed_lp      — PrefixLpLoss(p=FIXED_LP_P); p=1 ≡ old PrefixL1 (dims reversed)
    learned_lp    — LearnedPrefixLpLoss (scalar p learned jointly with encoder)
    learned_lp_vec— VectorLearnedPrefixLpLoss (per-dim p learned jointly)

Conda environment: mrl_env

Usage:
    python experiments/exp13_mrl_cd34_supervised.py                     # full run
    python experiments/exp13_mrl_cd34_supervised.py --fast              # smoke test
    python experiments/exp13_mrl_cd34_supervised.py --use-weights PATH  # plots only
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import math
import time
import argparse

import numpy as np
import scipy.sparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ExpConfig
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss, LearnedPrefixLpLoss, VectorLearnedPrefixLpLoss
from training.trainer import train
from utility import create_run_dir, save_runtime, save_code_snapshot

# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATA_PATH = os.path.join(os.environ["HOME"], "Mat_embedding_hyperbole",
                         "data", "cd34_multiome",
                         "GSE200046_cd34_multiome_rna.h5ad")

# HVG selection
N_HVG         = 2000   # number of highly variable genes used as input
RECOMPUTE_HVG = False  # True: recompute via scanpy; False: use h5ad precomputed

# Model
EMBED_DIM  = 8    # match PCA dim used by SEACells
HIDDEN_DIM = 256
HEAD_MODE  = "shared_head"

# MRL training prefixes — used in MatryoshkaLoss during training only
# Evaluation always uses dense k=1..EMBED_DIM for all models regardless of this setting
MRL_TRAIN_PREFIXES = [2, 4, 8, 16, 30]  # sparse subset (fast)
MRL_TRAIN_DENSE    = True               # True: override above with k=1..EMBED_DIM (slow)

# Training
EPOCHS       = 15
PATIENCE     = 5
LR           = 1e-3
BATCH_SIZE   = 128
WEIGHT_DECAY = 1e-4
SEED         = 42

# Loss-specific
L1_LAMBDA        = 0.05  # regularisation weight for fixed_lp / learned_lp / learned_lp_vec
FIXED_LP_P       = 1     # exponent for fixed_lp (1 = old PrefixL1; 3 = soft ordering)
LEARNED_LP_P_INIT = 0.0  # initial p_raw for learned_lp / learned_lp_vec
                          # effective p = 1 + softplus(p_raw); at 0.0 → p ≈ 1.69
                          # set to e.g. 1.0 → p ≈ 2.31 (closer to L2)

# Which models to train and evaluate — edit this list to run any subset
MODELS_TO_RUN = ["pca", "ce", "mrl", "fixed_lp", "learned_lp", "learned_lp_vec"]

# Prefix sizes shown in the 6×3 UMAP grid figure (must be ≤ EMBED_DIM)
UMAP_GRID_KS = [2, 4, 8]

# Evaluation
N_CLUSTERS = 100   # k-means clusters; set to match SEACells n_metacells

# Visualization — t-SNE / UMAP / HTML (per-model, per-prefix-k)
VIZ_PREFIXES    = [1, 2, 4, 8, 16, 30]  # which k values to visualise
MAX_VIZ_SAMPLES = 2000                   # subsample for projection speed
TSNE_N_ITER     = 500
TSNE_PERPLEXITY = 30
# ==============================================================================

# Display names for plot legends
LEGEND = {
    "pca":            "PCA",
    "ce":             "CE",
    "mrl":            "MRL",
    "fixed_lp":       f"FixedLp(p={FIXED_LP_P}) (rev)",
    "learned_lp":     "LearnedLp",
    "learned_lp_vec": "LearnedLpVec",
}

# Map model tag → build_loss model_type string
def _loss_type(tag: str) -> str:
    if tag == "ce":             return "standard"
    if tag == "mrl":            return "matryoshka"
    if tag == "fixed_lp":       return f"prefix_l{FIXED_LP_P}"
    if tag == "learned_lp":     return "prefix_lp_learned"
    if tag == "learned_lp_vec": return "prefix_lp_vector_learned"
    raise ValueError(f"Unknown model tag: {tag}")


# Named container matching the trainer's expectations
DataSplit = namedtuple(
    "DataSplit",
    ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "input_dim", "n_classes"]
)

# Cell type display colours (consistent with seacells_cd34.py)
CELLTYPE_ORDER  = ["HSC", "HMP", "MEP", "Ery", "Mono", "cDC", "pDC", "CLP"]
CELLTYPE_COLORS = {
    "HSC":  "#B34B78", "HMP": "#985275", "MEP": "#3D611A", "Ery": "#4A8C47",
    "Mono": "#E68A98", "cDC": "#62B1EC", "pDC": "#294D64", "CLP": "#583E18",
}


# ==============================================================================
# Data loading
# ==============================================================================

def load_cd34_data(data_path: str, n_hvg: int, recompute_hvg: bool,
                   fast: bool, seed: int):
    """
    Load CD34 h5ad, extract HVG expression matrix, cell type labels, PCA, UMAP.

    Returns
    -------
    X_hvg       : (n_cells, n_hvg) float32 array — log-normalised HVG expression
    y_int       : (n_cells,) int64 array — encoded cell type labels (0–7)
    X_pca       : (n_cells, 30) float32 array — precomputed PCA
    X_umap      : (n_cells, 2)  float32 array — precomputed UMAP
    seacell_ids : (n_cells,) str array — paper's original SEACell assignments
    label_names : list of str — class names indexed by y_int
    y_str       : (n_cells,) str array — raw cell type strings
    """
    print(f"Loading {data_path} ...")
    adata = sc.read_h5ad(data_path)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  Cell types: {dict(adata.obs['celltype'].value_counts())}")

    if fast:
        sc.pp.subsample(adata, n_obs=500, random_state=seed)
        print(f"  --fast: subsampled to {adata.n_obs} cells")

    # HVG selection
    if recompute_hvg:
        print(f"  Recomputing HVGs (n_top_genes={n_hvg}) ...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    else:
        # Verify precomputed HVGs exist
        if "highly_variable" not in adata.var.columns:
            raise RuntimeError(
                "h5ad has no 'highly_variable' column. Set RECOMPUTE_HVG=True."
            )

    hvg_mask = adata.var["highly_variable"].values
    n_used   = hvg_mask.sum()
    print(f"  Using {n_used} HVGs as input features")

    X_sub = adata[:, hvg_mask].X
    if scipy.sparse.issparse(X_sub):
        X_hvg = np.asarray(X_sub.todense(), dtype=np.float32)
    else:
        X_hvg = np.asarray(X_sub, dtype=np.float32)

    y_str       = adata.obs["celltype"].values.astype(str)
    le          = LabelEncoder()
    y_int       = le.fit_transform(y_str).astype(np.int64)
    label_names = list(le.classes_)

    X_pca       = adata.obsm["X_pca"].astype(np.float32)
    X_umap      = adata.obsm["X_umap"].astype(np.float32)
    seacell_ids = adata.obs["SEACell"].values.astype(str)

    print(f"  X_hvg shape: {X_hvg.shape}")
    print(f"  Classes ({len(label_names)}): {label_names}")

    return X_hvg, y_int, X_pca, X_umap, seacell_ids, label_names, y_str


def make_data_split(X_hvg, y_int, test_size, val_size, seed):
    """Stratified train/val/test split → DataSplit namedtuple of torch tensors."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_hvg, y_int, test_size=test_size, stratify=y_int, random_state=seed
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed
    )

    def _t(arr, dtype):
        return torch.tensor(arr, dtype=dtype)

    return DataSplit(
        X_train=_t(X_tr, torch.float32), y_train=_t(y_tr, torch.long),
        X_val  =_t(X_va, torch.float32), y_val  =_t(y_va, torch.long),
        X_test =_t(X_te, torch.float32), y_test =_t(y_te, torch.long),
        input_dim=X_hvg.shape[1],
        n_classes=len(np.unique(y_int)),
    )


# ==============================================================================
# Training
# ==============================================================================

def build_cfg(embed_dim, hidden_dim, head_mode, mrl_train_prefixes,
              lr, epochs, batch_size, patience, weight_decay, seed, l1_lambda):
    """Build ExpConfig used during training (eval_prefixes = MRL training prefixes)."""
    return ExpConfig(
        dataset="cd34",
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        head_mode=head_mode,
        eval_prefixes=mrl_train_prefixes,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        weight_decay=weight_decay,
        seed=seed,
        l1_lambda=l1_lambda,
        experiment_name="exp13_mrl_cd34_supervised",
    )


def train_model(tag, cfg, data, run_dir):
    """
    Build, train, and save one model.

    Returns
    -------
    encoder : trained MLPEncoder (best weights loaded)
    history : dict with train_losses, val_losses, best_epoch
    """
    print(f"\n{'='*60}")
    print(f"  Training: {tag}  ({LEGEND[tag]})")
    print(f"{'='*60}")

    torch.manual_seed(cfg.seed)

    encoder = MLPEncoder(input_dim=data.input_dim, hidden_dim=cfg.hidden_dim,
                         embed_dim=cfg.embed_dim)
    head    = build_head(cfg, n_classes=data.n_classes)

    # Learned losses are constructed directly so p_init is honoured;
    # all other losses go through the shared build_loss() factory.
    if tag == "learned_lp":
        loss_fn = LearnedPrefixLpLoss(
            embed_dim=cfg.embed_dim, lambda_l1=cfg.l1_lambda,
            p_init=LEARNED_LP_P_INIT,
        )
    elif tag == "learned_lp_vec":
        loss_fn = VectorLearnedPrefixLpLoss(
            embed_dim=cfg.embed_dim, lambda_l1=cfg.l1_lambda,
            p_init=LEARNED_LP_P_INIT,
        )
    else:
        loss_fn = build_loss(cfg, _loss_type(tag))

    # Always include loss_fn parameters — harmless for non-learnable losses
    params = (list(encoder.parameters())
              + list(head.parameters())
              + list(loss_fn.parameters()))
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = train(encoder, head, loss_fn, opt, data, cfg, run_dir, tag)

    # Save final (best) weights for --use-weights reloading
    torch.save(encoder.state_dict(), os.path.join(run_dir, f"{tag}_encoder.pt"))
    torch.save(head.state_dict(),    os.path.join(run_dir, f"{tag}_head.pt"))

    # Save loss state dict for learned-p variants (captures final p_raw value)
    if hasattr(loss_fn, "p_raw"):
        torch.save(loss_fn.state_dict(), os.path.join(run_dir, f"{tag}_loss.pt"))
        p_eff = loss_fn.p if loss_fn.p.ndim == 0 else loss_fn.p
        if p_eff.ndim == 0:
            print(f"  [{tag}] Final learned p = {p_eff.item():.4f}")
        else:
            vals = p_eff.detach().tolist()
            print(f"  [{tag}] Final learned p (per dim) = "
                  f"[{', '.join(f'{v:.4f}' for v in vals)}]")
            print(f"  [{tag}]   mean={sum(vals)/len(vals):.4f}  "
                  f"min={min(vals):.4f}  max={max(vals):.4f}")

    return encoder, history


# ==============================================================================
# Embedding extraction
# ==============================================================================

def extract_embeddings(models_to_run, trained_encoders, X_hvg_all,
                       X_pca, embed_dim):
    """
    Extract full-dataset embeddings for all trained models + PCA baseline.

    Returns
    -------
    embeddings : dict {tag: np.ndarray (n_cells, embed_dim)}
    """
    embeddings = {}
    X_tensor = torch.tensor(X_hvg_all, dtype=torch.float32)

    for tag in models_to_run:
        if tag == "pca":
            # Use first embed_dim PCA components as the PCA embedding
            embeddings["pca"] = X_pca[:, :embed_dim].copy()
            print(f"  [pca] Using precomputed X_pca[:, :{embed_dim}]")
            continue

        encoder = trained_encoders[tag]
        encoder.eval()
        with torch.no_grad():
            Z = encoder(X_tensor).numpy()   # (n_cells, embed_dim)

        # FixedLp reversal: dim 0 is most penalised → least informative → flip
        if tag == "fixed_lp":
            Z = np.ascontiguousarray(Z[:, ::-1])
            print(f"  [{tag}] Dims reversed (FixedLp convention)")

        embeddings[tag] = Z
        print(f"  [{tag}] Embedding shape: {Z.shape}")

    return embeddings


# ==============================================================================
# Clustering metrics
# ==============================================================================

def _celltype_purity(cluster_labels, y_int):
    """Median fraction of dominant cell type per cluster (higher = better)."""
    purities = []
    for c in np.unique(cluster_labels):
        mask   = cluster_labels == c
        counts = np.bincount(y_int[mask], minlength=int(y_int.max()) + 1)
        purities.append(counts.max() / mask.sum())
    return float(np.median(purities))


def _compactness(Z_k, cluster_labels):
    """
    Median mean distance from each cell to its cluster centroid (lower = better).
    Uses centroid distance instead of pairwise for efficiency.
    """
    scores = []
    unique = np.unique(cluster_labels)
    centroids = np.array([Z_k[cluster_labels == c].mean(axis=0) for c in unique])
    for i, c in enumerate(unique):
        mask = cluster_labels == c
        dists = np.linalg.norm(Z_k[mask] - centroids[i], axis=1)
        scores.append(dists.mean())
    return float(np.median(scores))


def _separation(Z_k, cluster_labels):
    """
    Median distance from each cluster centroid to its nearest-neighbour centroid
    (higher = better).
    """
    unique    = np.unique(cluster_labels)
    centroids = np.array([Z_k[cluster_labels == c].mean(axis=0) for c in unique])
    D         = pairwise_distances(centroids)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(axis=1)))


def _median_pairwise_dist(Z_k, max_subsample=2000, seed=42):
    """
    Estimate median pairwise Euclidean distance via subsampling.
    Used to normalize compactness/separation so they are scale-invariant
    and comparable across models with different embedding magnitudes.
    """
    n = Z_k.shape[0]
    if n > max_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_subsample, replace=False)
        Z_sub = Z_k[idx]
    else:
        Z_sub = Z_k
    D = pairwise_distances(Z_sub, metric="euclidean")
    triu = D[np.triu_indices(len(Z_sub), k=1)]
    return float(np.median(triu))


def run_prefix_sweep(embeddings, y_int, eval_prefixes, n_clusters, seed):
    """
    For each model × prefix k, run k-means and compute purity/compactness/separation.
    Compactness and separation are normalised by the median pairwise distance of Z_k
    so that values are scale-invariant and comparable across models.

    Returns
    -------
    results : dict {tag: {'purity': [...], 'compactness': [...], 'separation': [...]}}
    """
    results = {tag: {"purity": [], "compactness": [], "separation": []}
               for tag in embeddings}

    for tag, Z in embeddings.items():
        print(f"\n  Prefix sweep: {tag}")
        for k in eval_prefixes:
            Z_k  = Z[:, :k]
            km   = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
            cl   = km.fit_predict(Z_k)
            norm = _median_pairwise_dist(Z_k, seed=seed)

            p = _celltype_purity(cl, y_int)
            c = _compactness(Z_k, cl) / norm if norm > 0 else 0.0
            s = _separation(Z_k, cl)  / norm if norm > 0 else 0.0

            results[tag]["purity"].append(p)
            results[tag]["compactness"].append(c)
            results[tag]["separation"].append(s)

        print(f"    purity range:       [{min(results[tag]['purity']):.3f}, "
              f"{max(results[tag]['purity']):.3f}]")
        print(f"    compactness range:  [{min(results[tag]['compactness']):.4f}, "
              f"{max(results[tag]['compactness']):.4f}]")
        print(f"    separation range:   [{min(results[tag]['separation']):.4f}, "
              f"{max(results[tag]['separation']):.4f}]")

    return results


def compute_seacell_reference(seacell_ids, y_int, X_pca):
    """
    Compute SEACells reference metrics from the paper's original assignments.
    Compactness and separation use PCA space and are normalised by the median
    pairwise distance of X_pca for scale-invariant comparison.

    Returns dict with scalar purity, compactness, separation.
    """
    le_sc     = LabelEncoder()
    sc_labels = le_sc.fit_transform(seacell_ids)
    norm      = _median_pairwise_dist(X_pca, seed=42)

    purity  = _celltype_purity(sc_labels, y_int)
    compact = _compactness(X_pca, sc_labels) / norm if norm > 0 else 0.0
    sep     = _separation(X_pca, sc_labels)  / norm if norm > 0 else 0.0

    print(f"\n  SEACells reference (paper):")
    print(f"    purity={purity:.3f}  compactness={compact:.4f}  separation={sep:.4f}")

    return {"purity": purity, "compactness": compact, "separation": sep}


# ==============================================================================
# Plotting
# ==============================================================================

METRIC_LABELS = {
    "purity":      ("Cell Type Purity", "higher is better"),
    "compactness": ("Compactness (normalised by median pairwise dist)", "lower is better"),
    "separation":  ("Separation (normalised by median pairwise dist)", "higher is better"),
}


def plot_prefix_curves(results, seacell_ref, eval_prefixes, models_to_run,
                       run_dir, fig_stamp):
    """One figure per metric: all models + SEACells dashed reference."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {tag: colors[i % len(colors)]
                 for i, tag in enumerate(models_to_run)}

    for metric, (ylabel, direction) in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for tag in models_to_run:
            if tag not in results:
                continue
            ax.plot(eval_prefixes, results[tag][metric],
                    marker="o", markersize=3, linewidth=1.5,
                    label=LEGEND[tag], color=color_map[tag])

        # SEACells dashed reference line
        ref_val = seacell_ref[metric]
        ax.axhline(ref_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"SEACells (paper) = {ref_val:.3f}")

        ax.set_xlabel("Prefix size k (dims)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"CD34 — {ylabel}\n({direction})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        path = os.path.join(run_dir, f"prefix_{metric}_curve{fig_stamp}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_training_curves(histories, models_to_run, run_dir, fig_stamp):
    """Loss vs epoch for all trained models (train + val)."""
    trained = [t for t in models_to_run if t != "pca" and t in histories]
    if not trained:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for tag in trained:
        h = histories[tag]
        epochs = range(1, len(h["train_losses"]) + 1)
        axes[0].plot(epochs, h["train_losses"], label=LEGEND[tag], linewidth=1.5)
        axes[1].plot(epochs, h["val_losses"],   label=LEGEND[tag], linewidth=1.5)

    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(run_dir, f"training_curves{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_umap_comparison(X_umap, y_str, embeddings, seacell_ids,
                         results, eval_prefixes, models_to_run, run_dir, fig_stamp,
                         seacell_ref):
    """
    3-panel UMAP: (1) ground-truth cell types, (2) MRL clusters at best k,
    (3) SEACells paper assignments.

    Style: dark background, cells coloured by cell type in all panels,
    cluster / SEACell centroids overlaid as white rings.
    """
    # Find best prefix for MRL (highest purity)
    best_tag = "mrl" if "mrl" in results else next(
        (t for t in models_to_run if t not in ("pca", "ce")), None
    )

    if best_tag is None or best_tag not in results:
        print("  UMAP comparison skipped (no suitable trained model)")
        return

    best_k_idx  = int(np.argmax(results[best_tag]["purity"]))
    best_k      = eval_prefixes[best_k_idx]
    best_purity = results[best_tag]["purity"][best_k_idx]
    Z           = embeddings[best_tag]
    n_seacells  = len(np.unique(seacell_ids))
    km          = KMeans(n_clusters=n_seacells, n_init=10, random_state=SEED)
    mrl_cl      = km.fit_predict(Z[:, :best_k])

    le_sc     = LabelEncoder()
    sc_labels = le_sc.fit_transform(seacell_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")

    dot_kw = dict(s=5, alpha=0.7, rasterized=True)

    # Helper: scatter cells by cell type
    def _scatter_celltypes(ax, with_legend=False):
        for ct in CELLTYPE_ORDER:
            mask = y_str == ct
            if mask.sum() == 0:
                continue
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       color=CELLTYPE_COLORS.get(ct, "grey"),
                       label=ct, **dot_kw)
        if with_legend:
            ax.legend(markerscale=3, fontsize=8, loc="best",
                      framealpha=0.4, labelcolor="white",
                      facecolor="black", edgecolor="grey")

    # Helper: overlay cluster centroids (UMAP mean per cluster) as white rings
    def _overlay_centroids(ax, cluster_labels):
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            cx = X_umap[mask, 0].mean()
            cy = X_umap[mask, 1].mean()
            ax.scatter(cx, cy, s=100, facecolors="none",
                       edgecolors="white", linewidths=1.2, zorder=5)

    title_kw = dict(color="white", fontsize=11)

    # Panel 1 — ground-truth cell types
    _scatter_celltypes(axes[0], with_legend=True)
    axes[0].set_title("Ground-truth cell types", **title_kw)
    axes[0].axis("off")

    # Panel 2 — MRL clusters (cells by cell type, centroids as rings)
    _scatter_celltypes(axes[1])
    _overlay_centroids(axes[1], mrl_cl)
    axes[1].set_title(f"{LEGEND[best_tag]} clusters (k={best_k}, "
                      f"purity={best_purity:.3f})", **title_kw)
    axes[1].axis("off")

    # Panel 3 — SEACells (cells by cell type, SEACell centroids as rings)
    _scatter_celltypes(axes[2])
    _overlay_centroids(axes[2], sc_labels)
    seacell_purity = seacell_ref["purity"]
    axes[2].set_title(f"SEACells (paper, purity={seacell_purity:.3f})", **title_kw)
    axes[2].axis("off")

    plt.suptitle("CD34 — UMAP Comparison", fontsize=13, color="white", y=1.01)
    plt.tight_layout()
    path = os.path.join(run_dir, f"umap_comparison{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==============================================================================
# 6×3 UMAP grid — all models × selected prefix sizes
# ==============================================================================

def plot_all_models_umap_grid(embeddings, results, eval_prefixes, models_to_run,
                               X_umap, y_str, y_int, n_clusters,
                               grid_ks, run_dir, fig_stamp):
    """
    6-row × len(grid_ks)-col UMAP grid.

    Rows  : one per model in models_to_run (in order)
    Cols  : one per k in grid_ks (e.g. [2, 4, 8])
    Panel : cells coloured by ground-truth cell type (dark background)
            + white rings at k-means cluster centroids in UMAP space
    Title : model name, k, purity
    """
    n_rows = len(models_to_run)
    n_cols = len(grid_ks)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             facecolor="black")
    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.axis("off")

    dot_kw = dict(s=3, alpha=0.6, rasterized=True)

    def _scatter_celltypes(ax):
        for ct in CELLTYPE_ORDER:
            mask = y_str == ct
            if mask.sum() == 0:
                continue
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       color=CELLTYPE_COLORS.get(ct, "grey"),
                       **dot_kw)

    def _overlay_centroids(ax, cluster_labels):
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            cx = X_umap[mask, 0].mean()
            cy = X_umap[mask, 1].mean()
            # dominant cell type in this cluster → fill colour (SEACells paper style)
            cts_in_cluster = y_str[mask]
            vals, counts = np.unique(cts_in_cluster, return_counts=True)
            dominant_ct = vals[counts.argmax()]
            fill_color = CELLTYPE_COLORS.get(dominant_ct, "grey")
            ax.scatter(cx, cy, s=120, facecolors=fill_color,
                       edgecolors="white", linewidths=1.2, zorder=5)

    for row_idx, tag in enumerate(models_to_run):
        if tag not in embeddings:
            continue
        Z = embeddings[tag]

        for col_idx, k in enumerate(grid_ks):
            ax = axes[row_idx, col_idx]

            # k-means in embedding space
            Z_k = Z[:, :k]
            km  = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
            cl  = km.fit_predict(Z_k)

            # purity from pre-computed results (avoids recomputing)
            k_idx   = eval_prefixes.index(k) if k in eval_prefixes else None
            purity  = (results[tag]["purity"][k_idx]
                       if k_idx is not None else _celltype_purity(cl, y_int))

            _scatter_celltypes(ax)
            _overlay_centroids(ax, cl)

            ax.set_title(f"{LEGEND[tag]}  k={k}\npurity={purity:.3f}",
                         color="white", fontsize=9)

    plt.suptitle("CD34 — All Models × Prefix Size (cell type purity)",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(run_dir, f"umap_grid_all_models{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==============================================================================
# Per-prefix visualisation — t-SNE / UMAP / interactive HTML
# ==============================================================================

def _reduce_to_2d(Z_k, method, seed, n_iter=500, perplexity=30):
    """
    Project (n, k) prefix embeddings to 2D.
      k=1 → value on x-axis + uniform y-jitter
      k=2 → return as-is
      k>2 → t-SNE or UMAP
    """
    n, k = Z_k.shape
    if k == 1:
        rng = np.random.default_rng(seed)
        return np.hstack([Z_k, rng.uniform(-0.3, 0.3, (n, 1))])
    if k == 2:
        return Z_k.copy()
    if method == "UMAP":
        try:
            import umap as _umap
            return _umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                              random_state=seed, verbose=False).fit_transform(Z_k)
        except ImportError:
            raise RuntimeError("umap-learn not installed — run: conda install -c conda-forge umap-learn")
    else:
        safe_perp = min(perplexity, max(5, n // 4 - 1))
        return TSNE(n_components=2, perplexity=safe_perp, max_iter=n_iter,
                    init="pca", random_state=seed, learning_rate="auto").fit_transform(Z_k)


def _reduce_to_3d(Z_k, seed):
    """
    Project (n, k) to 3D.
      k==3 → return as-is (no reduction)
      k>3  → UMAP with n_components=3
    """
    if Z_k.shape[1] == 3:
        return Z_k.copy()
    try:
        import umap as _umap
        return _umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                          random_state=seed, verbose=False).fit_transform(Z_k)
    except ImportError:
        raise RuntimeError("umap-learn not installed")


def compute_projections(embeddings, y_str, viz_prefixes, seed,
                        max_viz_samples, tsne_n_iter, tsne_perplexity):
    """
    Subsample cells, then compute t-SNE and UMAP 2D + UMAP 3D for each (tag, k).
    UMAP 2D coords are reused for interactive HTML to avoid re-running UMAP.

    Returns
    -------
    tsne2d   : dict {(tag, k): np.ndarray (n_sub, 2)}
    umap2d   : dict {(tag, k): np.ndarray (n_sub, 2)}
    umap3d   : dict {(tag, k): np.ndarray (n_sub, 3)}  — only for k >= 3
    idx      : subsample indices into the full dataset
    y_sub    : y_str[idx]
    """
    n   = len(y_str)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(n, max_viz_samples), replace=False) if n > max_viz_samples else np.arange(n)
    y_sub = y_str[idx]

    tsne2d, umap2d, umap3d = {}, {}, {}

    for tag, Z in embeddings.items():
        Z_sub = Z[idx]
        for k in viz_prefixes:
            k_eff = min(k, Z_sub.shape[1])
            Z_k   = Z_sub[:, :k_eff]

            for method, store in [("t-SNE", tsne2d), ("UMAP", umap2d)]:
                try:
                    store[(tag, k)] = _reduce_to_2d(Z_k, method, seed,
                                                    tsne_n_iter, tsne_perplexity)
                    print(f"  [{tag}] {method} 2D k={k}: done")
                except Exception as e:
                    print(f"  [{tag}] {method} 2D k={k} failed: {e}")

            if k_eff >= 3:
                try:
                    umap3d[(tag, k)] = _reduce_to_3d(Z_k, seed)
                    print(f"  [{tag}] UMAP 3D k={k}: done")
                except Exception as e:
                    print(f"  [{tag}] UMAP 3D k={k} failed: {e}")

    return tsne2d, umap2d, umap3d, idx, y_sub


def plot_projection_grid(coords_by_k, y_sub, tag, viz_prefixes, method_name,
                         run_dir, fig_stamp):
    """
    One PNG per model: columns = VIZ_PREFIXES, cells coloured by cell type.
    """
    valid_ks = [k for k in viz_prefixes if k in coords_by_k and coords_by_k[k] is not None]
    if not valid_ks:
        return

    n_cols = len(valid_ks)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, k in zip(axes, valid_ks):
        coords = coords_by_k[k]
        for ct in CELLTYPE_ORDER:
            mask = y_sub == ct
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=5, alpha=0.7, color=CELLTYPE_COLORS.get(ct, "grey"),
                       label=ct, rasterized=True)
        ax.set_title(f"k={k}", fontsize=10)
        ax.axis("off")

    # Legend on first panel only
    axes[0].legend(markerscale=2, fontsize=7, loc="best", framealpha=0.7)
    fig.suptitle(f"{LEGEND[tag]} — {method_name} by prefix k", fontsize=12)
    plt.tight_layout()

    method_slug = "tsne" if method_name == "t-SNE" else "umap"
    path = os.path.join(run_dir, f"{method_slug}_{tag}{fig_stamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_html_plots(umap3d, y_sub, models_to_run, viz_prefixes,
                    run_dir, fig_stamp):
    """
    Interactive plotly HTML per (model, k): 3D rotating scatter only.
      k==3  → direct embedding dims as (x, y, z)
      k>3   → UMAP projected to 3D
    Skips silently if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("  plotly not installed — HTML plots skipped. pip install plotly")
        return

    html_dir = os.path.join(run_dir, "html_viz")
    os.makedirs(html_dir, exist_ok=True)

    layout_dark = dict(
        plot_bgcolor="black", paper_bgcolor="black",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="grey", borderwidth=1),
    )

    for tag in models_to_run:
        for k in viz_prefixes:

            # ── 3D rotating ─────────────────────────────────────────────────
            coords3 = umap3d.get((tag, k))
            if coords3 is not None:
                dim_label = "direct" if k == 3 else "UMAP-3D"
                traces3 = []
                for ct in CELLTYPE_ORDER:
                    mask = y_sub == ct
                    if mask.sum() == 0:
                        continue
                    traces3.append(go.Scatter3d(
                        x=coords3[mask, 0], y=coords3[mask, 1], z=coords3[mask, 2],
                        mode="markers",
                        marker=dict(size=3, color=CELLTYPE_COLORS.get(ct, "grey"),
                                    opacity=0.75),
                        name=ct, text=[ct] * mask.sum(),
                        hovertemplate="%{text}<extra></extra>",
                    ))
                fig3 = go.Figure(traces3)
                fig3.update_layout(
                    title=f"{LEGEND[tag]}  —  3D rotating ({dim_label})  k={k}",
                    scene=dict(
                        xaxis=dict(title="Dim 1", backgroundcolor="black",
                                   gridcolor="grey", showbackground=True),
                        yaxis=dict(title="Dim 2", backgroundcolor="black",
                                   gridcolor="grey", showbackground=True),
                        zaxis=dict(title="Dim 3", backgroundcolor="black",
                                   gridcolor="grey", showbackground=True),
                        bgcolor="black",
                    ),
                    **layout_dark,
                )
                path3 = os.path.join(html_dir, f"{tag}_k{k:02d}_3d{fig_stamp}.html")
                pio.write_html(fig3, path3, include_plotlyjs="cdn")
                print(f"  Saved HTML 3D: {path3}")


# ==============================================================================
# Results summary
# ==============================================================================

def save_results_summary(results, seacell_ref, eval_prefixes, models_to_run,
                         run_dir, cfg, n_hvg, n_input_features, fast):
    """Write a text table: model × prefix × purity × compactness × separation."""
    # Read back final learned p from saved loss state dicts (if present)
    learned_p_summary = {}
    for tag in ["learned_lp", "learned_lp_vec"]:
        loss_path = os.path.join(run_dir, f"{tag}_loss.pt")
        if os.path.exists(loss_path):
            sd = torch.load(loss_path, weights_only=True, map_location="cpu")
            p_raw = sd["p_raw"]
            import torch.nn.functional as _F
            p_eff = 1.0 + _F.softplus(p_raw).clamp(max=10.0)
            if p_eff.ndim == 0:
                learned_p_summary[tag] = f"{p_eff.item():.4f}"
            else:
                vals = p_eff.tolist()
                learned_p_summary[tag] = (
                    f"[{', '.join(f'{v:.4f}' for v in vals)}]  "
                    f"mean={sum(vals)/len(vals):.4f}  "
                    f"min={min(vals):.4f}  max={max(vals):.4f}"
                )

    lines = [
        "Experiment      : exp13_mrl_cd34_supervised",
        f"fast            : {fast}",
        f"EMBED_DIM       : {cfg.embed_dim}",
        f"N_HVG (config)  : {n_hvg}  (RECOMPUTE_HVG={RECOMPUTE_HVG})",
        f"Input features  : {n_input_features}  (actual HVGs used)",
        f"N_CLUSTERS : {N_CLUSTERS}",
        f"EPOCHS     : {cfg.epochs}",
        f"L1_LAMBDA  : {cfg.l1_lambda}",
        f"FIXED_LP_P : {FIXED_LP_P}",
        f"MODELS     : {models_to_run}",
        f"LEARNED_LP_P_INIT : {LEARNED_LP_P_INIT}  (eff p₀ ≈ {1.0 + math.log1p(math.exp(LEARNED_LP_P_INIT)):.3f})",
    ]

    if learned_p_summary:
        lines.append("")
        lines.append("Learned p (final, after training):")
        for tag, val in learned_p_summary.items():
            lines.append(f"  {LEGEND[tag]:<18} : p = {val}")

    lines += [
        "",
        "SEACells reference (paper):",
        f"  purity={seacell_ref['purity']:.4f}  "
        f"compactness={seacell_ref['compactness']:.4f}  "
        f"separation={seacell_ref['separation']:.4f}",
        "",
        f"{'Model':<18} {'k':>4}  {'Purity':>8}  {'Compact':>10}  {'Separat':>10}",
        "-" * 58,
    ]

    for tag in models_to_run:
        if tag not in results:
            continue
        for i, k in enumerate(eval_prefixes):
            p = results[tag]["purity"][i]
            c = results[tag]["compactness"][i]
            s = results[tag]["separation"][i]
            lines.append(
                f"{LEGEND[tag]:<18} {k:>4}  {p:>8.4f}  {c:>10.4f}  {s:>10.4f}"
            )
        lines.append("")

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {path}")


def save_experiment_description(run_dir, cfg, n_hvg, n_input_features, fast, use_weights):
    desc = (
        "Experiment  : exp13_mrl_cd34_supervised\n"
        "Goal        : Supervised MRL on CD34 HSPCs — compare prefix-sweep clustering\n"
        "              against SEACells metacell baseline (Persad et al. 2023)\n"
        "Dataset     : GSE200046_cd34_multiome_rna.h5ad "
        "(6,881 CD34+ HSPCs, 8 cell types)\n"
        "Eval        : Dense prefix sweep k=1..EMBED_DIM — purity / compactness / separation\n"
        "\nConfig:\n"
        f"  EMBED_DIM          = {cfg.embed_dim}\n"
        f"  HIDDEN_DIM         = {cfg.hidden_dim}\n"
        f"  N_HVG (config)     = {n_hvg}  (RECOMPUTE_HVG={RECOMPUTE_HVG})\n"
        f"  Input features     = {n_input_features}  (actual HVGs used)\n"
        f"  MRL_TRAIN_PREFIXES = {cfg.eval_prefixes}  (dense={MRL_TRAIN_DENSE})\n"
        f"  EPOCHS             = {cfg.epochs}\n"
        f"  LR                 = {cfg.lr}\n"
        f"  BATCH_SIZE         = {cfg.batch_size}\n"
        f"  PATIENCE           = {cfg.patience}\n"
        f"  WEIGHT_DECAY       = {cfg.weight_decay}\n"
        f"  L1_LAMBDA          = {cfg.l1_lambda}\n"
        f"  FIXED_LP_P         = {FIXED_LP_P}\n"
        f"  LEARNED_LP_P_INIT  = {LEARNED_LP_P_INIT}"
        f"  (eff p₀ ≈ {1.0 + math.log1p(math.exp(LEARNED_LP_P_INIT)):.3f})\n"
        f"  N_CLUSTERS         = {N_CLUSTERS}\n"
        f"  MODELS_TO_RUN      = {MODELS_TO_RUN}\n"
        f"  SEED               = {cfg.seed}\n"
        f"  --fast             = {fast}\n"
        f"  --use-weights      = {use_weights}\n"
    )
    path = os.path.join(run_dir, "experiment_description.log")
    with open(path, "w") as f:
        f.write(desc)
    print(desc)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp13: Supervised MRL on CD34 vs SEACells"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smoke test: 500 cells, 5 epochs, embed_dim=8")
    parser.add_argument("--use-weights", metavar="PATH",
                        help="Load saved weights from PATH; skip training, "
                             "regenerate plots")
    args = parser.parse_args()

    t0        = time.time()
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

    # ------------------------------------------------------------------
    # Fast overrides
    # ------------------------------------------------------------------
    embed_dim          = EMBED_DIM
    hidden_dim         = HIDDEN_DIM
    epochs             = EPOCHS
    n_hvg              = N_HVG
    n_clusters         = N_CLUSTERS
    mrl_train_prefixes = MRL_TRAIN_PREFIXES
    models_to_run      = list(MODELS_TO_RUN)
    viz_prefixes       = list(VIZ_PREFIXES)
    max_viz_samples    = MAX_VIZ_SAMPLES
    tsne_n_iter        = TSNE_N_ITER
    tsne_perplexity    = TSNE_PERPLEXITY

    if args.fast:
        embed_dim          = 8
        hidden_dim         = 128
        epochs             = 5
        n_hvg              = 200
        n_clusters         = 20
        mrl_train_prefixes = [2, 4, 8]
        viz_prefixes       = [1, 4, 8]
        max_viz_samples    = 200
        tsne_n_iter        = 250
        tsne_perplexity    = 20
        # Keep models_to_run as-is so smoke test covers all paths

    # Dense training override: replace sparse prefixes with k=1..embed_dim
    if MRL_TRAIN_DENSE:
        mrl_train_prefixes = list(range(1, embed_dim + 1))
        print(f"[exp13] MRL_TRAIN_DENSE=True → training prefixes: 1..{embed_dim}")

    # Evaluation always uses dense sweep over 1..embed_dim
    eval_prefixes = list(range(1, embed_dim + 1))

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------
    if args.use_weights:
        weights_dir = os.path.abspath(args.use_weights)
        # Resolve bare folder name relative to results root
        if not os.path.isdir(weights_dir):
            results_root = os.path.join(os.environ["HOME"],
                                        "Mat_embedding_hyperbole",
                                        "files", "results")
            weights_dir = os.path.join(results_root, args.use_weights)
        if not os.path.isdir(weights_dir):
            print(f"ERROR: weights folder not found: {weights_dir}")
            sys.exit(1)
        sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[exp13] Run dir (inside weights folder): {run_dir}")
    else:
        run_dir = create_run_dir(fast=args.fast)

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------
    cfg = build_cfg(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        head_mode=HEAD_MODE,
        mrl_train_prefixes=[p for p in mrl_train_prefixes if p <= embed_dim],
        lr=LR,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        l1_lambda=L1_LAMBDA,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found:\n  {DATA_PATH}")
        sys.exit(1)

    X_hvg, y_int, X_pca, X_umap, seacell_ids, _, y_str = load_cd34_data(
        DATA_PATH, n_hvg, RECOMPUTE_HVG, args.fast, SEED
    )
    n_input_features = X_hvg.shape[1]   # actual HVGs loaded (may differ from n_hvg config)

    save_experiment_description(run_dir, cfg, n_hvg, n_input_features,
                                args.fast, args.use_weights)

    # Trim PCA to embed_dim columns if needed
    X_pca_emb = X_pca[:, :embed_dim]

    # ------------------------------------------------------------------
    # Training or weight loading
    # ------------------------------------------------------------------
    trained_encoders = {}
    histories        = {}

    trained_tags = [t for t in models_to_run if t != "pca"]

    if args.use_weights:
        print("\nLoading saved weights ...")
        data = make_data_split(X_hvg, y_int, 0.2, 0.1, SEED)

        for tag in trained_tags:
            enc_path = os.path.join(weights_dir, f"{tag}_encoder.pt")
            if not os.path.exists(enc_path):
                print(f"  [{tag}] No weights found — skipping")
                models_to_run = [m for m in models_to_run if m != tag]
                continue
            encoder = MLPEncoder(input_dim=data.input_dim,
                                 hidden_dim=hidden_dim, embed_dim=embed_dim)
            encoder.load_state_dict(
                torch.load(enc_path, weights_only=True, map_location="cpu")
            )
            encoder.eval()
            trained_encoders[tag] = encoder
            print(f"  [{tag}] Weights loaded: {enc_path}")

    else:
        data = make_data_split(X_hvg, y_int, 0.2, 0.1, SEED)
        print(f"\nData split: train={len(data.X_train)}  "
              f"val={len(data.X_val)}  test={len(data.X_test)}")

        for tag in trained_tags:
            encoder, history        = train_model(tag, cfg, data, run_dir)
            trained_encoders[tag]   = encoder
            histories[tag]          = history

        print("\nPlotting training curves ...")
        plot_training_curves(histories, models_to_run, run_dir, fig_stamp)

    # ------------------------------------------------------------------
    # Extract embeddings (full dataset — all cells)
    # ------------------------------------------------------------------
    print("\nExtracting embeddings ...")
    embeddings = extract_embeddings(
        models_to_run, trained_encoders, X_hvg, X_pca_emb, embed_dim
    )

    # Save embeddings
    emb_path = os.path.join(run_dir, "cd34_embeddings.npz")
    np.savez(emb_path, **embeddings)
    print(f"  Embeddings saved: {emb_path}")

    # ------------------------------------------------------------------
    # SEACells reference metrics
    # ------------------------------------------------------------------
    print("\nComputing SEACells reference metrics ...")
    seacell_ref = compute_seacell_reference(seacell_ids, y_int, X_pca)

    # ------------------------------------------------------------------
    # Prefix sweep evaluation
    # ------------------------------------------------------------------
    print(f"\nRunning prefix sweep (k=1..{embed_dim}) with "
          f"k-means (n_clusters={n_clusters}) ...")
    results = run_prefix_sweep(embeddings, y_int, eval_prefixes, n_clusters, SEED)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nPlotting prefix curves ...")
    plot_prefix_curves(results, seacell_ref, eval_prefixes, models_to_run,
                       run_dir, fig_stamp)

    print("\nPlotting UMAP comparison ...")
    plot_umap_comparison(X_umap, y_str, embeddings, seacell_ids,
                         results, eval_prefixes, models_to_run, run_dir, fig_stamp,
                         seacell_ref)

    print("\nPlotting all-models UMAP grid ...")
    grid_ks = [k for k in UMAP_GRID_KS if k <= embed_dim]
    plot_all_models_umap_grid(embeddings, results, eval_prefixes, models_to_run,
                               X_umap, y_str, y_int, N_CLUSTERS,
                               grid_ks, run_dir, fig_stamp)

    # ------------------------------------------------------------------
    # Per-prefix projections: t-SNE / UMAP PNGs + interactive HTML (2D & 3D)
    # ------------------------------------------------------------------
    # Clip viz_prefixes to eval range so we never request k > embed_dim
    viz_prefixes_clipped = [k for k in viz_prefixes if k <= embed_dim]

    print(f"\nComputing t-SNE / UMAP projections "
          f"(viz_prefixes={viz_prefixes_clipped}, n_sub≤{max_viz_samples}) ...")
    tsne2d, umap2d, umap3d, _, y_viz = compute_projections(
        embeddings, y_str, viz_prefixes_clipped,
        SEED, max_viz_samples, tsne_n_iter, tsne_perplexity,
    )

    print("\nPlotting per-model projection grids ...")
    for tag in models_to_run:
        plot_projection_grid(
            {k: tsne2d.get((tag, k)) for k in viz_prefixes_clipped},
            y_viz, tag, viz_prefixes_clipped, "t-SNE", run_dir, fig_stamp,
        )
        plot_projection_grid(
            {k: umap2d.get((tag, k)) for k in viz_prefixes_clipped},
            y_viz, tag, viz_prefixes_clipped, "UMAP", run_dir, fig_stamp,
        )

    print("\nSaving interactive HTML plots ...")
    save_html_plots(umap3d, y_viz, models_to_run, viz_prefixes_clipped,
                    run_dir, fig_stamp)

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    print("\nSaving results summary ...")
    save_results_summary(results, seacell_ref, eval_prefixes, models_to_run,
                         run_dir, cfg, n_hvg, n_input_features, args.fast)

    # ------------------------------------------------------------------
    # Mandatory outputs
    # ------------------------------------------------------------------
    save_runtime(run_dir, time.time() - t0)
    save_code_snapshot(run_dir)
    print(f"\n[exp13] Output: {run_dir}")


if __name__ == "__main__":
    main()
