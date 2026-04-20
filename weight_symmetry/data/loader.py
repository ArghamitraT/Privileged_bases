"""
Script: weight_symmetry/data/loader.py
---------------------------------------
Dataset loading for weight_symmetry experiments.
Copied and extended from code/data/loader.py.

Supports: mnist, fashion_mnist, digits (sklearn).
Returns raw (unstandardised) tensors plus a StandardScaler fit on train — callers
decide whether to apply it (Exp 1 needs raw pixels to compute ground-truth PCA).

Also provides load_data_with_directions() for Exp 2 (real datasets):
    Supports: "20newsgroups", "mnist_noise", "fashion_mnist_noise"
    Returns (DataSplit, pca_dirs, lda_dirs, dataset_info)

Outputs:
    DataSplit namedtuple:
        X_train, y_train  : training tensors  (float32, long)
        X_val,   y_val    : validation tensors
        X_test,  y_test   : test tensors
        input_dim         : number of input features
        n_classes         : number of target classes
        scaler            : fitted StandardScaler (apply manually if needed)

Usage:
    from weight_symmetry.data.loader import load_data, load_data_with_directions
    split = load_data("mnist", seed=42)
    split, pca_dirs, lda_dirs, info = load_data_with_directions("20newsgroups", seed=42)
"""

import numpy as np
import torch
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DataSplit = namedtuple(
    "DataSplit",
    ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
     "input_dim", "n_classes", "scaler"]
)


# ==============================================================================
# Raw loaders
# ==============================================================================

def _load_mnist():
    from sklearn.datasets import fetch_openml
    print("[loader] Fetching MNIST via fetch_openml ...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    print(f"[loader] MNIST: X={X.shape}, y={y.shape}")
    return X, y


def _load_fashion_mnist():
    from sklearn.datasets import fetch_openml
    print("[loader] Fetching Fashion-MNIST via fetch_openml ...")
    fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    X = fmnist.data.astype(np.float32) / 255.0
    y = fmnist.target.astype(np.int64)
    print(f"[loader] Fashion-MNIST: X={X.shape}, y={y.shape}")
    return X, y


def _load_digits():
    from sklearn.datasets import load_digits
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    print(f"[loader] Digits: X={X.shape}, y={y.shape}")
    return X, y


# ==============================================================================
# Public API
# ==============================================================================

def load_data(
    dataset: str,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
    standardise: bool = True,
    synthetic_variant: str = "nonOrderedLDA",
) -> DataSplit:
    """
    Load dataset, split into train/val/test, optionally standardise.

    Args:
        dataset    : "mnist", "fashion_mnist", or "digits"
        seed       : random seed for splits
        test_size  : fraction of full data for test set
        val_size   : fraction of full data for val set
        standardise: if True, fit StandardScaler on train and apply to all splits

    Returns:
        DataSplit namedtuple (scaler is fitted even if standardise=False,
        so callers can apply it manually)
    """
    # Synthetic dataset has its own load path
    if dataset == "synthetic":
        from weight_symmetry.data.synthetic import load_synthetic, SYNTHETIC_VARIANTS
        vparams = SYNTHETIC_VARIANTS.get(synthetic_variant,
                                         SYNTHETIC_VARIANTS["nonOrderedLDA"])
        raw = load_synthetic(seed=seed, **vparams)
        p = raw["params"]
        print(f"[loader] Synthetic: p={p['p']} C={p['C']} "
              f"train={raw['X_train'].shape[0]} val={raw['X_val'].shape[0]} "
              f"test={raw['X_test'].shape[0]}")
        scaler = StandardScaler(with_std=False).fit(raw["X_train"])  # centred only; scaler for API compat
        return DataSplit(
            X_train   = torch.tensor(raw["X_train"]),
            y_train   = torch.tensor(raw["y_train"], dtype=torch.long),
            X_val     = torch.tensor(raw["X_val"]),
            y_val     = torch.tensor(raw["y_val"],   dtype=torch.long),
            X_test    = torch.tensor(raw["X_test"]),
            y_test    = torch.tensor(raw["y_test"],  dtype=torch.long),
            input_dim = p["p"],
            n_classes = p["C"],
            scaler    = scaler,
        )

    loaders = {
        "mnist":         _load_mnist,
        "fashion_mnist": _load_fashion_mnist,
        "digits":        _load_digits,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(loaders) + ['synthetic']}")

    X, y = loaders[dataset]()
    input_dim = X.shape[1]
    n_classes = len(np.unique(y))

    # Train/test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Train/val split
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv
    )

    print(f"[loader] Split — train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")

    # Fit scaler on train
    scaler = StandardScaler()
    scaler.fit(X_train)

    if standardise:
        X_train = scaler.transform(X_train).astype(np.float32)
        X_val   = scaler.transform(X_val).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)
        print("[loader] Standardisation applied.")

    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    return DataSplit(
        X_train   = to_tensor(X_train),
        y_train   = to_tensor(y_train, dtype=torch.long),
        X_val     = to_tensor(X_val),
        y_val     = to_tensor(y_val,   dtype=torch.long),
        X_test    = to_tensor(X_test),
        y_test    = to_tensor(y_test,  dtype=torch.long),
        input_dim = input_dim,
        n_classes = n_classes,
        scaler    = scaler,
    )


# ==============================================================================
# Helpers for Exp 2 (real datasets with PCA + LDA directions)
# ==============================================================================

def _compute_pca_lda_directions(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    n_components: int = None,
):
    """
    Compute PCA eigenvectors and LDA discriminant directions from training data.

    Args:
        X_train_np   : (n, p) float32/float64 numpy array (centred)
        y_train_np   : (n,) int64 labels
        n_components : how many PCA columns to return (default: all = min(n, p))

    Returns:
        pca_dirs : (p, k) float64 — top-k PCA eigenvectors (SVD right singular vectors)
        lda_dirs : (p, C-1) float64 — QR-orthonormalised LDA scalings
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    p = X_train_np.shape[1]
    k = min(p, n_components or p)

    # PCA via truncated SVD of centred data
    _, _, Vh = np.linalg.svd(X_train_np.astype(np.float64), full_matrices=False)
    pca_dirs = Vh[:k, :].T.astype(np.float64)   # (p, k)

    # LDA
    n_classes = int(len(np.unique(y_train_np)))
    n_lda     = n_classes - 1
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_np, y_train_np)
    lda_raw  = lda.scalings_[:, :n_lda].astype(np.float64)   # (p, C-1), sorted by discriminability
    # MGS orthonormalisation: preserves column order so col 0 stays most discriminative.
    # QR would destroy this ordering.
    Q = np.zeros_like(lda_raw)
    for i in range(lda_raw.shape[1]):
        v = lda_raw[:, i].copy()
        for j in range(i):
            v -= np.dot(Q[:, j], v) * Q[:, j]
        Q[:, i] = v / np.linalg.norm(v)
    lda_dirs = Q

    return pca_dirs.astype(np.float64), lda_dirs.astype(np.float64)


def load_data_with_directions(
    dataset: str,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
    # 20newsgroups
    p_svd: int = 100,
    tfidf_max_features: int = 10000,
    # mnist_noise / fashion_mnist_noise
    p_pca_proj: int = 50,
    n_noise_dims: int = 25,
    sigma_noise: float = 5.0,
) -> tuple:
    """
    Load a real dataset for Exp 2 and return ground-truth subspace directions.

    Supported datasets:
        "20newsgroups"      — TF-IDF → TruncatedSVD(p_svd), input_dim=p_svd
        "mnist_noise"       — PCA-project MNIST to p_pca_proj dims,
                              prepend n_noise_dims of N(0, sigma_noise²) → p = p_pca_proj + n_noise_dims
        "fashion_mnist_noise" — same augmentation as mnist_noise on Fashion-MNIST

    All splits are centred (train mean subtracted).  No variance scaling so that
    the variance contrast between blocks (noise vs signal) is preserved.

    Returns:
        split       : DataSplit namedtuple (input_dim=p, float32 tensors)
        pca_dirs    : (p, k_pca) float64 — top-k_pca PCA eigenvectors of centred train data
        lda_dirs    : (p, C-1)  float64 — QR-orthonormalised LDA directions
        dataset_info: dict of dataset parameters for logging
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 20 Newsgroups
    # ------------------------------------------------------------------
    if dataset == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        print(f"[loader] Fetching 20newsgroups (seed={seed}) ...")
        news   = fetch_20newsgroups(subset="all",
                                    remove=("headers", "footers", "quotes"))
        X_text = news.data
        y_raw  = news.target.astype(np.int64)

        # Index-level split so we can handle sparse text
        idx = np.arange(len(X_text))
        idx_tv, idx_te, y_tv, y_te = train_test_split(
            idx, y_raw, test_size=test_size, random_state=seed, stratify=y_raw)
        val_frac = val_size / (1.0 - test_size)
        idx_tr, idx_va, y_tr, y_va = train_test_split(
            idx_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv)

        X_tr_text = [X_text[i] for i in idx_tr]
        X_va_text = [X_text[i] for i in idx_va]
        X_te_text = [X_text[i] for i in idx_te]

        # TF-IDF (fit on train only)
        tfidf     = TfidfVectorizer(max_features=tfidf_max_features, sublinear_tf=True)
        X_tr_sp   = tfidf.fit_transform(X_tr_text)
        X_va_sp   = tfidf.transform(X_va_text)
        X_te_sp   = tfidf.transform(X_te_text)

        # TruncatedSVD to p_svd dims (fit on train)
        svd       = TruncatedSVD(n_components=p_svd, random_state=seed)
        X_tr_np   = svd.fit_transform(X_tr_sp).astype(np.float32)
        X_va_np   = svd.transform(X_va_sp).astype(np.float32)
        X_te_np   = svd.transform(X_te_sp).astype(np.float32)

        # Centre on train mean
        mu        = X_tr_np.mean(axis=0, keepdims=True)
        X_tr_np  -= mu;  X_va_np -= mu;  X_te_np -= mu

        print(f"[loader] 20newsgroups: p={p_svd}  "
              f"train={len(X_tr_np)}  val={len(X_va_np)}  test={len(X_te_np)}")

        pca_dirs, lda_dirs = _compute_pca_lda_directions(X_tr_np, y_tr, n_components=p_svd)

        n_classes    = int(len(np.unique(y_tr)))
        dataset_info = dict(
            dataset="20newsgroups", p=p_svd, C=n_classes, n_lda=n_classes - 1,
            p_svd=p_svd, tfidf_max_features=tfidf_max_features,
            n_train=len(X_tr_np), n_val=len(X_va_np), n_test=len(X_te_np),
        )
        scaler = StandardScaler(with_std=False).fit(X_tr_np)
        split  = DataSplit(
            X_train=torch.tensor(X_tr_np),
            y_train=torch.tensor(y_tr, dtype=torch.long),
            X_val=torch.tensor(X_va_np),
            y_val=torch.tensor(y_va, dtype=torch.long),
            X_test=torch.tensor(X_te_np),
            y_test=torch.tensor(y_te, dtype=torch.long),
            input_dim=p_svd, n_classes=n_classes, scaler=scaler,
        )
        return split, pca_dirs, lda_dirs, dataset_info

    # ------------------------------------------------------------------
    # MNIST + noise  /  Fashion-MNIST + noise
    # ------------------------------------------------------------------
    elif dataset in ("mnist_noise", "fashion_mnist_noise"):
        from sklearn.decomposition import PCA as SklearnPCA

        loader_fn = _load_mnist if dataset == "mnist_noise" else _load_fashion_mnist
        X_raw, y_raw = loader_fn()

        X_tv, X_te_raw, y_tv, y_te = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=seed, stratify=y_raw)
        val_frac = val_size / (1.0 - test_size)
        X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
            X_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv)

        # PCA project to p_pca_proj dims (fit on train)
        pca_proj  = SklearnPCA(n_components=p_pca_proj, random_state=seed)
        X_tr_pca  = pca_proj.fit_transform(X_tr_raw).astype(np.float32)
        X_va_pca  = pca_proj.transform(X_va_raw).astype(np.float32)
        X_te_pca  = pca_proj.transform(X_te_raw).astype(np.float32)

        # Prepend noise dims (noise first → PCA of augmented data picks noise first)
        n_tr = len(X_tr_pca);  n_va = len(X_va_pca);  n_te = len(X_te_pca)
        noise_tr = rng.normal(0.0, sigma_noise, (n_tr, n_noise_dims)).astype(np.float32)
        noise_va = rng.normal(0.0, sigma_noise, (n_va, n_noise_dims)).astype(np.float32)
        noise_te = rng.normal(0.0, sigma_noise, (n_te, n_noise_dims)).astype(np.float32)

        X_tr_np = np.concatenate([noise_tr, X_tr_pca], axis=1)
        X_va_np = np.concatenate([noise_va, X_va_pca], axis=1)
        X_te_np = np.concatenate([noise_te, X_te_pca], axis=1)

        # Centre on train mean
        mu       = X_tr_np.mean(axis=0, keepdims=True)
        X_tr_np -= mu;  X_va_np -= mu;  X_te_np -= mu

        p = p_pca_proj + n_noise_dims
        print(f"[loader] {dataset}: p={p} (noise={n_noise_dims}, pca={p_pca_proj})  "
              f"train={n_tr}  val={n_va}  test={n_te}")

        pca_dirs, lda_dirs = _compute_pca_lda_directions(X_tr_np, y_tr, n_components=p)

        n_classes    = int(len(np.unique(y_tr)))
        dataset_info = dict(
            dataset=dataset, p=p, C=n_classes, n_lda=n_classes - 1,
            p_pca_proj=p_pca_proj, n_noise_dims=n_noise_dims, sigma_noise=sigma_noise,
            n_train=n_tr, n_val=n_va, n_test=n_te,
        )
        scaler = StandardScaler(with_std=False).fit(X_tr_np)
        split  = DataSplit(
            X_train=torch.tensor(X_tr_np),
            y_train=torch.tensor(y_tr, dtype=torch.long),
            X_val=torch.tensor(X_va_np),
            y_val=torch.tensor(y_va, dtype=torch.long),
            X_test=torch.tensor(X_te_np),
            y_test=torch.tensor(y_te, dtype=torch.long),
            input_dim=p, n_classes=n_classes, scaler=scaler,
        )
        return split, pca_dirs, lda_dirs, dataset_info

    else:
        raise ValueError(
            f"load_data_with_directions: unsupported dataset '{dataset}'. "
            f"Use 'synthetic' (via load_synthetic) or one of: "
            f"20newsgroups, mnist_noise, fashion_mnist_noise"
        )


# ==============================================================================
# Sanity check
# ==============================================================================

if __name__ == "__main__":
    for ds in ["digits"]:
        print(f"\n=== {ds} ===")
        split = load_data(ds, seed=42)
        assert split.X_train.shape[1] == split.input_dim
        assert split.X_test.shape[1]  == split.input_dim
        print(f"  X_train: {split.X_train.shape}  n_classes: {split.n_classes}")
        print("  PASSED")
