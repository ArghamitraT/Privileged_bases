"""
Script: weight_symmetry/data/loader.py
---------------------------------------
Dataset loading for weight_symmetry experiments.
Copied and extended from code/data/loader.py.

Supports: mnist, fashion_mnist, digits (sklearn).
Returns raw (unstandardised) tensors plus a StandardScaler fit on train — callers
decide whether to apply it (Exp 1 needs raw pixels to compute ground-truth PCA).

Outputs:
    DataSplit namedtuple:
        X_train, y_train  : training tensors  (float32, long)
        X_val,   y_val    : validation tensors
        X_test,  y_test   : test tensors
        input_dim         : number of input features
        n_classes         : number of target classes
        scaler            : fitted StandardScaler (apply manually if needed)

Usage:
    from weight_symmetry.data.loader import load_data
    split = load_data("mnist", seed=42)
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
        from weight_symmetry.data.synthetic import load_synthetic
        raw = load_synthetic(seed=seed)
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
