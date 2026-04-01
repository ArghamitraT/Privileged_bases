"""
Script: data/loader.py
----------------------
Flexible dataset loading for all experiments.

Supports sklearn toy datasets and MNIST (via torchvision).
Handles train/val/test splitting, standardisation, and conversion
to PyTorch tensors — all driven by ExpConfig.

Inputs:
    cfg (ExpConfig): experiment configuration (dataset name, split sizes, seed)

Outputs:
    DataSplit namedtuple with fields:
        X_train, y_train  : training tensors
        X_val,   y_val    : validation tensors
        X_test,  y_test   : test tensors
        input_dim         : number of input features
        n_classes         : number of target classes

Usage:
    from data.loader import load_data
    split = load_data(cfg)
    python data/loader.py   # quick sanity check across iris, digits, mnist
"""

import numpy as np
import torch
from collections import namedtuple
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Named container so callers can access splits by name, not index
DataSplit = namedtuple(
    "DataSplit",
    ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "input_dim", "n_classes"]
)


# ==============================================================================
# Internal helpers
# ==============================================================================

def _load_sklearn_dataset(name: str):
    """
    Load one of the supported sklearn toy datasets.

    Args:
        name (str): One of 'iris', 'wine', 'breast_cancer', 'digits'.

    Returns:
        Tuple (X, y) as numpy arrays.

    Raises:
        ValueError: If the dataset name is not recognised.
    """
    loaders = {
        "iris":          sk_datasets.load_iris,
        "wine":          sk_datasets.load_wine,
        "breast_cancer": sk_datasets.load_breast_cancer,
        "digits":        sk_datasets.load_digits,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown sklearn dataset '{name}'. "
            f"Choose from: {list(loaders.keys())}"
        )
    data = loaders[name]()
    return data.data.astype(np.float32), data.target.astype(np.int64)


def _load_mnist():
    """
    Load MNIST via sklearn fetch_openml, flatten images to 784-dim vectors.

    Uses sklearn instead of torchvision to avoid a macOS segfault in
    torchvision's SSL/urllib download mechanism (Issue 6 in CLAUDE.md).
    sklearn is already a project dependency and caches the dataset in
    ~/scikit_learn_data/ after the first download (~11 MB).

    Returns:
        Tuple (X, y) as numpy arrays of shape (70000, 784) and (70000,).
    """
    from sklearn.datasets import fetch_openml

    print("[loader] Fetching MNIST via sklearn fetch_openml ...")
    print("[loader] First run downloads ~11 MB to ~/scikit_learn_data/ — subsequent runs are instant.")

    # as_frame=False  → plain numpy arrays (not pandas DataFrame)
    # parser='auto'   → suppress FutureWarning about default parser change
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    # data:   (70000, 784) float64, pixel values 0–255 → normalise to [0, 1]
    # target: (70000,) strings '0'–'9'                 → cast to int64
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    print(f"[loader] MNIST loaded: X={X.shape}, y={y.shape}")
    return X, y


def _to_tensors(*arrays):
    """
    Convert a sequence of numpy arrays to PyTorch tensors.

    Args:
        *arrays: numpy arrays. Float arrays become FloatTensor,
                 int/long arrays become LongTensor.

    Returns:
        Tuple of torch.Tensor in the same order.
    """
    out = []
    for arr in arrays:
        if arr.dtype in (np.float32, np.float64):
            out.append(torch.tensor(arr, dtype=torch.float32))
        else:
            out.append(torch.tensor(arr, dtype=torch.long))
    return tuple(out)


# ==============================================================================
# Public API
# ==============================================================================

def load_data(cfg) -> DataSplit:
    """
    Load a dataset, split into train/val/test, standardise, and return tensors.

    The split order is:
        1. Separate out test set (cfg.test_size fraction of full data)
        2. From remaining data, separate out val set (cfg.val_size fraction)
        3. Fit StandardScaler on training set only, apply to val and test

    Args:
        cfg (ExpConfig): Experiment config. Uses:
                         cfg.dataset, cfg.test_size, cfg.val_size, cfg.seed

    Returns:
        DataSplit: namedtuple with X_train, y_train, X_val, y_val,
                   X_test, y_test, input_dim, n_classes.
    """

    # ------------------------------------------------------------------
    # Step 1: Load raw data
    # ------------------------------------------------------------------
    print(f"[loader] Loading dataset: '{cfg.dataset}' ...")

    if cfg.dataset == "mnist":
        X, y = _load_mnist()
    else:
        X, y = _load_sklearn_dataset(cfg.dataset)

    print(f"[loader] Raw data shape: X={X.shape}, y={y.shape}")
    print(f"[loader] Classes: {np.unique(y).tolist()}  |  n_classes={len(np.unique(y))}")

    input_dim = X.shape[1]
    n_classes = len(np.unique(y))

    # ------------------------------------------------------------------
    # Step 2: Train / test split  (stratified to preserve class balance)
    # ------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Step 3: Train / val split  (from the non-test portion)
    # ------------------------------------------------------------------
    # val_size is expressed as a fraction of the FULL dataset in config,
    # so we adjust it relative to the trainval portion
    val_fraction_of_trainval = cfg.val_size / (1.0 - cfg.test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=cfg.seed,
        stratify=y_trainval,
    )

    print(f"[loader] Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Step 4: Standardise using ONLY training statistics
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print("[loader] Standardisation applied (fit on train only).")

    # ------------------------------------------------------------------
    # Step 5: Convert to PyTorch tensors
    # ------------------------------------------------------------------
    X_train_t, X_val_t, X_test_t = _to_tensors(
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
    )
    y_train_t, y_val_t, y_test_t = _to_tensors(y_train, y_val, y_test)

    print("[loader] Converted to PyTorch tensors.")
    print(f"[loader] Done. input_dim={input_dim}, n_classes={n_classes}\n")

    return DataSplit(
        X_train=X_train_t, y_train=y_train_t,
        X_val=X_val_t,     y_val=y_val_t,
        X_test=X_test_t,   y_test=y_test_t,
        input_dim=input_dim,
        n_classes=n_classes,
    )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from config import ExpConfig

    for ds_name in ["iris", "digits", "mnist"]:
        print(f"\n{'='*50}")
        print(f"Checkpoint: testing dataset = '{ds_name}'")
        print(f"{'='*50}")

        cfg = ExpConfig(dataset=ds_name, test_size=0.2, val_size=0.1)
        split = load_data(cfg)

        # Verify shapes are consistent
        assert split.X_train.shape[1] == split.input_dim, "input_dim mismatch"
        assert split.X_val.shape[1]   == split.input_dim, "val input_dim mismatch"
        assert split.X_test.shape[1]  == split.input_dim, "test input_dim mismatch"
        assert split.X_train.shape[0] == split.y_train.shape[0], "train size mismatch"
        assert split.X_val.shape[0]   == split.y_val.shape[0],   "val size mismatch"
        assert split.X_test.shape[0]  == split.y_test.shape[0],  "test size mismatch"

        print(f"  X_train: {split.X_train.shape}  y_train: {split.y_train.shape}")
        print(f"  X_val:   {split.X_val.shape}    y_val:   {split.y_val.shape}")
        print(f"  X_test:  {split.X_test.shape}   y_test:  {split.y_test.shape}")
        print(f"  PASSED")

    print("\nAll loader checks passed.")
