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
    Load MNIST via torchvision, flatten images to 784-dim vectors.

    Returns:
        Tuple (X, y) as numpy arrays of shape (70000, 784) and (70000,).
    """
    # torchvision is only imported here so sklearn-only users are not forced
    # to install it
    try:
        from torchvision import datasets as tv_datasets, transforms
    except ImportError:
        raise ImportError(
            "torchvision is required for MNIST. "
            "Install it with: pip install torchvision"
        )

    transform = transforms.ToTensor()

    # Download to a local cache inside the project's files/ folder
    # (will be skipped if already downloaded)
    import os, sys
    # Find cache dir relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, "..", "..", "files", "mnist_cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_ds = tv_datasets.MNIST(cache_dir, train=True,  download=True, transform=transform)
    test_ds  = tv_datasets.MNIST(cache_dir, train=False, download=True, transform=transform)

    # Stack into numpy arrays.
    # Convert via .tolist() first to avoid the PyTorch-NumPy bridge entirely
    # (np.array(tensor) calls tensor.__array__() -> .numpy(), which fails when
    # PyTorch was built without NumPy support; .tolist() is always available).
    X_train = np.array(train_ds.data.tolist(), dtype=np.float32).reshape(-1, 784) / 255.0
    y_train = np.array(train_ds.targets.tolist(), dtype=np.int64)
    X_test  = np.array(test_ds.data.tolist(),  dtype=np.float32).reshape(-1, 784) / 255.0
    y_test  = np.array(test_ds.targets.tolist(), dtype=np.int64)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
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
