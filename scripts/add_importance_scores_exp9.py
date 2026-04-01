"""
Script: scripts/add_importance_scores_exp9.py
----------------------------------------------
One-off script to add importance_scores.png to an existing exp9 run folder
without retraining.

For each seed subfolder (seed_42/, seed_123/) it:
  1. Loads saved encoder weights (standard, l1, mat).
  2. Loads the matching data split (using the same seed).
  3. Fits PCA on the training set.
  4. Extracts embeddings for all 4 models.
  5. Computes per-dimension importance scores (mean_abs, variance, probe_acc).
  6. Saves importance_scores.png into the seed subfolder.

Usage:
    conda activate mrl_env
    python scripts/add_importance_scores_exp9.py --run-dir PATH

Example:
    python scripts/add_importance_scores_exp9.py \
        --run-dir "../files/results/exprmnt_2026_03_20__22_15_23"

Inputs:
    --run-dir  PATH : path to the exp9 timestamped output folder.
    --fast          : use digits dataset + small probe cap (for quick testing).

Outputs:
    {run_dir}/seed_42/importance_scores.png
    {run_dir}/seed_123/importance_scores.png  (if seed_123/ exists)
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from experiments.exp7_mrl_vs_ff import get_embeddings_np
from experiments.exp8_dim_importance import (
    get_pca_embeddings_np,
    compute_importance_scores,
    plot_importance_scores,
)


def load_neural_models(seed_dir: str, cfg: ExpConfig, data):
    """
    Load Standard, L1, and MRL encoder weights from a seed subfolder.

    Args:
        seed_dir (str)      : Path to e.g. run_dir/seed_42/.
        cfg      (ExpConfig): Must match the architecture used when saving.
        data     (DataSplit): Used for input_dim and n_classes.

    Returns:
        Tuple (std_enc, l1_enc, mat_enc): all MLPEncoder in eval mode.
    """
    def _load_enc(fname):
        enc = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        path = os.path.join(seed_dir, fname)
        assert os.path.isfile(path), f"Weight file not found: {path}"
        enc.load_state_dict(torch.load(path, map_location="cpu"))
        enc.eval()
        print(f"  Loaded {fname}")
        return enc

    std_enc = _load_enc("standard_encoder_best.pt")
    l1_enc  = _load_enc("l1_encoder_best.pt")
    mat_enc = _load_enc("mat_encoder_best.pt")
    return std_enc, l1_enc, mat_enc


def process_seed(seed: int, seed_dir: str, cfg_base: ExpConfig,
                  max_probe_samples: int):
    """
    Generate importance_scores.png for a single seed subfolder.

    Args:
        seed             (int)      : Data seed (controls the train/test split).
        seed_dir         (str)      : Path to the seed subfolder.
        cfg_base         (ExpConfig): Base config (architecture / dataset settings).
        max_probe_samples(int)      : Cap on samples for the per-dim LR probe.
    """
    print(f"\n[add_importance] ===== Seed {seed}  ({seed_dir}) =====")

    # Build per-seed config (same as exp9's run_one_seed)
    seed_cfg = ExpConfig(
        dataset         = cfg_base.dataset,
        embed_dim       = cfg_base.embed_dim,
        hidden_dim      = cfg_base.hidden_dim,
        head_mode       = cfg_base.head_mode,
        eval_prefixes   = cfg_base.eval_prefixes,
        epochs          = cfg_base.epochs,
        patience        = cfg_base.patience,
        lr              = cfg_base.lr,
        weight_decay    = cfg_base.weight_decay,
        batch_size      = cfg_base.batch_size,
        val_size        = cfg_base.val_size,
        seed            = seed,
        l1_lambda       = cfg_base.l1_lambda,
        experiment_name = cfg_base.experiment_name,
    )

    # Load data with this seed's split
    print(f"[add_importance] Loading data (seed={seed}) ...")
    data = load_data(seed_cfg)

    # Load saved encoder weights
    print(f"[add_importance] Loading encoder weights from {seed_dir} ...")
    std_enc, l1_enc, mat_enc = load_neural_models(seed_dir, seed_cfg, data)

    # Extract embeddings
    print(f"[add_importance] Extracting embeddings ...")
    Z_train_std = get_embeddings_np(std_enc, data.X_train)
    Z_test_std  = get_embeddings_np(std_enc, data.X_test)

    Z_train_l1  = get_embeddings_np(l1_enc,  data.X_train)
    Z_test_l1   = get_embeddings_np(l1_enc,  data.X_test)

    Z_train_mrl = get_embeddings_np(mat_enc, data.X_train)
    Z_test_mrl  = get_embeddings_np(mat_enc, data.X_test)

    Z_train_pca, Z_test_pca = get_pca_embeddings_np(data, seed_cfg)

    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    all_embeddings = {
        "Standard": (Z_train_std, Z_test_std),
        "L1":       (Z_train_l1,  Z_test_l1),
        "MRL":      (Z_train_mrl, Z_test_mrl),
        "PCA":      (Z_train_pca, Z_test_pca),
    }

    # Compute importance scores for all 4 models
    print(f"[add_importance] Computing importance scores ...")
    all_scores = {}
    for model_name, (Z_train, Z_test) in all_embeddings.items():
        all_scores[model_name] = compute_importance_scores(
            Z_test=Z_test, Z_train=Z_train,
            y_train=y_train_np, y_test=y_test_np,
            max_probe_samples=max_probe_samples,
            seed=seed, model_tag=model_name,
        )

    # Save importance_scores.png into the seed subfolder
    plot_importance_scores(all_scores, seed_dir, seed_cfg)
    print(f"[add_importance] Done — saved to {seed_dir}/importance_scores.png")


def main():
    """
    Parse args and run process_seed for every seed subfolder found in run_dir.
    """
    parser = argparse.ArgumentParser(
        description="Add importance_scores.png to an existing exp9 run folder."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to the exp9 timestamped output folder.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use digits dataset + small probe cap (for quick testing).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    assert os.path.isdir(run_dir), f"run_dir not found: {run_dir}"

    # Config must match what was used in the original exp9 run
    if args.fast:
        cfg = ExpConfig(
            dataset       = "digits",
            embed_dim     = 16,
            hidden_dim    = 128,
            head_mode     = "shared_head",
            eval_prefixes = list(range(1, 17)),
            epochs        = 5,
            patience      = 3,
            seed          = 42,
            l1_lambda     = 0.05,
            experiment_name = "exp9_dense_prefix",
        )
        max_probe_samples = 500
    else:
        cfg = ExpConfig(
            dataset       = "mnist",
            embed_dim     = 64,
            hidden_dim    = 256,
            head_mode     = "shared_head",
            eval_prefixes = list(range(1, 65)),
            epochs        = 20,
            patience      = 5,
            seed          = 42,
            l1_lambda     = 0.05,
            experiment_name = "exp9_dense_prefix",
        )
        max_probe_samples = 2000

    # Discover seed subfolders (seed_42/, seed_123/, etc.)
    seed_dirs = {}
    for entry in sorted(os.listdir(run_dir)):
        if entry.startswith("seed_") and os.path.isdir(os.path.join(run_dir, entry)):
            try:
                seed = int(entry.split("_")[1])
                seed_dirs[seed] = os.path.join(run_dir, entry)
            except ValueError:
                pass

    assert seed_dirs, f"No seed_* subdirectories found in {run_dir}"
    print(f"[add_importance] Found seed dirs: {list(seed_dirs.keys())}")

    for seed, seed_dir in seed_dirs.items():
        process_seed(seed, seed_dir, cfg, max_probe_samples)

    print(f"\n[add_importance] All done. importance_scores.png saved for seeds: "
          f"{list(seed_dirs.keys())}")


if __name__ == "__main__":
    main()
