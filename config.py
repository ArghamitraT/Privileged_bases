"""
Script: config.py
-----------------
Central configuration for all experiments.

All hyperparameters, dataset choices, and structural decisions live here.
Import ExpConfig and instantiate it at the top of any experiment script.
Changing one value here propagates everywhere — no need to hunt through
multiple files.

Inputs:  None (this is a pure configuration module)
Outputs: ExpConfig dataclass instance
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpConfig:
    """
    Master configuration for Experiment 1: Prefix Performance Curve.

    --- Data ---
    dataset     : which dataset to load. Options:
                    'mnist'          (torchvision, 784 input dims, 10 classes)
                    'iris'           (sklearn,      4 input dims,   3 classes)
                    'wine'           (sklearn,      13 input dims,  3 classes)
                    'breast_cancer'  (sklearn,      30 input dims,  2 classes)
                    'digits'         (sklearn,      64 input dims, 10 classes)
    test_size   : fraction of data held out for testing
    val_size    : fraction of TRAINING data used for validation / early stopping

    --- Model ---
    embed_dim   : size of the learned embedding (output of encoder)
    hidden_dim  : size of the hidden layer inside the MLP encoder
    head_mode   : classifier head design
                    'shared_head' (Mode A) — one Linear(embed_dim, n_classes) head,
                                            prefix is zero-padded back to embed_dim
                    'multi_head'  (Mode B) — one Linear(k, n_classes) head per scale,
                                            prefix is sliced to exactly k dims

    --- MRL ---
    eval_prefixes : list of prefix sizes k to evaluate at test time.
                    Must all be <= embed_dim.
                    Example: [1, 2, 4, 8, 16, 32, 64]

    --- Training ---
    lr              : learning rate for Adam optimizer
    epochs          : maximum number of training epochs
    batch_size      : mini-batch size
    patience        : early stopping — stop if val loss does not improve for
                      this many epochs. Set to None to disable early stopping.
    weight_decay    : L2 regularisation on model weights

    --- Reproducibility ---
    seed        : master random seed for model init + training (numpy, torch, sklearn)

    --- Experiment 5: Seed Stability ---
    data_seed   : seed used for the fixed data split in exp5 (kept separate from
                  model_seeds so only training randomness varies across runs)
    model_seeds : list of seeds used for model initialisation in exp5.
                  Each entry produces one independent Standard + Mat training run.

    --- Output ---
    experiment_name : used to name the timestamped run output folder
    """

    # --- Data ---
    dataset:    str   = "mnist" # "mnist" digits
    test_size:  float = 0.2
    val_size:   float = 0.1      # fraction of training set used for validation

    # --- Model ---
    embed_dim:  int   = 64
    hidden_dim: int   = 256
    head_mode:  str   = "shared_head"   # 'shared_head' or 'multi_head'

    # --- MRL: prefix sizes to evaluate ---
    eval_prefixes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])

    # --- Training ---
    lr:           float = 1e-3
    epochs:       int   = 2
    batch_size:   int   = 128
    patience:     int   = 5     # set to None to disable early stopping
    weight_decay: float = 1e-4

    # --- Experiment 7: L1 regularization strength ---
    l1_lambda: float = 0.05   # weight of L1 penalty on embedding activations

    # --- Reproducibility ---
    seed: int = 42

    # --- Experiment 5: Seed Stability ---
    data_seed:   int       = 42
    # model_seeds: List[int] = field(default_factory=lambda: [100, 200, 300])
    model_seeds: List[int] = field(default_factory=lambda: [100, 200])

    # --- Output ---
    experiment_name: str = "exp1_prefix_curve"

    def __post_init__(self):
        """Validate config values immediately after construction."""

        # All prefix sizes must fit within embed_dim
        for k in self.eval_prefixes:
            assert k <= self.embed_dim, (
                f"eval_prefix {k} exceeds embed_dim {self.embed_dim}"
            )

        # head_mode must be one of the two supported options
        assert self.head_mode in ("shared_head", "multi_head"), (
            f"head_mode must be 'shared_head' or 'multi_head', got '{self.head_mode}'"
        )

        # Fractions must be in (0, 1)
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < self.val_size  < 1, "val_size must be between 0 and 1"

        print("[config] ExpConfig validated successfully.")
        print(f"  dataset       : {self.dataset}")
        print(f"  embed_dim     : {self.embed_dim}")
        print(f"  head_mode     : {self.head_mode}")
        print(f"  eval_prefixes : {self.eval_prefixes}")
        print(f"  epochs        : {self.epochs}  |  lr: {self.lr}  |  batch: {self.batch_size}")
        print(f"  seed          : {self.seed}")
        print(f"  data_seed     : {self.data_seed}  |  model_seeds: {self.model_seeds}")


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    print("--- Testing default ExpConfig ---")
    cfg = ExpConfig()
    print()

    print("--- Testing custom ExpConfig ---")
    cfg2 = ExpConfig(
        dataset="digits",
        embed_dim=32,
        eval_prefixes=[1, 2, 4, 8, 16, 32],
        head_mode="multi_head",
        epochs=20,
    )
    print()

    print("--- Testing invalid config (should raise AssertionError) ---")
    try:
        bad = ExpConfig(embed_dim=8, eval_prefixes=[1, 2, 4, 16])  # 16 > 8
        print("  ERROR: should have raised AssertionError")
    except AssertionError as e:
        print(f"  Correctly caught: {e}")

    print("\nAll config checks passed.")
