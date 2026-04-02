"""
Script: config.py
-----------------
Schema and validation container for experiment configuration.

ExpConfig is a typed dataclass that shared modules (trainer.py, prefix_eval.py,
etc.) accept instead of long argument lists. It is NOT a source of defaults —
every experiment script defines its own CONFIG block at the top with all
parameters made explicit. ExpConfig is only instantiated inside each experiment's
main(), passing values from that script's CONFIG block.

Inputs:  All fields must be provided explicitly by the calling experiment.
         Only test_size and val_size carry stable defaults (never vary across runs).
Outputs: Validated ExpConfig instance

Usage:
    from config import ExpConfig
    cfg = ExpConfig(
        dataset="mnist", embed_dim=64, hidden_dim=256,
        head_mode="shared_head", eval_prefixes=[1,2,4,8,16,32,64],
        lr=1e-3, epochs=20, batch_size=128, patience=5,
        weight_decay=1e-4, l1_lambda=0.05, seed=42,
        data_seed=42, model_seeds=[100,200],
        experiment_name="exp1_prefix_curve",
    )
    python config.py   # prints a sample validated configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExpConfig:
    """
    Validated configuration container shared across all experiments.

    All fields must be set explicitly by the calling experiment script —
    there are no experiment-level defaults here (only test_size and val_size,
    which never vary). This ensures that reading a single experiment file
    gives a complete picture of what that run does.

    Required fields (no defaults — experiment must supply them):
        dataset, embed_dim, hidden_dim, head_mode, eval_prefixes,
        lr, epochs, batch_size, patience, weight_decay, seed,
        experiment_name

    Optional fields (None if not used by the experiment):
        l1_lambda   — L1 regularisation weight (exp7+)
        data_seed   — fixed data-split seed (exp5+)
        model_seeds — list of model-init seeds (exp5+)

    Stable infrastructure defaults (never touch these):
        test_size = 0.2
        val_size  = 0.1

    --- Data ---
    dataset     : 'mnist' | 'digits' | 'iris' | 'wine' | 'breast_cancer'
    test_size   : fraction of data held out for testing
    val_size    : fraction of TRAINING data used for validation

    --- Model ---
    embed_dim   : size of the learned embedding (output of encoder)
    hidden_dim  : size of the hidden layer inside the MLP encoder
    head_mode   : 'shared_head' (Mode A) or 'multi_head' (Mode B)

    --- MRL ---
    eval_prefixes : list of prefix sizes k; all must be <= embed_dim

    --- Training ---
    lr           : Adam learning rate
    epochs       : max training epochs
    batch_size   : mini-batch size
    patience     : early-stopping patience (epochs without val improvement)
    weight_decay : L2 regularisation on weights

    --- Regularisation ---
    l1_lambda : weight of L1 penalty on embedding activations (exp7+)

    --- Reproducibility ---
    seed        : master random seed for model init + training
    data_seed   : seed for the fixed data split (exp5+)
    model_seeds : seeds for independent training runs (exp5+)

    --- Output ---
    experiment_name : used to name the timestamped run output folder
    """

    # ------------------------------------------------------------------ #
    # Required — no defaults; every experiment must supply these          #
    # ------------------------------------------------------------------ #
    dataset:         str
    embed_dim:       int
    hidden_dim:      int
    head_mode:       str        # 'shared_head' or 'multi_head'
    eval_prefixes:   List[int]
    lr:              float
    epochs:          int
    batch_size:      int
    patience:        int
    weight_decay:    float
    seed:            int
    experiment_name: str

    # ------------------------------------------------------------------ #
    # Stable infrastructure — same in every experiment, safe to default   #
    # ------------------------------------------------------------------ #
    test_size: float = 0.2
    val_size:  float = 0.1

    # ------------------------------------------------------------------ #
    # Optional — only some experiments use these; default None / []       #
    # ------------------------------------------------------------------ #
    l1_lambda:   Optional[float]  = None   # L1 regularisation weight (exp7+)
    data_seed:   Optional[int]    = None   # fixed data-split seed (exp5+)
    model_seeds: List[int]        = field(default_factory=list)  # per-run seeds (exp5+)

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

        print("[config] ExpConfig validated.")
        print(f"  dataset       : {self.dataset}")
        print(f"  embed_dim     : {self.embed_dim}  |  hidden_dim : {self.hidden_dim}")
        print(f"  head_mode     : {self.head_mode}")
        print(f"  eval_prefixes : {self.eval_prefixes}")
        print(f"  epochs        : {self.epochs}  |  lr: {self.lr}  |  batch: {self.batch_size}")
        print(f"  patience      : {self.patience}  |  weight_decay: {self.weight_decay}")
        print(f"  seed          : {self.seed}")
        if self.l1_lambda   is not None: print(f"  l1_lambda     : {self.l1_lambda}")
        if self.data_seed   is not None: print(f"  data_seed     : {self.data_seed}")
        if self.model_seeds:             print(f"  model_seeds   : {self.model_seeds}")


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    print("--- Testing full ExpConfig ---")
    cfg = ExpConfig(
        dataset="mnist",
        embed_dim=64,
        hidden_dim=256,
        head_mode="shared_head",
        eval_prefixes=[1, 2, 4, 8, 16, 32, 64],
        lr=1e-3,
        epochs=20,
        batch_size=128,
        patience=5,
        weight_decay=1e-4,
        seed=42,
        experiment_name="exp1_prefix_curve",
    )
    print()

    print("--- Testing with optional fields ---")
    cfg2 = ExpConfig(
        dataset="digits",
        embed_dim=16,
        hidden_dim=128,
        head_mode="shared_head",
        eval_prefixes=[1, 2, 4, 8, 16],
        lr=1e-3,
        epochs=5,
        batch_size=128,
        patience=3,
        weight_decay=1e-4,
        seed=42,
        experiment_name="exp5_seed_stability",
        l1_lambda=0.05,
        data_seed=42,
        model_seeds=[100, 200],
    )
    print()

    print("--- Testing invalid config (should raise AssertionError) ---")
    try:
        bad = ExpConfig(
            dataset="digits", embed_dim=8, hidden_dim=128,
            head_mode="shared_head", eval_prefixes=[1, 2, 4, 16],
            lr=1e-3, epochs=5, batch_size=128, patience=3,
            weight_decay=1e-4, seed=42, experiment_name="bad",
        )
        print("  ERROR: should have raised AssertionError")
    except AssertionError as e:
        print(f"  Correctly caught: {e}")

    print("\nAll config checks passed.")
