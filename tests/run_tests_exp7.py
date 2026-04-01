"""
Script: run_tests_exp7.py
--------------------------
Test runner for Experiment 7 — MRL vs FF vs L1.

Tests:
  1. Unit tests for shared module changes (L1RegLoss, build_loss "l1").
  2. Unit tests for new evaluation helpers (evaluate_1nn, train_ff_models).
  3. End-to-end smoke test: run the full experiment with --fast flag.

Usage:
    python run_tests_exp7.py           # unit tests + e2e smoke test
    python run_tests_exp7.py --fast    # unit tests only (skip e2e smoke)

Inputs:  None (uses internal test data / digits dataset)
Outputs: PASS / FAIL messages printed to stdout; non-zero exit on failure.
"""

import os

# Cap BLAS thread count before any numpy/scipy/sklearn imports to prevent
# deadlocks on macOS when PyTorch and sklearn share the same OpenBLAS pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import argparse
import subprocess
import numpy as np
import torch

# Absolute path to code/ — tests/ is one level below code/
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from config import ExpConfig
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss, L1RegLoss

from experiments.exp7_mrl_vs_ff import (
    evaluate_1nn,
    evaluate_prefix_1nn,
    train_ff_models,
    get_embeddings_np,
)


# ==============================================================================
# Unit tests
# ==============================================================================

def test_l1_reg_loss():
    """L1RegLoss produces a finite scalar, gradient flows back through it."""
    print("--- test_l1_reg_loss ---")

    cfg   = ExpConfig(dataset="digits", embed_dim=8,
                      eval_prefixes=[1, 2, 4, 8], epochs=1)
    batch = 16
    emb   = torch.randn(batch, cfg.embed_dim, requires_grad=True)
    labels = torch.randint(0, 10, (batch,))
    head  = build_head(cfg, n_classes=10)

    loss_fn = L1RegLoss(lambda_l1=0.05)
    loss    = loss_fn(emb, labels, head)

    assert loss.shape == torch.Size([]),     "Loss must be a scalar"
    assert torch.isfinite(loss).item(),      "Loss must be finite"
    assert loss.item() > 0,                  "Loss must be positive"

    # Gradient must flow back to the embedding
    loss.backward()
    assert emb.grad is not None,             "No gradient on embedding"
    assert torch.isfinite(emb.grad).all(),   "Gradient contains non-finite values"

    print(f"  loss={loss.item():.4f}  grad_norm={emb.grad.norm().item():.4f}")
    print("  PASSED\n")


def test_build_loss_l1():
    """build_loss('l1') returns an L1RegLoss with the correct lambda."""
    print("--- test_build_loss_l1 ---")

    cfg = ExpConfig(dataset="digits", embed_dim=8,
                    eval_prefixes=[1, 2, 4, 8], epochs=1, l1_lambda=0.1)
    loss_fn = build_loss(cfg, "l1")

    assert isinstance(loss_fn, L1RegLoss),         "Should return L1RegLoss"
    assert abs(loss_fn.lambda_l1 - 0.1) < 1e-6,   "lambda_l1 should match cfg"
    print(f"  lambda_l1={loss_fn.lambda_l1}")
    print("  PASSED\n")


def test_l1_vs_standard_sparsity():
    """
    L1 loss gradient should push embedding values toward zero more aggressively
    than standard CE loss. Verified via a single gradient step on a fixed batch —
    no full training required, runs in milliseconds.
    """
    print("--- test_l1_vs_standard_sparsity ---")

    torch.manual_seed(0)
    cfg  = ExpConfig(dataset="digits", embed_dim=8,
                     eval_prefixes=[1, 2, 4, 8], epochs=1, l1_lambda=1.0)
    batch, d = 32, 8

    # Use the same random embedding and labels for both models
    emb_init = torch.randn(batch, d)
    labels   = torch.randint(0, 10, (batch,))

    # --- Standard: one gradient step ---
    emb_std = emb_init.clone().detach().requires_grad_(True)
    head_std = build_head(cfg, n_classes=10)
    loss_std = build_loss(cfg, "standard")(emb_std, labels, head_std)
    loss_std.backward()
    grad_magnitude_std = emb_std.grad.abs().mean().item()

    # --- L1 (high lambda=1.0): one gradient step ---
    emb_l1 = emb_init.clone().detach().requires_grad_(True)
    head_l1 = build_head(cfg, n_classes=10)
    # Copy head weights so CE contribution is identical
    head_l1.load_state_dict(head_std.state_dict())
    loss_l1 = build_loss(cfg, "l1")(emb_l1, labels, head_l1)
    loss_l1.backward()
    grad_magnitude_l1 = emb_l1.grad.abs().mean().item()

    print(f"  Mean |grad|  standard={grad_magnitude_std:.4f}  l1={grad_magnitude_l1:.4f}")
    # L1 adds lambda * sign(emb) to the gradient — must be strictly larger
    assert grad_magnitude_l1 > grad_magnitude_std, (
        f"L1 gradient should be larger than standard "
        f"(l1={grad_magnitude_l1:.4f} vs std={grad_magnitude_std:.4f})"
    )
    print("  PASSED\n")


def test_evaluate_1nn_basic():
    """evaluate_1nn returns correct accuracy on trivially separable embeddings."""
    print("--- test_evaluate_1nn_basic ---")

    # Perfect separation: each class cluster far from others
    rng = np.random.default_rng(0)
    n_cls, n_per = 5, 20
    train_emb = np.vstack([
        rng.normal(loc=i * 10, scale=0.1, size=(n_per, 4))
        for i in range(n_cls)
    ]).astype(np.float32)
    y_train = np.repeat(np.arange(n_cls), n_per)

    test_emb = np.vstack([
        rng.normal(loc=i * 10, scale=0.1, size=(5, 4))
        for i in range(n_cls)
    ]).astype(np.float32)
    y_test = np.repeat(np.arange(n_cls), 5)

    acc = evaluate_1nn(train_emb, test_emb, y_train, y_test)
    assert acc == 1.0, f"Expected perfect accuracy, got {acc:.4f}"
    print(f"  accuracy={acc:.4f}  (expected 1.0)")
    print("  PASSED\n")


def test_evaluate_1nn_subsample():
    """evaluate_1nn subsampling does not crash and returns a valid accuracy."""
    print("--- test_evaluate_1nn_subsample ---")

    rng = np.random.default_rng(1)
    train_emb = rng.standard_normal((1000, 8)).astype(np.float32)
    test_emb  = rng.standard_normal((100,  8)).astype(np.float32)
    y_train   = rng.integers(0, 5, 1000).astype(np.int64)
    y_test    = rng.integers(0, 5, 100).astype(np.int64)

    acc = evaluate_1nn(train_emb, test_emb, y_train, y_test, max_db_samples=200, seed=42)
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"
    print(f"  accuracy={acc:.4f}")
    print("  PASSED\n")


def test_train_ff_models():
    """train_ff_models trains two FF models and saves the expected weight files."""
    print("--- test_train_ff_models ---")

    import tempfile
    cfg = ExpConfig(
        dataset="digits", embed_dim=16, hidden_dim=64,
        eval_prefixes=[1, 2, 4, 8, 16], epochs=2, patience=2, seed=0,
    )
    data = load_data(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        ff = train_ff_models(cfg, eval_prefixes=[1, 4], data=data, run_dir=tmp)

        # Both models should exist in the dict
        assert set(ff.keys()) == {1, 4}, f"Expected keys {{1, 4}}, got {set(ff.keys())}"

        for k in [1, 4]:
            enc, hd = ff[k]
            assert enc.embed_dim == k, f"FF-{k} encoder embed_dim should be {k}"
            # Weight files should have been saved
            assert os.path.isfile(os.path.join(tmp, f"ff_k{k}_encoder_best.pt"))
            assert os.path.isfile(os.path.join(tmp, f"ff_k{k}_head_best.pt"))
            # Forward pass should work
            enc.eval()
            hd.eval()
            with torch.no_grad():
                emb    = enc(data.X_test[:4])
                logits = hd(emb)
            assert emb.shape    == (4, k),              f"Wrong emb shape for k={k}"
            assert logits.shape == (4, data.n_classes), f"Wrong logit shape for k={k}"

    print("  PASSED\n")


def test_evaluate_prefix_1nn():
    """evaluate_prefix_1nn returns a dict with correct keys and valid accuracies."""
    print("--- test_evaluate_prefix_1nn ---")

    cfg = ExpConfig(
        dataset="digits", embed_dim=8,
        eval_prefixes=[1, 2, 4, 8], epochs=1, seed=42,
    )
    data    = load_data(cfg)
    encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)

    results = evaluate_prefix_1nn(
        encoder, data, cfg.eval_prefixes, "test_model",
        max_db_samples=200, seed=42,
    )

    assert set(results.keys()) == set(cfg.eval_prefixes), "Missing prefix keys"
    for k, acc in results.items():
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range for k={k}: {acc}"

    print(f"  results={results}")
    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_fast():
    """
    Run the full experiment with --fast flag and verify all output files exist.
    """
    print("--- test_e2e_fast (end-to-end smoke test) ---")
    print("  Running: python experiments/exp7_mrl_vs_ff.py --fast")
    print("  (This trains digits models for 5 epochs — takes ~1-2 min) ...")

    result = subprocess.run(
        [sys.executable, "experiments/exp7_mrl_vs_ff.py", "--fast"],
        capture_output=True, text=True,
        cwd=CODE_DIR,
    )

    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-2000:])
        print("  STDERR:", result.stderr[-2000:])
        raise AssertionError(f"exp7 --fast failed with return code {result.returncode}")

    # Find the most recently created run folder
    from utility import get_path
    results_dir = get_path("files/results")
    exp7_runs = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith("exprmnt_") and
        os.path.isdir(os.path.join(results_dir, d))
    ])
    assert exp7_runs, "No output folder found after e2e run"
    run_dir = os.path.join(results_dir, exp7_runs[-1])

    required_files = [
        "experiment_description.log",
        "training_curves.png",
        "linear_accuracy_curve.png",
        "1nn_accuracy_curve.png",
        "combined_comparison.png",
        "results_summary.txt",
        "runtime.txt",
        "code_snapshot",
    ]
    for fname in required_files:
        fpath = os.path.join(run_dir, fname)
        assert os.path.exists(fpath), f"Missing output: {fname}"
        print(f"  Found: {fname}")

    print(f"  Run folder: {run_dir}")
    print("  PASSED\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="run_tests_exp7.py")
    parser.add_argument(
        "--fast", action="store_true",
        help="Run unit tests only; skip the end-to-end smoke test.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("run_tests_exp7.py — Experiment 7 test suite")
    print("=" * 60)

    print("\nUnit tests")
    print("-" * 40)
    test_l1_reg_loss()
    test_build_loss_l1()
    test_l1_vs_standard_sparsity()
    test_evaluate_1nn_basic()
    test_evaluate_1nn_subsample()
    test_train_ff_models()
    test_evaluate_prefix_1nn()

    if not args.fast:
        print("\nEnd-to-end smoke test")
        print("-" * 40)
        test_e2e_fast()

    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
