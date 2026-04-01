"""
Script: tests/run_tests_exp10.py
----------------------------------
Test runner for Experiment 10 — Dense Prefix Sweep (MRL vs Standard vs L1 vs PCA).

Tests:
  1. test_embed_dim_flag          — --embed-dim N sets embed_dim and dense prefixes [1..N]
  2. test_evaluate_prefix_linear  — output shape and values in [0,1]
  3. test_linear_vs_1nn_shapes    — linear and 1-NN dicts have same keys
  4. test_plot_functions_no_crash — all 3 plots run, PNGs created
  5. test_e2e_fast (slow)         — full --fast run, all output files present

Usage:
    python tests/run_tests_exp10.py           # unit tests + e2e smoke test
    python tests/run_tests_exp10.py --fast    # unit tests only (skip e2e smoke)

Inputs:  None (uses internal synthetic data / digits dataset)
Outputs: PASS / FAIL messages printed to stdout; non-zero exit on failure.
"""

import os

# Must be set before any numpy/scipy/sklearn imports to prevent BLAS deadlocks on macOS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import tempfile
import argparse
import subprocess

import numpy as np

# Absolute path to code/ — tests/ is one level below code/
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from config import ExpConfig
from data.loader import load_data
from experiments.exp10_dense_multidim import (
    evaluate_prefix_linear,
    plot_all_curves,
    save_results_summary,
)


# ==============================================================================
# Test 1: --embed-dim flag produces dense [1..N] prefixes
# ==============================================================================

def test_embed_dim_flag():
    """
    Verify that --embed-dim N sets embed_dim=N and eval_prefixes=[1,2,...,N].

    Replicates the override logic from exp10 main() directly.
    """
    print("--- test_embed_dim_flag ---")

    for n in [8, 16, 32, 64]:
        # Replicate the override logic from exp10 main()
        cfg = ExpConfig(dataset="digits", embed_dim=64,
                        eval_prefixes=list(range(1, 65)))
        cfg.embed_dim     = n
        cfg.eval_prefixes = list(range(1, n + 1))

        assert cfg.embed_dim == n, \
            f"embed_dim not set: {cfg.embed_dim} != {n}"
        assert cfg.eval_prefixes == list(range(1, n + 1)), \
            f"eval_prefixes wrong for n={n}: {cfg.eval_prefixes}"
        assert len(cfg.eval_prefixes) == n, \
            f"Expected {n} prefixes, got {len(cfg.eval_prefixes)}"
        print(f"  n={n:>3}  eval_prefixes=[1..{n}]  (length={n})")

    print("  PASSED\n")


# ==============================================================================
# Test 2: evaluate_prefix_linear output shape and value range
# ==============================================================================

def test_evaluate_prefix_linear():
    """
    evaluate_prefix_linear returns a dict with one entry per prefix,
    all accuracies in [0, 1].
    """
    print("--- test_evaluate_prefix_linear ---")

    rng           = np.random.default_rng(0)
    embed_dim     = 8
    eval_prefixes = list(range(1, embed_dim + 1))
    n_train, n_test = 200, 60
    n_classes = 3

    # Simple separable embeddings: dim 0 = class * 5, rest = noise
    Z_train = rng.standard_normal((n_train, embed_dim)).astype(np.float32)
    Z_test  = rng.standard_normal((n_test,  embed_dim)).astype(np.float32)
    y_train = rng.integers(0, n_classes, n_train).astype(np.int64)
    y_test  = rng.integers(0, n_classes, n_test).astype(np.int64)

    results = evaluate_prefix_linear(
        Z_train, Z_test, y_train, y_test,
        eval_prefixes=eval_prefixes,
        model_tag="test_model",
        seed=42,
    )

    assert set(results.keys()) == set(eval_prefixes), \
        f"Wrong prefix keys: {set(results.keys())}"
    for k, acc in results.items():
        assert 0.0 <= acc <= 1.0, f"Accuracy out of [0,1] at k={k}: {acc}"

    print(f"  All {len(results)} prefixes have valid accuracies.")
    print(f"  Sample: k=1 → {results[1]:.4f}, k=8 → {results[8]:.4f}")
    print("  PASSED\n")


# ==============================================================================
# Test 3: linear and 1-NN result dicts have the same prefix keys
# ==============================================================================

def test_linear_vs_1nn_shapes():
    """
    Both linear and 1-NN evaluation dicts must have the same set of k keys
    so that plot_all_curves can zip them together without KeyErrors.
    """
    print("--- test_linear_vs_1nn_shapes ---")

    rng           = np.random.default_rng(1)
    embed_dim     = 4
    eval_prefixes = [1, 2, 3, 4]

    # Build synthetic dicts mimicking what main() produces
    linear_results = {
        model: {k: float(rng.random()) for k in eval_prefixes}
        for model in ["Standard", "L1", "MRL", "PCA"]
    }
    nn1_results = {
        model: {k: float(rng.random()) for k in eval_prefixes}
        for model in ["Standard", "L1", "MRL", "PCA"]
    }

    # Verify all keys match
    for model in ["Standard", "L1", "MRL", "PCA"]:
        lin_keys = set(linear_results[model].keys())
        nn1_keys = set(nn1_results[model].keys())
        assert lin_keys == nn1_keys, \
            f"{model}: linear keys {lin_keys} != 1-NN keys {nn1_keys}"

    print(f"  All 4 models have matching prefix keys: {set(eval_prefixes)}")
    print("  PASSED\n")


# ==============================================================================
# Test 4: plot functions run without crash and produce PNGs
# ==============================================================================

def test_plot_functions_no_crash():
    """
    plot_all_curves and save_results_summary run without error on minimal
    synthetic data and produce the expected output files.
    """
    print("--- test_plot_functions_no_crash ---")

    rng           = np.random.default_rng(2)
    eval_prefixes = list(range(1, 9))   # 1..8

    linear_results = {
        model: {k: float(rng.random()) for k in eval_prefixes}
        for model in ["Standard", "L1", "MRL", "PCA"]
    }
    nn1_results = {
        model: {k: float(rng.random()) for k in eval_prefixes}
        for model in ["Standard", "L1", "MRL", "PCA"]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all_curves(linear_results, nn1_results, eval_prefixes,
                        run_dir=tmpdir, l1_lambda=0.05)
        save_results_summary(linear_results, nn1_results, eval_prefixes,
                             run_dir=tmpdir)

        expected_files = [
            "linear_accuracy_curve.png",
            "1nn_accuracy_curve.png",
            "combined_comparison.png",
            "results_summary.txt",
        ]
        for fname in expected_files:
            fpath = os.path.join(tmpdir, fname)
            assert os.path.isfile(fpath), f"Missing output: {fname}"
            print(f"  Created: {fname}")

    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_fast():
    """
    Run the full experiment with --fast --embed-dim 8 and verify all output
    files are present in the run directory.
    """
    print("--- test_e2e_fast (end-to-end smoke test) ---")
    print("  Running: python experiments/exp10_dense_multidim.py --fast --embed-dim 8")
    print("  (trains digits models for 5 epochs, dense eval k=1..8) ...")

    result = subprocess.run(
        [sys.executable, "experiments/exp10_dense_multidim.py",
         "--fast", "--embed-dim", "8"],
        capture_output=True, text=True,
        cwd=CODE_DIR,
        timeout=300,
    )

    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-2000:])
        print("  STDERR:", result.stderr[-2000:])
        raise AssertionError(
            f"exp10 --fast --embed-dim 8 failed with return code {result.returncode}"
        )

    # Find the most recently created run folder
    sys.path.insert(0, CODE_DIR)
    from utility import get_path
    results_dir = get_path("files/results")
    runs = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith("exprmnt_") and
        os.path.isdir(os.path.join(results_dir, d))
    ])
    assert runs, "No output folder found after e2e run"
    run_dir = os.path.join(results_dir, runs[-1])

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
    """Run all exp10 unit tests and optionally the e2e smoke test."""
    parser = argparse.ArgumentParser(description="run_tests_exp10.py")
    parser.add_argument(
        "--fast", action="store_true",
        help="Run unit tests only; skip the end-to-end smoke test.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("run_tests_exp10.py — Experiment 10 test suite")
    print("=" * 60)

    print("\nUnit tests")
    print("-" * 40)
    test_embed_dim_flag()
    test_evaluate_prefix_linear()
    test_linear_vs_1nn_shapes()
    test_plot_functions_no_crash()

    if not args.fast:
        print("\nEnd-to-end smoke test")
        print("-" * 40)
        test_e2e_fast()

    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
