"""
Script: tests/run_tests_exp9.py
--------------------------------
Test runner for Experiment 9 — Dense Prefix Evaluation + Multi-Seed Best-k vs First-k.

Tests:
  1. test_dense_eval_prefixes     — eval_prefixes = [1..embed_dim] in fast config
  2. test_run_one_seed_smoke      — run_one_seed on tiny synthetic data, check outputs
  3. test_plot_prefix_accuracy    — creates prefix_accuracy.png without crashing
  4. test_plot_best_vs_first_k    — creates best_vs_first_k.png without crashing
  5. test_plot_gap_comparison     — creates gap_comparison.png for 2 synthetic seeds
  6. test_e2e_fast (slow)         — full --fast run, all output files present

Usage:
    python tests/run_tests_exp9.py           # all tests including e2e
    python tests/run_tests_exp9.py --fast    # unit tests only (skip e2e smoke)

Inputs:  None (uses internal synthetic data / digits dataset)
Outputs: PASS / FAIL messages to stdout; non-zero exit on failure.
"""

import os

# Cap BLAS threads before numpy/sklearn imports — prevents macOS deadlocks.
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
from experiments.exp9_dense_prefix import (
    plot_prefix_accuracy,
    plot_best_vs_first_k_dense,
    plot_gap_comparison,
    MODEL_NAMES,
    DATA_SEEDS,
    IMPORTANCE_METHODS,
)
from experiments.exp8_dim_importance import compute_importance_scores


# ==============================================================================
# Helpers
# ==============================================================================

def _make_synthetic_gap_results(embed_dim: int, eval_prefixes: list,
                                  n_train: int = 200, n_test: int = 80,
                                  n_classes: int = 3, seed: int = 0) -> dict:
    """
    Build a synthetic gap_results dict matching the shape expected by exp9 plots.

    Creates random embeddings for all 4 models, runs importance scoring, then
    runs compute_best_vs_first_k to populate the dict.

    Args:
        embed_dim     (int)      : Number of embedding dimensions.
        eval_prefixes (list[int]): k values.
        n_train       (int)      : Number of synthetic training samples.
        n_test        (int)      : Number of synthetic test samples.
        n_classes     (int)      : Number of classes.
        seed          (int)      : RNG seed.

    Returns:
        dict: {model_name: {curve_key: {k: float}}} for all MODEL_NAMES.
    """
    from experiments.exp8_dim_importance import (
        compute_importance_scores,
        compute_best_vs_first_k,
    )

    rng = np.random.default_rng(seed)
    y_train = (rng.integers(0, n_classes, n_train)).astype(np.int64)
    y_test  = (rng.integers(0, n_classes, n_test)).astype(np.int64)

    gap_results = {}
    for model_name in MODEL_NAMES:
        Z_train = rng.standard_normal((n_train, embed_dim)).astype(np.float32)
        Z_test  = rng.standard_normal((n_test,  embed_dim)).astype(np.float32)
        scores  = compute_importance_scores(
            Z_test=Z_test, Z_train=Z_train,
            y_train=y_train, y_test=y_test,
            max_probe_samples=50, seed=seed, model_tag=model_name,
        )
        gap_results[model_name] = compute_best_vs_first_k(
            Z_train=Z_train, Z_test=Z_test,
            y_train=y_train, y_test=y_test,
            importance_scores=scores,
            eval_prefixes=eval_prefixes,
            seed=seed, model_tag=model_name,
        )
    return gap_results


# ==============================================================================
# Unit tests
# ==============================================================================

def test_dense_eval_prefixes():
    """
    Fast config uses eval_prefixes = list(range(1, embed_dim+1)) — verify this.
    """
    print("--- test_dense_eval_prefixes ---")

    cfg = ExpConfig(
        dataset       = "digits",
        embed_dim     = 16,
        eval_prefixes = list(range(1, 17)),
        epochs        = 1,
    )
    expected = list(range(1, 17))
    assert cfg.eval_prefixes == expected, \
        f"Expected {expected}, got {cfg.eval_prefixes}"
    assert len(cfg.eval_prefixes) == 16, \
        f"Dense eval should have 16 prefixes, got {len(cfg.eval_prefixes)}"

    print(f"  eval_prefixes: {cfg.eval_prefixes}")
    print("  PASSED\n")


def test_plot_prefix_accuracy(tmp_dir: str):
    """
    plot_prefix_accuracy creates prefix_accuracy.png without crashing.

    Uses synthetic gap_results for 4 models and 8 dense prefix values.

    Args:
        tmp_dir (str): Temporary directory for output files.
    """
    print("--- test_plot_prefix_accuracy ---")

    embed_dim     = 8
    eval_prefixes = list(range(1, embed_dim + 1))
    gap_results   = _make_synthetic_gap_results(embed_dim, eval_prefixes, seed=1)

    plot_prefix_accuracy(gap_results, tmp_dir, eval_prefixes, seed=42)

    out_path = os.path.join(tmp_dir, "prefix_accuracy.png")
    assert os.path.isfile(out_path),          "prefix_accuracy.png not created"
    assert os.path.getsize(out_path) > 0,     "prefix_accuracy.png is empty"

    print(f"  File size: {os.path.getsize(out_path)} bytes")
    print("  PASSED\n")


def test_plot_best_vs_first_k(tmp_dir: str):
    """
    plot_best_vs_first_k_dense creates best_vs_first_k.png without crashing.

    Args:
        tmp_dir (str): Temporary directory for output files.
    """
    print("--- test_plot_best_vs_first_k_dense ---")

    embed_dim     = 8
    eval_prefixes = list(range(1, embed_dim + 1))
    gap_results   = _make_synthetic_gap_results(embed_dim, eval_prefixes, seed=2)

    plot_best_vs_first_k_dense(gap_results, tmp_dir, eval_prefixes, seed=42)

    out_path = os.path.join(tmp_dir, "best_vs_first_k.png")
    assert os.path.isfile(out_path),      "best_vs_first_k.png not created"
    assert os.path.getsize(out_path) > 0, "best_vs_first_k.png is empty"

    print(f"  File size: {os.path.getsize(out_path)} bytes")
    print("  PASSED\n")


def test_plot_gap_comparison(tmp_dir: str):
    """
    plot_gap_comparison creates gap_comparison.png for 2 synthetic seeds.

    Args:
        tmp_dir (str): Temporary directory for output files.
    """
    print("--- test_plot_gap_comparison ---")

    embed_dim     = 8
    eval_prefixes = list(range(1, embed_dim + 1))

    # Build all_seed_results for two seeds
    all_seed_results = {}
    for seed in [42, 123]:
        gap_results = _make_synthetic_gap_results(embed_dim, eval_prefixes, seed=seed)
        all_seed_results[seed] = {"gap_results": gap_results}

    plot_gap_comparison(all_seed_results, tmp_dir, eval_prefixes, data_seeds=[42, 123])

    out_path = os.path.join(tmp_dir, "gap_comparison.png")
    assert os.path.isfile(out_path),      "gap_comparison.png not created"
    assert os.path.getsize(out_path) > 0, "gap_comparison.png is empty"

    print(f"  File size: {os.path.getsize(out_path)} bytes")
    print("  PASSED\n")


def test_importance_methods_constant():
    """
    IMPORTANCE_METHODS from exp8 is the expected triple.
    This guards against accidental changes in the imported constant.
    """
    print("--- test_importance_methods_constant ---")

    expected = ["mean_abs", "variance", "probe_acc"]
    assert IMPORTANCE_METHODS == expected, \
        f"Expected {expected}, got {IMPORTANCE_METHODS}"

    print(f"  IMPORTANCE_METHODS = {IMPORTANCE_METHODS}")
    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_fast():
    """
    Full end-to-end smoke test using --fast flag.

    Runs: python experiments/exp9_dense_prefix.py --fast
    Expected: exits 0, creates all required output files.
    """
    print("--- test_e2e_fast (end-to-end smoke test) ---")
    print("  Running: python experiments/exp9_dense_prefix.py --fast")
    print("  (trains digits models for 5 epochs × 1 seed) ...")

    result = subprocess.run(
        [sys.executable, "experiments/exp9_dense_prefix.py", "--fast"],
        capture_output=True, text=True,
        cwd=CODE_DIR,
    )

    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-3000:])
        print("  STDERR:", result.stderr[-3000:])
        raise AssertionError(
            f"exp9 --fast exited with code {result.returncode}"
        )

    # Find the most recently created run folder
    from utility import get_path
    import glob as _glob
    results_dir = get_path("files/results")
    run_dirs    = sorted(_glob.glob(os.path.join(results_dir, "exprmnt_*")))
    assert run_dirs, "No run directory was created"
    latest = run_dirs[-1]

    # Check mandatory root-level outputs
    required_root = [
        "experiment_description.log",
        "training_curves.png",
        "results_summary.txt",
        "runtime.txt",
    ]
    for fname in required_root:
        path = os.path.join(latest, fname)
        assert os.path.isfile(path),          f"Missing root output: {fname}"
        assert os.path.getsize(path) > 0,     f"Root output is empty: {fname}"

    # Check per-seed subfolder outputs (fast mode uses seed 42 only)
    seed_dir = os.path.join(latest, "seed_42")
    assert os.path.isdir(seed_dir), "seed_42/ subdirectory not created"

    required_seed = [
        "prefix_accuracy.png",
        "best_vs_first_k.png",
        "method_agreement.png",
    ]
    for fname in required_seed:
        path = os.path.join(seed_dir, fname)
        assert os.path.isfile(path),      f"Missing seed output: seed_42/{fname}"
        assert os.path.getsize(path) > 0, f"Seed output is empty: seed_42/{fname}"

    print(f"  Output directory: {latest}")
    print(f"  All required files present.")
    print("  PASSED\n")


# ==============================================================================
# Runner
# ==============================================================================

def main():
    """Run all unit tests and optionally the end-to-end smoke test."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast", action="store_true",
        help="Run unit tests only — skip the slow end-to-end smoke test.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("run_tests_exp9.py — Experiment 9 test suite")
    print("=" * 60 + "\n")

    # Shared temp dir for tests that write files
    tmp_dir = tempfile.mkdtemp(prefix="exp9_test_")

    # --- Unit tests (always run) ---
    print("Unit tests\n" + "-" * 40)
    test_dense_eval_prefixes()
    test_importance_methods_constant()
    test_plot_prefix_accuracy(tmp_dir)
    test_plot_best_vs_first_k(tmp_dir)
    test_plot_gap_comparison(tmp_dir)

    if args.fast:
        print("=" * 60)
        print("All unit tests PASSED  (e2e smoke test skipped with --fast)")
        print("=" * 60)
        return

    # --- End-to-end smoke test ---
    print("End-to-end smoke test\n" + "-" * 40)
    test_e2e_fast()

    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
