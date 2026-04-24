"""
Script: tests/run_tests_exp14.py
---------------------------------
Test runner for Experiment 14 — Two-Evaluation Comparison.

Tests:
  1. test_eval2_linear_sweep       — Eval 2 linear-acc produces correct shape/values
  2. test_cd34_config_constants    — CD34 CONFIG block is syntactically valid
  3. test_e2e_mnist_fast (slow)    — `exp14 --fast` smoke run on digits; checks outputs
  4. test_e2e_cd34_fast  (slow)    — `exp14 --dataset cd34 --fast` smoke run;
                                     skipped if h5ad file not present

Usage:
    python tests/run_tests_exp14.py           # unit tests + e2e smokes
    python tests/run_tests_exp14.py --fast    # unit tests only (skip e2e)

Conda environment: mrl_env
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import argparse
import subprocess

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)


# ==============================================================================
# Test 1: Eval 2 linear sweep on synthetic data
# ==============================================================================

def test_eval2_linear_sweep():
    """eval2_linear_sweep: shape + endpoint-accuracy sanity check."""
    print("--- test_eval2_linear_sweep ---")

    from experiments.exp14_two_eval_compare import eval2_linear_sweep

    rng = np.random.default_rng(0)
    n_test, d, n_classes = 200, 8, 4
    eval_prefixes = list(range(1, d + 1))

    # Construct a classifier W whose k-th dim is a one-hot indicator for class k
    # when z has value +5 on that dim (trivially separable at full d).
    W = np.zeros((n_classes, d), dtype=np.float64)
    for c in range(n_classes):
        W[c, c] = 1.0
    b = np.zeros(n_classes, dtype=np.float64)

    # Generate z with large positive value in the dim matching its label
    y_test = rng.integers(0, n_classes, size=n_test)
    Z_test = rng.normal(0, 0.1, size=(n_test, d))
    for i, y in enumerate(y_test):
        Z_test[i, y] = 5.0

    result = eval2_linear_sweep(Z_test, y_test, W, b, eval_prefixes, "test")

    assert set(result.keys()) == set(eval_prefixes), (
        f"Expected keys {eval_prefixes}, got {sorted(result.keys())}"
    )
    for k in eval_prefixes:
        assert 0.0 <= result[k] <= 1.0, f"k={k} acc out of [0,1]: {result[k]}"

    # At full d, accuracy should be essentially perfect on this setup
    assert result[d] >= 0.95, (
        f"Expected near-perfect accuracy at k={d}, got {result[d]:.3f}"
    )
    # At k=1, only class 0 can be predicted correctly → acc ≈ 1/n_classes
    # (allow slack because W[c>0, 0] == 0 means ties broken by argmax order)
    assert result[1] <= 0.55, (
        f"Expected low accuracy at k=1 with single-dim W, got {result[1]:.3f}"
    )

    print(f"  k=1 acc={result[1]:.3f}  k={d} acc={result[d]:.3f}")
    print("  PASSED\n")


# ==============================================================================
# Test 2: CD34 CONFIG constants are defined and sane
# ==============================================================================

def test_cd34_config_constants():
    """Importing exp14 exposes CD34 + MNIST CONFIG constants with expected types."""
    print("--- test_cd34_config_constants ---")

    import experiments.exp14_two_eval_compare as exp14

    assert exp14.DATASET in ("cd34", "mnist"), (
        f"DATASET must be 'cd34' or 'mnist', got {exp14.DATASET!r}"
    )
    assert isinstance(exp14.EMBED_DIMS, list) and len(exp14.EMBED_DIMS) >= 1, (
        f"EMBED_DIMS must be a non-empty list, got {exp14.EMBED_DIMS}"
    )
    assert all(isinstance(d, int) and d > 0 for d in exp14.EMBED_DIMS), (
        f"EMBED_DIMS must contain positive ints, got {exp14.EMBED_DIMS}"
    )
    assert isinstance(exp14.EPOCHS, int) and exp14.EPOCHS > 0, (
        f"EPOCHS must be a positive int, got {exp14.EPOCHS}"
    )
    assert isinstance(exp14.HIDDEN_DIM, int) and exp14.HIDDEN_DIM > 0, (
        f"HIDDEN_DIM must be a positive int, got {exp14.HIDDEN_DIM}"
    )
    assert isinstance(exp14.CD34_DATA_PATH, str) and exp14.CD34_DATA_PATH.endswith(".h5ad"), (
        f"CD34_DATA_PATH should point to a .h5ad file, got {exp14.CD34_DATA_PATH}"
    )
    assert isinstance(exp14.CD34_N_HVG, int) and exp14.CD34_N_HVG > 0, (
        f"CD34_N_HVG must be a positive int, got {exp14.CD34_N_HVG}"
    )
    assert isinstance(exp14.MNIST_EMBED_DIMS, list) and len(exp14.MNIST_EMBED_DIMS) >= 1, (
        f"MNIST_EMBED_DIMS must be a non-empty list, got {exp14.MNIST_EMBED_DIMS}"
    )
    assert isinstance(exp14.MNIST_EPOCHS, int) and exp14.MNIST_EPOCHS > 0, (
        f"MNIST_EPOCHS must be a positive int, got {exp14.MNIST_EPOCHS}"
    )

    print(f"  DATASET          = {exp14.DATASET}")
    print(f"  EMBED_DIMS       = {exp14.EMBED_DIMS}")
    print(f"  HIDDEN_DIM       = {exp14.HIDDEN_DIM}")
    print(f"  EPOCHS           = {exp14.EPOCHS}")
    print(f"  CD34_N_HVG       = {exp14.CD34_N_HVG}")
    print(f"  MNIST_EMBED_DIMS = {exp14.MNIST_EMBED_DIMS}")
    print("  PASSED\n")


# ==============================================================================
# Test 3: End-to-end MNIST --fast run
# ==============================================================================

def _find_run_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        if "All outputs saved to:" in line:
            return line.split("All outputs saved to:")[-1].strip()
    return ""


def test_e2e_mnist_fast():
    """`exp14 --fast` runs to completion on digits; required outputs exist."""
    print("--- test_e2e_mnist_fast ---")

    script = os.path.join(CODE_DIR, "experiments", "exp14_two_eval_compare.py")
    result = subprocess.run(
        [sys.executable, script, "--fast"],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        print(f"  FAILED\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}")
        raise AssertionError("exp14 --fast exited non-zero")

    run_dir = _find_run_dir(result.stdout)
    assert run_dir and os.path.isdir(run_dir), (
        f"Could not find run_dir in output:\n{result.stdout[-2000:]}"
    )

    # --fast uses embed_dim=8 → one embed_8 subdir expected
    subdirs = [d for d in os.listdir(run_dir)
               if d.startswith("embed_") and os.path.isdir(os.path.join(run_dir, d))]
    assert subdirs, f"No embed_* subdir in {run_dir}: {os.listdir(run_dir)}"

    required_prefixes = [
        "training_curves",
        "combined_comparison_eval1",
        "combined_comparison_eval2",
        "importance_scores",
        "method_agreement",
        "results_summary.txt",
    ]
    for sub in subdirs:
        files = os.listdir(os.path.join(run_dir, sub))
        missing = [p for p in required_prefixes
                   if not any(f.startswith(p) for f in files)]
        assert not missing, (
            f"Missing outputs in {sub}: {missing}\nFound: {files}"
        )

    # Root-level mandatory outputs
    root_files = os.listdir(run_dir)
    assert "runtime.txt"   in root_files, f"Missing runtime.txt in {run_dir}"
    assert "code_snapshot" in root_files, f"Missing code_snapshot in {run_dir}"

    print(f"  Run dir: {run_dir}")
    print(f"  Dim subdirs: {subdirs}")
    print("  PASSED\n")


# ==============================================================================
# Test 4: End-to-end CD34 --fast run (skips if data file missing)
# ==============================================================================

def test_e2e_cd34_fast():
    """`exp14 --dataset cd34 --fast` runs end-to-end; required outputs exist.

    Skipped if the CD34 h5ad file is not present.
    """
    print("--- test_e2e_cd34_fast ---")

    from experiments.exp14_two_eval_compare import CD34_DATA_PATH
    if not os.path.exists(CD34_DATA_PATH):
        print(f"  SKIPPED: CD34 data file not found: {CD34_DATA_PATH}")
        return

    script = os.path.join(CODE_DIR, "experiments", "exp14_two_eval_compare.py")
    result = subprocess.run(
        [sys.executable, script, "--dataset", "cd34", "--fast"],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        print(f"  FAILED\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}")
        raise AssertionError("exp14 --dataset cd34 --fast exited non-zero")

    run_dir = _find_run_dir(result.stdout)
    assert run_dir and os.path.isdir(run_dir), (
        f"Could not find run_dir in output:\n{result.stdout[-2000:]}"
    )

    subdirs = [d for d in os.listdir(run_dir)
               if d.startswith("embed_") and os.path.isdir(os.path.join(run_dir, d))]
    assert subdirs, f"No embed_* subdir in {run_dir}: {os.listdir(run_dir)}"

    required_prefixes = [
        "training_curves",
        "combined_comparison_eval1",
        "combined_comparison_eval2",
        "results_summary.txt",
    ]
    for sub in subdirs:
        files = os.listdir(os.path.join(run_dir, sub))
        missing = [p for p in required_prefixes
                   if not any(f.startswith(p) for f in files)]
        assert not missing, (
            f"Missing outputs in {sub}: {missing}\nFound: {files}"
        )

    print(f"  Run dir: {run_dir}")
    print(f"  Dim subdirs: {subdirs}")
    print("  PASSED\n")


# ==============================================================================
# Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Unit tests only; skip e2e smoke runs")
    args = parser.parse_args()

    tests = [
        test_eval2_linear_sweep,
        test_cd34_config_constants,
    ]
    if not args.fast:
        tests.append(test_e2e_mnist_fast)
        tests.append(test_e2e_cd34_fast)

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}\n")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
