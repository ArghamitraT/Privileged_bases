"""
Script: tests/run_tests_exp12.py
----------------------------------
Test runner for Experiment 12 — VectorLearnedPrefixLp.

Tests:
  1. test_vector_learned_p_loss    — forward, backward, p shape, per-dim gradients
  2. test_evaluate_prefix_linear   — output shape and values in [0,1]
  3. test_p_trajectory_plots       — all 4 p plots produce PNGs without error
  4. test_accuracy_plots_no_crash  — plot_all_curves + save_results_summary produce files
  5. test_e2e_fast (slow)          — full --fast run, all output files present

Usage:
    python tests/run_tests_exp12.py           # all unit tests + e2e smoke test
    python tests/run_tests_exp12.py --fast    # unit tests only (skip e2e smoke)

Inputs:  None (uses internal synthetic data)
Outputs: PASS / FAIL messages printed to stdout; non-zero exit on failure.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import tempfile
import argparse
import subprocess

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from experiments.exp12_vector_learned_p import (
    evaluate_prefix_linear,
    plot_scalar_p_trajectory,
    plot_scalar_p_and_val_acc,
    plot_vector_p_trajectory,
    plot_vector_p_and_val_acc,
    plot_all_curves,
    save_results_summary,
)


# ==============================================================================
# Test 1: VectorLearnedPrefixLpLoss — shape, range, per-dim gradients
# ==============================================================================

def test_vector_learned_p_loss():
    """Run in subprocess to keep torch out of the test runner process."""
    print("--- test_vector_learned_p_loss ---")

    script = os.path.join(CODE_DIR, "tests", "helper_exp12_loss.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  FAILED\n{result.stderr}")
        raise AssertionError("helper_exp12_loss.py exited non-zero")

    output = result.stdout
    assert "p_shape: OK"   in output, f"p_shape check missing:\n{output}"
    assert "forward: OK"   in output, f"forward check missing:\n{output}"
    assert "backward: OK"  in output, f"backward check missing:\n{output}"

    print("  All vector loss checks passed.")
    print("  PASSED\n")


# ==============================================================================
# Test 2: evaluate_prefix_linear output shape and value range
# ==============================================================================

def test_evaluate_prefix_linear():
    print("--- test_evaluate_prefix_linear ---")

    rng           = np.random.default_rng(0)
    embed_dim     = 8
    eval_prefixes = list(range(1, embed_dim + 1))
    n_train, n_test = 200, 60

    Z_train = rng.standard_normal((n_train, embed_dim)).astype(np.float32)
    Z_test  = rng.standard_normal((n_test,  embed_dim)).astype(np.float32)
    y_train = rng.integers(0, 3, n_train).astype(np.int64)
    y_test  = rng.integers(0, 3, n_test).astype(np.int64)

    results = evaluate_prefix_linear(
        Z_train, Z_test, y_train, y_test,
        eval_prefixes=eval_prefixes, model_tag="test", seed=42,
    )

    assert set(results.keys()) == set(eval_prefixes)
    for k, acc in results.items():
        assert 0.0 <= acc <= 1.0, f"Accuracy out of [0,1] at k={k}: {acc}"

    print(f"  All {len(results)} prefixes have valid accuracies.")
    print("  PASSED\n")


# ==============================================================================
# Test 3: all 4 p plots produce PNGs without error
# ==============================================================================

def test_p_trajectory_plots():
    print("--- test_p_trajectory_plots ---")

    rng      = np.random.default_rng(1)
    n_epochs = 5
    embed_dim = 8

    # Scalar trajectory: list of 0-d arrays
    scalar_traj = [np.array(1.69 + 0.1 * i + float(rng.standard_normal()) * 0.02)
                   for i in range(n_epochs)]
    # Vector trajectory: list of (embed_dim,) arrays
    vector_traj = [np.array([1.69 + 0.05 * i + float(rng.standard_normal()) * 0.02
                              for _ in range(embed_dim)])
                   for i in range(n_epochs)]
    val_accs = [0.5 + 0.05 * i for i in range(n_epochs)]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_scalar_p_trajectory(scalar_traj, run_dir=tmpdir, fig_stamp="")
        plot_scalar_p_and_val_acc(scalar_traj, val_accs, run_dir=tmpdir, fig_stamp="")
        plot_vector_p_trajectory(vector_traj, embed_dim=embed_dim,
                                  run_dir=tmpdir, fig_stamp="")
        plot_vector_p_and_val_acc(vector_traj, val_accs, run_dir=tmpdir, fig_stamp="")

        for fname in ["scalar_p_trajectory.png", "scalar_p_and_val_acc.png",
                      "vector_p_trajectory.png", "vector_p_and_val_acc.png"]:
            assert os.path.isfile(os.path.join(tmpdir, fname)), f"Missing: {fname}"
            print(f"  Created: {fname}")

    print("  PASSED\n")


# ==============================================================================
# Test 4: accuracy plots and results summary produce expected files
# ==============================================================================

def test_accuracy_plots_no_crash():
    print("--- test_accuracy_plots_no_crash ---")

    rng           = np.random.default_rng(2)
    eval_prefixes = list(range(1, 9))
    model_names   = ["MRL", "ScalarLearnedPrefixLp (rev)", "VectorLearnedPrefixLp (rev)"]

    linear_results = {m: {k: float(rng.random()) for k in eval_prefixes}
                      for m in model_names}
    nn1_results    = {m: {k: float(rng.random()) for k in eval_prefixes}
                      for m in model_names}

    scalar_traj = [np.array(1.7 + i * 0.1) for i in range(5)]
    vector_traj = [np.ones(8) * (1.7 + i * 0.1) for i in range(5)]
    all_agreement = {
        m: {("mean_abs", "variance"): 0.9,
            ("mean_abs", "probe_acc"): 0.8,
            ("variance", "probe_acc"): 0.85}
        for m in model_names
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all_curves(linear_results, nn1_results, eval_prefixes,
                        run_dir=tmpdir, fig_stamp="")
        save_results_summary(linear_results, nn1_results, eval_prefixes,
                             scalar_traj, vector_traj, all_agreement, run_dir=tmpdir)

        for fname in ["linear_accuracy_curve.png", "1nn_accuracy_curve.png",
                      "combined_comparison.png", "results_summary.txt"]:
            assert os.path.isfile(os.path.join(tmpdir, fname)), f"Missing: {fname}"
            print(f"  Created: {fname}")

        with open(os.path.join(tmpdir, "results_summary.txt")) as f:
            content = f.read()
        assert "SCALAR LEARNED P SUMMARY" in content
        assert "VECTOR LEARNED P SUMMARY" in content
        assert "METHOD AGREEMENT"         in content
        print("  results_summary.txt contains all expected sections.")

    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_fast():
    print("--- test_e2e_fast (end-to-end smoke test) ---")
    print("  Running: python experiments/exp12_vector_learned_p.py --fast")
    print("  This may take ~2-3 minutes ...\n")

    exp_script = os.path.join(CODE_DIR, "experiments", "exp12_vector_learned_p.py")
    result = subprocess.run(
        [sys.executable, exp_script, "--fast"],
        capture_output=True, text=True, timeout=400,
        cwd=CODE_DIR,
    )

    if result.returncode != 0:
        print("  FAILED — non-zero exit code")
        print("  STDOUT:", result.stdout[-2000:])
        print("  STDERR:", result.stderr[-2000:])
        raise AssertionError("exp12 --fast exited with non-zero code")

    run_dir = None
    for line in result.stdout.splitlines():
        if "Outputs will be saved to:" in line:
            run_dir = line.split("Outputs will be saved to:")[-1].strip()
            break

    assert run_dir and os.path.isdir(run_dir), f"run_dir missing: {run_dir}"
    print(f"  run_dir: {run_dir}")

    for fname in ["experiment_description.log", "results_summary.txt", "runtime.txt"]:
        assert os.path.isfile(os.path.join(run_dir, fname)), f"Missing: {fname}"
        print(f"  Found: {fname}")

    import glob
    png_prefixes = [
        "training_curves", "linear_accuracy_curve", "1nn_accuracy_curve",
        "combined_comparison", "scalar_p_trajectory", "scalar_p_and_val_acc",
        "vector_p_trajectory", "vector_p_and_val_acc",
        "importance_scores", "method_agreement",
    ]
    for prefix in png_prefixes:
        matches = glob.glob(os.path.join(run_dir, f"{prefix}*.png"))
        assert len(matches) >= 1, f"Missing PNG with prefix '{prefix}'"
        print(f"  Found: {os.path.basename(matches[0])}")

    with open(os.path.join(run_dir, "results_summary.txt")) as f:
        content = f.read()
    assert "SCALAR LEARNED P SUMMARY" in content
    assert "VECTOR LEARNED P SUMMARY" in content
    assert "Vector final p" in result.stdout

    print("  PASSED\n")


# ==============================================================================
# Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip the slow e2e smoke test")
    args = parser.parse_args()

    tests = [
        test_vector_learned_p_loss,
        test_evaluate_prefix_linear,
        test_p_trajectory_plots,
        test_accuracy_plots_no_crash,
    ]
    if not args.fast:
        tests.append(test_e2e_fast)

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {test_fn.__name__}: {e}\n")
            failed += 1

    print("=" * 40)
    print(f"  {passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
