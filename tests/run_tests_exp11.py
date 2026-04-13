"""
Script: tests/run_tests_exp11.py
----------------------------------
Test runner for Experiment 11 — LearnedPrefixLp.

Tests:
  1. test_learned_prefix_lp_loss  — forward pass, backward pass, p tracking
  2. test_evaluate_prefix_linear  — output shape and values in [0,1]
  3. test_p_trajectory_plots      — both p plots run without error and produce PNGs
  4. test_accuracy_plots_no_crash — plot_all_curves runs and produces PNGs
  5. test_e2e_fast (slow)         — full --fast run, all output files present

Usage:
    python tests/run_tests_exp11.py           # all unit tests + e2e smoke test
    python tests/run_tests_exp11.py --fast    # unit tests only (skip e2e smoke)

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

from experiments.exp11_learned_prefix_lp import (
    evaluate_prefix_linear,
    plot_p_trajectory,
    plot_p_and_val_acc,
    plot_all_curves,
    save_results_summary,
)


# ==============================================================================
# Test 1: LearnedPrefixLpLoss forward, backward, and p tracking
# ==============================================================================

def test_learned_prefix_lp_loss():
    """
    Run in a subprocess to avoid importing torch in the test runner process
    (macOS hang — see Known Issues in CLAUDE.md).
    """
    print("--- test_learned_prefix_lp_loss ---")

    script = os.path.join(CODE_DIR, "tests", "helper_exp11_loss.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  FAILED\n{result.stderr}")
        raise AssertionError("helper_exp11_loss.py exited non-zero")

    output = result.stdout
    assert "forward: OK"   in output, f"Forward check missing from output:\n{output}"
    assert "backward: OK"  in output, f"Backward check missing from output:\n{output}"
    assert "p_property: OK" in output, f"p property check missing:\n{output}"
    assert "p_grad: OK"    in output, f"p_raw gradient check missing:\n{output}"

    print("  All loss checks passed.")
    print("  PASSED\n")


# ==============================================================================
# Test 2: evaluate_prefix_linear output shape and value range
# ==============================================================================

def test_evaluate_prefix_linear():
    """
    evaluate_prefix_linear returns one entry per prefix, all accuracies in [0,1].
    """
    print("--- test_evaluate_prefix_linear ---")

    rng           = np.random.default_rng(0)
    embed_dim     = 8
    eval_prefixes = list(range(1, embed_dim + 1))
    n_train, n_test = 200, 60
    n_classes = 3

    Z_train = rng.standard_normal((n_train, embed_dim)).astype(np.float32)
    Z_test  = rng.standard_normal((n_test,  embed_dim)).astype(np.float32)
    y_train = rng.integers(0, n_classes, n_train).astype(np.int64)
    y_test  = rng.integers(0, n_classes, n_test).astype(np.int64)

    results = evaluate_prefix_linear(
        Z_train, Z_test, y_train, y_test,
        eval_prefixes=eval_prefixes, model_tag="test", seed=42,
    )

    assert set(results.keys()) == set(eval_prefixes), \
        f"Wrong prefix keys: {set(results.keys())}"
    for k, acc in results.items():
        assert 0.0 <= acc <= 1.0, f"Accuracy out of [0,1] at k={k}: {acc}"

    print(f"  All {len(results)} prefixes have valid accuracies.")
    print(f"  Sample: k=1 → {results[1]:.4f}, k={embed_dim} → {results[embed_dim]:.4f}")
    print("  PASSED\n")


# ==============================================================================
# Test 3: p trajectory plots run without crash and produce PNGs
# ==============================================================================

def test_p_trajectory_plots():
    """
    plot_p_trajectory and plot_p_and_val_acc run on synthetic data and
    produce the expected PNG files.
    """
    print("--- test_p_trajectory_plots ---")

    rng          = np.random.default_rng(1)
    n_epochs     = 5
    p_trajectory = [1.69 + 0.3 * i + float(rng.standard_normal()) * 0.05
                    for i in range(n_epochs)]
    val_accs     = [0.5 + 0.05 * i + float(rng.standard_normal()) * 0.02
                    for i in range(n_epochs)]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_p_trajectory(p_trajectory, p_fixed=5, run_dir=tmpdir, fig_stamp="")
        plot_p_and_val_acc(p_trajectory, val_accs,  run_dir=tmpdir, fig_stamp="")

        for fname in ["p_trajectory.png", "p_and_val_acc.png"]:
            fpath = os.path.join(tmpdir, fname)
            assert os.path.isfile(fpath), f"Missing plot: {fname}"
            print(f"  Created: {fname}")

    print("  PASSED\n")


# ==============================================================================
# Test 4: accuracy plots run without crash and produce PNGs
# ==============================================================================

def test_accuracy_plots_no_crash():
    """
    plot_all_curves and save_results_summary run on synthetic data and
    produce the expected output files.
    """
    print("--- test_accuracy_plots_no_crash ---")

    rng           = np.random.default_rng(2)
    eval_prefixes = list(range(1, 9))
    model_names   = ["MRL", "PrefixLp (rev)", "LearnedPrefixLp (rev)"]

    linear_results = {
        m: {k: float(rng.random()) for k in eval_prefixes} for m in model_names
    }
    nn1_results = {
        m: {k: float(rng.random()) for k in eval_prefixes} for m in model_names
    }
    p_trajectory = [1.7 + i * 0.1 for i in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all_curves(linear_results, nn1_results, eval_prefixes,
                        run_dir=tmpdir, fig_stamp="")
        save_results_summary(linear_results, nn1_results, eval_prefixes,
                             run_dir=tmpdir, p_trajectory=p_trajectory, p_fixed=5)

        expected = [
            "linear_accuracy_curve.png",
            "1nn_accuracy_curve.png",
            "combined_comparison.png",
            "results_summary.txt",
        ]
        for fname in expected:
            fpath = os.path.join(tmpdir, fname)
            assert os.path.isfile(fpath), f"Missing output: {fname}"
            print(f"  Created: {fname}")

        # Verify results_summary contains learned p section
        with open(os.path.join(tmpdir, "results_summary.txt")) as f:
            content = f.read()
        assert "LEARNED P SUMMARY" in content, "Missing LEARNED P SUMMARY section"
        assert "p_trajectory" in content, "Missing p_trajectory in summary"
        print("  results_summary.txt contains LEARNED P SUMMARY section.")

    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_fast():
    """
    Run the full experiment with --fast flag and verify all output files are created.
    """
    print("--- test_e2e_fast (end-to-end smoke test) ---")
    print("  Running: python experiments/exp11_learned_prefix_lp.py --fast")
    print("  This may take ~1-2 minutes ...\n")

    exp_script = os.path.join(CODE_DIR, "experiments", "exp11_learned_prefix_lp.py")
    result = subprocess.run(
        [sys.executable, exp_script, "--fast"],
        capture_output=True, text=True, timeout=300,
        cwd=CODE_DIR,
    )

    if result.returncode != 0:
        print("  FAILED — non-zero exit code")
        print("  STDOUT:", result.stdout[-2000:])
        print("  STDERR:", result.stderr[-2000:])
        raise AssertionError("exp11 --fast exited with non-zero code")

    # Extract run_dir from stdout
    run_dir = None
    for line in result.stdout.splitlines():
        if "Outputs will be saved to:" in line:
            run_dir = line.split("Outputs will be saved to:")[-1].strip()
            break

    assert run_dir is not None, "Could not find run_dir in stdout"
    assert os.path.isdir(run_dir), f"run_dir does not exist: {run_dir}"
    print(f"  run_dir: {run_dir}")

    expected_files = [
        "experiment_description.log",
        "results_summary.txt",
        "runtime.txt",
    ]
    for fname in expected_files:
        fpath = os.path.join(run_dir, fname)
        assert os.path.isfile(fpath), f"Missing output: {fname}"
        print(f"  Found: {fname}")

    # Check for PNGs with timestamp suffix (glob by prefix)
    import glob
    png_prefixes = [
        "training_curves",
        "linear_accuracy_curve",
        "1nn_accuracy_curve",
        "combined_comparison",
        "p_trajectory",
        "p_and_val_acc",
    ]
    for prefix in png_prefixes:
        matches = glob.glob(os.path.join(run_dir, f"{prefix}*.png"))
        assert len(matches) >= 1, f"Missing PNG with prefix '{prefix}'"
        print(f"  Found: {os.path.basename(matches[0])}")

    # Check results_summary has learned p trajectory
    with open(os.path.join(run_dir, "results_summary.txt")) as f:
        content = f.read()
    assert "LEARNED P SUMMARY" in content
    assert "p_trajectory" in content
    print("  results_summary.txt contains LEARNED P SUMMARY.")

    # Check p_trajectory length in stdout
    assert "Final learned p" in result.stdout, "Missing 'Final learned p' in stdout"

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
        test_learned_prefix_lp_loss,
        test_evaluate_prefix_linear,
        test_p_trajectory_plots,
        test_accuracy_plots_no_crash,
    ]
    if not args.fast:
        tests.append(test_e2e_fast)

    passed = 0
    failed = 0
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
