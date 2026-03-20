"""
Script: run_tests_exp5.py
--------------------------
Test runner for Experiment 5: Seed Stability.

Runs two kinds of checks in order:
  1. Shared infrastructure modules (same as run_tests.py --fast) — ensures the
     modules exp5 depends on are healthy before running the experiment.
  2. Exp5-specific tests:
       - Unit test for compute_cross_seed_correlation (synthetic data, no GPU needed).
       - End-to-end smoke test of exp5_seed_stability.py --fast  (digits dataset,
         2 seeds, 3 epochs — runs in ~30s).

Usage (from code/):
    python run_tests_exp5.py          # all tests
    python run_tests_exp5.py --fast   # skip the end-to-end smoke test

Inputs:  None
Outputs: Console PASS/FAIL table. Exit code 0 if all pass, 1 if any fail.
"""

import subprocess
import sys
import time
import os
import argparse
import importlib
import numpy as np


# ==============================================================================
# Module list (shared infra — same as run_tests.py, fast subset)
# ==============================================================================

# (short_name, script_path_relative_to_code_dir)
INFRA_MODULES = [
    ("utility",     "utility.py"),
    ("config",      "config.py"),
    ("encoder",     "models/encoder.py"),
    ("heads",       "models/heads.py"),
    ("mat_loss",    "losses/mat_loss.py"),
    ("prefix_eval", "evaluation/prefix_eval.py"),
]


# ==============================================================================
# Helpers (reused from run_tests.py pattern)
# ==============================================================================

def run_subprocess(script_path: str, code_dir: str, extra_args: list = None) -> tuple:
    """
    Run a Python script as a subprocess and return (passed, elapsed, stdout, stderr).

    Args:
        script_path (str)      : Path to .py file relative to code_dir.
        code_dir    (str)      : Absolute path to code/ directory.
        extra_args  (list)     : Additional CLI arguments to pass to the script.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    abs_path = os.path.join(code_dir, script_path)
    cmd      = [sys.executable, abs_path] + (extra_args or [])

    env             = os.environ.copy()
    env["PYTHONPATH"] = code_dir + os.pathsep + env.get("PYTHONPATH", "")

    start  = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=code_dir, env=env,
    )
    elapsed = time.time() - start
    return (result.returncode == 0), elapsed, result.stdout, result.stderr


def print_result(name: str, passed: bool, elapsed: float,
                 stdout: str, stderr: str, idx: int, n_total: int,
                 suite_start: float, col_w: int = 20, bar_w: int = 20):
    """Print a single test result line with progress bar, then output on failure."""
    post_elapsed = time.time() - suite_start
    filled  = int(bar_w * (idx + 1) / n_total)
    bar     = "=" * filled + (">" if filled < bar_w else "") + " " * (bar_w - filled)
    bar_str = f"[{bar}] {idx+1}/{n_total}  {post_elapsed:.1f}s"
    status  = "PASS" if passed else "FAIL"

    print(f"  {name:<{col_w}}  {status:<8}  {elapsed:>5.1f}s  {bar_str}")

    if not passed:
        print()
        for line in (stdout or "").strip().splitlines():
            print(f"    [stdout] {line}")
        for line in (stderr or "").strip().splitlines()[-30:]:
            print(f"    [stderr] {line}")
        print()


# ==============================================================================
# Exp5-specific unit test
# ==============================================================================

def test_cross_seed_correlation(code_dir: str) -> tuple:
    """
    Unit test for compute_cross_seed_correlation using synthetic embeddings.

    Verifies:
      - A perfectly stable model (same embeddings each seed) yields diagonal=1.
      - A random/independent model yields low diagonal.

    Args:
        code_dir (str): Absolute path to code/ — used for sys.path setup.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    start = time.time()
    output_lines = []

    try:
        # Import the function under test
        sys.path.insert(0, code_dir)
        from experiments.exp5_seed_stability import compute_cross_seed_correlation

        np.random.seed(0)
        N, D = 200, 16

        # --- Case 1: identical embeddings across seeds → diagonal should be 1.0 ---
        base_emb = np.random.randn(N, D)
        identical = {100: base_emb.copy(), 200: base_emb.copy(), 300: base_emb.copy()}
        res_identical = compute_cross_seed_correlation(identical)
        diag_identical = res_identical["mean_diag_corr"]
        output_lines.append(f"Identical embeddings — mean |diag corr| = {diag_identical:.4f}")
        assert diag_identical > 0.99, (
            f"Identical embeddings should give diag~1.0, got {diag_identical:.4f}"
        )
        output_lines.append("  PASSED: identical embeddings → diag ≈ 1.0")

        # --- Case 2: independent random embeddings → diagonal should be ~0 ---
        random_seeds = {
            100: np.random.randn(N, D),
            200: np.random.randn(N, D),
            300: np.random.randn(N, D),
        }
        res_random = compute_cross_seed_correlation(random_seeds)
        diag_random = res_random["mean_diag_corr"]
        output_lines.append(f"Random embeddings   — mean |diag corr| = {diag_random:.4f}")
        assert diag_random < 0.3, (
            f"Random embeddings should give low diag, got {diag_random:.4f}"
        )
        output_lines.append("  PASSED: random embeddings → diag ≈ 0")

        # --- Case 3: check output dict structure ---
        assert "mean_corr_matrix" in res_identical
        assert "per_pair" in res_identical
        assert len(res_identical["per_pair"]) == 3  # C(3,2)=3 pairs
        output_lines.append("  PASSED: output dict structure correct")

        passed = True

    except Exception as e:
        output_lines.append(f"ERROR: {e}")
        import traceback
        output_lines.append(traceback.format_exc())
        passed = False

    elapsed = time.time() - start
    stdout  = "\n".join(output_lines)
    return passed, elapsed, stdout, ""


# ==============================================================================
# Main runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test runner for Experiment 5.")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip the end-to-end smoke test (only run unit tests and infra checks).",
    )
    args = parser.parse_args()

    code_dir    = os.path.dirname(os.path.abspath(__file__))
    suite_start = time.time()
    col_w       = 28
    results     = []   # list of (name, passed, elapsed)

    print()
    print("Experiment 5 — Seed Stability — Test Suite")
    print("=" * 70)
    print(f"  {'MODULE':<{col_w}}  {'STATUS':<8}  {'TIME':>6}  PROGRESS")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Part 1: Shared infra modules (fast subset — no MNIST, no training)
    # ------------------------------------------------------------------
    print(f"\n  [Shared infrastructure]\n")

    infra_total = len(INFRA_MODULES)
    if args.fast:
        # Exclude prefix_eval (it has a slow path internally — just keep core ones)
        infra_total = len(INFRA_MODULES)   # still run all fast ones

    for idx, (name, script) in enumerate(INFRA_MODULES):
        passed, elapsed, stdout, stderr = run_subprocess(script, code_dir)
        results.append((name, passed, elapsed))
        n_done = idx + 1
        print_result(name, passed, elapsed, stdout, stderr,
                     n_done - 1, len(INFRA_MODULES), suite_start, col_w)

    # ------------------------------------------------------------------
    # Part 2: Exp5-specific unit test (correlation function)
    # ------------------------------------------------------------------
    print(f"\n  [Exp5-specific unit tests]\n")

    passed, elapsed, stdout, stderr = test_cross_seed_correlation(code_dir)
    results.append(("cross_seed_corr", passed, elapsed))
    print_result("cross_seed_corr", passed, elapsed, stdout, stderr,
                 0, 1, suite_start, col_w)

    # ------------------------------------------------------------------
    # Part 3: End-to-end smoke tests (skipped with --fast)
    # ------------------------------------------------------------------
    if not args.fast:
        print(f"\n  [End-to-end smoke tests]\n")

        # Default mode (--fast only, no --low-dim)
        passed, elapsed, stdout, stderr = run_subprocess(
            "experiments/exp5_seed_stability.py", code_dir, extra_args=["--fast"],
        )
        results.append(("exp5_e2e_smoke", passed, elapsed))
        print_result("exp5_e2e_smoke", passed, elapsed, stdout, stderr,
                     0, 2, suite_start, col_w)

        # Low-dim mode (--fast --low-dim)
        passed, elapsed, stdout, stderr = run_subprocess(
            "experiments/exp5_seed_stability.py", code_dir,
            extra_args=["--fast", "--low-dim"],
        )
        results.append(("exp5_e2e_low_dim", passed, elapsed))
        print_result("exp5_e2e_low_dim", passed, elapsed, stdout, stderr,
                     1, 2, suite_start, col_w)
    else:
        print(f"\n  [End-to-end smoke tests: SKIPPED (--fast)]\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = len(results) - n_pass
    total  = sum(e for _, _, e in results)

    print("-" * 60)
    print(f"  {n_pass}/{len(results)} passed   {total:.1f}s total")

    if n_fail > 0:
        failed_names = [n for n, p, _ in results if not p]
        print(f"  FAILED: {', '.join(failed_names)}")
        print()
        sys.exit(1)
    else:
        print("  All tests passed.")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
