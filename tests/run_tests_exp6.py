"""
Script: run_tests_exp6.py
--------------------------
Test runner for Experiment 6: Orthogonal Matryoshka Autoencoder ≈ PCA.

Runs two kinds of checks in order:
  1. Shared infrastructure modules (fast subset) — ensures modules exp6 depends
     on are healthy.
  2. Exp6-specific tests:
       - Unit test for LinearAutoencoder (forward pass, orthogonalize, prefix ops).
       - End-to-end smoke test of exp6_ortho_mat_ae.py --fast (skipped with --fast).

Usage (from code/):
    python run_tests_exp6.py          # all tests
    python run_tests_exp6.py --fast   # skip the end-to-end smoke test

Inputs:  None
Outputs: Console PASS/FAIL table. Exit code 0 if all pass, 1 if any fail.
"""

import os

# Cap BLAS/OMP threads before numpy/torch imports — prevents macOS deadlocks.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import subprocess
import sys
import time
import argparse
import importlib
import numpy as np

# Absolute path to code/ — tests/ is one level below code/
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ==============================================================================
# Module list (shared infra — fast subset)
# ==============================================================================

INFRA_MODULES = [
    ("utility",    "utility.py"),
    ("config",     "config.py"),
    ("linear_ae",  "models/linear_ae.py"),
]


# ==============================================================================
# Helpers (same pattern as run_tests_exp5.py)
# ==============================================================================

def run_subprocess(script_path: str, code_dir: str, extra_args: list = None) -> tuple:
    """
    Run a Python script as a subprocess and return (passed, elapsed, stdout, stderr).

    Args:
        script_path (str)  : Path to .py file relative to code_dir.
        code_dir    (str)  : Absolute path to code/ directory.
        extra_args  (list) : Additional CLI arguments.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    abs_path = os.path.join(code_dir, script_path)
    cmd      = [sys.executable, abs_path] + (extra_args or [])

    env               = os.environ.copy()
    env["PYTHONPATH"] = code_dir + os.pathsep + env.get("PYTHONPATH", "")
    # Prevent macOS OMP/MKL thread-pool deadlock when torch subprocesses are
    # spawned in rapid sequence.  Single-threaded is fine for unit tests.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=code_dir, env=env,
            timeout=120,   # never hang forever
        )
        passed  = (result.returncode == 0)
        stdout  = result.stdout
        stderr  = result.stderr
    except subprocess.TimeoutExpired:
        passed  = False
        stdout  = ""
        stderr  = "TIMEOUT: subprocess did not complete within 120 s"
    elapsed = time.time() - start
    return passed, elapsed, stdout, stderr


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
# Exp6-specific unit test
# ==============================================================================

def test_linear_ae(code_dir: str) -> tuple:
    """
    Unit test for LinearAutoencoder — runs as a subprocess to avoid macOS
    torch-init hangs and in-process segfaults (recurring issue on this machine).

    Delegates to tests/helper_linear_ae.py which contains the actual assertions.

    Args:
        code_dir (str): Absolute path to code/ — passed to the helper script.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    helper = os.path.join(os.path.dirname(__file__), "helper_linear_ae.py")
    return run_subprocess(helper, code_dir, extra_args=[code_dir])


# ==============================================================================
# MNIST loader unit test
# ==============================================================================

def test_mnist_loader(code_dir: str) -> tuple:
    """
    Regression test for the MNIST loader segfault fix — runs as a subprocess
    to avoid macOS torch-init hangs and in-process segfaults.

    Delegates to tests/helper_mnist_loader.py which contains the actual assertions.

    Args:
        code_dir (str): Absolute path to code/ — passed to the helper script.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    helper = os.path.join(os.path.dirname(__file__), "helper_mnist_loader.py")
    return run_subprocess(helper, code_dir, extra_args=[code_dir])


# ==============================================================================
# Main runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test runner for Experiment 6.")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip the end-to-end smoke test (only run unit tests and infra checks).",
    )
    args = parser.parse_args()

    code_dir    = CODE_DIR
    suite_start = time.time()
    col_w       = 28
    results     = []

    print()
    print("Experiment 6 — Ortho Mat AE ≈ PCA — Test Suite")
    print("=" * 70)
    print(f"  {'MODULE':<{col_w}}  {'STATUS':<8}  {'TIME':>6}  PROGRESS")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Part 1: Shared infra modules
    # ------------------------------------------------------------------
    print(f"\n  [Shared infrastructure]\n")

    for idx, (name, script) in enumerate(INFRA_MODULES):
        passed, elapsed, stdout, stderr = run_subprocess(script, code_dir)
        results.append((name, passed, elapsed))
        print_result(name, passed, elapsed, stdout, stderr,
                     idx, len(INFRA_MODULES), suite_start, col_w)

    # ------------------------------------------------------------------
    # Part 2: Exp6-specific unit test (LinearAutoencoder)
    # ------------------------------------------------------------------
    print(f"\n  [Exp6-specific unit tests]\n")

    passed, elapsed, stdout, stderr = test_linear_ae(code_dir)
    results.append(("linear_ae_unit", passed, elapsed))
    print_result("linear_ae_unit", passed, elapsed, stdout, stderr,
                 0, 2, suite_start, col_w)

    # MNIST loader regression test — verifies .numpy() path doesn't segfault
    passed, elapsed, stdout, stderr = test_mnist_loader(code_dir)
    results.append(("mnist_loader", passed, elapsed))
    print_result("mnist_loader", passed, elapsed, stdout, stderr,
                 1, 2, suite_start, col_w)

    # ------------------------------------------------------------------
    # Part 3: End-to-end smoke test (skipped with --fast)
    # ------------------------------------------------------------------
    if not args.fast:
        print(f"\n  [End-to-end smoke test]\n")
        passed, elapsed, stdout, stderr = run_subprocess(
            "experiments/exp6_ortho_mat_ae.py", code_dir, extra_args=["--fast"],
        )
        results.append(("exp6_e2e_smoke", passed, elapsed))
        print_result("exp6_e2e_smoke", passed, elapsed, stdout, stderr,
                     0, 1, suite_start, col_w)
    else:
        print(f"\n  [End-to-end smoke test: SKIPPED (--fast)]\n")

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
