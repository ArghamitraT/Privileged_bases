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

import subprocess
import sys
import time
import os
import argparse
import importlib
import numpy as np


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
# Exp6-specific unit test
# ==============================================================================

def test_linear_ae(code_dir: str) -> tuple:
    """
    Unit test for LinearAutoencoder.

    Verifies:
      - Forward pass produces correct shape.
      - encode_prefix and decode_prefix produce correct shapes.
      - orthogonalize() makes W W^T ≈ I.
      - Gradient flows through the model.

    Args:
        code_dir (str): Absolute path to code/ — used for sys.path setup.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    start = time.time()
    output_lines = []

    try:
        sys.path.insert(0, code_dir)
        import torch
        import torch.nn.functional as F
        from models.linear_ae import LinearAutoencoder

        np.random.seed(0)
        torch.manual_seed(0)

        input_dim  = 64
        embed_dim  = 10
        batch_size = 32

        model = LinearAutoencoder(input_dim=input_dim, embed_dim=embed_dim)
        x     = torch.randn(batch_size, input_dim)

        # --- Case 1: forward pass shape ---
        x_hat = model(x)
        assert x_hat.shape == (batch_size, input_dim), \
            f"Expected ({batch_size},{input_dim}), got {x_hat.shape}"
        output_lines.append(f"  PASSED: forward pass shape {x_hat.shape}")

        # --- Case 2: encode/decode shapes ---
        z = model.encode(x)
        assert z.shape == (batch_size, embed_dim)
        x_rec = model.decode(z)
        assert x_rec.shape == (batch_size, input_dim)
        output_lines.append(f"  PASSED: encode {x.shape} -> {z.shape}, decode -> {x_rec.shape}")

        # --- Case 3: prefix shapes ---
        for k in [1, 5, 10]:
            zk = model.encode_prefix(x, k)
            assert zk.shape == (batch_size, k), f"encode_prefix k={k}: wrong shape"
            xk = model.decode_prefix(zk, k)
            assert xk.shape == (batch_size, input_dim), f"decode_prefix k={k}: wrong shape"
        output_lines.append("  PASSED: encode_prefix/decode_prefix shapes for k in [1,5,10]")

        # --- Case 4: orthogonalize makes W W^T ≈ I ---
        # Corrupt weights first
        model.encoder.weight.data = torch.randn(embed_dim, input_dim)
        model.orthogonalize()
        WWT = model.encoder.weight @ model.encoder.weight.T
        residual = (WWT - torch.eye(embed_dim)).abs().mean().item()
        assert residual < 1e-5, f"orthogonalize residual too large: {residual:.6f}"
        output_lines.append(f"  PASSED: orthogonalize — W W^T residual = {residual:.2e}")

        # --- Case 5: gradient flow ---
        x2    = torch.randn(batch_size, input_dim)
        x_hat2 = model(x2)
        loss  = F.mse_loss(x_hat2, x2)
        loss.backward()
        assert model.encoder.weight.grad is not None, "No gradient on encoder weight"
        output_lines.append(f"  PASSED: gradient flows (grad norm={model.encoder.weight.grad.norm().item():.4f})")

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
    parser = argparse.ArgumentParser(description="Test runner for Experiment 6.")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip the end-to-end smoke test (only run unit tests and infra checks).",
    )
    args = parser.parse_args()

    code_dir    = os.path.dirname(os.path.abspath(__file__))
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
                 0, 1, suite_start, col_w)

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
