"""
Script: run_tests.py
---------------------
Unified test runner for all modules in this project.

Runs each module's __main__ block as a subprocess (in dependency order),
captures its output, and prints a clean PASS / FAIL summary.

Usage (from code/):
    python run_tests.py                  # run all tests
    python run_tests.py --fast           # skip slow tests (loader w/ mnist, trainer)
    python run_tests.py --module config  # run only one module

Inputs:
    None — discovers tests from the MODULES list below.

Outputs:
    Console summary table with status, timing, and any error output.
    Exits with code 0 if all tests pass, 1 if any fail.
"""

import subprocess
import sys
import time
import os
import argparse

# Absolute path to code/ — works regardless of where this test file lives.
# tests/ is one level below code/, so we go up one directory.
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ==============================================================================
# Test definitions — in strict dependency order
# ==============================================================================

# Each entry: (short_name, script_path_relative_to_code_dir, is_slow)
# is_slow=True means it's skipped when --fast is passed
MODULES = [
    ("utility",      "utility.py",              False),
    ("config",       "config.py",               False),
    ("loader",       "data/loader.py",           True),   # downloads/loads MNIST
    ("encoder",      "models/encoder.py",        False),
    ("heads",        "models/heads.py",          False),
    ("mat_loss",     "losses/mat_loss.py",       False),
    ("trainer",      "training/trainer.py",      True),   # trains for a few epochs
    ("prefix_eval",  "evaluation/prefix_eval.py",False),
]


# ==============================================================================
# Runner
# ==============================================================================

def run_module(script_path: str, code_dir: str) -> tuple:
    """
    Run a single Python module as __main__ and return results.

    Args:
        script_path (str): Path to the .py file, relative to code_dir.
        code_dir    (str): Absolute path to the code/ directory.

    Returns:
        Tuple (passed: bool, elapsed: float, stdout: str, stderr: str)
    """
    abs_path = os.path.join(code_dir, script_path)
    start    = time.time()

    # Add code_dir to PYTHONPATH so subdirectory scripts (models/, losses/, etc.)
    # can import top-level modules like config.py via `from config import ...`.
    # Without this, sys.path.insert(0, "..") in their __main__ blocks resolves
    # ".." relative to cwd (= code/), landing in the project root instead of code/.
    env = os.environ.copy()
    env["PYTHONPATH"] = code_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, abs_path],
        capture_output=True,
        text=True,
        cwd=code_dir,          # run from code/ so relative imports resolve
        env=env,
    )

    elapsed = time.time() - start
    passed  = (result.returncode == 0)
    return passed, elapsed, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Run all module-level tests.")
    parser.add_argument("--fast",   action="store_true",
                        help="Skip slow tests (loader, trainer).")
    parser.add_argument("--module", type=str, default=None,
                        help="Run only this module by short name (e.g. 'config').")
    args = parser.parse_args()

    # Absolute path to code/ (defined at module level as CODE_DIR)
    code_dir = CODE_DIR

    # Filter which modules to run
    to_run = MODULES
    if args.module:
        to_run = [(n, p, s) for n, p, s in MODULES if n == args.module]
        if not to_run:
            print(f"ERROR: unknown module '{args.module}'. "
                  f"Choices: {[n for n,_,_ in MODULES]}")
            sys.exit(1)
    elif args.fast:
        to_run = [(n, p, s) for n, p, s in MODULES if not s]

    # ------------------------------------------------------------------
    # Run each module and collect results
    # ------------------------------------------------------------------
    col_w    = 14   # width of name column
    bar_w    = 20   # width of the progress bar fill
    n_total  = len(to_run)
    suite_start = time.time()

    print()
    print(f"{'MODULE':<{col_w}}  {'STATUS':<8}  {'TIME':>6}  PROGRESS")
    print("-" * 70)

    results = []
    for idx, (name, script, is_slow) in enumerate(to_run):
        # --- Live progress bar while the module is running ---
        def _progress_bar(n_done, n_total, elapsed):
            """Render a simple ASCII bar: [=====>    ] n/N  12.3s"""
            filled = int(bar_w * n_done / n_total)
            bar    = "=" * filled + (">" if filled < bar_w else "") + " " * (bar_w - filled)
            return f"[{bar}] {n_done}/{n_total}  {elapsed:.1f}s"

        # Show the bar at the start of this module (before it finishes)
        pre_elapsed = time.time() - suite_start
        bar_str     = _progress_bar(idx, n_total, pre_elapsed)
        sys.stdout.write(f"  {name:<{col_w}}  {'running...':<8}         {bar_str}\r")
        sys.stdout.flush()

        passed, elapsed, stdout, stderr = run_module(script, code_dir)
        results.append((name, passed, elapsed, stdout, stderr))

        # --- Print final result line with updated bar ---
        post_elapsed = time.time() - suite_start
        status       = "PASS" if passed else "FAIL"
        bar_str      = _progress_bar(idx + 1, n_total, post_elapsed)
        print(f"  {name:<{col_w}}  {status:<8}  {elapsed:>5.1f}s  {bar_str}")

        # On failure, show captured output indented
        if not passed:
            print()
            if stdout:
                for line in stdout.strip().splitlines():
                    print(f"    [stdout] {line}")
            if stderr:
                for line in stderr.strip().splitlines()[-30:]:  # last 30 lines
                    print(f"    [stderr] {line}")
            print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = sum(1 for _, p, *_ in results if p)
    n_fail = len(results) - n_pass
    total  = sum(e for _, _, e, *_ in results)

    print("-" * 60)
    print(f"  {n_pass}/{len(results)} passed   {total:.1f}s total")

    if n_fail > 0:
        failed_names = [n for n, p, *_ in results if not p]
        print(f"  FAILED: {', '.join(failed_names)}")
        print()
        sys.exit(1)
    else:
        print("  All tests passed.")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
