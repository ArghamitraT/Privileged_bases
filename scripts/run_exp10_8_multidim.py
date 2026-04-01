"""
Script: scripts/run_exp10_8_multidim.py
-----------------------------------------
Wrapper that runs Experiment 10 then Experiment 8 sequentially for each
requested embedding dimension (default: 8, 16, 32).

For each dim:
  1. Run exp10 --embed-dim {dim}  (train Standard/L1/MRL, dense prefix sweep)
  2. Parse the "[utility] Run directory created: PATH" line from stdout
     to locate the exp10 output folder automatically.
  3. Run exp8 --embed-dim {dim} --use-weights {exp10_dir}
     (per-dimension importance scoring, best-k vs first-k, method agreement)

Each subprocess streams live output to the terminal. The wrapper stops
immediately if any subprocess fails (non-zero exit code).

A summary table is printed at the end showing which output folders correspond
to which (dim, experiment) pair.

Inputs:
  --fast        : forward --fast to both exp10 and exp8 (digits, 5 epochs)
  --dims N ...  : run only the specified dims (default: 8 16 32)

Usage:
    python scripts/run_exp10_8_multidim.py                # full run, dims=[8,16,32]
    python scripts/run_exp10_8_multidim.py --fast         # smoke test, all 3 dims
    python scripts/run_exp10_8_multidim.py --dims 8 16    # only dims 8 and 16
    python scripts/run_exp10_8_multidim.py --dims 32 --fast  # dim=32 smoke test
"""

import os
import sys
import re
import subprocess
import argparse
import time

# Cap BLAS threads on child processes via environment (set before subprocess launch)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

# Locate code/ directory (this script is one level below code/scripts/)
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Pattern emitted by utility.create_run_dir()
_RUN_DIR_PATTERN = re.compile(r"\[utility\] Run directory created: (.+)")


def run_subprocess(cmd, label):
    """
    Run a command as a subprocess, streaming output live to stdout.

    Passes the same BLAS thread-cap env vars to the child process to
    prevent macOS OpenMP deadlocks (Known Issue #3 in CLAUDE.md).

    Args:
        cmd   (list[str]) : Command and arguments.
        label (str)       : Short label for log messages.

    Returns:
        str: The run directory path parsed from the subprocess output.

    Raises:
        RuntimeError: If the subprocess exits with non-zero code.
        RuntimeError: If no run directory line is found in the output.
    """
    print(f"\n{'='*70}")
    print(f"[wrapper] Starting: {label}")
    print(f"[wrapper] Command:  {' '.join(cmd)}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"]      = "1"
    env["MKL_NUM_THREADS"]      = "1"

    run_dir_found = None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=CODE_DIR,
    )

    for line in proc.stdout:
        # Stream line live to terminal
        print(line, end="", flush=True)
        # Check for run directory announcement
        m = _RUN_DIR_PATTERN.search(line)
        if m:
            run_dir_found = m.group(1).strip()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"[wrapper] FAILED: {label} exited with code {proc.returncode}"
        )

    if run_dir_found is None:
        raise RuntimeError(
            f"[wrapper] Could not parse run directory from output of: {label}\n"
            "Expected a line matching: [utility] Run directory created: PATH"
        )

    print(f"\n[wrapper] Completed: {label}")
    print(f"[wrapper] Output dir: {run_dir_found}\n")
    return run_dir_found


def main():
    """
    Parse args and run exp10 + exp8 for each requested embedding dimension.
    """
    total_start = time.time()

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Wrapper: run exp10 then exp8 for each embed_dim"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Forward --fast to exp10 and exp8 (digits, 5 epochs).",
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=[8, 16, 32],
        metavar="N",
        help="Embedding dimensions to run (default: 8 16 32).",
    )
    args = parser.parse_args()

    dims     = args.dims
    fast_arg = ["--fast"] if args.fast else []

    print(f"[wrapper] embed_dims to run: {dims}")
    print(f"[wrapper] fast mode: {args.fast}")
    print(f"[wrapper] code dir:  {CODE_DIR}\n")

    # ------------------------------------------------------------------
    # Run each dim
    # ------------------------------------------------------------------
    summary = []   # list of (dim, exp10_dir, exp8_dir)

    for dim in dims:
        dim_start = time.time()
        print(f"\n{'#'*70}")
        print(f"# embed_dim = {dim}")
        print(f"{'#'*70}")

        # --- Step 1: Run exp10 ---
        exp10_cmd = [
            sys.executable,
            os.path.join(CODE_DIR, "experiments", "exp10_dense_multidim.py"),
            "--embed-dim", str(dim),
        ] + fast_arg

        exp10_dir = run_subprocess(
            exp10_cmd,
            label=f"exp10 (embed_dim={dim})",
        )

        # --- Step 2: Run exp8 with weights from exp10 ---
        exp8_cmd = [
            sys.executable,
            os.path.join(CODE_DIR, "experiments", "exp8_dim_importance.py"),
            "--embed-dim", str(dim),
            "--use-weights", exp10_dir,
        ] + fast_arg

        exp8_dir = run_subprocess(
            exp8_cmd,
            label=f"exp8 (embed_dim={dim}, weights from exp10)",
        )

        dim_elapsed = time.time() - dim_start
        summary.append((dim, exp10_dir, exp8_dir, dim_elapsed))
        print(f"[wrapper] dim={dim} done in {dim_elapsed/60:.1f} min")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"[wrapper] ALL DONE  ({total_elapsed/60:.1f} min total)")
    print(f"{'='*70}")
    print(f"\n{'embed_dim':<12}  {'exp10 output dir':<45}  {'exp8 output dir'}")
    print("-" * 120)
    for dim, exp10_dir, exp8_dir, elapsed in summary:
        # Print just the folder name (last component) to keep the table readable
        exp10_name = os.path.basename(exp10_dir.rstrip("/"))
        exp8_name  = os.path.basename(exp8_dir.rstrip("/"))
        print(f"{dim:<12}  {exp10_name:<45}  {exp8_name}  ({elapsed/60:.1f}m)")

    print(f"\n[wrapper] Full paths:")
    for dim, exp10_dir, exp8_dir, _ in summary:
        print(f"  dim={dim}")
        print(f"    exp10: {exp10_dir}")
        print(f"    exp8:  {exp8_dir}")


if __name__ == "__main__":
    main()
