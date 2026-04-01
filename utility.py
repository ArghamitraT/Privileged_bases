"""
Script: utility.py
------------------
Shared utility helpers used across all experiments in this project.

Provides:
- create_timestamped_filename : build a timestamped filename string
- get_path                    : resolve an absolute path inside the project root
- create_run_dir              : create a self-contained timestamped output folder
                                under files/results/ for a given experiment name
- save_runtime                : write total experiment wall-clock time to runtime.txt
- save_code_snapshot          : copy the entire code/ folder into run_dir/code_snapshot/
                                so any result folder is fully self-contained and
                                exactly reproducible from its own snapshot

Usage:
    from utility import create_run_dir, save_runtime, save_code_snapshot, get_path
    python utility.py   # quick sanity check (prints resolved project root + test path)
"""

import os
import time
import shutil


# ==============================================================================
# Filename / path helpers
# ==============================================================================

def create_timestamped_filename(kind: str) -> str:
    """
    Create a timestamped filename string with the given prefix.

    Args:
        kind (str): Prefix for the filename (e.g. 'log', 'exp1').

    Returns:
        str: String in the format '{kind}_YYYY_MM_DD__HH_MM_SS'.
    """
    stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    return f"{kind}{stamp}"


def get_path(target_suffix: str, root_name: str = "Mat_embedding_hyperbole") -> str:
    """
    Resolve an absolute path inside the project root, creating the folder if needed.

    Walks upward from the current working directory to find the project root
    folder by name, then appends the given suffix.

    Args:
        target_suffix (str): Relative path inside the project (e.g. 'code/figure').
        root_name     (str): Name of the project root folder
                             (default: 'Mat_embedding_hyperbole').

    Returns:
        str: Absolute path to the target directory (created if it did not exist).

    Raises:
        FileNotFoundError: If the project root cannot be found upward from cwd.
    """
    cwd = os.path.abspath(os.getcwd())
    parts = cwd.split(os.sep)

    # Walk upward to find the project root by folder name
    root_path = None
    for i in range(len(parts), 0, -1):
        candidate = os.sep.join(parts[:i])
        if os.path.basename(candidate) == root_name:
            root_path = candidate
            break

    if root_path is None:
        raise FileNotFoundError(
            f"Project root folder '{root_name}' not found upward from {cwd}"
        )

    # Build and create the target path
    target_path = os.path.join(root_path, os.path.normpath(target_suffix))
    os.makedirs(target_path, exist_ok=True)

    return target_path


# ==============================================================================
# Run output directory
# ==============================================================================

def create_run_dir() -> str:
    """
    Create a timestamped output folder for a single experiment run.

    All results (logs, figures, model weights) for one run go here.
    The folder is always named:
        exprmnt_{YYYY_MM_DD__{HH_MM_SS}

    and is created at:
        Mat_embedding_hyperbole/files/results/exprmnt_{timestamp}/

    This folder is outside the code/ directory and is NOT tracked by git.

    Returns:
        str: Absolute path to the newly created run directory.

    Example:
        >>> run_dir = create_run_dir()
        >>> # returns something like:
        >>> # .../Mat_embedding_hyperbole/files/results/exprmnt_2026_03_06__14_30_00/
    """
    # Folder name is always exprmnt + timestamp
    folder_name = create_timestamped_filename("exprmnt")

    # Place it under files/results/ in the project root
    results_base = get_path("files/results")
    run_dir = os.path.join(results_base, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[utility] Run directory created: {run_dir}")
    return run_dir


# ==============================================================================
# Runtime recording
# ==============================================================================

def save_runtime(run_dir: str, elapsed_seconds: float) -> str:
    """
    Write the total wall-clock runtime of an experiment run to runtime.txt.

    Call this at the very end of main() after all work is done:
        elapsed = time.time() - run_start
        save_runtime(run_dir, elapsed)

    Args:
        run_dir         (str)  : Path to the run output directory.
        elapsed_seconds (float): Total elapsed time in seconds.

    Returns:
        str: Path to the written runtime.txt file.
    """
    h = int(elapsed_seconds // 3600)
    m = int((elapsed_seconds % 3600) // 60)
    s = elapsed_seconds % 60

    runtime_str = f"{h:02d}h {m:02d}m {s:05.2f}s  ({elapsed_seconds:.1f}s total)"
    path = os.path.join(run_dir, "runtime.txt")

    with open(path, "w") as f:
        f.write(runtime_str + "\n")

    print(f"[utility] Runtime: {runtime_str}")
    print(f"[utility] Saved to {path}")
    return path


# ==============================================================================
# Code snapshot
# ==============================================================================

def save_code_snapshot(run_dir: str) -> str:
    """
    Copy the entire code/ folder into run_dir/code_snapshot/.

    This makes every result folder fully self-contained: you can reproduce
    any past run by activating the env and running the snapshot directly,
    without needing to know the git state at the time.

    Excludes: __pycache__, *.pyc, .git, .DS_Store, figure/ (old plots).
    Includes: all .py source files, env/ yml files, README.md, CLAUDE.md.

    Args:
        run_dir (str): Path to the run output directory.

    Returns:
        str: Path to the created code_snapshot/ directory.
    """
    # The code/ folder is the parent of this file (utility.py lives in code/)
    code_dir      = os.path.dirname(os.path.abspath(__file__))
    snapshot_dest = os.path.join(run_dir, "code_snapshot")

    # Patterns to skip — env/ conda dirs are huge, figure/ is just old plots
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".DS_Store", "figure",
    )

    shutil.copytree(code_dir, snapshot_dest, ignore=ignore)
    print(f"[utility] Code snapshot saved to {snapshot_dest}")
    return snapshot_dest


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    # Checkpoint: verify all helpers work correctly

    print("--- Testing create_timestamped_filename ---")
    fname = create_timestamped_filename("test")
    print(f"  Result: {fname}")
    assert fname.startswith("test_"), "Should start with 'test_'"
    print("  PASSED\n")

    print("--- Testing get_path ---")
    p = get_path("code/figure")
    print(f"  Result: {p}")
    assert os.path.isdir(p), "Directory should exist after get_path()"
    print("  PASSED\n")

    print("--- Testing create_run_dir ---")
    run_dir = create_run_dir()
    print(f"  Result: {run_dir}")
    assert os.path.isdir(run_dir), "Run directory should have been created"
    assert os.path.basename(run_dir).startswith("exprmnt_"), "Folder should start with 'exprmnt_'"
    print("  PASSED\n")

    print("All utility checks passed.")
