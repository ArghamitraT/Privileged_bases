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
import json
import dataclasses


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

def create_run_dir(fast: bool = False) -> str:
    """
    Create a timestamped output folder for a single experiment run.

    All results (logs, figures, model weights) for one run go here.
    The folder is always named:
        exprmnt_{YYYY_MM_DD__{HH_MM_SS}

    Full runs are created at:
        Mat_embedding_hyperbole/files/results/exprmnt_{timestamp}/

    --fast (smoke test) runs are created at:
        Mat_embedding_hyperbole/files/results/test_runs/exprmnt_{timestamp}/

    This folder is outside the code/ directory and is NOT tracked by git.

    Args:
        fast (bool): If True, place the run under files/results/test_runs/.
                     Pass args.fast from the experiment's argparse.

    Returns:
        str: Absolute path to the newly created run directory.

    Example:
        >>> run_dir = create_run_dir()
        >>> run_dir = create_run_dir(fast=True)
    """
    folder_name = create_timestamped_filename("exprmnt")

    base = get_path("files/results/test_runs") if fast else get_path("files/results")
    run_dir = os.path.join(base, folder_name)
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
# Config serialisation — save / load ExpConfig as JSON
# ==============================================================================

def save_config_json(cfg, run_dir: str) -> str:
    """
    Serialise an ExpConfig dataclass to config.json in the run directory.

    This makes every run folder self-describing: any later experiment
    (e.g. exp8 loading exp10 weights) can reconstruct the exact config
    without parsing log files or guessing parameters.

    Args:
        cfg     : ExpConfig dataclass instance.
        run_dir (str): Path to the run output directory.

    Returns:
        str: Path to the written config.json.
    """
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    print(f"[utility] Config saved to {path}")
    return path


def load_config_json(run_dir: str) -> dict:
    """
    Load the config.json saved by save_config_json from a run directory.

    Args:
        run_dir (str): Path to the run output directory (or weights folder).

    Returns:
        dict: The saved config fields, or {} if config.json does not exist.
    """
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        cfg_dict = json.load(f)
    print(f"[utility] Config loaded from {path}")
    return cfg_dict


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
