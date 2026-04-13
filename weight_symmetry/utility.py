"""
Script: weight_symmetry/utility.py
-----------------------------------
Shared utility helpers for weight_symmetry experiments.
Copied and adapted from code/utility.py.

Provides:
- create_timestamped_filename : build a timestamped filename string
- get_path                    : resolve an absolute path inside the project root
- create_run_dir              : create a timestamped output folder under files/results/
- save_runtime                : write total wall-clock time to runtime.txt
- save_code_snapshot          : copy weight_symmetry/ into run_dir/code_snapshot/
- save_config                 : dump config dict to config.json

Usage:
    from weight_symmetry.utility import create_run_dir, save_runtime, save_code_snapshot
"""

import os
import time
import shutil
import json


def create_timestamped_filename(kind: str) -> str:
    stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    return f"{kind}{stamp}"


def get_path(target_suffix: str, root_name: str = "Mat_embedding_hyperbole") -> str:
    """
    Resolve an absolute path inside the project root, creating the folder if needed.
    Walks upward from cwd to find the project root by name.
    """
    cwd = os.path.abspath(os.getcwd())
    parts = cwd.split(os.sep)

    root_path = None
    for i in range(len(parts), 0, -1):
        candidate = os.sep.join(parts[:i])
        if os.path.basename(candidate) == root_name:
            root_path = candidate
            break

    if root_path is None:
        raise FileNotFoundError(
            f"Project root '{root_name}' not found upward from {cwd}"
        )

    target_path = os.path.join(root_path, os.path.normpath(target_suffix))
    os.makedirs(target_path, exist_ok=True)
    return target_path


def create_run_dir(fast: bool = False) -> str:
    """
    Create a timestamped output folder.
    Full runs  -> files/results/exprmnt_{timestamp}/
    Fast runs  -> files/results/test_runs/exprmnt_{timestamp}/
    """
    folder_name = create_timestamped_filename("exprmnt")
    base = get_path("files/results/test_runs") if fast else get_path("files/results")
    run_dir = os.path.join(base, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[utility] Run directory created: {run_dir}")
    return run_dir


def save_runtime(run_dir: str, elapsed_seconds: float) -> str:
    h = int(elapsed_seconds // 3600)
    m = int((elapsed_seconds % 3600) // 60)
    s = elapsed_seconds % 60
    runtime_str = f"{h:02d}h {m:02d}m {s:05.2f}s  ({elapsed_seconds:.1f}s total)"
    path = os.path.join(run_dir, "runtime.txt")
    with open(path, "w") as f:
        f.write(runtime_str + "\n")
    print(f"[utility] Runtime: {runtime_str}")
    return path


def save_code_snapshot(run_dir: str) -> str:
    """
    Copy the weight_symmetry/ folder into run_dir/code_snapshot/.
    """
    source_dir    = os.path.dirname(os.path.abspath(__file__))
    snapshot_dest = os.path.join(run_dir, "code_snapshot")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".DS_Store")
    shutil.copytree(source_dir, snapshot_dest, ignore=ignore)
    print(f"[utility] Code snapshot saved to {snapshot_dest}")
    return snapshot_dest


def save_config(cfg: dict, run_dir: str) -> str:
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[utility] Config saved to {path}")
    return path
