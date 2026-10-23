"""Run folders, provenance and shared inputs for Phase 0 / Experiment A.

Conda environment: mrl_env_cuda12
Usage:
    Import from the Phase 0 and Experiment A entry points.
"""

from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

from .data import DSprites, file_checksum
from .tasks import make_split, split_checksum, subsample, task_split, SPLIT_NAMES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # NonLinearPBExperiments/
PROJECT_ROOT = Path(__file__).resolve().parents[4]          # Mat_embedding/
DATA_PATH = PROJECT_ROOT / "data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
OUTPUT_ROOT = PROJECT_ROOT / "files/results"
PRETRAINED_MANIFEST = OUTPUT_ROOT / "exprmnt_2026_10_13__21_22_10/manifest.json"
PLAN = PACKAGE_ROOT.parent / "plans/dsprites_task_specific_mrl_v3_AEOLUS.md"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def make_run_dir(output_root, smoke):
    """files/results/exprmnt_<stamp>/, or files/results/test_runs/ for smoke runs."""
    root = Path(output_root) / ("test_runs" if smoke else "")
    stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_dir = root / f"exprmnt_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def snapshot(run_dir):
    """Copy this package and the governing plan; caches excluded."""
    shutil.copytree(PACKAGE_ROOT, run_dir / "code_snapshot/NonLinearPBExperiments",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if PLAN.exists():
        shutil.copy2(PLAN, run_dir / "code_snapshot" / PLAN.name)


def environment():
    versions = {}
    for package in ("numpy", "scipy", "torch", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    gpus = []
    if torch.cuda.is_available():
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        commit = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "cuda": torch.version.cuda, "gpus": gpus, "git_commit": commit,
            "command": sys.argv}


def pretrained_checkpoints(condition="beta_vae", seeds=(0, 1, 2)):
    """label -> checkpoint path for the existing three-seed reference models."""
    models = json.loads(PRETRAINED_MANIFEST.read_text())["models"][condition]
    return {f"{condition}_seed_{seed}": models[str(seed)]["checkpoint"] for seed in seeds}


def load_images(data_path):
    return DSprites(data_path).images


def prepare_split(run_dir, split_seed, fractions):
    split = make_split(split_seed, fractions)
    np.savez(run_dir / "split_manifest.npz", **split)
    info = {"split_seed": split_seed, "fractions": list(fractions),
            "sizes": {name: int(len(split[name])) for name in SPLIT_NAMES},
            "sha256": split_checksum(split)}
    return split, info


def task_data(features, split, task, limits, seed):
    """{split: (features, labels)} for one task, with optional smoke subsampling."""
    data = {}
    for offset, name in enumerate(SPLIT_NAMES):
        indices, _ = task_split(task, split[name])
        indices = subsample(indices, limits.get(name) if limits else None, seed + offset)
        indices, labels = task_split(task, indices)
        data[name] = (features[indices], labels)
    return data


def dataset_record(data_path):
    return {"data_path": str(data_path), "sha256": file_checksum(data_path)}
