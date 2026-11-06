"""Frozen hierarchical image data (v2) for Experiment 1, checksum-verified.

Conda environment: mrl_env_cuda12
Usage:
    from hier_exp1.data import load_split, dataset_record
    images, labels = load_split("train")      # uint8 (N, 64, 64), dict of int64 arrays

Spec: NonLinearPB/plans/hierarchical_Exp1.md. Every load checks the npz SHA-256
against the dataset's own config.json, so a run can never silently use a
different dataset version.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]          # Mat_embedding/
DATA_DIR = PROJECT_ROOT / "data/synthetic_data/hierarchical_images_v2"
OUTPUT_ROOT = PROJECT_ROOT / "files/results"
SPLITS = ("train", "val", "test")
LABELS = ("coarse", "mid", "fine", "leaf_id")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_config(data_dir=DATA_DIR):
    return json.loads((Path(data_dir) / "config.json").read_text())


def load_split(name, data_dir=DATA_DIR, verify=True):
    """uint8 images (N, 64, 64) and labels {coarse, mid, fine, leaf_id, mid_node}; mid_node = 3c + m."""
    path = Path(data_dir) / f"{name}.npz"
    if verify:
        expected = dataset_config(data_dir)["sha256"][f"{name}.npz"]
        if file_sha256(path) != expected:
            raise ValueError(f"{path} does not match the checksum in its config.json")
    d = np.load(path)
    labels = {k: d[k].astype(np.int64) for k in LABELS}
    labels["mid_node"] = 3 * labels["coarse"] + labels["mid"]
    return d["images"], labels


def dataset_record(data_dir=DATA_DIR):
    cfg = dataset_config(data_dir)
    return {"data_dir": str(data_dir), "version": cfg["version"], "frozen": cfg["frozen"], "sha256": cfg["sha256"]}
