"""CIFAR-100 (32x32 colour) for the Experiment 1 trainer, checksum-verified like the synthetic data.

Conda environment: mrl_env_cuda12
Usage:
    from hier_exp1.cifar import load_split, dataset_record
    images, labels = load_split("train")      # uint8 (N, 3, 32, 32), {coarse (20 superclasses), fine (100), mid_node}

Made once by scripts/prepare_cifar100.py: the official 50,000 training images split into train 45,000 / val 5,000
(fixed permutation, seed 0, recorded in config.json); the official 10,000 test images are the test split.
Labels are not used in training. mid_node = coarse here (the W&B grid picks one val image per value).
"""

import json
from pathlib import Path

import numpy as np

from hier_exp1.data import PROJECT_ROOT, file_sha256

DATA_DIR = PROJECT_ROOT / "data/cifar100/v1"
SPLITS = ("train", "val", "test")


def dataset_config(data_dir=DATA_DIR):
    return json.loads((Path(data_dir) / "config.json").read_text())


def load_split(name, data_dir=DATA_DIR, verify=True):
    path = Path(data_dir) / f"{name}.npz"
    if verify:
        if file_sha256(path) != dataset_config(data_dir)["sha256"][f"{name}.npz"]:
            raise ValueError(f"{path} does not match the checksum in its config.json")
    d = np.load(path)
    labels = {"coarse": d["coarse"].astype(np.int64), "fine": d["fine"].astype(np.int64)}
    labels["mid_node"] = labels["coarse"]
    return d["images"], labels


def dataset_record(data_dir=DATA_DIR):
    cfg = dataset_config(data_dir)
    return {"data_dir": str(data_dir), "version": cfg["version"], "frozen": cfg["frozen"], "sha256": cfg["sha256"]}
