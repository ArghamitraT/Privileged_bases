"""Compact dSprites loading and exact factorial indexing.

Conda environment: mrl_env_cuda12
Usage:
    Import DSprites from the reproduction entry point.
"""

import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

FACTOR_NAMES = ("shape", "scale", "rotation", "x", "y")
FACTOR_SIZES = np.array([3, 6, 40, 32, 32], dtype=np.int64)
FACTOR_BASES = np.array([245760, 40960, 1024, 32, 1], dtype=np.int64)
TARGET_COLUMNS = (3, 4, 1, 2)
TARGET_NAMES = ("x", "y", "scale", "rotation")


def file_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_indices(factors):
    factors = np.asarray(factors)
    if factors.shape[-1] != 5 or np.any(factors < 0) or np.any(factors >= FACTOR_SIZES):
        raise ValueError("Invalid dSprites factor indices")
    return factors @ FACTOR_BASES


class DSprites(Dataset):
    def __init__(self, path, limit=None, seed=0):
        with np.load(path, allow_pickle=False) as archive:
            self.images = archive["imgs"]
            classes = archive["latents_classes"]
        if self.images.shape != (737280, 64, 64):
            raise ValueError(f"Unexpected image shape: {self.images.shape}")
        if self.images.dtype not in (np.dtype("uint8"), np.dtype("bool")):
            raise ValueError("Expected binary uint8/bool images")
        # Chunk checks avoid full-sized temporary image arrays.
        for start in range(0, len(self.images), 8192):
            if self.images[start:start + 8192].max() > 1:
                raise ValueError("Nonbinary image values")
        if classes.shape != (737280, 6) or np.any(classes[:, 0] != 0):
            raise ValueError("Unexpected factor annotations / color factor")
        if not np.array_equal(factor_indices(classes[:, 1:]), np.arange(len(classes))):
            raise ValueError("Archive factor order differs from the metric lookup")
        self.indices = np.arange(len(self.images))
        if limit is not None:
            if not 1 <= limit <= len(self.images):
                raise ValueError("Invalid image subset size")
            self.indices = np.random.default_rng(seed).choice(
                len(self.images), limit, replace=False
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[self.indices[index]].reshape(4096))
