"""Image-level splits and factor-classification targets for supervised tasks.

Conda environment: mrl_env_cuda12
Usage:
    Import make_split, task_split and factor_labels from the Phase 0 /
    Experiment A (and later Experiment C) entry points.

Labels come from the exact factorial archive order (see data.factor_indices),
so no archive annotations are needed to build them. dSprites orientation is
linspace(0, 2*pi, 40): index 39 is the same angle as index 0 (verified
2026-10-22: pixel-identical images for all three shapes), so rotation uses
39 classes with 39 -> 0.
"""

import hashlib

import numpy as np

from .data import FACTOR_NAMES, FACTOR_SIZES

N_IMAGES = int(np.prod(FACTOR_SIZES))
HEART = 2  # dSprites shape order: square, ellipse, heart
ROTATION_CLASSES = 39
SPLIT_NAMES = ("train", "validation", "test")
# task name -> (factor predicted, heart images only)
TASKS = {
    "x": ("x", False), "y": ("y", False), "shape": ("shape", False),
    "scale": ("scale", False), "rotation_heart": ("rotation", True),
    "x_heart": ("x", True), "y_heart": ("y", True),
}


def image_factors(indices):
    """Factor indices (shape, scale, rotation, x, y) for archive image indices."""
    return np.stack(np.unravel_index(np.asarray(indices), FACTOR_SIZES), axis=1)


def make_split(seed=0, fractions=(0.8, 0.1, 0.1)):
    """Random image-level disjoint split; each image is one factor combination."""
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9 or min(fractions) <= 0:
        raise ValueError("Split fractions must be three positive values summing to one")
    order = np.random.default_rng(seed).permutation(N_IMAGES)
    n_train = int(round(fractions[0] * N_IMAGES))
    n_val = int(round(fractions[1] * N_IMAGES))
    return {"train": np.sort(order[:n_train]),
            "validation": np.sort(order[n_train:n_train + n_val]),
            "test": np.sort(order[n_train + n_val:])}


def split_checksum(split):
    digest = hashlib.sha256()
    for name in SPLIT_NAMES:
        digest.update(name.encode())
        digest.update(np.asarray(split[name], dtype=np.int64).tobytes())
    return digest.hexdigest()


def n_classes(factor):
    if factor == "rotation":
        return ROTATION_CLASSES
    return int(FACTOR_SIZES[FACTOR_NAMES.index(factor)])


def factor_labels(indices, factor):
    labels = image_factors(indices)[:, FACTOR_NAMES.index(factor)]
    if factor == "rotation":
        labels = labels % ROTATION_CLASSES
    return labels.astype(np.int64)


def regression_targets(labels, factor):
    """Ordered factors as numbers for R^2; rotation as (cos, sin); shape has none."""
    if factor == "shape":
        return None
    if factor == "rotation":
        angle = labels * (2 * np.pi / ROTATION_CLASSES)
        return np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return labels.astype(np.float64)[:, None]


def task_split(task, indices):
    """Restrict to the task's image subset; return (indices, labels)."""
    factor, heart_only = TASKS[task]
    indices = np.asarray(indices)
    if heart_only:
        indices = indices[image_factors(indices)[:, 0] == HEART]
    return indices, factor_labels(indices, factor)


def subsample(indices, limit, seed):
    """Fixed random subset for smoke runs; the full split when limit is None."""
    if limit is None or limit >= len(indices):
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, limit, replace=False))
