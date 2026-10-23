"""Frozen-encoder posterior means for probes and fixed-feature experiments.

Conda environment: mrl_env_cuda12
Usage:
    Import load_encoder, encode_means and check_saved_means.
"""

from pathlib import Path

import numpy as np
import torch

from .model import model_from_config


def load_encoder(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(state["config"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["config"]


@torch.no_grad()
def encode_means(model, images, device, batch_size=8192):
    """mu(x) for every image, in archive order."""
    means = np.empty((len(images), model.mean.out_features), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(images[start:start + batch_size].reshape(-1, 4096))
        means[start:start + batch_size] = model.encode(x.to(device, torch.float32))[0].cpu().numpy()
    if not np.isfinite(means).all():
        raise RuntimeError("Nonfinite posterior means")
    return means


def check_saved_means(checkpoint, means):
    """Compare with latent_means.npy saved next to the checkpoint, if present."""
    saved = Path(checkpoint).parent / "latent_means.npy"
    if not saved.exists():
        return {"saved_means": None}
    reference = np.load(saved, mmap_mode="r")
    if reference.shape != means.shape:
        return {"saved_means": str(saved), "shape_mismatch": list(reference.shape)}
    return {"saved_means": str(saved),
            "max_abs_difference": float(np.max(np.abs(reference - means)))}
