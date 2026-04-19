"""
Script: weight_symmetry/scripts/plot_reconstruction.py
-------------------------------------------------------
Visualise prefix reconstructions from a saved exp1 run — no retraining.

For a single test image (first occurrence of --recon-class in the test split),
encode the full embedding z (all embed_dim dims), then decode from the prefix
z[:k] for increasing k.  This shows how much information the first k dimensions
carry: a model that front-loads information (MRL + ortho) should produce
recognisable images at small k; a plain MSE LAE should not.

--recon-class : integer digit/class label (0–9 for MNIST) whose first test-set
    image is used.  Default: 3.  Choose a class with distinctive shape so
    reconstruction quality differences are visually obvious.

Models shown (subset of exp1 MODEL_CONFIGS):
    mse_lae              — MSE LAE (baseline, no ordering)
    fullprefix_mrl_ortho — Full-prefix MRL + dec ortho (expected best ordering)

Layout:
    rows  = models (one per RECON_TAGS entry)
    cols  = original | k=2 | k=4 | k=8 | ... | k=embed_dim

Output saved to --weights-dir as:
    reconstruction_grid_{stamp}.png

Usage:
    Conda environment: mrl_env_cuda12  (GPU, CUDA 12.4, torch 2.5.1+cu124)
                       mrl_env         (CPU fallback — same code, no GPU)

    python weight_symmetry/scripts/plot_reconstruction.py \\
        --weights-dir exprmnt_2026_04_12__10_00_00

    python weight_symmetry/scripts/plot_reconstruction.py \\
        --weights-dir exprmnt_2026_04_12__10_00_00 --recon-class 7
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_WS_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.data.loader import load_data
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.utility import get_path

# Models to include — must match tags saved in the exp1 run folder
RECON_TAGS = [
    ("mse_lae",              "MSE LAE"),
    ("fullprefix_mrl_ortho", "Full-prefix MRL + dec ortho"),
]

DEFAULT_RECON_CLASS = 3   # overridable via --recon-class


def _prefix_sizes(embed_dim: int) -> list:
    """Powers of 2 from 2 up to embed_dim, always including embed_dim."""
    sizes = []
    k = 2
    while k < embed_dim:
        sizes.append(k)
        k *= 2
    sizes.append(embed_dim)
    return sizes


def _to_img(vec_tensor, scaler, img_shape):
    v = scaler.inverse_transform(vec_tensor.detach().numpy().reshape(1, -1))[0]
    return np.clip(v, 0, 1).reshape(img_shape)


def plot_reconstruction_grid(models, data, embed_dim: int, dataset: str,
                              recon_class: int, out_dir: str, fig_stamp: str):
    """
    models : list of (tag, label, LinearAE) for rows
    data   : DataSplit (needs X_test, y_test, scaler)
    """
    if dataset not in ("mnist", "fashion_mnist", "digits"):
        print(f"[recon] Dataset '{dataset}' is not an image dataset — skipping.")
        return

    img_shape    = (28, 28) if dataset in ("mnist", "fashion_mnist") else (8, 8)
    prefix_sizes = _prefix_sizes(embed_dim)
    n_cols       = 1 + len(prefix_sizes)
    n_rows       = len(models)

    # First test image of recon_class
    y_test_np = data.y_test.numpy()
    idxs = np.where(y_test_np == recon_class)[0]
    if len(idxs) == 0:
        print(f"[recon] Class {recon_class} not found in test set — skipping.")
        return
    x_sample = data.X_test[idxs[0]:idxs[0] + 1]   # (1, input_dim)

    _, axes = plt.subplots(n_rows, n_cols,
                           figsize=(n_cols * 1.5, n_rows * 1.9))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (_, label, model) in enumerate(models):
        model.eval()
        with torch.no_grad():
            z_full = model.encode(x_sample)   # (1, embed_dim)

        # Original
        ax0 = axes[row, 0]
        ax0.imshow(_to_img(x_sample[0], data.scaler, img_shape),
                   cmap="gray", vmin=0, vmax=1)
        ax0.axis("off")
        if row == 0:
            ax0.set_title("orig", fontsize=8)
        # Row label using text (ylabel is hidden by axis("off"))
        ax0.text(-0.15, 0.5, label, transform=ax0.transAxes,
                 ha="right", va="center", fontsize=7, wrap=True)

        # Prefix reconstructions
        for ci, k in enumerate(prefix_sizes):
            with torch.no_grad():
                x_hat = model.decode_prefix(z_full[:, :k], k)
            ax = axes[row, ci + 1]
            ax.imshow(_to_img(x_hat[0], data.scaler, img_shape),
                      cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"k={k}", fontsize=8)

    plt.suptitle(
        f"Prefix reconstructions — {dataset}, class {recon_class}  "
        f"(embed_dim={embed_dim})",
        fontsize=9
    )
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])
    path = os.path.join(out_dir, f"reconstruction_grid{fig_stamp}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[recon] Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot prefix reconstructions from a saved exp1 run"
    )
    parser.add_argument("--weights-dir", required=True, metavar="FOLDER",
                        help="Saved exp1 run folder (contains config.json + seed*_best.pt)")
    parser.add_argument("--recon-class", type=int, default=DEFAULT_RECON_CLASS,
                        metavar="C",
                        help=f"Digit class to reconstruct (default: {DEFAULT_RECON_CLASS})")
    args = parser.parse_args()

    weights_dir = args.weights_dir
    if not os.path.isabs(weights_dir):
        weights_dir = os.path.join(get_path("files/results"), weights_dir)

    print(f"[recon] Loading run from: {weights_dir}")

    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg = json.load(f)

    dataset   = cfg["dataset"]
    embed_dim = cfg["embed_dim"]
    seed      = cfg["seeds"][0]

    print(f"[recon] Dataset: {dataset}  embed_dim: {embed_dim}  seed: {seed}")

    # Load data for test images + scaler
    print(f"[recon] Loading {dataset} (seed={seed}) ...")
    data = load_data(dataset, seed=seed)

    # Load model weights (first seed only — reconstruction is qualitative)
    models = []
    for tag, label in RECON_TAGS:
        ckpt = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
        if not os.path.exists(ckpt):
            print(f"[recon] WARNING: checkpoint not found, skipping: {ckpt}")
            continue
        model = LinearAE(input_dim=data.input_dim, embed_dim=embed_dim)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        models.append((tag, label, model))
        print(f"[recon] Loaded {tag} from {ckpt}")

    if not models:
        print("[recon] No models loaded — exiting.")
        return

    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    plot_reconstruction_grid(
        models, data, embed_dim, dataset,
        recon_class=args.recon_class,
        out_dir=weights_dir,
        fig_stamp=fig_stamp,
    )

    print(f"\n[recon] Done. Output written to: {weights_dir}")


if __name__ == "__main__":
    main()
