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

    # Default tags (exp1 folders)
    python weight_symmetry/scripts/plot_reconstruction.py \\
        --weights-dir exprmnt_2026_04_12__10_00_00

    python weight_symmetry/scripts/plot_reconstruction.py \\
        --weights-dir exprmnt_2026_04_12__10_00_00 --recon-class 7

    # Override tags (e.g. exp2 MSE folder: fp_mrl_mse_ortho instead of
    # fullprefix_mrl_ortho)
    python weight_symmetry/scripts/plot_reconstruction.py \\
        --weights-dir exprmnt_2026_04_20__21_33_52 \\
        --tags "mse_lae:MSE LAE" "fp_mrl_mse_ortho:FP-MRL (MSE)"
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
from weight_symmetry.plotting.style import apply_style, save_fig

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
                              recon_classes, out_dir: str, fig_stamp: str):
    """
    models          : list of (tag, label, LinearAE) for model rows (repeated per class)
    data            : DataSplit (needs X_test, y_test, scaler)
    recon_classes   : list[int] — test images of each class are stacked as blocks of
                      rows (one block per class, n_models rows inside each block).
    """
    if dataset not in ("mnist", "fashion_mnist", "digits"):
        print(f"[recon] Dataset '{dataset}' is not an image dataset — skipping.")
        return

    apply_style()  # serif 9pt base (matches plot_fig_combined.py)

    img_shape    = (28, 28) if dataset in ("mnist", "fashion_mnist") else (8, 8)
    prefix_sizes = _prefix_sizes(embed_dim)
    n_cols       = 1 + len(prefix_sizes)
    n_models     = len(models)

    # Resolve one test image per class (first occurrence in test split).
    y_test_np = data.y_test.numpy()
    class_samples = []
    for c in recon_classes:
        idxs = np.where(y_test_np == c)[0]
        if len(idxs) == 0:
            print(f"[recon] Class {c} not found in test set — skipping.")
            continue
        class_samples.append((c, data.X_test[idxs[0]:idxs[0] + 1]))
    if not class_samples:
        print("[recon] No requested classes available — nothing to plot.")
        return

    n_rows = n_models * len(class_samples)

    # Nested gridspec: outer = class blocks (larger gap), inner = model rows
    # inside each block (tight gap).
    # Square cells (images are 28×28) so hspace actually bites — otherwise each
    # axes cell contains unreachable vertical whitespace around the image.
    fig = plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    outer = fig.add_gridspec(len(class_samples), 1,
                              left=0.10, right=0.98,
                              top=0.86, bottom=0.03,
                              hspace=0.12)   # gap between class blocks

    for block_idx, (cls, x_sample) in enumerate(class_samples):
        inner = outer[block_idx].subgridspec(n_models, n_cols,
                                              hspace=0.0005, wspace=0.08)
        for m_idx, (_, label, model) in enumerate(models):
            model.eval()
            with torch.no_grad():
                z_full = model.encode(x_sample)   # (1, embed_dim)

            # Original
            ax0 = fig.add_subplot(inner[m_idx, 0])
            ax0.imshow(_to_img(x_sample[0], data.scaler, img_shape),
                       cmap="gray", vmin=0, vmax=1)
            ax0.axis("off")
            if block_idx == 0 and m_idx == 0:
                ax0.set_title("Original")
            # Bold rotated row label (model name) — combined-style
            ax0.annotate(label,
                         xy=(-0.28, 0.5), xycoords="axes fraction",
                         fontsize=9, rotation=90, va="center", ha="center",
                         fontweight="bold")

            # Prefix reconstructions
            for ci, k in enumerate(prefix_sizes):
                with torch.no_grad():
                    x_hat = model.decode_prefix(z_full[:, :k], k)
                ax = fig.add_subplot(inner[m_idx, ci + 1])
                ax.imshow(_to_img(x_hat[0], data.scaler, img_shape),
                          cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if block_idx == 0 and m_idx == 0:
                    ax.set_title(f"$k={k}$")

    classes_str = ", ".join(str(c) for c, _ in class_samples)
    fig.suptitle(f"Partial reconstruction of class {classes_str}",
                 x=(0.10 + 0.98) / 2, y=0.95)

    # Save to the canonical ICMLWorkshop figures folder (pdf/png/svg)
    cls_tag = "_".join(str(c) for c, _ in class_samples)
    # fig_stamp already begins with "_"; save_fig appends "_{stamp}" itself,
    # so strip the leading underscore to avoid double "__".
    stamp_for_save = fig_stamp.lstrip("_")
    save_fig(fig, f"reconstruction_grid_class{cls_tag}", stamp_for_save)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot prefix reconstructions from a saved LinearAE run"
    )
    parser.add_argument("--weights-dir", required=True, metavar="FOLDER",
                        help="Saved run folder (contains config.json + seed*_best.pt)")
    parser.add_argument("--recon-class", type=int, nargs="+",
                        default=[DEFAULT_RECON_CLASS], metavar="C",
                        help="One or more digit classes to reconstruct "
                             f"(default: {DEFAULT_RECON_CLASS}). "
                             "Example: --recon-class 3 8")
    parser.add_argument("--tags", nargs="+", default=None, metavar="TAG:LABEL",
                        help="Override model tags/labels. Format: 'tag:Display label'. "
                             "Example: --tags 'mse_lae:MSE LAE' "
                             "'fp_mrl_mse_ortho:FP-MRL (MSE)'")
    args = parser.parse_args()

    if args.tags:
        recon_tags = []
        for t in args.tags:
            if ":" not in t:
                raise ValueError(f"--tags entry missing ':' separator: {t!r}")
            tag, label = t.split(":", 1)
            recon_tags.append((tag.strip(), label.strip()))
    else:
        recon_tags = RECON_TAGS

    weights_dir = args.weights_dir
    if not os.path.isabs(weights_dir):
        weights_dir = os.path.join(get_path("files/results"), weights_dir)

    print(f"[recon] Loading run from: {weights_dir}")

    with open(os.path.join(weights_dir, "config.json")) as f:
        cfg = json.load(f)

    dataset = cfg["dataset"]
    seed    = cfg["seeds"][0]

    # embed_dim may be absent (e.g. exp2 folders). Infer from the first
    # available checkpoint: encoder.weight has shape (embed_dim, input_dim).
    if "embed_dim" in cfg:
        embed_dim = cfg["embed_dim"]
    else:
        embed_dim = None
        for tag, _ in recon_tags:
            ckpt = os.path.join(weights_dir, f"seed{seed}_{tag}_best.pt")
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                embed_dim = state["encoder.weight"].shape[0]
                print(f"[recon] Inferred embed_dim={embed_dim} from {tag}")
                break
        if embed_dim is None:
            raise FileNotFoundError(
                f"Could not infer embed_dim: no checkpoints found for tags "
                f"{[t for t,_ in recon_tags]} in {weights_dir}"
            )

    print(f"[recon] Dataset: {dataset}  embed_dim: {embed_dim}  seed: {seed}")

    # Load data for test images + scaler
    print(f"[recon] Loading {dataset} (seed={seed}) ...")
    data = load_data(dataset, seed=seed)

    # Load model weights (first seed only — reconstruction is qualitative)
    models = []
    for tag, label in recon_tags:
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
        recon_classes=args.recon_class,
        out_dir=weights_dir,
        fig_stamp=fig_stamp,
    )

    print(f"\n[recon] Done. Output written to: {weights_dir}")


if __name__ == "__main__":
    main()
