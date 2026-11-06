"""Tree figure of the hierarchical synthetic dataset: root -> 3 coarse -> 9 mid -> 27 leaves.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/scripts/plot_hier_tree.py
    python NonLinearPB/NonLinearPBExperiments/scripts/plot_hier_tree.py --samples 4 --seed 1 --output /path/to/dir

Presentation figure from the current renderer; no dataset is written.
Internal nodes show the pixel average of the clean leaf images below them; each
leaf shows its clean image and, underneath, random samples with nuisance
variation (shape jitter, brightness, pixel noise).
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_synthetic.render import COARSE_NAMES, FINE_NAMES, MID_NAMES, render, sample_nuisance

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "files/results/hier_synthetic_pilot"
BRANCH_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")    # one per coarse family
INK, MUTED, EDGE = "#222222", "#6b6b6b", "#9a9a9a"

# CONFIG: layout in inches.
LEFT, RIGHT, TOP, BOTTOM = 1.5, 0.3, 0.3, 0.5
COL, MID_GAP, COARSE_GAP = 0.62, 0.12, 0.4
LEAF_IMG, MID_IMG, COARSE_IMG = 0.52, 0.85, 1.1


def leaf_positions():
    """x centre (inches) of each of the 27 leaves, with gaps between groups."""
    xs, x = [], LEFT
    for c in range(3):
        for m in range(3):
            for f in range(3):
                xs.append(x + COL / 2)
                x += COL
            x += MID_GAP if m < 2 else 0
        x += COARSE_GAP if c < 2 else 0
    return np.array(xs).reshape(3, 3, 3), x + RIGHT


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=3, help="nuisance samples shown under each leaf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    clean = np.array([[[render(c, m, f) for f in range(3)] for m in range(3)] for c in range(3)])
    samples = [[[[render(c, m, f, sample_nuisance(rng)) for _ in range(args.samples)]
                 for f in range(3)] for m in range(3)] for c in range(3)]

    xs, width = leaf_positions()
    # Row centres from the top, in inches.
    y_root = TOP + 0.3
    y_coarse = y_root + 0.55 + COARSE_IMG / 2
    y_mid = y_coarse + COARSE_IMG / 2 + 0.75 + MID_IMG / 2
    y_leaf = y_mid + MID_IMG / 2 + 0.8 + LEAF_IMG / 2
    y_samples = [y_leaf + LEAF_IMG / 2 + 0.3 + LEAF_IMG / 2 + i * (LEAF_IMG + 0.06) for i in range(args.samples)]
    height = y_samples[-1] + LEAF_IMG / 2 + BOTTOM

    fig = plt.figure(figsize=(width, height))

    def fx(x):
        return x / width

    def fy(y):
        return 1 - y / height

    def image(img, x, y, size, color):
        ax = fig.add_axes([fx(x - size / 2), fy(y + size / 2), size / width, size / height])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    def text(x, y, s, **kw):
        kw = {"ha": "center", "va": "center", "color": INK, "fontsize": 8, **kw}
        fig.text(fx(x), fy(y), s, **kw)

    def elbow(x0, y0, children_x, y1):
        """Parent bottom (x0, y0) to each child top at y1, via a horizontal bus."""
        ybus = (y0 + y1) / 2
        lines = [((x0, x0), (y0, ybus)), ((min(children_x), max(children_x)), (ybus, ybus))]
        lines += [((cx, cx), (ybus, y1)) for cx in children_x]
        for (xa, xb), (ya, yb) in lines:
            fig.add_artist(Line2D([fx(xa), fx(xb)], [fy(ya), fy(yb)], color=EDGE, lw=1.0))

    # Root.
    x_root = xs.mean()
    text(x_root, y_root, "All images  (27 leaves, labels hidden from the model)", fontsize=11,
         fontweight="bold", bbox={"boxstyle": "round,pad=0.4", "fc": "#f2f2f2", "ec": EDGE})
    elbow(x_root, y_root + 0.2, xs.mean(axis=(1, 2)), y_coarse - COARSE_IMG / 2 - 0.22)

    for c in range(3):
        color = BRANCH_COLORS[c]
        xc = xs[c].mean()
        text(xc, y_coarse - COARSE_IMG / 2 - 0.12, f"c={c}: {COARSE_NAMES[c]}", fontsize=10, fontweight="bold")
        image(clean[c].mean(axis=(0, 1)), xc, y_coarse, COARSE_IMG, color)
        elbow(xc, y_coarse + COARSE_IMG / 2, xs[c].mean(axis=1), y_mid - MID_IMG / 2 - 0.2)
        for m in range(3):
            xm = xs[c, m].mean()
            text(xm, y_mid - MID_IMG / 2 - 0.1, f"m={m}: {MID_NAMES[c][m]}", fontsize=8.5)
            image(clean[c, m].mean(axis=0), xm, y_mid, MID_IMG, color)
            elbow(xm, y_mid + MID_IMG / 2, xs[c, m], y_leaf - LEAF_IMG / 2 - 0.2)
            for f in range(3):
                xf = xs[c, m, f]
                text(xf, y_leaf - LEAF_IMG / 2 - 0.13, f"f={f}\n{FINE_NAMES[c][m][f]}", fontsize=6.5, linespacing=1.0)
                image(clean[c, m, f], xf, y_leaf, LEAF_IMG, color)
                for y, img in zip(y_samples, samples[c][m][f]):
                    image(img, xf, y, LEAF_IMG, EDGE)

    # Level labels down the left margin, and a bracket around the samples.
    for y, s in ((y_root, "Root"), (y_coarse, "Coarse\n3 families"), (y_mid, "Mid\n9 refinements"),
                 (y_leaf, "Fine\n27 leaves")):
        text(0.15, y, s, ha="left", fontsize=10, fontweight="bold")
    y_s = (y_samples[0] + y_samples[-1]) / 2
    text(0.15, y_s, f"Samples\n{args.samples} per leaf\n(small shape\njitter, brightness,\npixel noise)",
         ha="left", fontsize=9, color=MUTED)
    fig.add_artist(Rectangle((fx(LEFT - 0.08), fy(y_samples[-1] + LEAF_IMG / 2 + 0.06)),
                             (xs.max() - LEFT + COL / 2 + 0.16) / width,
                             (y_samples[-1] - y_samples[0] + LEAF_IMG + 0.12) / height,
                             fill=False, ec=EDGE, lw=0.8, ls="--"))
    text(LEFT, height - BOTTOM / 2,
         "Coarse and mid nodes show the pixel average of the leaves below them. "
         "Fine details are attached to their parent's geometry, so f=0 means a different detail in each family. "
         "Sizes tuned so every coarse change costs more pixel MSE than every mid change, and every mid more than every fine.",
         ha="left", fontsize=8, color=MUTED)

    out = args.output or OUTPUT_ROOT / f"tree_{datetime.now():%Y_%m_%d__%H_%M_%S}"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"hierarchy_tree.{ext}", dpi=200)
    print(out / "hierarchy_tree.png")


if __name__ == "__main__":
    main()
