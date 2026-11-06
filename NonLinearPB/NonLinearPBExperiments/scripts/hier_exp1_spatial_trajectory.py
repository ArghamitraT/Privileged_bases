"""Spatial construction trajectory: where the reconstruction brightens most at each prefix step.

Conda environment: mrl_env_cuda12
Reads figures/prefix_change/data/<img>_e<ep>.npz (recon [model(ae, mrl), seed, k = 0..10, 64, 64], x).
For each model and the chosen seed(s), steps k = 2..10 (1:(k-1) -> 1:k), raw decoder output:
    D+_k(p) = max(0, xhat_{1:k}(p) - xhat_{1:k-1}(p))
    A_k     = sum_p D+_k(p)
    P_k     = smallest pixel set with sum D+_k >= SHARE * A_k (pixels ranked by D+ descending, ties by
              pixel index ascending); empty if A_k = 0
A step is marked "nearly inactive" if A_k < INACTIVE_FRAC * max_k A_k for that model and seed.
Figures: (A) spatial_trajectory_ae_mrl: per model, the first step at which each pixel is selected (colour =
step) and how many steps select it; (B) dominant_change_grid: rows = models, columns = steps, selected pixels
over a faint input, with A_k and |P_k| under each tile.
Outputs in figures/prefix_change/spatial_trajectory/<img>_e<ep>_seed<s>/ (png + svg, npz, csv, json); with several
--seeds also <img>_e<ep>_seeds/dominant_change_grid_<model>.png/.svg (rows = seeds).

Usage: python scripts/hier_exp1_spatial_trajectory.py [--run R] [--image c1m0f0] [--epoch 2] [--seeds 0]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_03__15_08_36"))
_ap.add_argument("--image", default="c1m0f0")
_ap.add_argument("--epoch", type=int, default=2)
_ap.add_argument("--seeds", type=int, nargs="+", default=[0])
_args = _ap.parse_args()
SRC = _args.run / "figures" / "prefix_change"
MODELS = ("ae", "mrl")                         # order inside the npz
SHOW = ("mrl", "ae")
SHARE = 0.8
INACTIVE_FRAC = 0.05
STEPS = list(range(2, 11))
STEP_LBL = [f"{k-1}→{k}" for k in STEPS]
STEP_COLORS = [matplotlib.colormaps["turbo"](v) for v in np.linspace(0.08, 0.92, len(STEPS))]   # blue (early) -> red (late)
NAMES = {"c1m0f0": "ring + stem / stem through", "c1m2f0": "ring + stem / stem slanted",
         "c0m2f0": "dumbbell / bowed", "c2m2f0": "boomerang / stepped"}


def dominant_set(d, share=SHARE):
    """d [4096] >= 0 -> bool mask of the smallest set holding >= share of the total (ties: lower index first)."""
    total = d.sum()
    mask = np.zeros(d.size, bool)
    if total <= 0:
        return mask
    order = np.lexsort((np.arange(d.size), -d))                     # by value desc, then index asc
    n = int(np.searchsorted(np.cumsum(d[order]), share * total) + 1)
    mask[order[:n]] = True
    return mask


def faint(ax, x):
    ax.imshow(x, cmap="gray", vmin=0, vmax=1, alpha=0.25, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    d = np.load(SRC / "data" / f"{_args.image}_e{_args.epoch:03d}.npz")
    rec, x = d["recon"], d["x"]
    meta_img = json.loads((SRC / "metadata.json").read_text())["images"][_args.image]
    for s in _args.seeds:
        out = SRC / "spatial_trajectory" / f"{_args.image}_e{_args.epoch:03d}_seed{s}"
        out.mkdir(parents=True, exist_ok=True)
        Dp = {m: np.maximum(0, rec[MODELS.index(m), s, 2:] - rec[MODELS.index(m), s, 1:-1]).reshape(9, -1) for m in SHOW}
        A = {m: Dp[m].sum(1) for m in SHOW}
        P = {m: np.stack([dominant_set(Dp[m][j]) for j in range(9)]) for m in SHOW}
        inactive = {m: A[m] < INACTIVE_FRAC * A[m].max() for m in SHOW}

        # (A) spatial trajectory: first selecting step + number of selecting steps
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.2), gridspec_kw={"width_ratios": [1, 1, 1]}, layout="constrained")
        cmap = ListedColormap(STEP_COLORS)
        for r, m in enumerate(SHOW):
            first = np.full(4096, -1)
            for j in range(9):
                first[(first < 0) & P[m][j]] = j
            count = P[m].sum(0)
            axes[r, 0].imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="none")
            axes[r, 0].set_title(f"{m.upper()} — input", fontsize=10); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            faint(axes[r, 1], x)
            im1 = axes[r, 1].imshow(np.ma.masked_less(first.reshape(64, 64), 0), cmap=cmap, vmin=-0.5, vmax=8.5, interpolation="none")
            axes[r, 1].set_title(f"{m.upper()} seed {s} — first step that selects the pixel", fontsize=10)
            faint(axes[r, 2], x)
            im2 = axes[r, 2].imshow(np.ma.masked_equal(count.reshape(64, 64), 0), cmap="viridis", vmin=1, vmax=max(3, count.max()), interpolation="none")
            axes[r, 2].set_title(f"{m.upper()} seed {s} — number of steps that select it", fontsize=10)
            cb = fig.colorbar(im2, ax=axes[r, 2], shrink=0.8)
            cb.set_label("steps", fontsize=8)
        cb = fig.colorbar(im1, ax=axes[:, 1], ticks=range(9), shrink=0.6, location="bottom")
        cb.ax.set_xticklabels(STEP_LBL, fontsize=8, rotation=45)
        cb.set_label("prefix step (blue = early, red = late)", fontsize=9)
        fig.suptitle(f"Spatial construction — {NAMES.get(_args.image)}, epoch {_args.epoch}, seed {s}: pixels giving "
                     f"{SHARE:.0%} of each step's brightening", fontsize=11)
        fig.savefig(out / "spatial_trajectory_ae_mrl.png", dpi=130); fig.savefig(out / "spatial_trajectory_ae_mrl.svg")
        plt.close(fig)

        # (B) step-by-step grid
        fig, axes = plt.subplots(2, 10, figsize=(17, 4.6), layout="constrained")
        for r, m in enumerate(SHOW):
            axes[r, 0].imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="none")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(f"{m.upper()}\nseed {s}", fontsize=10)
            axes[r, 0].set_title("input", fontsize=9)
            for j in range(9):
                ax = axes[r, j + 1]
                faint(ax, x)
                rgba = np.zeros((4096, 4)); rgba[P[m][j]] = STEP_COLORS[j]
                ax.imshow(rgba.reshape(64, 64, 4), interpolation="none")
                ax.set_title(STEP_LBL[j], fontsize=9, color="black" if not inactive[m][j] else "0.55")
                ax.set_xlabel(f"A {A[m][j]:.2f} · {P[m][j].sum()} px" + ("\nnearly inactive" if inactive[m][j] else ""),
                              fontsize=7.5, color="0.45" if inactive[m][j] else "black")
                if inactive[m][j]:
                    for sp in ax.spines.values():
                        sp.set_linestyle("--"); sp.set_color("0.6")
        fig.suptitle(f"Dominant-change pixels per step ({SHARE:.0%} of the brightening) — {NAMES.get(_args.image)}, "
                     f"epoch {_args.epoch}, seed {s}. A = total brightening (sum of positive pixel changes); "
                     f"dashed = nearly inactive (A < {INACTIVE_FRAC:.0%} of the row's largest A)", fontsize=10)
        fig.savefig(out / "dominant_change_grid.png", dpi=130); fig.savefig(out / "dominant_change_grid.svg")
        plt.close(fig)

        np.savez_compressed(out / "positive_change_maps.npz", **{m: Dp[m].reshape(9, 64, 64) for m in SHOW},
                            steps=np.array(STEPS), x=x)
        np.savez_compressed(out / "dominant_pixel_masks.npz", **{m: P[m].reshape(9, 64, 64) for m in SHOW}, steps=np.array(STEPS))
        with open(out / "change_summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "seed", "epoch", "image", "test_index", "step", "total_positive_change", "n_selected", "nearly_inactive"])
            for m in SHOW:
                for j, k in enumerate(STEPS):
                    w.writerow([m, s, _args.epoch, _args.image, meta_img["test_index"], f"1:{k-1}->1:{k}",
                                round(float(A[m][j]), 5), int(P[m][j].sum()), bool(inactive[m][j])])
        (out / "analysis_config.json").write_text(json.dumps({
            "image": _args.image, "name": NAMES.get(_args.image), "test_index": meta_img["test_index"],
            "epoch": _args.epoch, "seed": s, "share": SHARE, "tie_rule": "value descending, then pixel index ascending",
            "inactive_rule": f"A_k < {INACTIVE_FRAC} x max_k A_k (same model and seed)",
            "reconstructions": "raw decoder output, prefixes 1:1..1:10, rest of code zero (from figures/prefix_change/data)",
            "checkpoints": f"checkpoints/<ae|mrl>/seed_{s}/epochs/epoch_{_args.epoch:03d}.pt",
            "step_colors": "turbo, blue (1->2) to red (9->10)"}, indent=1) + "\n")
        for m in SHOW:
            print(m, "A:", A[m].round(2).tolist(), "| |P|:", P[m].sum(1).tolist(), "| inactive:", np.flatnonzero(inactive[m]).tolist())
        print(out)
    if len(_args.seeds) > 1:
        seed_grids(rec, x)


def seed_grids(rec, x):
    """One figure per model: rows = seeds, columns = input + steps 1->2 ... 9->10 (same rules as the per-seed grid)."""
    out = SRC / "spatial_trajectory" / f"{_args.image}_e{_args.epoch:03d}_seeds"
    out.mkdir(parents=True, exist_ok=True)
    for m in SHOW:
        mi = MODELS.index(m)
        n = len(_args.seeds)
        fig, axes = plt.subplots(n, 10, figsize=(17, 2.25 * n + 0.6), layout="constrained")
        for r, s in enumerate(_args.seeds):
            Dp = np.maximum(0, rec[mi, s, 2:] - rec[mi, s, 1:-1]).reshape(9, -1)
            A = Dp.sum(1)
            P = np.stack([dominant_set(Dp[j]) for j in range(9)])
            inactive = A < INACTIVE_FRAC * A.max()
            axes[r, 0].imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="none")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(f"seed {s}", fontsize=10)
            if r == 0:
                axes[r, 0].set_title("input", fontsize=9)
            for j in range(9):
                ax = axes[r, j + 1]
                faint(ax, x)
                rgba = np.zeros((4096, 4)); rgba[P[j]] = STEP_COLORS[j]
                ax.imshow(rgba.reshape(64, 64, 4), interpolation="none")
                if r == 0:
                    ax.set_title(STEP_LBL[j], fontsize=9)
                ax.set_xlabel(f"A {A[j]:.2f} · {P[j].sum()} px" + (" · inactive" if inactive[j] else ""),
                              fontsize=7.5, color="0.45" if inactive[j] else "black")
                if inactive[j]:
                    for sp in ax.spines.values():
                        sp.set_linestyle("--"); sp.set_color("0.6")
        fig.suptitle(f"{m.upper()} — dominant-change pixels per step ({SHARE:.0%} of the brightening), {NAMES.get(_args.image)}, "
                     f"epoch {_args.epoch}. A = total brightening; dashed = nearly inactive (A < {INACTIVE_FRAC:.0%} of that seed's largest A)",
                     fontsize=10)
        fig.savefig(out / f"dominant_change_grid_{m}.png", dpi=130); fig.savefig(out / f"dominant_change_grid_{m}.svg")
        plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
