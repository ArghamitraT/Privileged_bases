"""Noise-sweep figures for Experiment 1 (seed 0).

1. prefix_recon_tree_coarse0_seed0.png: 13 rows (c0 subtree: coarse, 3 mids, 9 leaves), columns
   input, z1, z1:2 ... z1:10; AE (left) vs MRL (right) from one sigma folder. Same image picks as
   the final run's figure (rng seed 0).
2. z1_codebook_path_m0f0.png: z1 of all test images, one panel per model (no-noise MRL from the
   final run for reference, then the sigma folder's MRL and AE). Colours: 3 per coarse family
   (dark = c (m1, m2), light = c m0 (f1, f2), lightest = c m0 f0).
Also prints z1 numbers: SD, nearest-leaf-mean accuracy, coarse runs along z1 (3 = contiguous).

Usage: python scripts/hier_exp1_noise_figs.py <sigma_folder> [--final <final_run_folder>]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from hier_exp1.data import load_split
from hier_exp1_train import build_model

SEED, COARSE = 0, 0
NAMES = ["dumbbell", "ring", "hinge"]
PAL = {0: ["#1b7a3e", "#5cc27f", "#b5ebc6"],
       1: ["#a33c0c", "#f08a4b", "#fdd0b1"],
       2: ["#1f4e9c", "#5f95e0", "#b9d3f7"]}


def load(path):
    m = build_model(SEED)
    m.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    return m.cuda().eval()


def recon_tree(run, label, imgs, lab, out):
    rng = np.random.default_rng(0)
    c, m, f = lab["coarse"], lab["mid"], lab["fine"]
    rows, names = [], []
    i = int(rng.choice(np.flatnonzero(c == COARSE))); rows.append(i)
    names.append(f"COARSE c{COARSE}\n(random: m{m[i]} f{f[i]})")
    for mm in range(3):
        i = int(rng.choice(np.flatnonzero((c == COARSE) & (m == mm)))); rows.append(i)
        names.append(f"MID c{COARSE} m{mm}\n(random: f{f[i]})")
        for ff in range(3):
            i = int(rng.choice(np.flatnonzero((c == COARSE) & (m == mm) & (f == ff)))); rows.append(i)
            names.append(f"LEAF c{COARSE} m{mm} f{ff}")
    x = torch.as_tensor(imgs[rows]).unsqueeze(1).float().cuda() / 255
    dec = {}
    for obj in ("ae", "mrl"):
        mdl = load(run / f"checkpoints/{obj}/seed_{SEED}/best.pt")
        with torch.no_grad():
            z = mdl.encode(x)
            dec[obj] = [mdl.decode(torch.cat([z[:, :k], torch.zeros_like(z[:, k:])], 1))
                        .clamp(0, 1).cpu().numpy()[:, 0] for k in range(1, 11)]
    n = len(rows)
    fig, axes = plt.subplots(n, 23, figsize=(23 * 0.95, n * 1.0))
    for r in range(n):
        inp = x[r, 0].cpu().numpy()
        cols = [inp] + [o[r] for o in dec["ae"]] + [None] + [inp] + [o[r] for o in dec["mrl"]]
        kind = names[r].split()[0]
        ec = {"COARSE": "#d62728", "MID": "#1f77b4", "LEAF": "#999999"}[kind]
        for j, im in enumerate(cols):
            a = axes[r, j]; a.set_xticks([]); a.set_yticks([])
            if im is None:
                a.axis("off"); continue
            a.imshow(im, cmap="gray", vmin=0, vmax=1)
            for s in a.spines.values():
                s.set_edgecolor(ec); s.set_linewidth(2 if kind != "LEAF" else 0.5)
        axes[r, 0].text(-6, 32, names[r], ha="right", va="center", fontsize=8.5,
                        weight="normal" if kind == "LEAF" else "bold")
    heads = ["input", "z1"] + [f"z1:{k}" for k in range(2, 11)]
    for j, h in enumerate(heads):
        axes[0, j].set_title(h, fontsize=9); axes[0, j + 12].set_title(h, fontsize=9)
    fig.text(0.29, 0.997, f"Standard AE, {label} (seed {SEED})", ha="center", va="top", fontsize=13, weight="bold")
    fig.text(0.76, 0.997, f"MRL-AE, {label} (seed {SEED})", ha="center", va="top", fontsize=13, weight="bold")
    plt.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.005, wspace=0.05, hspace=0.1)
    p = out / f"prefix_recon_tree_coarse{COARSE}_seed{SEED}.png"
    fig.savefig(p, dpi=110); plt.close(fig); print(p)


def z1_panels(panels, imgs, lab, out):
    x = torch.as_tensor(imgs).unsqueeze(1).float().cuda() / 255
    c, mm, ff, leaf = lab["coarse"], lab["mid"], lab["fine"], lab["leaf_id"]
    lvl = np.where(mm != 0, 0, np.where(ff != 0, 1, 2))
    col = np.array([PAL[a][b] for a, b in zip(c, lvl)])
    leaf_coarse = np.array([c[leaf == l][0] for l in range(27)])
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.9 * len(panels)))
    print(f"{'model':<28}{'z1 SD':>8}{'nearest-mean acc':>18}{'coarse runs':>13}  order of coarse along z1")
    for ax, (title, path) in zip(axes, panels):
        mdl = load(path)
        with torch.no_grad():
            z1 = torch.cat([mdl.encode(x[i:i + 512])[:, 0] for i in range(0, len(x), 512)]).cpu().numpy()
        means = np.array([z1[leaf == l].mean() for l in range(27)])
        order = np.argsort(means)
        rank = np.empty(27, int); rank[order] = np.arange(27)
        nn_acc = (np.abs(z1[:, None] - means[None]).argmin(1) == leaf).mean()
        seq = leaf_coarse[order]
        runs = 1 + int((seq[1:] != seq[:-1]).sum())
        print(f"{title:<28}{z1.std():>8.2f}{nn_acc:>18.3f}{runs:>13}  {''.join(map(str, seq))}")
        y = rank[leaf] + np.random.default_rng(0).uniform(-.3, .3, len(leaf))
        ax.set_facecolor("#ececec")
        ax.scatter(z1, y, s=4, c=col, rasterized=True)
        ax.set_title(f"{title}: z1 of {len(z1):,} test images  (coarse runs along z1: {runs}; 3 = contiguous)",
                     fontsize=10, loc="left")
        ax.set_yticks([]); ax.set_ylabel("leaf (sorted by mean z1)", fontsize=8); ax.grid(axis="x", color="white")
    h = []
    for a in range(3):
        h += [Line2D([], [], marker="o", ls="", color=PAL[a][0], label=f"c{a} {NAMES[a]} (m1, m2)"),
              Line2D([], [], marker="o", ls="", color=PAL[a][1], label=f"c{a} m0 (f1, f2)"),
              Line2D([], [], marker="o", ls="", color=PAL[a][2], label=f"c{a} m0 f0")]
    fig.legend(handles=h, ncol=3, fontsize=8.5, loc="lower center", frameon=True, facecolor="#ececec",
               bbox_to_anchor=(0.5, 0.0))
    axes[-1].set_xlabel("z1")
    plt.tight_layout(rect=(0, 0.1 if len(panels) <= 3 else 0.07, 1, 1))
    p = out / "z1_codebook_path_m0f0.png"
    fig.savefig(p, dpi=110); plt.close(fig); print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sigma_folder", type=Path)
    ap.add_argument("--final", type=Path,
                    default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_02__19_43_27"))
    a = ap.parse_args()
    run = a.sigma_folder
    sig = run.name.replace("sigma_", "")
    out = run / "figures"; out.mkdir(exist_ok=True)
    imgs, lab = load_split("test")
    label = f"noise sigma {sig}"
    recon_tree(run, label, imgs, lab, out)
    panels = [("MRL, no noise (final run)", a.final / f"checkpoints/mrl/seed_{SEED}/best.pt"),
              (f"MRL, sigma {sig}", run / f"checkpoints/mrl/seed_{SEED}/best.pt")]
    if (run / f"checkpoints/ae/seed_{SEED}/best.pt").exists():
        panels.append((f"AE, sigma {sig}", run / f"checkpoints/ae/seed_{SEED}/best.pt"))
    z1_panels(panels, imgs, lab, out)


if __name__ == "__main__":
    main()
