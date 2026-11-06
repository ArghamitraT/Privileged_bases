"""Reconstruction trajectories across prefixes in one shared 2-D PCA (one image, one epoch).

Conda environment: mrl_env_cuda12
Reads figures/prefix_change/data/<img>_e<ep>.npz (recon [model(ae, mrl), seed, k = 0..10, 64, 64], x).
Uses xhat_{1:k}, k = 1..10 (raw decoder output, not clipped), 2 models x 5 seeds x 10 = 100 vectors of 4096.
One PCA (2 components, centred) fitted on those 100 vectors only; the input x is projected with the same
transform. Figure: AE | MRL, same axes and limits, equal aspect, plus a third panel: MRL zoomed (same PCA,
window around the MRL points and the input; dashed box on the MRL panel); one colour per seed (same in both panels);
arrows 1:(k-1) -> 1:k; start (k = 1) open circle, end (k = 10) filled square; x = black star.
Outputs in figures/prefix_change/pca_trajectory/: <img>_e<ep>.png/.svg and <img>_e<ep>.npz (coords, PCA
components/mean/explained variance, the 100 reconstructions) + .json (ids, epoch, checkpoints).

Usage: python scripts/hier_exp1_prefix_pca.py [--run R] [--image c1m0f0] [--epoch 2]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_03__15_08_36"))
_ap.add_argument("--image", default="c1m0f0")
_ap.add_argument("--epoch", type=int, default=2)
_args = _ap.parse_args()
SRC = _args.run / "figures" / "prefix_change"
OUT = SRC / "pca_trajectory"
MODELS = ("ae", "mrl")                        # order inside the npz and in the figure
SEED_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")   # Okabe-Ito, CVD-safe
LABEL_K = (2, 3, 5)
NAMES = {"c1m0f0": "ring + stem / stem through", "c1m2f0": "ring + stem / stem slanted",
         "c0m2f0": "dumbbell / bowed", "c2m2f0": "boomerang / stepped"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SRC / "data" / f"{_args.image}_e{_args.epoch:03d}.npz")
    rec = d["recon"][:, :, 1:]                                   # [2, 5, 10, 64, 64], prefixes 1:1 .. 1:10
    x = d["x"]
    R = rec.reshape(-1, 64 * 64).astype(np.float64)              # 100 x 4096, order (model, seed, k)
    mean = R.mean(0)
    _, sv, vt = np.linalg.svd(R - mean, full_matrices=False)
    comps = vt[:2]
    evr = (sv ** 2 / (sv ** 2).sum())[:2]
    U = ((R - mean) @ comps.T).reshape(2, 5, 10, 2)
    ux = (x.ravel() - mean) @ comps.T

    allpts = np.concatenate([U.reshape(-1, 2), ux[None]])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.06 * (hi - lo)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), layout="constrained")
    # panels: AE, MRL (identical axes), MRL zoomed (same PCA, window around MRL points + input)
    for ai, mi in enumerate((0, 1, 1)):
        m = MODELS[mi]
        ax = axes[ai]
        zoom = ai == 2
        for s in range(5):
            c, p = SEED_COLORS[s], U[mi, s]
            ax.plot(p[:, 0], p[:, 1], color=c, lw=1.2, alpha=0.55, zorder=2)
            for k in range(9):
                ax.annotate("", xy=p[k + 1], xytext=p[k],
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, mutation_scale=11, shrinkA=3, shrinkB=3), zorder=3)
            ax.scatter(*p[0], s=60, facecolor="white", edgecolor=c, lw=2, zorder=4)
            ax.scatter(*p[-1], s=55, marker="s", color=c, edgecolor="white", lw=1, zorder=4)
            ax.scatter(p[1:-1, 0], p[1:-1, 1], s=12, color=c, zorder=4)
            ax.annotate("1", p[0], xytext=(5, 5), textcoords="offset points", fontsize=8, color=c, weight="bold")
            ax.annotate("10", p[-1], xytext=(5, -10), textcoords="offset points", fontsize=8, color=c, weight="bold")
            for k in LABEL_K:
                ax.annotate(str(k), p[k - 1], xytext=(4, 3), textcoords="offset points", fontsize=7, color=c)
        ax.scatter(*ux, marker="*", s=260, color="black", zorder=5, label="input image")
        ax.set_title(f"{m.upper()} — zoomed (same PCA)" if zoom else m.upper(), fontsize=12, weight="bold")
        ax.set_xlabel(f"PC1 ({evr[0]:.1%} of variance)")
        ax.set_aspect("equal")
        if zoom:
            zp = np.concatenate([U[1].reshape(-1, 2), ux[None]])
            zlo, zhi = zp.min(0), zp.max(0)
            zc, zr = (zlo + zhi) / 2, (zhi - zlo).max() / 2 * 1.12
            ax.set_xlim(zc[0] - zr, zc[0] + zr); ax.set_ylim(zc[1] - zr, zc[1] + zr)
            ax.set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
            # show the zoom window on the full MRL panel
            axes[1].add_patch(plt.Rectangle((zc[0] - zr, zc[1] - zr), 2 * zr, 2 * zr, fill=False, ls="--", lw=1, color="0.4", zorder=1))
        else:
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
    handles = [plt.Line2D([], [], color=SEED_COLORS[s], lw=2, label=f"seed {s}") for s in range(5)]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", mew=2, label="start (1:1)"),
                plt.Line2D([], [], marker="s", ls="", color="0.3", label="end (1:10)"),
                plt.Line2D([], [], marker="*", ls="", color="black", ms=13, label="input image")]
    fig.legend(handles=handles, loc="outside lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle(f"Prefix reconstructions 1:1 → 1:10 in one shared PCA — {NAMES.get(_args.image, _args.image)}, "
                 f"epoch {_args.epoch}\nPCA fitted on all 100 AE + MRL reconstructions; PC1 + PC2 = {evr.sum():.1%} of their variance", fontsize=11)
    tag = f"{_args.image}_e{_args.epoch:03d}"
    fig.savefig(OUT / f"{tag}.png", dpi=130)
    fig.savefig(OUT / f"{tag}.svg")
    plt.close(fig)
    np.savez_compressed(OUT / f"{tag}.npz", coords=U, input_coords=ux, components=comps, mean=mean,
                        explained_variance_ratio=evr, recon=rec, x=x, models=np.array(MODELS))
    meta = {"image": _args.image, "name": NAMES.get(_args.image), "epoch": _args.epoch, "run": str(_args.run),
            "test_index": json.loads((SRC / "metadata.json").read_text())["images"][_args.image]["test_index"],
            "checkpoints": f"checkpoints/<ae|mrl>/seed_<0-4>/epochs/epoch_{_args.epoch:03d}.pt",
            "reconstructions": "raw decoder output (not clipped), prefixes 1:1..1:10, rest of code zero",
            "pca": "2 components, centred, fitted on the 100 reconstructions only; input projected after",
            "explained_variance_ratio": [round(float(v), 4) for v in evr],
            "coords_layout": "[model(ae, mrl), seed, k-1, PC]"}
    (OUT / f"{tag}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(OUT / f"{tag}.png", "EVR", evr.round(3))


if __name__ == "__main__":
    main()
