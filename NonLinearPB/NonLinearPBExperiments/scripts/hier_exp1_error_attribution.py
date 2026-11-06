"""Which pixels account for the drop in reconstruction error as the prefix grows (one image, one epoch).

Conda environment: mrl_env_cuda12
Reads figures/prefix_change/data/<img>_e<ep>.npz (recon [model(ae, mrl), seed, k = 0..10, 64, 64], x).
    err_k        = mean_p (xhat_{1:k}(p) - x(p))^2            (k = 0: decode of the all-zero code)
    R_k(p)       = (xhat_{1:k-1}(p) - x(p))^2 - (xhat_{1:k}(p) - x(p))^2,   k = 1..10
    sum_p R_k(p) / 4096 = err_{k-1} - err_k exactly, so R_k splits the error drop of step k over pixels:
    R_k(p) > 0: pixel p got closer to the input; R_k(p) < 0: it got further away.
Per tile: dMSE = err_{k-1} - err_k, and n90 = the fewest pixels whose R_k adds up to 90% of the positive part.

Figure (zero-code start left out, user 2026-11-04): (A) err_k for k = 1..10, MRL and AE, 5 seeds (thin) + mean;
(B, C) per model, for one seed: input + reconstructions xhat_{1:k} (k = 1..10), and under xhat_{1:k} the R_k map
for step 1:(k-1) -> 1:k (k = 2..10) (BrBG: teal = error reduced, brown = error
increased, one symmetric limit per tile, printed). Output in figures/prefix_change/error_attribution/.

Usage: python scripts/hier_exp1_error_attribution.py [--run R] [--image c1m0f0] [--epoch 5] [--seed 0]
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
_ap.add_argument("--epoch", type=int, default=5)
_ap.add_argument("--seed", type=int, default=0)
_args = _ap.parse_args()
SRC = _args.run / "figures" / "prefix_change"
OUT = SRC / "error_attribution"
MODELS = ("ae", "mrl")           # order inside the npz
SHOW = ("mrl", "ae")             # order in the figure
COLORS = {"mrl": "#7a3fc2", "ae": "#0f8a6a"}
CMAP = "BrBG"                    # brown = error increased, teal = error reduced, white = 0
NAMES = {"c1m0f0": "ring + stem / stem through", "c1m2f0": "ring + stem / stem slanted",
         "c0m2f0": "dumbbell / bowed", "c2m2f0": "boomerang / stepped"}


def n_for_share(r, share=0.9):
    pos = np.sort(r[r > 0].ravel())[::-1]
    if pos.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(pos), share * pos.sum()) + 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SRC / "data" / f"{_args.image}_e{_args.epoch:03d}.npz")
    rec, x = d["recon"], d["x"]                                     # [2, 5, 11, 64, 64]
    sq = (rec - x) ** 2
    err = sq.mean(axis=(3, 4))                                      # [2, 5, 11]
    R = sq[:, :, :-1] - sq[:, :, 1:]                                # [2, 5, 10, 64, 64], step k = 1..10
    assert np.allclose(R.mean(axis=(3, 4)), err[:, :, :-1] - err[:, :, 1:], atol=1e-6)

    fig = plt.figure(figsize=(15, 12.6))
    gs = fig.add_gridspec(7, 11, height_ratios=[1.5, 0.42, 1, 1, 0.42, 1, 1], hspace=0.55, wspace=0.08,
                          left=0.05, right=0.99, top=0.96, bottom=0.03)
    ax = fig.add_subplot(gs[0, 2:9])                                # small error plot, centred
    ks = np.arange(1, 11)                                           # from 1:1 (the zero-code start is left out)
    for m in SHOW:
        mi = MODELS.index(m)
        for s in range(err.shape[1]):
            ax.plot(ks, err[mi, s, 1:] * 1e3, color=COLORS[m], lw=0.8, alpha=0.35)
        ax.plot(ks, err[mi, :, 1:].mean(0) * 1e3, color=COLORS[m], lw=2.2, marker="o", ms=4, label=f"{m.upper()} (mean of 5 seeds; thin = seeds)")
    ax.set_xticks(ks, [f"1:{k}" for k in ks])
    ax.set_xlabel("prefix used for the reconstruction")
    ax.set_ylabel("MSE on this image (×10⁻³, log)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    ax.set_title(f"(A) Reconstruction error vs prefix — {NAMES.get(_args.image, _args.image)}, epoch {_args.epoch}", fontsize=10)

    s = _args.seed
    summary = {}
    for bi, m in enumerate(SHOW):
        mi = MODELS.index(m)
        r0 = 1 + 3 * bi
        hdr = fig.add_subplot(gs[r0, :]); hdr.axis("off")
        hdr.text(0, 0.1, f"({'B' if bi == 0 else 'C'}) {m.upper()} seed {s}: reconstructions (top row) and per-pixel error reduction R_k (bottom row). "
                 f"Teal = pixel got closer to the input, brown = further; each tile has its own scale (±limit under it). "
                 f"ΔMSE ×10⁻³; '90%' = fewest pixels that give 90% of the reduction.", fontsize=9, va="bottom", wrap=True)
        top = [fig.add_subplot(gs[r0 + 1, k]) for k in range(11)]
        bot = [fig.add_subplot(gs[r0 + 2, k]) for k in range(11)]
        top[0].imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="none")
        top[0].set_title("input", fontsize=8)
        for k in range(1, 11):
            top[k].imshow(np.clip(rec[mi, s, k], 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="none")
            top[k].set_title(f"x̂ 1:{k}\nMSE {err[mi, s, k] * 1e3:.1f}", fontsize=7.5)
        for a in bot[:2]:
            a.axis("off")                                           # no map under the input and x̂ 1:1
        rows = []
        for k in range(2, 11):                                      # step 1:(k-1) -> 1:k, under x̂ 1:k
            r = R[mi, s, k - 1]
            lim = float(np.abs(r).max()) or 1e-12
            bot[k].imshow(r, cmap=CMAP, vmin=-lim, vmax=lim, interpolation="none")
            dm = float((err[mi, s, k - 1] - err[mi, s, k]) * 1e3)
            n90 = n_for_share(r)
            bot[k].set_title(f"1:{k-1}→1:{k}\nΔMSE {dm:+.2f}", fontsize=7.5)
            bot[k].set_xlabel(f"±{lim:.3f} · 90%: {n90} px", fontsize=6.5)
            rows.append({"step": k, "dMSE_e-3": round(dm, 4), "tile_limit": round(lim, 5), "n90": n90,
                         "gain_e-3": round(float(r[r > 0].sum() / r.size * 1e3), 4),
                         "loss_e-3": round(float(-r[r < 0].sum() / r.size * 1e3), 4)})
        for a in top + bot[2:]:
            a.set_xticks([]); a.set_yticks([])
        top[0].set_ylabel("recon", fontsize=8); bot[2].set_ylabel("R_k", fontsize=8)
        summary[m] = {"seed": s, "steps": rows}
    tag = f"{_args.image}_e{_args.epoch:03d}_seed{s}"
    fig.savefig(OUT / f"{tag}.png", dpi=120)
    fig.savefig(OUT / f"{tag}.svg")     # vector text/lines; interpolation="none" embeds raw 64x64 pixels marked
                                         # image-rendering:pixelated by matplotlib, so zooming keeps sharp pixels
    plt.close(fig)
    np.savez_compressed(OUT / f"{_args.image}_e{_args.epoch:03d}.npz", R=R, err=err, models=np.array(MODELS))
    (OUT / f"{tag}.json").write_text(json.dumps({"image": _args.image, "epoch": _args.epoch,
                                                 "err_e-3": {m: (err[MODELS.index(m)] * 1e3).astype(float).round(3).tolist() for m in MODELS},
                                                 "seed_detail": summary}, indent=1) + "\n")
    print(OUT / f"{tag}.png")


if __name__ == "__main__":
    main()
