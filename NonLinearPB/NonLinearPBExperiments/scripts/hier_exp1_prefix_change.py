"""Hierarchical prefix-change analysis (plan: NonLinearPB/plans/hierarchical_Exp1_eval.md).

Conda environment: mrl_env_cuda12
For each selected test image x, epoch t, model a in {ae, mrl} and seed s (frozen weights):
    xhat_{1:k} = D([z_1..z_k, 0..0]),  xhat_{1:0} = D(0)
    D_k        = xhat_{1:k} - xhat_{1:(k-1)},  k = 1..10           (signed 64x64 maps)
References from TRAINING-set conditional means for the image's (c, m, f):
    T_C = G_C - G_0,  T_M = G_CM - G_C,  T_F = G_CMF - G_CM
Alignment A_{l,k} = cos(D_k, T_l), l in {C, M, F}; NaN only if a norm is exactly zero (no cut-off).

Outputs (in <run>/figures/prefix_change/):
    fig1/<img>_e<ep>.png   signed change maps, AE block and MRL block, 5 seeds x 10 prefix steps,
                           one symmetric colour limit per model (shared over its seeds and steps)
    fig2/<img>_e<ep>.png   T_C, T_M, T_F and the three alignment curves for seed FIG2_SEED (AE, MRL)
    fig3/<img>_e<ep>.png   alignment across seeds: faint seed lines, mean, min-max band; AE | MRL
    data/<img>_e<ep>.npz   D [model, seed, k, 64, 64], recon [model, seed, k=0..10, 64, 64], z, T, G
    alignment.csv          image, epoch, model, seed, ref, k, cos
    summary.csv            image, epoch, model, ref, k, mean, min, max, n_valid
    metadata.json          images (test index, c, m, f), epochs, colour limits, settings

Usage: python scripts/hier_exp1_prefix_change.py [--run R] [--epochs 2 3 4 5 10 15 20 25]
"""
import argparse
import csv
import json
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

from hier_exp1.data import load_split
from hier_exp1_train import build_model

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_03__15_08_36"))
_ap.add_argument("--epochs", type=int, nargs="+", default=[2, 3, 4, 5, 10, 15, 20, 25])
_ap.add_argument("--latent", type=int, default=10, help="size of z in the run (10; 5 for exprmnt_2026_11_05__11_44_34)")
_ap.add_argument("--data-only", action="store_true", help="write data/*.npz and the CSVs only, no fig1-3 (fig1-3 assume 10-d)")
_args = _ap.parse_args()
RUN, EPOCHS = _args.run, _args.epochs
OUT = RUN / "figures" / "prefix_change"

# CONFIG (plan "Settings", decided 2026-11-03)
IMAGES = {"c1m0f0": (1, 0, 0), "c1m2f0": (1, 2, 0), "c0m2f0": (0, 2, 0), "c2m2f0": (2, 2, 0)}
IMAGE_NAMES = {"c1m0f0": "ring + stem / stem through", "c1m2f0": "ring + stem / stem slanted",
               "c0m2f0": "dumbbell / bowed", "c2m2f0": "boomerang / stepped"}
MODELS = ("ae", "mrl")
SEEDS = range(5)
LATENT = _args.latent
FIG2_SEED = 0
REFS = ("C", "M", "F")
REF_COLORS = {"C": "#0072B2", "M": "#E69F00", "F": "#009E73"}     # Okabe-Ito, CVD-safe
STEP_LABELS = ["∅→1"] + [f"1:{k-1}→1:{k}" for k in range(2, LATENT + 1)]
CMAP = "RdBu"            # red = negative (darker), white = 0, blue = positive (brighter)


def references(train_x, lab, c, m, f):
    sel = lambda mask: train_x[mask].mean(0)
    g0 = train_x.mean(0)
    gc = sel(lab["coarse"] == c)
    gcm = sel((lab["coarse"] == c) & (lab["mid"] == m))
    gcmf = sel((lab["coarse"] == c) & (lab["mid"] == m) & (lab["fine"] == f))
    G = np.stack([g0, gc, gcm, gcmf])
    T = np.stack([gc - g0, gcm - gc, gcmf - gcm])
    return G, T


def prefix_recons(model, x):
    """x [1,1,64,64] -> z [10], recon [11, 64, 64] for prefixes 0..10 (0 = all-zero code)."""
    with torch.no_grad():
        z = model.encode(x)                                        # [1, 10]
        codes = torch.zeros(LATENT + 1, LATENT, device=x.device)
        for k in range(1, LATENT + 1):
            codes[k, :k] = z[0, :k]
        rec = model.decode(codes)[:, 0]                            # no clamp: exact decoder output
    return z[0].cpu().numpy(), rec.cpu().numpy()


def load(obj, seed, ep):
    model = build_model(seed, latent_dim=LATENT)
    path = RUN / "checkpoints" / obj / f"seed_{seed}" / "epochs" / f"epoch_{ep:03d}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.cuda().eval()


def cosine(d, t):
    nd, nt = np.linalg.norm(d), np.linalg.norm(t)
    return float("nan") if nd == 0 or nt == 0 else float(np.dot(d.ravel(), t.ravel()) / (nd * nt))


def fig1(path, x, D, lims, title):
    fig = plt.figure(figsize=(14.5, 13.8), layout="constrained")
    subs = fig.subfigures(2, 1, hspace=0.04)
    for mi, obj in enumerate(MODELS):
        sf = subs[mi]
        sf.suptitle(f"{obj.upper()}  (red = darker, blue = brighter)", fontsize=11, weight="bold")
        gs = sf.add_gridspec(5, 12, width_ratios=[1] * 11 + [0.08])
        for s in SEEDS:
            ax = sf.add_subplot(gs[s, 0])
            ax.imshow(x, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(f"seed {s}", fontsize=9)
            if s == 0:
                ax.set_title("input", fontsize=8)
            for k in range(LATENT):
                ax = sf.add_subplot(gs[s, k + 1])
                im = ax.imshow(D[mi, s, k], cmap=CMAP, vmin=-lims[obj], vmax=lims[obj])
                ax.set_xticks([]); ax.set_yticks([])
                if s == 0:
                    ax.set_title(STEP_LABELS[k], fontsize=8)
        cb = sf.colorbar(im, cax=sf.add_subplot(gs[:, 11]))
        cb.set_label(f"pixel change (±{lims[obj]:.3f})", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig2(path, x, T, A, title):
    """T [3,64,64]; A [model, seed, ref, k]."""
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], hspace=0.35, wspace=0.4,
                          width_ratios=[1, 1, 1, 1, 0.06])
    tl = float(np.abs(T).max())
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(x, cmap="gray", vmin=0, vmax=1); ax.set_title("input", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    names = {"C": "T_C = G_C − G_0", "M": "T_M = G_CM − G_C", "F": "T_F = G_CMF − G_CM"}
    for j, r in enumerate(REFS):
        ax = fig.add_subplot(gs[0, j + 1])
        im = ax.imshow(T[j], cmap=CMAP, vmin=-tl, vmax=tl)
        ax.set_title(names[r], fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 4]))
    cb.set_label(f"pixel change (±{tl:.3f})", fontsize=8); cb.ax.tick_params(labelsize=7)
    ks = np.arange(1, LATENT + 1)
    for mi, obj in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, mi * 2:mi * 2 + 2])
        for j, r in enumerate(REFS):
            ax.plot(ks, A[mi, FIG2_SEED, j], marker="o", ms=4, lw=2, color=REF_COLORS[r], label=f"T_{r}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylim(-1, 1); ax.set_xticks(ks); ax.set_xticklabels(STEP_LABELS, rotation=45, fontsize=7)
        ax.set_title(f"{obj.upper()} seed {FIG2_SEED}", fontsize=10)
        ax.grid(alpha=0.25)
        if mi == 0:
            ax.set_ylabel("cos(D_k, T)", fontsize=9)
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig3(path, A, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    ks = np.arange(1, LATENT + 1)
    for mi, obj in enumerate(MODELS):
        ax = axes[mi]
        for j, r in enumerate(REFS):
            a = A[mi, :, j]                                         # [seed, k]
            for s in SEEDS:
                ax.plot(ks, a[s], color=REF_COLORS[r], lw=0.8, alpha=0.3)
            with np.errstate(all="ignore"):
                ax.fill_between(ks, np.nanmin(a, 0), np.nanmax(a, 0), color=REF_COLORS[r], alpha=0.12, lw=0)
                ax.plot(ks, np.nanmean(a, 0), color=REF_COLORS[r], lw=2.2, marker="o", ms=4, label=f"T_{r}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylim(-1, 1); ax.set_xticks(ks); ax.set_xticklabels(STEP_LABELS, rotation=45, fontsize=7)
        ax.set_title(f"{obj.upper()}  (5 seeds: faint lines, mean, min–max band)", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cos(D_k, T)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    for d in ("fig1", "fig2", "fig3", "data"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    train_x, train_lab = load_split("train")
    train_x = train_x.astype(np.float32) / 255
    test_x, test_lab = load_split("test")
    picks, refs = {}, {}
    for name, (c, m, f) in IMAGES.items():
        idx = int(np.flatnonzero((test_lab["coarse"] == c) & (test_lab["mid"] == m) & (test_lab["fine"] == f))[0])
        picks[name] = idx
        refs[name] = references(train_x, train_lab, c, m, f)
    xs = {n: test_x[i].astype(np.float32) / 255 for n, i in picks.items()}

    rows, summary, lims_all = [], [], {}
    for ep in EPOCHS:
        # recon [image, model, seed, 11, 64, 64]; z [image, model, seed, 10]
        rec = np.zeros((len(IMAGES), 2, 5, LATENT + 1, 64, 64), np.float32)
        zs = np.zeros((len(IMAGES), 2, 5, LATENT), np.float32)
        for mi, obj in enumerate(MODELS):
            for s in SEEDS:
                model = load(obj, s, ep)
                for ii, n in enumerate(IMAGES):
                    x = torch.as_tensor(xs[n])[None, None].cuda()
                    zs[ii, mi, s], rec[ii, mi, s] = prefix_recons(model, x)
        for ii, n in enumerate(IMAGES):
            c, m, f = IMAGES[n]
            G, T = refs[n]
            D = np.diff(rec[ii], axis=2)                            # [model, seed, 10, 64, 64]
            A = np.array([[[[cosine(D[mi, s, k], T[j]) for k in range(LATENT)] for j in range(3)]
                           for s in SEEDS] for mi in range(2)])     # [model, seed, ref, k]
            lims = {obj: float(np.abs(D[mi]).max()) for mi, obj in enumerate(MODELS)}
            lims_all[f"{n}_e{ep:03d}"] = lims
            tag = f"{n}_e{ep:03d}"
            title = f"{IMAGE_NAMES[n]} (c{c} m{m} f{f}, test #{picks[n]}), epoch {ep}"
            np.savez_compressed(OUT / "data" / f"{tag}.npz", D=D, recon=rec[ii], z=zs[ii], T=T, G=G,
                                x=xs[n], alignment=A, models=np.array(MODELS), refs=np.array(REFS))
            if not _args.data_only:
                fig1(OUT / "fig1" / f"{tag}.png", xs[n], D, lims, "Signed prefix-change maps D_k — " + title)
                fig2(OUT / "fig2" / f"{tag}.png", xs[n], T, A, "GT transitions and alignment — " + title)
                fig3(OUT / "fig3" / f"{tag}.png", A, "Alignment across seeds — " + title)
            for mi, obj in enumerate(MODELS):
                for j, r in enumerate(REFS):
                    for k in range(LATENT):
                        for s in SEEDS:
                            rows.append([n, ep, obj, s, r, k + 1, A[mi, s, j, k]])
                        a = A[mi, :, j, k]
                        v = a[~np.isnan(a)]
                        summary.append([n, ep, obj, r, k + 1, *(
                            [v.mean(), v.min(), v.max()] if len(v) else [np.nan] * 3), len(v)])
        print(f"epoch {ep} done", flush=True)

    with open(OUT / "alignment.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "epoch", "model", "seed", "ref", "k", "cos"]); w.writerows(rows)
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "epoch", "model", "ref", "k", "mean", "min", "max", "n_valid"])
        w.writerows(summary)
    meta = {"run": str(RUN), "latent_dim": LATENT, "figures_made": not _args.data_only, "plan": "NonLinearPB/plans/hierarchical_Exp1_eval.md", "epochs": EPOCHS,
            "images": {n: {"test_index": picks[n], "c": IMAGES[n][0], "m": IMAGES[n][1], "f": IMAGES[n][2],
                           "name": IMAGE_NAMES[n]} for n in IMAGES},
            "references": "training-set conditional means; T_C=G_C-G_0, T_M=G_CM-G_C, T_F=G_CMF-G_CM",
            "empty_prefix": "decode of the all-zero code", "decoder_output": "raw (not clamped)",
            "nan_rule": "cosine NaN only if a norm is exactly zero (no cut-off)",
            "fig2_seed": FIG2_SEED, "colour_limits_fig1": lims_all,
            "colour_map": "RdBu: red negative (darker), blue positive (brighter)"}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
