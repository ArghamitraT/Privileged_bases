"""Change maps and hierarchy alignment for "pair" codes {z1}, {z1, z2}, ..., {z1, zd} (one image set, many epochs).

Conda environment: mrl_env_cuda12
For each model (name=run pairs, e.g. the pair-objective run and the 5-d MRL run), seed, epoch and test image:
    xhat_0   = decode(all-zero code)
    xhat_1   = decode([z1, 0, ..., 0])
    xhat_j   = decode(z1 and zj kept, rest zero), j = 2..d
    D_1      = xhat_1 - xhat_0                       (what z1 adds to the zero-code image)
    D_j      = xhat_j - xhat_1, j = 2..d             (what adding zj on top of z1 does)
    cos(D_j, T_C / T_M / T_F) with the training-set conditional-mean references of plan hierarchical_Exp1_eval.md:
    T_C = G_C - G_0, T_M = G_CM - G_C, T_F = G_CMF - G_CM. Decoder output raw (not clamped), no cut-off.
Writes <out>/data/<img>_e<ep>.npz with the same keys as hier_exp1_prefix_change.py (recon [model, seed, d+1, 64, 64]
in the order xhat_0, xhat_1, xhat_2..d; D [model, seed, d, 64, 64]; alignment [model, seed, ref, d]; z; T; G; x; models)
plus metadata.json and alignment.csv, so hier_exp1_recon_cosine_html.py can draw it.

Usage: python scripts/hier_exp1_pair_change.py --out DIR --model pair=RUN_A --model mrl=RUN_B [--latent 5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hier_exp1.data import load_split
from hier_exp1_train import build_model

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", type=Path, required=True)
_ap.add_argument("--model", action="append", required=True, help="name=run_dir; checkpoints/<name>/seed_<s>/epochs/ in that run")
_ap.add_argument("--latent", type=int, default=5)
_ap.add_argument("--epochs", type=int, nargs="+", default=[2, 3, 4, 5, 10, 15, 20, 25])
_ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
_args = _ap.parse_args()
MODELS = [m.split("=", 1) for m in _args.model]
IMAGES = {"c1m0f0": (1, 0, 0), "c1m2f0": (1, 2, 0), "c0m2f0": (0, 2, 0), "c2m2f0": (2, 2, 0)}
IMAGE_NAMES = {"c1m0f0": "ring + stem / stem through", "c1m2f0": "ring + stem / stem slanted",
               "c0m2f0": "dumbbell / bowed", "c2m2f0": "boomerang / stepped"}
REFS = ("C", "M", "F")
D_LAT = _args.latent


def references(train_x, lab, c, m, f):
    sel = lambda mask: train_x[mask].mean(0)
    g0, gc = train_x.mean(0), sel(lab["coarse"] == c)
    gcm = sel((lab["coarse"] == c) & (lab["mid"] == m))
    gcmf = sel((lab["coarse"] == c) & (lab["mid"] == m) & (lab["fine"] == f))
    return np.stack([g0, gc, gcm, gcmf]), np.stack([gc - g0, gcm - gc, gcmf - gcm])


def pair_codes(z):
    """z [d] -> codes [d+1, d]: zero, {z1}, {z1, z2}, ..., {z1, zd}."""
    codes = torch.zeros(D_LAT + 1, D_LAT, device=z.device)
    codes[1:, 0] = z[0]
    for j in range(1, D_LAT):
        codes[j + 1, j] = z[j]
    return codes


def cosine(d, t):
    nd, nt = np.linalg.norm(d), np.linalg.norm(t)
    return float("nan") if nd == 0 or nt == 0 else float(np.dot(d.ravel(), t.ravel()) / (nd * nt))


def main():
    (_args.out / "data").mkdir(parents=True, exist_ok=True)
    train_x, train_lab = load_split("train")
    train_x = train_x.astype(np.float32) / 255
    test_x, test_lab = load_split("test")
    picks, refs = {}, {}
    for n, (c, m, f) in IMAGES.items():
        picks[n] = int(np.flatnonzero((test_lab["coarse"] == c) & (test_lab["mid"] == m) & (test_lab["fine"] == f))[0])
        refs[n] = references(train_x, train_lab, c, m, f)
    xs = {n: test_x[i].astype(np.float32) / 255 for n, i in picks.items()}
    S = len(_args.seeds)
    rows = []
    for ep in _args.epochs:
        rec = np.zeros((len(IMAGES), len(MODELS), S, D_LAT + 1, 64, 64), np.float32)
        zs = np.zeros((len(IMAGES), len(MODELS), S, D_LAT), np.float32)
        for mi, (name, run) in enumerate(MODELS):
            for si, s in enumerate(_args.seeds):
                model = build_model(s, latent_dim=D_LAT)
                path = Path(run) / "checkpoints" / name / f"seed_{s}" / "epochs" / f"epoch_{ep:03d}.pt"
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                model = model.cuda().eval()
                with torch.no_grad():
                    for ii, n in enumerate(IMAGES):
                        z = model.encode(torch.as_tensor(xs[n])[None, None].cuda())[0]
                        zs[ii, mi, si] = z.cpu().numpy()
                        rec[ii, mi, si] = model.decode(pair_codes(z))[:, 0].cpu().numpy()
        for ii, n in enumerate(IMAGES):
            G, T = refs[n]
            r = rec[ii]
            D = np.concatenate([r[:, :, 1:2] - r[:, :, 0:1], r[:, :, 2:] - r[:, :, 1:2]], axis=2)   # [model, seed, d, 64, 64]
            A = np.array([[[[cosine(D[mi, si, k], T[j]) for k in range(D_LAT)] for j in range(3)]
                           for si in range(S)] for mi in range(len(MODELS))])
            np.savez_compressed(_args.out / "data" / f"{n}_e{ep:03d}.npz", D=D, recon=r, z=zs[ii], T=T, G=G, x=xs[n],
                                alignment=A, models=np.array([m for m, _ in MODELS]), refs=np.array(REFS))
            for mi, (name, _) in enumerate(MODELS):
                for si, s in enumerate(_args.seeds):
                    for j, ref in enumerate(REFS):
                        for k in range(D_LAT):
                            rows.append([n, ep, name, s, ref, "z1" if k == 0 else f"z1+z{k + 1}", A[mi, si, j, k]])
        print(f"epoch {ep} done", flush=True)
    with open(_args.out / "alignment.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "epoch", "model", "seed", "ref", "code", "cos"]); w.writerows(rows)
    meta = {"codes": "pair: zero, {z1}, {z1, zj} j = 2..d", "latent_dim": D_LAT, "epochs": _args.epochs, "seeds": _args.seeds,
            "models": {m: r for m, r in MODELS},
            "change_maps": "D_1 = xhat_{z1} - xhat_zero; D_j = xhat_{z1,zj} - xhat_{z1}",
            "images": {n: {"test_index": picks[n], "c": IMAGES[n][0], "m": IMAGES[n][1], "f": IMAGES[n][2],
                           "name": IMAGE_NAMES[n]} for n in IMAGES},
            "references": "training-set conditional means; T_C=G_C-G_0, T_M=G_CM-G_C, T_F=G_CMF-G_CM",
            "decoder_output": "raw (not clamped)", "nan_rule": "cosine NaN only if a norm is exactly zero"}
    (_args.out / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(_args.out)


if __name__ == "__main__":
    main()
