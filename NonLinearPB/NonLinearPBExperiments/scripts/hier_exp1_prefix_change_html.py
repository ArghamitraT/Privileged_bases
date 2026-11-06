"""Epoch-slider pages for Figure 1 of the prefix-change analysis: one page per model (ae, mrl).

Conda environment: mrl_env_cuda12
Reads figures/prefix_change/data/<img>_e<ep>.npz (written by hier_exp1_prefix_change.py).
For each seed and epoch: one PNG, 4 rows (test images) x 11 columns, 64 px tiles, 2 px gap:
input (grey), the z1-only reconstruction decode([z1, 0..0]) (grey, clipped to [0, 1]), then
D_2..D_10 in RdBu (red negative, blue positive, white 0). Two colour-scale modes:
    tile : every change tile has its own limit, max |D_k| of that tile (printed on the tile)
    seed : one limit per seed, max |D_k| over k = 2..10, its 4 images and all epochs
    pos  : brighter only, max(D_k, 0), white -> blue, per-tile limit = max positive change of that tile
    posseed : brighter only, one limit per seed = max positive change over k = 2..10, 4 images, all epochs
Plus a "recon" view: input, then decode(z1:k) for k = 1..10, all grey, clipped to [0, 1].
Writes slider/img/{model}_{mode}_s{seed}_e{ep}.png, slider/limits.json and slider/{model}.html
(self-contained, WebP-embedded).

Usage: python scripts/hier_exp1_prefix_change_html.py [--run R]
"""
import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_03__15_08_36"))
_ap.add_argument("--desc", default="lr 6e-5", help="short run description shown on the page")
_ap.add_argument("--models", nargs="+", default=["ae", "mrl"])
_args = _ap.parse_args()
SRC = _args.run / "figures" / "prefix_change"
OUT = SRC / "slider"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "prefix_change_slider.html"

IMAGES = ("c1m0f0", "c1m2f0", "c0m2f0", "c2m2f0")
ROW_LABELS = ("ring+stem through", "ring+stem slanted", "dumbbell bowed", "boomerang stepped")
EPOCHS = (2, 3, 4, 5, 10, 15, 20, 25)
MODELS = ("ae", "mrl")
SEEDS = range(5)
GAP, GAP_RGB = 2, (150, 150, 150)
CMAP = matplotlib.colormaps["RdBu"]
CAVEAT = {"ae": "AE caveat: the AE never saw zeroed coordinates, so a zero is a specific code, not \"unknown\"; "
                "and its coordinate order is arbitrary, so step k means different things in different seeds.",
          "mrl": ""}


def tile_sheet(x, z1rec, D, lim):
    """x, z1rec [4,64,64]; D [4,10,64,64] (D[:, 0] unused); lim scalar or [4,10]; -> RGB uint8 sheet.
    For the brighter-only view pass max(D, 0): the colours then use only the white -> blue half."""
    lim = np.broadcast_to(np.asarray(lim, float), D.shape[:2])
    grey = lambda a: np.repeat(np.clip(a, 0, 1)[..., None], 3, -1)
    rows = []
    for i in range(len(IMAGES)):
        tiles = [grey(x[i]), grey(z1rec[i])]
        tiles += [CMAP(np.clip(0.5 + 0.5 * D[i, k] / max(lim[i, k], 1e-12), 0, 1))[..., :3]
                  for k in range(1, D.shape[1])]
        rows.append(tiles)
    h = w = 64
    n_r, n_c = len(rows), len(rows[0])
    sheet = np.empty((n_r * (h + GAP) - GAP, n_c * (w + GAP) - GAP, 3), np.uint8)
    sheet[:] = GAP_RGB
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sheet[r * (h + GAP):r * (h + GAP) + h, c * (w + GAP):c * (w + GAP) + w] = (t * 255).round()
    return sheet


def recon_sheet(x, rec):
    """x [4,64,64]; rec [4,11,64,64] (prefix 0..10) -> grey sheet: input, z1, z1:2 ... z1:10."""
    rows = np.concatenate([x[:, None], rec[:, 1:]], 1).clip(0, 1)            # [4, 11, 64, 64]
    r, c, h, w = rows.shape
    sheet = np.full((r * (h + GAP) - GAP, c * (w + GAP) - GAP), GAP_RGB[0] / 255, np.float32)
    for i in range(r):
        for j in range(c):
            sheet[i * (h + GAP):i * (h + GAP) + h, j * (w + GAP):j * (w + GAP) + w] = rows[i, j]
    return np.repeat((sheet * 255).round().astype(np.uint8)[..., None], 3, -1)


def webp_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "WEBP", quality=90, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    data = {ep: [np.load(SRC / "data" / f"{n}_e{ep:03d}.npz") for n in IMAGES] for ep in EPOCHS}
    x = np.stack([d["x"] for d in data[EPOCHS[0]]])
    grad = ", ".join(matplotlib.colors.to_hex(CMAP(v)) for v in np.linspace(1, 0, 11))   # top = +lim (blue)
    grad_pos = ", ".join(matplotlib.colors.to_hex(CMAP(v)) for v in np.linspace(1, 0.5, 6))  # +lim .. 0
    limits = {}
    for mi, obj in enumerate(MODELS):
        if obj not in _args.models:
            continue
        D = {ep: np.stack([d["D"][mi] for d in data[ep]], 1) for ep in EPOCHS}        # [seed, img, k, 64, 64]
        REC = {ep: np.stack([d["recon"][mi] for d in data[ep]], 1) for ep in EPOCHS}  # [seed, img, 11, 64, 64]
        Z1 = {ep: REC[ep][:, :, 1] for ep in EPOCHS}                                    # [seed, img, 64, 64]
        lim_epoch = [[float(np.abs(D[ep][s][:, 1:]).max()) for ep in EPOCHS] for s in SEEDS]
        lim_fixed = [max(v) for v in lim_epoch]
        lim_pos = [max(float(np.maximum(D[ep][s][:, 1:], 0).max()) for ep in EPOCHS) for s in SEEDS]
        imgs, tile_lims, pos_lims = {}, {}, {}
        for s in SEEDS:
            for ei, ep in enumerate(EPOCHS):
                tile_lim = np.abs(D[ep][s]).max(axis=(2, 3))                   # [img, k]
                tile_lims[f"s{s}_e{ep:03d}"] = tile_lim[:, 1:].round(4).tolist()
                Dpos = np.maximum(D[ep][s], 0)
                pos_lim = Dpos.max(axis=(2, 3))                                # [img, k]
                pos_lims[f"s{s}_e{ep:03d}"] = pos_lim[:, 1:].round(4).tolist()
                for mode, dd, lim in (("tile", D[ep][s], tile_lim), ("seed", D[ep][s], lim_fixed[s]),
                                      ("pos", Dpos, pos_lim), ("posseed", Dpos, lim_pos[s])):
                    sheet = tile_sheet(x, Z1[ep][s], dd, lim)
                    key = f"{mode}_s{s}_e{ep:03d}"
                    Image.fromarray(sheet).save(OUT / "img" / f"{obj}_{key}.png", optimize=True)
                    imgs[key] = webp_uri(sheet)
                rs = recon_sheet(x, REC[ep][s])
                Image.fromarray(rs).save(OUT / "img" / f"{obj}_recon_s{s}_e{ep:03d}.png", optimize=True)
                imgs[f"recon_s{s}_e{ep:03d}"] = webp_uri(rs)
        d_lat = D[EPOCHS[0]].shape[2]                                     # number of prefix steps = latent size
        payload = {"model": obj.upper(), "ncol": d_lat + 1, "epochs": list(EPOCHS), "rows": list(ROW_LABELS),
                   "w": int(sheet.shape[1]), "h": int(sheet.shape[0]), "lim": lim_fixed, "tile_lim": tile_lims,
                   "pos_lim": pos_lims, "lim_pos": lim_pos, "img": imgs}
        html = (TEMPLATE.read_text()
                .replace("__TITLE__", f"{obj.upper()} Prefix Changes" + (f" z{d_lat}" if d_lat != 10 else ""))
                .replace("__H1__", f"{obj.upper()}: signed prefix-change maps by epoch")
                .replace("__GRADIENT_POS__", grad_pos)
                .replace("__GRADIENT__", grad)
                .replace("__CAVEAT__", CAVEAT[obj])
                .replace("__NCOL__", str(d_lat + 1)).replace("__W__", str(int(sheet.shape[1])))
                .replace("__RUN__", _args.run.name).replace("__RUNDESC__", _args.desc)
                .replace("__LASTSTEP__", f"{d_lat - 1}→{d_lat}").replace("__D__", str(d_lat))
                .replace("__DATA__", json.dumps(payload)))
        (OUT / f"{obj}.html").write_text(html)
        limits[obj] = {"seed": lim_fixed, "tile": tile_lims, "pos": pos_lims, "pos_seed": lim_pos}
        print(obj, f"{len(html) / 1e6:.1f} MB", flush=True)
    (OUT / "limits.json").write_text(json.dumps({"epochs": list(EPOCHS), "limits": limits}, indent=1) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
