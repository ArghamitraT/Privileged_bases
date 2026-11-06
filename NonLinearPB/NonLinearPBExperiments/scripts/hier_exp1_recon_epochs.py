"""Prefix-reconstruction grids per model, seed and epoch (input for the epoch-slider page).

For each objective (ae, mrl), seed 0-4 and epoch in EPOCHS: decode the same 9 validation images
(first image of each mid node, c0m0 ... c2m2; the W&B recon/grid picks) from z1, z1:2, ... z1:10
(remaining coordinates zeroed, clipped to [0, 1]). One PNG per (objective, seed, epoch): 9 rows x
11 columns (input + 10 prefixes), 64 px tiles with a 2 px grey gap. Epoch 0 = untrained init
(build_model(seed), identical to the run's init); other epochs from diagnostics/<model>/epochs/.
Also writes mse.json: full-validation MSE per k for every epoch, from training/<model>.csv.

Weights come from checkpoints/<obj>/seed_<s>/epochs/epoch_XXX.pt (--epoch-weights runs) when present,
else from the full states in diagnostics/<model>/epochs/.

Usage: python scripts/hier_exp1_recon_epochs.py [--run R] [--out O] [--epochs 0 1 ... 20 200]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

import numpy as np
import torch
from PIL import Image

from hier_exp1.data import load_split
from hier_exp1_train import build_model

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_02__19_43_27"))
_ap.add_argument("--out", type=Path, default=None)
_ap.add_argument("--epochs", type=int, nargs="+", default=list(range(0, 21)) + [200])
_args = _ap.parse_args()
RUN = _args.run
OUT = _args.out or RUN / "figures" / "recon_epochs"
EPOCHS = _args.epochs
SEEDS = range(5)
GAP = 2


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    val, lab = load_split("val")
    mid = lab["mid_node"]
    picks = [int(np.flatnonzero(mid == m)[0]) for m in range(9)]
    x = torch.as_tensor(val[picks]).unsqueeze(1).float().cuda() / 255
    mse = {}
    for obj in ("ae", "mrl"):
        for seed in SEEDS:
            name = f"{obj}_seed{seed}"
            with open(RUN / "training" / f"{name}.csv") as fh:
                rows = {int(r["epoch"]): r for r in csv.DictReader(fh)}
            mse[name] = {str(e): [float(rows[e][f"val_mse_k{k}"]) for k in range(1, 11)] for e in EPOCHS if e in rows}
            for ep in EPOCHS:
                model = build_model(seed)
                light = RUN / "checkpoints" / obj / f"seed_{seed}" / "epochs" / f"epoch_{ep:03d}.pt"
                if light.exists():
                    model.load_state_dict(torch.load(light, map_location="cpu", weights_only=True))
                elif ep > 0:
                    state = torch.load(RUN / "diagnostics" / name / "epochs" / f"epoch_{ep:03d}.pt",
                                       map_location="cpu", weights_only=False)["model"]
                    model.load_state_dict(state)
                model = model.cuda().eval()
                with torch.no_grad():
                    z = model.encode(x)
                    cols = [x] + [model.decode(torch.cat([z[:, :k], torch.zeros_like(z[:, k:])], 1)).clamp(0, 1)
                                  for k in range(1, 11)]
                tiles = torch.stack(cols, 1)[:, :, 0]                                  # [9, 11, 64, 64]
                tiles = torch.nn.functional.pad(tiles, (0, GAP, 0, GAP), value=0.35)
                r, c, h, w = tiles.shape
                img = tiles.permute(0, 2, 1, 3).reshape(r * h, c * w)[:-GAP, :-GAP]
                Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(
                    OUT / "img" / f"{name}_e{ep:03d}.png", optimize=True)
            print(name, "done", flush=True)
    (OUT / "mse.json").write_text(json.dumps({"epochs": EPOCHS, "picks_val_index": picks, "mse": mse}) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
