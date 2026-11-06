"""Cross-seed consistency of binary positive-change maps across prefix steps (Jaccard).

Conda environment: mrl_env_cuda12
Spec (agreed 2026-11-04): one test image (c1 m0 f0, "ring + stem / stem through", first test image of
that leaf), epochs 2-6 of the lr 6e-5 run, AE and MRL, seeds 0-4. For each (model, seed, epoch):
    xhat_{1:k} = D([z_1..z_k, 0..0]),   delta_k = xhat_{1:k} - xhat_{1:k-1},   k = 2..10
    B_k(p) = 1 if delta_k(p) > tau else 0,  tau = 0 (background included)
    B in {0,1}^(4096 x 9): rows = pixels (row-major 64 x 64), columns = steps 1->2 ... 9->10
For each step and each of the 10 seed pairs: Jaccard |P_s & P_s'| / |P_s | P_s'|; NaN if both empty.
AE is compared exactly like MRL (same k in both seeds), although AE coordinate order is arbitrary.

Outputs in <run>/figures/prefix_change/jaccard/:
    binary_trajectories.npz  B [epoch, model, seed, 4096, 9] (bool, tau = 0), delta [epoch, model, seed, 4096, 9]
                             (float32 signed change, for other thresholds), x, z1 recon [epoch, model, seed, 64, 64]
    pairwise_jaccard.csv     epoch, model, seed_a, seed_b, step(k), jaccard, n_a, n_b
    summary.csv              epoch, model, step, mean, min, max, n_valid_pairs
    counts.csv               epoch, model, seed, step, n_brightened
    metadata.json
Then the page: scripts/hier_exp1_seed_jaccard_html.py.

Usage: python scripts/hier_exp1_seed_jaccard.py [--run R]
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

import numpy as np
import torch

from hier_exp1.data import load_split
from hier_exp1_train import build_model

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_03__15_08_36"))
_args = _ap.parse_args()
RUN = _args.run
OUT = RUN / "figures" / "prefix_change" / "jaccard"

# CONFIG (agreed 2026-11-04)
IMAGE = (1, 0, 0)            # c1 m0 f0, ring + stem / stem through
EPOCHS = (2, 3, 4, 5, 6)
MODELS = ("mrl", "ae")
SEEDS = tuple(range(5))
LATENT = 10
TAU = 0.0
PAIRS = list(itertools.combinations(SEEDS, 2))


def jaccard(a, b):
    union = np.logical_or(a, b).sum()
    return float("nan") if union == 0 else float(np.logical_and(a, b).sum() / union)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_x, lab = load_split("test")
    c, m, f = IMAGE
    idx = int(np.flatnonzero((lab["coarse"] == c) & (lab["mid"] == m) & (lab["fine"] == f))[0])
    x = test_x[idx].astype(np.float32) / 255
    xt = torch.as_tensor(x)[None, None].cuda()
    codes_mask = torch.tril(torch.ones(LATENT, LATENT)).cuda()            # row k-1 keeps z_1..z_k

    B = np.zeros((len(EPOCHS), len(MODELS), len(SEEDS), 64 * 64, LATENT - 1), bool)
    DELTA = np.zeros(B.shape, np.float32)
    z1rec = np.zeros((len(EPOCHS), len(MODELS), len(SEEDS), 64, 64), np.float32)
    for ei, ep in enumerate(EPOCHS):
        for mi, obj in enumerate(MODELS):
            for s in SEEDS:
                model = build_model(s)
                path = RUN / "checkpoints" / obj / f"seed_{s}" / "epochs" / f"epoch_{ep:03d}.pt"
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                model = model.cuda().eval()
                with torch.no_grad():
                    z = model.encode(xt)                                      # [1, 10]
                    rec = model.decode(z * codes_mask)[:, 0].cpu().numpy()    # [10, 64, 64], prefixes 1..10
                delta = rec[1:] - rec[:-1]                                    # [9]: steps 1->2 ... 9->10
                DELTA[ei, mi, s] = delta.reshape(LATENT - 1, -1).T
                B[ei, mi, s] = DELTA[ei, mi, s] > TAU
                z1rec[ei, mi, s] = rec[0]

    pair_rows, summary, counts = [], [], []
    for ei, ep in enumerate(EPOCHS):
        for mi, obj in enumerate(MODELS):
            for k in range(LATENT - 1):
                for s in SEEDS:
                    counts.append([ep, obj, s, k + 2, int(B[ei, mi, s, :, k].sum())])
                js = []
                for a, b in PAIRS:
                    j = jaccard(B[ei, mi, a, :, k], B[ei, mi, b, :, k])
                    js.append(j)
                    pair_rows.append([ep, obj, a, b, k + 2, j, int(B[ei, mi, a, :, k].sum()), int(B[ei, mi, b, :, k].sum())])
                v = np.array([j for j in js if not np.isnan(j)])
                summary.append([ep, obj, k + 2, *([v.mean(), v.min(), v.max()] if len(v) else [np.nan] * 3), len(v)])

    np.savez_compressed(OUT / "binary_trajectories.npz", B=B, delta=DELTA, x=x, z1_recon=z1rec, epochs=np.array(EPOCHS),
                        models=np.array(MODELS), seeds=np.array(SEEDS), steps=np.arange(2, LATENT + 1))
    for name, header, rows in (("pairwise_jaccard.csv", ["epoch", "model", "seed_a", "seed_b", "step", "jaccard", "n_a", "n_b"], pair_rows),
                               ("summary.csv", ["epoch", "model", "step", "mean", "min", "max", "n_valid_pairs"], summary),
                               ("counts.csv", ["epoch", "model", "seed", "step", "n_brightened"], counts)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    meta = {"run": str(RUN), "image": {"test_index": idx, "c": c, "m": m, "f": f, "name": "ring + stem / stem through"},
            "epochs": list(EPOCHS), "models": list(MODELS), "seeds": list(SEEDS), "tau": TAU,
            "steps": "k = 2..10, step k is 1:(k-1) -> 1:k", "pairs": PAIRS,
            "B_layout": "[epoch, model, seed, pixel (row-major 64x64), step]",
            "jaccard_empty_rule": "NaN if both seeds brighten no pixel at that step",
            "decoder_output": "raw (not clamped)"}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
