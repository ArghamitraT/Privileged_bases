"""Page for the cross-seed Jaccard analysis with an adjustable threshold tau.

Conda environment: mrl_env_cuda12
Reads <run>/figures/prefix_change/jaccard/binary_trajectories.npz (from hier_exp1_seed_jaccard.py) and embeds
the positive changes max(delta, 0) as float16 [epoch, model, seed, step, pixel], the z1 reconstructions and the
input as uint8. The browser recomputes, for the chosen epoch and tau: binary maps (delta > tau), Jaccard for
the 10 seed pairs per step (mean, min-max band, pair lines; MRL vs AE), a 5x5 pair grid for a selected step,
and brightened-pixel counts. Positive changes that underflow float16 are stored as its smallest positive
value, so tau = 0 reproduces the saved B exactly (checked below).
Writes <run>/figures/prefix_change/jaccard/seed_jaccard.html.

Usage: python scripts/hier_exp1_seed_jaccard_html.py [--run R]
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_03__15_08_36"))
_args = _ap.parse_args()
SRC = _args.run / "figures" / "prefix_change" / "jaccard"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "seed_jaccard.html"
BLUE = (33, 102, 172)          # RdBu positive end (#2166ac), as on the change-map pages
F16_MIN = np.float16(np.finfo(np.float16).smallest_subnormal)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def main():
    meta = json.loads((SRC / "metadata.json").read_text())
    d = np.load(SRC / "binary_trajectories.npz")
    delta, B, x, z1 = d["delta"], d["B"], d["x"], d["z1_recon"]            # delta [E, M, S, 4096, 9]
    pos = np.maximum(delta, 0).transpose(0, 1, 2, 4, 3)                     # [E, M, S, 9, 4096]
    pos16 = pos.astype("<f2")
    pos16[(pos > 0) & (pos16 == 0)] = F16_MIN                               # keep sign of tiny increases
    assert np.array_equal(pos16.astype(np.float32) > 0, B.transpose(0, 1, 2, 4, 3)), "tau = 0 would not match B"
    grey = (np.clip(z1, 0, 1) * 255).round().astype(np.uint8).reshape(*z1.shape[:3], -1)
    xg = (np.clip(x, 0, 1) * 255).round().astype(np.uint8).ravel()
    payload = {"epochs": [int(e) for e in d["epochs"]], "models": [str(m) for m in d["models"]],
               "pairs": [list(p) for p in meta["pairs"]], "blue": list(BLUE),
               "pos": b64(pos16.view("<u2")), "grey": b64(grey), "x": b64(xg)}
    html = (TEMPLATE.read_text()
            .replace("__IDX__", str(meta["image"]["test_index"]))
            .replace("__BLUE__", "#%02x%02x%02x" % BLUE)
            .replace("__DATA__", json.dumps(payload)))
    (SRC / "seed_jaccard.html").write_text(html)
    print(SRC / "seed_jaccard.html", f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
