"""
Scratch: sweep Standard-CE with dimensions sorted by mean |z| (ascending).

Quick sanity check — shows that the Standard-CE prefix curve is ordering-dependent.
Not a formal figure.

Usage:
    python weight_symmetry/scripts/scratch_std_ce_sorted.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PROJ = os.path.dirname(_CODE)
sys.path.insert(0, _CODE)

from weight_symmetry.data.loader import load_data

STD_CE_DIR = os.path.join(_PROJ, "files", "results", "exprmnt_2026_04_24__14_54_10")
OUT_DIR    = os.path.join(_PROJ, "files", "results", "ICMLWorkshop_weightSymmetry2026", "figures")


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


def _embed(enc, X):
    enc.eval()
    with torch.no_grad():
        return np.concatenate(
            [enc(X[i:i+512]).cpu().numpy() for i in range(0, len(X), 512)], axis=0)


def _sweep(Z, y, W, b, ks):
    return [float((np.argmax(Z[:, :k] @ W[:, :k].T + b, axis=1) == y).mean()) for k in ks]


def main():
    with open(os.path.join(STD_CE_DIR, "config.json")) as fh:
        cfg = json.load(fh)
    embed_dim  = cfg["embed_dim"]
    hidden_dim = cfg["hidden_dim"]
    seed       = cfg["seed"]

    data = load_data(cfg["dataset"], seed=seed)
    y_test = np.array(data.y_test.tolist(), dtype=np.int64)

    enc  = MLPEncoder(data.input_dim, hidden_dim, embed_dim)
    enc.load_state_dict(torch.load(
        os.path.join(STD_CE_DIR, "standard_ce_encoder_best.pt"),
        map_location="cpu", weights_only=True))
    head = nn.Linear(embed_dim, data.n_classes)
    head.load_state_dict(torch.load(
        os.path.join(STD_CE_DIR, "standard_ce_head_best.pt"),
        map_location="cpu", weights_only=True))

    Z_te = _embed(enc, data.X_test)
    W    = head.weight.detach().cpu().numpy()
    b    = head.bias.detach().cpu().numpy()

    mean_abs = np.mean(np.abs(Z_te), axis=0)
    order_asc = np.argsort(mean_abs)                # low → high
    print("[scratch] mean |z| per dim:", np.round(mean_abs, 4))
    print("[scratch] ascending order :", order_asc.tolist())

    Z_sorted = Z_te[:, order_asc]
    W_sorted = W[:, order_asc]

    ks = list(range(1, embed_dim + 1))
    acc_orig   = _sweep(Z_te,     y_test, W,        b, ks)
    acc_sorted = _sweep(Z_sorted, y_test, W_sorted, b, ks)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ks, acc_orig,   ls="--", color="#888888", lw=1.2,
            label="Std CE (as-is)")
    ax.plot(ks, acc_sorted, ls="-",  color="#CC3300", lw=1.6,
            label="Std CE (sorted by mean |z|, low→high)")
    ax.set_xlabel("Prefix size $k$")
    ax.set_ylabel("Linear acc. (trained W, no refit)")
    ax.set_title("Std CE — prefix sweep is ordering-dependent")
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "scratch_std_ce_sorted.png")
    fig.savefig(out, dpi=150)
    print(f"[scratch] saved → {out}")


if __name__ == "__main__":
    main()
