"""
Compute `_lda_paired` (dim-to-dim paired cosine) for the synthetic-Fisher
models offline from saved encoder checkpoints and directions.npz.

Outputs:
  - per folder: `lda_paired_offline.npz` containing each model's `{tag}_lda_paired`
                with shape (1, 19), matching the layout of keys in metrics_raw.npz.
  - a quick visualization figure at
    `files/results/ICMLWorkshop_weightSymmetry2026/figures/scratch_synth_fisher_lda_paired.png`

Usage:
    python weight_symmetry/scripts/scratch_compute_synth_fisher_paired.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/argha/Mat_embedding_hyperbole/files/results"
FIG_DIR = os.path.join(ROOT, "ICMLWorkshop_weightSymmetry2026", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# (model_tag, folder, flip_dims)
SPEC = [
    ("fisher",           "exprmnt_2026_04_20__01_31_36", False),
    ("fp_fisher",        "exprmnt_2026_04_20__01_44_24", False),
    ("std_mrl_fisher",   "exprmnt_2026_04_20__11_33_48", False),
    ("prefix_l1_fisher", "exprmnt_2026_04_20__11_33_48", True),
]
CKPT_PREFIX = "seed47_"

# Style matches plot_fig_combined.py
STYLE = {
    "fisher":           dict(label="Unordered", color="#888888",  ls="--", lw=1.2),
    "fp_fisher":        dict(label="FP-MRL",    color="#009E73",  ls="-",  lw=1.8),
    "std_mrl_fisher":   dict(label="S-MRL",     color="#E07B00",  ls="-",  lw=1.8),
    "prefix_l1_fisher": dict(label=r"MD-$\ell_1$", color="#CC79A7", ls="-", lw=1.8),
}


def paired_cosine(a, u):
    a = a / (np.linalg.norm(a) + 1e-10)
    u = u / (np.linalg.norm(u) + 1e-10)
    return float(np.abs(a @ u))


def compute(tag, folder, flip):
    ckpt = os.path.join(ROOT, folder, f"{CKPT_PREFIX}{tag}_best.pt")
    sd   = torch.load(ckpt, map_location="cpu", weights_only=True)
    B    = sd["encoder.weight"].numpy().astype(np.float64)    # (d, p)
    B_T  = B.T.copy()                                         # (p, d)
    if flip:
        B_T = np.ascontiguousarray(B_T[:, ::-1])
    d = B_T.shape[1]

    dirs  = np.load(os.path.join(ROOT, folder, "directions.npz"), allow_pickle=True)
    lda   = dirs["lda_dirs"].astype(np.float64)               # (p, n_lda)
    n_lda = lda.shape[1]

    vals = np.full(d, np.nan, dtype=np.float64)
    for k in range(1, d + 1):
        if k <= n_lda:
            vals[k - 1] = paired_cosine(B_T[:, k - 1], lda[:, k - 1])
    return vals[np.newaxis, :], n_lda                         # (1, d), n_lda


def main():
    # Collect per-folder → {key: array}
    per_folder = {}
    all_vals   = {}

    print(f"{'tag':<20} {'flip':<5} {'d':<3} {'n_lda':<6} first 10 paired cosines")
    print("-" * 100)
    for tag, folder, flip in SPEC:
        vals, n_lda = compute(tag, folder, flip)
        d = vals.shape[1]
        key = f"{tag}_lda_paired"
        per_folder.setdefault(folder, {})[key] = vals
        all_vals[tag] = vals[0]
        print(f"{tag:<20} {str(flip):<5} {d:<3} {n_lda:<6} "
              f"{np.round(vals[0, :10], 3).tolist()}")

    # Save per-folder npz
    print("\nSaving offline npz files …")
    for folder, arrs in per_folder.items():
        out = os.path.join(ROOT, folder, "lda_paired_offline.npz")
        np.savez(out, **arrs)
        keys = ", ".join(sorted(arrs.keys()))
        print(f"  {out}\n    keys: {keys}")

    # Save visualization figure
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ks = np.arange(1, 20)
    for tag in ["fisher", "std_mrl_fisher", "fp_fisher", "prefix_l1_fisher"]:
        s = STYLE[tag]
        ax.plot(ks, all_vals[tag], label=s["label"], color=s["color"],
                ls=s["ls"], lw=s["lw"])
    ax.set_xlabel("Prefix size $k$")
    ax.set_ylabel("Paired cos. sim. to LDA")
    ax.set_title("Synthetic Fisher — paired cosine to LDA (offline)")
    ax.set_xlim(0.5, 19.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "scratch_synth_fisher_lda_paired.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {out_png}")


if __name__ == "__main__":
    main()
