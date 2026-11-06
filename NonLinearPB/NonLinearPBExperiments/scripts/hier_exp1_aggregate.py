"""Combine per-model Experiment 1 evaluations: cross-seed agreement, summary.csv and figures.

Conda environment: mrl_env_cuda12
Usage (from code/; run last by scripts/run_hier_exp1.py):
    python NonLinearPB/NonLinearPBExperiments/scripts/hier_exp1_aggregate.py --run-dir /path/to/exprmnt_x

Spec: NonLinearPB/plans/hierarchical_Exp1.md sections 5C-5D, 7. Cross-seed
comparisons use the same test images and identical clustering settings (they
come from the per-model partitions): ARI between the partitions of every pair
of seeds, and Spearman correlation of pairwise latent distances on one fixed
set of test pairs. Latent axes are never compared across seeds.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_exp1.model import LATENT_DIM

# ==============================================================================
# CONFIG
# ==============================================================================
MODELS = ("ae", "mrl", "random")
COLORS = {"ae": "#2a78d6", "mrl": "#eb6834", "random": "#8a8a8a"}
LABELS = {"ae": "standard AE", "mrl": "MRL-AE", "random": "untrained"}
PRIMARY_COUNTS = (3, 9, 27)
REFERENCE_MSE = {"family only": 0.0147, "family + mid": 0.0048, "leaf": 0.0005}   # v2 conditional variance
CROSS_SEED_PAIRS = 20000
CROSS_SEED_PAIR_SEED = 777
KS = list(range(1, LATENT_DIM + 1))
# ==============================================================================


def concat(folder, pattern):
    files = sorted(Path(folder).glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def cross_seed(run_dir):
    emb = {}
    for f in sorted((run_dir / "embeddings").glob("*.npz")):
        model, seed = f.stem.rsplit("_seed", 1)
        emb[model, int(seed)] = np.load(f)
    ari_rows, rank_rows = [], []
    n_test = next(iter(emb.values()))["z_test"].shape[0]
    rng = np.random.default_rng(CROSS_SEED_PAIR_SEED)
    i, j = rng.integers(n_test, size=CROSS_SEED_PAIRS), rng.integers(n_test, size=CROSS_SEED_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    for model in MODELS:
        seeds = sorted(s for m, s in emb if m == model)
        for a, b in combinations(seeds, 2):
            ea, eb = emb[model, a], emb[model, b]
            for k in KS:
                for key in ea.files:
                    if key.startswith(f"part_k{k}_n"):
                        n = int(key.split("_n")[1])
                        ari_rows.append({"model": model, "seed_a": a, "seed_b": b, "k": k, "n_clusters": n,
                                         "primary": n in PRIMARY_COUNTS, "ari": adjusted_rand_score(ea[key], eb[key])})
                da = np.linalg.norm(ea["z_test"][i, :k] - ea["z_test"][j, :k], axis=1)
                db = np.linalg.norm(eb["z_test"][i, :k] - eb["z_test"][j, :k], axis=1)
                rank_rows.append({"model": model, "seed_a": a, "seed_b": b, "k": k,
                                  "spearman_distance": spearmanr(da, db).statistic})
    return pd.DataFrame(ari_rows), pd.DataFrame(rank_rows)


def mean_sd(df, keys, value, name):
    g = df.groupby(keys)[value]
    return pd.DataFrame({f"{name}_mean": g.mean(), f"{name}_sd": g.std(ddof=1)})


def summarize(rec, probes, truth, stab, nest, xari, xrank):
    parts = [mean_sd(rec, ["model", "k"], "test_mse", "test_mse")]
    for h in ("coarse", "mid_node", "leaf_id"):
        parts.append(mean_sd(rec, ["model", "k"], f"clf_acc_{h}", f"recon_clf_{h}"))
    for level in ("coarse", "mid_node", "leaf_id"):
        parts.append(mean_sd(probes[probes.level == level], ["model", "k"], "test_acc", f"probe_{level}"))
    for level in ("mid_within_coarse", "fine_within_mid"):
        per_seed = probes[probes.level == level].groupby(["model", "seed", "k"]).test_acc.mean().reset_index()
        parts.append(mean_sd(per_seed, ["model", "k"], "test_acc", f"probe_{level}"))
    for n, level in zip(PRIMARY_COUNTS, ("coarse", "mid_node", "leaf_id")):
        parts.append(mean_sd(truth[(truth.n_clusters == n)], ["model", "k"], "ari_truth", f"ari_truth_{n}"))
        s = stab[stab.n_clusters == n].rename(columns={"k_next": "kk"})
        parts.append(mean_sd(s.assign(k=s.kk), ["model", "k"], "ari_consecutive", f"stability_ari_{n}_vs_prev_k"))
        if not xari.empty:
            parts.append(mean_sd(xari[xari.n_clusters == n], ["model", "k"], "ari", f"cross_seed_ari_{n}"))
    for nf, nc in ((9, 3), (27, 9)):
        p = nest[(nest.fine_clusters == nf) & (nest.coarse_clusters == nc)]
        same = p[p.k_fine == p.k_coarse].rename(columns={"k_fine": "k"})
        parts.append(mean_sd(same, ["model", "k"], "containment_minus_shuffled", f"nest_{nf}in{nc}_same_k"))
        prev = p[p.k_coarse == p.k_fine - 1].rename(columns={"k_fine": "k"})
        parts.append(mean_sd(prev, ["model", "k"], "containment_minus_shuffled", f"nest_{nf}in{nc}_coarse_at_k-1"))
    if not xrank.empty:
        parts.append(mean_sd(xrank, ["model", "k"], "spearman_distance", "cross_seed_distance_spearman"))
    return pd.concat(parts, axis=1).reset_index()


def curve(ax, summary, column, title, ylabel, refs=None, log=False):
    for model in MODELS:
        s = summary[summary.model == model]
        if s.empty or f"{column}_mean" not in s or s[f"{column}_mean"].isna().all():
            continue
        m, sd = s[f"{column}_mean"].to_numpy(), s[f"{column}_sd"].fillna(0).to_numpy()
        ax.plot(s.k, m, color=COLORS[model], lw=2, marker="o", ms=4, label=LABELS[model])
        ax.fill_between(s.k, m - sd, m + sd, color=COLORS[model], alpha=0.15, lw=0)
    for label, y in (refs or {}).items():
        ax.axhline(y, color="#555555", lw=0.8, ls="--")
        ax.text(10.1, y, label, fontsize=7, va="center", color="#444444")
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("prefix length k", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(KS)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)


def figures(run_dir, summary, nest):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    panels = [
        ("test_mse", "Reconstruction (test MSE)", "MSE", REFERENCE_MSE, True),
        ("probe_coarse", "Linear probe: coarse (3)", "test accuracy", None, False),
        ("probe_mid_node", "Linear probe: mid (9)", "test accuracy", None, False),
        ("probe_leaf_id", "Linear probe: leaf (27)", "test accuracy", None, False),
        ("probe_mid_within_coarse", "Parent-specific: mid within coarse", "test accuracy (mean of 3)", None, False),
        ("probe_fine_within_mid", "Parent-specific: fine within mid", "test accuracy (mean of 9)", None, False),
        ("recon_clf_coarse", "Frozen classifier on reconstructions: coarse", "accuracy", None, False),
        ("recon_clf_mid_node", "Frozen classifier on reconstructions: mid", "accuracy", None, False),
        ("recon_clf_leaf_id", "Frozen classifier on reconstructions: leaf", "accuracy", None, False),
        ("ari_truth_3", "k-means ARI to truth: 3 clusters", "ARI", None, False),
        ("ari_truth_9", "k-means ARI to truth: 9 clusters", "ARI", None, False),
        ("ari_truth_27", "k-means ARI to truth: 27 clusters", "ARI", None, False),
        ("nest_9in3_coarse_at_k-1", "Nesting: 9 @k inside 3 @k-1 (minus shuffled)", "containment - shuffled", None, False),
        ("nest_27in9_coarse_at_k-1", "Nesting: 27 @k inside 9 @k-1 (minus shuffled)", "containment - shuffled", None, False),
        ("stability_ari_9_vs_prev_k", "Stability: 9 clusters, k-1 vs k", "ARI", None, False),
        ("cross_seed_ari_3", "Cross-seed ARI: 3 clusters", "ARI", None, False),
        ("cross_seed_ari_9", "Cross-seed ARI: 9 clusters", "ARI", None, False),
        ("cross_seed_ari_27", "Cross-seed ARI: 27 clusters", "ARI", None, False),
    ]
    fig, axes = plt.subplots(6, 3, figsize=(13, 22))
    for ax, (col, title, ylabel, refs, log) in zip(axes.flat, panels):
        curve(ax, summary, col, title, ylabel, refs, log)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Experiment 1: prefix curves (mean ± SD over seeds, test images)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fig_dir / "prefix_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(MODELS), figsize=(4.2 * len(MODELS), 7.6))
    for row, (nf, nc) in enumerate(((9, 3), (27, 9))):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            p = nest[(nest.model == model) & (nest.fine_clusters == nf) & (nest.coarse_clusters == nc)]
            mat = np.full((len(KS), len(KS)), np.nan)
            if not p.empty:
                g = p.groupby(["k_coarse", "k_fine"]).containment_minus_shuffled.mean()
                for (kc, kf), v in g.items():
                    mat[kc - 1, kf - 1] = v
            im = ax.imshow(mat, origin="lower", cmap="Blues", vmin=0, vmax=np.nanmax([0.01, np.nanmax(mat) if np.isfinite(mat).any() else 0.01]))
            ax.set_title(f"{LABELS[model]}: {nf} inside {nc}", fontsize=9)
            ax.set_xlabel(f"k for the {nf} clusters", fontsize=8)
            ax.set_ylabel(f"k for the {nc} clusters", fontsize=8)
            ax.set_xticks(range(len(KS)), KS, fontsize=7)
            ax.set_yticks(range(len(KS)), KS, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, label="containment - shuffled")
    fig.suptitle("Nesting: does a finer partition at a later prefix sit inside a coarser one at an earlier prefix? (mean over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "nesting_matrix.png", dpi=130)
    plt.close(fig)


def training_curves(run_dir):
    files = sorted(f for f in (run_dir / "training").glob("*.csv") if not f.name.startswith("latent_"))   # latent_*: per-coordinate ranges
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for f in files:
        model = f.stem.rsplit("_seed", 1)[0]
        h = pd.read_csv(f)
        axes[0].plot(h.epoch, h.val_full_mse, color=COLORS[model], lw=1, alpha=0.8)
        axes[1].plot(h.epoch, h.val_mean_prefix_mse, color=COLORS[model], lw=1, alpha=0.8)
    for ax, t in zip(axes, ("validation full-width MSE (k=10)", "validation mean prefix MSE")):
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    for m in ("ae", "mrl"):
        axes[0].plot([], [], color=COLORS[m], label=LABELS[m])
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "training_curves.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    rec = concat(run / "reconstructions", "*_seed*.csv")
    probes = concat(run / "probes", "*_seed*.csv")
    dist = concat(run / "distances", "dist_*.csv")
    aucs = concat(run / "distances", "auc_*.csv")
    truth = concat(run / "clustering", "truth_ari_*.csv")
    stab = concat(run / "clustering", "stability_*.csv")
    nest = concat(run / "clustering", "nesting_*.csv")
    rec.to_csv(run / "reconstructions" / "prefix_reconstruction.csv", index=False)
    probes.to_csv(run / "probes" / "linear_probes.csv", index=False)
    dist.to_csv(run / "distances" / "representation_distances.csv", index=False)
    aucs.to_csv(run / "distances" / "distance_auc.csv", index=False)
    truth.to_csv(run / "clustering" / "truth_ari.csv", index=False)
    stab.to_csv(run / "clustering" / "stability.csv", index=False)
    nest.to_csv(run / "clustering" / "nesting.csv", index=False)
    xari, xrank = cross_seed(run)
    xari.to_csv(run / "clustering" / "cross_seed_ari.csv", index=False)
    xrank.to_csv(run / "distances" / "cross_seed_distance_spearman.csv", index=False)
    summary = summarize(rec, probes, truth, stab, nest, xari, xrank)
    summary.to_csv(run / "summary.csv", index=False)
    figures(run, summary, nest)
    training_curves(run)
    print(f"aggregated {len(rec.groupby(['model', 'seed']))} models -> {run / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
