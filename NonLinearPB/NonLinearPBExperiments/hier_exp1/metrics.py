"""Partition and distance metrics for Experiment 1 (plan section 5D).

Conda environment: mrl_env_cuda12
Usage:
    from hier_exp1.metrics import containment, shuffled_containment, pair_indices, auc

containment(fine, coarse): size-weighted share of each finer cluster that lies
in its best coarser cluster (1 = perfectly nested). shuffled_containment: the
same after permuting the finer labels (cluster sizes kept), the chance level.
Stability and cross-seed agreement use sklearn's adjusted_rand_score.
"""

import numpy as np
from scipy.stats import rankdata

PAIR_TYPES = ("same_leaf", "same_mid_diff_leaf", "same_coarse_diff_mid", "diff_coarse")


def contingency(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    return table


def containment(fine, coarse):
    """sum_B |B|/N * max_A |A n B|/|B| = (1/N) sum_B max_A |A n B|."""
    table = contingency(fine, coarse)          # rows: finer clusters
    return float(table.max(axis=1).sum() / len(fine))


def shuffled_containment(fine, coarse, n_perm, rng):
    """Containment after permuting the finer labels across images; cluster sizes unchanged."""
    fine = np.asarray(fine)
    return np.array([containment(rng.permutation(fine), coarse) for _ in range(n_perm)])


def pair_indices(labels, n_per_type, rng):
    """{pair type: (i, j)} with n_per_type pairs each, i != j, sampled uniformly within the type."""
    c, mid, leaf = labels["coarse"], labels["mid_node"], labels["leaf_id"]
    n = len(c)
    out, need = {}, {t: n_per_type for t in PAIR_TYPES}
    buckets = {t: ([], []) for t in PAIR_TYPES}
    while any(v > 0 for v in need.values()):
        i = rng.integers(n, size=200_000)
        j = rng.integers(n, size=200_000)
        keep = i != j
        i, j = i[keep], j[keep]
        kind = np.where(leaf[i] == leaf[j], 0, np.where(mid[i] == mid[j], 1, np.where(c[i] == c[j], 2, 3)))
        for t_idx, t in enumerate(PAIR_TYPES):
            sel = np.flatnonzero(kind == t_idx)[:need[t]]
            buckets[t][0].append(i[sel])
            buckets[t][1].append(j[sel])
            need[t] -= len(sel)
    for t in PAIR_TYPES:
        out[t] = (np.concatenate(buckets[t][0]), np.concatenate(buckets[t][1]))
    return out


def auc(lower, higher):
    """P(random draw from `higher` > random draw from `lower`), ties count 1/2 (Mann-Whitney)."""
    lower, higher = np.asarray(lower, dtype=np.float64), np.asarray(higher, dtype=np.float64)
    ranks = rankdata(np.concatenate([lower, higher]))          # 1-based, ties averaged
    n_l, n_h = len(lower), len(higher)
    return float((ranks[n_l:].sum() - n_h * (n_h + 1) / 2) / (n_l * n_h))
