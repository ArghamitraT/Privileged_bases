"""Leakage probe for the hierarchical synthetic dataset (spec section 5, item 6).

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/scripts/leakage_probe_hier.py --data /home/argha/Mat_embedding/data/synthetic_data/hierarchical_images_v1
    python NonLinearPB/NonLinearPBExperiments/scripts/leakage_probe_hier.py --data /path/to/run --output /path/to/dir

Question: can a fine-label classifier fitted inside one parent (c, m) also sort
the fine labels of a different parent? If so, a shared pixel feature carries the
fine label across parents, so fine details are not fully parent-specific.

For each source parent A, a multinomial logistic regression on raw pixels is fit
to A's train images (3 fine classes) and scored on every parent B's val images.
Fine indices are arbitrary across parents (f=0 is a different detail in each
family), so transfer is reported two ways: raw accuracy (labels as given) and
best-permutation accuracy (the best of the 6 relabelings of B's classes), which
catches a shared feature even when the index differs. Chance is 1/3. The
diagonal (A = B) checks that each parent's fine labels are learnable at all.
Train and val only; test is never touched.
"""

import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# CONFIG: fixed probe; no tuning against the transfer numbers.
C_REG = 0.1
MAX_ITER = 2000
FLAG = 0.6      # transfer accuracy above this is reported as leakage
PARENTS = [(c, m) for c in range(3) for m in range(3)]


def load(path):
    d = np.load(path)
    x = d["images"].reshape(len(d["images"]), -1).astype(np.float32) / 255.0
    return x, d["coarse"], d["mid"], d["fine"]


def best_permutation_accuracy(pred, true):
    return max(np.mean(np.array(p)[pred] == true) for p in permutations(range(3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="folder with train.npz and val.npz")
    ap.add_argument("--output", type=Path, default=None, help="default: <data>/leakage_probe")
    args = ap.parse_args()
    out = args.output or args.data / "leakage_probe"
    out.mkdir(parents=True, exist_ok=True)

    xtr, ctr, mtr, ftr = load(args.data / "train.npz")
    xva, cva, mva, fva = load(args.data / "val.npz")
    raw = np.zeros((9, 9))
    perm = np.zeros((9, 9))
    for i, (ca, ma) in enumerate(PARENTS):
        sel = (ctr == ca) & (mtr == ma)
        clf = LogisticRegression(C=C_REG, max_iter=MAX_ITER).fit(xtr[sel], ftr[sel])
        for j, (cb, mb) in enumerate(PARENTS):
            selb = (cva == cb) & (mva == mb)
            pred = clf.predict(xva[selb])
            raw[i, j] = np.mean(pred == fva[selb])
            perm[i, j] = best_permutation_accuracy(pred, fva[selb])
        print(f"source c={ca},m={ma}: " + " ".join(f"{v:.2f}" for v in perm[i]), flush=True)

    rows = []
    for i, (ca, ma) in enumerate(PARENTS):
        for j, (cb, mb) in enumerate(PARENTS):
            kind = "same parent" if i == j else ("same family" if ca == cb else "other family")
            rows.append({"source": f"c={ca},m={ma}", "target": f"c={cb},m={mb}", "kind": kind,
                         "raw_acc": raw[i, j], "best_perm_acc": perm[i, j]})
    with open(out / "leakage_matrix.csv", "w") as fh:
        fh.write("source,target,kind,raw_acc,best_perm_acc\n")
        for r in rows:
            fh.write(f"\"{r['source']}\",\"{r['target']}\",{r['kind']},{r['raw_acc']:.4f},{r['best_perm_acc']:.4f}\n")

    summary = {}
    for kind in ("same parent", "same family", "other family"):
        v = np.array([r["best_perm_acc"] for r in rows if r["kind"] == kind])
        summary[kind] = {"n": len(v), "mean_best_perm": float(v.mean()), "max_best_perm": float(v.max()),
                         "n_above_flag": int((v > FLAG).sum())}
    summary["flagged_pairs"] = [f"{r['source']} -> {r['target']} ({r['best_perm_acc']:.2f})"
                                for r in rows if r["kind"] != "same parent" and r["best_perm_acc"] > FLAG]
    summary["settings"] = {"classifier": "multinomial logistic regression on raw pixels", "C": C_REG,
                           "fit": "train split", "score": "val split", "flag": FLAG, "chance": 1 / 3}
    (out / "leakage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
