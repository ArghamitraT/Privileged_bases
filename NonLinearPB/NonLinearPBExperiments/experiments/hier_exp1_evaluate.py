"""Evaluate one frozen Experiment 1 model (ae / mrl / random, one seed) at every prefix k = 1..10.

Conda environment: mrl_env_cuda12
Usage (from code/; normally queued by scripts/run_hier_exp1.py):
    python NonLinearPB/NonLinearPBExperiments/experiments/hier_exp1_evaluate.py --objective mrl --seed 0 --run-dir /path/to/exprmnt_x --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/hier_exp1_evaluate.py --objective random --seed 0 --run-dir /path --fast

Spec: NonLinearPB/plans/hierarchical_Exp1.md sections 5A-5D. 'random' is the
untrained network with the same seed's initial weights. All evaluation is on
the fixed v2 test images (same order for every model); probes fit on train and
pick regularization on val. Writes one CSV per view under the run folder
(probes/, reconstructions/, distances/, clustering/), the test embeddings and
cluster partitions (embeddings/) for the cross-seed aggregation, and a
reconstruction grid. Probe features are standardized with train statistics
inside each probe only; distances and clustering use raw z.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hier_exp1.data import file_sha256, load_split
from hier_exp1.metrics import PAIR_TYPES, auc, containment, pair_indices, shuffled_containment
from hier_exp1.model import LATENT_DIM, ReconClassifier, prefix_mse
from hier_exp1_train import build_model

# ==============================================================================
# CONFIG — plan section 5D; --fast overrides inside main().
# ==============================================================================
KS = list(range(1, LATENT_DIM + 1))
PROBE_C = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PROBE_MAX_ITER = 1000
CLUSTER_COUNTS = {"coarse": (2, 3, 4), "mid_node": (8, 9, 10), "leaf_id": (26, 27, 28)}
PRIMARY = {"coarse": 3, "mid_node": 9, "leaf_id": 27}
KMEANS_N_INIT = 20
KMEANS_SEED = 0
N_SHUFFLE = 100
N_SHUFFLE_SENSITIVITY = 20
PAIRS_PER_TYPE = 20000
PAIR_SEED = 12345                 # same pairs for every model and seed
GRID_K = (1, 2, 3, 5, 10)
GRID_PER_LEAF = 2
CHUNK = 512                       # images per encode/decode pass; ~2.1 GiB peak
# ==============================================================================


def load_model(args, device):
    model = build_model(args.seed, args.bounded)
    if args.objective != "random":
        ckpt = torch.load(args.run_dir / "checkpoints" / args.objective / f"seed_{args.seed}" / "best.pt",
                          map_location="cpu", weights_only=True)
        if bool(ckpt["config"].get("bounded", False)) != args.bounded:
            raise ValueError(f"checkpoint bounded={ckpt['config'].get('bounded', False)} but --bounded={args.bounded}")
        model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def embed(model, images, device):
    x_all = torch.as_tensor(images, device=device).unsqueeze(1)
    return torch.cat([model.encode(x_all[s:s + CHUNK].float() / 255.0) for s in range(0, len(x_all), CHUNK)]).double().cpu().numpy()


@torch.no_grad()
def reconstructions(model, images, labels, classifier, device, name):
    """Per-k test MSE (raw linear output; overall, clipped to [0,1], by family) and frozen-classifier
    accuracy on the clipped reconstructions."""
    x_all = torch.as_tensor(images, device=device).unsqueeze(1)
    n = len(x_all)
    mse = np.zeros((len(KS), n))
    mse_clip = np.zeros((len(KS), n))
    outside = np.zeros(len(KS))
    pred = {h: np.zeros((len(KS), n), dtype=np.int64) for h in ReconClassifier.HEADS}
    masks = torch.tril(torch.ones(LATENT_DIM, LATENT_DIM, device=device))
    for s in range(0, n, CHUNK):
        x = x_all[s:s + CHUNK].float() / 255.0
        z = model.encode(x)
        for i, k in enumerate(KS):
            xhat = model.decode(z * masks[k - 1])
            mse[i, s:s + len(x)] = ((xhat - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            clipped = xhat.clamp(0, 1)
            mse_clip[i, s:s + len(x)] = ((clipped - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            outside[i] += float(((xhat < 0) | (xhat > 1)).float().sum())
            if classifier is not None:
                o = classifier(clipped)
                for h in pred:
                    pred[h][i, s:s + len(x)] = o[h].argmax(1).cpu().numpy()
    rows = []
    for i, k in enumerate(KS):
        row = {"model": name[0], "seed": name[1], "k": k, "test_mse": mse[i].mean(),
               "test_mse_clipped": mse_clip[i].mean(), "frac_pixels_outside_0_1": outside[i] / (n * 64 * 64),
               **{f"test_mse_c{c}": mse[i][labels["coarse"] == c].mean() for c in range(3)}}
        for h in ReconClassifier.HEADS:
            row[f"clf_acc_{h}"] = (pred[h][i] == labels[h]).mean() if classifier is not None else np.nan
        rows.append(row)
    return rows


@torch.no_grad()
def grid(model, images, labels, device, path, title):
    """27 leaves x GRID_PER_LEAF samples; columns original | k in GRID_K for each sample."""
    idx = np.concatenate([np.flatnonzero(labels["leaf_id"] == l)[:GRID_PER_LEAF] for l in range(27)])
    x = torch.as_tensor(images[idx], device=device).unsqueeze(1).float() / 255.0
    z = model.encode(x)
    masks = torch.tril(torch.ones(LATENT_DIM, LATENT_DIM, device=device))
    cols = [x] + [model.decode(z * masks[k - 1]).clamp(0, 1) for k in GRID_K]
    tiles = torch.stack(cols, 1).squeeze(2).cpu().numpy()           # [n, 1 + len(GRID_K), 64, 64]
    pad = 2
    per_row = GRID_PER_LEAF * len(cols)
    canvas = np.ones((27 * (64 + pad), per_row * (64 + pad) + GRID_PER_LEAF * 6)) * 0.35
    for r in range(27):
        for s in range(GRID_PER_LEAF):
            for c in range(len(cols)):
                y0 = r * (64 + pad)
                x0 = (s * len(cols) + c) * (64 + pad) + s * 6
                canvas[y0:y0 + 64, x0:x0 + 64] = tiles[r * GRID_PER_LEAF + s, c]
    fig = plt.figure(figsize=(canvas.shape[1] / 60, canvas.shape[0] / 60 + 0.6))
    ax = fig.add_axes([0, 0, 1, canvas.shape[0] / (canvas.shape[0] + 36)])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.suptitle(f"{title} — per sample: original | k = {', '.join(map(str, GRID_K))}; rows: leaves 0..26", fontsize=9)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def probe_tasks(train, val, test):
    """(task, level, parent, labels per split, row selections per split)."""
    tasks = [("global", "coarse", "all", "coarse", None), ("pooled", "mid_node", "all", "mid_node", None),
             ("pooled", "leaf_id", "all", "leaf_id", None)]
    tasks += [("parent", "mid_within_coarse", f"c={c}", "mid", ("coarse", c)) for c in range(3)]
    tasks += [("parent", "fine_within_mid", f"c={p // 3},m={p % 3}", "fine", ("mid_node", p)) for p in range(9)]
    for kind, level, parent, target, sel in tasks:
        def rows(lab):
            return np.ones(len(lab["coarse"]), bool) if sel is None else lab[sel[0]] == sel[1]
        yield kind, level, parent, target, rows(train), rows(val), rows(test)


def probes(z, labels, name, c_grid):
    out = []
    ztr, zva, zte = z["train"], z["val"], z["test"]
    ltr, lva, lte = labels["train"], labels["val"], labels["test"]
    for kind, level, parent, target, rtr, rva, rte in probe_tasks(ltr, lva, lte):
        for k in KS:
            scaler = StandardScaler().fit(ztr[rtr, :k])
            xtr, xva, xte = (scaler.transform(a[r, :k]) for a, r in ((ztr, rtr), (zva, rva), (zte, rte)))
            best = None
            for c in c_grid:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf = LogisticRegression(C=c, max_iter=PROBE_MAX_ITER).fit(xtr, ltr[target][rtr])
                v = clf.score(xva, lva[target][rva])
                if best is None or v > best[0]:
                    best = (v, c, clf)
            pred = best[2].predict(xte)
            y = lte[target][rte]
            out.append({"model": name[0], "seed": name[1], "k": k, "probe": kind, "level": level, "parent": parent,
                        "n_classes": len(np.unique(y)), "C": best[1], "val_acc": best[0],
                        "test_acc": float(np.mean(pred == y)), "test_balanced_acc": balanced_accuracy_score(y, pred)})
    return out


def distances(z, labels, name, pairs_per_type):
    rng = np.random.default_rng(PAIR_SEED)
    pairs = pair_indices(labels, pairs_per_type, rng)
    ri, rj = rng.integers(len(z), size=pairs_per_type), rng.integers(len(z), size=pairs_per_type)
    keep = ri != rj
    ri, rj = ri[keep], rj[keep]
    rows, aucs = [], []
    for k in KS:
        zk = z[:, :k]
        scale = np.linalg.norm(zk[ri] - zk[rj], axis=1).mean()
        d = {t: np.linalg.norm(zk[i] - zk[j], axis=1) for t, (i, j) in pairs.items()}
        for family in ("all", 0, 1, 2):
            for t, (i, j) in pairs.items():
                sel = np.ones(len(i), bool) if family == "all" else labels["coarse"][i] == family
                v = d[t][sel]
                rows.append({"model": name[0], "seed": name[1], "k": k, "family": family, "pair_type": t, "n": len(v),
                             "mean": v.mean(), "median": np.median(v), "q10": np.quantile(v, 0.1),
                             "q90": np.quantile(v, 0.9), "random_pair_mean": scale, "norm_mean": v.mean() / scale})
            for lo, hi in zip(PAIR_TYPES[:-1], PAIR_TYPES[1:]):
                slo = np.ones(len(d[lo]), bool) if family == "all" else labels["coarse"][pairs[lo][0]] == family
                shi = np.ones(len(d[hi]), bool) if family == "all" else labels["coarse"][pairs[hi][0]] == family
                aucs.append({"model": name[0], "seed": name[1], "k": k, "family": family, "closer": lo, "farther": hi,
                             "auc": auc(d[lo][slo], d[hi][shi])})
    return rows, aucs, pairs


def clustering(z, labels, name, n_init, n_shuffle, n_shuffle_sens):
    parts = {}
    for k in KS:
        for level, counts in CLUSTER_COUNTS.items():
            for n in counts:
                parts[k, n] = KMeans(n_clusters=n, n_init=n_init, random_state=KMEANS_SEED).fit_predict(z[:, :k])
    truth, nest, stab = [], [], []
    for k in KS:
        for level, counts in CLUSTER_COUNTS.items():
            for n in counts:
                truth.append({"model": name[0], "seed": name[1], "k": k, "level": level, "n_clusters": n,
                              "primary": n == PRIMARY[level], "ari_truth": adjusted_rand_score(labels[level], parts[k, n])})
                if k < KS[-1]:
                    stab.append({"model": name[0], "seed": name[1], "k": k, "k_next": k + 1, "n_clusters": n,
                                 "primary": n == PRIMARY[level], "ari_consecutive": adjusted_rand_score(parts[k, n], parts[k + 1, n])})
    rng = np.random.default_rng(KMEANS_SEED)
    for fine_level, coarse_level in (("mid_node", "coarse"), ("leaf_id", "mid_node")):
        for nf in CLUSTER_COUNTS[fine_level]:
            for nc in CLUSTER_COUNTS[coarse_level]:
                primary = nf == PRIMARY[fine_level] and nc == PRIMARY[coarse_level]
                for k_late in KS:
                    for k_early in KS:
                        if k_early > k_late:
                            continue
                        obs = containment(parts[k_late, nf], parts[k_early, nc])
                        sh = shuffled_containment(parts[k_late, nf], parts[k_early, nc],
                                                  n_shuffle if primary else n_shuffle_sens, rng)
                        nest.append({"model": name[0], "seed": name[1], "fine_clusters": nf, "coarse_clusters": nc,
                                     "primary": primary, "k_fine": k_late, "k_coarse": k_early, "containment": obs,
                                     "shuffled_mean": sh.mean(), "shuffled_sd": sh.std(ddof=1),
                                     "containment_minus_shuffled": obs - sh.mean()})
    return truth, stab, nest, parts


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objective", choices=("ae", "mrl", "random"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classifier", type=Path, help="default: <run-dir>/classifier")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--bounded", action="store_true", help="models with tanh on z (must match the checkpoints)")
    args = ap.parse_args()
    args.classifier = args.classifier or args.run_dir / "classifier"
    start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    tag = f"{args.objective}_seed{args.seed}"
    name = (args.objective, args.seed)
    c_grid = [1e-2, 1.0] if args.fast else PROBE_C
    n_init, n_shuf, n_shuf_s = (2, 5, 2) if args.fast else (KMEANS_N_INIT, N_SHUFFLE, N_SHUFFLE_SENSITIVITY)
    pairs_per_type = 2000 if args.fast else PAIRS_PER_TYPE

    data = {s: load_split(s) for s in ("train", "val", "test")}
    if args.fast:
        data = {s: (im[::8], {k: v[::8] for k, v in lab.items()}) for s, (im, lab) in data.items()}
    labels = {s: lab for s, (_, lab) in data.items()}
    model = load_model(args, device)

    metrics = json.loads((args.classifier / "metrics.json").read_text())
    classifier, clf_note = None, "gate failed: reconstruction classifier scores are NaN"
    if metrics["gate_passed"]:
        if file_sha256(args.classifier / "classifier.pt") != metrics["checkpoint_sha256"]:
            raise ValueError("classifier checkpoint does not match its recorded checksum")
        classifier = ReconClassifier().to(device).eval()
        classifier.load_state_dict(torch.load(args.classifier / "classifier.pt", map_location=device, weights_only=True)["state_dict"])
        clf_note = "gate passed"

    z = {s: embed(model, data[s][0], device) for s in data}
    emb_dir = args.run_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    print(f"{tag}: embedded ({time.time() - start:.0f}s)", flush=True)

    rec = reconstructions(model, data["test"][0], labels["test"], classifier, device, name)
    write_csv(args.run_dir / "reconstructions" / f"{tag}.csv", rec)
    grid(model, data["test"][0], labels["test"], device, args.run_dir / "reconstructions" / f"grid_{tag}.png", tag)
    print(f"{tag}: reconstructions ({time.time() - start:.0f}s)", flush=True)

    rows, aucs, pairs = distances(z["test"], labels["test"], name, pairs_per_type)
    write_csv(args.run_dir / "distances" / f"dist_{tag}.csv", rows)
    write_csv(args.run_dir / "distances" / f"auc_{tag}.csv", aucs)
    print(f"{tag}: distances ({time.time() - start:.0f}s)", flush=True)

    truth, stab, nest, parts = clustering(z["test"], labels["test"], name, n_init, n_shuf, n_shuf_s)
    write_csv(args.run_dir / "clustering" / f"truth_ari_{tag}.csv", truth)
    write_csv(args.run_dir / "clustering" / f"stability_{tag}.csv", stab)
    write_csv(args.run_dir / "clustering" / f"nesting_{tag}.csv", nest)
    np.savez_compressed(emb_dir / f"{tag}.npz", z_test=z["test"].astype(np.float32),
                        **{f"part_k{k}_n{n}": p.astype(np.int16) for (k, n), p in parts.items()})
    print(f"{tag}: clustering ({time.time() - start:.0f}s)", flush=True)

    write_csv(args.run_dir / "probes" / f"{tag}.csv", probes(z, labels, name, c_grid))
    print(f"{tag}: probes ({time.time() - start:.0f}s)", flush=True)

    status = {"model": tag, "classifier": str(args.classifier), "classifier_note": clf_note,
              "classifier_test_accuracy": metrics["test_accuracy"], "fast": args.fast,
              "seconds": round(time.time() - start, 1)}
    (args.run_dir / "probes" / f"status_{tag}.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"{tag}: done in {status['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
