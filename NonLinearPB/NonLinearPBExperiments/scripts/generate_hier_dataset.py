"""Generate the 3 x 3 x 3 hierarchical synthetic image dataset and its validation tables.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/scripts/generate_hier_dataset.py
    python NonLinearPB/NonLinearPBExperiments/scripts/generate_hier_dataset.py --per-leaf 200 --output /tmp/hier_smoke
    python NonLinearPB/NonLinearPBExperiments/scripts/generate_hier_dataset.py --final   # frozen VERSION into data/

Spec: NonLinearPB/plans/hierarchical_synthetic_data_spec.md (sections 1, 3-6).
Without --final, output goes to files/results/hier_synthetic_pilot/data_<stamp>/;
--final writes data/synthetic_data/hierarchical_images_<VERSION>/ and refuses to overwrite.
v1 (2026-11-02) is frozen; v2 gives dumbbell "straight" and ring "stem below" their own fine details.
Each split has its own seed stream; every sample's nuisance comes from its own
recorded seed. Cost ladder pairs use a separate calibration stream, never test.
Images are stored as uint8; load as float32 / 255.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_synthetic import render as R

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PILOT_ROOT = PROJECT_ROOT / "files/results/hier_synthetic_pilot"
VERSION = "v2"
FINAL_DIR = PROJECT_ROOT / f"data/synthetic_data/hierarchical_images_{VERSION}"

# CONFIG: spec section 1 sizes and independent seed streams.
SPLITS = {"train": 1600, "val": 200, "test": 200}
SPLIT_SEEDS = {"train": 1001, "val": 2002, "test": 3003}
CALIBRATION_SEED = 4004
CALIBRATION_REPLICATES = 100        # nuisance seeds per leaf for the cost ladder
LEAVES = [(c, m, f) for c in range(3) for m in range(3) for f in range(3)]


def leaf_id(c, m, f):
    return 9 * c + 3 * m + f


def to_uint8(img):
    return np.round(img * 255).astype(np.uint8)


def render_sample(c, m, f, seed):
    return to_uint8(R.render(c, m, f, R.sample_nuisance(np.random.default_rng(seed))))


def make_split(name, per_leaf):
    """Images and labels in fixed (leaf, index) order; one nuisance seed per sample."""
    rng = np.random.default_rng(SPLIT_SEEDS[name])
    seeds = rng.choice(2**62, size=len(LEAVES) * per_leaf, replace=False)
    images = np.empty((len(seeds), R.SIZE, R.SIZE), np.uint8)
    labels = np.empty((len(seeds), 4), np.int64)
    for i, (c, m, f) in enumerate(LEAVES):
        for j in range(per_leaf):
            k = i * per_leaf + j
            images[k] = render_sample(c, m, f, int(seeds[k]))
            labels[k] = (c, m, f, leaf_id(c, m, f))
    return images, labels, seeds


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)) / 255.0 ** 2


def cost_ladder(replicates):
    """Matched-pair MSE per level: same nuisance seed, exactly one label changed."""
    seeds = np.random.default_rng(CALIBRATION_SEED).choice(2**62, size=replicates + 1, replace=False)
    grid = {}
    for r, s in enumerate(seeds):
        for c, m, f in LEAVES:
            grid[r, c, m, f] = render_sample(c, m, f, int(s))
    rows = []   # (level, parent, pair, cost)
    for r in range(replicates):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            for x in range(3):
                for y in range(3):
                    rows.append(("coarse", f"m={x},f={y}", f"c{a}-c{b}", mse(grid[r, a, x, y], grid[r, b, x, y])))
                    rows.append(("mid", f"c={x}", f"m{a}-m{b}", mse(grid[r, x, a, y], grid[r, x, b, y])))
                    rows.append(("fine", f"c={x},m={y}", f"f{a}-f{b}", mse(grid[r, x, y, a], grid[r, x, y, b])))
        for c, m, f in LEAVES:      # nuisance: same leaf, consecutive independent seeds
            rows.append(("nuisance", f"c={c},m={m},f={f}", "seed-seed", mse(grid[r, c, m, f], grid[r + 1, c, m, f])))
    return rows


def single_pair_overlap(rows):
    """Per adjacent levels: share of single upper-level pairs cheaper than the costliest lower-level pair."""
    cost = {lv: np.array([c for l, _, _, c in rows if l == lv]) for lv in ("coarse", "mid", "fine", "nuisance")}
    out = {}
    for hi, lo in (("coarse", "mid"), ("mid", "fine"), ("fine", "nuisance")):
        out[f"{hi}_vs_{lo}"] = {"min_" + hi: float(cost[hi].min()), "max_" + lo: float(cost[lo].max()),
                                "frac_" + hi + "_below_max_" + lo: float(np.mean(cost[hi] < cost[lo].max())),
                                "p_random_" + hi + "_below_" + lo: float(np.mean(cost[hi][:, None] < cost[lo][None, :]))}
    return out


def summarize(rows):
    out, levels = [], ("coarse", "mid", "fine", "nuisance")
    for level in levels:
        groups = {"all": []}
        for lv, parent, pair, cost in rows:
            if lv == level:
                groups["all"].append(cost)
                groups.setdefault(parent, []).append(cost)
        for parent, costs in groups.items():
            v = np.array(costs)
            out.append({"level": level, "parent": parent, "n": len(v), "mean": v.mean(), "var": v.var(ddof=1),
                        "ci95": 1.96 * v.std(ddof=1) / np.sqrt(len(v)), "min": v.min(), "max": v.max()})
    return out


def conditional_variance(images, labels):
    """Pixel MSE of predicting each image by its root / coarse / mid / leaf group mean."""
    x = images.astype(np.float64) / 255.0
    out, prev = [], None
    for level, cols in (("root", []), ("coarse", [0]), ("mid", [0, 1]), ("leaf", [0, 1, 2])):
        keys = [tuple(r) for r in labels[:, cols]] if cols else [()] * len(x)
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        err = sum(((x[idx] - x[idx].mean(axis=0)) ** 2).sum() for idx in groups.values()) / x.size
        out.append({"known_up_to": level, "n_groups": len(groups), "mse": err,
                    "reduction_vs_previous": None if prev is None else prev - err})
        prev = err
    return out


def contact_sheet(path, per_leaf_images, n):
    """Rows: the 9 (c, m) parents; columns: 3 fine leaves x n samples, with gaps."""
    pad, gap = 2, 8
    tile = R.SIZE + 2 * pad
    w = 3 * n * tile + 2 * gap
    rows = []
    for c in range(3):
        for m in range(3):
            row = np.full((tile, w), 90, np.uint8)
            for f in range(3):
                for j in range(n):
                    x0 = (f * n + j) * tile + f * gap + pad
                    row[pad:pad + R.SIZE, x0:x0 + R.SIZE] = per_leaf_images[leaf_id(c, m, f)][j]
            rows.append(row)
        rows.append(np.full((gap, w), 255, np.uint8))
    sheet = np.concatenate(rows[:-1], axis=0)
    Image.fromarray(sheet).resize((sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(path)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-leaf", type=int, default=None, help="override total images per leaf (smoke runs)")
    ap.add_argument("--replicates", type=int, default=CALIBRATION_REPLICATES)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--final", action="store_true", help=f"write the frozen dataset to {FINAL_DIR}")
    args = ap.parse_args()

    if args.final and (args.output or args.per_leaf):
        ap.error("--final uses the fixed location and spec sizes")
    out = FINAL_DIR if args.final else args.output or PILOT_ROOT / f"data_{datetime.now():%Y_%m_%d__%H_%M_%S}"
    if args.final and out.exists():
        sys.exit(f"{out} exists; a changed generator must get a new VERSION, not overwrite {VERSION}")
    out.mkdir(parents=True, exist_ok=not args.final)
    splits = SPLITS if args.per_leaf is None else {
        k: max(1, round(v * args.per_leaf / 2000)) for k, v in SPLITS.items()}
    t0 = time.time()

    # 1. Splits.
    data, hashes, all_seeds, meta_rows = {}, {}, [], []
    for name, per_leaf in splits.items():
        images, labels, seeds = make_split(name, per_leaf)
        np.savez_compressed(out / f"{name}.npz", images=images, coarse=labels[:, 0], mid=labels[:, 1],
                            fine=labels[:, 2], leaf_id=labels[:, 3], nuisance_seed=seeds)
        data[name] = (images, labels)
        all_seeds.append(seeds)
        hashes[name] = {hashlib.sha1(img.tobytes()).hexdigest() for img in images}
        meta_rows += [{"sample_id": f"{name}_{k:05d}", "split": name, "coarse": l[0], "mid": l[1], "fine": l[2],
                       "leaf_id": l[3], "nuisance_seed": int(s)} for k, (l, s) in enumerate(zip(labels, seeds))]
        counts = np.bincount(labels[:, 3], minlength=27)
        print(f"{name}: {len(images)} images, per-leaf counts {counts.min()}..{counts.max()}, "
              f"{len(hashes[name])} unique, {time.time() - t0:.0f}s", flush=True)
    write_csv(out / "metadata.csv", meta_rows)

    # 2. Split integrity: no reused nuisance seeds, no identical images across splits.
    seeds_all = np.concatenate(all_seeds)
    checks = {
        "seeds_unique_across_splits": bool(len(np.unique(seeds_all)) == len(seeds_all)),
        "duplicate_images_within_splits": {k: len(data[k][0]) - len(h) for k, h in hashes.items()},
        "images_shared_train_val": len(hashes["train"] & hashes["val"]),
        "images_shared_train_test": len(hashes["train"] & hashes["test"]),
        "images_shared_val_test": len(hashes["val"] & hashes["test"]),
        "leaves_balanced": all(len(set(np.bincount(data[k][1][:, 3], minlength=27))) == 1 for k in data),
    }
    print("integrity:", json.dumps(checks), flush=True)

    # 3. Cost ladder on a separate calibration stream.
    pairs = cost_ladder(args.replicates)
    summary = summarize(pairs)
    write_csv(out / "cost_validation.csv", summary)
    means = {s["level"]: s["mean"] for s in summary if s["parent"] == "all"}
    ladder = {
        "delta": means,
        "ratio_to_fine": {k: v / means["fine"] for k, v in means.items()},
        "ordering_holds": bool(means["coarse"] > means["mid"] > means["fine"] > means["nuisance"]),
        "single_pair_overlap": single_pair_overlap(pairs),
        "nuisance_below_quarter_fine": bool(means["nuisance"] < 0.25 * means["fine"]),
        "note": "costs are nested: coarse pairs also change mid/fine geometry, mid pairs also change the fine detail",
    }
    print("cost ladder:", json.dumps(ladder, indent=2), flush=True)

    # 4. Conditional variance on train only.
    cv_rows = conditional_variance(*data["train"])
    write_csv(out / "conditional_variance.csv", cv_rows)
    red = [r["reduction_vs_previous"] for r in cv_rows[1:]]
    cv_check = {"gains": dict(zip(("coarse", "mid", "fine"), red)), "residual_leaf_mse": cv_rows[-1]["mse"],
                "ordering_holds": bool(red[0] > red[1] > red[2] > cv_rows[-1]["mse"])}
    print("conditional variance:", json.dumps(cv_check), flush=True)

    # 5. Contact sheet from train.
    images, labels = data["train"]
    n_show = min(4, splits["train"])
    per_leaf = {lid: images[labels[:, 3] == lid][:n_show] for lid in range(27)}
    contact_sheet(out / "contact_sheet.png", per_leaf, n_show)

    config = {
        "version": VERSION, "frozen": args.final, "created": datetime.now().isoformat(timespec="seconds"),
        "image_size": R.SIZE, "supersample": R.SUPERSAMPLE, "storage": "uint8; float32 = images / 255",
        "per_leaf": splits, "split_seeds": SPLIT_SEEDS, "calibration_seed": CALIBRATION_SEED,
        "calibration_replicates": args.replicates,
        "nuisance": {"jitter_px": R.JITTER_PX, "brightness": R.BRIGHTNESS, "noise_sigma": R.NOISE_SIGMA,
                     "translation": "none"},
        "cost": "pixel MSE on [0,1] images", "cost_targets": {"coarse:mid:fine": "4:2:1", "nuisance": "< 0.25 x fine"},
        "names": {"coarse": R.COARSE_NAMES, "mid": R.MID_NAMES, "fine": R.FINE_NAMES},
        "integrity": checks, "cost_ladder": ladder, "conditional_variance": cv_check,
        "sha256": {f"{k}.npz": sha256(out / f"{k}.npz") for k in splits},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rel = Path(__file__).resolve().relative_to(PROJECT_ROOT / "code")
    (out / "README.md").write_text(
        f"# Hierarchical synthetic images ({VERSION})\n\n"
        f"3 coarse x 3 mid x 3 fine = 27 leaves, 64 x 64 grayscale, uint8 (divide by 255).\n"
        f"Spec: `code/NonLinearPB/plans/hierarchical_synthetic_data_spec.md`.\n\n"
        f"Regenerate exactly (from `code/`):\n\n```bash\n"
        f"python {rel}{' --final' if args.final else ''}"
        f"{f' --per-leaf {args.per_leaf}' if args.per_leaf else ''}\n```\n\n"
        f"Seeds, nuisance settings, cost ladder, integrity checks and file checksums are in `config.json`.\n"
        f"npz keys: images, coarse, mid, fine, leaf_id, nuisance_seed.\n")
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
