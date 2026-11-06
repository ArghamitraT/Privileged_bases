"""Tier 1 frozen-encoder target alignment: where do target directions sit in a pretrained encoder's coordinate order?

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/tier1_target_alignment.py --run-dir /path/to/exprmnt_x --betas 0 --gpus 0 1
    python NonLinearPB/NonLinearPBExperiments/scripts/tier1_target_alignment.py --run-dir /path/to/exprmnt_x --jobs recon_beta0_seed0 --device cuda:1
    python NonLinearPB/NonLinearPBExperiments/scripts/tier1_target_alignment.py --run-dir /path/to/exprmnt_x --betas 0 --combine-only

Spec: NonLinearPB/plans/FrozenEncoderTargetAlignmentExp.md. No encoder training.
For each frozen encoder, outputs go to <job>/tier1_alignment_<stamp>/:
  - posterior means mu for train / validation / test, cached separately; the test
    split is cached but never scored. mu is standardized per coordinate with the
    training mean and std (keeps coordinate order); raw mu is a sensitivity check;
  - six targets: alpha in {0, 30, 60, 90} deg, 6 equal-frequency classes of
    round(cos(alpha) X + sin(alpha) Y, 9) with cut points at the train quantiles
    1/6..5/6 (method "higher"), a score equal to a cut point goes to the upper
    class; scale (6 classes); shape (3 classes);
  - Method 1 (primary): eigenvectors of the equal-class between-class scatter of
    the training embeddings, S_B = sum_c (mu_c - mean_c mu_c)(...)^T; rank =
    eigenvalues > RANK_TOLERANCE * largest, at most C - 1. A(1) = m1[1]^2,
    O(k, j) = ||P_k M_j||_F^2 / min(k, j), leading and eigenvalue-weighted depth,
    concentration lambda1 / sum(lambda > 0). Frequency-weighted scatter and raw
    mu are recorded as sensitivity checks;
  - nulls for O(k, j): uniform random rank-dim subspaces, either in all 10
    coordinates ("all10") or inside the active coordinates of one activity rule
    (variance, recon_gain, kl), each rule kept separate; masks come from the
    job's newest pretraining analysis;
  - Method 2 (secondary): orthogonal Q_T = Q0 expm(S - S^T) with one linear head
    per prefix, equal-weight mean of prefix cross-entropy k = 1..10, encoder
    frozen; Q_RESTARTS restarts, each from its own random orthogonal Q0;
    early stopping on validation loss (vae_reproduction.bases.fit_nested_rotation);
  - fresh linear prefix probes (L-BFGS, PROBE_TRAIN training images, full
    validation) in four bases: original order, scatter (PCA of the remaining
    space after the rank directions; only k <= rank is interpretable), each
    learned Q_T, and PROBE_NULL_DRAWS random orthogonal matrices. Validation
    balanced accuracy, chance-normalized (b - 1/C)/(1 - 1/C); k* = smallest k
    with normalized(k) >= 0.9 * normalized(10), flagged uninformative when
    normalized(10) < UNINFORMATIVE_GAIN; ordinal R^2 against the continuous
    angle score; principal angles between scatter and Q_T subspaces and between
    Q_T restarts.
--gpus runs one subprocess per job (round-robin over the GPUs); every job writes
its own folder as soon as it finishes. Several jobs also write
<run-dir>/tier1_combined_<stamp>/. --combine-only reuses each job's newest saved
result. Hypothesis inputs are reported as numbers only, no verdicts.
Each job folder also gets the Method 2 contribution matrices C = Q_T^2 and
scores from scripts/tier1_contribution.py (also runnable standalone).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import torch

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE_ROOT))
from utility import save_runtime
from vae_reproduction.bases import (_fix_signs, fit_nested_rotation, head_accuracies, orthogonality_error,
                                    principal_angles, random_orthogonal)
from vae_reproduction.model import model_from_config
from vae_reproduction.probes import fit_linear_probe, fit_linear_regression
from vae_reproduction.runs import DATA_PATH, write_json
from vae_reproduction.tasks import split_checksum, subsample
from scripts.analyze_pretrain import encode, latest_analysis, load_context
from scripts.tier1_contribution import contribution_job

# ==============================================================================
# CONFIG
# ==============================================================================
ANGLES = (0, 30, 60, 90)       # degrees; score = cos(a) X + sin(a) Y
N_BINS = 6                     # equal-frequency classes per angle target
TARGETS = tuple(f"alpha{a}" for a in ANGLES) + ("scale", "shape")
STD_FLOOR = 1e-6               # standardization: std clamped from below
RANK_TOLERANCE = 1e-3          # scatter eigenvalue / largest eigenvalue
ACTIVE_RULES = ("variance", "recon_gain", "kl")   # kept separate, never mixed
NULL_DRAWS = 1000              # random subspaces per null convention
PROBE_NULL_DRAWS = 10          # random orthogonal bases for the probe null
PROBE_TRAIN = 50000            # probe training images (subset of the train split)
PROBE_SUBSET_SEED = 0          # same subset as analyze_pretrain.py
Q_RESTARTS = 5
Q_INIT_SEED = 100              # restart r starts from random_orthogonal(d, Q_INIT_SEED + r)
NULL_SEED = 2026
K_STAR_FRACTION = 0.9
UNINFORMATIVE_GAIN = 0.05      # chance-normalized full-probe accuracy below this: k* uninformative
SMOKE = {"probe_train": 4000, "validation": 4000, "rotation_train": 8192, "null_draws": 50,
         "probe_null_draws": 2, "q_restarts": 2, "rotation_config": {"max_epochs": 3}}
# ==============================================================================


# ------------------------------------------------------------------ targets ---
def angle_scores(factors, angle):
    radians = np.deg2rad(angle)
    return np.round(np.cos(radians) * factors[:, 3] + np.sin(radians) * factors[:, 4], 9)


def equal_frequency_labels(scores, train, n_bins=N_BINS):
    """Cut points from training scores only; ties at a cut point go to the upper class."""
    edges = np.quantile(scores[train], np.arange(1, n_bins) / n_bins, method="higher")
    if len(np.unique(edges)) != len(edges):
        raise ValueError(f"Duplicate cut points {edges}")
    return np.searchsorted(edges, scores, side="right").astype(np.int64), edges


def make_targets(factors, split):
    """{target: labels for every archive image, classes, continuous score (angles), cut points}."""
    targets = {}
    for angle in ANGLES:
        scores = angle_scores(factors, angle)
        labels, edges = equal_frequency_labels(scores, split["train"])
        targets[f"alpha{angle}"] = {"labels": labels, "classes": N_BINS, "score": scores, "edges": edges.tolist()}
    targets["scale"] = {"labels": factors[:, 1].astype(np.int64), "classes": 6, "score": None, "edges": None}
    targets["shape"] = {"labels": factors[:, 0].astype(np.int64), "classes": 3, "score": None, "edges": None}
    return targets


def label_record(target, split):
    """Cut points and class balance; train and validation only (test stays untouched)."""
    record = {"classes": target["classes"], "edges": target["edges"],
              "tie_rule": "score equal to a cut point goes to the upper class" if target["edges"] else None}
    for name in ("train", "validation"):
        counts = np.bincount(target["labels"][split[name]], minlength=target["classes"])
        share = counts / counts.sum()
        record[f"{name}_counts"] = counts.tolist()
        record[f"{name}_normalized_entropy"] = float(
            -(share[share > 0] * np.log(share[share > 0])).sum() / np.log(len(share)))
    return record


# ------------------------------------------------------- Method 1: scatter ---
def scatter_directions(z, labels, classes, weighting="equal"):
    """Eigenpairs of the between-class scatter, largest first; returns (values, vectors, rank)."""
    z = np.asarray(z, dtype=np.float64)
    means = np.stack([z[labels == c].mean(0) for c in range(classes)])
    if weighting == "equal":
        deviation = means - means.mean(0)
        between = deviation.T @ deviation
    elif weighting == "frequency":
        counts = np.bincount(labels, minlength=classes)
        deviation = means - z.mean(0)
        between = (deviation * (classes * counts / counts.sum())[:, None]).T @ deviation
    else:
        raise ValueError(weighting)
    values, vectors = np.linalg.eigh(between)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], _fix_signs(vectors[:, order])
    rank = int(min(classes - 1, np.sum(values > RANK_TOLERANCE * values[0])))
    return values, vectors, rank


def overlap(m):
    """O[k-1, j-1] = ||P_k M_j||_F^2 / min(k, j) for orthonormal columns m (d x r)."""
    d, r = m.shape
    captured = np.cumsum(np.cumsum(m ** 2, axis=0), axis=1)
    return captured / np.minimum(np.arange(1, d + 1)[:, None], np.arange(1, r + 1)[None])


def geometry(m, values=None):
    """Alignment of the columns of m (d x r) with the original coordinate order."""
    positions = np.arange(1, len(m) + 1)
    depth = (positions[:, None] * m ** 2).sum(0)
    o = overlap(m)
    out = {"A1": float(m[0, 0] ** 2), "O": o.tolist(),
           "A_k": [float(o[k, k]) for k in range(m.shape[1])], "depth_leading": float(depth[0])}
    if values is not None:
        weights = values[:m.shape[1]] / values[:m.shape[1]].sum()
        out["depth_weighted"] = float(weights @ depth)
    return out


def scatter_record(z, labels, classes, weighting="equal"):
    values, vectors, rank = scatter_directions(z, labels, classes, weighting)
    positive = values[values > 0]
    return {"weighting": weighting, "rank": rank, "eigenvalues": values.tolist(),
            "concentration": float(values[0] / positive.sum()),
            "directions": vectors[:, :rank].tolist(), **geometry(vectors[:, :rank], values)}, vectors[:, :rank]


def completed_basis(lead, x):
    """lead (d x r) followed by the PCA directions of x inside its orthogonal complement."""
    complement = linalg.null_space(lead.T)
    if complement.shape[1] == 0:
        return lead
    inside = (x - x.mean(0)) @ complement
    values, vectors = np.linalg.eigh(np.cov(inside, rowvar=False).reshape(complement.shape[1], -1))
    return np.hstack([lead, complement @ vectors[:, np.argsort(values)[::-1]]])


# -------------------------------------------------------------------- nulls ---
def null_conventions(activity, d):
    """{name: 0/1 mask}: all coordinates, plus each available activity rule separately."""
    masks = {"all10": [1] * d}
    for rule in ACTIVE_RULES:
        if f"{rule}_mask" in activity:
            masks[rule] = list(activity[f"{rule}_mask"])
    return masks


def overlap_null(rank, mask, draws, seed):
    """O(k, j) of uniform random rank-dim subspaces inside span(e_i : mask_i = 1); draws x d x min(rank, support)."""
    d = len(mask)
    support = np.eye(d)[:, np.flatnonzero(mask)]
    j = min(rank, support.shape[1])
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, d, j))
    for i in range(draws):
        q, _ = np.linalg.qr(rng.standard_normal((support.shape[1], j)))
        samples[i] = overlap(support @ q)
    return samples


def null_summary(observed, samples, mask):
    """Null distribution of O(k, j); upper_tail = P(null >= observed) with a +1 correction."""
    j = samples.shape[2]
    hits = (samples >= np.asarray(observed)[None, :, :j] - 1e-12).sum(0)
    return {"support": int(sum(mask)), "mask": list(mask), "ranks_covered": j,
            "mean": samples.mean(0).tolist(), "p05": np.quantile(samples, .05, axis=0).tolist(),
            "p95": np.quantile(samples, .95, axis=0).tolist(),
            "upper_tail": ((hits + 1) / (len(samples) + 1)).tolist()}


# -------------------------------------------------------------------- probes ---
def curve_record(balanced, accuracy, r2, classes):
    chance = 1 / classes
    normalized = [(b - chance) / (1 - chance) for b in balanced]
    full = normalized[-1]
    k_star = next(k for k, v in enumerate(normalized, 1) if v >= K_STAR_FRACTION * full)
    return {"balanced_accuracy": balanced, "accuracy": accuracy, "normalized": normalized, "r2": r2,
            "chance": chance, "k_star": k_star, "k_star_informative": bool(full >= UNINFORMATIVE_GAIN)}


def prefix_curve(data, basis, classes, device):
    """Fresh linear probes on the first k coordinates of z @ basis, k = 1..d; validation scores."""
    (x_train, y_train, s_train), (x_val, y_val, s_val) = data["train"], data["validation"]
    basis = np.asarray(basis, dtype=np.float32)
    z_train, z_val = x_train @ basis, x_val @ basis
    balanced, accuracy, r2 = [], [], []
    for k in range(1, basis.shape[1] + 1):
        result = fit_linear_probe((z_train[:, :k], y_train), (z_val[:, :k], y_val), None, classes, device=device)
        balanced.append(result["validation"]["balanced_accuracy"])
        accuracy.append(result["validation"]["accuracy"])
        if s_train is not None:
            r2.append(fit_linear_regression((z_train[:, :k], s_train[:, None]),
                                            (z_val[:, :k], s_val[:, None]), None)["validation"])
    return curve_record(balanced, accuracy, r2 or None, classes)


def max_angle(a, b):
    """Largest principal angle (degrees) between the column spans of a and b."""
    return float(np.degrees(principal_angles(a, b).max()))


# --------------------------------------------------------- one target, one job ---
def analyze_target(target, z, raw, split, probe_split, rotation_split, masks, device, settings):
    labels, classes, score = target["labels"], target["classes"], target["score"]
    train = split["train"]
    record = {"labels": label_record(target, split)}

    scatter, lead = scatter_record(z[train], labels[train], classes)
    rank = scatter["rank"]
    record["scatter"] = scatter
    record["sensitivity"] = {
        "frequency_weighted": scatter_record(z[train], labels[train], classes, "frequency")[0],
        "raw_mu": scatter_record(raw[train], labels[train], classes)[0]}
    for item in record["sensitivity"].values():
        item.pop("directions")
    nulls, null_draws = {}, {}
    for convention, mask in masks.items():
        if sum(mask) == 0:
            continue
        samples = overlap_null(rank, mask, settings["null_draws"], NULL_SEED)
        nulls[convention] = null_summary(scatter["O"], samples, mask)
        null_draws[convention] = samples[:, 0, 0]          # A(1) draws
    record["null"] = nulls

    data = {part: (z[idx], labels[idx], None if score is None else score[idx])
            for part, idx in probe_split.items()}
    d = z.shape[1]
    access = {"original": prefix_curve(data, np.eye(d), classes, device),
              "scatter": prefix_curve(data, completed_basis(lead, z[train]), classes, device)}
    access["scatter"]["interpretable_up_to_k"] = rank
    access["random"] = [prefix_curve(data, random_orthogonal(d, NULL_SEED + i), classes, device)
                        for i in range(settings["probe_null_draws"])]

    rotation_train = (z[rotation_split["train"]], labels[rotation_split["train"]])
    rotation_val = (z[rotation_split["validation"]], labels[rotation_split["validation"]])
    learned, qs = [], []
    for restart in range(settings["q_restarts"]):
        q, heads, center, log = fit_nested_rotation(
            rotation_train, rotation_val, classes, range(1, d + 1), random_orthogonal(d, Q_INIT_SEED + restart),
            device=device, seed=restart, config=settings["rotation_config"])
        log.pop("history")
        curve = prefix_curve(data, q, classes, device)
        heads_val = head_accuracies(q, heads, range(1, d + 1), center, rotation_val, device)
        learned.append({"restart": restart, "init_seed": Q_INIT_SEED + restart, "fit": log,
                        "head_validation_accuracy": heads_val, "curve": curve,
                        "geometry": geometry(q[:, :rank]),
                        "max_angle_to_scatter_deg": [max_angle(q[:, :j], lead[:, :j]) for j in range(1, rank + 1)]})
        qs.append(q)
    access["learned"] = [item["curve"] for item in learned]
    record["learned_rotation"] = learned
    record["restart_agreement_deg"] = {
        f"{a}-{b}": [max_angle(qs[a][:, :j], qs[b][:, :j]) for j in range(1, rank + 1)]
        for a, b in combinations(range(len(qs)), 2)}
    full = [access["original"]["balanced_accuracy"][-1], access["scatter"]["balanced_accuracy"][-1]] + \
           [c["balanced_accuracy"][-1] for c in access["learned"] + access["random"]]
    record["full_probe_spread"] = float(max(full) - min(full))   # orthogonal bases: should be ~0
    record["access"] = access
    return record, {"null_A1": null_draws, "Q": np.stack(qs)}


# ------------------------------------------------------------------- one job ---
def analyze_job(job_dir, context, device, smoke=False):
    """Tier 1 for one frozen encoder; outputs to <job_dir>/tier1_alignment_<stamp>/. Returns its result."""
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    job_dir = Path(job_dir)
    job = job_dir.name
    folder = job_dir / f"tier1_alignment_{stamp}{'_smoke' if smoke else ''}"
    folder.mkdir()
    print(f"{job}: tier 1 target alignment", flush=True)
    settings = {"probe_train": PROBE_TRAIN, "validation": None, "rotation_train": None, "null_draws": NULL_DRAWS,
                "probe_null_draws": PROBE_NULL_DRAWS, "q_restarts": Q_RESTARTS, "rotation_config": None}
    if smoke:
        settings.update(SMOKE)

    # Stage 0: frozen model, split check, activity masks, embeddings.
    state = torch.load(job_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    with np.load(job_dir / "split_manifest.npz") as manifest:
        split = {name: manifest[name] for name in ("train", "validation", "test")}
    if split_checksum(split) != config["split"]["sha256"]:
        raise ValueError(f"{job}: split manifest does not match its config checksum")
    pretrain = latest_analysis(job_dir)
    if pretrain is None:
        raise FileNotFoundError(f"{job}: no pretraining analysis (activity masks); run analyze_pretrain.py first")
    model = model_from_config(config).to(device).eval()
    model.load_state_dict(state["model"])
    raw, _ = encode(model, context["images"], device)
    center = raw[split["train"]].mean(0)
    scale = np.maximum(raw[split["train"]].std(0), STD_FLOOR)
    z = ((raw - center) / scale).astype(np.float32)
    np.savez(folder / "embeddings_mu.npz", **{f"{name}_indices": split[name] for name in split},
             **{f"{name}_mu": raw[split[name]] for name in split}, train_mean=center, train_std=scale)
    d = z.shape[1]
    masks = null_conventions(pretrain["activity"], d)

    probe_split = {"train": subsample(split["train"], settings["probe_train"], PROBE_SUBSET_SEED),
                   "validation": subsample(split["validation"], settings["validation"], 1)}
    rotation_split = {"train": subsample(split["train"], settings["rotation_train"], 0),
                      "validation": probe_split["validation"]}
    targets = make_targets(context["factors"], split)
    result = {"objective": config["objective"], "beta": config["beta"], "seed": config["seed"],
              "lambda_sup": config["lambda_sup"], "split_sha256": config["split"]["sha256"],
              "standardization": {"train_mean": center.tolist(), "train_std": scale.tolist()},
              "activity_masks": masks, "targets": {}}
    arrays = {}
    for name in TARGETS:
        print(f"{job}: {name}", flush=True)
        result["targets"][name], extra = analyze_target(
            targets[name], z, raw, split, probe_split, rotation_split, masks, device, settings)
        arrays[f"{name}_Q"] = extra["Q"]
        for convention, draws in extra["null_A1"].items():
            arrays[f"{name}_null_A1_{convention}"] = draws
        arrays[f"{name}_labels"] = targets[name]["labels"].astype(np.int8)
    np.savez(folder / "arrays.npz", **arrays)
    write_outputs({job: result}, folder, stamp, {"job": job, "smoke": smoke, "settings": settings,
                                                "device": str(device), **config_record()})
    contribution_job(folder, folder, stamp)          # Method 2 contribution matrices, same folder
    save_runtime(str(folder), time.perf_counter() - started)
    print(f"{job}: tier 1 done -> {folder}", flush=True)
    return result


def latest_tier1(job_dir):
    folders = sorted((p for p in Path(job_dir).glob("tier1_alignment_*") if not p.name.endswith("_smoke")),
                     key=lambda p: p.name, reverse=True)
    for folder in folders:
        path = folder / "tier1.json"
        if path.exists():
            return json.loads(path.read_text())[Path(job_dir).name]
    return None


def config_record():
    return {"angles": ANGLES, "n_bins": N_BINS, "targets": TARGETS, "std_floor": STD_FLOOR,
            "rank_tolerance": RANK_TOLERANCE, "active_rules": ACTIVE_RULES, "null_seed": NULL_SEED,
            "probe_subset_seed": PROBE_SUBSET_SEED, "q_init_seed": Q_INIT_SEED,
            "k_star_fraction": K_STAR_FRACTION, "uninformative_gain": UNINFORMATIVE_GAIN}


# ------------------------------------------------------------------- outputs ---
def short(job):
    return job.replace("_lam100", "").replace("_beta", " b").replace("_seed", " s")


def figures(results, folder, stamp):
    jobs = list(results)
    reference = [np.cos(np.deg2rad(a)) ** 2 for a in ANGLES]

    fig, axes = plt.subplots(1, len(jobs), figsize=(3.4 * len(jobs), 3.4), sharey=True, squeeze=False)
    for ax, job in zip(axes[0], jobs):
        t = results[job]["targets"]
        names = [f"alpha{a}" for a in ANGLES]
        low = [t[n]["null"]["all10"]["p05"][0][0] for n in names]
        high = [t[n]["null"]["all10"]["p95"][0][0] for n in names]
        ax.fill_between(ANGLES, low, high, color="0.85", label="null all10 (5–95%)")
        for rule, style in (("variance", ":"), ("recon_gain", "-."), ("kl", "--")):
            if all(rule in t[n]["null"] for n in names):
                ax.plot(ANGLES, [t[n]["null"][rule]["mean"][0][0] for n in names], style, color="0.5",
                        linewidth=1, label=f"null {rule} (mean)")
        ax.plot(ANGLES, reference, "--", color="black", linewidth=1, label="cos²α (idealized)")
        for restart in range(len(t[names[0]]["learned_rotation"])):
            ax.plot(ANGLES, [t[n]["learned_rotation"][restart]["geometry"]["A1"] for n in names], "x",
                    color="tab:orange", markersize=4, label="learned Q_T" if restart == 0 else None)
        ax.plot(ANGLES, [t[n]["scatter"]["A1"] for n in names], "o-", color="tab:blue", label="scatter m1")
        ax.set_title(short(job), fontsize=9)
        ax.set_xticks(ANGLES)
        ax.set_xlabel("α (deg)")
    axes[0][0].set_ylabel("A(1) = m1[1]²")
    axes[0][-1].legend(fontsize=6)
    fig.suptitle("Alignment of the leading target direction with source coordinate z1 (standardized mu)")
    fig.tight_layout()
    fig.savefig(folder / f"alignment_{stamp}.png", dpi=150)
    plt.close(fig)

    ks = np.arange(1, len(results[jobs[0]]["standardization"]["train_mean"]) + 1)
    fig, axes = plt.subplots(len(TARGETS), len(jobs), figsize=(3 * len(jobs), 2.2 * len(TARGETS)),
                             sharex=True, sharey=True, squeeze=False)
    for col, job in enumerate(jobs):
        for row, name in enumerate(TARGETS):
            ax = axes[row][col]
            access = results[job]["targets"][name]["access"]
            random = np.array([c["normalized"] for c in access["random"]])
            ax.fill_between(ks, random.min(0), random.max(0), color="0.85", label="random rotations")
            for i, curve in enumerate(access["learned"]):
                ax.plot(ks, curve["normalized"], color="tab:orange", linewidth=.8, alpha=.7,
                        label="learned Q_T" if i == 0 else None)
            ax.plot(ks, access["scatter"]["normalized"], color="tab:blue", label="scatter basis")
            ax.plot(ks, access["original"]["normalized"], color="black", marker="o", markersize=2.5, label="original")
            ax.axvline(access["scatter"]["interpretable_up_to_k"] + .5, color="tab:blue", linewidth=.5, linestyle=":")
            if row == 0:
                ax.set_title(short(job), fontsize=9)
            if col == 0:
                ax.set_ylabel(name, fontsize=9)
            if row == len(TARGETS) - 1:
                ax.set_xlabel("prefix k")
    axes[0][-1].legend(fontsize=6)
    fig.suptitle("Linear prefix probes, chance-normalized balanced accuracy (validation); "
                 "dotted line: scatter rank")
    fig.tight_layout()
    fig.savefig(folder / f"accessibility_{stamp}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(jobs), figsize=(3.4 * len(jobs), 3.2), sharey=True, squeeze=False)
    for ax, job in zip(axes[0], jobs):
        for x, name in enumerate(TARGETS):
            t = results[job]["targets"][name]
            to_scatter = [r["max_angle_to_scatter_deg"][0] for r in t["learned_rotation"]]
            between = [v[0] for v in t["restart_agreement_deg"].values()]
            ax.plot([x - .12] * len(to_scatter), to_scatter, "o", color="tab:blue", markersize=3,
                    label="Q_T vs scatter" if x == 0 else None)
            ax.plot([x + .12] * len(between), between, "o", color="tab:orange", markersize=3,
                    label="Q_T restart pairs" if x == 0 else None)
        ax.set_xticks(range(len(TARGETS)), TARGETS, rotation=45, fontsize=7)
        ax.set_title(short(job), fontsize=9)
    axes[0][0].set_ylabel("angle between leading directions (deg)")
    axes[0][-1].legend(fontsize=6)
    fig.suptitle("Estimator agreement (leading direction, j = 1)")
    fig.tight_layout()
    fig.savefig(folder / f"agreement_{stamp}.png", dpi=150)
    plt.close(fig)


def summary_text(results):
    lines = ["Tier 1 frozen-encoder target alignment (validation; standardized mu). Numbers only, no verdicts.",
             "A1: leading scatter direction on z1. null: all10 mean [p95]. acc: chance-normalized balanced accuracy.",
             "k*: 90% of full-probe gain ('!' = full probe near chance)."]
    for job, r in results.items():
        lines.append(f"\n{job}   masks: " + "  ".join(f"{k}={sum(v)}" for k, v in r["activity_masks"].items()))
        lines.append(f"  {'target':8s} rank conc   A1   null-all10    depth  Q_T A1(mean)  "
                     f"acc k=1 orig/scat/Q    acc k=10   k* orig/scat/Q   full spread")
        for name in TARGETS:
            t = r["targets"][name]
            s, a = t["scatter"], t["access"]
            null = t["null"]["all10"]
            q_a1 = np.mean([x["geometry"]["A1"] for x in t["learned_rotation"]])
            q_k1 = np.mean([c["normalized"][0] for c in a["learned"]])
            q_k = int(np.median([c["k_star"] for c in a["learned"]]))
            flag = "" if a["original"]["k_star_informative"] else "!"
            lines.append(
                f"  {name:8s} {s['rank']:4d} {s['concentration']:.2f}  {s['A1']:.2f}  {null['mean'][0][0]:.2f} "
                f"[{null['p95'][0][0]:.2f}]   {s['depth_leading']:5.2f}   {q_a1:.2f}        "
                f"{a['original']['normalized'][0]:.2f}/{a['scatter']['normalized'][0]:.2f}/{q_k1:.2f}"
                f"       {a['original']['normalized'][-1]:.2f}      "
                f"{a['original']['k_star']}{flag}/{a['scatter']['k_star']}/{q_k}"
                f"            {t['full_probe_spread']:.3f}")
    lines += ["", "Hypothesis inputs (S_posX minus S_recon, same beta and seed; H1-H4 per target):"]
    for job, r in results.items():
        if r["objective"] != "posx":
            continue
        partner = next((j for j, x in results.items() if x["objective"] == "recon"
                        and x["beta"] == r["beta"] and x["seed"] == r["seed"]), None)
        if partner is None:
            continue
        p = results[partner]
        lines.append(f"  beta {r['beta']:g} seed {r['seed']}:")
        for name in TARGETS:
            ta, tb = r["targets"][name], p["targets"][name]
            lines.append(
                f"    {name:8s} dA1 {ta['scatter']['A1'] - tb['scatter']['A1']:+.2f}   "
                f"d acc k=1 {ta['access']['original']['normalized'][0] - tb['access']['original']['normalized'][0]:+.2f}   "
                f"d acc k=2 {ta['access']['original']['normalized'][1] - tb['access']['original']['normalized'][1]:+.2f}   "
                f"d acc k=10 {ta['access']['original']['normalized'][-1] - tb['access']['original']['normalized'][-1]:+.2f}")
    lines += ["", "H9 inputs: (1 - A1, mean Q_T gain over original at k=1, at k=2) per pair:"]
    for job, r in results.items():
        for name in TARGETS:
            t = r["targets"][name]
            gain = [np.mean([c["normalized"][k] for c in t["access"]["learned"]]) - t["access"]["original"]["normalized"][k]
                    for k in (0, 1)]
            lines.append(f"  {short(job):18s} {name:8s} {1 - t['scatter']['A1']:.2f}  {gain[0]:+.2f}  {gain[1]:+.2f}")
    return "\n".join(lines) + "\n"


def write_outputs(results, folder, stamp, settings):
    write_json(folder / "config.json", settings)
    write_json(folder / "tier1.json", results)
    figures(results, folder, stamp)
    (folder / "results_summary.txt").write_text(summary_text(results))
    (folder / "experiment_description.log").write_text(
        __doc__ + "\nConfig:\n" + json.dumps(settings, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------- main ---
def select_jobs(run_dir, jobs, betas):
    if jobs:
        return jobs
    found = sorted(p.parent.name for p in run_dir.glob("*/checkpoint_final.pt") if "_ep" not in p.parent.name)
    if betas is not None:
        found = [j for j in found if any(re.search(rf"_beta{b:g}_", j) for b in betas)]
    return found


def run_parallel(args, jobs):
    """One subprocess per job, round-robin over GPUs; each writes its own folder when done."""
    processes = []
    for i, job in enumerate(jobs):
        command = [sys.executable, "-u", __file__, "--run-dir", str(args.run_dir), "--jobs", job,
                   "--device", f"cuda:{args.gpus[i % len(args.gpus)]}", "--data-path", str(args.data_path),
                   "--no-combine"] + (["--smoke"] if args.smoke else [])
        processes.append((job, subprocess.Popen(command)))
    failed = [job for job, process in processes if process.wait() != 0]
    if failed:
        print(f"Failed jobs: {failed}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--betas", nargs="+", type=float, help="Restrict default job list to these betas")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", nargs="+", type=int, help="Run jobs in parallel subprocesses on these GPUs")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--combine-only", action="store_true", help="No recomputation: combine newest saved results")
    parser.add_argument("--no-combine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    jobs = select_jobs(args.run_dir, args.jobs, args.betas)
    if not jobs:
        parser.error("No finished jobs found")
    failed = []
    if args.gpus and not args.combine_only:
        failed = run_parallel(args, jobs)
        if args.smoke:                        # smoke folders are never combined
            raise SystemExit(1 if failed else 0)
        args.combine_only = True
    results = {}
    if args.combine_only:
        for job in jobs:
            if job in failed:
                continue
            result = latest_tier1(args.run_dir / job)
            if result is None:
                parser.error(f"No saved tier 1 result for {job}; run without --combine-only")
            results[job] = result
    else:
        context = load_context(args.data_path)
        for job in jobs:
            results[job] = analyze_job(args.run_dir / job, context, torch.device(args.device), args.smoke)
    if len(results) > 1 and not args.no_combine and not args.smoke:
        folder = args.run_dir / f"tier1_combined_{stamp}"
        folder.mkdir()
        write_outputs(results, folder, stamp, {"jobs": list(results), "failed": failed, **config_record()})
        save_runtime(str(folder), time.perf_counter() - started)
        print(f"Combined tier 1: {folder}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
