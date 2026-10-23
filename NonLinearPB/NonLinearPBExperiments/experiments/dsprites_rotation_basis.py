"""Experiment A: learned orthogonal task basis on a frozen encoder (plan v3, sections 5, 15).

Conda environment: mrl_env_cuda12
Usage (from code/; launch real runs through detached tmux):
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh rotation --smoke
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh rotation
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_rotation_basis.py --smoke --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_rotation_basis.py --methods original mrl pca oracle greedy_nested --angles
    # later, C0+Q: any frozen checkpoint(s)
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_rotation_basis.py --checkpoint c0_x_seed0=/path/checkpoint_final.pt --tasks x y

For every frozen encoder and task, build a 10x10 orthonormal basis Q by each
method, then evaluate ALL methods the same way on the first k transformed
coordinates, k = 1..10: a fresh linear classifier (test accuracy) and, for
ordered factors, a fresh linear regression (test R^2; rotation as cos/sin).
Default methods (2026-10-22): original order, greedy permutation
(validation), LDA (orthonormalized, PCA fill), joint MRL Q (4 inits).
Optional (future plan): pca, oracle (separate k-frame per k), greedy_nested;
--angles saves principal angles inside each encoder.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utility import save_runtime
from vae_reproduction.bases import (ROTATION_CONFIG, initial_basis, fit_nested_rotation, head_accuracies,
                                    fit_greedy_nested, pca_basis, lda_basis, permutation_basis,
                                    orthogonality_error, principal_angles)
from vae_reproduction.features import load_encoder, encode_means, check_saved_means
from vae_reproduction.probes import (fit_linear_probe, fit_linear_regression, greedy_coordinate_order,
                                     LINEAR_CONFIG)
from vae_reproduction.runs import (DATA_PATH, OUTPUT_ROOT, make_run_dir, snapshot, environment,
                                   pretrained_checkpoints, load_images, prepare_split, task_data,
                                   dataset_record, write_json)
from vae_reproduction.tasks import TASKS, n_classes, regression_targets

# ==============================================================================
# CONFIG
# ==============================================================================
EXPERIMENT_NOTE = (
    "Experiment A (linear bridge): frozen beta=4 beta-VAE mu(x) (ordinary training, no MRL in the "
    "encoder); orthogonal Q = Q0 expm(S-S^T) with independent linear heads on every prefix, mean "
    "prefix CE, Adam. MRL acts only on Q and heads. Compared with original order, validation greedy "
    "permutation, and orthonormalized LDA. Every method scored by a fresh L-BFGS linear classifier "
    "(accuracy) and a fresh least-squares regression (R^2) on each prefix. Same image-level split "
    "(seed 0) as Phase 0. Linear problem in fixed features, not evidence about nonlinear basis learning."
)
MODEL_SEEDS = [0]
TASK_LIST = ["x", "y", "scale"]   # 2026-10-22: shape and rotation dropped for now
METHODS = ["original", "permutation", "lda", "mrl"]
ALL_METHODS = ["original", "permutation", "lda", "mrl", "pca", "oracle", "greedy_nested"]
SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
MRL_INITS = ["identity", "random0", "random1", "random2"]
ORACLE_INITS = ["identity", "random0"]       # best validation loss per k
GREEDY_SEED = 0
TRAIN_SEED = 0
SMOKE_LIMITS = {"train": 20000, "validation": 4000, "test": 4000}
SMOKE_ROTATION = {"max_epochs": 3}
DEVICE = "cuda:0"
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
METHOD_STYLE = {"original": (WONG[0], ":"), "permutation": (WONG[7], "-"), "pca": (WONG[2], "-"),
                "lda": (WONG[3], "-"), "greedy_nested": (WONG[1], "-"), "oracle": (WONG[6], "--"),
                "mrl": (WONG[5], "-")}
# ==============================================================================


def evaluate_frames(frames, data, targets, classes, device):
    """frames[k-1] is a d x k matrix; fresh linear classifier and regression on x @ frame."""
    curve = []
    for frame in frames:
        project = lambda name: data[name][0] @ frame
        r = fit_linear_probe(*[(project(n), data[n][1]) for n in ("train", "validation", "test")],
                             classes, device=device)
        point = {"validation": r["validation"]["accuracy"], "test": r["test"]["accuracy"],
                 "converged": r["converged"]}
        if targets["train"] is not None:
            r2 = fit_linear_regression(*[(project(n), targets[n]) for n in ("train", "validation", "test")])
            point["r2_validation"], point["r2_test"] = r2["validation"], r2["test"]
        curve.append(point)
    return curve


def basis_frames(q):
    return [q[:, :k] for k in range(1, q.shape[1] + 1)]


def run_task(data, classes, methods, device, rotation_config):
    train, validation = data["train"], data["validation"]
    d = train[0].shape[1]
    prefixes = list(range(1, d + 1))
    bases, logs, head_scores, oracle_frames = {}, {}, {}, None
    if "original" in methods:
        bases["original"] = np.eye(d)
    if "permutation" in methods:
        order, _ = greedy_coordinate_order(train, validation, classes, device=device)
        bases["permutation"] = permutation_basis(order)
        logs["permutation"] = {"order": order}
    if "pca" in methods:
        bases["pca"], variances = pca_basis(train[0])
        logs["pca"] = {"variances": variances.tolist()}
    if "lda" in methods:
        bases["lda"], eigenvalues = lda_basis(train[0], train[1], classes)
        logs["lda"] = {"eigenvalues": eigenvalues.tolist(), "discriminant_directions": int(len(eigenvalues))}
    if "mrl" in methods:
        for init in MRL_INITS:
            q, heads, center, log = fit_nested_rotation(train, validation, classes, prefixes,
                                                        initial_basis(init, d), device=device,
                                                        seed=TRAIN_SEED, config=rotation_config)
            bases[f"mrl_{init}"] = q
            logs[f"mrl_{init}"] = log
            head_scores[f"mrl_{init}"] = head_accuracies(q, heads, prefixes, center, data["test"], device)
    if "greedy_nested" in methods:
        bases["greedy_nested"], logs["greedy_nested"] = fit_greedy_nested(
            train, validation, classes, device=device, seed=GREEDY_SEED, config=rotation_config)
    if "oracle" in methods:
        oracle_frames, oracle_logs = [], []
        for k in prefixes:
            best = None
            for init in ORACLE_INITS:
                q, _, _, log = fit_nested_rotation(train, validation, classes, [k], initial_basis(init, d),
                                                   device=device, seed=TRAIN_SEED, config=rotation_config)
                log.pop("history")
                if best is None or log["best_validation_loss"] < best[1]["best_validation_loss"]:
                    best = (q[:, :k], {"init": init, **log})
            oracle_frames.append(best[0])
            oracle_logs.append(best[1])
        logs["oracle"] = oracle_logs
    return bases, oracle_frames, logs, head_scores


def curve_of(curve, key="test"):
    return np.array([v[key] for v in curve])


def summary(curves):
    """Mean over k of each curve; MRL also as mean over its inits."""
    out = {}
    for key in ("test", "r2_test"):
        if key not in next(iter(curves.values()))[0]:
            continue
        values = {name: float(curve_of(c, key).mean()) for name, c in curves.items()}
        mrl = [curve_of(curves[f"mrl_{i}"], key) for i in MRL_INITS if f"mrl_{i}" in curves]
        if mrl:
            values["mrl_mean"] = float(np.mean(mrl))
        out[f"mean_over_k_{key}"] = values
    return out


def angles(task_bases):
    """Principal angles in degrees, per k, inside one encoder's feature space."""
    out = {"same_task_mrl_inits": {}, "cross_task_same_init": {}, "mrl_vs_closed_form": {}}
    degrees = lambda a, b, k: np.degrees(principal_angles(a[:, :k], b[:, :k])).tolist()
    d = next(iter(task_bases.values()))[f"mrl_{MRL_INITS[0]}"].shape[0]
    for task, bases in task_bases.items():
        for a, b in combinations(MRL_INITS, 2):
            out["same_task_mrl_inits"][f"{task}:{a}|{b}"] = {k: degrees(bases[f"mrl_{a}"], bases[f"mrl_{b}"], k)
                                                             for k in range(1, d)}
        for init in MRL_INITS:
            for other in ("lda", "pca", "permutation"):
                if other in bases:
                    out["mrl_vs_closed_form"][f"{task}:{init}|{other}"] = {
                        k: degrees(bases[f"mrl_{init}"], bases[other], k) for k in range(1, d)}
    for (t1, b1), (t2, b2) in combinations(task_bases.items(), 2):
        for init in MRL_INITS:
            out["cross_task_same_init"][f"{t1}|{t2}:{init}"] = {k: degrees(b1[f"mrl_{init}"], b2[f"mrl_{init}"], k)
                                                                for k in range(1, d)}
    return out


def plot(model_results, run_dir, stamp, label):
    tasks = list(model_results)
    rows = [("test", "test accuracy")]
    if any("r2_test" in model_results[t]["curves"]["original" if "original" in model_results[t]["curves"]
                                                    else next(iter(model_results[t]["curves"]))][0] for t in tasks):
        rows.append(("r2_test", "test R$^2$"))
    figure, axes = plt.subplots(len(rows), len(tasks), figsize=(3.4 * len(tasks), 3.0 * len(rows)), squeeze=False)
    for column, task in enumerate(tasks):
        curves, chance = model_results[task]["curves"], model_results[task]["chance"]
        for row, (key, ylabel) in enumerate(rows):
            axis = axes[row][column]
            if key not in next(iter(curves.values()))[0]:
                axis.set_visible(False)
                continue
            k = np.arange(1, 11)
            for name in ("original", "permutation", "pca", "lda", "greedy_nested", "oracle"):
                if name in curves:
                    colour, style = METHOD_STYLE[name]
                    axis.plot(k, curve_of(curves[name], key), color=colour, ls=style, lw=1.2, label=name)
            mrl = np.array([curve_of(curves[f"mrl_{i}"], key) for i in MRL_INITS if f"mrl_{i}" in curves])
            if len(mrl):
                colour, _ = METHOD_STYLE["mrl"]
                axis.plot(k, mrl.mean(0), color=colour, lw=2, label=f"MRL Q (mean of {len(mrl)})")
                axis.fill_between(k, mrl.min(0), mrl.max(0), color=colour, alpha=0.2, lw=0)
            if key == "test":
                axis.axhline(chance, color="0.6", lw=0.8)
            axis.set(title=task, xlabel="first k coordinates", ylabel=ylabel, ylim=(0, 1.02), xticks=k)
    axes[0][0].legend(fontsize=6)
    figure.suptitle(label, fontsize=9)
    figure.tight_layout()
    for ext in ("png", "pdf"):
        figure.savefig(run_dir / f"A_prefix_curves_{label}_{stamp}.{ext}", dpi=200)
    plt.close(figure)


def parse_checkpoints(values, seeds):
    if not values:
        return pretrained_checkpoints("beta_vae", seeds)
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("--checkpoint expects label=path")
    return dict(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--checkpoint", action="append", help="label=path; replaces the default beta-VAEs")
    parser.add_argument("--tasks", nargs="+", default=TASK_LIST, choices=sorted(TASKS))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=ALL_METHODS)
    parser.add_argument("--angles", action="store_true", help="save principal angles (needs mrl)")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.angles and "mrl" not in args.methods:
        parser.error("--angles needs the mrl method")
    start = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    run_dir = make_run_dir(args.output_root, args.smoke)
    stamp = run_dir.name.removeprefix("exprmnt_")
    print(f"Run folder: {run_dir}", flush=True)
    snapshot(run_dir)
    split, split_info = prepare_split(run_dir, SPLIT_SEED, SPLIT_FRACTIONS)
    checkpoints = parse_checkpoints(args.checkpoint, args.seeds)
    limits = SMOKE_LIMITS if args.smoke else None
    rotation_config = {**ROTATION_CONFIG, **(SMOKE_ROTATION if args.smoke else {})}
    write_json(run_dir / "config.json", {
        "experiment_note": EXPERIMENT_NOTE, "checkpoints": checkpoints, "tasks": args.tasks,
        "methods": args.methods, "angles": args.angles, "split": split_info, "mrl_inits": MRL_INITS,
        "oracle_inits": ORACLE_INITS, "greedy_seed": GREEDY_SEED, "train_seed": TRAIN_SEED,
        "rotation": rotation_config, "linear_probe": LINEAR_CONFIG, "smoke": args.smoke,
        "smoke_limits": limits, "device": str(device), "dataset": dataset_record(args.data_path),
        "environment": environment()})
    images = load_images(args.data_path)
    rows = []
    for label, checkpoint in checkpoints.items():
        model, _ = load_encoder(checkpoint, device)
        features = encode_means(model, images, device)
        del model
        folder = run_dir / label
        folder.mkdir()
        write_json(folder / "feature_check.json", {"checkpoint": checkpoint, **check_saved_means(checkpoint, features)})
        model_results, task_bases = {}, {}
        for task in args.tasks:
            t0 = time.time()
            factor = TASKS[task][0]
            classes = n_classes(factor)
            data = task_data(features, split, task, limits, SPLIT_SEED)
            targets = {name: regression_targets(split_data[1], factor) for name, split_data in data.items()}
            bases, oracle_frames, logs, head_scores = run_task(data, classes, args.methods, device, rotation_config)
            curves = {name: evaluate_frames(basis_frames(q), data, targets, classes, device)
                      for name, q in bases.items()}
            if oracle_frames is not None:
                curves["oracle"] = evaluate_frames(oracle_frames, data, targets, classes, device)
            task_bases[task] = bases
            np.savez(folder / f"{task}_bases.npz", **bases,
                     **{f"oracle_k{k}": f for k, f in enumerate(oracle_frames or [], 1)})
            model_results[task] = {"classes": classes, "chance": 1.0 / classes,
                                   "sizes": {n: int(len(s[1])) for n, s in data.items()},
                                   "curves": curves, "training_head_test_accuracy": head_scores,
                                   "orthogonality_error": {n: orthogonality_error(q) for n, q in bases.items()},
                                   "logs": logs, "summary": summary(curves), "seconds": time.time() - t0}
            write_json(folder / f"{task}.json", model_results[task])
            for name, curve in curves.items():
                for k, v in enumerate(curve, 1):
                    rows.append({"model": label, "task": task, "method": name, "k": k,
                                 "validation_accuracy": v["validation"], "test_accuracy": v["test"],
                                 "validation_r2": v.get("r2_validation", ""), "test_r2": v.get("r2_test", ""),
                                 "probe_converged": v["converged"]})
            for key, values in model_results[task]["summary"].items():
                print(f"{label} {task} {key}: " + " ".join(f"{n}={v:.3f}" for n, v in values.items()), flush=True)
            print(f"{label} {task} done ({time.time() - t0:.0f}s)", flush=True)
        if args.angles:
            write_json(folder / "principal_angles_degrees.json", angles(task_bases))
        plot(model_results, run_dir, stamp, label)
    with open(run_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run_dir / "status.json", {"status": "complete", "seconds": time.time() - start})
    save_runtime(str(run_dir), time.time() - start)
    print(f"Done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
