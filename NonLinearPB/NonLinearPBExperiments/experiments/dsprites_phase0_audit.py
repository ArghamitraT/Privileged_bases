"""Phase 0: what do the existing beta-VAEs already encode? (plan v3, section 4)

Conda environment: mrl_env_cuda12
Usage (from code/; launch real runs through detached tmux):
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh phase0 --smoke
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_phase0_A_tmux.sh phase0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_phase0_audit.py --smoke --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_phase0_audit.py --seeds 0 --tasks x y --no-prefix-mlp

For each frozen beta-VAE (beta=4, seeds 0-2) and each task, on mu(x) and the
image-level split: full 10-D linear and MLP probes; each coordinate alone;
first-k prefixes in native order; first-k prefixes in the greedy best order
(forward selection on validation), linear and MLP. k* comes from the greedy
validation curve, because native order in an unsupervised beta-VAE is
arbitrary. Ordered factors (x, y, scale, rotation) also get linear-regression
R^2 (rotation as cos/sin), because exact-class accuracy counts a one-step
miss as fully wrong. --smoke uses small split subsets; diagnostics only.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
import csv
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
from vae_reproduction.features import load_encoder, encode_means, check_saved_means
from vae_reproduction.probes import (fit_linear_probe, fit_mlp_probe, greedy_coordinate_order,
                                     fit_linear_regression, greedy_order_r2, k_star,
                                     LINEAR_CONFIG, MLP_CONFIG)
from vae_reproduction.runs import (DATA_PATH, OUTPUT_ROOT, make_run_dir, snapshot, environment,
                                   pretrained_checkpoints, load_images, prepare_split, task_data,
                                   dataset_record, write_json)
from vae_reproduction.tasks import TASKS, n_classes, regression_targets

# ==============================================================================
# CONFIG
# ==============================================================================
EXPERIMENT_NOTE = (
    "Phase 0 audit of the existing beta=4 beta-VAEs (exprmnt_2026_10_13__21_22_10). "
    "Frozen mu(x); image-level 80/10/10 split (seed 0); tasks x, y, shape, scale on all "
    "shapes and rotation on hearts with 39 classes (orientation 39 == 0). Linear probe: "
    "L-BFGS logistic regression to tolerance; MLP: 2x256 ReLU, best validation accuracy. "
    "k* = smallest k with validation accuracy >= full - epsilon on the greedy order. "
    "Ordered factors also get least-squares R^2 (rotation: cos/sin of angle), with its own greedy "
    "order and k*. Note: these encoders were trained without labels on the full archive, test images included."
)
MODEL_SEEDS = [0, 1, 2]
TASK_LIST = ["x", "y", "scale"]   # 2026-10-22: shape and rotation dropped for now
HEATMAP_TASKS = ["x", "y", "scale"]   # single-coordinate map columns
SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
EPSILONS = [0.01, 0.02, 0.05]
PRIMARY_EPSILON = 0.01
MLP_SEED = 0
PREFIX_MLP = True
SMOKE_LIMITS = {"train": 20000, "validation": 4000, "test": 4000}
SMOKE_MLP = {"max_epochs": 3}
DEVICE = "cuda:0"
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# ==============================================================================


def accuracy_pair(result):
    return {"validation": result["validation"]["accuracy"], "test": result["test"]["accuracy"]}


def audit_task(data, factor, classes, device, prefix_mlp, mlp_config):
    train, validation, test = data["train"], data["validation"], data["test"]
    d = train[0].shape[1]
    columns = lambda split, keep: (split[0][:, keep], split[1])
    probe = lambda keep, **kw: fit_linear_probe(columns(train, keep), columns(validation, keep),
                                                columns(test, keep), classes, device=device, **kw)
    mlp = lambda keep: fit_mlp_probe(columns(train, keep), columns(validation, keep),
                                     columns(test, keep), classes, device=device,
                                     seed=MLP_SEED, config=mlp_config)
    every = list(range(d))
    full_linear = probe(every, confusion=True)
    full_mlp = mlp(every)
    single = [probe([i]) for i in every]
    native = [probe(every[:k]) for k in range(1, d + 1)]
    order, greedy_validation = greedy_coordinate_order(train, validation, classes, device=device)
    greedy = [probe(order[:k]) for k in range(1, d + 1)]
    greedy_mlp = [mlp(order[:k]) for k in range(1, d + 1)] if prefix_mlp else None
    unconverged = sum(not r["converged"] for r in [full_linear, *single, *native, *greedy])
    curve = [r["validation"]["accuracy"] for r in greedy]
    result = {
        "classes": classes, "chance": 1.0 / classes,
        "sizes": {name: int(len(split[1])) for name, split in data.items()},
        "full_linear": full_linear, "full_mlp": full_mlp,
        "single_coordinate_linear": [accuracy_pair(r) for r in single],
        "native_prefix_linear": [accuracy_pair(r) for r in native],
        "greedy_order": order, "greedy_selection_validation": greedy_validation,
        "greedy_prefix_linear": [accuracy_pair(r) for r in greedy],
        "greedy_prefix_mlp": [accuracy_pair(r) for r in greedy_mlp] if greedy_mlp else None,
        "k_star_greedy": {str(e): k_star(curve, e) for e in EPSILONS},
        "k_star_native": {str(e): k_star([r["validation"]["accuracy"] for r in native], e) for e in EPSILONS},
        "linear_probes_not_converged": unconverged,
        "full_mlp_minus_linear_test": full_mlp["test"]["accuracy"] - full_linear["test"]["accuracy"],
    }
    targets = {name: regression_targets(split[1], factor) for name, split in data.items()}
    if targets["train"] is not None:
        regress = lambda keep: fit_linear_regression(
            *[(data[name][0][:, keep], targets[name]) for name in ("train", "validation", "test")])
        r2_order, r2_curve = greedy_order_r2((train[0], targets["train"]), (validation[0], targets["validation"]))
        result["r2"] = {
            "full": regress(every), "single_coordinate": [regress([i]) for i in every],
            "native_prefix": [regress(every[:k]) for k in range(1, d + 1)],
            "greedy_order": r2_order, "greedy_selection_validation": r2_curve,
            "greedy_prefix": [regress(r2_order[:k]) for k in range(1, d + 1)],
            "k_star_greedy": {str(e): k_star(r2_curve, e) for e in EPSILONS}}
    if greedy_mlp:
        result["k_star_greedy_mlp"] = {str(e): k_star([r["validation"]["accuracy"] for r in greedy_mlp], e)
                                       for e in EPSILONS}
    return result


def normalized(accuracy, chance):
    return (accuracy - chance) / (1.0 - chance)


def plot(results, run_dir, stamp):
    labels, tasks = list(results), list(next(iter(results.values())))
    figure, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.2), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        for colour, label in zip(WONG[1:], labels):
            r = results[label][task]
            k = np.arange(1, len(r["greedy_prefix_linear"]) + 1)
            axis.plot(k, [v["test"] for v in r["greedy_prefix_linear"]], color=colour, label=f"{label} greedy")
            axis.plot(k, [v["test"] for v in r["native_prefix_linear"]], color=colour, lw=0.8, ls=":")
            if r["greedy_prefix_mlp"]:
                axis.plot(k, [v["test"] for v in r["greedy_prefix_mlp"]], color=colour, ls="--", lw=1)
        axis.axhline(results[labels[0]][task]["chance"], color="0.6", lw=0.8)
        axis.set(title=task, xlabel="prefix k", ylabel="test accuracy", ylim=(0, 1.02), xticks=k)
    axes[0][0].legend(fontsize=6, title="solid greedy, dotted native,\ndashed greedy MLP", title_fontsize=6)
    figure.tight_layout()
    for ext in ("png", "pdf"):
        figure.savefig(run_dir / f"phase0_prefix_curves_{stamp}.{ext}", dpi=200)
    plt.close(figure)

    columns = [t for t in HEATMAP_TASKS if t in tasks]
    heatmap(results, labels, columns, run_dir, f"phase0_single_coordinate_map_{stamp}",
            lambda r, t: [normalized(v["test"], r["chance"]) for v in r["single_coordinate_linear"]],
            "(acc - chance) / (1 - chance), one coordinate")
    heatmap(results, labels, columns, run_dir, f"phase0_single_coordinate_r2_map_{stamp}",
            lambda r, t: ([v["test"] for v in r["r2"]["single_coordinate"]] if "r2" in r else
                          [normalized(v["test"], r["chance"]) for v in r["single_coordinate_linear"]]),
            "test R^2, one coordinate (shape: normalized accuracy)")


def heatmap(results, labels, columns, run_dir, name, values, colour_label):
    figure, axes = plt.subplots(1, len(labels), figsize=(2.2 + 1.0 * len(columns) * len(labels), 4), squeeze=False)
    for axis, label in zip(axes[0], labels):
        matrix = np.array([values(results[label][t], t) for t in columns]).T
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                      color="white" if value < 0.5 else "black")
        axis.set(title=label, xticks=range(len(columns)), xticklabels=columns,
                 yticks=range(matrix.shape[0]), yticklabels=[f"z{i + 1}" for i in range(matrix.shape[0])])
        axis.tick_params(axis="x", rotation=45, labelsize=7)
    figure.colorbar(image, ax=axes[0].tolist(), label=colour_label)
    for ext in ("png", "pdf"):
        figure.savefig(run_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def summarize(results, run_dir):
    rows = []
    for label, tasks in results.items():
        for task, r in tasks.items():
            rows.append({"model": label, "task": task, "classes": r["classes"], "chance": r["chance"],
                         "full_linear_test": r["full_linear"]["test"]["accuracy"],
                         "full_mlp_test": r["full_mlp"]["test"]["accuracy"],
                         "full_linear_r2_test": r["r2"]["full"]["test"] if "r2" in r else "",
                         "greedy_order": " ".join(f"z{i + 1}" for i in r["greedy_order"]),
                         "greedy_order_r2": " ".join(f"z{i + 1}" for i in r["r2"]["greedy_order"]) if "r2" in r else "",
                         "k_star_r2_eps0.01": r["r2"]["k_star_greedy"]["0.01"] if "r2" in r else "",
                         **{f"k_star_eps{e}": r["k_star_greedy"][str(e)] for e in EPSILONS},
                         "unconverged_linear_probes": r["linear_probes_not_converged"]})
    with open(run_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tasks = list(next(iter(results.values())))
    locked = {task: {"per_model": {label: results[label][task]["k_star_greedy"][str(PRIMARY_EPSILON)]
                                   for label in results},
                     "common_k": max(results[label][task]["k_star_greedy"][str(PRIMARY_EPSILON)]
                                     for label in results)}
              for task in tasks}
    xy = [t for t in ("x", "y") if t in locked]
    if len(xy) == 2:
        locked["x_y_common_k"] = max(locked["x"]["common_k"], locked["y"]["common_k"])
    write_json(run_dir / "k_star.json", {"epsilon": PRIMARY_EPSILON, "source": "greedy order, validation, linear",
                                        "status": "provisional until reviewed and locked in the plan",
                                        "tasks": locked})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--tasks", nargs="+", default=TASK_LIST, choices=sorted(TASKS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-prefix-mlp", action="store_true")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    start = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    run_dir = make_run_dir(args.output_root, args.smoke)
    stamp = run_dir.name.removeprefix("exprmnt_")
    print(f"Run folder: {run_dir}", flush=True)
    snapshot(run_dir)
    split, split_info = prepare_split(run_dir, SPLIT_SEED, SPLIT_FRACTIONS)
    checkpoints = pretrained_checkpoints("beta_vae", args.seeds)
    limits = SMOKE_LIMITS if args.smoke else None
    mlp_config = SMOKE_MLP if args.smoke else None
    config = {"experiment_note": EXPERIMENT_NOTE, "checkpoints": checkpoints, "tasks": args.tasks,
              "split": split_info, "epsilons": EPSILONS, "primary_epsilon": PRIMARY_EPSILON,
              "linear_probe": LINEAR_CONFIG, "mlp_probe": {**MLP_CONFIG, **(mlp_config or {})},
              "mlp_seed": MLP_SEED, "prefix_mlp": not args.no_prefix_mlp, "smoke": args.smoke,
              "smoke_limits": limits, "device": str(device), "dataset": dataset_record(args.data_path),
              "environment": environment()}
    write_json(run_dir / "config.json", config)
    images = load_images(args.data_path)
    results = {}
    for label, checkpoint in checkpoints.items():
        model, model_config = load_encoder(checkpoint, device)
        if model_config.get("beta") != 4.0:
            raise ValueError(f"{checkpoint} is not a beta=4 model")
        features = encode_means(model, images, device)
        folder = run_dir / label
        folder.mkdir()
        np.save(folder / "latent_means.npy", features)
        write_json(folder / "feature_check.json", {"checkpoint": checkpoint, **check_saved_means(checkpoint, features)})
        del model
        results[label] = {}
        for task in args.tasks:
            t0 = time.time()
            data = task_data(features, split, task, limits, SPLIT_SEED)
            results[label][task] = audit_task(data, TASKS[task][0], n_classes(TASKS[task][0]), device,
                                              not args.no_prefix_mlp, mlp_config)
            results[label][task]["seconds"] = time.time() - t0
            write_json(folder / f"{task}.json", results[label][task])
            r = results[label][task]
            print(f"{label} {task}: full linear {r['full_linear']['test']['accuracy']:.4f} "
                  f"MLP {r['full_mlp']['test']['accuracy']:.4f} chance {r['chance']:.4f} "
                  f"greedy order {r['greedy_order']} k*={r['k_star_greedy']} "
                  + (f"R2 full {r['r2']['full']['test']:.4f} R2 order {r['r2']['greedy_order']} "
                     f"R2 k*={r['r2']['k_star_greedy']} " if "r2" in r else "")
                  + f"({r['seconds']:.0f}s)", flush=True)
    summarize(results, run_dir)
    plot(results, run_dir, stamp)
    write_json(run_dir / "status.json", {"status": "complete", "seconds": time.time() - start})
    save_runtime(str(run_dir), time.time() - start)
    print(f"Done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
