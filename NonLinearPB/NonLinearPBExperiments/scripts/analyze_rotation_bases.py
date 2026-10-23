"""What do the rotated coordinates of Experiment A mean? (plan v3, sections 8, 15)

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/analyze_rotation_bases.py --run /home/argha/Mat_embedding/files/results/exprmnt_<stamp> --device cuda:0

No training. For every encoder, task, and saved basis Q in an Experiment A run
(original, permutation, lda, mrl_<init>, ...), the coordinates are
z = Q^T (mu(x) - c), with c the train mean used when Q was fitted. Outputs,
in <run>/analysis_<stamp>/<model>/<task>/:
  1. single-coordinate map: test R^2 of each z_i alone for x, y, scale,
     rotation (hearts, cos/sin) and normalized linear accuracy for shape;
  2. Figure 6 in native Q order: mean z_i over position, scale, rotation
     (exact factorial marginal means), traversals of one reference image,
     and the data standard deviation of z_i;
  3. traversals for four reference images: vary one z_i, map back with
     mu = c + Q z, decode with the frozen decoder.
Traversal values are +3 ... -3 around the train mean c (for the original
basis this is about the old Figure 6 convention, since c is near 0).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.data import FACTOR_SIZES
from vae_reproduction.features import load_encoder, encode_means
from vae_reproduction.probes import fit_linear_probe, fit_linear_regression
from vae_reproduction.runs import load_images, task_data, write_json
from vae_reproduction.tasks import TASKS, n_classes, regression_targets
from scripts.plot_dsprites_figure6 import factor_responses, SHAPE_COLORS

# ==============================================================================
# CONFIG
# ==============================================================================
MAP_TASKS = ["x", "y", "scale", "shape", "rotation_heart"]   # columns of the meaning map
TRAVERSAL_VALUES = np.linspace(3, -3, 7)
FIGURE6_TRAVERSAL = np.linspace(3, -3, 5)
REFERENCE_SEED = 2026          # same draw as plot_dsprites_figure6.py; first = 19477
N_REFERENCES = 4
SPLIT_SEED = 0
# ==============================================================================


def coordinates(features, center, q):
    return (features - center) @ q


def meaning_map(z_all, split, device):
    """Rows z_1..z_10, columns MAP_TASKS: R^2 (ordered factors) or normalized accuracy (shape)."""
    matrix = np.zeros((z_all.shape[1], len(MAP_TASKS)))
    for column, task in enumerate(MAP_TASKS):
        factor = TASKS[task][0]
        data = task_data(z_all, split, task, None, SPLIT_SEED)
        targets = {n: regression_targets(s[1], factor) for n, s in data.items()}
        for i in range(z_all.shape[1]):
            one = lambda n: data[n][0][:, [i]]
            if targets["train"] is not None:
                matrix[i, column] = fit_linear_regression(*[(one(n), targets[n]) for n in data])["test"]
            else:
                chance = 1.0 / n_classes(factor)
                accuracy = fit_linear_probe(*[(one(n), data[n][1]) for n in data], n_classes(factor),
                                            device=device)["test"]["accuracy"]
                matrix[i, column] = (accuracy - chance) / (1 - chance)
    return matrix


@torch.no_grad()
def traversals(model, z_reference, center, q, values, device):
    """images[i, v] = decode(c + Q z) with z = z_reference except z_i = values[v]."""
    d = len(z_reference)
    codes = np.repeat(z_reference[None], d * len(values), axis=0)
    for i in range(d):
        codes[i * len(values):(i + 1) * len(values), i] = values
    means = center + codes @ q.T
    logits = model.decoder(torch.as_tensor(means, dtype=torch.float32, device=device))
    return logits.sigmoid().cpu().numpy().reshape(d, len(values), 64, 64)


def plot_figure6(responses, std, traversal, scales, rotations, title, path):
    d = traversal.shape[0]
    limit = max(float(np.max(np.abs(v))) for v in responses.values())
    figure = plt.figure(figsize=(13, 13))
    grid = figure.add_gridspec(9, d, height_ratios=[1.5, 1, 1, 1, 1, 1, 1, 1, .35],
                               left=.05, right=.95, top=.93, bottom=.08, hspace=.28, wspace=.19)
    for i in range(d):
        axis = figure.add_subplot(grid[0, i])
        axis.imshow(responses["position"][:, :, i].T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(f"$z_{{{i + 1}}}$", fontsize=10)
        axis.set_xticks([0, 31], labels=["0", "1"])
        axis.set_yticks([0, 31], labels=["0", "1"])
        axis.tick_params(labelsize=6, length=2)
        if i == 0:
            axis.set_ylabel("Position\nY", fontsize=9)
        for row, factor, values in ((1, "scale", scales), (2, "rotation", rotations)):
            axis = figure.add_subplot(grid[row, i])
            for shape, colour in enumerate(SHAPE_COLORS):
                axis.plot(values, responses[factor][shape, :, i], color=colour, lw=1)
            axis.set_ylim(-limit, limit)
            axis.set_xlim(values[0], values[-1])
            axis.tick_params(labelsize=6, length=2)
            axis.spines[["top", "right"]].set_visible(False)
            if i == 0:
                axis.set_ylabel(factor.capitalize(), fontsize=9)
            else:
                axis.set_yticklabels([])
        for offset, value in enumerate(FIGURE6_TRAVERSAL):
            axis = figure.add_subplot(grid[3 + offset, i])
            axis.imshow(traversal[i, offset], cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if i == 0:
                axis.set_ylabel(f"z = {value:g}", fontsize=9)
        axis = figure.add_subplot(grid[8, i])
        axis.imshow([[min(std[i], 1)]], cmap="Greys", norm=Normalize(0, 1), aspect="auto")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(.5, .5, f"{std[i]:.2f}", ha="center", va="center", transform=axis.transAxes, fontsize=7,
                  color="white" if std[i] > .5 else "black")
        if i == 0:
            axis.set_ylabel("Data\nstd", fontsize=8)
    figure.text(.5, .965, title, ha="center", fontsize=14)
    figure.text(.5, .03, "Square: red   Ellipse: green   Heart: blue   |   Native Q order z = Q^T(mu - c)   |   "
                "Traversal: z_i from +3 to -3, others fixed at the reference, decoded from mu = c + Q z",
                ha="center", fontsize=9)
    colour_axis = figure.add_axes([.965, .70, .006, .2])
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"), cax=colour_axis,
                    label="Mean z_i")
    for ext in ("png", "pdf"):
        figure.savefig(f"{path}.{ext}", dpi=160)
    plt.close(figure)


def plot_traversals(grids, references, title, path):
    d, n_values = grids[0].shape[:2]
    figure, axes = plt.subplots(d, len(grids) * (n_values + 1) - 1,
                                figsize=(0.75 * len(grids) * (n_values + 1), 0.8 * d + 1), squeeze=False)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
    for panel, (grid, reference) in enumerate(zip(grids, references)):
        for i in range(d):
            for v in range(n_values):
                axis = axes[i, panel * (n_values + 1) + v]
                axis.imshow(grid[i, v], cmap="gray", vmin=0, vmax=1)
                if i == 0:
                    axis.set_title(f"{TRAVERSAL_VALUES[v]:+.0f}" + (f"\nref {reference}" if v == 0 else ""), fontsize=7)
            axes[i, panel * (n_values + 1)].set_ylabel(f"$z_{{{i + 1}}}$", fontsize=9, rotation=0, labelpad=12)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    for ext in ("png", "pdf"):
        figure.savefig(f"{path}.{ext}", dpi=140)
    plt.close(figure)


def plot_maps(maps, title, path):
    names = list(maps)
    figure, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names) + 1, 4.2), squeeze=False)
    for axis, name in zip(axes[0], names):
        matrix = maps[name]
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                      color="white" if value < .5 else "black")
        axis.set(title=name, xticks=range(len(MAP_TASKS)), yticks=range(matrix.shape[0]),
                 yticklabels=[f"z{i + 1}" for i in range(matrix.shape[0])])
        axis.set_xticklabels(["x", "y", "scale", "shape", "rot."], rotation=45, fontsize=7)
    figure.colorbar(image, ax=axes[0].tolist(), label="one coordinate: test R$^2$ (shape: normalized accuracy)")
    figure.suptitle(title, fontsize=10)
    for ext in ("png", "pdf"):
        figure.savefig(f"{path}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Experiment A run folder")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    start = time.time()
    config = json.loads((args.run / "config.json").read_text())
    if config.get("smoke_limits"):
        print("Note: smoke run; Q bases are diagnostics only", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    out_root = args.run / f"analysis_{stamp}"
    out_root.mkdir()
    shutil.copy2(__file__, out_root / Path(__file__).name)
    with np.load(args.run / "split_manifest.npz") as archive:
        split = {name: archive[name] for name in ("train", "validation", "test")}
    data_path = Path(config["dataset"]["data_path"])
    images = load_images(data_path)
    with np.load(data_path, allow_pickle=False) as archive:
        values = archive["latents_values"]
    scales, rotations = np.unique(values[:, 2]), np.unique(values[:, 3])
    references = [int(i) for i in np.random.default_rng(REFERENCE_SEED).choice(len(images), N_REFERENCES, replace=False)]
    record = {"run": str(args.run), "references": references, "traversal_values": TRAVERSAL_VALUES.tolist(),
              "figure6_traversal_values": FIGURE6_TRAVERSAL.tolist(), "map_tasks": MAP_TASKS, "models": {}}
    for label, checkpoint in config["checkpoints"].items():
        model, _ = load_encoder(checkpoint, device)
        features = encode_means(model, images, device)
        record["models"][label] = {"checkpoint": checkpoint, "tasks": {}}
        for task in config["tasks"]:
            folder = out_root / label / task
            folder.mkdir(parents=True)
            center = task_data(features, split, task, config.get("smoke_limits"), SPLIT_SEED)["train"][0].mean(0)
            with np.load(args.run / label / f"{task}_bases.npz") as archive:
                bases = {name: archive[name] for name in archive.files if not name.startswith("oracle_k")}
            maps, saved = {}, {}
            for name, q in bases.items():
                t0 = time.time()
                z_all = coordinates(features, center, q)
                maps[name] = meaning_map(z_all, split, device)
                responses = factor_responses(z_all)
                std = z_all.std(0)
                grids = [traversals(model, z_all[r], center, q, TRAVERSAL_VALUES, device) for r in references]
                figure6 = traversals(model, z_all[references[0]], center, q, FIGURE6_TRAVERSAL, device)
                title = f"{label}, task {task}, basis {name}"
                plot_figure6(responses, std, figure6, scales, rotations, title,
                             folder / f"figure6_{task}_{name}_{stamp}")
                plot_traversals(grids, references, title, folder / f"traversals_{task}_{name}_{stamp}")
                np.savez(folder / f"responses_{name}.npz", **responses, data_std=std,
                         meaning_map=maps[name], figure6_traversal=figure6)
                saved[name] = {"meaning_map": maps[name].tolist(), "data_std": std.tolist()}
                print(f"{label} {task} {name}: done ({time.time() - t0:.0f}s)", flush=True)
            plot_maps(maps, f"{label}, task {task}: what each coordinate holds",
                      folder / f"meaning_map_{task}_{stamp}")
            write_json(folder / "meaning_maps.json", {"columns": MAP_TASKS, "bases": saved})
            record["models"][label]["tasks"][task] = sorted(bases)
        del model
    record["seconds"] = time.time() - start
    write_json(out_root / "provenance.json", record)
    print(f"Done: {out_root} ({record['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
