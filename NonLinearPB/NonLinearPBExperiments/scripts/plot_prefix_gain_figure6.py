"""Figure 6 factor responses plus the gain each coordinate adds to the prefix.

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/plot_prefix_gain_figure6.py --run /home/argha/Mat_embedding/files/results/exprmnt_2026_10_22__22_16_48 --analysis analysis_2026_10_22__22_37_06

No fitting. For every task and basis of an Experiment A run, one figure:
rows 1-3 are the Figure 6 mean responses of z_i to position, scale and
rotation (from <analysis>/<model>/<task>/responses_<basis>.npz); rows 4-5
are the gain of adding z_k to the prefix z_1..z_{k-1}:
    delta_acc(k) = acc(1:k) - acc(1:k-1),  delta_R2(k) = R2(1:k) - R2(1:k-1),
test set, fresh linear classifier / regression, with acc(1:0) = chance and
R2(1:0) = 0. Bars are labelled with the delta and, below, the running total.
Figures go to <analysis>/<model>/<task>/prefix_gain_figure6_<task>_<basis>_<stamp>.png/.pdf.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_dsprites_figure6 import SHAPE_COLORS

GAIN_COLORS = {"accuracy": "#0072B2", "r2": "#D55E00"}   # Wong blue / vermillion


def deltas(curve, start):
    values = np.asarray(curve, dtype=float)
    return np.diff(np.concatenate([[start], values])), values


def plot(responses, gains, scales, rotations, title, path):
    d = responses["position"].shape[-1]
    limit = max(float(np.max(np.abs(responses[key]))) for key in ("position", "scale", "rotation"))
    rows = [name for name in ("accuracy", "r2") if gains.get(name) is not None]
    figure = plt.figure(figsize=(13, 5.6 + 1.6 * len(rows)))
    grid = figure.add_gridspec(3 + len(rows), d, height_ratios=[1.5, 1, 1] + [1.3] * len(rows),
                               left=.06, right=.95, top=.91, bottom=.07, hspace=.45, wspace=.22)
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
    for offset, name in enumerate(rows):
        delta, total = gains[name]
        top = max(float(np.max(np.abs(delta))), 1e-3) * 1.35
        for i in range(d):
            axis = figure.add_subplot(grid[3 + offset, i])
            axis.bar([0], [delta[i]], width=.6, color=GAIN_COLORS[name])
            axis.axhline(0, color="0.5", lw=.6)
            axis.set_ylim(-top * .35, top)
            axis.set_xlim(-.6, .6)
            axis.set_xticks([])
            axis.tick_params(labelsize=6, length=2)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(0, max(delta[i], 0) + top * .04, f"{delta[i]:+.2f}", ha="center", va="bottom", fontsize=7)
            axis.set_xlabel(f"total {total[i]:.2f}", fontsize=6.5, labelpad=2)
            if i == 0:
                axis.set_ylabel("Δ accuracy" if name == "accuracy" else "Δ R$^2$", fontsize=9)
            else:
                axis.set_yticklabels([])
    figure.text(.5, .955, title, ha="center", fontsize=13)
    figure.text(.5, .02, "Square: red   Ellipse: green   Heart: blue   |   Native basis order   |   "
                "Δ = test score of z_1..z_k minus z_1..z_{k-1} (k=1: minus chance accuracy / minus 0 R²); "
                "'total' = score of z_1..z_k", ha="center", fontsize=8.5)
    colour_axis = figure.add_axes([.962, .66, .006, .22])
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"), cax=colour_axis,
                    label="Mean z_i")
    for ext in ("png", "pdf"):
        figure.savefig(f"{path}.{ext}", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Experiment A run folder")
    parser.add_argument("--analysis", required=True, help="analysis_<stamp> folder name inside --run")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    with np.load(config["dataset"]["data_path"], allow_pickle=False) as archive:
        values = archive["latents_values"]
    scales, rotations = np.unique(values[:, 2]), np.unique(values[:, 3])
    stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    written = []
    for label in config["checkpoints"]:
        for task in config["tasks"]:
            result = json.loads((args.run / label / f"{task}.json").read_text())
            folder = args.run / args.analysis / label / task
            for name, curve in result["curves"].items():
                source = folder / f"responses_{name}.npz"
                if not source.exists():
                    continue
                with np.load(source) as archive:
                    responses = {key: archive[key] for key in ("position", "scale", "rotation")}
                gains = {"accuracy": deltas([v["test"] for v in curve], result["chance"])}
                gains["r2"] = deltas([v["r2_test"] for v in curve], 0.0) if "r2_test" in curve[0] else None
                path = folder / f"prefix_gain_figure6_{task}_{name}_{stamp}"
                plot(responses, gains, scales, rotations, f"{label}, task {task}, basis {name}", path)
                written.append(f"{path}.png")
                print(f"{path}.png", flush=True)
    (args.run / args.analysis / f"prefix_gain_figure6_{stamp}.json").write_text(
        json.dumps({"figures": written, "definition": "delta(k) = score(1:k) - score(1:k-1), test set; "
                    "score(1:0) = chance accuracy or 0 R2"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
