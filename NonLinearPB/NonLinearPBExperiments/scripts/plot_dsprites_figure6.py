"""Figure 6 A/B-style factor responses, mean traversals, and posterior variances.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/scripts/plot_dsprites_figure6.py --output /tmp/figure6 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/scripts/plot_dsprites_figure6.py --help
    python NonLinearPB/NonLinearPBExperiments/scripts/plot_dsprites_figure6.py --vae-checkpoint /path/to/mrl_beta1.pt --beta-checkpoint /path/to/mrl_beta4.pt --output /path/to/new_figures

This is a visualization of existing checkpoints, with no model training. The
paper leaves variance aggregation unspecified: ordinary columns use ascending
full-dataset mean posterior variance. MRL retains native z1..z10 order, with
variance shown only as a diagnostic. Pixel plots show probabilities.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.data import DSprites, FACTOR_SIZES
from vae_reproduction.model import model_from_config

# CONFIG: fixed existing seed-0, 100-epoch pilot; no fitting or score selection.
RESULTS = Path("/home/argha/Mat_embedding/files/results")
DEFAULT_VAE = RESULTS / "exprmnt_2026_10_12__22_29_14/beta_1_seed_0/checkpoint_final.pt"
DEFAULT_BETA = RESULTS / "exprmnt_2026_10_12__22_22_53/beta_4_seed_0/checkpoint_final.pt"
TRAVERSAL_VALUES = np.linspace(3, -3, 5)
SHAPE_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def factor_responses(means):
    """Exact marginal means in validated shape/scale/rotation/X/Y order."""
    grid = np.asarray(means).reshape(*FACTOR_SIZES, means.shape[1])
    return {
        "position": grid.mean(axis=(0, 1, 2), dtype=np.float64),
        "scale": grid.mean(axis=(2, 3, 4), dtype=np.float64),
        "rotation": grid.mean(axis=(1, 3, 4), dtype=np.float64),
    }


def coordinate_order(config, variance, requested="auto"):
    mrl = config.get("reconstruction_mrl", False)
    if mrl and requested == "variance":
        raise ValueError("MRL Figure 6 must retain native coordinate order")
    native = mrl or requested == "native"
    return (np.arange(len(variance)) if native else np.argsort(variance, kind="stable"),
            "native" if native else "variance")


def load_analysis(path, dataset, device, batch_size, reference_index,
                  requested_order="auto", allow_smoke=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg.get("image_limit") is not None and not allow_smoke:
        raise ValueError("Figure 6 requires a full-archive checkpoint")
    model = model_from_config(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    cache_path = path.parent / "latent_means.npy"
    means = np.load(cache_path, mmap_mode="r") if cache_path.exists() else None
    if means is not None and means.shape != (len(dataset.images), cfg["latent_dim"]):
        raise ValueError("Cached means have incorrect dimensions")
    if means is None:
        means = np.empty((len(dataset.images), cfg["latent_dim"]), dtype=np.float32)
    variance_sum = np.zeros(cfg["latent_dim"], dtype=np.float64)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(dataset.images), batch_size):
            x = torch.as_tensor(dataset.images[start:start + batch_size], device=device).float()
            mean, logvar = model.encode(x)
            if not cache_path.exists():
                means[start:start + len(x)] = mean.cpu().numpy()
            variance_sum += logvar.exp().double().sum(dim=0).cpu().numpy()
            if start % (batch_size * 100) == 0:
                print(f"beta={cfg['beta']}: posterior variance {start}/{len(dataset.images)}; {time.monotonic()-started:.1f}s", flush=True)
        reference = torch.as_tensor(dataset.images[reference_index:reference_index + 1], device=device).float()
        mean = model.encode(reference)[0][0]
        codes = mean.repeat(cfg["latent_dim"] * len(TRAVERSAL_VALUES), 1)
        for coordinate in range(cfg["latent_dim"]):
            codes[coordinate * len(TRAVERSAL_VALUES):(coordinate + 1) * len(TRAVERSAL_VALUES), coordinate] = torch.as_tensor(TRAVERSAL_VALUES, device=device, dtype=mean.dtype)
        traversal = model.decoder(codes).sigmoid().cpu().numpy().reshape(cfg["latent_dim"], len(TRAVERSAL_VALUES), 64, 64)
    variance = variance_sum / len(dataset.images)
    if not np.all(np.isfinite(variance)):
        raise ValueError("Nonfinite posterior variance")
    order, ordering = coordinate_order(cfg, variance, requested_order)
    return {"responses": factor_responses(means), "variance": variance,
            "order": order, "coordinate_order": ordering, "traversal": traversal,
            "config": cfg, "epoch": state["epoch"], "checkpoint": str(path.resolve()),
            "cached_means": str(cache_path.resolve()) if cache_path.exists() else None,
            "reference_mean": mean.cpu().tolist()}


def plot_panel(fig, subgrid, analysis, label, scales, rotations, response_limit):
    order = analysis["order"]
    response = analysis["responses"]
    heatmap = None
    for column, coordinate in enumerate(order):
        ax = fig.add_subplot(subgrid[0, column])
        heatmap = ax.imshow(response["position"][:, :, coordinate].T, origin="lower", cmap="RdBu_r", vmin=-response_limit, vmax=response_limit)
        ax.set_title(f"$z_{{{coordinate + 1}}}$", fontsize=10)
        ax.set_xticks([0, 31], labels=["0", "1"])
        ax.set_yticks([0, 31], labels=["0", "1"])
        ax.tick_params(labelsize=6, length=2)
        if column == 0:
            ax.set_ylabel("Position\nY", fontsize=9)
        for row, factor, values in ((1, "scale", scales), (2, "rotation", rotations)):
            ax = fig.add_subplot(subgrid[row, column])
            for shape, color in enumerate(SHAPE_COLORS):
                ax.plot(values, response[factor][shape, :, coordinate], color=color, linewidth=1)
            ax.set_ylim(-response_limit, response_limit)
            ax.set_xlim(values[0], values[-1])
            ax.tick_params(labelsize=6, length=2)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel(factor.capitalize(), fontsize=9)
            else:
                ax.set_yticklabels([])
        for offset, value in enumerate(TRAVERSAL_VALUES):
            ax = fig.add_subplot(subgrid[3 + offset, column])
            ax.imshow(analysis["traversal"][coordinate, offset], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(f"z = {value:g}", fontsize=9)
        ax = fig.add_subplot(subgrid[8, column])
        variance = analysis["variance"][coordinate]
        ax.imshow([[variance]], cmap="Greys", norm=Normalize(0, 1), aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(.5, .5, f"{variance:.3f}", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="white" if variance > .5 else "black")
        if column == 0:
            ax.set_ylabel("Mean\nvariance", fontsize=8)
    return heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--beta-checkpoint", type=Path, default=DEFAULT_BETA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reference-index", type=int, default=None)
    parser.add_argument("--coordinate-order", choices=("auto", "native", "variance"), default="auto")
    parser.add_argument("--allow-smoke", action="store_true", help="Allow subset-trained checkpoints; label plots as smoke diagnostics")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "status.json").write_text(json.dumps({"status": "running"}))
    started = time.monotonic()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    state = torch.load(args.beta_checkpoint, map_location="cpu", weights_only=False)
    data_path = Path(state["config"]["data_path"])
    dataset = DSprites(data_path)
    reference_index = args.reference_index
    if reference_index is None:
        reference_index = int(np.random.default_rng(2026).choice(len(dataset.images), 4, replace=False)[0])
    if not 0 <= reference_index < len(dataset.images):
        parser.error("Reference index outside dataset")
    with np.load(data_path, allow_pickle=False) as archive:
        values = archive["latents_values"]
        scales = np.unique(values[:, 2])
        rotations = np.unique(values[:, 3])
        reference_factors = archive["latents_classes"][reference_index].tolist()
    analyses = [load_analysis(path, dataset, args.device, args.batch_size, reference_index,
                              args.coordinate_order, args.allow_smoke)
                for path in (args.beta_checkpoint, args.vae_checkpoint)]
    if analyses[1]["config"].get("objective", "beta") != "beta" or analyses[1]["config"]["beta"] != 1:
        raise ValueError("Panel B requires a fixed-beta1 VAE checkpoint")
    if analyses[0]["config"].get("objective", "beta") == "beta" and analyses[0]["config"]["beta"] != 4:
        raise ValueError("Panel A requires beta4 or capacity-controlled training")
    if analyses[0]["config"]["dataset_sha256"] != analyses[1]["config"]["dataset_sha256"]:
        raise ValueError("Models trained on different dataset archives")
    limit = max(float(np.max(np.abs(v))) for a in analyses for v in a["responses"].values())
    limit = max(limit, 1e-6)
    labels = []
    for panel, analysis in zip(("A", "B"), analyses):
        cfg = analysis["config"]
        kind = (f"Capacity VAE (γ={cfg.get('gamma', 1000):g}, Cmax={cfg.get('capacity_max', 25):g})"
                if cfg.get("objective", "beta") == "capacity"
                else ("β-VAE (β=4)" if panel == "A" else "VAE (β=1)"))
        options = ""
        if any(cfg.get(key, default) != default for key, default in (
            ("optimizer", "adagrad"), ("initialization", "default"), ("decoder_activation", "tanh"))):
            options = f"; {cfg.get('optimizer', 'adagrad')}, {cfg.get('initialization', 'default')}, {cfg.get('decoder_activation', 'tanh')}"
        if cfg.get("reconstruction_mrl", False):
            kind = "MRL " + kind
        smoke_label = "SMOKE — " if cfg.get("image_limit") is not None else ""
        labels.append(f"{panel}  {smoke_label}{kind}, seed {cfg['seed']}, {analysis['epoch']} epochs{options}")
    files = []
    for selected, name in (([0, 1], "AB"), ([0], "A_beta_vae"), ([1], "B_vae")):
        fig = plt.figure(figsize=(13 * len(selected), 14))
        outer = fig.add_gridspec(1, len(selected), left=.035, right=.97, top=.925, bottom=.09, wspace=.12)
        for panel, index in enumerate(selected):
            grid = outer[panel].subgridspec(9, 10, height_ratios=[1.5, 1, 1, 1, 1, 1, 1, 1, .35], hspace=.28, wspace=.19)
            plot_panel(fig, grid, analyses[index], labels[index], scales, rotations, limit)
            fig.text((panel + .5) / len(selected), .962, labels[index], ha="center", fontsize=16)
        ordering = "; ".join(f"{('A', 'B')[i]}: {analyses[i]['coordinate_order']} coordinate order" for i in selected)
        fig.text(.5, .035, "Square: red   Ellipse: green   Heart: blue   |   Traversals: +3 → −3; remaining coordinates fixed to reference μ(x)\n" + ordering + "; original coordinate IDs retained. Shared response limits across A/B.", ha="center", fontsize=10)
        color_ax = fig.add_axes([.978, .69, .006, .2])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"), cax=color_ax, label="Mean latent activation")
        for suffix in ("png", "pdf"):
            path = args.output / f"figure6_{name}_{stamp}.{suffix}"
            fig.savefig(path, dpi=180)
            files.append(str(path))
        plt.close(fig)
    plt.imsave(args.output / f"reference_{reference_index}_{stamp}.png", dataset.images[reference_index], cmap="gray", vmin=0, vmax=1)
    for name, analysis in zip(("beta4", "beta1"), analyses):
        if analysis["config"].get("reconstruction_mrl", False):
            name = "mrl_" + name
        np.savez(args.output / f"factor_responses_{name}.npz", **analysis["responses"], posterior_variance=analysis["variance"], coordinate_order=analysis["order"], traversal=analysis["traversal"], traversal_values=TRAVERSAL_VALUES)
    provenance = {"figure": "Higgins et al. ICLR2017 under-review Figure6 A/B-style analysis", "reference_index": reference_index, "reference_factor_classes_color_shape_scale_rotation_x_y": reference_factors, "traversal_values": TRAVERSAL_VALUES.tolist(), "variance_convention": "mean(exp(logvar(x))) across all 737280 images; MRL native order, ordinary variance order by default", "factor_aggregation": "exact full-factorial marginal mean of encoder means", "plot_limits": [-limit, limit], "device": args.device, "models": [{k: a[k] for k in ("checkpoint", "cached_means", "epoch", "config", "reference_mean", "coordinate_order")} | {"coordinate_order_zero_based": a["order"].tolist(), "mean_posterior_variance": a["variance"].tolist()} for a in analyses], "files": files, "elapsed_seconds": time.monotonic() - started}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output / "experiment_description.log").write_text("Post-hoc Figure6 A/B-style plots of supplied checkpoints. No training or metric refitting. MRL uses native order; ordinary models use variance order by default. Subset-trained checkpoints are smoke diagnostics. See provenance.json for conventions and model budgets.\n")
    (args.output / "runtime.txt").write_text(f"{provenance['elapsed_seconds']:.3f} seconds\n")
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), Path(__file__).parents[1] / "vae_reproduction/model.py", Path(__file__).parents[1] / "vae_reproduction/data.py"):
        shutil.copy2(source, snapshot / source.name)
    (args.output / "status.json").write_text(json.dumps({"status": "completed", "figures": files}, indent=2))
    print(f"Completed: {args.output}; runtime {provenance['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
