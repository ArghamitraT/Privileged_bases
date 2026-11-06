"""Analysis of finished dsprites_pretrain.py runs: activity, factor maps, prefix probes, Figure 6.

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/analyze_pretrain.py --run-dir /path/to/exprmnt_x --device cuda:1
    python NonLinearPB/NonLinearPBExperiments/scripts/analyze_pretrain.py --run-dir /path/to/exprmnt_x --jobs recon_beta0_seed0 recon_beta0_seed1
    python NonLinearPB/NonLinearPBExperiments/scripts/analyze_pretrain.py --run-dir /path/to/exprmnt_x --combine-only

Every training job calls analyze_job() itself as soon as it finishes, so each
job folder gets <job>/analysis_<stamp>/. This script is the standalone entry
point: it re-analyzes the given (default: all finished) jobs into new per-job
folders, then writes combined cross-run figures to <run-dir>/analysis_combined_<stamp>/.
--combine-only skips recomputation and combines each job's newest saved analysis.

No training. On the posterior mean mu(x): validation reconstruction per prefix;
three activity rules (variance ratio, per-coordinate KL for beta > 0,
reconstruction gain per added prefix coordinate); factor-by-coordinate
correlation ratio eta^2; native-order prefix linear probes (accuracy, and R^2
for ordered factors) for shape, scale, rotation (hearts only), X and Y; and a
single-model Figure 6 (factor responses of each standardized coordinate, and
traversals between each coordinate's 1st and 99th validation percentile, since
beta = 0 has no fixed latent scale). --smoke subsamples the probes.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE_ROOT))
from utility import save_runtime
from vae_reproduction.data import DSprites, FACTOR_SIZES
from vae_reproduction.model import model_from_config, posterior_kl, prefix_masks
from vae_reproduction.probes import fit_linear_probe, fit_linear_regression
from vae_reproduction.runs import DATA_PATH, write_json
from vae_reproduction.tasks import image_factors, n_classes, regression_targets, subsample, task_split
from scripts.plot_dsprites_figure6 import factor_responses, SHAPE_COLORS

# ==============================================================================
# CONFIG
# ==============================================================================
FACTORS = ("shape", "scale", "rotation", "x", "y")          # image_factors column order
PROBE_TASKS = {"shape": "shape", "scale": "scale", "rotation": "rotation_heart", "x": "x", "y": "y"}
PROBE_TRAIN = 50000          # probe training images, drawn from the encoder's train split
VARIANCE_THRESHOLD = 0.01    # v_i / max_j v_j
KL_THRESHOLD = 0.01          # nats, validation mean, beta > 0 only
RECON_GAIN_THRESHOLD = 0.5   # nats drop in validation recon when coordinate k is added
TRAVERSAL_STEPS = 7
REFERENCE_SEED = 2026
ENCODE_BATCH = 8192
# ==============================================================================


@torch.no_grad()
def encode(model, images, device):
    means = np.empty((len(images), model.mean.out_features), dtype=np.float32)
    logvars = np.empty_like(means)
    for start in range(0, len(images), ENCODE_BATCH):
        x = torch.from_numpy(images[start:start + ENCODE_BATCH].reshape(-1, 4096)).to(device, torch.float32)
        mean, logvar = model.encode(x)
        means[start:start + len(x)] = mean.cpu().numpy()
        logvars[start:start + len(x)] = logvar.cpu().numpy()
    return means, logvars


@torch.no_grad()
def prefix_reconstruction(model, images, indices, means, device):
    d = means.shape[1]
    masks = prefix_masks(d, device)
    total = torch.zeros(d, device=device, dtype=torch.float64)
    for start in range(0, len(indices), 2048):
        batch = indices[start:start + 2048]
        x = torch.from_numpy(images[batch].reshape(-1, 4096)).to(device, torch.float32)
        mu = torch.from_numpy(means[batch]).to(device)
        logits = model.decoder((mu[None] * masks[:, None]).reshape(-1, d))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, x.repeat(d, 1), reduction="none").sum(1).reshape(d, -1)
        total += bce.sum(1).double()
    return (total / len(indices)).cpu().numpy()


def correlation_ratio(values, labels):
    """eta^2 per column: between-class variance of values / total variance."""
    values = values.astype(np.float64)
    centered = values - values.mean(0)
    counts = np.bincount(labels)
    sums = np.stack([np.bincount(labels, weights=centered[:, i], minlength=len(counts))
                     for i in range(values.shape[1])], axis=1)
    between = (sums ** 2 / np.maximum(counts, 1)[:, None]).sum(0)
    return between / np.maximum((centered ** 2).sum(0), 1e-30)


def activity(means_val, logvars_val, recon, beta):
    variance = means_val.astype(np.float64).var(0)
    relative = variance / variance.max()
    gain = np.concatenate([[np.nan], recon[:-1] - recon[1:]])     # drop when coordinate k is added
    out = {"variance": variance.tolist(), "relative_variance": relative.tolist(),
           "variance_mask": (relative > VARIANCE_THRESHOLD).astype(int).tolist(),
           "recon_gain": gain.tolist(),
           "recon_gain_mask": [1] + (gain[1:] > RECON_GAIN_THRESHOLD).astype(int).tolist()}
    if beta > 0:
        kl = posterior_kl(torch.from_numpy(means_val), torch.from_numpy(logvars_val)).double().numpy()
        out.update(kl=kl.tolist(), kl_mask=(kl > KL_THRESHOLD).astype(int).tolist())
    for key in ("variance_mask", "recon_gain_mask", "kl_mask"):
        if key in out:
            out[key.replace("_mask", "_d_active")] = int(sum(out[key]))
    return out


def prefix_probes(means, split, device, smoke):
    """Native-order prefix linear probes; validation accuracy, k = 1..d."""
    results = {}
    limits = {"train": 4000 if smoke else PROBE_TRAIN, "validation": 4000 if smoke else None}
    for factor, task in PROBE_TASKS.items():
        data = {}
        for offset, name in enumerate(("train", "validation")):
            indices, _ = task_split(task, split[name])
            indices = subsample(indices, limits[name], offset)
            indices, labels = task_split(task, indices)
            data[name] = (means[indices], labels)
        classes = n_classes(factor)
        curve = []
        for k in range(1, means.shape[1] + 1):
            result = fit_linear_probe((data["train"][0][:, :k], data["train"][1]),
                                      (data["validation"][0][:, :k], data["validation"][1]),
                                      None, classes, device=device)
            curve.append(result["validation"]["accuracy"])
        targets = {name: regression_targets(labels, factor) for name, (_, labels) in data.items()}
        r2 = None if targets["train"] is None else [
            fit_linear_regression((data["train"][0][:, :k], targets["train"]),
                                  (data["validation"][0][:, :k], targets["validation"]), None)["validation"]
            for k in range(1, means.shape[1] + 1)]
        chance = 1 / classes
        normalized = [(a - chance) / (1 - chance) for a in curve]
        target = 0.95 * normalized[-1]
        results[factor] = {"accuracy": curve, "chance": chance, "normalized": normalized,
                           "k_star_chance_adjusted": next(k for k, v in enumerate(normalized, 1) if v >= target),
                           "r2": r2, "task": task}
    return results


@torch.no_grad()
def traversals(model, means_val, reference, device):
    low, high = np.quantile(means_val, [0.01, 0.99], axis=0)
    d = means_val.shape[1]
    codes = torch.as_tensor(reference, device=device).repeat(d * TRAVERSAL_STEPS, 1)
    for i in range(d):
        codes[i * TRAVERSAL_STEPS:(i + 1) * TRAVERSAL_STEPS, i] = torch.linspace(
            float(low[i]), float(high[i]), TRAVERSAL_STEPS, device=device)
    images = model.decoder(codes).sigmoid().cpu().numpy().reshape(d, TRAVERSAL_STEPS, 64, 64)
    return images, low, high


def figure6(job, means, means_val, traversal, low, high, posterior_variance, folder, stamp, scales, rotations):
    """Higgins Figure 6 layout for one model, on per-coordinate standardized mu."""
    d = means.shape[1]
    standardized = (means - means_val.mean(0)) / np.maximum(means_val.std(0), 1e-12)
    responses = factor_responses(standardized)
    limit = max(float(np.max(np.abs(v))) for v in responses.values())
    fig = plt.figure(figsize=(14, 15))
    grid = fig.add_gridspec(4 + TRAVERSAL_STEPS, d, height_ratios=[1.5, 1, 1] + [1] * TRAVERSAL_STEPS + [.35],
                            left=.06, right=.95, top=.93, bottom=.07, hspace=.3, wspace=.2)
    for i in range(d):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(responses["position"][:, :, i].T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"$z_{{{i + 1}}}$", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Position\n(X right, Y up)", fontsize=8)
        for row, factor, values in ((1, "scale", scales), (2, "rotation", rotations)):
            ax = fig.add_subplot(grid[row, i])
            for shape, color in enumerate(SHAPE_COLORS):
                ax.plot(values, responses[factor][shape, :, i], color=color, linewidth=1)
            ax.set_ylim(-limit, limit)
            ax.tick_params(labelsize=5, length=2)
            if i == 0:
                ax.set_ylabel(factor.capitalize(), fontsize=8)
            else:
                ax.set_yticklabels([])
        for step in range(TRAVERSAL_STEPS):
            ax = fig.add_subplot(grid[3 + step, i])
            ax.imshow(traversal[i, step], cmap="gray_r", vmin=0, vmax=1)   # black shape on white
            ax.set_xticks([])
            ax.set_yticks([])
            if step == 0:
                ax.set_xlabel(f"{low[i]:.2g}", fontsize=6, labelpad=1)
                ax.xaxis.set_label_position("top")
            if step == TRAVERSAL_STEPS - 1:
                ax.set_xlabel(f"{high[i]:.2g}", fontsize=6, labelpad=1)
            if i == 0 and step == TRAVERSAL_STEPS // 2:
                ax.set_ylabel("Traversal\n1% → 99%", fontsize=8)
        ax = fig.add_subplot(grid[3 + TRAVERSAL_STEPS, i])
        ax.axis("off")
        text = "n/a" if posterior_variance is None else f"{posterior_variance[i]:.3f}"
        ax.text(.5, .5, text, ha="center", va="center", fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.text(-.6, .5, "posterior\nvariance", ha="center", va="center", fontsize=7, transform=ax.transAxes)
    fig.suptitle(f"{job}: factor responses (standardized mu) and traversals", fontsize=14)
    fig.text(.5, .025, "Square red, ellipse green, heart blue. Responses: mean of standardized mu_i over the other factors "
             "(full archive). Traversals: coordinate i from its 1st to 99th validation percentile, others at the reference image's mu.",
             ha="center", fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"),
                 cax=fig.add_axes([.96, .8, .008, .12]), label="std. units")
    path = folder / f"figure6_{job}_{stamp}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary_figures(results, folder, stamp):
    jobs = list(results)
    d = len(next(iter(results.values()))["prefix_reconstruction_nll"])
    ks = np.arange(1, d + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for job in jobs:
        ax.plot(ks, results[job]["prefix_reconstruction_nll"], marker="o", markersize=3, label=job)
    ax.set_xlabel("prefix k")
    ax.set_ylabel("validation reconstruction (nats)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Reconstruction from the first k coordinates of mu")
    fig.tight_layout()
    fig.savefig(folder / f"prefix_reconstruction_{stamp}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(jobs), figsize=(2.6 * len(jobs), 4.2), squeeze=False)
    for ax, job in zip(axes[0], jobs):
        eta = np.array(results[job]["eta_squared"])          # d x factors
        ax.imshow(eta, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(FACTORS)), FACTORS, rotation=45, fontsize=7)
        ax.set_yticks(range(d), [f"z{i + 1}" for i in range(d)], fontsize=7)
        for i in range(d):
            for j in range(len(FACTORS)):
                ax.text(j, i, f"{eta[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if eta[i, j] < .5 else "black")
        ax.set_title(job, fontsize=7)
    fig.suptitle("Factor-by-coordinate eta^2 (share of mu_i variance explained by the factor), validation")
    fig.tight_layout()
    fig.savefig(folder / f"factor_coordinate_eta2_{stamp}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(FACTORS), figsize=(18, 3.8), sharey=True)
    for ax, factor in zip(axes, FACTORS):
        for job in jobs:
            ax.plot(ks, results[job]["prefix_probes"][factor]["normalized"], marker="o", markersize=3, label=job)
        ax.set_title(f"{factor} ({PROBE_TASKS[factor]})")
        ax.set_xlabel("prefix k")
        ax.set_ylim(-.05, 1.05)
    axes[0].set_ylabel("chance-normalized accuracy")
    axes[-1].legend(fontsize=6)
    fig.suptitle("Native-order prefix linear probes on mu (validation)")
    fig.tight_layout()
    fig.savefig(folder / f"prefix_probes_{stamp}.png", dpi=150)
    plt.close(fig)


def summary_text(results):
    lines = ["Post-hoc analysis of pretraining runs (validation split, posterior mean mu)."]
    for job, r in results.items():
        a = r["activity"]
        lines += [f"\n{job}",
                  "  recon per prefix (nats): " + " ".join(f"{v:.1f}" for v in r["prefix_reconstruction_nll"]),
                  f"  d_active: variance={a['variance_d_active']} recon_gain={a['recon_gain_d_active']}"
                  + (f" KL={a['kl_d_active']}" if "kl_d_active" in a else " KL=n/a"),
                  "  recon gain when adding z_k: " + " ".join("-" if np.isnan(v) else f"{v:.1f}" for v in a["recon_gain"])]
        if "kl" in a:
            lines.append("  KL per coordinate: " + " ".join(f"{v:.2f}" for v in a["kl"]))
        eta = np.array(r["eta_squared"])
        best = [f"z{i + 1}:{FACTORS[int(np.argmax(eta[i]))]}({eta[i].max():.2f})" for i in range(len(eta))]
        lines.append("  top factor per coordinate (eta^2): " + " ".join(best))
        for factor in FACTORS:
            p = r["prefix_probes"][factor]
            lines.append(f"  probe {factor:8s} acc k=1..10: " + " ".join(f"{v:.2f}" for v in p["accuracy"])
                         + f"  (chance {p['chance']:.3f}, k*={p['k_star_chance_adjusted']})")
            if p["r2"] is not None:
                lines.append(f"  R2    {factor:8s}     k=1..10: " + " ".join(f"{v:.2f}" for v in p["r2"]))
    return "\n".join(lines) + "\n"


def load_context(data_path=DATA_PATH):
    """Archive images and factor tables shared by every job's analysis."""
    images = DSprites(data_path).images
    with np.load(data_path, allow_pickle=False) as archive:
        values = archive["latents_values"]
    factors_all = image_factors(np.arange(len(images)))
    factors_all[:, 2] %= n_classes("rotation")
    return {"images": images, "scales": np.unique(values[:, 2]),
            "rotations": np.unique(values[:, 3]), "factors": factors_all}


def write_outputs(results, folder, stamp, settings):
    write_json(folder / "config.json", settings)
    write_json(folder / "analysis.json", results)
    summary_figures(results, folder, stamp)
    (folder / "results_summary.txt").write_text(summary_text(results))
    (folder / "experiment_description.log").write_text(
        __doc__ + "\nConfig:\n" + json.dumps(settings, indent=2) + "\n")


def analyze_job(job_dir, context, device, smoke=False):
    """Analyze one finished job; outputs go to <job_dir>/analysis_<stamp>/. Returns its result."""
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    job_dir = Path(job_dir)
    job = job_dir.name
    folder = job_dir / f"analysis_{stamp}"
    folder.mkdir()
    print(f"{job}: analysis", flush=True)
    state = torch.load(job_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    model = model_from_config(config).to(device).eval()
    model.load_state_dict(state["model"])
    with np.load(job_dir / "split_manifest.npz") as manifest:
        split = {name: manifest[name] for name in ("train", "validation", "test")}
    if config.get("fast"):          # fast training uses subsets; analysis must match
        smoke = True
    val = split["validation"] if not smoke else subsample(split["validation"], 8192, 0)
    images = context["images"]
    means, logvars = encode(model, images, device)
    recon = prefix_reconstruction(model, images, val, means, device)
    eta = np.stack([correlation_ratio(means[val], context["factors"][val, j])
                    for j in range(len(FACTORS))], axis=1)
    probes = prefix_probes(means, split, device, smoke)
    reference = int(np.random.default_rng(REFERENCE_SEED).choice(split["validation"]))
    traversal, low, high = traversals(model, means[val], means[reference], device)
    posterior_variance = np.exp(logvars[val]).mean(0) if config["beta"] > 0 else None
    path = figure6(job, means, means[val], traversal, low, high, posterior_variance, folder, stamp,
                   context["scales"], context["rotations"])
    result = {"objective": config["objective"], "beta": config["beta"], "seed": config["seed"],
              "lambda_sup": config["lambda_sup"], "epochs": state["epoch"],
              "prefix_reconstruction_nll": recon.tolist(),
              "activity": activity(means[val], logvars[val], recon, config["beta"]),
              "eta_squared": eta.tolist(), "prefix_probes": probes,
              "figure6": str(path), "reference_index": reference,
              "traversal_range": {"low": low.tolist(), "high": high.tolist()}}
    write_outputs({job: result}, folder, stamp,
                  {"job": job, "smoke": smoke, "probe_train": PROBE_TRAIN,
                   "variance_threshold": VARIANCE_THRESHOLD, "kl_threshold": KL_THRESHOLD,
                   "recon_gain_threshold": RECON_GAIN_THRESHOLD, "device": str(device)})
    save_runtime(str(folder), time.perf_counter() - started)
    print(f"{job}: analysis done -> {folder}", flush=True)
    return result


def latest_analysis(job_dir):
    """Newest saved analysis of this job: <job>/analysis_<stamp>/ or an older multi-job
    <run-dir>/analysis_<stamp>/ that contains it; None if there is none."""
    job_dir = Path(job_dir)
    folders = list(job_dir.glob("analysis_*")) + list(job_dir.parent.glob("analysis_2*"))
    for folder in sorted(folders, key=lambda f: f.name, reverse=True):
        path = folder / "analysis.json"
        if path.exists():
            results = json.loads(path.read_text())
            if job_dir.name in results:
                return results[job_dir.name]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--combine-only", action="store_true",
                        help="No recomputation: combine each job's newest saved analysis")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    jobs = args.jobs or sorted(p.parent.name for p in args.run_dir.glob("*/checkpoint_final.pt"))
    if not jobs:
        parser.error("No finished jobs found")
    results = {}
    if args.combine_only:
        for job in jobs:
            result = latest_analysis(args.run_dir / job)
            if result is None:
                parser.error(f"No saved analysis for {job}; run without --combine-only")
            results[job] = result
    else:
        context = load_context(args.data_path)
        for job in jobs:
            results[job] = analyze_job(args.run_dir / job, context, torch.device(args.device), args.smoke)
    if len(results) > 1:
        folder = args.run_dir / f"analysis_combined_{stamp}"
        folder.mkdir()
        write_outputs(results, folder, stamp, {"jobs": jobs, "combine_only": args.combine_only,
                                               "smoke": args.smoke})
        save_runtime(str(folder), time.perf_counter() - started)
        print(f"Combined analysis: {folder}", flush=True)


if __name__ == "__main__":
    main()
