"""Reproduce Higgins et al. Figure 5 with VAE and beta-VAE on dSprites.

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --fast
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --trial --epochs 2 --seeds 0 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --betas 1 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --betas 4 --device cuda:1
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --benchmark --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --trial --untrained-only --seeds 0 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --betas 1 --seeds 0 --epochs 100 --skip-untrained --resume-checkpoint /path/to/checkpoint_epoch_10.pt --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --output-root /tmp/dsprites_runs --device cpu --fast
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --betas 4 --optimizer adam --seeds 0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --objective capacity --optimizer adam --initialization xavier --decoder-activation relu --seeds 0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --mrl-reconstruction --betas 1 4 --seeds 0 1 2 --classifier-seeds 0 1 2 --classifier-epochs 2000 --no-score-filtering --skip-untrained

--trial keeps full data/metric settings but saves under test_runs. --fast uses
an image subset and reduced metric settings. --benchmark performs 50 warm-up
and 300 timed updates without a scientific metric evaluation. Runs use final
fixed-budget VAE checkpoints. Optional --epochs, --seeds, --betas, --device,
--batch-size, --data-path, and --output-root overrides are logged.
Untrained controls are evaluated once per model seed by default. Use
--untrained-only for evaluation without VAE optimization, or --skip-untrained
to omit that control. --classifier-epochs overrides the classifier epoch cap.
--resume-checkpoint restores model, optimizer, and RNG states into a fresh run
folder; --epochs is the total target epoch count, not additional epochs.
Default --objective beta --optimizer adagrad --initialization default
--decoder-activation tanh preserves the existing paper-style implementation.
Adam defaults to LR 0.0005; Adagrad to 0.01; --lr overrides either. Capacity
uses gamma * abs(batch-mean KL - C(step)), with C ramping 0 to 25 over
100000 updates. --gamma, --capacity-max and --capacity-steps override this.
Capacity mode trains one condition per seed; beta is not its penalty multiplier.
--mrl-reconstruction uses every masked prefix, one shared sample/decoder,
mean scalar reconstruction loss, and one full-posterior KL. It requires fixed
beta. --no-score-filtering retains all classifier scores without best-half stats.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE_ROOT))
from utility import save_runtime
from vae_reproduction.data import DSprites, TARGET_NAMES, file_checksum
from vae_reproduction.model import VAE, model_from_config, vae_loss, mrl_vae_loss
from vae_reproduction.metric import metric_examples, fit_classifier

# ==============================================================================
# CONFIG — complete full-run settings; --fast overrides are inside main().
# ==============================================================================
EXPERIMENT_NOTE = (
    "Unsupervised VAE/beta-VAE Figure 5 reproduction, before adding MRL. "
    "Paper: MLP architecture, Adagrad 0.01, beta 1/4, 10 model replicas, "
    "3 classifier initializations, 5000 metric test examples. Our choices: "
    "full archive for VAE training, 100 epochs, batch 256, pair average 64, "
    "10000/2000 metric train/validation examples and validation stopping. "
    "Report all scores; historical best-half summaries are optional and disabled "
    "by --no-score-filtering. Reconstruction MRL averages all prefix BCE losses "
    "with a shared sample/decoder and adds full-posterior KL once. "
    "Optional PyTorch variants: Adam, Xavier-uniform weights/zero biases, "
    "ReLU decoder, or gamma*abs(batch-mean KL-C(step)) capacity objective; "
    "see resolved configuration for the actual training condition."
)
DATA_PATH = PROJECT_ROOT / "data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
OUTPUT_ROOT = PROJECT_ROOT / "files/results"
LATENT_DIM = 10
HIDDEN_DIM = 1200
BETAS = [1.0, 4.0]
MODEL_SEEDS = list(range(10))
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.01
OBJECTIVE = "beta"
OPTIMIZER = "adagrad"
INITIALIZATION = "default"
DECODER_ACTIVATION = "tanh"
RECONSTRUCTION_MRL = False
ADAM_LR = 5e-4
GAMMA = 1000.0
CAPACITY_MAX = 25.0
CAPACITY_STEPS = 100000
NUM_WORKERS = 0
CHECKPOINT_EVERY = 10
DEVICE = "cuda:0"
METRIC_PAIRS = 64
METRIC_TRAIN = 10000
METRIC_VALIDATION = 2000
METRIC_TEST = 5000
METRIC_SAMPLING_SEEDS = {"train": 1729, "validation": 1730, "test": 1731}
CLASSIFIER_SEEDS = [0, 1, 2]
CLASSIFIER_CONFIG = {"lr": 0.01, "batch_size": 256, "epochs": 200,
                     "patience": 20, "min_delta": 1e-4}
INCLUDE_UNTRAINED = True
# ==============================================================================


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def snapshot_code(run_dir):
    # Same whole-code snapshot convention as utility.py, excluding existing
    # generated reference reproduction outputs as well as caches and git data.
    shutil.copytree(CODE_ROOT, run_dir / "code_snapshot", ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", ".DS_Store", "figure", "reproduction_outputs"
    ))


def save_checkpoint(path, model, optimizer, loader_generator, epoch, config, global_step=None):
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "config": config, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "loader_rng": loader_generator.get_state(),
    }
    if global_step is not None:
        state["global_step"] = global_step
    torch.save(state, path)


def build_optimizer(model, config):
    """Only the VAE optimizer changes; metric-classifier optimization is separate."""
    kind = config.get("optimizer", "adagrad")
    if kind == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=config["lr"])
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    raise ValueError("optimizer must be adagrad or adam")


def capacity_at_step(config, step):
    """Step zero has zero target capacity; resume preserves the update counter."""
    return config.get("capacity_max", CAPACITY_MAX) * min(
        step / config.get("capacity_steps", CAPACITY_STEPS), 1.0
    )


def training_loss(logits, images, mean, logvar, config, step):
    return vae_loss(logits, images, mean, logvar, config.get("beta", 4.0),
                    objective=config.get("objective", "beta"),
                    capacity=capacity_at_step(config, step),
                    gamma=config.get("gamma", GAMMA))


def train_model(model, dataset, config, device, folder):
    if config.get("reconstruction_mrl", False):
        if config.get("objective", "beta") != "beta":
            raise ValueError("Reconstruction MRL requires fixed-beta training")
        if config.get("mrl_prefixes", list(range(1, config["latent_dim"] + 1))) != list(range(1, config["latent_dim"] + 1)):
            raise ValueError("Reconstruction MRL requires every native prefix")
    optimizer = build_optimizer(model, config)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=config["num_workers"],
                        pin_memory=device.type == "cuda")
    history = []
    start_epoch = 1
    global_step = 0
    if config.get("resume_checkpoint"):
        checkpoint_path = Path(config["resume_checkpoint"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        previous = state["config"]
        for key in ("beta", "seed", "latent_dim", "hidden_dim", "batch_size", "lr",
                    "num_workers", "image_limit", "dataset_sha256"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Resume configuration mismatch for {key}")
        legacy_defaults = {"objective": "beta", "optimizer": "adagrad",
                           "initialization": "default", "decoder_activation": "tanh",
                           "gamma": GAMMA, "capacity_max": CAPACITY_MAX,
                           "capacity_steps": CAPACITY_STEPS,
                           "reconstruction_mrl": False,
                           "mrl_prefixes": list(range(1, config["latent_dim"] + 1)),
                           "mrl_reduction": "mean_scalar_losses",
                           "mrl_sampling": "one_full_sample_masked",
                           "mrl_kl": "full_posterior_once"}
        for key, default in legacy_defaults.items():
            if previous.get(key, default) != config.get(key, default):
                raise ValueError(f"Resume configuration mismatch for {key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch = state["epoch"] + 1
        global_step = state.get("global_step", state["epoch"] * len(loader))
        if start_epoch > config["epochs"]:
            raise ValueError("Checkpoint already reached the requested total epoch budget")
        source_history = checkpoint_path.parent / "training.jsonl"
        if source_history.exists():
            history = [json.loads(line) for line in source_history.read_text().splitlines()]
            history = [row for row in history if row["epoch"] < start_epoch]
            (folder / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        print(f"Resuming {checkpoint_path} at epoch {start_epoch}/{config['epochs']}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config["epochs"] + 1):
        synchronize(device)
        started = time.perf_counter()
        totals = torch.zeros(5 + config["latent_dim"], device=device)
        prefix_totals = torch.zeros(config["latent_dim"], device=device)
        seen = 0
        model.train()
        for images in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            target_capacity = capacity_at_step(config, global_step)
            if config.get("reconstruction_mrl", False):
                loss, reconstruction, kl, coordinates, prefixes = mrl_vae_loss(
                    model, images, config["beta"], backward=True)
                prefix_totals += prefixes * len(images)
            else:
                logits, mean, logvar = model(images)
                loss, reconstruction, kl, coordinates = training_loss(
                    logits, images, mean, logvar, config, global_step)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite VAE loss at epoch {epoch}")
            if not config.get("reconstruction_mrl", False):
                loss.backward()
            optimizer.step()
            penalty = (config.get("gamma", GAMMA) * (kl - target_capacity).abs()
                       if config.get("objective", "beta") == "capacity" else config["beta"] * kl)
            totals += torch.cat((torch.stack((loss, reconstruction, kl)), coordinates,
                                 torch.stack((penalty, kl.new_tensor(target_capacity))))).detach() * len(images)
            seen += len(images)
            global_step += 1
        synchronize(device)
        averages = (totals / seen).cpu().tolist()
        row = {"epoch": epoch, "loss": averages[0], "reconstruction_nll": averages[1],
               "kl": averages[2], "regularization_penalty": averages[-2],
               "capacity_target_mean": averages[-1] if config.get("objective", "beta") == "capacity" else None,
               "coordinate_kl": averages[3:3 + config["latent_dim"]],
               "global_step": global_step, "objective": config.get("objective", "beta"),
               "seconds": time.perf_counter() - started,
               "images": seen, "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        if config.get("objective", "beta") == "beta":
            row["weighted_kl"] = config["beta"] * averages[2]
        row["reconstruction_mrl"] = config.get("reconstruction_mrl", False)
        if row["reconstruction_mrl"]:
            row["prefix_reconstruction_nll"] = (prefix_totals / seen).cpu().tolist()
            row["full_reconstruction_nll"] = row["prefix_reconstruction_nll"][-1]
        else:
            row["full_reconstruction_nll"] = row["reconstruction_nll"]
        history.append(row)
        with (folder / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        condition = "capacity" if config.get("objective", "beta") == "capacity" else f"beta={config['beta']:g}"
        if config.get("reconstruction_mrl", False):
            condition = f"MRL {condition}"
        print(f"{condition} seed={config['seed']} epoch={epoch}/{config['epochs']} "
              f"NLL={row['reconstruction_nll']:.3f} KL={row['kl']:.3f} "
              f"total={row['loss']:.3f} time={row['seconds']:.1f}s", flush=True)
        if epoch % config["checkpoint_every"] == 0:
            save_checkpoint(folder / f"checkpoint_epoch_{epoch}.pt", model, optimizer, generator, epoch, config, global_step)
    save_checkpoint(folder / "checkpoint_final.pt", model, optimizer, generator, config["epochs"], config, global_step)
    return history


@torch.no_grad()
def encode_archive(model, images, batch_size, device):
    model.eval()
    means = np.empty((len(images), model.mean.out_features), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start:start + batch_size].reshape(-1, 4096))
        mean, _ = model.encode(batch.to(device=device, dtype=torch.float32))
        means[start:start + len(batch)] = mean.cpu().numpy()
    if not np.isfinite(means).all():
        raise RuntimeError("Nonfinite encoded means")
    return means


@torch.no_grad()
def diagnostic_figures(model, images, means, folder, stamp, device):
    references = np.random.default_rng(2026).choice(len(images), 4, replace=False)
    original = images[references]
    z = torch.from_numpy(means[references]).to(device)
    decoded = model.decoder(z).sigmoid().cpu().numpy().reshape(-1, 64, 64)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(4):
        axes[0, i].imshow(original[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(decoded[i], cmap="gray", vmin=0, vmax=1)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Input (top); reconstruction from posterior mean (bottom)")
    fig.tight_layout()
    fig.savefig(folder / f"reconstructions_{stamp}.png", dpi=140)
    plt.close(fig)
    ranges = np.quantile(means, [0.05, 0.95], axis=0)
    np.save(folder / "traversal_quantiles.npy", ranges)
    for reference, reference_z in enumerate(z):
        rows = []
        for coordinate in range(means.shape[1]):
            traversal = reference_z.repeat(7, 1)
            traversal[:, coordinate] = torch.linspace(float(ranges[0, coordinate]), float(ranges[1, coordinate]), 7, device=device)
            rows.append(model.decoder(traversal).sigmoid().cpu().numpy().reshape(7, 64, 64))
        fig, axes = plt.subplots(len(rows), 7, figsize=(9, 12))
        for i, row in enumerate(rows):
            for j, image in enumerate(row):
                axes[i, j].imshow(image, cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            axes[i, 0].set_ylabel(f"z{i + 1}")
        fig.suptitle(f"Reference {references[reference]}; empirical 5–95% traversals")
        fig.tight_layout()
        fig.savefig(folder / f"traversal_ref_{reference}_{stamp}.png", dpi=120)
        plt.close(fig)


def plot_training(histories, run_dir, stamp):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for tag, history in histories.items():
        for ax, key in zip(axes, ("loss", "reconstruction_nll", "kl")):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=tag)
            ax.set_xlabel("Epoch")
            ax.set_title(key)
    if histories:
        axes[0].legend(fontsize=7)
    else:
        fig.suptitle("Untrained baseline: no VAE optimization performed")
    fig.tight_layout()
    fig.savefig(run_dir / f"training_curves_{stamp}.png", dpi=150)
    plt.close(fig)


def report(scores, run_dir, trial, discard_bottom_half=True):
    lines = ["dSprites Higgins disentanglement metric", "Trial/exploratory" if trial else "Fixed-budget reproduction",
             "Scores are held-fixed-factor classification accuracies, not factor-value accuracy."]
    if any(r.get("objective") == "capacity" for r in scores):
        lines.append("Capacity-controlled extension: beta metadata is unused by this training loss; no fixed-beta paper reference is assigned.")
    summary = {}
    def group_key(record):
        if record["beta"] is None:
            return "untrained"
        if record.get("objective", "beta") == "capacity":
            return "capacity"
        return ("mrl_" if record.get("reconstruction_mrl", False) else "") + str(record["beta"])
    for key in sorted({group_key(r) for r in scores}):
        records = [r for r in scores if group_key(r) == key]
        beta = records[0]["beta"]
        raw = np.array([r["accuracy"] for r in records])
        model_means = [np.mean([r["accuracy"] for r in records if r["model_seed"] == seed])
                       for seed in sorted({r["model_seed"] for r in records})]
        kept = np.sort(raw)[len(raw) // 2:] if discard_bottom_half else raw
        capacity_mode = beta is not None and records[0].get("objective", "beta") == "capacity"
        reference = None if capacity_mode or records[0].get("reconstruction_mrl", False) else {None: 0.4414, 1.0: 0.6158, 4.0: 0.9923}.get(beta)
        stats = {"all_mean": float(raw.mean()), "all_sd": float(raw.std(ddof=1)) if len(raw) > 1 else None,
                 "model_seed_means": model_means,
                 "model_seed_sd": float(np.std(model_means, ddof=1)) if len(model_means) > 1 else None,
                 "best_half_mean": float(kept.mean()),
                 "best_half_sd": float(kept.std(ddof=1)) if len(kept) > 1 else None,
                 "count": len(raw), "kept_count": len(kept), "paper_reference": reference,
                 "filtered_difference_percentage_points": float(100 * (kept.mean() - reference)) if reference else None}
        if not discard_bottom_half:
            for field in ("best_half_mean", "best_half_sd", "kept_count", "filtered_difference_percentage_points"):
                stats.pop(field)
        stats["score_filtering"] = "bottom_half" if discard_bottom_half else "none"
        label = "untrained" if beta is None else ("capacity-controlled" if capacity_mode else f"{'MRL ' if records[0].get('reconstruction_mrl', False) else ''}beta={beta:g}")
        summary[key] = stats
        lines.append(f"\n{label}: all mean={100 * raw.mean():.2f}%; n={len(raw)}")
        if discard_bottom_half:
            lines.append(f"  best-half mean={100 * kept.mean():.2f}% (paper-style statistic; ± convention unspecified)")
        else:
            lines.append("  All scores retained; no filtering.")
        for record in records:
            lines.append(f"  model_seed={record['model_seed']} classifier_seed={record['classifier_seed']}: {100 * record['accuracy']:.2f}%")
    (run_dir / "results_summary.txt").write_text("\n".join(lines) + "\n")
    write_json(run_dir / "metric_scores.json", scores)
    write_json(run_dir / "metric_summary.json", summary)


def evaluate_model(model, dataset, config, device, folder, stamp, beta, seed):
    """Identical frozen-encoder evaluation for trained and random models."""
    print(f"{folder.name}: encoding full archive for metric", flush=True)
    means = encode_archive(model, dataset.images, config["batch_size"], device)
    np.save(folder / "latent_means.npy", means)
    diagnostic_figures(model, dataset.images, means, folder, stamp, device)
    metric_sets = {}
    for split, sampling_seed in config["metric_sampling_seeds"].items():
        metric_sets[split] = metric_examples(
            means, config[f"metric_{split}"], config["metric_pairs"], sampling_seed
        )
        x, y = metric_sets[split]
        np.savez(folder / f"metric_{split}.npz", features=x, labels=y)
    scores = []
    for classifier_seed in config["classifier_seeds"]:
        result, state, classifier_history = fit_classifier(
            metric_sets["train"], metric_sets["validation"], metric_sets["test"],
            classifier_seed, config["classifier"]
        )
        result.update(beta=beta, model_seed=seed,
                      model_kind="untrained" if beta is None else "trained",
                      objective=config.get("objective", "beta"),
                      optimizer=config.get("optimizer", "adagrad"),
                      initialization=config.get("initialization", "default"),
                      decoder_activation=config.get("decoder_activation", "tanh"))
        result["reconstruction_mrl"] = config.get("reconstruction_mrl", False)
        scores.append(result)
        torch.save(state, folder / f"classifier_seed_{classifier_seed}.pt")
        write_json(folder / f"classifier_seed_{classifier_seed}.json",
                   {"result": result, "history": classifier_history})
        print(f"{folder.name}: classifier_seed={classifier_seed} "
              f"disentanglement={100 * result['accuracy']:.2f}%", flush=True)
    return scores


def benchmark(model, dataset, config, device):
    optimizer = build_optimizer(model, config)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        pin_memory=device.type == "cuda", num_workers=0)
    iterator = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = None
    for step in range(350):
        if step == 50:
            synchronize(device)
            start = time.perf_counter()
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if config.get("reconstruction_mrl", False):
            loss, *_ = mrl_vae_loss(model, images, config["betas"][0], backward=True)
        else:
            logits, mean, logvar = model(images)
            loss, *_ = training_loss(logits, images, mean, logvar,
                                     {**config, "beta": config["betas"][0]}, step)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite benchmark loss")
        if not config.get("reconstruction_mrl", False):
            loss.backward()
        optimizer.step()
    synchronize(device)
    seconds = time.perf_counter() - start
    return {"timed_updates": 300, "seconds": seconds, "updates_per_second": 300 / seconds,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "note": "Single-process measurement; concurrent GPU throughput not measured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--untrained-only", action="store_true")
    baseline.add_argument("--skip-untrained", action="store_true")
    parser.add_argument("--classifier-epochs", type=int)
    parser.add_argument("--classifier-seeds", type=int, nargs="+")
    parser.add_argument("--no-score-filtering", action="store_true")
    parser.add_argument("--mrl-reconstruction", action="store_true", default=RECONSTRUCTION_MRL)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--betas", type=float, nargs="+")
    parser.add_argument("--objective", choices=("beta", "capacity"), default=OBJECTIVE)
    parser.add_argument("--optimizer", choices=("adagrad", "adam"), default=OPTIMIZER)
    parser.add_argument("--lr", type=float, help="VAE learning rate; classifier settings are independent")
    parser.add_argument("--initialization", choices=("default", "xavier"), default=INITIALIZATION)
    parser.add_argument("--decoder-activation", choices=("tanh", "relu"), default=DECODER_ACTIVATION)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--capacity-max", type=float, default=CAPACITY_MAX)
    parser.add_argument("--capacity-steps", type=int, default=CAPACITY_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.objective == "capacity" and args.betas is not None:
        parser.error("--betas applies only to --objective beta; capacity uses --gamma")
    if args.mrl_reconstruction and args.objective != "beta":
        parser.error("Reconstruction MRL requires fixed-beta training")
    betas = args.betas if args.betas is not None else (BETAS if args.objective == "beta" else [4.0])
    lr = args.lr if args.lr is not None else (ADAM_LR if args.optimizer == "adam" else LR)
    if not np.isfinite(lr) or lr <= 0:
        parser.error("VAE learning rate must be finite and positive")
    if not np.isfinite(args.gamma) or args.gamma <= 0 or not np.isfinite(args.capacity_max) or args.capacity_max < 0 or args.capacity_steps < 1:
        parser.error("Gamma must be finite/positive, capacity-max finite/nonnegative, and capacity-steps positive")
    config = {"experiment_note": EXPERIMENT_NOTE, "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
              "betas": betas, "seeds": args.seeds if args.seeds is not None else MODEL_SEEDS,
              "epochs": EPOCHS, "batch_size": args.batch_size, "lr": lr, "num_workers": NUM_WORKERS,
              "objective": args.objective, "optimizer": args.optimizer,
              "reconstruction_mrl": args.mrl_reconstruction,
              "mrl_prefixes": list(range(1, LATENT_DIM + 1)),
              "mrl_reduction": "mean_scalar_losses", "mrl_sampling": "one_full_sample_masked",
              "mrl_kl": "full_posterior_once", "score_filtering": "none" if args.no_score_filtering else "bottom_half",
              "initialization": args.initialization, "decoder_activation": args.decoder_activation,
              "gamma": args.gamma, "capacity_max": args.capacity_max, "capacity_steps": args.capacity_steps,
              "checkpoint_every": CHECKPOINT_EVERY, "device": args.device, "image_limit": None,
              "metric_pairs": METRIC_PAIRS, "metric_train": METRIC_TRAIN,
              "metric_validation": METRIC_VALIDATION, "metric_test": METRIC_TEST,
              "metric_sampling_seeds": METRIC_SAMPLING_SEEDS, "classifier_seeds": CLASSIFIER_SEEDS,
              "classifier": dict(CLASSIFIER_CONFIG), "data_path": str(args.data_path.resolve()),
              "fast": args.fast, "trial": args.trial, "benchmark": args.benchmark,
              "include_untrained": INCLUDE_UNTRAINED and not args.skip_untrained,
              "untrained_only": args.untrained_only,
              "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None}
    config["beta_used_in_loss"] = args.objective == "beta"
    if args.benchmark and args.untrained_only:
        parser.error("--benchmark performs optimization; cannot combine with --untrained-only")
    if args.fast:
        config.update(epochs=2, image_limit=8192, metric_pairs=16, metric_train=1000,
                      metric_validation=400, metric_test=1000, classifier_seeds=[0])
        config["classifier"].update(epochs=40, patience=10)
        if args.seeds is None:
            config["seeds"] = [0]
    elif args.trial and args.seeds is None:
        config["seeds"] = [0]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.classifier_seeds is not None:
        config["classifier_seeds"] = args.classifier_seeds
    if len(set(config["classifier_seeds"])) != len(config["classifier_seeds"]):
        parser.error("Duplicate classifier seeds would overwrite artifacts")
    if args.classifier_epochs is not None:
        if args.classifier_epochs < 1:
            parser.error("Classifier epochs must be positive")
        config["classifier"]["epochs"] = args.classifier_epochs
    if config["epochs"] < 1 or args.batch_size < 1 or any(not np.isfinite(b) or b <= 0 for b in betas):
        parser.error("Epochs, batch size and beta must be positive")
    if len(set(betas)) != len(betas) or len(set(config["seeds"])) != len(config["seeds"]):
        parser.error("Duplicate betas/seeds would overwrite run artifacts")
    if args.resume_checkpoint:
        if len(betas) != 1 or len(config["seeds"]) != 1 or args.untrained_only or args.benchmark:
            parser.error("Resume requires exactly one beta and model seed, without benchmark/untrained-only")
        if not args.resume_checkpoint.is_file():
            parser.error("Resume checkpoint does not exist")
        config["include_untrained"] = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; verify GPU permissions or explicitly use --device cpu")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    base = args.output_root / "test_runs" if args.fast or args.trial or args.benchmark else args.output_root
    run_dir = base / f"exprmnt_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run_dir}", flush=True)
    write_json(run_dir / "config.json", config)
    (run_dir / "experiment_description.log").write_text(
        EXPERIMENT_NOTE + "\n\nResolved configuration:\n" + json.dumps(config, indent=2) + "\n"
    )
    snapshot_code(run_dir)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                   "cuda_build": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "optimizer": args.optimizer,
                   "optimizer_defaults": build_optimizer(torch.nn.Linear(1, 1), config).defaults,
                   "tf32": False}
    if args.optimizer == "adagrad":
        environment["adagrad_defaults"] = environment["optimizer_defaults"]
    environment["packages"] = {
        package: metadata.version(package)
        for package in ("torch", "numpy", "matplotlib", "scikit-learn", "scipy", "tqdm")
    }
    if shutil.which("nvidia-smi"):
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        environment["nvidia_smi"] = {"returncode": driver.returncode,
                                      "stdout": driver.stdout.strip(), "stderr": driver.stderr.strip()}
    write_json(run_dir / "environment.json", environment)
    histories, scores = {}, []
    try:
        config["dataset_sha256"] = file_checksum(args.data_path)
        dataset = DSprites(args.data_path, limit=config["image_limit"], seed=0)
        config["training_images"] = len(dataset)
        write_json(run_dir / "config.json", config)
        if args.benchmark:
            seed_everything(config["seeds"][0])
            result = benchmark(model_from_config(config).to(device), dataset, config, device)
            write_json(run_dir / "benchmark.json", result)
            (run_dir / "results_summary.txt").write_text(json.dumps(result, indent=2) + "\n")
            write_json(run_dir / "status.json", {"status": "completed"})
            print(json.dumps(result), flush=True)
            return
        if config["include_untrained"] or args.untrained_only:
            for seed in config["seeds"]:
                seed_everything(seed)
                folder = run_dir / f"untrained_seed_{seed}"
                folder.mkdir()
                write_json(folder / "config.json", {**config, "seed": seed, "vae_training_epochs": 0})
                model = model_from_config(config).to(device)
                initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
                torch.save({"model": initial, "epoch": 0, "config": config}, folder / "checkpoint_untrained.pt")
                scores.extend(evaluate_model(model, dataset, config, device, folder, stamp, None, seed))
                if any(not torch.equal(initial[key], value) for key, value in model.state_dict().items()):
                    raise RuntimeError("Untrained model parameters changed during evaluation")
                report(scores, run_dir, args.fast or args.trial, discard_bottom_half=not args.no_score_filtering)
                plot_training(histories, run_dir, stamp)
                del model, initial
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        for beta in ([] if args.untrained_only else config["betas"]):
            for seed in config["seeds"]:
                seed_everything(seed)
                tag = f"capacity_seed_{seed}" if args.objective == "capacity" else f"beta_{beta:g}_seed_{seed}"
                if args.mrl_reconstruction:
                    tag = "mrl_" + tag
                folder = run_dir / tag
                folder.mkdir()
                run_config = {**config, "beta": beta, "seed": seed}
                write_json(folder / "config.json", run_config)
                model = model_from_config(config).to(device)
                histories[tag] = train_model(model, dataset, run_config, device, folder)
                plot_training(histories, run_dir, stamp)
                scores.extend(evaluate_model(model, dataset, config, device, folder, stamp, beta, seed))
                report(scores, run_dir, args.fast or args.trial, discard_bottom_half=not args.no_score_filtering)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        write_json(run_dir / "status.json", {"status": "completed"})
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        save_runtime(str(run_dir), time.perf_counter() - started)


if __name__ == "__main__":
    main()
