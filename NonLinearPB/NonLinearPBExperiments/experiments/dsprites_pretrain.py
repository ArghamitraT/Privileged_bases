"""Pretrain one MRL source encoder on dSprites: S_recon or S_posX, beta in {0, 1, 4}.

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_pretrain.py --fast --objective posx --beta 4 --lambda-sup 30 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_pretrain.py --objective recon --beta 0 --seed 0 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_pretrain.py --objective posx --beta 4 --seed 0 --lambda-sup 30 --epochs 10 --run-dir /path/to/exprmnt_x --device cuda:1

Spec: NonLinearPB/plans/PrivilagedBasisRotation.md (pretraining). One process
trains one (objective, beta, seed[, lambda]) model on the shared seed-0 80% train
split; scripts/run_dsprites_pretrain.py queues many of these across GPUs.
--run-dir puts the job folder inside an existing experiment folder (the
queue's); otherwise a new exprmnt_<stamp> folder is created. --fast trains 2
epochs on an 8192-image train subset under test_runs. --tf32 enables TF32
matmuls (about 1.3x faster on RTX 3090; off by default for continuity).
When training completes, the same process analyzes the model at once
(scripts/analyze_pretrain.analyze_job) into <job>/analysis_<stamp>/; an
analysis error is logged to analysis_status.json and does not fail the run.
--no-analysis skips it; re-run standalone with scripts/analyze_pretrain.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE_ROOT))
from utility import save_runtime
from vae_reproduction.data import DSprites
from vae_reproduction.model import PrefixHeads, model_from_config, mrl_pretrain_loss
from vae_reproduction.runs import (DATA_PATH, OUTPUT_ROOT, dataset_record, environment,
                                   make_run_dir, prepare_split, snapshot, write_json)
from vae_reproduction.tasks import factor_labels, n_classes, subsample
from scripts.analyze_pretrain import analyze_job, load_context

# ==============================================================================
# CONFIG — complete full-run settings; --fast overrides are inside main().
# ==============================================================================
EXPERIMENT_NOTE = (
    "Pretraining for the privileged-basis study (PrivilagedBasisRotation.md). "
    "Same MLP beta-VAE as vae_reproduction (Adagrad 0.01, batch 256, 100 epochs, "
    "tanh decoder, default init). Changes from the earlier MRL runs: train on the "
    "seed-0 80% split instead of the full archive (supervised labels must not see "
    "validation images); all ten prefixes decoded as one batched pass (same mean "
    "prefix BCE, faster); beta = 0 is deterministic (z = mu, no KL); S_posX adds "
    "lambda_sup * mean_k CE(head_k(z[:k]), X) / ln(32). KL is never nested."
)
LATENT_DIM = 10
HIDDEN_DIM = 1200
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.01
SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
OBJECTIVES = {"recon": None, "posx": "x"}   # objective -> supervised factor
VALIDATE_EVERY = 10
CHECKPOINT_EVERY = 10
ACTIVE_THRESHOLD = 0.01                    # v_i / max_j v_j
EVAL_BATCH = 4096
# ==============================================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def job_name(objective, beta, seed, lambda_sup, epochs=EPOCHS):
    """Short runs (e.g. lambda pilots) get an _ep<N> suffix so they never collide with full runs."""
    name = f"{objective}_beta{beta:g}_seed{seed}"
    name += f"_lam{lambda_sup:g}" if objective != "recon" else ""
    return name + (f"_ep{epochs}" if epochs != EPOCHS else "")


@torch.no_grad()
def validate(model, heads, images, indices, x_labels, config, device):
    """Posterior-mean diagnostics on validation images (mu everywhere, never samples)."""
    model.eval()
    d = config["latent_dim"]
    means, recon = [], torch.zeros(d, device=device)
    correct = torch.zeros(d, device=device) if heads is not None else None
    for start in range(0, len(indices), EVAL_BATCH):
        batch = torch.as_tensor(indices[start:start + EVAL_BATCH], device=device)
        x = images[batch].float()
        out = mrl_pretrain_loss(model, x, 0.0)        # beta 0 -> z = mu, recon only
        recon += out["prefix_recon"] * len(batch)
        mean, _ = model.encode(x)
        means.append(mean)
        if heads is not None:
            labels = torch.as_tensor(x_labels[start:start + EVAL_BATCH], device=device)
            correct += (heads(mean).argmax(-1) == labels[None]).float().sum(dim=1)
    model.train()
    mu = torch.cat(means).cpu().numpy().astype(np.float64)
    variance = mu.var(axis=0)
    relative = variance / variance.max()
    rho = [float(spearmanr(mu[:, i], x_labels).statistic) for i in range(d)]
    result = {"prefix_reconstruction_nll": (recon / len(indices)).tolist(),
              "mu_variance": variance.tolist(), "relative_variance": relative.tolist(),
              "active_mask": (relative > ACTIVE_THRESHOLD).astype(int).tolist(),
              "d_active": int((relative > ACTIVE_THRESHOLD).sum()),
              "spearman_x_per_coordinate": rho}
    if heads is not None:
        result["head_accuracy_x"] = (correct / len(indices)).tolist()
        with torch.no_grad():
            pred = torch.cat([heads(torch.as_tensor(mu[s:s + EVAL_BATCH], dtype=torch.float32, device=device))[0].argmax(-1)
                              for s in range(0, len(mu), EVAL_BATCH)]).cpu().numpy()
        result["head1_within_one_bin_x"] = float(np.mean(np.abs(pred - x_labels) <= 1))
    return result


def train(model, heads, images, split, config, device, folder):
    d = config["latent_dim"]
    train_idx = torch.as_tensor(split["train"], device=device)
    factor = OBJECTIVES[config["objective"]]
    all_x = torch.as_tensor(factor_labels(np.arange(len(images)), "x"), device=device)
    val_idx = split["validation"]
    val_x = factor_labels(val_idx, "x")
    parameters = list(model.parameters()) + (list(heads.parameters()) if heads is not None else [])
    optimizer = torch.optim.Adagrad(parameters, lr=config["lr"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    ln_classes = math.log(n_classes(factor)) if factor else 1.0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        order = train_idx[torch.randperm(len(train_idx), generator=generator, device=device)]
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "sup": 0.0}
        prefix_recon = torch.zeros(d, device=device)
        coordinate_kl = torch.zeros(d, device=device)
        prefix_acc = torch.zeros(d, device=device)
        totals = torch.zeros(4, device=device)
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            x = images[batch].float()
            out = mrl_pretrain_loss(model, x, config["beta"], heads,
                                    all_x[batch] if heads is not None else None,
                                    config["lambda_sup"], ln_classes)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()
            n = len(batch)
            totals += torch.stack((out["loss"].detach(), out["recon"].detach(), out["kl"].detach(),
                                   out["sup"].detach() if "sup" in out else out["kl"].new_zeros(()))) * n
            prefix_recon += out["prefix_recon"].detach() * n
            coordinate_kl += out["coordinate_kl"].detach() * n
            if "prefix_acc" in out:
                prefix_acc += out["prefix_acc"] * n
        seen = len(order)
        if not torch.isfinite(totals).all():
            raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
        torch.cuda.synchronize(device)
        values = (totals / seen).tolist()
        row = {"epoch": epoch, **dict(zip(sums, values)),
               "prefix_reconstruction_nll": (prefix_recon / seen).tolist(),
               "coordinate_kl": (coordinate_kl / seen).tolist(),
               "seconds": time.perf_counter() - started}
        if heads is not None:
            row["train_head_accuracy_x"] = (prefix_acc / seen).tolist()
        if epoch % VALIDATE_EVERY == 0 or epoch == config["epochs"]:
            row["validation"] = validate(model, heads, images, val_idx, val_x, config, device)
        history.append(row)
        with (folder / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{folder.name} epoch={epoch}/{config['epochs']} recon={row['recon']:.2f} "
              f"full={row['prefix_reconstruction_nll'][-1]:.2f} KL={row['kl']:.2f} "
              f"sup={row['sup']:.3f} time={row['seconds']:.1f}s", flush=True)
        if epoch % CHECKPOINT_EVERY == 0 or epoch == config["epochs"]:
            name = "checkpoint_final.pt" if epoch == config["epochs"] else f"checkpoint_epoch_{epoch}.pt"
            torch.save({"model": model.state_dict(),
                        "heads": heads.state_dict() if heads is not None else None,
                        "optimizer": optimizer.state_dict(), "epoch": epoch, "config": config,
                        "torch_rng": torch.get_rng_state(), "cuda_generator": generator.get_state()},
                       folder / name)
    return history


def plot_training(history, folder, stamp):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    axes[0].plot(epochs, [r["loss"] for r in history])
    axes[0].set_title("total loss")
    prefix = np.array([r["prefix_reconstruction_nll"] for r in history])
    for k in range(prefix.shape[1]):
        axes[1].plot(epochs, prefix[:, k], label=f"k={k + 1}", color=plt.cm.viridis(k / 9))
    axes[1].set_title("train reconstruction per prefix (nats)")
    axes[1].legend(fontsize=6, ncol=2)
    axes[2].plot(epochs, [r["kl"] for r in history])
    axes[2].set_title("KL (full latent)")
    if "train_head_accuracy_x" in history[0]:
        acc = np.array([r["train_head_accuracy_x"] for r in history])
        for k in range(acc.shape[1]):
            axes[3].plot(epochs, acc[:, k], color=plt.cm.viridis(k / 9))
        axes[3].set_title("train head accuracy, X (per prefix)")
    else:
        axes[3].set_title("no supervised head")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.suptitle(folder.name)
    fig.tight_layout()
    fig.savefig(folder / f"training_curves_{stamp}.png", dpi=130)
    plt.close(fig)


def summary_text(config, final):
    lines = [f"{config['job']}: objective={config['objective']} beta={config['beta']:g} "
             f"seed={config['seed']} lambda_sup={config['lambda_sup']:g} epochs={config['epochs']}",
             "Validation, posterior mean:",
             "  prefix recon (nats): " + " ".join(f"{v:.2f}" for v in final["prefix_reconstruction_nll"]),
             f"  d_active (v_i/max v > {ACTIVE_THRESHOLD}): {final['d_active']}  mask={final['active_mask']}",
             "  relative variance: " + " ".join(f"{v:.3g}" for v in final["relative_variance"]),
             "  Spearman(mu_i, X): " + " ".join(f"{v:+.3f}" for v in final["spearman_x_per_coordinate"])]
    if "head_accuracy_x" in final:
        lines += ["  X head accuracy per prefix: " + " ".join(f"{v:.3f}" for v in final["head_accuracy_x"]),
                  f"  X head prefix-1 within +-1 bin: {final['head1_within_one_bin_x']:.3f}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-sup", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()
    if args.beta < 0 or not math.isfinite(args.beta):
        parser.error("beta must be finite and >= 0")
    if args.objective == "recon" and args.lambda_sup != 0:
        parser.error("recon has no supervised head; leave --lambda-sup at 0")
    if args.objective != "recon" and args.lambda_sup <= 0:
        parser.error("supervised objectives need --lambda-sup > 0")
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    run_dir = args.run_dir or make_run_dir(args.output_root, args.fast)
    job = job_name(args.objective, args.beta, args.seed, args.lambda_sup, 2 if args.fast else args.epochs)
    folder = run_dir / job
    folder.mkdir(parents=True, exist_ok=False)
    config = {"experiment_note": EXPERIMENT_NOTE, "job": job, "objective": args.objective,
              "supervised_factor": OBJECTIVES[args.objective], "beta": args.beta,
              "deterministic_z": args.beta == 0, "seed": args.seed, "lambda_sup": args.lambda_sup,
              "sup_normalization": "mean_k CE / ln(C)", "latent_dim": LATENT_DIM,
              "hidden_dim": HIDDEN_DIM, "decoder_activation": "tanh", "initialization": "default",
              "optimizer": "adagrad", "lr": LR, "batch_size": BATCH_SIZE, "epochs": args.epochs,
              "mrl_prefixes": list(range(1, LATENT_DIM + 1)), "mrl_reduction": "mean_scalar_losses",
              "mrl_kl": "full_posterior_once", "split_seed": SPLIT_SEED,
              "split_fractions": list(SPLIT_FRACTIONS), "tf32": args.tf32, "fast": args.fast,
              "device": args.device, "active_threshold": ACTIVE_THRESHOLD}
    if args.fast:
        config["epochs"] = 2
    try:
        if args.run_dir is None:
            snapshot(run_dir)
        split, split_info = prepare_split(folder, SPLIT_SEED, SPLIT_FRACTIONS)
        if args.fast:
            split = {"train": subsample(split["train"], 8192, 0),
                     "validation": subsample(split["validation"], 2048, 1), "test": split["test"]}
        config.update(split=split_info, data=dataset_record(args.data_path),
                      training_images=int(len(split["train"])))
        write_json(folder / "config.json", config)
        write_json(folder / "environment.json", environment())
        (folder / "experiment_description.log").write_text(
            EXPERIMENT_NOTE + "\n\nResolved configuration:\n" + json.dumps(config, indent=2) + "\n")
        seed_everything(args.seed)
        images = torch.from_numpy(DSprites(args.data_path).images.reshape(-1, 4096)).to(device)
        model = model_from_config(config).to(device)
        factor = OBJECTIVES[args.objective]
        heads = PrefixHeads(LATENT_DIM, n_classes(factor)).to(device) if factor else None
        history = train(model, heads, images, split, config, device, folder)
        plot_training(history, folder, stamp)
        final = history[-1]["validation"]
        write_json(folder / "final_validation.json", final)
        (folder / "results_summary.txt").write_text(summary_text(config, final))
        write_json(folder / "status.json", {"status": "completed"})
    except Exception as error:
        write_json(folder / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        save_runtime(str(folder), time.perf_counter() - started)
    if not args.no_analysis:
        del images, model, heads
        torch.cuda.empty_cache()
        try:
            analyze_job(folder, load_context(args.data_path), device)
            write_json(folder / "analysis_status.json", {"status": "completed"})
        except Exception as error:
            write_json(folder / "analysis_status.json", {"status": "failed", "error": repr(error)})
            print(f"{job}: analysis failed: {error!r}; rerun scripts/analyze_pretrain.py", flush=True)


if __name__ == "__main__":
    main()
