"""Train one Experiment 1 autoencoder: objective 'ae' (full-width MSE) or 'mrl' (mean prefix MSE), one seed.

Conda environment: mrl_env_cuda12
Usage (from code/; normally queued by scripts/run_hier_exp1.py):
    python NonLinearPB/NonLinearPBExperiments/experiments/hier_exp1_train.py --objective mrl --seed 0 --run-dir /path/to/exprmnt_x --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/hier_exp1_train.py --objective ae --seed 0 --run-dir /tmp/x --device cuda:0 --fast

Spec: NonLinearPB/plans/hierarchical_Exp1.md sections 1-4. The model is built
right after seeding, on CPU, so 'ae' and 'mrl' with the same seed start from
identical weights (and the 'random' control in evaluation is that same init).
Batch order comes from a generator seeded by the seed, so it also matches
across objectives. No labels are used. Every epoch logs train loss and
validation full-width and per-prefix MSE for both objectives; the checkpoint
with the best validation value of the run's own objective is kept as best.pt.
Data stay on the GPU as uint8; torch.compile on the encoder/decoder (about 2x
faster here; numerics are the same ops). --fast: 2 epochs on every 21st train image
(2,058, all 27 leaves).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_exp1 import cifar
from hier_exp1.data import dataset_record, load_split
from hier_exp1.model import CNNAE, LATENT_DIM, objective_loss, pair_mse, prefix_mse
from hier_exp1.diagnostics import Recorder
from hier_exp1.wandb_log import GRID_EVERY, WandbLogger

# ==============================================================================
# CONFIG — plan section 4; --fast overrides inside main().
# ==============================================================================
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
WANDB_PROJECT = "hier_exp1"   # W&B project for --wandb
COLLAPSE_MSE = 0.075       # all-black output scores ~0.080 on val; healthy runs are ~0.004 by epoch 2
COLLAPSE_EPOCH = 2
LATE_COLLAPSE_MSE = 0.05   # --diagnostics: stop when val full-width MSE jumps above this (healthy ~0.0003; dead 0.072);
                           # only after the run has been below it once (a slow, still-learning run is not a crash)
COLLAPSE_REF_LR = 3e-4     # COLLAPSE_EPOCH was set at this lr; slower runs get proportionally more epochs
VAL_CHUNK = 64             # images per validation pass (x 10 prefixes); 1024 peaked at 16.8 GiB, 64 at 2.4
# ==============================================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DATASETS = {"hier": {"load": load_split, "record": dataset_record, "channels": 1, "size": 64},
            "cifar100": {"load": cifar.load_split, "record": cifar.dataset_record, "channels": 3, "size": 32}}


def build_model(seed, bounded=False, latent_dim=LATENT_DIM, dataset="hier"):
    """Same seed -> same initial weights for every objective (and for the random control);
    bounded (tanh on z) adds no parameters, so it does not change them either."""
    seed_everything(seed)
    ds = DATASETS[dataset]
    return CNNAE(latent_dim, bounded=bounded, channels=ds["channels"], size=ds["size"])


class Compiled:
    """encode/decode through torch.compile'd submodules; parameters stay in the plain model."""

    def __init__(self, model, enabled):
        self.latent_dim = model.latent_dim
        wrap = torch.compile if enabled else (lambda m: m)
        self._enc, self._din, self._dec = wrap(model.encoder), wrap(model.decoder_in), wrap(model.decoder)

    def encode(self, x):
        return self._enc(x)

    def decode(self, z):
        return self._dec(self._din(z).view(-1, 128, 8, 8))


def to_device(images, device):
    x = torch.as_tensor(images, device=device)
    return x.unsqueeze(1) if x.ndim == 3 else x                      # uint8 [N, 1, 64, 64] or [N, 3, 32, 32]


@torch.no_grad()
def validate_pairs(net, images):
    """Per-code mean MSE for {z1}, {z1, z2}, ..., {z1, zd} over all images: list of d."""
    total = torch.zeros(net.latent_dim, device=images.device, dtype=torch.float64)
    for s in range(0, len(images), VAL_CHUNK):
        x = images[s:s + VAL_CHUNK].float() / 255.0
        total += pair_mse(net, x).sum(dim=1).double()
    return (total / len(images)).tolist()


@torch.no_grad()
def validate(net, images, noise_sigma=0.0):
    """Per-prefix mean MSE over all images: list of 10 (k = 1..10). noise_sigma > 0: the training noise."""
    total = torch.zeros(net.latent_dim, device=images.device, dtype=torch.float64)
    for s in range(0, len(images), VAL_CHUNK):
        x = images[s:s + VAL_CHUNK].float() / 255.0
        total += prefix_mse(net, x, noise_sigma=noise_sigma).sum(dim=1).double()
    return (total / len(images)).tolist()


LATENT_STATS = ("min", "q01", "median", "q99", "max", "sd", "frac_abs_gt_0.99")


@torch.no_grad()
def latent_stats(net, images):
    """Per-coordinate range of the clean validation code z: {stat: list of 10}."""
    z = torch.cat([net.encode(images[s:s + VAL_CHUNK * 8].float() / 255.0) for s in range(0, len(images), VAL_CHUNK * 8)]).float()
    q = torch.quantile(z, torch.tensor([0.01, 0.5, 0.99], device=z.device), dim=0)
    return {"min": z.min(0).values.tolist(), "q01": q[0].tolist(), "median": q[1].tolist(), "q99": q[2].tolist(),
            "max": z.max(0).values.tolist(), "sd": z.std(0).tolist(), "frac_abs_gt_0.99": (z.abs() > 0.99).float().mean(0).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objective", choices=("ae", "mrl", "pair"), required=True,
                    help="pair: mean MSE over codes {z1}, {z1, z2}, ..., {z1, zd} (full code never used in training)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR, help=f"Adam learning rate (default {LR:g}, the plan's)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--diagnostics", action="store_true",
                    help="record per-step numbers, replayable snapshots and loss spikes (hier_exp1/diagnostics.py); "
                         "stop when validation collapses. Training itself is unchanged.")
    ap.add_argument("--wandb", action="store_true", help="mirror the epoch log (and, with --diagnostics, the step log) "
                    "to Weights & Biases (hier_exp1/wandb_log.py); logging only")
    ap.add_argument("--wandb-project", default=WANDB_PROJECT)
    ap.add_argument("--bounded", action="store_true", help="tanh on the encoder output: z in [-1, 1]")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="hier",
                    help="hier: synthetic hierarchical_images_v2 (64x64 grey); cifar100: 32x32 colour (hier_exp1/cifar.py)")
    ap.add_argument("--latent-dim", type=int, default=LATENT_DIM, help=f"size of z (default {LATENT_DIM})")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="training only: Gaussian noise SD added to the kept code entries before decoding")
    ap.add_argument("--save-all", action="store_true",
                    help="keep everything (implies --diagnostics): every epoch checkpoint incl. epoch 0, model weights "
                         "after every update, the clean test-set code every epoch, and the W&B recon grid every epoch")
    ap.add_argument("--epoch-weights", action="store_true",
                    help="save model weights (state_dict only, ~3.3 MB) at the init and after every epoch: "
                         "checkpoints/<objective>/seed_<s>/epochs/epoch_XXX.pt")
    ap.add_argument("--grid-every", type=int, default=None,
                    help=f"W&B recon/grid every N epochs (default {GRID_EVERY}; --save-all sets 1)")
    ap.add_argument("--diag-warmup", type=int, default=None, help="testing only: spike-detection warmup steps")
    ap.add_argument("--diag-spike-factor", type=float, default=None, help="testing only: spike threshold factor")
    args = ap.parse_args()
    if args.save_all:
        args.diagnostics = True
    start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    epochs = 2 if args.fast else args.epochs

    name = f"{args.objective}_seed{args.seed}"
    ckpt_dir = args.run_dir / "checkpoints" / args.objective / f"seed_{args.seed}"
    hist_dir = args.run_dir / "training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    ds = DATASETS[args.dataset]
    train_np, _ = ds["load"]("train")
    val_np, val_labels = ds["load"]("val")
    val_mid = val_labels["mid_node"]
    if args.fast:            # strided, not a prefix: the splits are stored sorted by leaf
        train_np, val_np, val_mid = train_np[::21], val_np[::10], val_mid[::10]
    train_x, val_x = to_device(train_np, device), to_device(val_np, device)
    test_x = to_device(ds["load"]("test")[0], device) if args.save_all else None

    model = build_model(args.seed, args.bounded, args.latent_dim, args.dataset).to(device)
    net = Compiled(model, enabled=not args.no_compile)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    n_batches = len(train_x) // BATCH_SIZE            # drop_last; the remainder rotates every epoch

    config = {"name": name, "objective": args.objective, "seed": args.seed, "latent_dim": args.latent_dim,
              "epochs": epochs, "batch_size": BATCH_SIZE, "optimizer": "Adam", "lr": args.lr,
              "batches_per_epoch": n_batches, "compile": not args.no_compile, "tf32": True,
              "checkpoint_rule": "best clean (noise-free) validation own objective (ae: k=10 MSE; mrl: mean prefix MSE; pair: mean pair MSE)",
              "bounded": args.bounded, "noise_sigma": args.noise_sigma,
              "dataset": args.dataset, "data": ds["record"](), "fast": args.fast, "device": str(device),
              "torch": str(torch.__version__), "diagnostics": args.diagnostics, "save_all": args.save_all, "epoch_weights": args.epoch_weights, "wandb": args.wandb, "command": sys.argv}
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    fields = ["epoch", "train_loss", "val_own", "val_full_mse", "val_mean_prefix_mse"] + \
             [f"val_mse_k{k}" for k in range(1, args.latent_dim + 1)] + \
             ([f"val_pair_mse_1"] + [f"val_pair_mse_1_{j}" for j in range(2, args.latent_dim + 1)] if args.objective == "pair" else []) + \
             (["val_noisy_own", "val_noisy_full_mse", "val_noisy_mean_prefix_mse"] if args.noise_sigma > 0 else []) + ["seconds"]
    lat_fh = open(hist_dir / f"latent_{name}.csv", "w", newline="")
    lat_writer = csv.writer(lat_fh)
    lat_writer.writerow(["epoch", "coord", *LATENT_STATS])
    best, best_epoch = float("inf"), -1
    collapse_epoch = max(COLLAPSE_EPOCH, round(COLLAPSE_EPOCH * COLLAPSE_REF_LR / args.lr))
    collapse_mse, late_collapse_mse = COLLAPSE_MSE, LATE_COLLAPSE_MSE
    if args.dataset != "hier":       # natural images: a constant output scores the mean-image MSE; use that as both guards
        vf = val_x.float() / 255.0
        collapse_mse = late_collapse_mse = float(((vf - vf.mean(0, keepdim=True)) ** 2).mean())
        del vf
    config["collapse_guard_mse"] = [collapse_mse, late_collapse_mse]
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    was_healthy = False                                  # val full-width MSE has been below LATE_COLLAPSE_MSE
    rec = None
    if args.diagnostics:
        kw = {k: v for k, v in (("warmup", args.diag_warmup), ("factor", args.diag_spike_factor)) if v is not None}
        rec = Recorder(model, optimizer, args.run_dir / "diagnostics" / name, save_all=args.save_all, **kw)
    codes_dir = args.run_dir / "diagnostics" / name / "test_codes"

    def save_codes(ep):                     # clean code of every test image (split order), [5400, 10] float32
        if test_x is None:
            return
        codes_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        with torch.no_grad():
            z = torch.cat([net.encode(test_x[s:s + VAL_CHUNK * 8].float() / 255.0)
                           for s in range(0, len(test_x), VAL_CHUNK * 8)]).float()
        np.save(codes_dir / f"epoch_{ep:03d}.npy", z.cpu().numpy())

    wb = None
    if args.wandb:
        variant = (f"_{args.dataset}" if args.dataset != "hier" else "") + (f"_z{args.latent_dim}" if args.latent_dim != LATENT_DIM else "") + (f"_tanh" if args.bounded else "") + (f"_noise{args.noise_sigma:g}" if args.noise_sigma > 0 else "")
        group = args.run_dir.name if not args.run_dir.name.startswith("sigma_") else f"{args.run_dir.parent.name}/{args.run_dir.name}"
        wb = WandbLogger(args.wandb_project, f"{name}{variant}_lr{args.lr:g}_ep{epochs}", group, config,
                         args.run_dir, model, val_x, val_mid,
                         tags=[args.objective, f"lr{args.lr:g}"] + (["tanh"] if args.bounded else []) +
                              ([f"noise{args.noise_sigma:g}"] if args.noise_sigma > 0 else []) +
                              (["save_all"] if args.save_all else []) +
                              ([f"z{args.latent_dim}"] if args.latent_dim != LATENT_DIM else []) +
                              ([args.dataset] if args.dataset != "hier" else []),
                         grid_every=args.grid_every or (1 if args.save_all else GRID_EVERY))
    step = 0
    wdir = ckpt_dir / "epochs"
    if args.epoch_weights:
        wdir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), wdir / "epoch_000.pt")
    if args.save_all:
        rec.save_initial(generator)
        save_codes(0)
    with open(hist_dir / f"{name}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            order = torch.randperm(len(train_x), device=device, generator=generator)
            running = torch.zeros((), device=device)
            for b in range(n_batches):
                idx = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                x = train_x[idx].float() / 255.0
                per_prefix = None
                if rec is not None:
                    rec.before_step(step, epoch, idx)
                if rec is not None and args.objective == "mrl":
                    per_kb = prefix_mse(net, x, noise_sigma=args.noise_sigma)   # [10, B]
                    loss = per_kb.mean()                              # exactly objective_loss("mrl")
                    per_prefix = per_kb.detach().mean(dim=1)          # logging only
                else:
                    loss = objective_loss(net, x, args.objective, noise_sigma=args.noise_sigma)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if rec is not None:
                    rec.after_backward()
                optimizer.step()
                if rec is not None:
                    rec.after_step(loss, per_prefix, x)
                    if wb is not None:
                        wb.log_step(rec.last_row, rec.spike_steps)
                running += loss.detach()
                step += 1
            model.eval()
            per_k = validate(net, val_x)                                   # clean: selection and reporting
            pairs = validate_pairs(net, val_x) if args.objective == "pair" else None
            own = per_k[-1] if args.objective == "ae" else float(np.mean(pairs if pairs is not None else per_k))
            guard = own if args.objective == "pair" else per_k[-1]       # collapse guard: a quantity the model is trained on
            noisy = {}
            if args.noise_sigma > 0:                                       # the training objective, for the curve only
                pk = validate(net, val_x, args.noise_sigma)
                noisy = {"val_noisy_own": pk[-1] if args.objective == "ae" else float(np.mean(pk)),
                         "val_noisy_full_mse": pk[-1], "val_noisy_mean_prefix_mse": float(np.mean(pk))}
            lat = latent_stats(net, val_x)
            for i in range(args.latent_dim):
                lat_writer.writerow([epoch, i + 1, *(lat[k][i] for k in LATENT_STATS)])
            lat_fh.flush()
            row = {"epoch": epoch, "train_loss": float(running) / n_batches, "val_own": own,
                   "val_full_mse": per_k[-1], "val_mean_prefix_mse": float(np.mean(per_k)),
                   **{f"val_mse_k{k}": v for k, v in enumerate(per_k, 1)},
                   **({"val_pair_mse_1": pairs[0], **{f"val_pair_mse_1_{j}": v for j, v in enumerate(pairs[1:], 2)}} if pairs else {}),
                   **noisy, "seconds": round(time.time() - start, 1)}
            writer.writerow(row)
            fh.flush()
            if wb is not None:
                wb.log_epoch(row, epoch, optimizer.param_groups[0]["lr"], round(time.time() - epoch_start, 1), device, lat)
            if args.epoch_weights:
                torch.save(model.state_dict(), wdir / f"epoch_{epoch:03d}.pt")
            if own < best:
                best, best_epoch = own, epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": config}, ckpt_dir / "best.pt")
            print(f"{name} epoch {epoch}/{epochs} train {row['train_loss']:.6f} val own {own:.6f} "
                  f"full {per_k[-1]:.6f} k1 {per_k[0]:.6f} ({row['seconds']:.0f}s)", flush=True)
            if rec is not None:
                rec.end_epoch(epoch, step, generator, order)
                save_codes(epoch)
                if guard > late_collapse_mse and was_healthy:
                    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": config}, ckpt_dir / "last.pt")
                    report = {"collapsed_at_epoch": epoch, "val_full_mse": per_k[-1], "val_mse_per_k": per_k,
                              "spike_steps": rec.spike_steps, "steps_done": step, "seconds": round(time.time() - start, 1)}
                    (args.run_dir / "diagnostics" / name / "crash.json").write_text(json.dumps(report, indent=2) + "\n")
                    rec.close()
                    if wb is not None:
                        wb.finish("crashed", exit_code=3, **report)
                    print(f"{name}: CRASH detected at epoch {epoch} (val guard MSE {guard:.4f} > {late_collapse_mse:.4f}); "
                          f"spikes at steps {rec.spike_steps}; stopping", flush=True)
                    raise SystemExit(3)
            was_healthy = was_healthy or guard < late_collapse_mse
            if epoch == collapse_epoch and guard > collapse_mse and not args.fast:
                if wb is not None:
                    wb.finish("collapsed_early", exit_code=1, epoch=epoch, val_guard_mse=guard)
                sys.exit(f"{name}: collapsed (val guard MSE {guard:.4f} > {collapse_mse:.4f} after epoch {epoch}; "
                         "a constant output scores about that)")
    torch.save({"state_dict": model.state_dict(), "epoch": epochs, "config": config}, ckpt_dir / "last.pt")
    lat_fh.close()
    if rec is not None:
        rec.close()
    summary = {"best_epoch": best_epoch, "best_val_own": best, "epochs": epochs,
               "converged_note": "best epoch == last epoch suggests training had not converged" if best_epoch == epochs else "",
               "seconds": round(time.time() - start, 1)}
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wb is not None:
        wb.finish("finished", **summary)
    print(f"{name} done: best epoch {best_epoch}, val own {best:.6f}, {summary['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
