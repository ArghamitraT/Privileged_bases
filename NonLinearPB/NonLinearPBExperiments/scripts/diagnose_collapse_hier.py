"""Diagnose a collapsed Experiment 1 run from its saved checkpoints (inference only, no training).

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/diagnose_collapse_hier.py --run-dir /path/to/exprmnt_x --objective mrl --seed 1 --reference-seed 0 --output /path/to/diagnosis

Context: 2026-11-02, MRL seed 1 trained normally to epoch 27, then epoch 28's
mean train loss jumped from 0.0003 to 0.034 and validation MSE became 0.072 at
every prefix for the remaining 72 epochs. 0.072 is roughly the error of
predicting one constant image, so the network stopped depending on its input.
This script copies every artefact needed to re-examine the collapse later into
--output, then compares the healthy checkpoint (best.pt), the collapsed one
(last.pt) and a healthy reference seed (both its checkpoints) on validation
images:
  1. weight norm of every parameter tensor, and its change best -> last;
  2. activation statistics at every encoder and decoder stage: mean, overall
     SD, SD across images (does the stage still depend on the input?), share of
     positive pre-GELU values (are units dead?), max |value|;
  3. the output: SD across images, distance to the validation mean image;
  4. MRL-loss gradient norm per parameter tensor on one validation batch.
Limits: weights at the moment of collapse (epochs 27-28) and Adam's state were
not saved; last.pt has had 72 more epochs of Adam updates after the collapse.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hier_exp1.data import file_sha256, load_split
from hier_exp1.model import objective_loss
from hier_exp1_train import build_model

# ==============================================================================
# CONFIG
# ==============================================================================
N_IMAGES = 1024              # validation images used for activations and gradients
GRAD_BATCH = 256             # one training-size batch for the gradient check
# ==============================================================================


def load(run_dir, objective, seed, which, device):
    model = build_model(seed)
    ckpt = torch.load(run_dir / "checkpoints" / objective / f"seed_{seed}" / f"{which}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["epoch"]


def stages(model):
    """(name, module, is_pre_gelu) for every stage whose output we record."""
    e, din, d = model.encoder, model.decoder_in, model.decoder
    return [("enc.conv1(pre-GELU)", e[0], True), ("enc.res32", e[2], False), ("enc.conv2(pre-GELU)", e[3], True),
            ("enc.res64", e[5], False), ("enc.conv3(pre-GELU)", e[6], True), ("enc.z", e[9], False),
            ("dec_in.linear(pre-GELU)", din[0], True), ("dec_in.gelu", din[1], False),
            ("dec.res128", d[0], False), ("dec.up1.conv(pre-GELU)", d[1][1], True), ("dec.res64", d[2], False),
            ("dec.up2.conv(pre-GELU)", d[3][1], True), ("dec.up3.conv(pre-GELU)", d[4][1], True),
            ("dec.out", d[5], False)]


@torch.no_grad()
def activations(model, x):
    out, hooks = {}, []
    for name, module, pre in stages(model):
        hooks.append(module.register_forward_hook(lambda m, i, o, name=name: out.__setitem__(name, o.detach().float())))
    model.decode(model.encode(x))
    for h in hooks:
        h.remove()
    rows = []
    for name, _, pre in stages(model):
        a = out[name].flatten(1)                                  # [images, features]
        rows.append({"stage": name, "mean": a.mean().item(), "sd_overall": a.std().item(),
                     "sd_across_images_mean": a.std(dim=0).mean().item(),
                     "frac_positive": (a > 0).float().mean().item() if pre else np.nan,
                     "frac_units_never_positive": ((a > 0).sum(dim=0) == 0).float().mean().item() if pre else np.nan,
                     "max_abs": a.abs().max().item()})
    return rows, out


def gradients(model, x):
    model.zero_grad(set_to_none=True)
    objective_loss(model, x, "mrl").backward()
    return {n: (p.grad.norm().item() if p.grad is not None else 0.0) for n, p in model.named_parameters()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--objective", default="mrl")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reference-seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False          # exact fp32 for diagnosis
    torch.backends.cudnn.allow_tf32 = False

    # 1. Preserve every artefact needed to re-examine this later.
    keep = {}
    for seed, role in ((args.seed, "collapsed"), (args.reference_seed, "reference")):
        tag = f"{args.objective}_seed{seed}"
        dst = out / "artefacts" / f"{role}_{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        src = args.run_dir / "checkpoints" / args.objective / f"seed_{seed}"
        for f in ("best.pt", "last.pt", "config.json", "summary.json"):
            shutil.copy2(src / f, dst / f)
        shutil.copy2(args.run_dir / "training" / f"{tag}.csv", dst / "training_history.csv")
        shutil.copy2(args.run_dir / "logs" / f"train_{tag}.log", dst / "train.log")
        keep[role] = {"tag": tag, "source": str(src), "sha256": {f: file_sha256(dst / f) for f in ("best.pt", "last.pt")}}
    shutil.copy2(args.run_dir / "config.json", out / "artefacts" / "run_config.json")
    shutil.copytree(args.run_dir / "code_snapshot", out / "artefacts" / "code_snapshot", dirs_exist_ok=True)

    images, _ = load_split("val")
    x = torch.as_tensor(images[:: max(1, len(images) // N_IMAGES)][:N_IMAGES], device=device).unsqueeze(1).float() / 255.0
    mean_image = x.mean(dim=0, keepdim=True)

    models = {}
    for seed, role in ((args.seed, "collapsed"), (args.reference_seed, "reference")):
        for which in ("best", "last"):
            models[f"{role}_{which}"] = load(args.run_dir, args.objective, seed, which, device)

    # 2. Weight norms and their change.
    weight_rows = []
    names = [n for n, _ in models["collapsed_best"][0].named_parameters()]
    params = {k: dict(m.named_parameters()) for k, (m, _) in models.items()}
    for n in names:
        row = {"parameter": n, "numel": params["collapsed_best"][n].numel()}
        for k in models:
            row[f"norm_{k}"] = params[k][n].norm().item()
        for role in ("collapsed", "reference"):
            b, l = params[f"{role}_best"][n], params[f"{role}_last"][n]
            row[f"rel_change_{role}"] = ((l - b).norm() / b.norm().clamp_min(1e-12)).item()
        weight_rows.append(row)

    # 3. Activations, 4. output dependence, 5. gradients.
    act_rows, out_rows, grad_rows = [], [], []
    grads = {}
    for k, (m, epoch) in models.items():
        rows, acts = activations(m, x)
        act_rows += [{"model": k, "epoch": epoch, **r} for r in rows]
        y = acts["dec.out"]
        with torch.no_grad():
            per_k = []
            masks = torch.tril(torch.ones(10, 10, device=device))
            z = m.encode(x)
            for kk in range(10):
                per_k.append(((m.decode(z * masks[kk]) - x) ** 2).mean().item())
        out_rows.append({"model": k, "epoch": epoch, "mse_k10": per_k[-1], "mse_k1": per_k[0],
                         "mse_constant_mean_image": ((mean_image - x) ** 2).mean().item(),
                         "output_sd_across_images": y.flatten(1).std(dim=0).mean().item(),
                         "output_mean": y.mean().item(),
                         "output_dist_to_mean_image": (y - mean_image).pow(2).mean().sqrt().item(),
                         "z_sd_across_images_per_coord": z.std(dim=0).tolist()})
        grads[k] = gradients(m, x[:GRAD_BATCH])
    for n in names:
        grad_rows.append({"parameter": n, **{f"grad_norm_{k}": grads[k][n] for k in models}})

    for name, rows in (("weight_norms.csv", weight_rows), ("activations.csv", act_rows),
                       ("outputs.csv", out_rows), ("gradient_norms.csv", grad_rows)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    hist = list(csv.DictReader(open(out / "artefacts" / f"collapsed_{keep['collapsed']['tag']}" / "training_history.csv")))
    window = [{k: hist[i][k] for k in ("epoch", "train_loss", "val_full_mse", "val_mse_k1")} for i in range(20, 32)]
    summary = {"collapsed": keep["collapsed"], "reference": keep["reference"],
               "epochs": {k: e for k, (_, e) in models.items()}, "n_val_images": len(x),
               "history_epochs_21_to_32": window,
               "limits": "weights/optimizer state at the collapse (epochs 27-28) were not saved; "
                         "last.pt has 72 further epochs of Adam updates; no per-step gradient logging existed",
               "outputs": out_rows}
    (out / "diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
