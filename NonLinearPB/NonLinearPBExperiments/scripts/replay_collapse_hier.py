"""Replay a recorded Experiment 1 collapse from a saved pre-spike state, with candidate optimizer fixes.

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/replay_collapse_hier.py --diag-dir /path/to/exprmnt_x/diagnostics/mrl_seed1 --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/scripts/replay_collapse_hier.py --diag-dir ... --variants baseline beta2_0.99 --steps 400

Loads the earliest state of the first spike dump (weights + Adam state, taken
before that step's update; hier_exp1/diagnostics.py), then trains MRL forward
on exactly the batches the original run used: the rest of that epoch from the
saved order, later epochs from the saved data-order generator state. Each
variant changes one optimizer setting from that same starting point. Output:
<diag-dir>/replay/<variant>.csv (loss and encoder grad/update norms per step)
and summary.json. GPU kernels are not bit-deterministic, so 'baseline' is run
twice to show how much the replay itself varies.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hier_exp1.data import load_split
from hier_exp1.model import prefix_mse
from hier_exp1_train import BATCH_SIZE, LR, Compiled, build_model, to_device

# ==============================================================================
# CONFIG: one change per variant, everything else as in the original run
# ==============================================================================
VARIANTS = {
    "baseline": {},
    "baseline_repeat": {},
    "beta2_0.99": {"betas": (0.9, 0.99)},
    "lr_3e-4": {"lr": 3e-4},
    "eps_1e-6": {"eps": 1e-6},
    "clip_1.0": {"clip": 1.0},
}
COLLAPSE_LOSS = 0.05          # batch loss of a dead network is ~0.072; healthy ~0.0004
# ==============================================================================


def batches(diag, start_step, n_steps, n_batches, device):
    """Batch indices for steps start_step.. using the saved orders and the saved generator."""
    first_epoch = start_step // n_batches + 1
    out, epoch = [], first_epoch
    order = torch.as_tensor(np.load(diag / "order" / f"epoch_{epoch:03d}.npy").astype(np.int64), device=device)
    gen = None
    step = start_step
    while len(out) < n_steps:
        b = step - (epoch - 1) * n_batches
        if b >= n_batches:
            epoch += 1
            saved = diag / "order" / f"epoch_{epoch:03d}.npy"
            if saved.exists():
                order = torch.as_tensor(np.load(saved).astype(np.int64), device=device)
            else:                                   # continue the run's own generator past the last saved epoch
                if gen is None:
                    gen = torch.Generator(device=device)
                    gen.set_state(torch.load(diag / "epochs" / f"epoch_{epoch - 1:03d}.pt", weights_only=True)["generator_state"])
                order = torch.randperm(order.numel(), device=device, generator=gen)      # as in training: randperm(len(train))
            continue
        out.append(order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE])
        step += 1
    return out


def restore(state, seed, device, variant):
    model = build_model(seed)
    model.load_state_dict(state["weights"])
    model = model.to(device)
    kw = {k: v for k, v in variant.items() if k in ("lr", "betas", "eps")}
    opt = torch.optim.Adam(model.parameters(), lr=kw.pop("lr", LR), **kw)
    params = dict(model.named_parameters())
    for n, s in state["adam"].items():
        if s:
            opt.state[params[n]] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()}
    return model, opt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None, help="default: from the folder name mrl_seed<N>")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    diag = args.diag_dir
    seed = args.seed if args.seed is not None else int(diag.name.rsplit("seed", 1)[1])
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True          # as in training
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out_dir = diag / "replay"
    out_dir.mkdir(exist_ok=True)

    spike = sorted(diag.glob("spike_step*"), key=lambda p: int(p.name.split("step")[1]))[0]
    pre = sorted(spike.glob("pre_*.pt"))[0]
    state = torch.load(pre, map_location="cpu", weights_only=True)
    start = state["step"]
    train_x = to_device(load_split("train")[0], device)
    n_batches = len(train_x) // BATCH_SIZE
    idx = batches(diag, start, args.steps, n_batches, device)
    assert torch.equal(idx[0].cpu(), state["batch_idx"].cpu()), "saved batch does not match the reconstructed order"
    print(f"replaying from step {start} ({pre.name}, spike {spike.name}) for {args.steps} steps", flush=True)

    summary = {"start_state": str(pre), "start_step": start, "steps": args.steps, "spike_dir": spike.name, "variants": {}}
    for name in args.variants:
        v = VARIANTS[name]
        t0 = time.time()
        model, opt = restore(state, seed, device, v)
        net = Compiled(model, True)
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
        rows = []
        for i, b in enumerate(idx):
            x = train_x[b].float() / 255.0
            before = [p.detach().clone() for p in enc]
            loss = prefix_mse(net, x).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.cat([p.grad.flatten() for p in enc]).norm()
            if "clip" in v:
                torch.nn.utils.clip_grad_norm_(model.parameters(), v["clip"])
            opt.step()
            unorm = torch.cat([(p.detach() - q).flatten() for p, q in zip(enc, before)]).norm()
            rows.append([start + i, *torch.stack([loss.detach(), gnorm, unorm]).tolist()])
        losses = np.array([r[1] for r in rows])
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "loss", "grad_norm_encoder", "update_norm_encoder"])
            w.writerows(rows)
        dead = np.flatnonzero(losses > COLLAPSE_LOSS)
        res = {"settings": {k: list(x) if isinstance(x, tuple) else x for k, x in v.items()},
               "max_loss": float(losses.max()), "max_loss_step": int(rows[int(losses.argmax())][0]),
               "mean_loss_last_50": float(losses[-50:].mean()),
               "first_step_above_collapse": int(rows[dead[0]][0]) if len(dead) else None,
               "collapsed_at_end": bool(losses[-50:].mean() > COLLAPSE_LOSS),
               "max_update_norm_encoder": float(max(r[3] for r in rows)), "seconds": round(time.time() - t0, 1)}
        summary["variants"][name] = res
        print(f"{name:16s} max loss {res['max_loss']:.4f} @ {res['max_loss_step']}  last-50 mean {res['mean_loss_last_50']:.5f}  "
              f"collapsed {res['collapsed_at_end']}  max update {res['max_update_norm_encoder']:.3f}  ({res['seconds']:.0f}s)", flush=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
