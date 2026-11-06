"""Training-collapse recorder for Experiment 1 (used by hier_exp1_train.py --diagnostics).

Conda environment: mrl_env_cuda12
Usage:
    rec = Recorder(model, optimizer, folder)        # then, every step:
    rec.before_step(step, epoch, batch_idx)         # snapshot of weights + Adam state (GPU ring buffer)
    rec.after_backward(); optimizer.step(); rec.after_step(loss, per_prefix, x)
    rec.end_epoch(epoch, generator, order)           # replayable checkpoint + data order

Written for the 2026-11-02 MRL seed 1 collapse (diagnosis in
exprmnt_2026_11_02__13_07_00/diagnosis_mrl_seed1_collapse/), where only epoch
means and best/last weights existed. Nothing here changes training: all
measurements are read-only, and activations come from a separate no-grad
eager pass over the same batch (no hooks, so the compiled graph is untouched).

Outputs in <folder>:
  steps.csv               every step: loss (+10 prefix losses for MRL); per part
                          (encoder / decoder_in / decoder): gradient norm and max
                          |entry| before Adam, update norm and max |entry| after,
                          weight norm, mean sqrt(Adam v); every ACT_EVERY steps:
                          activation sizes and dead-unit share
  epochs/epoch_XXX.pt     model + Adam state + data-order generator state (replayable);
                          only the last KEEP_LAST_EPOCHS and every KEEP_EVERY_EPOCHS-th are kept
  order/epoch_XXX.npy     the epoch's training-image order (batch b = order[b*256:(b+1)*256])
  spike_<step>/pre_XX.pt  on a loss spike: the RING states before it, and
  spike_<step>/post_XX.pt the POST states after it (each: weights, Adam state, batch indices)
With save_all=True (hier_exp1_train.py --save-all), additionally:
  epochs/epoch_000.pt     the initial state (before any update); no epoch checkpoint is pruned
  step_weights/upd_NNNNNN.pt  model weights (state_dict, no Adam) after NNNNNN updates, every step;
                          upd_{168 e} is the model at the end of epoch e (load: model.load_state_dict)
Each saved state is the state *before* that step's update, so replaying step s
from state s with its batch reproduces the update (up to cuDNN/compile nondeterminism).
"""

from collections import deque
import csv
from pathlib import Path

import numpy as np
import torch

# ==============================================================================
# CONFIG
# ==============================================================================
RING = 5                  # states saved before a spike (replay starts from the earliest); was 20
POST = 0                  # states saved after a spike; was 50 (forensics only, ~0.5 GB per dump)
KEEP_LAST_EPOCHS = 2      # epoch checkpoints kept on a rolling basis (enough to replay a collapse)
KEEP_EVERY_EPOCHS = 50    # plus every 50th epoch; was every epoch (1.9 GB per 200-epoch run)
ACT_EVERY = 10            # steps between activation measurements
SPIKE_FACTOR = 10.0       # batch loss > factor x median of the last SPIKE_WINDOW losses
SPIKE_WINDOW = 200
SPIKE_WARMUP = 500        # no spike detection during the first steps (loss falls fast there)
MAX_SPIKES = 3            # cap on spike dumps (disk)
SPIKE_COOLDOWN = 50       # steps after a dump before another can trigger (one event = one dump)
GROUPS = ("encoder", "decoder_in", "decoder")
# ==============================================================================


def _group(name):
    return name.split(".")[0]


class Recorder:
    def __init__(self, model, optimizer, folder, warmup=SPIKE_WARMUP, factor=SPIKE_FACTOR, save_all=False):
        self.model, self.opt = model, optimizer
        self.folder = Path(folder)
        self.save_all = save_all
        for sub in ("epochs", "order") + (("step_weights",) if save_all else ()):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        self.params = dict(model.named_parameters())
        self.ring = deque(maxlen=RING)
        self.losses = deque(maxlen=SPIKE_WINDOW)
        self.warmup, self.factor = warmup, factor
        self.post_left, self.post_dir, self.n_spikes = 0, None, 0
        self.next_dump_step = 0
        self.spike_steps = []
        self.fh = None
        self.writer = None

    # -- state capture -------------------------------------------------------
    def _state(self):
        adam = {}
        for n, p in self.params.items():
            s = self.opt.state.get(p, {})
            adam[n] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in s.items()}
        return {"weights": {n: p.detach().clone() for n, p in self.params.items()}, "adam": adam}

    @staticmethod
    def _to_cpu(state):
        return {"weights": {n: t.cpu() for n, t in state["weights"].items()},
                "adam": {n: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()}
                         for n, s in state["adam"].items()},
                **{k: v for k, v in state.items() if k not in ("weights", "adam")}}

    def before_step(self, step, epoch, batch_idx):
        self.cur = {"step": step, "epoch": epoch, **self._state(), "batch_idx": batch_idx.detach().clone()}
        self.ring.append(self.cur)
        if self.post_left > 0:
            torch.save(self._to_cpu(self.cur), self.post_dir / f"post_{POST - self.post_left:02d}_step{step}.pt")
            self.post_left -= 1

    # -- per-step numbers ----------------------------------------------------
    def after_backward(self):
        g = {k: [] for k in GROUPS}
        for n, p in self.params.items():
            if p.grad is not None:
                g[_group(n)].append(p.grad.detach().flatten())
        self.grad = {k: torch.cat(v) for k, v in g.items()}

    def after_step(self, loss, per_prefix, x):
        row_t = {"loss": loss.detach()}
        if per_prefix is not None:
            for k, v in enumerate(per_prefix.detach(), 1):
                row_t[f"loss_k{k}"] = v
        before = self.cur["weights"]
        for grp in GROUPS:
            names = [n for n in self.params if _group(n) == grp]
            gvec = self.grad[grp]
            upd = torch.cat([(self.params[n].detach() - before[n]).flatten() for n in names])
            w = torch.cat([self.params[n].detach().flatten() for n in names])
            v = [self.opt.state[self.params[n]]["exp_avg_sq"].flatten() for n in names if "exp_avg_sq" in self.opt.state[self.params[n]]]
            row_t[f"grad_norm_{grp}"] = gvec.norm()
            row_t[f"grad_max_{grp}"] = gvec.abs().max()
            row_t[f"update_norm_{grp}"] = upd.norm()
            row_t[f"update_max_{grp}"] = upd.abs().max()
            row_t[f"weight_norm_{grp}"] = w.norm()
            row_t[f"adam_sqrt_v_mean_{grp}"] = torch.cat(v).sqrt().mean() if v else torch.zeros((), device=w.device)
        if self.cur["step"] % ACT_EVERY == 0:
            row_t.update(self._activations(x))
        keys = list(row_t)
        values = torch.stack([row_t[k].float().reshape(()) for k in keys]).tolist()      # one GPU sync per step
        row = {"step": self.cur["step"], "epoch": self.cur["epoch"], **dict(zip(keys, values))}
        self._write(row)
        if self.save_all:                                  # weights after this update (for inference at any step)
            torch.save({n: t.detach().cpu() for n, t in self.model.state_dict().items()},
                       self.folder / "step_weights" / f"upd_{self.cur['step'] + 1:06d}.pt")
        self.last_row = row                                # read by the optional W&B mirror
        self._spike_check(row["loss"])

    @torch.no_grad()
    def _activations(self, x):
        """Stage-by-stage eager pass (no hooks), same weights and batch as the training step just taken."""
        m = self.model
        e = m.encoder
        h = x
        out = {}
        for i, layer in enumerate(e):
            h = layer(h)
            if i == 6:
                out["act_enc_conv3_absmax"] = h.abs().max()
        z = h
        out["act_z_absmax"] = z.abs().max()
        out["act_z_sd_mean"] = z.std(dim=0).mean()
        h = m.decoder_in(z).view(-1, 128, 8, 8)
        last_up = max(i for i, l in enumerate(m.decoder)            # 4 in the 64-px net, 3 in the 32-px net
                      if isinstance(l, torch.nn.Sequential) and isinstance(l[0], torch.nn.Upsample))
        for i, layer in enumerate(m.decoder):
            if i == last_up:                             # last upsampling block: Upsample, Conv, GELU
                pre = layer[1](layer[0](h))
                out["act_dec_up3_absmax"] = pre.abs().max()
                # unit = one (channel, position); dead = never positive on any image of the batch
                # (same definition as the seed 1 diagnosis: 0.74 healthy at epoch 24, 1.00 collapsed)
                out["act_dec_up3_frac_dead_units"] = ((pre.flatten(1) > 0).sum(dim=0) == 0).float().mean()
                h = layer[2](pre)
            else:
                h = layer(h)
        out["act_out_sd_across_images"] = h.flatten(1).std(dim=0).mean()
        return out

    def _write(self, row):
        if self.writer is None:
            self.fields = list(row) + [f for f in ("act_enc_conv3_absmax", "act_z_absmax", "act_z_sd_mean",
                                                    "act_dec_up3_absmax", "act_dec_up3_frac_dead_units",
                                                    "act_out_sd_across_images") if f not in row]
            self.fh = open(self.folder / "steps.csv", "w", newline="")
            self.writer = csv.DictWriter(self.fh, fieldnames=self.fields, restval="")
            self.writer.writeheader()
        self.writer.writerow(row)

    def _spike_check(self, loss):
        step = self.cur["step"]
        if (step >= self.warmup and len(self.losses) >= 20 and self.n_spikes < MAX_SPIKES
                and self.post_left == 0 and step >= self.next_dump_step and loss > self.factor * float(np.median(self.losses))):
            self.n_spikes += 1
            self.spike_steps.append(step)
            self.post_dir = self.folder / f"spike_step{step}"
            self.post_dir.mkdir(parents=True, exist_ok=True)
            for i, st in enumerate(self.ring):
                torch.save(self._to_cpu(st), self.post_dir / f"pre_{i:02d}_step{st['step']}.pt")
            (self.post_dir / "trigger.txt").write_text(
                f"step {step}: batch loss {loss:.6g} > {self.factor} x median {float(np.median(self.losses)):.6g} "
                f"of the last {len(self.losses)} steps\n")
            self.post_left = POST
            self.next_dump_step = step + SPIKE_COOLDOWN
            self.fh.flush()
            print(f"[diagnostics] loss spike at step {step} (loss {loss:.4g}); saved {len(self.ring)} prior states, "
                  f"saving the next {POST}", flush=True)
        self.losses.append(loss)

    # -- per epoch -------------------------------------------------------------
    def save_initial(self, generator):
        """Epoch 0: the state before any update (same layout as the epoch checkpoints)."""
        torch.save({"epoch": 0, "next_step": 0, "model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(), "generator_state": generator.get_state().cpu()},
                   self.folder / "epochs" / "epoch_000.pt")

    def end_epoch(self, epoch, step, generator, order):
        torch.save({"epoch": epoch, "next_step": step, "model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(), "generator_state": generator.get_state().cpu()},
                   self.folder / "epochs" / f"epoch_{epoch:03d}.pt")
        np.save(self.folder / "order" / f"epoch_{epoch:03d}.npy", order.cpu().numpy().astype(np.int32))
        old = epoch - KEEP_LAST_EPOCHS                     # prune: keep the last few and every KEEP_EVERY_EPOCHS-th
        if not self.save_all and old >= 1 and old % KEEP_EVERY_EPOCHS != 0:
            (self.folder / "epochs" / f"epoch_{old:03d}.pt").unlink(missing_ok=True)
        if self.fh:
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
