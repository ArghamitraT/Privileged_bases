"""Optional Weights & Biases live view for Experiment 1 training (hier_exp1_train.py --wandb).

Conda environment: mrl_env_cuda12
Usage:
    wb = WandbLogger(project, run_name, group, config, folder, model, val_x, val_mid_node)   # files go to <folder>/wandb/
    wb.log_step(row, spike_steps)     # with --diagnostics: one steps.csv row per training step
    wb.log_epoch(row, epoch, lr, epoch_seconds)
    wb.finish(status, exit_code, **summary)

Logging only. Step numbers are the Recorder's row (already synced to CPU);
epoch extras (latent spread, reconstruction grid) come from separate no-grad
eager passes on validation images, with no randomness, so training's weights,
optimizer state and batch order are untouched. The CSV files in the run folder
stay the record; W&B is a mirror for watching. Login comes from ~/.netrc
(`wandb login`); no key is stored here or in the run folder.

Charts:  step/*   against global_step  (batch loss, per-prefix loss, gradient,
                                        update, relative update, weight norm, Adam
                                        sqrt(v), activation sizes, dead units)
         epoch/*  against epoch        (the training CSV row, latent SD per
                                        coordinate, lr, seconds/epoch, peak GPU memory)
         recon/grid every grid_every epochs (default GRID_EVERY): rows = one validation image per mid
                    node (9), columns = input, then decodes at GRID_KS
Alerts (W&B email / app, per your W&B notification settings): a crash, an early collapse, and the end of the run.
"""

import numpy as np
import torch

# ==============================================================================
# CONFIG
# ==============================================================================
GRID_EVERY = 5                    # epochs between reconstruction grids (also epoch 1)
GRID_KS = (1, 2, 3, 5, 10)        # prefixes shown in the grid
GRID_SCALE = 2                    # pixel upscaling of the grid image
SPREAD_CHUNK = 512                # validation images per encoder pass for the latent spread
STEP_PREFIXES = ("loss", "grad_norm_", "grad_max_", "update_norm_", "update_max_", "weight_norm_",
                 "adam_sqrt_v_mean_", "act_")
GROUPS = ("encoder", "decoder_in", "decoder")
# ==============================================================================


class WandbLogger:
    def __init__(self, project, run_name, group, config, folder, model, val_x, val_mid_node, tags=(),
                 grid_every=GRID_EVERY):
        self.run, self.model = None, model
        self.pair = config.get("objective") == "pair"     # grid shows {z1}, {z1, zj} codes instead of prefixes
        self.grid_every = grid_every
        self.n_spikes_seen = 0
        # one validation image per mid node (first of each), fixed for the whole run
        picks = [int(np.flatnonzero(val_mid_node == m)[0]) for m in range(9) if (val_mid_node == m).any()]
        self.grid_x = val_x[picks].float() / 255.0
        self.val_x = val_x
        try:
            import wandb
            self.wandb = wandb
            folder.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(project=project, name=run_name, group=group, config=config, dir=str(folder),
                                  tags=list(tags), settings=wandb.Settings(init_timeout=120, console="off"))
            self.run.define_metric("global_step")
            self.run.define_metric("step/*", step_metric="global_step")
            self.run.define_metric("epoch")
            self.run.define_metric("epoch/*", step_metric="epoch")
            self.run.define_metric("recon/*", step_metric="epoch")
            print(f"[wandb] {self.run.url}", flush=True)
        except Exception as exc:                 # never let the live view stop a training run
            print(f"[wandb] disabled ({type(exc).__name__}: {exc}); CSV logging continues", flush=True)
            self.run = None

    @property
    def active(self):
        return self.run is not None

    def _log(self, data):
        if self.run is None:
            return
        try:
            self.run.log(data)
        except Exception as exc:
            print(f"[wandb] log failed ({exc}); disabling W&B for the rest of the run", flush=True)
            self.run = None

    def _alert(self, title, text, level="WARN"):
        if self.run is None:
            return
        try:
            self.run.alert(title=title, text=text, level=getattr(self.wandb.AlertLevel, level))
        except Exception as exc:
            print(f"[wandb] alert failed ({exc})", flush=True)

    # -- per step ----------------------------------------------------------------
    def log_step(self, row, spike_steps=()):
        if self.run is None:
            return
        data = {"global_step": row["step"]}
        for k, v in row.items():
            if v == "" or v is None or k in ("step", "epoch"):
                continue
            if k.startswith(STEP_PREFIXES):
                data[f"step/{k}"] = v
        for g in GROUPS:                         # update size relative to the weights: the runaway signal
            u, w = row.get(f"update_norm_{g}"), row.get(f"weight_norm_{g}")
            if u not in (None, "") and w not in (None, "") and w > 0:
                data[f"step/rel_update_{g}"] = u / w
        self._log(data)
        if len(spike_steps) > self.n_spikes_seen:     # chart marker only; no alert (user: no emails for spikes)
            self.n_spikes_seen = len(spike_steps)
            self._log({"global_step": row["step"], "step/spike_recorded": self.n_spikes_seen})

    # -- per epoch ---------------------------------------------------------------
    @torch.no_grad()
    def _latent_sd(self):
        was_training = self.model.training
        self.model.eval()
        z = torch.cat([self.model.encode(self.val_x[s:s + SPREAD_CHUNK].float() / 255.0)
                       for s in range(0, len(self.val_x), SPREAD_CHUNK)])
        self.model.train(was_training)
        return z.std(dim=0).tolist(), float(z.abs().max())

    def _grid_ks(self):                  # GRID_KS cut to the latent size, always ending at the full code
        d = self.model.latent_dim
        if self.pair:
            return list(range(1, d + 1))
        return sorted({k for k in GRID_KS if k < d} | {d})

    @torch.no_grad()
    def _grid(self):
        was_training = self.model.training
        self.model.eval()
        x = self.grid_x
        z = self.model.encode(x)
        cols = [x]
        for k in self._grid_ks():
            if self.pair:                    # k = 1: z1 only; k = j: z1 and zj
                keep = torch.zeros_like(z[:1]); keep[:, 0] = 1; keep[:, k - 1] = 1
                zk = z * keep
            else:
                zk = torch.cat([z[:, :k], torch.zeros_like(z[:, k:])], dim=1)
            cols.append(self.model.decode(zk).clamp(0, 1))
        self.model.train(was_training)
        tiles = torch.stack(cols, dim=1)                               # [9 rows, cols, C, H, W]
        tiles = torch.nn.functional.pad(tiles, (1, 1, 1, 1), value=0.5)  # grey frame between tiles
        r, c, ch, h, w = tiles.shape
        img = tiles.permute(0, 3, 1, 4, 2).reshape(r * h, c * w, ch)
        img = img.repeat_interleave(GRID_SCALE, 0).repeat_interleave(GRID_SCALE, 1)
        img = (img.cpu().numpy() * 255).astype(np.uint8)
        return img[..., 0] if ch == 1 else img                         # grey (synthetic) or RGB (CIFAR)

    def log_epoch(self, row, epoch, lr, epoch_seconds, device, latent=None):
        if self.run is None:
            return
        data = {"epoch": epoch, **{f"epoch/{k}": v for k, v in row.items() if k != "epoch"},
                "epoch/lr": lr, "epoch/epoch_seconds": epoch_seconds}
        if device.type == "cuda":
            data["epoch/peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        try:
            if latent is None:
                sd, zmax = self._latent_sd()
                data.update({f"epoch/z_sd_coord{i + 1}": v for i, v in enumerate(sd)})
                data["epoch/z_absmax"] = zmax
            else:                    # per-coordinate range of the clean val code (trainer's latent_stats)
                for stat, values in latent.items():
                    key = {"frac_abs_gt_0.99": "z_saturated"}.get(stat, f"z_{stat}")
                    data.update({f"epoch/{key}_coord{i + 1}": v for i, v in enumerate(values)})
            if epoch == 1 or epoch % self.grid_every == 0:
                cap = ("rows: one val image per mid node (CIFAR: per superclass 0-8); columns: input, then codes {z1}, " + ", ".join(f"{{z1,z{j}}}" for j in self._grid_ks()[1:])
                       if self.pair else "rows: one val image per mid node (CIFAR: per superclass 0-8); columns: input, then k = " + ", ".join(map(str, self._grid_ks())))
                data["recon/grid"] = self.wandb.Image(self._grid(), caption=f"epoch {epoch}. {cap}")
        except Exception as exc:
            print(f"[wandb] epoch extras failed ({exc})", flush=True)
        self._log(data)

    # -- end ---------------------------------------------------------------------
    def finish(self, status, exit_code=0, **summary):
        if self.run is None:
            return
        try:
            self.run.summary.update({"status": status, **summary})
            level = "INFO" if status == "finished" else "ERROR"
            text = ", ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}"
                             for k, v in summary.items() if not isinstance(v, (list, dict)))
            self._alert(f"{self.run.name}: {status}", text or status, level)
            self.run.finish(exit_code=exit_code)
            self.wandb.teardown(exit_code=int(exit_code))   # shut the W&B service now, not in an atexit hook
        except Exception:
            pass
        self.run = None
