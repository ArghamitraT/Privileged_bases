"""Epoch-slider page for the CIFAR-100 run: reconstructions of a few test images by AE, MRL-prefix and MRL-pair.

Conda environment: mrl_env_cuda12
For each model and epoch (0 = init .. 25, from checkpoints/<model>/seed_<s>/epochs/epoch_XXX.pt) decodes the test images with
    ae, mrl : prefix codes  z1, z1:2, ..., z1:d   (rest of the code zero; for the AE only z1:d was trained)
    pair    : pair codes    {z1}, {z1, z2}, ..., {z1, zd}   (the codes it was trained on; the full code was not)
Clipped to [0, 1]. One WebP sheet per (model, epoch): rows = images, columns = input then the d codes (32 px tiles, 2 px gap).
Also per tile: MSE x1e-3 of that image; per column: the whole-val-set MSE of that code from training/<model>_seed<s>.csv.
Writes <run>/figures/recon_epochs/recon_epochs.html (+ .json with the tile MSEs and image indices).

Usage: python scripts/cifar_recon_epochs_html.py [--run R] [--seed 0] [--images 34 27 37 35]
"""
import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hier_exp1.cifar import dataset_config, load_split
from hier_exp1_train import build_model

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_05__20_43_41"))
_ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--models", nargs="+", default=["ae", "mrl", "pair"])
_ap.add_argument("--images", type=int, nargs="+", default=[34, 27, 37, 35],
                 help="test indices (default: first test image of superclasses trees, vehicles_1, people, fish)")
_ap.add_argument("--epochs", type=int, default=25)
_args = _ap.parse_args()
OUT = _args.run / "figures" / "recon_epochs"
GAP, D = 2, 10
NOTES = {"ae": "trained on the full code z1:10 only; shorter prefixes are zero-filled codes it never saw",
         "mrl": "trained on every prefix z1, z1:2, …, z1:10",
         "pair": "trained on {z1} and each {z1, zj}; codes shown are those pairs (full code never trained)"}


def codes(z, kind):
    """z [n, d] -> [n, d, d]: row k is the k-th code."""
    n, d = z.shape
    m = torch.zeros(d, d, device=z.device)
    if kind == "pair":
        m[:, 0] = 1
        m[torch.arange(1, d), torch.arange(1, d)] = 1
    else:
        m = torch.tril(torch.ones(d, d, device=z.device))
    return z[:, None] * m[None]


def sheet(x, rec):
    """x [n, 3, 32, 32], rec [n, d, 3, 32, 32] in [0, 1] -> uint8 [H, W, 3]."""
    tiles = np.concatenate([x[:, None], rec], 1)                       # [n, d+1, 3, 32, 32]
    n, c = tiles.shape[:2]
    out = np.full((n * (32 + GAP) - GAP, c * (32 + GAP) - GAP, 3), 0.55, np.float32)
    for i in range(n):
        for j in range(c):
            out[i * (32 + GAP):i * (32 + GAP) + 32, j * (32 + GAP):j * (32 + GAP) + 32] = tiles[i, j].transpose(1, 2, 0)
    return (out * 255).round().astype(np.uint8)


def webp(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "WEBP", lossless=True, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dataset_config()
    test_x, lab = load_split("test")
    xs = test_x[_args.images].astype(np.float32) / 255
    names = [f"{cfg['fine_names'][lab['fine'][i]]} ({cfg['coarse_names'][lab['coarse'][i]].replace('_', ' ')})" for i in _args.images]
    xt = torch.as_tensor(xs).cuda()
    EP = list(range(0, _args.epochs + 1))
    payload = {"epochs": EP, "images": [{"index": i, "name": n} for i, n in zip(_args.images, names)], "models": []}
    for m in _args.models:
        sheets, mse = [], []
        ck = _args.run / "checkpoints" / m / f"seed_{_args.seed}" / "epochs"
        for ep in EP:
            model = build_model(_args.seed, dataset="cifar100")
            model.load_state_dict(torch.load(ck / f"epoch_{ep:03d}.pt", map_location="cpu", weights_only=True))
            model = model.cuda().eval()
            with torch.no_grad():
                z = model.encode(xt)
                rec = model.decode(codes(z, "pair" if m == "pair" else "prefix").reshape(-1, D)).view(len(xs), D, 3, 32, 32)
            r = rec.clamp(0, 1).cpu().numpy()
            sheets.append(webp(sheet(xs, r)))
            mse.append((((r - xs[:, None]) ** 2).mean(axis=(2, 3, 4)) * 1e3).round(2).tolist())   # [img][code]
        val = {}
        with open(_args.run / "training" / f"{m}_seed{_args.seed}.csv") as fh:
            for row in csv.DictReader(fh):
                keys = ["val_pair_mse_1"] + [f"val_pair_mse_1_{j}" for j in range(2, D + 1)] if m == "pair" else \
                       [f"val_mse_k{k}" for k in range(1, D + 1)]
                val[int(row["epoch"])] = [round(float(row[k]) * 1e3, 2) for k in keys]
        labels = ["z1"] + ([f"z1,z{j}" for j in range(2, D + 1)] if m == "pair" else [f"z1:{k}" for k in range(2, D + 1)])
        payload["models"].append({"key": m, "note": NOTES[m], "labels": labels, "sheets": sheets, "mse": mse,
                                  "val": [val.get(ep) for ep in EP]})
        print(m, "done", flush=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(payload)).replace("__RUN__", _args.run.name)
    (OUT / "recon_epochs.html").write_text(html)
    (OUT / "recon_epochs.json").write_text(json.dumps({"run": str(_args.run), "seed": _args.seed, "test_indices": _args.images,
        "names": names, "tile_mse_x1e3": {m["key"]: m["mse"] for m in payload["models"]},
        "val_mse_x1e3": {m["key"]: m["val"] for m in payload["models"]}, "layout": "[epoch][image][code]"}, indent=1) + "\n")
    print(OUT / "recon_epochs.html", f"{len(html) / 1e6:.2f} MB")


TEMPLATE = r"""<title>CIFAR-100 Reconstructions by Epoch</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #1b1f27; --muted: #5d6572; --line: #dde1e7; --plate: #e9ecf0; --accent: #2d5bd7;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111318; --surface: #191c23; --ink: #e6e8ec; --muted: #9aa2b1; --line: #2c313b; --plate: #252931; --accent: #7ea2ff; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111318; --surface: #191c23; --ink: #e6e8ec; --muted: #9aa2b1; --line: #2c313b; --plate: #252931; --accent: #7ea2ff; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--surface); }
.wrap { max-width: 1240px; margin: 0 auto; padding-block: 20px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 16px; }
h1 { font-size: 1.4rem; font-weight: 600; margin: 0; text-wrap: balance; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 0; }
h2 small { font-weight: 400; color: var(--muted); font-size: 0.8rem; margin-left: 8px; }
p { margin: 0; max-width: 80ch; }
code { font-family: var(--mono); font-size: 0.85em; }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
.bar label { font-weight: 600; }
.bar output { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 2ch; font-size: 1.1rem; }
.slider { flex: 1 1 260px; min-width: 0; display: grid; gap: 2px; }
.slider input { width: 100%; accent-color: var(--accent); margin: 0; }
.ticks { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.68rem; color: var(--muted); }
button { font: inherit; font-size: 0.85rem; border: 1px solid var(--line); background: var(--plate); color: var(--ink); border-radius: 6px; padding: 4px 12px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.hint { font-size: 0.75rem; color: var(--muted); }
section { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 8px; min-width: 0; }
.scroll { overflow-x: auto; }
.grid { display: grid; grid-template-columns: 150px repeat(11, 68px); gap: 2px 6px; align-items: center; width: max-content; }
.grid .h { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); text-align: center; }
.grid .h b { color: var(--ink); font-weight: 500; }
.grid .row { font-size: 0.8rem; color: var(--muted); line-height: 1.25; padding-right: 6px; }
.grid .row b { color: var(--ink); font-weight: 500; display: block; }
.cell { display: grid; justify-items: center; gap: 1px; }
.cell canvas { width: 64px; height: 64px; image-rendering: pixelated; border-radius: 2px; background: var(--plate); }
.cell span { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.val { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }
footer { font-size: 0.8rem; color: var(--muted); display: grid; gap: 6px; }
</style>

<div class="wrap">
  <header style="display:grid;gap:6px">
    <h1>CIFAR-100: AE vs MRL-prefix vs MRL-pair, by epoch</h1>
    <p>Run <code>__RUN__</code>, seed 0, 10-d code, lr 6e-5. Four test images, decoded from partial codes at the epoch on the slider
      (epoch 0 = initial weights). The number under each tile is that image's MSE ×10⁻³; the row under each panel is the
      MSE of that code over the whole validation set (5,000 images).</p>
  </header>

  <div class="bar" role="group" aria-label="Epoch">
    <label for="ep">Epoch</label>
    <output id="ep-out" for="ep"></output>
    <div class="slider">
      <input type="range" id="ep" min="0" step="1" value="0">
      <div class="ticks" id="ticks" aria-hidden="true"></div>
    </div>
    <button type="button" id="play">Play</button>
    <span class="hint">Arrow keys step epochs</span>
  </div>

  <div id="panels" style="display:grid;gap:14px"></div>

  <footer>
    <p>Codes: AE and MRL use prefixes (keep z<sub>1</sub>…z<sub>k</sub>, zero the rest); the pair model uses the codes it was trained on
      ({z<sub>1</sub>} and {z<sub>1</sub>, z<sub>j</sub>}). Reconstructions clipped to [0, 1]; 32×32 pixels shown at 2×. A constant
      mean-image output scores 69.5 on the validation set.</p>
    <p>Generator: <code>scripts/cifar_recon_epochs_html.py</code>.</p>
  </footer>
</div>

<script>
const DATA = __DATA__;
const EP = DATA.epochs, NI = DATA.images.length, T = 32, G = 2;
let idx = EP.length - 1, timer = null;
const ticks = document.getElementById("ticks");
EP.forEach(e => { const s = document.createElement("span"); s.textContent = e % 5 === 0 ? e : "·"; ticks.appendChild(s); });
const range = document.getElementById("ep");
range.max = EP.length - 1;

const panels = [];
DATA.models.forEach(m => {
  const sec = document.createElement("section");
  let h = `<h2>${m.key === "pair" ? "MRL-pair" : m.key === "mrl" ? "MRL-prefix" : "AE"}<small>${m.note}</small></h2><div class="scroll"><div class="grid">`;
  h += `<div></div><div class="h"><b>input</b></div>` + m.labels.map(l => `<div class="h"><b>${l}</b></div>`).join("");
  DATA.images.forEach((im, i) => {
    h += `<div class="row"><b>${im.name.split(" (")[0]}</b>${im.name.split(" (")[1].replace(")", "")} · #${im.index}</div>`;
    for (let c = 0; c <= m.labels.length; c++) h += `<div class="cell"><canvas width="${T}" height="${T}" data-i="${i}" data-c="${c}"></canvas><span data-i="${i}" data-c="${c}"></span></div>`;
  });
  h += `<div class="row">val set</div><div></div>` + m.labels.map((l, k) => `<div class="val" data-k="${k}"></div>`).join("");
  h += `</div></div>`;
  sec.innerHTML = h;
  document.getElementById("panels").appendChild(sec);
  const imgs = m.sheets.map(src => { const im = new Image(); im.src = src; return im; });
  panels.push({ m, sec, imgs });
});

function render() {
  const ep = EP[idx];
  document.getElementById("ep-out").textContent = ep;
  range.value = idx;
  range.setAttribute("aria-valuetext", "epoch " + ep);
  panels.forEach(p => {
    const sheet = p.imgs[idx];
    const draw = () => p.sec.querySelectorAll("canvas").forEach(cv => {
      const i = +cv.dataset.i, c = +cv.dataset.c, ctx = cv.getContext("2d");
      ctx.clearRect(0, 0, T, T);
      ctx.drawImage(sheet, c * (T + G), i * (T + G), T, T, 0, 0, T, T);
      cv.setAttribute("aria-label", `${p.m.key} ${DATA.images[i].name} ${c === 0 ? "input" : p.m.labels[c - 1]} epoch ${ep}`);
    });
    if (sheet.complete) draw(); else sheet.onload = () => { if (EP[idx] === ep) draw(); };
    p.sec.querySelectorAll(".cell span").forEach(s => {
      const i = +s.dataset.i, c = +s.dataset.c;
      s.textContent = c === 0 ? "" : p.m.mse[idx][i][c - 1].toFixed(1);
    });
    p.sec.querySelectorAll(".val").forEach(v => { const r = p.m.val[idx]; v.textContent = r ? r[+v.dataset.k].toFixed(1) : "—"; });
  });
}
function step(d) { idx = Math.max(0, Math.min(EP.length - 1, idx + d)); render(); }
function stop() { clearInterval(timer); timer = null; document.getElementById("play").textContent = "Play"; }
range.addEventListener("input", e => { idx = +e.target.value; render(); });
document.getElementById("play").addEventListener("click", () => {
  if (timer) return stop();
  if (idx >= EP.length - 1) idx = 0;
  document.getElementById("play").textContent = "Pause";
  render();
  timer = setInterval(() => { if (idx >= EP.length - 1) return stop(); step(1); }, 700);
});
document.addEventListener("keydown", e => {
  if (e.target.type === "range") return;
  if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); step(1); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); step(-1); }
});
render();
</script>
"""


if __name__ == "__main__":
    main()
