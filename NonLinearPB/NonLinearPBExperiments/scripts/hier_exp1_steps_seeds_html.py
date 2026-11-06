"""Epoch-slider page: every prefix step of one model, all seeds side by side, five views, one image at a time.

Conda environment: mrl_env_cuda12
Reads <run>/figures/prefix_change/data/<img>_e<ep>.npz (hier_exp1_prefix_change.py): recon [model, seed, d+1, 64, 64]
(prefix 0 = all-zero code, then z1, z1:2, ..., z1:d), D [model, seed, d, 64, 64] with D_k = recon_k - recon_{k-1},
alignment [model, seed, ref, d] = cos(D_k, T_C / T_M / T_F).
Per image: one row per seed; columns: input, then for each step k = 1..d a tile (view below) and cos(D_k, C/M/F) vs epoch.
Views (as on the prefix-change slider pages, but step 1 is shown as a change D_1 = decode(z1) - decode(0)):
    tile    : signed change, own limit per tile (max |D_k| of that tile)
    seed    : signed change, one limit per seed = max |D_k| over k = 1..d, the 4 images and all epochs
    recon   : decode(z1:k), grey, clipped to [0, 1], MSE x1e-3 vs the input
    pos     : brighter only, max(D_k, 0), own limit per tile
    posseed : brighter only, one limit per seed (max positive change over k, images, epochs)
The maps are drawn in the browser from embedded data (recon uint8, D int8 per-tile scaled, gzip + base64).
Writes <run>/figures/prefix_change/steps_seeds/<model>_steps_seeds.html and a .json with the limits and cosines.

Usage: python scripts/hier_exp1_steps_seeds_html.py [--run R] [--model mrl] [--desc "..."]
"""
import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_05__11_44_34"))
_ap.add_argument("--model", default="mrl")
_ap.add_argument("--images", nargs="+", default=["c1m0f0", "c1m2f0", "c0m2f0", "c2m2f0"])
_ap.add_argument("--epochs", type=int, nargs="+", default=[2, 3, 4, 5, 10, 15, 20, 25])
_ap.add_argument("--desc", default="5-d latent, lr 6e-5, prefix codes")
_ap.add_argument("--title", default=None)
_args = _ap.parse_args()
SRC = _args.run / "figures" / "prefix_change"
OUT = SRC / "steps_seeds"
CMAP = matplotlib.colormaps["RdBu"]          # red = darker, blue = brighter, white = 0
SCALE = 2


def uri_div(a):
    lim = float(np.abs(a).max()) or 1e-12
    rgb = (CMAP(np.clip(0.5 + 0.5 * a / lim, 0, 1))[..., :3] * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((64 * SCALE, 64 * SCALE), Image.NEAREST).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), round(lim, 3)


def uri(a):
    g = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(g).resize((64 * SCALE, 64 * SCALE), Image.NEAREST).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pack(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "metadata.json").read_text())
    EP = _args.epochs
    raw = {n: [np.load(SRC / "data" / f"{n}_e{ep:03d}.npz") for ep in EP] for n in _args.images}
    models = [str(m) for m in raw[_args.images[0]][0]["models"]]
    mi = models.index(_args.model)
    REC = {n: np.stack([d["recon"][mi] for d in raw[n]], 1) for n in _args.images}   # [seed, ep, d+1, 64, 64]
    D = {n: np.stack([d["D"][mi] for d in raw[n]], 1) for n in _args.images}         # [seed, ep, d, 64, 64]
    S, _, d_lat = D[_args.images[0]].shape[:3]
    lim_seed = [float(max(np.abs(D[n][s]).max() for n in _args.images)) for s in range(S)]
    pos_seed = [float(max(np.maximum(D[n][s], 0).max() for n in _args.images)) for s in range(S)]
    images = []
    for n in _args.images:
        info, x = meta["images"][n], raw[n][0]["x"]
        dd = D[n]
        tile_max = np.abs(dd).max(axis=(3, 4))                                         # [seed, ep, d]
        q = np.round(dd / np.maximum(tile_max, 1e-12)[..., None, None] * 127).astype(np.int8)
        rec = REC[n][:, :, 1:]                                                        # z1 .. z1:d
        rec8 = (np.clip(rec, 0, 1) * 255).round().astype(np.uint8)
        mse = (((np.clip(rec, 0, 1) - x) ** 2).mean(axis=(3, 4)) * 1e3).round(2)       # [seed, ep, d]
        pos_max = np.maximum(dd, 0).max(axis=(3, 4))
        cos = np.stack([d["alignment"][mi] for d in raw[n]], 1)                        # [seed, ep, ref, d]
        images.append({"key": n, "name": info["name"], "c": info["c"], "m": info["m"], "f": info["f"],
                       "test_index": info["test_index"], "x": uri(x),
                       "refs": [{"label": lab, "src": u, "lim": lim} for lab, (u, lim) in
                                zip(("T_C", "T_M", "T_F"), (uri_div(raw[n][0]["T"][j]) for j in range(3)))],
                       "D": pack(q), "R": pack(rec8), "tmax": tile_max.round(5).tolist(), "pmax": pos_max.round(5).tolist(),
                       "mse": mse.tolist(), "cos": cos.round(3).tolist()})
    lut = (CMAP(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(int).ravel().tolist()
    payload = {"epochs": EP, "d": d_lat, "seeds": S, "lim_seed": lim_seed, "pos_seed": pos_seed, "lut": lut, "images": images}
    title = _args.title or f"{_args.model.upper()} Steps Across Seeds" + (f" z{d_lat}" if d_lat != 10 else "")
    html = (TEMPLATE.replace("__DATA__", json.dumps(payload)).replace("__TITLE__", title)
            .replace("__MODEL__", _args.model.upper()).replace("__RUN__", _args.run.name).replace("__DESC__", _args.desc))
    path = OUT / f"{_args.model}_steps_seeds.html"
    path.write_text(html)
    (OUT / f"{_args.model}_steps_seeds.json").write_text(json.dumps({
        "run": str(_args.run), "model": _args.model, "epochs": EP, "images": _args.images,
        "lim_seed": lim_seed, "pos_seed": pos_seed,
        "cos": {im["key"]: im["cos"] for im in images},
        "layout": "cos[img][seed][epoch][ref C, M, F][step k = 1..d]; D_k = decode(z1:k) - decode(z1:k-1), D_1 from decode(0)"},
        indent=1) + "\n")
    print(path, f"{len(html) / 1e6:.2f} MB")


TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #1b1f27; --muted: #5d6572; --line: #dde1e7; --plate: #e9ecf0;
  --accent: #2d5bd7; --refC: #0072B2; --refM: #C27F00; --refF: #009E73;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111318; --surface: #191c23; --ink: #e6e8ec; --muted: #9aa2b1; --line: #2c313b; --plate: #252931;
  --accent: #7ea2ff; --refC: #4aa8e0; --refM: #E6A532; --refF: #2fc29a; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111318; --surface: #191c23; --ink: #e6e8ec; --muted: #9aa2b1; --line: #2c313b; --plate: #252931;
  --accent: #7ea2ff; --refC: #4aa8e0; --refM: #E6A532; --refF: #2fc29a; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--surface); }
.wrap { max-width: 1480px; margin: 0 auto; padding-block: 20px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 16px; }
h1 { font-size: 1.4rem; font-weight: 600; margin: 0; text-wrap: balance; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 80ch; }
code, .mono { font-family: var(--mono); font-size: 0.85em; }
header { display: grid; gap: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; color: var(--muted); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 14px; display: grid; gap: 8px; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
.row > b { font-weight: 600; min-width: 4.5em; }
.bar output { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 2ch; font-size: 1.1rem; }
.slider { flex: 1 1 260px; min-width: 0; display: grid; gap: 2px; }
.slider input { width: 100%; accent-color: var(--accent); margin: 0; }
.ticks { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
button { font: inherit; font-size: 0.85rem; border: 1px solid var(--line); background: var(--plate); color: var(--ink); border-radius: 6px; padding: 4px 12px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--surface); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.seg { display: flex; flex-wrap: wrap; gap: 6px; }
.hint { font-size: 0.75rem; color: var(--muted); }
section.img { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.scroll { overflow-x: auto; }
.tbl { display: grid; gap: 8px 8px; align-items: center; }
.th { font-size: 0.8rem; color: var(--muted); font-weight: 500; text-align: center; }
.th.step { color: var(--ink); font-weight: 600; text-align: left; border-bottom: 1px solid var(--line); padding-bottom: 2px; }
.seed { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }
img.px, canvas.px { image-rendering: pixelated; display: block; width: 72px; height: 72px; border-radius: 2px; background: var(--plate); }
.cap { font-family: var(--mono); font-size: 0.64rem; color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }
.chart { display: grid; gap: 2px; min-width: 0; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart text { font-family: var(--mono); font-size: 9px; fill: var(--muted); }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 2; }
.chart .now { stroke: var(--accent); stroke-width: 1.5; opacity: 0.5; }
.chart .hit { fill: transparent; cursor: crosshair; }
.vals { font-family: var(--mono); font-size: 0.68rem; font-variant-numeric: tabular-nums; display: flex; gap: 8px; }
.tip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; font-family: var(--mono); font-size: 0.72rem; line-height: 1.45; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); }
.refs { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
.refs figure { margin: 0; display: grid; gap: 3px; justify-items: center; }
.refs img { width: 88px; height: 88px; image-rendering: pixelated; border-radius: 2px; }
.refs figcaption { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); text-align: center; line-height: 1.35; }
.refnote { font-size: 0.78rem; color: var(--muted); max-width: 46ch; align-self: center; }
.cbar { display: flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }
.cbar i { display: inline-block; width: 160px; height: 10px; border-radius: 2px; border: 1px solid var(--line); }
footer { font-size: 0.8rem; color: var(--muted); display: grid; gap: 6px; }
</style>

<div class="wrap">
  <header>
    <h1>__MODEL__: every prefix step, every seed</h1>
    <p>Run <code>__RUN__</code> (__DESC__). Rows are seeds. For each step k: a tile (choose the view) and cos(D<sub>k</sub>, T) over all
      epochs, with D<sub>k</sub> = decode(z<sub>1:k</sub>) − decode(z<sub>1:k−1</sub>) and step 1 measured from the all-zero code.</p>
    <div class="legend">
      <span><span class="sw" style="background:var(--refC)"></span><b>C</b> family (T<sub>C</sub>)</span>
      <span><span class="sw" style="background:var(--refM)"></span><b>M</b> mid (T<sub>M</sub>)</span>
      <span><span class="sw" style="background:var(--refF)"></span><b>F</b> fine (T<sub>F</sub>)</span>
      <span>Charts: −1 to +1, vertical line = slider epoch, hover for values.</span>
    </div>
  </header>

  <div class="bar">
    <div class="row" role="group" aria-label="Image">
      <b>Image</b><div class="seg" id="imgs"></div>
    </div>
    <div class="row" role="group" aria-label="View">
      <b>View</b><div class="seg" id="views"></div>
      <span class="cbar" id="cbar"></span>
    </div>
    <div class="row" role="group" aria-label="Epoch">
      <b><label for="ep">Epoch</label></b>
      <output id="ep-out" for="ep"></output>
      <div class="slider">
        <input type="range" id="ep" min="0" step="1" value="0">
        <div class="ticks" id="ticks" aria-hidden="true"></div>
      </div>
      <button type="button" id="play">Play</button>
      <span class="hint">Arrow keys step epochs</span>
    </div>
  </div>

  <section class="img" id="sec"><p class="hint" id="loading">Loading data…</p></section>

  <footer>
    <p>Change views: red = pixel gets darker, blue = brighter, white = no change. <b>Own scale</b>: each tile scaled to its own
      max |D<sub>k</sub>| (printed under it), so small steps look as strong as big ones. <b>Shared scale</b>: one limit per seed over all steps,
      the 4 images and all epochs, so tile strengths compare. <b>Brighter only</b>: max(D<sub>k</sub>, 0). Reconstructions: decode(z<sub>1:k</sub>),
      clipped to [0, 1], MSE ×10⁻³ against the input.</p>
    <p>References from training-set means: T<sub>C</sub> = G<sub>C</sub> − G<sub>0</sub>, T<sub>M</sub> = G<sub>CM</sub> − G<sub>C</sub>,
      T<sub>F</sub> = G<sub>CMF</sub> − G<sub>CM</sub>. Cosine over the 4,096 pixels, signed, no cut-off, from the unquantised maps.
      Change tiles are drawn from 8-bit copies of the maps (±127 steps per tile).</p>
    <p>Generator: <code>scripts/hier_exp1_steps_seeds_html.py</code> from <code>figures/prefix_change/data/</code>.</p>
  </footer>
</div>

<script>
const DATA = __DATA__;
const EP = DATA.epochs, DL = DATA.d, NS_ = DATA.seeds, LUT = DATA.lut;
const REF = [["C", "--refC"], ["M", "--refM"], ["F", "--refF"]];
const VIEWS = [["tile", "Change, own scale"], ["seed", "Change, shared scale"], ["recon", "Reconstructions"],
               ["pos", "Brighter, own scale"], ["posseed", "Brighter, shared scale"]];
const fmt = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const NS = "http://www.w3.org/2000/svg";
const X0 = 22, X1 = 196, Y0 = 6, Y1 = 66, W = 200, H = 80;
const xAt = i => X0 + (X1 - X0) * i / (EP.length - 1);
const yAt = v => Y0 + (Y1 - Y0) * (1 - v) / 2;
let idx = 0, timer = null, view = "tile", cur = 0, cells = [];
const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); parent.appendChild(e); return e; }
const stepName = k => k === 0 ? "∅→1" : `${k}→${k + 1}`;
const codeName = k => k === 0 ? "z1" : `z1:${k + 1}`;

async function unpack(b64, Type) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const buf = await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  return new Type(buf);
}

const ticks = document.getElementById("ticks");
EP.forEach(e => { const s = document.createElement("span"); s.textContent = e; ticks.appendChild(s); });
const range = document.getElementById("ep");
range.max = EP.length - 1;

function chart(svg, cos, k, label) {        // cos[ep][ref][step]
  const v = (i, j) => cos[i][j][k];
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${label}: cosine with C, M, F over epochs ` + EP.map((e, i) => `${e}: ` + [0, 1, 2].map(j => v(i, j).toFixed(2)).join("/")).join(", "));
  for (const y of [1, 0.5, -0.5, -1]) el("line", {x1: X0, x2: X1, y1: yAt(y), y2: yAt(y), class: "grid"}, svg);
  el("line", {x1: X0, x2: X1, y1: yAt(0), y2: yAt(0), class: "zero"}, svg);
  for (const y of [1, 0, -1]) { const t = el("text", {x: X0 - 3, y: yAt(y) + 3, "text-anchor": "end"}, svg); t.textContent = y === 0 ? "0" : (y > 0 ? "+1" : "−1"); }
  EP.forEach((e, i) => { if (i % 2 === 0 || i === EP.length - 1) { const t = el("text", {x: xAt(i), y: 78, "text-anchor": "middle"}, svg); t.textContent = e; } });
  const now = el("line", {x1: xAt(0), x2: xAt(0), y1: Y0, y2: Y1, class: "now"}, svg);
  REF.forEach(([l, c], j) => {
    el("polyline", {points: EP.map((e, i) => `${xAt(i).toFixed(1)},${yAt(v(i, j)).toFixed(1)}`).join(" "),
      fill: "none", stroke: `var(${c})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg);
  });
  const hit = el("rect", {x: X0 - 4, y: 0, width: X1 - X0 + 8, height: H, class: "hit"}, svg);
  hit.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect();
    const vx = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(EP.length - 1, Math.round((vx - X0) / (X1 - X0) * (EP.length - 1))));
    tip.innerHTML = `${label} · epoch ${EP[i]}<br>` + REF.map(([l, c], j) => `<span style="color:var(${c})">${l}</span> ${fmt(v(i, j))}`).join("&nbsp; ");
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (ev.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; });
  return now;
}

function draw(ctx, im, s, e, k) {
  const off = ((s * EP.length + e) * DL + k) * 4096;
  const out = ctx.createImageData(64, 64), px = out.data;
  if (view === "recon") {
    for (let p = 0; p < 4096; p++) { const g = im.R[off + p]; px[4*p] = px[4*p+1] = px[4*p+2] = g; px[4*p+3] = 255; }
  } else {
    const tmax = im.tmax[s][e][k], pos = view === "pos" || view === "posseed";
    const lim = view === "tile" ? tmax : view === "seed" ? DATA.lim_seed[s] : view === "pos" ? im.pmax[s][e][k] : DATA.pos_seed[s];
    for (let p = 0; p < 4096; p++) {
      let d = im.D[off + p] / 127 * tmax;
      if (pos && d < 0) d = 0;
      const t = lim > 0 ? Math.max(0, Math.min(1, 0.5 + 0.5 * d / lim)) : 0.5;
      const li = Math.round(t * 255) * 3;
      px[4*p] = LUT[li]; px[4*p+1] = LUT[li+1]; px[4*p+2] = LUT[li+2]; px[4*p+3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
}

function caption(im, s, e, k) {
  if (view === "recon") return `MSE ${im.mse[s][e][k].toFixed(2)}`;
  if (view === "tile") return `±${im.tmax[s][e][k].toFixed(3)}`;
  if (view === "pos") return `max +${im.pmax[s][e][k].toFixed(3)}`;
  return `|D| ${im.tmax[s][e][k].toFixed(3)}`;
}

function cbar() {
  const grad = (a, b) => Array.from({length: 11}, (_, i) => { const li = Math.round((a + (b - a) * i / 10) * 255) * 3; return `rgb(${LUT[li]},${LUT[li+1]},${LUT[li+2]})`; }).join(",");
  const cb = document.getElementById("cbar");
  if (view === "recon") { cb.innerHTML = `<i style="background:linear-gradient(90deg,#000,#fff)"></i> 0 … 1`; return; }
  const pos = view === "pos" || view === "posseed";
  const shared = view === "seed" ? "seed limits " + DATA.lim_seed.map(v => v.toFixed(2)).join(" / ") :
                 view === "posseed" ? "seed limits " + DATA.pos_seed.map(v => v.toFixed(2)).join(" / ") : "own limit per tile";
  cb.innerHTML = pos ? `<i style="background:linear-gradient(90deg,${grad(0.5, 1)})"></i> 0 … +lim · ${shared}`
                     : `<i style="background:linear-gradient(90deg,${grad(0, 1)})"></i> −lim (darker) … +lim (brighter) · ${shared}`;
}

function build() {
  const im = DATA.images[cur], sec = document.getElementById("sec");
  const REFTXT = ["family: G<sub>C</sub> − G<sub>0</sub>", "mid: G<sub>CM</sub> − G<sub>C</sub>", "fine: G<sub>CMF</sub> − G<sub>CM</sub>"];
  sec.innerHTML = `<h2>${im.name} <span class="mono" style="color:var(--muted);font-weight:400">c${im.c} m${im.m} f${im.f} · test #${im.test_index}</span></h2>
    <div class="refs">
      <figure><img src="${im.x}" alt="input image" width="88" height="88"><figcaption>input</figcaption></figure>
      ${im.refs.map((r, j) => `<figure><img src="${r.src}" alt="ground-truth change ${r.label}, limit ${r.lim}" width="88" height="88">
        <figcaption><span style="color:var(${REF[j][1]})">${r.label.replace("_", "")}</span> ±${r.lim.toFixed(2)}<br>${REFTXT[j]}</figcaption></figure>`).join("")}
      <p class="refnote">Ground-truth changes for this image's leaf, from training-set mean images (red = darker, blue = brighter, each scaled to its own ± limit). The charts compare each step D<sub>k</sub> with these.</p>
    </div>`;
  const sc = document.createElement("div"); sc.className = "scroll";
  const g = document.createElement("div"); g.className = "tbl";
  g.style.gridTemplateColumns = `48px 72px repeat(${DL}, 72px minmax(150px, 1fr))`;
  g.style.minWidth = `${48 + 72 + DL * 238 + 8}px`;
  let html = `<div></div><div class="th">input</div>`;
  for (let k = 0; k < DL; k++) html += `<div class="th step" style="grid-column: span 2" data-k="${k}"></div>`;
  for (let s = 0; s < NS_; s++) {
    html += `<div class="seed">seed ${s}</div><div><img class="px" src="${im.x}" alt="input" width="72" height="72"></div>`;
    for (let k = 0; k < DL; k++) html += `<div class="tile" data-s="${s}" data-k="${k}"><canvas class="px" width="64" height="64"></canvas><div class="cap"></div></div>
      <div class="chart" data-s="${s}" data-k="${k}"><svg></svg><div class="vals"></div></div>`;
  }
  g.innerHTML = html;
  sc.appendChild(g); sec.appendChild(sc);
  cells = [];
  g.querySelectorAll(".tile").forEach(t => {
    const s = +t.dataset.s, k = +t.dataset.k;
    const c = g.querySelector(`.chart[data-s="${s}"][data-k="${k}"]`);
    const cos = im.cos[s];
    const now = chart(c.querySelector("svg"), cos, k, `seed ${s}, step ${stepName(k)}`);
    cells.push({s, k, cos, now, cv: t.querySelector("canvas"), ctx: t.querySelector("canvas").getContext("2d"), cap: t.querySelector(".cap"), vals: c.querySelector(".vals")});
  });
}

function render() {
  const im = DATA.images[cur], ep = EP[idx];
  document.getElementById("ep-out").textContent = ep;
  range.value = idx;
  range.setAttribute("aria-valuetext", "epoch " + ep);
  document.querySelectorAll(".th.step").forEach(h => { const k = +h.dataset.k;
    h.textContent = view === "recon" ? `${codeName(k)}  ·  cos of step ${stepName(k)}` : `step ${stepName(k)}`; });
  cells.forEach(c => {
    draw(c.ctx, im, c.s, idx, c.k);
    c.cv.setAttribute("aria-label", `seed ${c.s}, ${view === "recon" ? "reconstruction " + codeName(c.k) : "step " + stepName(c.k)}, epoch ${ep}`);
    c.cap.textContent = caption(im, c.s, idx, c.k);
    c.now.setAttribute("x1", xAt(idx)); c.now.setAttribute("x2", xAt(idx));
    c.vals.innerHTML = REF.map(([l, col], j) => `<span><span style="color:var(${col})">${l}</span> ${fmt(c.cos[idx][j][c.k])}</span>`).join("");
  });
}

function seg(host, items, isOn, onPick) {
  host.innerHTML = "";
  items.forEach(([key, label]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.dataset.key = key;
    b.setAttribute("aria-pressed", isOn(key));
    b.addEventListener("click", () => { onPick(key); host.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", isOn(x.dataset.key))); });
    host.appendChild(b);
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
  timer = setInterval(() => { if (idx >= EP.length - 1) return stop(); step(1); }, 1000);
});
document.addEventListener("keydown", e => {
  if (e.target.type === "range") return;
  if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); step(1); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); step(-1); }
});

(async () => {
  try {
    for (const im of DATA.images) { im.D = await unpack(im.D, Int8Array); im.R = await unpack(im.R, Uint8Array); }
  } catch (err) {
    document.getElementById("loading").textContent = "This browser could not unpack the image data (needs DecompressionStream).";
    return;
  }
  seg(document.getElementById("imgs"), DATA.images.map((im, i) => [String(i), im.name]), k => +k === cur, k => { cur = +k; build(); render(); });
  seg(document.getElementById("views"), VIEWS, k => k === view, k => { view = k; cbar(); render(); });
  cbar(); build(); render();
})();
</script>
"""


if __name__ == "__main__":
    main()
