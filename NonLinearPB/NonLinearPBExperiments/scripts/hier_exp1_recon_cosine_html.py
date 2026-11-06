"""Epoch-slider page: prefix reconstructions with the cosine of each step to the hierarchy references.

Conda environment: mrl_env_cuda12
Reads figures/prefix_change/data/<img>_e<ep>.npz (written by hier_exp1_prefix_change.py; any latent size).
For each chosen image, one seed, AE and MRL, each epoch:
    row 1: input, then xhat_{1:k} = decode([z1..zk, 0..]), k = 1..d (grey, clipped to [0, 1])
    row 2: D_k = xhat_{1:k} - xhat_{1:k-1} (k = 1: minus the decode of the all-zero code), RdBu, own limit per tile
    under each step: cos(D_k, T_C), cos(D_k, T_M), cos(D_k, T_F) as bars on [-1, 1] for the chosen epoch,
    then a small chart of the three cosines against epoch (current epoch marked), and a table of all values
References (per image, from training-set conditional means): T_C = G_C - G_0, T_M = G_CM - G_C, T_F = G_CMF - G_CM.
Writes figures/prefix_change/recon_cosine/recon_cosine.html (self-contained).

Usage: python scripts/hier_exp1_recon_cosine_html.py --run R [--images c1m0f0 c0m2f0] [--seed 0] [--desc "..."]
"""
import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_05__11_44_34"))
_ap.add_argument("--images", nargs="+", default=["c1m0f0", "c1m2f0", "c0m2f0", "c2m2f0"])
_ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--epochs", type=int, nargs="+", default=[2, 3, 4, 5, 10, 15, 20, 25])
_ap.add_argument("--desc", default="lr 6e-5, 5-d latent")
_ap.add_argument("--src", type=Path, help="folder with data/ and metadata.json (default <run>/figures/prefix_change)")
_ap.add_argument("--out", type=Path, help="output folder (default <src>/recon_cosine)")
_ap.add_argument("--show", nargs="+", default=["mrl", "ae"], help="models in page order (names as in the npz)")
_ap.add_argument("--codes", choices=("prefix", "pair"), default="prefix",
                 help="prefix: z1:k, steps 1:(k-1) -> 1:k; pair: {z1}, {z1, zj}, changes relative to {z1}")
_ap.add_argument("--title", default=None, help="page <title> (default Prefix Cosine z<d> / Pair Cosine z<d>)")
_args = _ap.parse_args()
SRC = _args.src or _args.run / "figures" / "prefix_change"
OUT = _args.out or SRC / "recon_cosine"
SHOW = tuple(_args.show)          # order on the page
NOTES = {"mrl": "trained on every prefix", "ae": "trained on the full code", "pair": "trained on {z1} and each {z1, zj}"}
CMAP = matplotlib.colormaps["RdBu"]
SCALE = 2                         # upsample tiles (nearest) so they stay sharp


def uri(rgb):
    img = Image.fromarray(rgb).resize((rgb.shape[1] * SCALE, rgb.shape[0] * SCALE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def grey(a):
    return np.repeat((np.clip(a, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, -1)


def diverging(a):
    lim = float(np.abs(a).max()) or 1e-12
    return (CMAP(np.clip(0.5 + 0.5 * a / lim, 0, 1))[..., :3] * 255).round().astype(np.uint8), lim


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "metadata.json").read_text())
    s = _args.seed
    images = []
    for name in _args.images:
        info = meta["images"][name]
        first = np.load(SRC / "data" / f"{name}_e{_args.epochs[0]:03d}.npz")
        MODELS = [str(m) for m in first["models"]]                                      # order inside the npz
        refs = []
        for j, lab in enumerate(("T_C", "T_M", "T_F")):
            rgb, lim = diverging(first["T"][j])
            refs.append({"label": lab, "src": uri(rgb), "lim": round(lim, 3)})
        entry = {"key": name, "name": info["name"], "c": info["c"], "m": info["m"], "f": info["f"],
                 "test_index": info["test_index"], "x": uri(grey(first["x"])), "refs": refs, "epochs": {}}
        for ep in _args.epochs:
            d = np.load(SRC / "data" / f"{name}_e{ep:03d}.npz")
            per_model = {}
            for m in SHOW:
                mi = MODELS.index(m)
                rec, D, A = d["recon"][mi, s], d["D"][mi, s], d["alignment"][mi, s]     # [d+1,64,64], [d,64,64], [3, d]
                mse = ((rec[1:] - d["x"]) ** 2).mean(axis=(1, 2))
                steps = []
                for k in range(D.shape[0]):
                    rgb, lim = diverging(D[k])
                    steps.append({"rec": uri(grey(rec[k + 1])), "d": uri(rgb), "lim": round(lim, 3),
                                  "mse": round(float(mse[k]) * 1e3, 2), "cos": [round(float(v), 3) for v in A[:, k]]})
                per_model[m] = steps
            entry["epochs"][str(ep)] = per_model
        images.append(entry)
    d_lat = len(images[0]["epochs"][str(_args.epochs[0])]["mrl"])
    pair = _args.codes == "pair"
    payload = {"run": _args.run.name, "desc": _args.desc, "seed": s, "epochs": _args.epochs, "d": d_lat, "images": images,
               "show": list(SHOW), "notes": {m: NOTES.get(m, "") for m in SHOW}, "codes": _args.codes}
    if pair:
        h1 = "Adding one coordinate to z<sub>1</sub>: reconstructions and alignment with the hierarchy"
        intro = ("Each column is one code: {z<sub>1</sub>}, then {z<sub>1</sub>, z<sub>j</sub>} for j = 2…__D__ (rest of the code zero). "
                 "Under each reconstruction: the change, D<sub>1</sub> = x̂<sub>z1</sub> − decode(all-zero code) and "
                 "D<sub>j</sub> = x̂<sub>z1,zj</sub> − x̂<sub>z1</sub> (what adding z<sub>j</sub> on top of z<sub>1</sub> does), "
                 "and the cosine of that change with the three reference changes.")
    else:
        h1 = "Prefix reconstructions and their alignment with the hierarchy"
        intro = ("Each column is one prefix: the reconstruction from z<sub>1:k</sub> (rest of the code zero), "
                 "the change that step made, D<sub>k</sub> = x̂<sub>1:k</sub> − x̂<sub>1:k−1</sub> (for k = 1: minus the decode of the all-zero code), "
                 "and the cosine of that change with the three reference changes.")
    title = _args.title or f"{'Pair' if pair else 'Prefix'} Cosine z{d_lat}"
    caveat = "AE caveat: the AE never saw zeroed coordinates during training, and its coordinate order is arbitrary." if "ae" in SHOW else ""
    if pair:
        caveat += (" The MRL was trained on prefixes, so it never saw {z1, zj} codes with a zero in between; "
                   "the pair model never saw the full code." if "mrl" in SHOW else "")
    html = TEMPLATE.replace("__CAVEAT__", caveat).replace("__H1__", h1).replace("__INTRO__", intro).replace("__TITLE__", title) \
                   .replace("__SRC__", str(SRC.relative_to(_args.run)) if SRC.is_relative_to(_args.run) else str(SRC)) \
                   .replace("__DATA__", json.dumps(payload)).replace("__RUN__", _args.run.name) \
                   .replace("__DESC__", _args.desc).replace("__SEED__", str(s)).replace("__D__", str(d_lat))
    path = OUT / "recon_cosine.html"
    path.write_text(html)
    (OUT / "recon_cosine.json").write_text(json.dumps({
        "run": str(_args.run), "src": str(SRC), "codes": _args.codes, "images": _args.images, "seed": s, "epochs": _args.epochs, "latent": d_lat,
        "cos": {im["key"]: {ep: {m: [st["cos"] for st in v] for m, v in pm.items()} for ep, pm in im["epochs"].items()}
                for im in images},
        "cos_layout": "[step k = 1..d (pair: {z1}, {z1,z2}..) ][ref C, M, F]"}, indent=1) + "\n")
    print(path, f"{len(html) / 1e6:.2f} MB")


TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one column of image sections; each has the hierarchy references, then MRL and AE strips
   (reconstruction / change map / cosine bars per prefix step), all driven by one sticky epoch bar. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #1b1f27; --muted: #5d6572; --line: #dde1e7; --plate: #e9ecf0;
  --accent: #2d5bd7; --accent-ink: #ffffff;
  --refC: #0072B2; --refM: #C27F00; --refF: #009E73; --neg: #b3424a;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111318; --surface: #191c23; --ink: #e6e8ec; --muted: #9aa2b1; --line: #2c313b; --plate: #252931;
  --accent: #7ea2ff; --accent-ink: #0d1120; --refC: #4aa8e0; --refM: #E6A532; --refF: #2fc29a; --neg: #e07a80; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111318; --surface: #191c23; --ink: #e6e8ec; --muted: #9aa2b1; --line: #2c313b; --plate: #252931;
  --accent: #7ea2ff; --accent-ink: #0d1120; --refC: #4aa8e0; --refM: #E6A532; --refF: #2fc29a; --neg: #e07a80; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--surface); }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 20px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 18px; }
h1 { font-size: 1.45rem; font-weight: 600; margin: 0; text-wrap: balance; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 0; text-wrap: balance; }
h3 { font-size: 0.8rem; font-weight: 600; margin: 0; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
p { margin: 0; max-width: 72ch; }
code, .mono { font-family: var(--mono); font-size: 0.85em; }
header { display: grid; gap: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; color: var(--muted); }
.legend b { font-weight: 600; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
.bar label { font-weight: 600; }
.bar output { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 2ch; font-size: 1.1rem; }
.slider { flex: 1 1 260px; min-width: 0; display: grid; gap: 2px; }
.slider input { width: 100%; accent-color: var(--accent); margin: 0; }
.ticks { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
button { font: inherit; font-size: 0.85rem; border: 1px solid var(--line); background: var(--plate); color: var(--ink);
  border-radius: 6px; padding: 4px 12px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.hint { font-size: 0.75rem; color: var(--muted); }
section.img { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 14px; min-width: 0; }
.imghead { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: flex-end; }
.refs { display: flex; flex-wrap: wrap; gap: 10px; }
figure { margin: 0; display: grid; gap: 3px; justify-items: center; }
figcaption { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); text-align: center; line-height: 1.3; }
.refs img { width: 76px; height: 76px; }
img.px { image-rendering: pixelated; display: block; border-radius: 2px; background: var(--plate); }
.strip { overflow-x: auto; }
.model { display: grid; grid-template-columns: 88px repeat(var(--cols), minmax(84px, 1fr)); gap: 6px 8px; min-width: 620px; align-items: start; }
.mname { font-weight: 600; font-size: 0.95rem; padding-top: 4px; }
.mname small { display: block; font-weight: 400; color: var(--muted); font-size: 0.72rem; line-height: 1.3; }
.cell { display: grid; gap: 4px; justify-items: stretch; }
.cell img { width: 100%; height: auto; aspect-ratio: 1; max-width: 100%; }
.cap { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }
.cos { display: grid; gap: 3px; }
.cr { display: grid; grid-template-columns: 12px 1fr 3.4em; align-items: center; gap: 4px; font-family: var(--mono); font-size: 0.7rem; font-variant-numeric: tabular-nums; }
.cr .lab { color: var(--muted); }
.track { position: relative; height: 9px; background: var(--plate); border-radius: 2px; }
.track::after { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--muted); }
.fill { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
.val { text-align: right; }
.divider { height: 1px; background: var(--line); }
.trend { width: 100%; height: auto; display: block; overflow: visible; }
.trend text { font-family: var(--mono); font-size: 9px; fill: var(--muted); }
.trend .grid { stroke: var(--line); stroke-width: 1; }
.trend .zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 2; }
.trend .now { stroke: var(--accent); stroke-width: 1.5; opacity: 0.5; }
.trend .hit { fill: transparent; cursor: crosshair; }
.trendcap { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); text-align: center; }
.tip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; font-family: var(--mono); font-size: 0.72rem; line-height: 1.45; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); }
details { font-size: 0.8rem; }
summary { cursor: pointer; color: var(--muted); }
.tbl { overflow-x: auto; margin-top: 6px; }
table { border-collapse: collapse; font-family: var(--mono); font-size: 0.72rem; font-variant-numeric: tabular-nums; }
th, td { padding: 2px 8px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; }
td.lead, th.lead { text-align: left; }
td.cur, th.cur { background: var(--plate); }
footer { font-size: 0.8rem; color: var(--muted); display: grid; gap: 6px; }
@media (prefers-reduced-motion: no-preference) { .fill { transition: left 0.2s, width 0.2s; } }
</style>

<div class="wrap">
  <header>
    <h1>__H1__</h1>
    <p>Run <code>__RUN__</code> (__DESC__), seed __SEED__. __INTRO__ Under each step, a chart of the three cosines over all epochs
      (epochs evenly spaced; the vertical line is the epoch on the slider; hover for values), and below each model a table of every value.</p>
    <div class="legend">
      <span><span class="sw" style="background:var(--refC)"></span><b>C</b> cos(D<sub>k</sub>, T<sub>C</sub>): family</span>
      <span><span class="sw" style="background:var(--refM)"></span><b>M</b> cos(D<sub>k</sub>, T<sub>M</sub>): mid</span>
      <span><span class="sw" style="background:var(--refF)"></span><b>F</b> cos(D<sub>k</sub>, T<sub>F</sub>): fine</span>
      <span>Bars run from −1 (left) to +1 (right); the line is 0.</span>
    </div>
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

  <div id="images" style="display:grid;gap:18px"></div>

  <footer>
    <p>References from training-set means: G<sub>0</sub> all images, G<sub>C</sub> the image's family, G<sub>CM</sub> its mid node, G<sub>CMF</sub> its leaf.
      T<sub>C</sub> = G<sub>C</sub> − G<sub>0</sub>, T<sub>M</sub> = G<sub>CM</sub> − G<sub>C</sub>, T<sub>F</sub> = G<sub>CMF</sub> − G<sub>CM</sub>.
      Cosine over the 4,096 pixels, no cut-off, so a very small change can still score high: check the ± limit under its map.</p>
    <p>Change maps: red = darker, blue = brighter, each scaled to its own largest |change| (± under it). Reconstructions clipped to [0, 1]; MSE ×10⁻³ against the input.
      __CAVEAT__</p>
    <p>Generator: <code>scripts/hier_exp1_recon_cosine_html.py</code> from <code>__SRC__/data/</code>.</p>
  </footer>
</div>

<script>
const DATA = __DATA__;
const EP = DATA.epochs, D = DATA.d;
const REF = [["C", "--refC"], ["M", "--refM"], ["F", "--refF"]];
const MODEL_NOTE = DATA.notes, SHOWN = DATA.show, PAIR = DATA.codes === "pair";
const fmt = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
let idx = 0, timer = null;

const ticks = document.getElementById("ticks");
EP.forEach(e => { const s = document.createElement("span"); s.textContent = e; ticks.appendChild(s); });
const range = document.getElementById("ep");
range.max = EP.length - 1;

const NS = "http://www.w3.org/2000/svg";
const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
const X0 = 22, X1 = 116, Y0 = 6, Y1 = 70;
const xAt = i => X0 + (X1 - X0) * i / (EP.length - 1);
const yAt = v => Y0 + (Y1 - Y0) * (1 - v) / 2;
const stepName = k => k === 0 ? "∅→1" : PAIR ? `+z${k + 1}` : `${k}→${k + 1}`;
const codeName = k => PAIR ? (k === 0 ? "z1" : `z1,z${k + 1}`) : `z1:${k + 1}`;
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); parent.appendChild(e); return e; }
function drawTrend(svg, im, m, k) {
  svg.setAttribute("aria-label", `${m.toUpperCase()} step ${stepName(k)}: cosine with C, M, F over epochs ` + EP.map(e => `${e}: ` +
    im.epochs[e][m][k].cos.map(v => v.toFixed(2)).join("/")).join(", "));
  for (const v of [1, 0.5, -0.5, -1]) el("line", {x1: X0, x2: X1, y1: yAt(v), y2: yAt(v), class: "grid"}, svg);
  el("line", {x1: X0, x2: X1, y1: yAt(0), y2: yAt(0), class: "zero"}, svg);
  for (const v of [1, 0, -1]) { const t = el("text", {x: X0 - 3, y: yAt(v) + 3, "text-anchor": "end"}, svg); t.textContent = v === 0 ? "0" : (v > 0 ? "+1" : "−1"); }
  [0, EP.length - 1].forEach(i => { const t = el("text", {x: xAt(i), y: 82, "text-anchor": "middle"}, svg); t.textContent = EP[i]; });
  el("line", {x1: xAt(0), x2: xAt(0), y1: Y0, y2: Y1, class: "now"}, svg);
  REF.forEach(([l, c], j) => {
    const pts = EP.map((e, i) => `${xAt(i).toFixed(1)},${yAt(im.epochs[e][m][k].cos[j]).toFixed(1)}`).join(" ");
    el("polyline", {points: pts, fill: "none", stroke: `var(${c})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg);
    EP.forEach((e, i) => el("circle", {cx: xAt(i), cy: yAt(im.epochs[e][m][k].cos[j]), r: 1.8, fill: `var(${c})`}, svg));
  });
  const hit = el("rect", {x: X0 - 4, y: 0, width: X1 - X0 + 8, height: 84, class: "hit"}, svg);
  hit.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect();
    const vx = (ev.clientX - r.left) / r.width * 120;
    const i = Math.max(0, Math.min(EP.length - 1, Math.round((vx - X0) / (X1 - X0) * (EP.length - 1))));
    const cs = im.epochs[EP[i]][m][k].cos;
    tip.innerHTML = `${m.toUpperCase()} ${stepName(k)} · epoch ${EP[i]}<br>` + REF.map(([l, c], j) => `<span style="color:var(${c})">${l}</span> ${fmt(cs[j])}`).join("&nbsp; ");
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (ev.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; });
}
function makeTable(im, m) {
  const t = document.createElement("table");
  t.dataset.model = m;
  t.innerHTML = `<thead><tr><th class="lead">step</th><th class="lead">ref</th>${EP.map(e => `<th data-ep="${e}">ep ${e}</th>`).join("")}</tr></thead><tbody>` +
    Array.from({length: D}, (_, k) => REF.map(([l, c], j) => `<tr><td class="lead">${j === 0 ? stepName(k) : ""}</td>
      <td class="lead" style="color:var(${c})">${l}</td>${EP.map(e => `<td data-ep="${e}">${fmt(im.epochs[e][m][k].cos[j])}</td>`).join("")}</tr>`).join("")).join("") + "</tbody>";
  return t;
}

const host = document.getElementById("images");
const slots = [];
DATA.images.forEach(im => {
  const sec = document.createElement("section");
  sec.className = "img";
  sec.innerHTML = `<div class="imghead"><h2>${im.name} <span class="mono" style="color:var(--muted);font-weight:400">c${im.c} m${im.m} f${im.f} · test #${im.test_index}</span></h2></div>
    <div class="refs">
      <figure><img class="px" src="${im.x}" alt="input image" width="76" height="76"><figcaption>input</figcaption></figure>
      ${im.refs.map((r, j) => `<figure><img class="px" src="${r.src}" alt="reference ${r.label}" width="76" height="76">
        <figcaption><span style="color:var(${REF[j][1]})">${r.label}</span> ±${r.lim.toFixed(2)}</figcaption></figure>`).join("")}
    </div>`;
  const models = {};
  SHOWN.forEach((m, mi) => {
    if (mi) { const dv = document.createElement("div"); dv.className = "divider"; sec.appendChild(dv); }
    const strip = document.createElement("div");
    strip.className = "strip";
    const g = document.createElement("div");
    g.className = "model";
    g.style.setProperty("--cols", D + 1);
    g.innerHTML = `<div class="mname">${m.toUpperCase()}<small>${MODEL_NOTE[m]}</small></div>
      <div class="cell"><img class="px" src="${im.x}" alt="input" width="128" height="128"><div class="cap">input</div></div>` +
      Array.from({length: D}, (_, k) => `<div class="cell" data-k="${k}">
        <img class="px rec" alt="" width="128" height="128"><div class="cap mse"></div>
        <img class="px dmap" alt="" width="128" height="128"><div class="cap lim"></div>
        <div class="cos">${REF.map(([l, c]) => `<div class="cr"><span class="lab" style="color:var(${c})">${l}</span>
          <div class="track"><div class="fill" style="background:var(${c})"></div></div><span class="val"></span></div>`).join("")}</div>
        <svg class="trend" viewBox="0 0 120 84" role="img"></svg><div class="trendcap">cos vs epoch</div>
      </div>`).join("");
    strip.appendChild(g);
    sec.appendChild(strip);
    const det = document.createElement("details");
    det.innerHTML = `<summary>${m.toUpperCase()}: all cosines by epoch (table)</summary><div class="tbl"></div>`;
    sec.appendChild(det);
    [...g.querySelectorAll(".cell[data-k] .trend")].forEach((svg, k) => drawTrend(svg, im, m, k));
    det.querySelector(".tbl").appendChild(makeTable(im, m));
    models[m] = [...g.querySelectorAll(".cell[data-k]")].map(c => ({
      rec: c.querySelector(".rec"), mse: c.querySelector(".mse"), dmap: c.querySelector(".dmap"), lim: c.querySelector(".lim"),
      fills: [...c.querySelectorAll(".fill")], vals: [...c.querySelectorAll(".val")], now: c.querySelector(".now") }));
  });
  host.appendChild(sec);
  slots.push({ im, models });
});

function render() {
  const ep = EP[idx];
  document.getElementById("ep-out").textContent = ep;
  range.value = idx;
  range.setAttribute("aria-valuetext", "epoch " + ep);
  slots.forEach(({ im, models }) => {
    for (const m of SHOWN) {
      im.epochs[ep][m].forEach((st, k) => {
        const c = models[m][k];
        const pre = stepName(k);
        c.rec.src = st.rec;
        c.rec.alt = `${m.toUpperCase()} reconstruction from ${codeName(k)}, epoch ${ep}`;
        c.mse.textContent = `${codeName(k)} · MSE ${st.mse.toFixed(2)}`;
        c.dmap.src = st.d;
        c.dmap.alt = `${m.toUpperCase()} change ${pre}, epoch ${ep}, C ${st.cos[0]}, M ${st.cos[1]}, F ${st.cos[2]}`;
        c.lim.textContent = `${pre} · ±${st.lim.toFixed(2)}`;
        const nowLine = c.rec.closest(".cell").querySelector(".now");
        nowLine.setAttribute("x1", xAt(idx)); nowLine.setAttribute("x2", xAt(idx));
        st.cos.forEach((v, j) => {
          const w = Math.min(1, Math.abs(v)) * 50;
          c.fills[j].style.left = (v >= 0 ? 50 : 50 - w) + "%";
          c.fills[j].style.width = w + "%";
          c.vals[j].textContent = fmt(v);
        });
      });
    }
  });
  markTables();
}
function markTables() {
  const ep = String(EP[idx]);
  document.querySelectorAll("table [data-ep]").forEach(c => c.classList.toggle("cur", c.dataset.ep === ep));
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
render();
</script>
"""


if __name__ == "__main__":
    main()
