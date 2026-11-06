"""Epoch-slider page: the z1-only reconstruction and its alignment over epochs, all seeds side by side.

Conda environment: mrl_env_cuda12
Reads <src>/data/<img>_e<ep>.npz (hier_exp1_pair_change.py or hier_exp1_prefix_change.py; recon[:, :, 1] is decode([z1, 0..]),
D[:, :, 0] = decode([z1, 0..]) - decode(0), alignment[:, :, :, 0] = its cosine with T_C, T_M, T_F).
Per image: one row per seed; per model: input | z1 reconstruction at the slider's epoch | cos(D_1, T_C/T_M/T_F) vs epoch
(current epoch marked, values printed). Writes <src>/z1_seeds/z1_seeds.html.

Usage: python scripts/hier_exp1_z1_seeds_html.py --run R --src DIR [--images c1m0f0 c0m2f0] [--show pair mrl] [--desc "..."]
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
_ap.add_argument("--run", type=Path, default=Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_11_05__13_45_22"))
_ap.add_argument("--src", type=Path, help="folder with data/ and metadata.json (default <run>/figures/pair_change)")
_ap.add_argument("--images", nargs="+", default=["c1m0f0", "c1m2f0", "c0m2f0", "c2m2f0"])
_ap.add_argument("--show", nargs="+", default=["pair", "mrl"])
_ap.add_argument("--epochs", type=int, nargs="+", default=[2, 3, 4, 5, 10, 15, 20, 25])
_ap.add_argument("--desc", default="pair objective vs 5-d MRL, lr 6e-5")
_args = _ap.parse_args()
SRC = _args.src or _args.run / "figures" / "pair_change"
OUT = SRC / "z1_seeds"
NOTES = {"mrl": "trained on every prefix", "ae": "trained on the full code", "pair": "trained on {z1} and each {z1, zj}"}
SCALE = 2
CMAP = matplotlib.colormaps["RdBu"]          # red = darker, blue = brighter, white = 0


def uri_div(a):
    """Signed map -> RdBu image with its own symmetric limit; returns (uri, limit)."""
    lim = float(np.abs(a).max()) or 1e-12
    rgb = (CMAP(np.clip(0.5 + 0.5 * a / lim, 0, 1))[..., :3] * 255).round().astype(np.uint8)
    img = Image.fromarray(rgb).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), round(lim, 3)


def uri(a):
    g = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    img = Image.fromarray(g).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "metadata.json").read_text())
    images = []
    for name in _args.images:
        info = meta["images"][name]
        d0 = np.load(SRC / "data" / f"{name}_e{_args.epochs[0]:03d}.npz")
        models = [str(m) for m in d0["models"]]
        n_seeds = d0["recon"].shape[1]
        entry = {"key": name, "name": info["name"], "c": info["c"], "m": info["m"], "f": info["f"],
                 "test_index": info["test_index"], "x": uri(d0["x"]), "models": {},
                 "refs": [{"label": lab, "src": u, "lim": lim} for lab, (u, lim) in
                          zip(("T_C", "T_M", "T_F"), (uri_div(d0["T"][j]) for j in range(3)))]}
        for m in _args.show:
            mi = models.index(m)
            seeds = []
            for s in range(n_seeds):
                rec, cos, mse = [], [], []
                for ep in _args.epochs:
                    d = np.load(SRC / "data" / f"{name}_e{ep:03d}.npz")
                    z1 = d["recon"][mi, s, 1]
                    rec.append(uri(z1))
                    cos.append([round(float(v), 3) for v in d["alignment"][mi, s, :, 0]])
                    mse.append(round(float(((z1 - d["x"]) ** 2).mean()) * 1e3, 2))
                seeds.append({"rec": rec, "cos": cos, "mse": mse})
            entry["models"][m] = seeds
        images.append(entry)
    payload = {"epochs": _args.epochs, "images": images, "show": _args.show, "notes": {m: NOTES.get(m, "") for m in _args.show}}
    html = TEMPLATE.replace("__DATA__", json.dumps(payload)).replace("__RUN__", _args.run.name).replace("__DESC__", _args.desc) \
                   .replace("__SRC__", str(SRC.relative_to(_args.run)) if SRC.is_relative_to(_args.run) else str(SRC))
    path = OUT / "z1_seeds.html"
    path.write_text(html)
    (OUT / "z1_seeds.json").write_text(json.dumps({"src": str(SRC), "images": _args.images, "models": _args.show,
        "epochs": _args.epochs, "cos_D1": {im["key"]: {m: [sd["cos"] for sd in v] for m, v in im["models"].items()} for im in images},
        "layout": "[seed][epoch][ref C, M, F]; D_1 = decode([z1,0..]) - decode(0)"}, indent=1) + "\n")
    print(path, f"{len(html) / 1e6:.2f} MB")


TEMPLATE = r"""<title>z1 Across Seeds</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: per image a card; one row per seed; per model three cells (input, z1 reconstruction, cos-vs-epoch chart). */
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
.wrap { max-width: 1240px; margin: 0 auto; padding-block: 20px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 18px; }
h1 { font-size: 1.4rem; font-weight: 600; margin: 0; text-wrap: balance; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 76ch; }
code, .mono { font-family: var(--mono); font-size: 0.85em; }
header { display: grid; gap: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; color: var(--muted); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
.bar label { font-weight: 600; }
.bar output { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 2ch; font-size: 1.1rem; }
.slider { flex: 1 1 260px; min-width: 0; display: grid; gap: 2px; }
.slider input { width: 100%; accent-color: var(--accent); margin: 0; }
.ticks { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
button { font: inherit; font-size: 0.85rem; border: 1px solid var(--line); background: var(--plate); color: var(--ink); border-radius: 6px; padding: 4px 12px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.hint { font-size: 0.75rem; color: var(--muted); }
section.img { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.scroll { overflow-x: auto; }
.tbl { display: grid; grid-template-columns: 52px repeat(var(--nm), 72px 72px minmax(170px, 1fr)); gap: 8px 10px; align-items: center; min-width: calc(52px + var(--nm) * 340px); }
.th { font-size: 0.78rem; color: var(--muted); font-weight: 500; text-align: center; }
.th.model { font-size: 0.95rem; color: var(--ink); font-weight: 600; text-align: left; border-bottom: 1px solid var(--line); padding-bottom: 2px; }
.th.model small { font-weight: 400; color: var(--muted); font-size: 0.72rem; margin-left: 6px; }
.seed { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }
img.px { image-rendering: pixelated; display: block; width: 72px; height: 72px; border-radius: 2px; background: var(--plate); }
.cap { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }
.chart { display: grid; gap: 2px; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart text { font-family: var(--mono); font-size: 9px; fill: var(--muted); }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 2; }
.chart .now { stroke: var(--accent); stroke-width: 1.5; opacity: 0.5; }
.chart .hit { fill: transparent; cursor: crosshair; }
.vals { font-family: var(--mono); font-size: 0.7rem; font-variant-numeric: tabular-nums; display: flex; gap: 10px; }
.tip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; font-family: var(--mono); font-size: 0.72rem; line-height: 1.45; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); }
.refs { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
.refs figure { margin: 0; display: grid; gap: 3px; justify-items: center; }
.refs img { width: 88px; height: 88px; }
.refs figcaption { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); text-align: center; line-height: 1.35; }
.refnote { font-size: 0.78rem; color: var(--muted); max-width: 46ch; align-self: center; }
footer { font-size: 0.8rem; color: var(--muted); display: grid; gap: 6px; }
</style>

<div class="wrap">
  <header>
    <h1>z<sub>1</sub> alone, every seed</h1>
    <p>Run <code>__RUN__</code> (__DESC__). For each seed and model: the input, the reconstruction from z<sub>1</sub> alone
      (rest of the code zero) at the epoch on the slider, and how the first step lines up with the hierarchy over all epochs:
      cos(D<sub>1</sub>, T) with D<sub>1</sub> = decode([z<sub>1</sub>, 0, …]) − decode(all-zero code).</p>
    <div class="legend">
      <span><span class="sw" style="background:var(--refC)"></span><b>C</b> family (T<sub>C</sub>)</span>
      <span><span class="sw" style="background:var(--refM)"></span><b>M</b> mid (T<sub>M</sub>)</span>
      <span><span class="sw" style="background:var(--refF)"></span><b>F</b> fine (T<sub>F</sub>)</span>
      <span>Charts: −1 to +1, epochs evenly spaced, vertical line = slider epoch, hover for values.</span>
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
    <p>References from training-set means: T<sub>C</sub> = G<sub>C</sub> − G<sub>0</sub>, T<sub>M</sub> = G<sub>CM</sub> − G<sub>C</sub>,
      T<sub>F</sub> = G<sub>CMF</sub> − G<sub>CM</sub>. Cosine over the 4,096 pixels, signed, no cut-off. Reconstructions clipped to [0, 1];
      MSE ×10⁻³ against the input. The all-zero code decodes to a fixed image per seed and epoch, so for an image of that same family
      the first step has little family change left to make.</p>
    <p>Generator: <code>scripts/hier_exp1_z1_seeds_html.py</code> from <code>__SRC__/data/</code>.</p>
  </footer>
</div>

<script>
const DATA = __DATA__;
const EP = DATA.epochs, SHOWN = DATA.show;
const REF = [["C", "--refC"], ["M", "--refM"], ["F", "--refF"]];
const fmt = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const NS = "http://www.w3.org/2000/svg";
const X0 = 22, X1 = 196, Y0 = 6, Y1 = 66, W = 200, H = 80;
const xAt = i => X0 + (X1 - X0) * i / (EP.length - 1);
const yAt = v => Y0 + (Y1 - Y0) * (1 - v) / 2;
let idx = 0, timer = null;
const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); parent.appendChild(e); return e; }

const ticks = document.getElementById("ticks");
EP.forEach(e => { const s = document.createElement("span"); s.textContent = e; ticks.appendChild(s); });
const range = document.getElementById("ep");
range.max = EP.length - 1;

function chart(svg, sd, label) {
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${label}: cosine of the first step with C, M, F over epochs ` + EP.map((e, i) => `${e}: ` + sd.cos[i].map(v => v.toFixed(2)).join("/")).join(", "));
  for (const v of [1, 0.5, -0.5, -1]) el("line", {x1: X0, x2: X1, y1: yAt(v), y2: yAt(v), class: "grid"}, svg);
  el("line", {x1: X0, x2: X1, y1: yAt(0), y2: yAt(0), class: "zero"}, svg);
  for (const v of [1, 0, -1]) { const t = el("text", {x: X0 - 3, y: yAt(v) + 3, "text-anchor": "end"}, svg); t.textContent = v === 0 ? "0" : (v > 0 ? "+1" : "−1"); }
  EP.forEach((e, i) => { const t = el("text", {x: xAt(i), y: 78, "text-anchor": "middle"}, svg); t.textContent = e; });
  const now = el("line", {x1: xAt(0), x2: xAt(0), y1: Y0, y2: Y1, class: "now"}, svg);
  REF.forEach(([l, c], j) => {
    el("polyline", {points: EP.map((e, i) => `${xAt(i).toFixed(1)},${yAt(sd.cos[i][j]).toFixed(1)}`).join(" "),
      fill: "none", stroke: `var(${c})`, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg);
    EP.forEach((e, i) => el("circle", {cx: xAt(i), cy: yAt(sd.cos[i][j]), r: 2, fill: `var(${c})`}, svg));
  });
  const hit = el("rect", {x: X0 - 4, y: 0, width: X1 - X0 + 8, height: H, class: "hit"}, svg);
  hit.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect();
    const vx = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(EP.length - 1, Math.round((vx - X0) / (X1 - X0) * (EP.length - 1))));
    tip.innerHTML = `${label} · epoch ${EP[i]}<br>` + REF.map(([l, c], j) => `<span style="color:var(${c})">${l}</span> ${fmt(sd.cos[i][j])}`).join("&nbsp; ");
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (ev.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; });
  return now;
}

const cells = [];
const host = document.getElementById("images");
DATA.images.forEach(im => {
  const sec = document.createElement("section");
  sec.className = "img";
  const REFTXT = ["family: G<sub>C</sub> − G<sub>0</sub>", "mid: G<sub>CM</sub> − G<sub>C</sub>", "fine: G<sub>CMF</sub> − G<sub>CM</sub>"];
  sec.innerHTML = `<h2>${im.name} <span class="mono" style="color:var(--muted);font-weight:400">c${im.c} m${im.m} f${im.f} · test #${im.test_index}</span></h2>
    <div class="refs">
      <figure><img class="px" src="${im.x}" alt="input image" width="88" height="88"><figcaption>input</figcaption></figure>
      ${im.refs.map((r, j) => `<figure><img class="px" src="${r.src}" alt="ground-truth change ${r.label}, limit ${r.lim}" width="88" height="88">
        <figcaption><span style="color:var(${REF[j][1]})">${r.label.replace("_", "")}</span> ±${r.lim.toFixed(2)}<br>${REFTXT[j]}</figcaption></figure>`).join("")}
      <p class="refnote">Ground-truth changes for this image's leaf, from training-set mean images (red = darker, blue = brighter, each scaled to its own ± limit). The charts below compare the first step D<sub>1</sub> with these.</p>
    </div>`;
  const sc = document.createElement("div"); sc.className = "scroll";
  const g = document.createElement("div"); g.className = "tbl"; g.style.setProperty("--nm", SHOWN.length);
  let html = `<div></div>` + SHOWN.map(m => `<div class="th model" style="grid-column: span 3">${m.toUpperCase()}<small>${DATA.notes[m]}</small></div>`).join("");
  html += `<div></div>` + SHOWN.map(() => `<div class="th">input</div><div class="th">z1 only</div><div class="th">first step: cos vs epoch</div>`).join("");
  const nSeeds = im.models[SHOWN[0]].length;
  for (let s = 0; s < nSeeds; s++) {
    html += `<div class="seed">seed ${s}</div>` + SHOWN.map(m => `<div><img class="px" src="${im.x}" alt="input" width="72" height="72"></div>
      <div data-m="${m}" data-s="${s}" class="z1"><img class="px" alt="" width="72" height="72"><div class="cap"></div></div>
      <div class="chart" data-m="${m}" data-s="${s}"><svg></svg><div class="vals"></div></div>`).join("");
  }
  g.innerHTML = html;
  sc.appendChild(g); sec.appendChild(sc); host.appendChild(sec);
  g.querySelectorAll(".chart").forEach(c => {
    const m = c.dataset.m, s = +c.dataset.s, sd = im.models[m][s];
    const now = chart(c.querySelector("svg"), sd, `${m.toUpperCase()} seed ${s}`);
    const z = g.querySelector(`.z1[data-m="${m}"][data-s="${s}"]`);
    cells.push({ sd, m, s, now, img: z.querySelector("img"), cap: z.querySelector(".cap"), vals: c.querySelector(".vals") });
  });
});

function render() {
  const ep = EP[idx];
  document.getElementById("ep-out").textContent = ep;
  range.value = idx;
  range.setAttribute("aria-valuetext", "epoch " + ep);
  cells.forEach(c => {
    c.img.src = c.sd.rec[idx];
    c.img.alt = `${c.m.toUpperCase()} seed ${c.s}: reconstruction from z1 alone, epoch ${ep}`;
    c.cap.textContent = `MSE ${c.sd.mse[idx].toFixed(2)}`;
    c.now.setAttribute("x1", xAt(idx)); c.now.setAttribute("x2", xAt(idx));
    c.vals.innerHTML = REF.map(([l, col], j) => `<span><span style="color:var(${col})">${l}</span> ${fmt(c.sd.cos[idx][j])}</span>`).join("");
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
render();
</script>
"""


if __name__ == "__main__":
    main()
