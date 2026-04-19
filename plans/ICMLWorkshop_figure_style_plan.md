# Figure Generation Plan — ICML Workshop 2026 (Weight Symmetry)

**Date**: 2026-04-19  
**Output folder**: `/home/argha/Mat_embedding_hyperbole/files/results/ICMLWorkshop_weightSymmetry2026/figures/`  
**Paper deadline**: April 24, 2026

---

## 1. Global Style Spec

All figures must import and apply a shared style config. No per-figure overrides.

### 1a. Font

| Element | Size | Notes |
|---------|------|-------|
| Family | `serif` | matches LaTeX default |
| Base / axis labels | `9 pt` | `font.size`, `axes.labelsize`, `axes.titlesize` |
| Tick labels | `8 pt` | `xtick.labelsize`, `ytick.labelsize` |
| Legend | `8 pt` | `legend.fontsize` (use rcParam, not per-call) |
| Annotations / italic notes | `7 pt` | pass `fontsize=7` to `ax.annotate()` only |

Use `matplotlib.rcParams` to set globally, never `fontsize=` per call (except annotations).

```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "pdf.fonttype":      42,   # embeds fonts for camera-ready
    "ps.fonttype":       42,
})
```

### 1b. Figure Size

| Use case | Width × Height (inches) |
|----------|------------------------|
| Single-panel (one column) | 3.5 × 2.8 |
| Two-panel side-by-side (one column) | 6.5 × 3.0 |
| Three-panel side-by-side | 7.0 × 2.5 |
| Bar chart | 3.5 × 2.5 |

Use `fig, ax = plt.subplots(figsize=(...))` — never `plt.figure()` alone.

### 1c. Axes

- Grid: `ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='gray')`
- Spines: keep all four; no `ax.spines[...].set_visible(False)` unless deliberate
- Line width: `1.8` for main curves, `1.0` for baselines/reference
- Marker size: `4` if used

---

## 2. Color Palette (per-method, consistent across all figures)

Every model gets one color and one linestyle. Never reuse a color for a different model.

| Method | Color (hex) | Linestyle | Marker |
|--------|------------|-----------|--------|
| LAE | `#888888` | `--` | none |
| MRL | `#E07B00` | `-` | none |
| Full prefix MRL | `#009E73` | `-` | none |
| Full prefix MRL + ortho | `#009E73` | `-` | `s` |
| Full-prefix MRL (CE) | `#D55E00` | `-` | none |
| Standard MRL (CE) | `#CC79A7` | `--` | none |
| PCA + linear probe | `#0072B2` | `:` | none |

Color scheme is colorblind-safe (Wong 2011 palette).

Define once in a shared helper file:

```python
# weight_symmetry/plotting/style.py

METHOD_STYLE = {
    "LAE":                     dict(color="#888888", ls="--",  lw=1.0),
    "MRL":                     dict(color="#E07B00", ls="-",   lw=1.8),
    "Full prefix MRL":         dict(color="#009E73", ls="-",   lw=1.8),
    "Full prefix MRL + ortho": dict(color="#009E73", ls="-",   lw=1.8, marker="s", ms=4),
    "Full-prefix MRL (CE)":    dict(color="#D55E00", ls="-",   lw=1.8),
    "Standard MRL (CE)":       dict(color="#CC79A7", ls="--",  lw=1.8),
    "PCA + linear probe":      dict(color="#0072B2", ls=":",   lw=1.0),
}
```

### Bar overlay colors (not model lines)

| Overlay | Color (hex) | Alpha |
|---------|------------|-------|
| Eigenvalue spectrum bars | `#89C4E1` (faded blue) | 0.35 |
| Eigenvalue gap bars | `#9B59B6` (purple) | 0.30 |

---

## 3. Saving Convention

### 3a. File naming

Pattern: `{exp_tag}_{description}_{YYYY_MM_DD__{HH_MM_SS}.{ext}`

Timestamp is set once per run:
```python
import time
fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
```

### 3b. Save helper

```python
def save_fig(fig, base_name, fig_stamp, out_dir=None):
    """Save figure as pdf, png, and svg with timestamp suffix."""
    out_dir = out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{base_name}_{fig_stamp}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    return stem
```

All figures go to:
```
/home/argha/Mat_embedding_hyperbole/files/results/ICMLWorkshop_weightSymmetry2026/figures/
```

---

## 4. Figure Specs

### Fig 2 — PCA Recovery (Exp 1)

**Script**: `weight_symmetry/scripts/plot_fig2_col_alignment.py`  
**Layout**: 2 panels, 1 row, `figsize=(6.5, 3.0)` — fits one paper column

**Panel 1 (left) — Cosine similarity + eigenvalue spectrum**
- Left y-axis: "Cosine similarity" (mean max cosine similarity, higher = better)
- Right y-axis: "Normalised eigenvalue" — light blue bars (`#89C4E1`, alpha=0.35, zorder=0)
- x-axis: "Prefix size $m$"

**Panel 2 (right) — Principal angle + eigenvalue gap**
- Left y-axis: "Principal angle (°)" (mean principal angle, lower = better)
- Right y-axis: "Normalised eigenvalue gap" — purple bars (`#9B59B6`, alpha=0.30, zorder=0)
  - Cap y-axis at 0.12 (dims 2–3 have large gaps ~0.35–0.40 that are real but compress the scale)
  - Shade 4 smallest-gap dims with faint purple strip (`alpha=0.08`)

**Models (3 only)**
| Legend label | Data key | Color | Style |
|---|---|---|---|
| LAE | `mse_lae` | `#888888` | `--` |
| MRL | `standard_mrl` | `#E07B00` | `-` |
| Full prefix MRL | `fullprefix_mrl_ortho` | `#009E73` | `-` |

**Legend**: upper-right of Panel 2 (principal angle panel)  
**No panel titles** — caption in LaTeX

**Caption (draft)**:
> **Figure 2.** PCA recovery across prefix sizes for three models on Fashion-MNIST (d=32).
> *(Left)* Mean max cosine similarity between decoder columns and PCA eigenvectors (higher = better), overlaid with the eigenvalue spectrum (blue bars, right axis).
> *(Right)* Mean principal angle between the learned prefix subspace and the PCA subspace (lower = better), overlaid with the normalised eigenvalue gap (purple bars, right axis, capped at 0.12). Full prefix MRL achieves near-perfect subspace and eigenvector recovery at all scales.

---

## 5. Figure Registry

Tracks every paper figure: source experiment, run folder, and generated file. Update after each figure is produced.

| Paper Fig | Description | Experiment | Source Run Folder | Generated File (stem) | Date Generated |
|-----------|-------------|------------|-------------------|-----------------------|----------------|
| Fig 2 | Cosine similarity (left) + principal angle (right), dual-axis | Exp 1 — PCA Subspace Recovery | `exprmnt_2026_04_16__20_54_49` | `exp1_pca_recovery_2026_04_19__14_19_37` | 2026-04-19 |

---

## 6. Checklist Before Submission

- [ ] All figures use the same font family and size
- [ ] All methods have consistent color across all figures
- [ ] All figures saved as `.pdf`, `.png`, `.svg` with timestamp
- [ ] PDF files embed fonts (check with `pdffonts fig.pdf` — no Type3)
- [ ] Figure width ≤ 6.5 in (one column)
- [ ] Legend entries match exactly the method names in `METHOD_STYLE`
- [ ] All output files are in the ICMLWorkshop figures folder
