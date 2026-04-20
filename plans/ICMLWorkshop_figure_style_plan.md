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

### Fig 1 — Loss-Family Comparison: Cosine Similarity to PCA and LDA (NEW PLAN — 2026-04-20)

**Script**: `weight_symmetry/scripts/plot_fig1_mse_pca_ce_lda.py`
**Layout**: 2 rows × 3 columns, `figsize=(7.0, 5.0)`

#### Step 0 — Prerequisite: Eigenvalue extraction script

Before generating Fig 1, run:
```
python weight_symmetry/scripts/compute_eigenvalues.py
```
This loads the `orderedBoth` synthetic dataset (seed=42), computes PCA and LDA eigenvalues from `X_train`, and saves:
```
files/results/ICMLWorkshop_weightSymmetry2026/eigenvalues/
    pca_eigenvalues.npy      # shape (50,) — raw PCA eigenvalues (variance), decreasing
    lda_eigenvalues.npy      # shape (19,) — generalised LDA eigenvalues, decreasing
    pca_eigenvalues_norm.npy # normalised by pca_eigenvalues[0]
    lda_eigenvalues_norm.npy # normalised by lda_eigenvalues[0]
```

#### Panel grid

| | Col 0: LAE (MSE loss) | Col 1: CE loss | Col 2: DeepLDA (Fisher loss) |
|---|---|---|---|
| **Row 0** | cos-sim to PCA + PCA eigenvalue bars (x ≤ 20) | cos-sim to PCA | cos-sim to PCA |
| **Row 1** | cos-sim to LDA + LDA eigenvalue bars (x ≤ 5) | cos-sim to LDA | cos-sim to LDA |

- All panels: left y-axis = "Max cosine similarity" [0, 1]
- Col 0 only: right y-axis = "Normalised eigenvalue" (faded bars, secondary axis)
- x-axis label: "Prefix size $k$"
- x-axis range: row 0 → [0, 20]; row 1 → [0, 5] (data is clipped to these ranges at plot time)

#### Data sources

| Column | Source file |
|--------|------------|
| Col 0 (LAE) | `exprmnt_2026_04_19__16_00_49/exprmnt_2026_04_19__21_55_11/metrics_raw.npz` |
| Col 1 (CE) | same file as Col 0 |
| Col 2 (Fisher) | fisher: `exprmnt_2026_04_20__01_31_36`; fp_fisher: `exprmnt_2026_04_20__01_44_24`; std_mrl_fisher + prefix_l1_fisher: `exprmnt_2026_04_20__11_33_48` |

**Pending Fisher models** (two jobs still running at plan time): `prefix_l1_fisher`, `std_mrl_fisher` — render as flat `NaN` dashed line with label "(pending)" until result folders are available.

#### Models per column

**Col 0 — LAE (MSE)**
| Legend label | npz key prefix | Color | Style |
|---|---|---|---|
| LAE | `mse_lae` | `#888888` | `--` lw=1.0 |
| MRL | `std_mrl_mse` | `#E07B00` | `-` lw=1.8 |
| FP MRL | `fp_mrl_mse_ortho` | `#009E73` | `-` lw=1.8 |
| PrefixL1 | `prefix_l1_mse` | `#D55E00` | `-.` lw=1.8 |
| NonUniform L2 | `nonuniform_l2` | `#56B4E9` | `-` lw=1.8 |

**Col 1 — CE**
| Legend label | npz key prefix | Color | Style |
|---|---|---|---|
| Normal CE | `normal_ce` | `#888888` | `--` lw=1.0 |
| MRL (CE) | `std_mrl_ce` | `#E07B00` | `-` lw=1.8 |
| FP MRL (CE) | `fp_mrl_ce` | `#009E73` | `-` lw=1.8 |
| PrefixL1 (CE) | `prefix_l1_ce` | `#D55E00` | `-.` lw=1.8 |

**Col 2 — DeepLDA (Fisher)**
| Legend label | npz key prefix | Color | Style |
|---|---|---|---|
| Fisher | `fisher` | `#888888` | `--` lw=1.0 |
| MRL Fisher | `std_mrl_fisher` | `#E07B00` | `-` lw=1.8 |
| FP MRL Fisher | `fp_fisher` | `#009E73` | `-` lw=1.8 |
| PrefixL1 Fisher | `prefix_l1_fisher` | `#D55E00` | `-.` lw=1.8 |

#### Bar overlays (col 0 only)
| Row | Eigenvalue file | Color | Alpha |
|-----|----------------|-------|-------|
| Row 0 | `pca_eigenvalues_norm.npy` | `#89C4E1` (faded blue) | 0.35 |
| Row 1 | `lda_eigenvalues_norm.npy` | `#89C4E1` (faded blue) | 0.35 |

#### Legend placement
- Each panel has its own legend (upper-right), font size 7pt.
- Legend only shows models that have real data (omit flat-NaN pending lines from legend, or append "(pending)" as grey italic).

#### Row / column titles
- Row titles (left margin, rotated): "Cosine sim. to PCA" (row 0), "Cosine sim. to LDA" (row 1)
- Column titles (top): "MSE models", "CE models", "Fisher models"

---

### Fig 2 — PCA Recovery (Exp 1)

**Script**: `weight_symmetry/scripts/plot_fig2_col_alignment.py`  
**Layout**: 2 panels, 1 row, `figsize=(6.5, 3.0)` — fits one paper column

**Panel 1 (left) — Column alignment to PCA**
- Title: "Column alignment to PCA"
- Left y-axis: "Max cosine similarity" — max cosine of each decoder column to any PCA eigenvector
- Right y-axis: "Normalised eigenvalue" — light blue bars (`#89C4E1`, alpha=0.35, zorder=0)
- x-axis: "Prefix size $m$", starts at 0
- Extra line: FP MRL pairwise cosine (`fullprefix_mrl_ortho_paired_cosines`), `#009E73` dashed, lw=1.4
- No legend on this panel (legend lives on right panel)

**Panel 2 (right) — Subspace angle to PCA**
- Title: "Subspace angle to PCA"
- Left y-axis: "Mean principal angle (°)" — mean over canonical angles, lower = better
- Right y-axis: "Normalised eigenvalue gap" — purple bars (`#9B59B6`, alpha=0.30, zorder=0)
  - y-axis capped at `gaps_norm.max() * 1.05` (data-driven, not hardcoded)
- x-axis: "Prefix size $m$", starts at 0, ends at embed_dim
- y-axis top: `max angle across all models * 1.05`

**Models (3 + 1 extra line)**
| Legend label | Data key | Color | Style | Panel |
|---|---|---|---|---|
| LAE | `mse_lae` | `#888888` | `--` | right legend |
| MRL | `standard_mrl` | `#E07B00` | `-` | right legend |
| FP MRL | `fullprefix_mrl_ortho` | `#009E73` | `-` | right legend |
| FP MRL (pairwise sim.) | `fullprefix_mrl_ortho_paired_cosines` | `#009E73` | `--` lw=1.4 | right legend (custom handle) |

**Legend**: upper-right of Panel 2 only — includes custom handle for pairwise sim. line from left panel

**Caption (draft)**:
> **Figure 2.** PCA recovery across prefix sizes for three models on Fashion-MNIST (d=32).
> *(Left)* Max cosine similarity between decoder columns and PCA eigenvectors (solid lines), with pairwise cosine for FP MRL (dashed green), overlaid with the eigenvalue spectrum (blue bars, right axis).
> *(Right)* Mean principal angle between the learned prefix subspace and the PCA subspace (lower = better), overlaid with the normalised eigenvalue gap (purple bars, right axis). FP MRL achieves near-perfect subspace and eigenvector recovery at all scales.

---

### Fig A1 — Appendix: Cluster Visualization + Classification Accuracy (CE Models)

**Script**: `weight_symmetry/scripts/plot_figA1_cluster_accuracy_ce.py`
**Layout**: 4 rows × 4 columns, `figsize=(7.5, 8.5)`

#### Overview

For each of 4 CE-family models × 3 prefix sizes $k \in \{2, 4, 8\}$, show a t-SNE scatter of test embeddings colored by class, with the linear-probe accuracy printed as the panel subtitle.

#### Data source

- Dataset: `orderedBoth` synthetic, seed=42 — use `X_test` / `y_test` (2,000 points, 20 classes)
- Model weights: `exprmnt_2026_04_19__16_00_49/seed42_<tag>_best.pt`
- Embed dim: 50; slice first $k$ dims for each model (reverse for PrefixL1 before slicing)

#### Panel grid

| | Col 0: Normal CE | Col 1: MRL (CE) | Col 2: FP MRL (CE) | Col 3: PrefixL1 (CE) |
|---|---|---|---|---|
| **Row 0** $k=2$  | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D |
| **Row 1** $k=4$  | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D |
| **Row 2** $k=8$  | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D |
| **Row 3** $k=16$ | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D | t-SNE 2-D |

- **Panel title** (top, 8 pt): column header on row 0 only (e.g. "Normal CE")
- **Panel subtitle** (below each scatter, 7 pt italic): `Lin. acc: XX.X%`
- **Row label** (left margin, rotated, 8 pt): "$k = 2$", "$k = 4$", "$k = 8$"
- **Point colors**: `tab20` colormap, one color per class (20 classes), marker size 2, alpha 0.5
- No axis ticks or labels on any panel (t-SNE axes are not interpretable)
- No per-panel legend (20 classes → too cluttered); add a single shared colorbar/legend strip at the bottom if space allows, otherwise omit

#### Model tags (weight file → npz key mapping)

| Column | Weight file tag | npz key prefix |
|---|---|---|
| Normal CE | `normal_ce` | `normal_ce` |
| MRL (CE) | `std_mrl_ce` | `std_mrl_ce` |
| FP MRL (CE) | `fp_mrl_ce` | `fp_mrl_ce` |
| PrefixL1 (CE) | `prefix_l1_ce` | `prefix_l1_ce` (**reverse dims** before slicing) |

#### t-SNE parameters

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
```

Run t-SNE independently for each (model, $k$) combination.

#### Accuracy computation

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Z_train[:, :k], y_train)   # use training embeddings for fitting
acc = clf.score(Z_test[:, :k], y_test)
```

Train split embeddings are also extracted (pass `X_train` through each encoder).

#### Saving

- Output dir: `ICMLWorkshop_weightSymmetry2026/figures/`
- Stem: `figA1_cluster_ce_{fig_stamp}`
- Formats: `.pdf`, `.png`, `.svg`

---

### Fig A1B — Appendix: Cluster Visualization + Classification Accuracy (Fisher Models)

**Script**: `weight_symmetry/scripts/plot_figA1_cluster_accuracy_fisher.py`
**Layout**: 4 rows × 4 columns, `figsize=(7.5, 8.5)`

Same structure as Fig A1 (CE) but for Fisher/LDA models.

#### Key differences from Fig A1

| Property | CE (A1) | Fisher (A1B) |
|---|---|---|
| Model arch | `LinearAEWithHeads` | `LinearAE` (no heads) |
| embed_dim | 50 | 19 (= n_lda = C−1) |
| Seed | 42 | 47 |
| flip_dims | `prefix_l1_ce` | `prefix_l1_fisher` |

#### Models (columns)

| Column | Tag | Weight folder | File |
|---|---|---|---|
| Fisher | `fisher` | `exprmnt_2026_04_20__01_31_36` | `seed47_fisher_best.pt` |
| FP Fisher | `fp_fisher` | `exprmnt_2026_04_20__01_44_24` | `seed47_fp_fisher_best.pt` |
| MRL Fisher | `std_mrl_fisher` | `exprmnt_2026_04_20__11_33_48` | `seed47_std_mrl_fisher_best.pt` |
| PrefixL1 Fisher | `prefix_l1_fisher` | `exprmnt_2026_04_20__11_33_48` | `seed47_prefix_l1_fisher_best.pt` |

#### Rows: $k \in \{2, 4, 8, 16\}$

#### Saving

- Output dir: `ICMLWorkshop_weightSymmetry2026/figures/`
- Stem: `figA1B_cluster_fisher_{fig_stamp}`
- Formats: `.pdf`, `.png`, `.svg`

---

## 5. Figure Registry

Tracks every paper figure: source experiment, run folder, and generated file. Update after each figure is produced.

| Paper Fig | Description | Experiment | Source Run Folder | Generated File (stem) | Date Generated |
|-----------|-------------|------------|-------------------|-----------------------|----------------|
| Fig 1 | 2×3 cosine-sim grid: PCA row + LDA row × MSE/CE/Fisher cols | Exp 2 divergence (LAE/CE) + Exp 2 Fisher | LAE/CE: `exprmnt_2026_04_19__16_00_49/exprmnt_2026_04_20__13_46_30`; Fisher: `exprmnt_2026_04_20__01_31_36` | `fig1_cosine_pca_lda_grid_2026_04_20__13_49_06` | 2026-04-20 |
| Fig 2 | Max cosine (left, + pairwise dashed) + mean principal angle (right), dual-axis | Exp 1 — PCA Subspace Recovery | `exprmnt_2026_04_16__20_54_49/exprmnt_2026_04_19__21_36_49` | `exp1_pca_recovery_2026_04_19__23_21_01` | 2026-04-19 |
| Fig A1 | 4×4 t-SNE cluster grid, CE models, k=2,4,8,16 | Exp 2 divergence (CE) | `exprmnt_2026_04_19__16_00_49` | `figA1_cluster_ce_2026_04_20__16_02_43` | 2026-04-20 |
| Fig A1B | 4×4 t-SNE cluster grid, Fisher models, k=2,4,8,16 | Exp 2 Fisher | `01_31_36` (fisher), `01_44_24` (fp_fisher), `11_33_48` (mrl+l1) | `figA1B_cluster_fisher_2026_04_20__16_19_44` | 2026-04-20 |

---

## 6. Checklist Before Submission

- [ ] All figures use the same font family and size
- [ ] All methods have consistent color across all figures
- [ ] All figures saved as `.pdf`, `.png`, `.svg` with timestamp
- [ ] PDF files embed fonts (check with `pdffonts fig.pdf` — no Type3)
- [ ] Figure width ≤ 6.5 in (one column)
- [ ] Legend entries match exactly the method names in `METHOD_STYLE`
- [ ] All output files are in the ICMLWorkshop figures folder
