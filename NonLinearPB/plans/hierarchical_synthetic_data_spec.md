# Experiment 1 — Hierarchical Synthetic Image Dataset

## Goal

Generate a **nonlinear, image-based, three-level hierarchy** with known labels and a measured reconstruction-cost ordering. This dataset tests whether a nonlinear Matryoshka autoencoder (MRL-AE) learns stable **coarse-to-fine groupings**, not simply whether it reconstructs coarse features first.

The ground-truth tree has **3 coarse branches × 3 middle branches × 3 fine branches = 27 leaves**. All labels are retained for evaluation but **never provided to the autoencoder during training**.

## 1. Dataset size and splits

- Image format: **64 × 64 grayscale**, stored as float32 in `[0, 1]` for model training (optionally uint8 PNG for inspection).
- **2,000 images per leaf**: `27 × 2,000 = 54,000 images`.
- Stratified split within every leaf: **1,600 train / 200 validation / 200 test**.
- Totals: **43,200 train / 5,400 validation / 5,400 test**.
- Generate a separate, fixed evaluation set of matched intervention pairs from the **test** split only; use no test data to tune generation parameters.
- If the experiment is too easy or too noisy, modify the generator using **training/validation** diagnostics, then freeze it and generate a fresh final test set.

## 2. Ground-truth hierarchy

Each image has three discrete labels:

- `coarse = c ∈ {0,1,2}`: which overall object family exists;
- `mid = m ∈ {0,1,2}`: which substantial *family-specific* refinement exists within coarse family `c`;
- `fine = f ∈ {0,1,2}`: which *parent-specific* local detail exists within `(c,m)`.

There are 3 coarse nodes, 9 mid nodes, and 27 leaf nodes. Each leaf is a **distribution** of images, not one fixed template.

**Proposed visual families (initial design, to be calibrated):**

| Coarse node | Overall structure | Mid-level refinements (3 alternatives) | Fine-level refinements (3 per mid, parent-anchored) |
|---|---|---|---|
| `c=0` | Horizontal double-lobed body connected by a bar | Connector profile: tapered, straight, bowed | Three distinct local junction/notch shapes placed at a position determined by the chosen connector |
| `c=1` | Vertical ring-and-stem body | Ring-stem attachment: top, central, offset | Three distinct small attachment-tip geometries, placed relative to that specific attachment |
| `c=2` | Diagonal bent/boomerang-like body | Bend profile: rounded, sharp, stepped | Three distinct short edge cutouts or bulges anchored to the bend defined by `m` |

These are **design candidates**, not assumed to have correct loss ratios. Render them, inspect them, and tune geometric sizes until the measured cost ladder holds. Use coherent silhouettes (antialiased shapes); keep major objects fully inside the canvas.

**Critical nesting rule:** Do not add an identical feature at a fixed absolute pixel coordinate to every branch. Implement each mid variation using its own coarse parent geometry, and each fine variation using the geometry/anchor provided by its `(c,m)` parent. A single numerical label `f=1` does **not** mean the same visual feature across all branches.

### Rendering algorithm (conceptual pseudocode)

```python
def render(c, m, f, nuisance_seed, config):
    canvas = zeros((64, 64))
    parent_geometry = draw_coarse_family(canvas, family=c, config=config)
    mid_geometry = refine_parent(canvas, parent_geometry, c=c, mid=m, config=config)
    add_local_detail(canvas, mid_geometry, c=c, m=m, fine=f, config=config)
    return apply_small_nuisance(canvas, nuisance_seed, config)
```

Use geometry-aware rasterization (e.g., Pillow/OpenCV), then downsample with antialiasing if rendering at higher resolution. The image generator is nonlinear because family-specific geometry and local changes are conditionally applied, not a linear sum of independent vectors.

## 3. Ground-truth reconstruction-cost ladder

**The desired ordering is**

`Δ_coarse > Δ_mid > Δ_fine > Δ_nuisance`.

For a fixed base configuration and *the same nuisance seed*, construct matched pairs which change **only one hierarchy label**:

- Coarse: `(c,m,f)` vs `((c+1) mod 3,m,f)`; sweep all valid coarse alternatives.
- Mid: `(c,m,f)` vs `(c,(m+1) mod 3,f)`; sweep all alternatives within the same `c`.
- Fine: `(c,m,f)` vs `(c,m,(f+1) mod 3)`; sweep all alternatives within the same `(c,m)`.
- Nuisance: same `(c,m,f)`, different nuisance seeds.

Define **pixel MSE cost** for two images:

`D(x, x') = mean_pixels((x - x')**2)`.

Define `Δ_level` as the average matched-pair cost over **all parent branches, all sibling choices, and many nuisance replicates**. Report the **distribution** of pair costs as well as the mean; a global mean can hide expensive/cheap branches. For noise-only pairs, record `Δ_nuisance`.

**Costs are nested.** Because mid and fine features are parent-specific, a coarse-swap pair also changes mid and fine geometry, and a mid-swap pair also changes the fine detail. This is intended (it is what a nested hierarchy means); report ratios with this stated rather than as independent per-level costs.

**Initial tuning target (not a fixed law):** approximately `Δ_coarse : Δ_mid : Δ_fine ≈ 4 : 2 : 1`, with `Δ_nuisance` substantially below `Δ_fine` (suggested mean `< 0.25 × Δ_fine`). Calibrate these on a separate pilot sample; do not treat a visual pixel-count estimate as evidence of the ratio.

**Important caveat:** Mean pairwise pixel cost measures the difference between siblings, not exactly the *minimum reconstruction error incurred when a level is missing*. For a stronger check, additionally compute the mean image within each group and quantify the extra MSE of predicting with a parent-level mean rather than a child-level mean (conditional-variance reduction). Verify the same coarse > mid > fine ordering for that measure. This better reflects the AE's squared reconstruction objective.

## 4. Nuisance variation

Introduce *small*, independent variation within every leaf:

- **No whole-object translation in v1.** Images stay centered; small shifts are deferred to a later experiment (a ±1 px shift of a large silhouette changes the whole outline and can exceed `Δ_fine`, and it blurs group means).
- Mild brightness/contrast perturbation.
- Very low-amplitude pixel noise (pilot `σ ≈ 0.01`; independent noise contributes `≈ 2σ²` per pixel to pair MSE over all 4,096 pixels).
- Small geometric jitter of part shapes that preserves the parent/child labels and never moves the whole object, e.g. for `c=0`: lobe radius, connector thickness, connector length. Frozen at ±0.2 px in v2 (±0.3 px in v1 made 9% of single fine pairs cheaper than the 99th-percentile noise pair).

Geometric jitter still changes pixels along part outlines, so its cost must be **measured, not assumed**. Rough scale: a 0.5 px radius change on a radius-10 lobe costs `≈ 63 × 0.5² / 4096 ≈ 0.004` MSE, comparable to a ~20-pixel fine detail. Expect sub-pixel jitter amplitudes (pilot ≈ ±0.2 px), which requires rendering at higher resolution (e.g. 256 × 256) and antialiased downsampling to 64 × 64. Jitter that displaces parts (e.g. connector length moving the lobes) is the most expensive; keep it smallest. Fine details are anchored to the *jittered* parent geometry so nesting is preserved.

Calibration: measure each nuisance component separately (radius, thickness, length, brightness, noise) on train/validation only, then scale so the combined `Δ_nuisance ≈ 0.1–0.2 × Δ_fine`.

Do **not** simply add many pixels of noise: the measured nuisance-only cost must remain below the fine-level cost. Use matched nuisance seeds when estimating label-change costs, so nuisance variation does not contaminate them.

Use independent random generators for splits and record all seeds. Prevent identical rendered samples from appearing in multiple splits.

## 5. Generator validation before any AE training

1. **Contact sheet:** 3 coarse rows, each with its 3 mid variants and their 3 fine leaves; show several samples per leaf.
2. **Cost ladder plot/table:** report `Δ_coarse`, `Δ_mid`, `Δ_fine`, `Δ_nuisance`, ratios, confidence intervals, and branch-level ranges.
3. **Conditional-variance check:** confirm that knowing coarse/mid/fine successively reduces expected pixel MSE in the desired order.
4. **Within-leaf diversity:** show that samples are not duplicates and nuisance is visible but does not obscure fine differences.
5. **Train/test check:** all 27 leaves equally represented in every split and no reused random seeds or exact duplicate images.
6. **Leakage probe (diagnostic):** check whether simple classifiers can predict a fine label across *unrelated* parent branches from a fixed, shared feature. Some information may be visually predictive without following the hierarchy; document this rather than assuming perfect non-leakage.

If costs fail, adjust family contrast/area, mid geometry size, fine detail prominence, and nuisance amplitude. Recompute all costs after every change. **Freeze the renderer and parameters before primary comparisons.**

## 6. Outputs

Output layout. The frozen dataset lives outside the git repo at `<project root>/data/synthetic_data/hierarchical_images_<version>/`. Pilot/tuning runs write to `<project root>/files/results/hier_synthetic_pilot/<run>/`; only frozen versions go in `data/`, and any post-freeze generator change gets a new version.

- **v1** (2026-11-02, superseded): shape jitter ±0.3 px; dumbbell "tapered"/"straight" and ring "stem through"/"stem below" put identical fine details in the same place (leakage probe transfer 1.00).
- **v2** (2026-11-02, current): those two parents (dumbbell "straight", ring "stem below") get their own fine-detail sets; shape jitter ±0.2 px; leakage probe stored in `leakage_probe/`. See `NonLinearPB/NonLinearPBMeetingNotes/2026_11_02_hierarchical_dataset_v2_report.md`.

```text
synthetic_data/hierarchical_images_<version>/
  config.json                 # render settings, cost targets, version, seeds
  train.npz                   # images [43200, 64, 64], coarse, mid, fine, leaf_id
  val.npz                     # images [5400, 64, 64], labels
  test.npz                    # images [5400, 64, 64], labels
  metadata.csv                # sample_id, split, coarse, mid, fine, leaf_id, nuisance_seed
  cost_validation.csv         # cost by level, branch, pair, mean, variance
  conditional_variance.csv    # reconstruction-cost improvements per hierarchy level
  contact_sheet.png           # example images for all 27 leaves
  README.md                   # generator version, how to regenerate exactly
```

For memory/storage efficiency, store images as uint8 internally if desired and convert to float32 `[0,1]` when loading. Keep deterministic identifiers and avoid array/dataframe ordering differences across runs.

## 7. What this enables (Experiment 1, later)

- Train the **same nonlinear architecture** as standard AE and full-prefix MRL-AE; change only reconstruction loss.
- Train on **images only**, no tree labels.
- Freeze encoders, analyze their prefix geometry, compare learned grouping against the known **3 → 9 → 27** ground-truth tree and across random seeds.
- Key scientific test: does increasing prefix length **refine** previous groupings instead of redrawing them? Does that organization recur across seeds?
- Do not presume that `k=1,2,3` must recover levels exactly. A 1-D continuous representation can encode many groups, and reconstruction optimization does not formally guarantee the semantic tree.

## 8. Design cautions

- Large pixel cost does not automatically imply semantic grouping; hence evaluate both the ordering and the *nesting of partitions*.
- Pixel-space MSE is the specified cost, and **v1 AE training also uses MSE** (not BCE) so the calibrated ladder matches the training objective. If a later run uses BCE or a different reconstruction loss, **recalibrate costs using that actual loss**.
- Use separate seed streams per split. The final test set (and its intervention pairs) is generated with a fresh seed stream **only after** the generator is frozen; record that seed in `config.json`.
- A standard AE's full 10-dimensional reconstruction and the MRL-AE's prefix reconstructions are different objectives by design; keep optimizer, training data, architecture, and full latent width otherwise matched.
- This specification describes **how to generate** the dataset; it does not claim the images have already been generated or that the desired cost ratios are achieved before measurement.
