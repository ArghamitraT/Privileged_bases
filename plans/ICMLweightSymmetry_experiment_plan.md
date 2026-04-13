# Experiment Plan — ICML 2026 Weight Symmetry Paper
**Deadline: April 24**
**Setting: Linear autoencoders on MNIST (p=100 after PCA projection, d=50)**
**Seeds: 5 random seeds for all experiments**
**Models: MSE LAE, Oftadeh loss, Standard MRL (M={5,10,25,50}), Full-prefix MRL (M=[50])**

---

## What We Are Proving (Narrative Arc)

1. **Ordering ≠ Basis-alignment** — these are genuinely distinct properties
2. **MRL's symmetry breaking is task-focused** — privileged directions depend on the objective, not data variance
3. **Full-prefix MRL has tighter structure than standard MRL** — validates the theoretical contribution

Experiments 1–2 are main paper. Experiments 3–4 are main paper or appendix depending on results.

---

## Experiment 1: PCA Subspace Recovery
**Validates: Theorem 1 (PCA recovery result)**
**In paper: Yes**

### Datasets
- MNIST and Fashion-MNIST (run both; decide which to show in paper based on results)
- Raw data, no PCA preprocessing (p=784 for MNIST/Fashion-MNIST)
  Ground-truth PCA eigenvectors computed from the raw data covariance directly.
  Pre-projecting via PCA then recovering PCA would be circular.
- Normalize to zero mean, unit variance per dimension

### Latent dimension
- Run both d=16 and d=32; decide based on which shows the standard MRL gap most clearly
- d=16: standard MRL uses M={2,4,8,16}; full prefix m=1..16
- d=32: standard MRL uses M={4,8,16,32}; full prefix m=1..32
- d=32 likely shows a cleaner gap since more non-evaluated prefix sizes exist

### Models (all trained with MSE reconstruction loss)
- **MSE LAE** — no constraint (baseline, expect no recovery)
- **MSE LAE + orthogonality** (A^T A = I) — known PCA recovery result, upper baseline
- **Standard MRL** (M={2,4,8,16} for d=16 or M={4,8,16,32} for d=32), no constraint
- **Full-prefix MRL, no orthogonality constraint** — validates Theorem 1 Part 1
- **Full-prefix MRL, with orthogonality constraint** (A^T A = I) — validates Theorem 1 Part 2

The 2x2 structure (orthogonality x full-prefix) is the key comparison:

|                    | No orthogonality          | With orthogonality         |
|--------------------|--------------------------|---------------------------|
| MSE LAE            | no recovery (baseline)   | PCA recovery (known)       |
| Full-prefix MRL    | subspace recovery (Thm 1 Part 1) | full PCA recovery (Thm 1 Part 2) |

Standard MRL sits in between: partial recovery only at evaluated prefix sizes.

### What to measure
For every prefix size m = 1, 2, ..., d:

1. **Subspace angle**: principal angles between recovered colspan(A*_{1:m})
   and ground-truth span(u_1,...,u_m) from PCA of the dataset.
   - Compute via scipy.linalg.subspace_angles(A_{1:m}, U_{1:m})
   - Report mean principal angle in degrees, mean ± std over 5 seeds

2. **Column alignment**: for each column of A*_{1:m}, cosine similarity to
   its nearest PCA eigenvector.
   - Report mean max cosine similarity across columns, mean ± std over 5 seeds

### Expected results
- MSE LAE, no constraint: large subspace angles, low column alignment (no recovery)
- MSE LAE + orthogonality: small angles, high alignment (PCA recovery — known result)
- Standard MRL: small angles at M sizes only, larger at non-evaluated sizes
  (shows the O(log d) gap — key evidence for full-prefix contribution)
- Full-prefix MRL, no constraint: small angles everywhere, low column alignment
  (subspace recovered without orthogonality — Theorem 1 Part 1)
- Full-prefix MRL + orthogonality: small angles everywhere, high alignment
  (full eigenvector recovery — Theorem 1 Part 2)

### Plots
- Main: line plot, prefix size m (x-axis, 1..d) vs mean subspace angle (y-axis),
  one line per model. Standard MRL should show visible spikes at non-M prefix sizes.
- Secondary: column alignment vs prefix size m, same layout.
- Run for both d=16 and d=32; use whichever shows the standard MRL gap more clearly.

---

## Experiment 2: Divergence Experiment — Objective-Specific Privilege
**Validates: Core claim that full-prefix MRL privilege is objective-specific, not variance-specific**
**In paper: Yes — this is the central empirical result**

### Core claim
Same architecture, same training procedure, only the loss changes → privileged directions change.
Full-prefix MRL with MSE → aligns with PCA (variance-optimal).
Full-prefix MRL with cross-entropy → aligns with LDA (classification-optimal).
This demonstrates that privilege is determined entirely by the objective, not by the method.

### Why Oftadeh is dropped from Exp 2
Oftadeh loss is mathematically derived for the reconstruction objective only — the S_d weighting
produces variance-based ordering via its interaction with the MSE gradient. There is no
principled extension to cross-entropy. Oftadeh stays in Exp 1 as a reconstruction sanity check
(mathematically equivalent to full-prefix MRL + MSE for linear AE).

### Models (5 core)
1. **MSE LAE** — no ordering baseline (encoder directions arbitrary)
2. **Full-prefix MRL (MSE)** — task = reconstruction → expect PCA alignment
3. **Full-prefix MRL (CE)** — task = classification → expect LDA alignment
4. **Standard MRL (CE)** — same as (3) but M = O(log d) prefixes → LDA alignment but weaker
5. **PCA + linear probe** — theoretical baseline: PCA subspace, probe trained on frozen PCA scores

Optional additions (implement after core is working):
- **Nested Dropout + MSE** — reconstruction-only ordered baseline, variance-based
- **Slimmable Networks** — ordered sub-networks for classification, no privilege guarantee

### The story the table tells

| Method                     | Prefix acc ↑ | Angle to PCA ↓ | Angle to LDA ↓ |
|----------------------------|-------------|----------------|----------------|
| PCA + linear probe         | medium      | 0°             | large          |
| MSE LAE                    | low         | large          | large          |
| Full-prefix MRL (MSE)      | medium      | small          | large          |
| Full-prefix MRL (CE)       | **high**    | large          | **small**      |
| Standard MRL (CE)          | high        | large          | medium         |

The crossover between Full-prefix MRL (MSE) and Full-prefix MRL (CE) on both metrics is the result.

### Architecture for CE-loss models
- LinearAE encoder B ∈ R^{d×p}: shared across all prefix sizes
- Per-prefix classifier head W_m ∈ R^{C×m} for m = 1..d (trained jointly)
- Full-prefix MRL (CE) loss:
    L = (1/d) * sum_{m=1}^d CE(W_m @ B_{1:m} x, y)
- Standard MRL (CE) loss:
    L = (1/|M|) * sum_{m in M} CE(W_m @ B_{1:m} x, y)
- Metric: compare encoder rows B^T[:,1:k] (directions in R^p) to PCA/LDA subspaces
  (analogous to comparing decoder columns in Exp 1)

### Datasets
Priority order — run cleanest first, rest in appendix:
1. **Synthetic** — fully controlled PCA ≠ LDA construction (start here):
   - p_signal=19, p_noise=50, p_total=69, C=20, n=10,000 (500/class)
   - Noise block (dims 1–50): x_noise ~ N(0, σ_noise² I), σ_noise=5.0, same for all classes
   - Signal block (dims 51–69): x_signal ~ N(μ_c, σ_signal² I), σ_signal=0.1, μ_c class-specific
   - Class means μ_c: C-1=19 random orthogonal vectors in R^{19}, scaled by class_sep=1.0
   - p_signal = C-1 = 19 exactly → between-class scatter is full rank in signal space
   - Every signal dim is discriminative; LDA directions span the entire signal block
   - Noise dims placed first so top-d PCA eigenvectors all land in noise block
   - LDA gives C-1=19 directions, all in signal block → PCA ⊥ LDA by construction
   - Verification: angle between top-19 PCA subspace and top-19 LDA subspace should be ~90°
2. **20 Newsgroups (TF-IDF)** — natural PCA ≠ LDA divergence
   - C=20 → 19 LDA directions (enough for a clean curve up to m=19)
   - Frequent words ≠ discriminative words, no augmentation needed
3. **MNIST + noise augmentation** — controlled real-data check
   - Append high-variance Gaussian noise dims (σ_noise=5.0, zero class signal)
   - Forces PCA to focus on noise, LDA to focus on digit structure
4. **Fashion-MNIST + noise** — same augmentation as MNIST

### LDA constraint
LDA gives at most C-1 discriminant directions.
- Synthetic: C=20, p_signal=19=C-1 → 19 LDA directions, full rank. Full curve up to m=19.
- MNIST/Fashion-MNIST: C=10 → 9 LDA directions. Report angle curves only up to m=9.
- 20 Newsgroups: C=20 → 19 LDA directions.
- Angle-to-LDA curves always plotted up to min(k, C-1).

### What to measure
For each prefix size k = 1..min(d, C-1):
1. **Angle to PCA subspace**: principal angles between encoder B^T[:,1:k] and top-k PCA eigenvectors
2. **Angle to LDA subspace**: principal angles between encoder B^T[:,1:k] and top-(C-1) LDA directions
3. **Prefix accuracy**: accuracy of W_k @ B_{1:k} x on test set (for CE models);
   accuracy of linear probe trained on frozen B_{1:k} x for MSE models and PCA baseline
Report mean ± std over 5 seeds.

### Plots
- Left panel: angle to PCA subspace vs k, one line per model
- Right panel: angle to LDA subspace vs k, one line per model
- Optional third panel: prefix accuracy vs k
- The crossover of Full-prefix MRL (MSE) vs (CE) on both panels is the key visual

---

## Experiment 3: Kurtosis + Drop-off Shape
**Validates: (a) task-signal vs variance-signal produce different latent geometry;
(b) basis-alignment conjecture — aligned dims show sharper importance drop-off**
**In paper: Yes (kurtosis bar chart) + potentially appendix (drop-off)**

### 3a — Kurtosis
Use trained model checkpoints from Experiments 1 and 2.

**What to measure**: for each model and each dimension k,
  kurtosis_k = E[(z_k - mu_k)^4] / E[(z_k - mu_k)^2]^2
  computed over test set. Report mean kappa_bar = (1/d) sum_k kurtosis_k,
  std over 5 seeds.

Also plot per-dimension kurtosis profile (kurtosis_k vs k sorted by index)
to show whether early dimensions are more concentrated.

**Expected results**:
- MSE LAE: kappa_bar ~ 3 (Gaussian, no privilege)
- Oftadeh: kappa_bar >> 3 (basis-aligned, concentrated activations)
- MRL: kappa_bar ~ 3 (task signal, residual rotational freedom within shells)
- Full-prefix MRL: kappa_bar slightly > 3 (more structured than standard MRL)

**Plot**: Bar chart, one bar per model, error bars over seeds.
Horizontal dashed line at 3 labeled "Gaussian baseline."

### 3b — Drop-off Shape
**What this tests**: whether basis-alignment produces sharper importance concentration.

**Procedure**:
- For each model, post-hoc sort dimensions by marginal accuracy gain:
  Delta_k = Acc(1:k) - Acc(1:k-1)
- Plot Delta_k vs rank k for each model
- Also run the same procedure on post-hoc sorted MSE LAE (permute its
  dimensions by importance) — if this matches MRL, ordering is trivially
  achievable post-hoc

**Expected results (conjecture — may not hold)**:
- Oftadeh: sharp early drop-off (basis-aligned dims capture information efficiently)
- MRL: more gradual drop-off (subspace-level ordering, less per-dimension concentration)
- Post-hoc sorted LAE: if it matches MRL, ordering is trivial; if worse, MRL's
  training-time ordering adds value

**Plot**: Line plot, Delta_k (log scale y-axis) vs rank k, one line per model.

---

## Experiment 4: Prefix Performance Curves + Subspace Stability
**Validates: (a) functional ordering — MRL front-loads task signal;
(b) prefix identifiability — MRL learns stable subspaces across seeds**
**In paper or appendix: depends on results**

### 4a — Prefix Performance Curves
For each model, freeze encoder and train linear probe on z_{1:k} for k=1..50.
Plot test accuracy vs prefix size k.

Key comparison: MRL vs post-hoc sorted LAE.
- If post-hoc sorted LAE matches MRL accuracy curves, MRL's ordering is trivially
  achievable and its real contribution is elsewhere
- If MRL is better, training-time ordering adds genuine value

### 4b — Prefix Subspace Stability (replaces failed coordinate correlation)
For each model, 5 seeds, each prefix size k:
- Compute principal angles between span(z_{1:k}^(s)) and span(z_{1:k}^(s'))
  for all seed pairs (s, s')
- Report mean angle per k

**Three levels of identifiability**:
1. Coordinate stability: |<e_k^(s), e_k^(s')>| (strong — probably fails for MRL)
2. Prefix subspace stability: principal angles (medium — likely holds for MRL)
3. Functional ordering stability: variance of Delta_k curves across seeds (weak — should hold)

**Expected results**:
- MSE LAE: all three unstable
- Oftadeh: all three stable
- MRL: functional ordering stable, prefix subspace stable, coordinates not
- This is the honest claim for MRL

**Plot**: Line plot of mean principal angle vs prefix size k, one line per model.
Table: Method | Coord stability | Subspace angle | Delta_k variance

---

## Implementation Notes

### Datasets

**Exp 1 (PCA recovery)**: MNIST
- Standard train/test split
- Project to p=100 via PCA, use d=50 (so rank constraint is binding)
- Clean baseline, validates theorem, everyone trusts it

**Exp 2 (Divergence — PCA vs LDA)**:
Try all of the following, report best two in paper:

1. **20 Newsgroups (TF-IDF)** — primary candidate
   - ~20 classes, TF-IDF features (high-variance dims = frequent/non-discriminative words,
     LDA dirs = rare topic-specific words)
   - PCA ≠ LDA naturally and strongly, no artificial construction needed
   - Linear methods work well on TF-IDF
   - Reduce to p=100 via TruncatedSVD, d=50

2. **MNIST + noise augmentation** — controlled sanity check
   - Project MNIST to p=50 via PCA, append 25 dims drawn from N(0, sigma^2)
     with sigma >> 1 (high variance, zero class signal) → p=75, d=50
   - Useful as an ablation: shows the divergence is not dataset-specific
   - Reviewer may call it artificial — use as supporting evidence, not primary

3. **Fashion-MNIST** — secondary visual dataset
   - Same structure as MNIST, slightly harder
   - PCA/LDA less correlated than MNIST
   - Project to p=100, d=50
   - Useful for cross-dataset consistency claim

4. **CIFAR-10 with fixed pretrained features** — if above datasets insufficient
   - Freeze ResNet features (512-dim), apply linear AE on top
   - More complex, visually intuitive
   - Adds pretrained model dependency — use only if other options don't show clean divergence

**Exps 3–4 (Kurtosis, drop-off, prefix curves, stability)**:
- Reuse checkpoints from Exps 1 and 2
- Run on whichever datasets work cleanest in Exps 1–2

**Normalization**: zero mean, unit variance per dimension for all datasets

### Training
- Optimizer: Adam, lr=1e-3
- Epochs: 500
- Batch size: 256
- MRL weights c_m = 1 (uniform)
- Oftadeh: exact S_d weight matrix formulation
- Orthogonality constraint (Exp 1): Cayley parametrization or Gram-Schmidt
  projection after each step

### Evaluation
- All metrics on test set (10,000 MNIST test images)
- Report mean ± std over 5 seeds
- Principal angles: scipy.linalg.subspace_angles
- Kurtosis: scipy.stats.kurtosis

### Priority order
1. Experiment 1 (PCA recovery) — validates main theorem
2. Experiment 2 (divergence) — proves title claim
3. Experiment 3a (kurtosis) — fast, uses same checkpoints
4. Experiment 3b (drop-off) — tests conjecture, uses same checkpoints
5. Experiment 4a (prefix curves) — supporting evidence
6. Experiment 4b (subspace stability) — if time permits
