# Tier 1 — Frozen-Encoder Target Alignment Experiment

**Status:** Decisions finalized 2026-10-29 (Section 0). Code written, not yet run: `NonLinearPBExperiments/scripts/tier1_target_alignment.py`.  
**Scope:** No encoder fine-tuning. Measure how target-relevant directions sit relative to the coordinate order of independently pretrained MRL β-VAE encoders.

## 0. Finalized decisions (2026-10-29)

1. **Standardization.** Per encoder, z-score each latent coordinate with the training-embedding mean and std; apply the same statistics to validation and test. Coordinate order is preserved. Raw μ is a sensitivity check; full whitening stays secondary.
2. **Active dimensions.** Variance, reconstruction gain, and KL (β > 0) stay separate diagnostics; they are never mixed into one rule. Nulls are empirical random subspaces, computed in all 10 coordinates and separately inside each rule's active coordinates, with the convention named in the output.
3. **Probes.** Linear only. MLP probes postponed.
4. **Coordinate permutation.** Postponed. H8 (order vs orientation) cannot be decided until it is added.
5. **Learned rotation.** Orthogonal `Q_T` (Q_T^T Q_T = I), equal-weight mean target CE over prefixes k = 1..10, encoder frozen, 5 restarts, each from a different random orthogonal start.
6. **k\*.** Smallest k with `A_k >= A_chance + 0.9 (A_full - A_chance)` on balanced accuracy; flagged uninformative when the chance-normalized full-probe accuracy is below 0.05.

Implementation choices: angle labels use `round(cos α X + sin α Y, 9)`, train-quantile cut points (method "higher", so a cut point is the first value of the upper class), ties to the upper class; scatter rank = eigenvalues above 1e-3 × largest, at most C − 1; the scatter basis is filled beyond its rank with PCA of the remaining space, and only k ≤ rank is interpreted; equal-class scatter is primary, frequency-weighted is recorded. First run: Stage 0 + β = 0 pilot (4 encoders), validation only.

## 1. Research question

Given a frozen source encoder and a new target task, can we describe (i) whether target information is accessible, (ii) where it sits in the ordered latent coordinates, and (iii) whether improving short-prefix accessibility requires no change, a coordinate permutation, or a genuine rotation?

**Do not equate geometric alignment with actual transfer-learning performance.** Tier 1 measures geometry and frozen-feature prediction; adaptation and forgetting belong to a later experiment.

## 2. Available pretrained models

- Existing MLP MRL β-VAE; **10-dimensional** posterior-mean embedding `mu(x)`.
- Sources: `S_recon` (nested reconstruction only) and `S_posX` (nested reconstruction plus X-position supervision).
- β ∈ {0, 1, 4}; **2 encoder seeds** per source–β cell = **12 trained models**.
- Same fixed 80/10/10 image split for every model. Train data is for fitting target directions/heads; validation for choices and checks; test remains untouched until final evaluation.
- `S_scale` source is postponed. **Shape and scale remain target tasks.**
- Source pretraining does **not** imply X must lie in z1 or that coordinates are disentangled. Do not alter or re-train encoders to force the desired hypotheses.
- Use `mu(x)`, not posterior samples, throughout frozen-encoder measurement.

## 3. Targets

### 3.1 Family A — position-angle tasks

For each image with integer factor labels X,Y ∈ {0,…,31}, set

`s_alpha(x) = cos(alpha)*X(x) + sin(alpha)*Y(x)`.

Primary angles: **0°, 30°, 60°, 90°**. Bin the scores into **6 approximately equal-frequency classes** (labels 0–5). Compute and save deterministic boundaries/tie handling from the training data only; reuse them on validation and test. Log class counts and entropy; ties at discrete X/Y values can prevent perfectly equal class sizes.

- 0° = pure X (in-domain supervised-target sanity check).
- 30°, 60° = **main mixed-target tests** (neither source directly trained on the mixtures).
- 90° = pure Y (**named transfer/control target**, despite no Y supervision).

### 3.2 Family B — other factors

- `T_scale`: 6-class scale label.
- `T_shape`: 3-class shape label.

Both use the original factor labels. Total: **6 target tasks × 12 frozen source models = 72 source–target evaluations** (no new encoder trainings). Maintain the same train/validation/test split across all targets.

## 4. Two ways to find target-relevant directions

### Method 1 — class-mean scatter (closed form; primary)

For each **source model and target**, on training embeddings only:

- Compute class-mean vectors `mu_c ∈ R^10` and overall mean `mu`.
- Compute `S_B = sum_c (mu_c-mu)(mu_c-mu)^T` (equal-class contribution; record class counts and optionally compare frequency-weighted scatter).
- Eigendecompose `S_B`: ordered eigenvectors `m1,...,mr`, eigenvalues `lambda1>=...>=lambdar>0`; use a pre-specified numerical rank tolerance.
- `r <= min(C-1, d_active, 10)`: at most **5** for each six-class task and at most **2** for shape. Do not claim `A(k)` from scatter beyond the identified target rank.

Scatter finds directions of **large class-mean separation**, not necessarily directions optimizing classification. Keep eigenvalues and check whether one direction dominates.

### Method 2 — learned orthogonal rotation (secondary)

Freeze source encoder. Fit an orthogonal `Q_T` together with nested **linear classification heads** for target labels; rotated prefixes see `(Q_T^T (z-center))[:k]`. Use **5 optimization restarts per fixed encoder–target pair**, separate from the 2 encoder seeds. No encoder retraining.

Compare fitted rotations by *prediction and target subspaces*, not by an arbitrary full-matrix angle: rotations of unused coordinates are unidentifiable. If several restarts yield similar accuracy but different `Q_T`, report that instability.

**The two methods are not mathematically identical:** scatter maximizes class-mean variance; `Q_T` optimizes nested classification. Disagreement is diagnostic, not automatic proof that one is broken. Record principal angles where both estimate the same rank and compare held-out prefix accuracy.

## 5. Metrics (compute per source, target, and encoder seed)

### 5.1 Original-order alignment

With `P_k` selecting the first `k` original latent coordinates and `M_j=[m1,...,mj]`, define

`O(k,j) = ||P_k M_j||_F^2 / min(k,j)`.

For the leading direction: **`A(1) = (e1^T m1)^2 = m1[1]^2`**; e1 is the *first original source coordinate*. For matched dimensions use `A(k)=O(k,k)` only when `k <= r`.

This is **alignment**, not actual rotation cost or transfer success. A low A(1) may coexist with good full-feature prediction.

### 5.2 Where target information sits

- Report `A(1)` and the full valid `O(k,j)` curve.
- Report **depth of the leading target direction**: `D1 = sum_{i=1}^{10} i * m1[i]^2`. Larger = later coordinates.
- Optionally report eigenvalue-weighted depth across the well-identified target subspace; disclose normalization and rank.
- For target factors, use existing factor-by-coordinate `eta^2` as descriptive support, **not** as a rule selecting a privileged first coordinate or as a substitute for target direction estimation.

### 5.3 Actual accessibility

Fit fresh, standardized *target probes* on **frozen embeddings**, not encoder parameters. Record:

- Linear probe performance on `z[:k]` for `k=1,...,10` and full `k=10`.
- The same after **best coordinate permutation** (learn permutation on training data; select on validation), and after learned `Q_T`.
- Chance-normalized balanced accuracy across tasks: `(accuracy - 1/C)/(1 - 1/C)`; also retain raw balanced accuracy and, for X/Y, ordinal R² where appropriate.
- Report target `k*` (smallest prefix reaching a predeclared fraction of the full-probe improvement over chance), plus full-probe performance as a separate adequacy check.

**Permutation vs rotation is the key additional control:** if permutation suffices, the issue was coordinate order; if an orthogonal mixture is needed, it was basis orientation. A full-dimensional unregularized linear classifier has no intrinsic information gain from any invertible rotation; compare **short prefixes**.

### 5.4 Nulls, active dimensions, and covariance

- Determine active coordinates per encoder from the pretraining diagnostics; keep each coordinate's activity mask and its position. Do not assume active dimensions are the first `d_active`.
- If target directions and random rotations are defined **inside the active subspace**, let `k_active(k)` be the count of active coordinates in prefix k. The analytic expected overlap for random j-dimensional task subspaces in the d_active-dimensional active subspace is:

`E[O(k,j)] = k_active(k) * j / (d_active * min(k,j))`.

This formula assumes `j <= d_active` and the same normalization as `O`. It does **not** directly apply when rotations mix active and inactive coordinates in all 10 dimensions; specify the null consistently with the rotation procedure.
- Compare **100 or more** random orthogonal rotations using the chosen active-space convention and report the *distribution*, not an assertion that every draw equals its expectation.
- With β > 0, inactive dimensions may have nearly zero variance. Compute covariance conditioning/whitening only on retained active coordinates, with a fixed documented regularization threshold. Preserve the original coordinate system when reporting source-axis alignment; whitened scatter is a **secondary sensitivity analysis**, not a replacement for original ordering.
- Report original raw `A(1)`, the appropriate null, and where useful a null-adjusted score. Do not simply divide all observed scores by `d_active`; the active coordinates' positions matter.

## 6. Hypotheses to pre-register (falsifiable, not guarantees)

**H1 — Objective × target angle interaction.** At β=0, `S_posX` may show better *early-prefix alignment/accessibility* than `S_recon` for α=0 and perhaps 30°, with the advantage weakening by 60° and 90°. **α=0 is a sanity check, not evidence of general transfer**; α=30° and 60° are the decisive tests. This can fail when supervision does not orient z1.

**H2 — Angle dependence is nonlinear, but not automatically cos²α.** In the **idealized** equal-scale, angle-preserving coordinate model with z1=X, `A(1)=cos²α`, giving {1, 0.75, 0.25, 0} at {0°,30°,60°,90°}. Real nonlinear pretrained embeddings and binned targets need not follow this line; measure the actual curve and do not enforce monotonicity by assumption.

**H3 — Non-X factors.** X supervision may shift early-prefix availability of scale and shape relative to `S_recon`, but no deterioration is guaranteed because both sources still optimize reconstruction. Test rather than presume a trade-off. Scale may be earlier than shape, but treat this as a prediction, not established pixel-variance fact.

**H4 — Y generalization.** Compare α=90° performance and alignment on `S_posX` versus `S_recon` at fixed β; ask whether X supervision organized position information beyond X. 'Above chance' alone is not compelling because both sources reconstruct positions.

**H5 — β and concentration.** From β=0 to 1 to 4, fewer coordinates may remain active while useful target information becomes more concentrated in early active prefixes. Raw scores across β must be contextualized by activity masks and matched nulls; a larger β does **not** guarantee better disentanglement or a different prediction for every target.

**H6 — Seed instability.** `S_recon` may have more variation across seeds in early target alignment than `S_posX`, but `S_posX` has already shown varying coordinate placement. **Show both encoder seeds individually**; do not average away disagreement. Two seeds support descriptive differences, not a robust variance estimate.

**H7 — Information vs alignment.** Some source–target pairs will have strong full 10-dimensional prediction despite poor `A(1)`; these represent accessible but non-early information.

**H8 — Reordering vs genuine rotation.** For some targets, a permutation will recover short-prefix accessibility; for others, the learned `Q_T` will outperform every permutation. The margin measures whether the limitation was order or orientation.

**H9 — Alignment vs rotation benefit.** Poor original-prefix alignment may predict a larger short-prefix accuracy improvement after target rotation. Test association empirically, holding source and target fixed when possible; do not equate angle with optimization effort.

**H10 — Multiple relevant directions.** For some targets, the leading scatter eigenvalue will dominate; for others, signal will be spread across several directions. Report `lambda1/sum(lambda_positive)` and avoid relying only on `A(1)` where multiple directions matter.

**H11 — β can change ordering without increasing information.** Full-probe accuracy may remain similar across β while early-prefix curves and null-adjusted overlap change. Separate capacity, representation content, and ordering.

## 7. Controls and analysis sequencing

### Stage 0 — prepare (no training)

1. Validate the 12 frozen model paths, common data split, matching image ordering, and saved active masks.
2. Cache posterior-mean embeddings **separately** for training, validation, and test.
3. Create the six target label arrays and freeze binning protocol.
4. Specify numerical-rank tolerance, active-space treatment, probe normalization, random-rotation convention, and primary metrics before inspecting test data.

### Stage 1 — β = 0 pilot (4 frozen encoders: 2 objectives × 2 seeds)

1. Start with α = 0°, 30°, 60°, 90°, scale, shape.
2. Estimate scatter directions on training embeddings; report eigenvalues, `A(1)`, relevant overlaps, depth, and prefix probes.
3. Run **random-rotation control** (many draws): preserved full-feature information and loss of systematic privileged-order signal relative to the random baseline. Rotation does not make each individual score equal its null.
4. Compare original prefix vs best **permutation** vs fitted **rotation**, using target-head validation performance and held-out reporting.
5. Report each encoder seed separately and examine their disagreement before interpreting a trend.

### Stage 2 — β = 1 and β = 4

Repeat exactly the same targets, estimators, probe settings, and controls. Examine differences in active dimensions, geometry, short-prefix access, and seed stability; do not assume greater β means cleaner task directions.

### Stage 3 — final test-set evaluation

After fixing analysis choices from training/validation data, evaluate final metrics on held-out test. Do not refit source encoders, scatters, bins, permutations, or rotations on test labels.

## 8. Deliverables

1. **Source-by-target table**: objective, β, seed, target, class counts, d_active, active mask, scatter rank, eigenvalues, `A(1)`, valid `O(k,j)`, depth, and active-space null.
2. **Alignment figure**: `A(1)` versus α separately for each source–β–seed, idealized cos²α as *reference only*, with random-rotation null distributions.
3. **Accessibility figure**: original/permuted/rotated prefix performance, with full-probe performance visible.
4. **Estimator agreement figure**: scatter vs learned-rotation leading/subspace principal angles and held-out prefix performance; differences are interpreted, not automatically deemed bugs.
5. **Factor analysis**: `T_scale`, `T_shape`, and α=90° Y explicitly reported, including eigenvalue concentration and rank limits.
6. **Seed report**: paired plots or values for seeds 0 and 1, not averaged into one apparently stable result.
7. **Machine-readable results**: fixed splits, bin edges/tie rules, active masks, rank tolerance, scatter eigenpairs, rotations and restart metrics, permutations, probe scores, random-rotation draws, and all frozen configuration details.

## 9. Important non-claims / exclusions

- **No direct cross-model basis-angle comparison** between independently trained `S_recon` and `S_posX` is required. Different encoders have different coordinate frames; target evaluation is the common comparison.
- Do **not** require X to occupy `z1`, and do not declare a run invalid because it occupies `z2` or `z3`.
- Do **not** assert the β=0 `S_recon` alignment is exactly random; measure it and compare with a null.
- Do **not** treat two encoder seeds as strong evidence of population-level stability. Keep five independent learned-`Q_T` restarts per fixed encoder/target.
- Do **not** assume disagreement between scatter and learned `Q_T` means either is broken.
- Do **not** equate a low alignment value with absent information or a rotation angle with actual fine-tuning cost.
- Do **not** enforce the ideal `cos²α` prediction as an identity for nonlinear β-VAE latent geometry.

## 10. Practical implementation note

Use existing trained models and analysis utilities in the `NonLinearPBExperiments` codebase. Do not create a new encoder architecture or train additional source models for this Tier 1 analysis. All computations here operate on frozen posterior-mean embeddings and newly fit target-side statistics, permutations, or classification heads.
