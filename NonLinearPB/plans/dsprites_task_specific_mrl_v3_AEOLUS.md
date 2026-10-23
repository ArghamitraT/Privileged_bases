# dSprites Task-Specific Matryoshka: Revised Experimental Plan (v3)

**Status:** Pre-implementation research specification, written 2026-10-22. No code written, no training launched.
**New document:** the earlier plans are unchanged:
`dsprites_task_specific_mrl_experiment_plan_AEOLUS.md` (old task-MRL proposal),
`dsprites_task_specific_mrl_two_approaches_AEOLUS.md` (frozen-Q vs fine-tuning),
`dsprites_vae_beta_vae_matryoshka_three_seed_plan.md` (completed reconstruction-MRL study).

**Core question:** Can jointly training a nonlinear representation with a nested task objective produce reproducible, task-dependent *native-coordinate* organization that cannot be explained by an equally supervised non-MRL encoder followed by an optimal post-hoc orthogonal change of basis?

## 0. Plan in short

1. **Phase 0 — check existing models.** Take the 3 existing β-VAEs. Predict X, Y, shape, rotation from their 10 latents. Find which tasks work and the k\* for each task.
2. **Experiment A — rotate a frozen model.** Freeze the encoder. Learn a rotation Q that moves task information to the first coordinates. Compare to PCA, LDA, best permutation, original order.
3. **C pilot.** X and Y, one seed. Choose λ (weight of the task loss).
4. **Experiment C — main: train from scratch.** C0 = β-VAE + one classifier on all 10 latents. C1 = β-VAE + a classifier on every prefix (MRL). C0+Q = C0, rotated afterwards with A's method. 4 tasks × 2 models × 5 seeds.
5. **Experiment B — optional.** Same as C, but start from the existing β-VAEs.

**For every model:** Figure 6, accuracy of each coordinate alone, prefix accuracy k = 1…10, agreement across seeds.
**Main test:** X-models put X first, Y-models put Y first, and C1 does this better than C0 and C0+Q.

## 1. Claims and hypotheses

Keep three claims distinct:

1. **Early-prefix task utility:** the chosen target can be predicted with fewer native early coordinates.
2. **Objective specificity:** holding images, architecture, and optimization protocols fixed, selecting task X versus task Y reverses which factor is more accessible in the earliest coordinates.
3. **Axis reproducibility:** across independent training seeds, coordinate-specific factor associations agree in native index order (allowing sign flips of coordinate values but **no permutations or learned rotations** for the primary claim).

A useful early prefix is not by itself an identifiable basis. A reproducible *subspace* is weaker than reproducible *axes*. No experiment here proves a unique nonlinear basis.

**Primary pre-registered contrast:** X-position versus Y-position on full dSprites. **Extensions:** shape and rotation; rotation is evaluated initially on hearts to reduce rotational ambiguity. Shape prediction necessarily uses multiple shapes; **do not evaluate shape accuracy on a heart-only subset**, because its label is constant there.

## 2. Existing work and reusable infrastructure

The completed October 2026 reconstruction study trained ordinary VAE (β=1), β-VAE (β=4), MRL-VAE (β=1), and MRL-β-VAE (β=4), with seeds 0–2, 100 epochs, Adagrad 0.01, decoder Tanh, and 10-dimensional latents (results root `/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__21_22_10/`). Treat the checkpoints, native-order Figure 6 panels, and full-latent metric as reference material — not as evidence for supervised task specificity. Reuse code/data plumbing and the unsupervised reconstruction-MRL checkpoints as *reference representations*. Do not overwrite existing run artifacts or plans.

For all new conditions, use the same dSprites archive (`/home/argha/Mat_embedding/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`), deterministic splits, encoder and decoder family, 10-dimensional Gaussian latent, Bernoulli image likelihood, β=4 initially, matched seeds where meaningful, and one declared optimizer protocol (§10). **Supervised** experiments require image-level disjoint train/validation/test partitions: unlike the previous unsupervised full-archive metric, their classification claims must be evaluated on images not used to supervise task heads. Prefer an additional held-out factor-combination stress test if practical.

**Test-image exposure of reference models.** The existing β-VAEs (used in Phase 0, A, B) and the reconstruction-MRL reference were trained *without labels* on the full archive, so their encoders have seen the new test images. This is acceptable for probing (no labels were used), but it differs from C, whose encoders never see test images. State this difference wherever reference models are compared to C.

## 3. Model notation and training principles

- $r_{\phi}(x)=\mu_{\phi}(x)\in\mathbb R^{10}$: deterministic posterior mean for probes and the principal classification protocol.
- $R(\phi,\theta)$: average Bernoulli reconstruction negative log likelihood, **sum over 4096 pixels per image**.
- $K(\phi)$: analytic KL to $\mathcal N(0,I)$, summed over all ten coordinates.
- $C_t(h,z_{1:k})$: cross-entropy for target $t$ from first $k$ coordinates.
- $\mathcal K=\{1,2,\ldots,10\}$ for the first controlled study, with weights $w_k=1/10$.

**Lock the classification input:** use $\mu_\phi(x)$ for A, B, and C classification heads, while reconstruction continues to use reparameterized sampled $z$. Optionally repeat with sampled $z$ as a robustness check. For B/C, classification gradients must propagate through $\mu_\phi$ into the encoder.

Use linear heads for the primary comparisons; matched small-MLP and k-NN probes are *secondary diagnostics*, not substitutes for linear results.

## 4. Phase 0: Audit pretrained representations (run first)

For each existing β-VAE seed (0, 1, 2), freeze the encoder and fit held-out full-latent linear probes for X, Y, shape, and rotation. Fit small-MLP or k-NN probes as secondary tests for information that is not linearly decodable. Record chance accuracy, class distribution, per-class failures, and linear–nonlinear gap. Also fit **single-coordinate probes** ($z_i$ alone) and **prefix probes** ($z_{1:k}$, $k=1..10$).

**Rotation labels.** dSprites orientation is `linspace(0, 2π, 40)`: labels 0 and 39 are the same angle. Verified 2026-10-22: for all three shapes, the images at orientation 0 and 39 are pixel-identical. Therefore **merge labels 0 and 39 (39 rotation classes)** for every rotation task. Rotation uses hearts initially; also check hearts for any further rasterization ambiguities.

**Gate:** if a frozen feature set cannot support a task even with its full 10-D linear probe, a linear orthogonal $Q$ cannot make that task linearly accessible. Such outcomes are reported but not counted as failures of MRL coordinate ordering. Phase 0 also gives provisional saturation thresholds for $k_t^*$ (§8).

## 5. Experiment A: Fixed-feature linear bridge, and the decisive post-hoc control

Fix encoder $r(x)$ and learn orthogonal $Q_t\in O(10)$, $z_t=Q_t^\top r(x)$, with linear nested classification heads $h_{t,k}$:

$$
\min_{Q_t\in O(10),\{h_{t,k}\}}\; \frac1{10}\sum_{k=1}^{10}C_t\!\left(h_{t,k},(Q_t^\top r(x))_{1:k}\right).
$$

Use a correct orthogonality-preserving parameterization or retraction, and verify numerically that $Q_t^\top Q_t\approx I$. Keep the *unrotated frozen β-VAE* as a reference. This is a **linear problem in fixed features**, not evidence that a nonlinear encoder learned a privileged basis.

### A baselines (genuinely different optimization problems)

1. **Fixed original order:** nested linear heads directly on $r_{1:k}$.
2. **Permutation-only:** coordinate permutation chosen on validation data, matched heads, evaluated once on test.
3. **PCA rotation:** orthonormal eigenvectors of train-feature covariance; unsupervised baseline.
4. **Fisher/LDA rotation:** discriminative directions from train features, converted to a *Euclidean-orthogonal* basis with an explicitly documented rule. LDA is a baseline, **not** a guaranteed optimum of nested cross-entropy. For $c$ classes there are at most $c-1$ nonzero discriminant directions (shape: 2); document how the remaining directions are filled.
5. **Per-$k$ oracle:** separate orthonormal $k$-frame and head for each $k$; subspaces need not nest. An approximate best-achievable per-$k$ reference, not a proven upper bound.
6. **Greedy nested:** optimize direction 1, freeze it, add and optimize the next orthogonal direction, etc.
7. **Joint MRL $Q$:** all nested prefix losses optimized together (main A condition).

*Per-$k$ oracle − joint MRL* = empirical cost of nesting. *Greedy − joint MRL* = benefit of joint rather than sequential optimization. Match hyperparameters and restarts; report local-optimum sensitivity.

### A as the principal control for Experiment C (C0+Q)

After training a supervised non-MRL β-VAE C0 from scratch for task $t$, **freeze that exact encoder** and fit A's joint nested $Q_t$ and heads. Each task gets its own rotation: C0\_X + Q\_X, C0\_Y + Q\_Y, etc. Compare to the matched C1 on the same task, splits, and test metrics.

- $\text{C1}\approx\text{C0+Q}$: post-hoc linear reorganization is sufficient for the measured criterion.
- $\text{C1}>\text{C0+Q}$: adapting the representation during training gives an additional benefit. Not by itself proof of an identifiable basis.

## 6. Experiment C (PRIMARY): Nonlinear representations learned from scratch

Random initialization, train from scratch. For each task $t$:

**C0 — supervised non-MRL β-VAE**
$$
\mathcal L_{\mathrm{C0},t}=R+4K+\lambda\,C_t(h_{10},\mu_\phi(x)).
$$

**C1 — task-MRL β-VAE**
$$
\mathcal L_{\mathrm{C1},t}=R+4K+\lambda\frac1{10}\sum_{k=1}^{10}C_t(h_k,\mu_\phi(x)_{1:k}).
$$

Match initial weights, data order, seeds, epochs, optimizer, splits, and total classification-loss weight. C0 and C1 use the same labels; only nesting differs. Nested heads add head parameters; report this and consider a head-capacity-matched ablation.

**Full-latent weight difference.** In C1 the 10-D term has weight $\lambda/10$, in C0 it has weight $\lambda$. C1 may therefore reach lower full-latent accuracy. Always report full-latent accuracy next to prefix results. If full-latent accuracy differs by more than a pre-set tolerance (set in the pilot), also report a C1 variant with an extra full-latent term so both have equal 10-D weight, and state that prefix gains are interpreted relative to full-latent accuracy.

**Required comparisons per task:** C1 vs C0 (native order); C1 vs C0 with validation-chosen permutation; **C1 vs C0+Q**; C1 vs reconstruction-MRL reference (matched probes; note no labels and full-archive training, §2). Also report full-latent accuracy, reconstruction loss, KL, and number of active latents.

### Pre-registered X↔Y reversal

For each trained X-model and Y-model, probe **both X and Y** from the same native prefixes with identical held-out probes on $\mu(x)$. Let $A_f^{(t)}(k)$ be the test accuracy of predicting factor $f$ from prefix $1:k$ of a model trained for task $t$:

$$
D(k)=\big[A_X^{(X)}(k)-A_Y^{(X)}(k)\big]-\big[A_X^{(Y)}(k)-A_Y^{(Y)}(k)\big].
$$

X and Y both have 32 balanced classes, so the accuracies share one scale. Require **both directions** (X ahead of Y in X-models, Y ahead of X in Y-models) and positive paired $D(k)$ across seeds.

**Which models enter $D(k)$:**
- C1: C1\_X and C1\_Y (native order).
- C0+Q: C0\_X + Q\_X and C0\_Y + Q\_Y. **This is the key comparison.**
- C0 native order: expected $D\approx 0$ because its coordinate order is arbitrary; reported but a weak comparison.
- Reconstruction-MRL: one model, no task, so $D=0$ by construction; report $A_X(k)-A_Y(k)$ instead as the unsupervised reference.

Lock $k$ (one common $k$ for X and Y), tolerance, probe training, and statistical summary from Phase 0 / pilot **before** looking at final test results. A reversal is evidence of *task-specific priority*, not of a semantic single-axis correspondence.

## 7. Experiment B (SECONDARY, optional): Fine-tuning pretrained β-VAE

Start from each existing β-VAE checkpoint. B0: full-latent supervised head. B1: nested MRL heads. Fine-tune encoder **and decoder** in both, with the same classification convention as C. Monitor $R$, $K$, full-latent accuracy, prefix curves, native-order maps.

Reset optimizer state **identically for B0/B1** and pilot a learning rate below Adagrad 0.01. Report $\Delta R$, $\Delta K$, and performance relative to the starting checkpoint. B vs C shows how pretrained coordinate organization affects the learned ordering. Run only after C.

## 8. Evaluations

**For every model:** Figure 6 (native order), single-coordinate accuracy ($z_i$ alone, each factor), prefix accuracy $k=1..10$, seed agreement.

**Primary:** prefix performance in *native order*; locked-$k$ X↔Y reversal; C1 vs C0+Q gap; native-order factor–coordinate heatmaps (10 coordinates × 5 factors); cross-seed axis agreement up to sign.

**Secondary:** small-MLP/k-NN probes (linear vs broader decodability); latent traversals; per-coordinate KL, reconstruction, full-latent performance; principal angles between leading subspaces; disentanglement metrics as diagnostics only.

**Seed-stability hierarchy:**
1. Unaligned native-index factor selectivity and activation correlations on the same held-out images; sign flips allowed. **Primary axis agreement is computed only over coordinates $1..k_t^*$.** Coordinates after $k_t^*$ are not constrained by the task, so their order is arbitrary; report them separately and do not average them into the primary score.
2. Matched/permuted correspondence, reported separately to diagnose permutation instability, never used to rescue the primary claim.
3. Principal angles of leading subspaces, as a weaker fallback.

**Choosing $k_t^*$:** from Phase 0 / pilot validation curves, the smallest $k$ near saturation, $k_t^*=\min\{k: A_t(k)\ge A_t(10)-\epsilon\}$, with $\epsilon$ fixed on validation. Then fix one *common* $k$ per task across seeds and conditions before the final study (the largest validation $k_t^*$ across seeds). Linear $k_t^*$ measures **linear decodability**, not the intrinsic dimension of a factor; nonlinear prefix probes are a sensitivity check. For X↔Y use one common $k$.

**Same-task null for principal angles:** angles between different seeds of the **same task** in the *same frozen feature space* (A or C0+Q). Do not compare Q-matrix angles across independently trained encoders in B/C; in C, native coordinate semantics is the primary test.

## 9. Rotation and shape safeguards

- Rotation: heart-only, 39 classes (labels 0 and 39 merged, §4). Consider symmetry-aware labels for mixed shapes later.
- For X/Y vs rotation geometry in A, fit X/Y models on the same heart-only inputs.
- Shape cannot be part of a heart-only comparison (constant label). Compare shape to X/Y on the full multi-shape data.
- Chance levels differ (shape 1/3, X/Y 1/32, rotation 1/39). Do not compare raw accuracies across tasks; use within-task improvement over chance.

## 10. Pilot, loss scaling, locked settings, run order

**Run order:** Phase 0 → Experiment A on existing features → **C pilot** (X and Y, one paired seed; λ sweep; inspect native maps) → full C → C0+Q fits on every C0 → B last (optional).

**Loss scaling:** the 4096-pixel BCE is much larger than per-image CE; do not use λ=1 by convention. For each candidate λ log $\|\nabla_\phi R\|$, $\|\nabla_\phi (4K)\|$, $\|\nabla_\phi (\lambda C)\|$ on matched pilot batches, plus accuracy, reconstruction, and KL. Pre-register a small sweep and a validation-based selection rule. Report results across several λ if conclusions depend on it.

**Settings to lock in the pilot (proposed defaults, not yet locked):**

| Item | Proposed default | Note |
|---|---|---|
| Optimizer (C) | Adagrad, lr 0.01, batch 256 | Same as all earlier runs |
| Optimizer (A, Q and heads) | Adam, lr tuned on validation | Linear problem only |
| Optimizer (B) | Adagrad, fresh state, lr < 0.01 | Pilot the lr |
| Epochs (C) | 100 | Same as earlier runs; confirm in pilot that losses have flattened |
| Split | 80 / 10 / 10 train / validation / test, image-level, stratified by factor combination where possible, fixed seed | Same split for every new condition |
| λ | Chosen from pilot sweep | §10 rule |
| Classification input | $\mu_\phi(x)$ | §3 |
| Q parameterization | To be chosen (e.g. matrix exponential of skew-symmetric, or QR retraction) | Log $\|Q^\top Q-I\|$ |

**Budget:** full C = 4 tasks × 2 conditions × 5 seeds = **40 trainings**, plus C0+Q fits. Earlier runtimes are not a reliable estimate for the supervised variants; measure GPU-hours in the pilot. B (4×2×3×3 = 72 fine-tunes) is optional and not a commitment.

## 11. Interpretation guardrails

| Finding | Supported inference | Not established |
|---|---|---|
| C1 improves native early-prefix accuracy over C0 | Nesting encourages early task accessibility | Novel semantic axes |
| C1 ties C0+Q | Post-hoc rotation is sufficient **for the measured criterion** | Identical representations or no general MRL advantage |
| C1 robustly exceeds C0+Q | Encoder adaptation under nested supervision helps | Uniqueness or identifiability |
| X↔Y reversal repeats across seeds | Task-dependent factor prioritization | Unique basis unless native axes recur |
| Leading subspaces stable; axes unstable | Task-relevant subspace may be reproducible | Coordinate-level privileged basis |
| Shape/rotation not linearly decodable in Phase 0 | Frozen linear rotation is a poor route for those tasks | Their information is absent |
| Prediction improves but reconstruction worsens | Task/reconstruction trade-off | Unqualified improvement |

**Do not conflate:** good LDA fit with proof about cross-entropy; linear $Q$ with nonlinear representation learning; optimal rotation with optimal permutation; axis repeatability with subspace repeatability; agreement in factor *ranking* with agreement in factor *meaning*.

## 12. Code, workflow, and outputs

- **Code location:** all new code in `/home/argha/Mat_embedding/code/NonLinearPB/NonLinearPBExperiments/` (new experiment entry points in `experiments/`, launchers in `scripts/`, tests in `tests/`). Reuse `vae_reproduction/model.py`, `vae_reproduction/data.py`, `experiments/dsprites_vae_reproduction.py`, `scripts/plot_dsprites_figure6.py`; do not replace them.
- **Training:** every training or retraining job runs in a detached tmux session with persistent logs and checkpoints, as in `NonLinearPBREADME.md`. Each launch prints its session name and results folder.
- **Results:** new timestamped root per launch, `/home/argha/Mat_embedding/files/results/exprmnt_<YYYY_MM_DD__HH_MM_SS>/`; smoke tests under `/home/argha/Mat_embedding/files/results/test_runs/`; launch logs under `/home/argha/Mat_embedding/files/results/launch_logs/<session>/`. Never overwrite earlier folders.
- **Authorization:** no training or tests are implied by this plan; each phase is run only after explicit go-ahead.

### Deliverables

1. Immutable config and image-level split manifest; model/code/dataset checksums.
2. Phase 0 table: full-latent linear and nonlinear accuracy, single-coordinate and prefix accuracy, per pretrained seed.
3. A benchmark: original order, permutation, PCA, LDA, per-$k$ oracle, greedy, joint-MRL $Q$.
4. C0/C1 checkpoints for every task/seed, plus C0+Q fits.
5. Native-prefix accuracy curves and probe fits; X↔Y $D(k)$ per seed.
6. Native-order Figure 6, 10×5 factor–coordinate heatmaps, and traversals for **all** seeds.
7. Axis-level seed agreement (over $1..k_t^*$), matched-axis diagnostic, same-task-null principal angles where valid.
8. B results only after C.
9. Report of failures, degeneracies, and the linear vs nonlinear distinction.

**Completion criterion:** a defensible answer to whether C1's early-coordinate organization and X↔Y reversal exceed C0, **C0+Q**, and reconstruction-MRL across all scheduled seeds. Negative results are valid; do not select favorable checkpoints or seeds.

## 13. Implementation: Phase 0 and Experiment A — code written 2026-10-22 (not yet run)

**Files** (all in `/home/argha/Mat_embedding/code/NonLinearPB/NonLinearPBExperiments/`):

| File | Role |
|---|---|
| `vae_reproduction/tasks.py` | Image-level 80/10/10 split (seed 0), task labels from factorial index, rotation 39 classes (39 → 0), heart-only subsets |
| `vae_reproduction/features.py` | Load checkpoint, compute $\mu(x)$ for all images, compare to saved `latent_means.npy` |
| `vae_reproduction/probes.py` | Linear probe (L-BFGS logistic regression to tolerance, no epoch cap), MLP probe (2×256 ReLU), greedy coordinate order, $k^*$ |
| `vae_reproduction/bases.py` | Orthogonal $Q=Q_0\exp(S-S^\top)$, joint-MRL $Q$, per-$k$ oracle, greedy nested, PCA, LDA, permutation, principal angles |
| `vae_reproduction/runs.py` | Run folders, code snapshot, split manifest, provenance |
| `experiments/dsprites_phase0_audit.py` | Phase 0 |
| `experiments/dsprites_rotation_basis.py` | Experiment A (also takes `--checkpoint label=path` for C0+Q later) |
| `scripts/run_phase0_A_tmux.sh` | tmux launcher: `phase0` or `rotation`, extra args passed on |
| `tests/helper_phase0_rotation.py` | Synthetic checks; added to `tests/run_tests.py` |

**Choices made in code (change here if not agreed):**
- **Phase 0 $k^*$ uses the greedy best order, not native order.** Native order of an unsupervised β-VAE is arbitrary, so native-order $k^*$ would count unused coordinates. Native-order curves and $k^*$ are still saved. Greedy order = forward selection by validation linear-probe accuracy.
- $\epsilon = 0.01$ (absolute accuracy) primary; 0.02 and 0.05 also saved. Common $k$ per task = max over the 3 seeds; X/Y common $k$ = max of the two. Written to `k_star.json` as *provisional*.
- Phase 0 tasks: x, y, shape, scale (all shapes), rotation_heart. Single-coordinate map columns: the same five.
- Experiment A tasks: x, y, shape, rotation_heart, x_heart, y_heart (heart X/Y for the rotation geometry comparison, §9).
- **All A methods are evaluated the same way:** a fresh L-BFGS linear probe on the first $k$ transformed coordinates. Trained MRL-head accuracies are saved separately.
- Joint MRL $Q$: Adam lr 0.01, batch 8192, ≤100 epochs, early stopping on validation loss (patience 10); inits identity, random0, random1, random2 (these four runs are the same-task null for principal angles).
- Per-$k$ oracle: inits identity and random0, best validation loss per $k$.
- Greedy nested: one run (seed 0).
- LDA: $S_b v=\lambda S_w v$, ridge $10^{-6}\,\mathrm{tr}(S_w)/d$, top $\min(c-1,10)$ directions, QR in eigenvalue order, rest filled by PCA inside the complement.
- Features are centered by the train mean before any basis is applied.
- `--smoke`: 20000/4000/4000 images per task, 3 epochs for MLP and $Q$ training; numbers are diagnostics only. Smoke runs go to `files/results/test_runs/`.

**Verified on data 2026-10-22:** shape index 2 is the heart (changes under 90° and 180° rotation; the square does not change under 90°). Orientation step is $2\pi/39$.

**Not yet done:** user runs `tests/run_tests.py`; then smoke runs; then full runs after go-ahead.

## 14. Decision 2026-10-22: tasks reduced to X, Y, scale

After Phase 0 on β-VAE seed 0 (`/home/argha/Mat_embedding/files/results/exprmnt_2026_10_22__21_46_28/`, seeds 1–2 cancelled), the active task set is **X, Y, scale** (all shapes). **Shape and rotation are dropped for now.** Defaults in `dsprites_phase0_audit.py` and `dsprites_rotation_basis.py` are set to these three; the shape/rotation code stays and can be selected with `--tasks`.

Phase 0, seed 0 (test set):

| Task | Chance | Linear acc | MLP acc | Linear $R^2$ |
|---|---|---|---|---|
| x | 0.031 | 0.23 | 0.79 | 0.81 |
| y | 0.031 | 0.27 | 0.80 | 0.83 |
| scale | 0.167 | 0.35 | 0.99 | 0.37 |
| shape (dropped) | 0.333 | 0.45 | 0.98 | — |
| rotation, hearts (dropped) | 0.026 | 0.07 | 0.58 | 0.01 |

Active: $z_1, z_4, z_6, z_7, z_9$ ($z_1$ carries rotation, mainly ellipse; corrected 2026-10-22). X and Y are mixed over $z_4, z_7, z_9$; scale is mainly in $z_6$ (nonlinear). Exact-class linear accuracy understates ordered factors, so Experiment A should also report $R^2$ for X, Y, scale.

## 15. Experiment A — reduced first run (agreed 2026-10-22)

**Model:** ordinary β-VAE (β = 4), **seed 0 only**. The encoder has no MRL; MRL acts only on $Q$ and its prefix heads.
**Tasks:** X, Y, scale.
**Methods now:**
1. Original order (no rotation).
2. Best permutation (greedy forward selection on validation).
3. LDA (closed form, seconds; included because it tests whether MRL $Q$ beats a standard rotation).
4. Joint MRL $Q$, 4 starting points (identity, random0, random1, random2).

**Scoring for every method:** first $k$ coordinates, $k=1..10$: fresh linear classifier (test accuracy) and fresh linear regression (test $R^2$).
**Questions:** original vs permutation = gain from reordering; permutation vs MRL $Q$ = extra gain from mixing (rotation); LDA vs MRL $Q$ = is MRL better than a textbook rotation.

**Future plan (not in this run):** PCA, per-$k$ oracle, greedy nested, principal angles, Figure 6 and single-coordinate map for $Q$ coordinates, traversals through $Q$, seeds 1–2, reconstruction-MRL β-VAE as input. Code supports PCA/oracle/greedy/angles via `--methods ... --angles`.
