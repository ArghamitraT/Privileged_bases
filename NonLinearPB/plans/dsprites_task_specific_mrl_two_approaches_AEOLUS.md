# dSprites Task-Specific Matryoshka Classification: Frozen-Basis vs Joint Fine-Tuning

**Status:** Experimental design and implementation plan; no training launched by this document.  
**Primary question:** Can Matryoshka classification training organize latent information for a chosen task, and how does task-specific reorganization of a *fixed* β-VAE representation compare with end-to-end adaptation?

## 1. Background and starting point

We have already trained VAE (β=1), β-VAE (β=4), reconstruction-MRL VAE, and reconstruction-MRL β-VAE on dSprites, with 10-dimensional latents, seeds 0–2, 100 epochs, and Figure 6-style factor responses/traversals. **Do not repeat that experiment by default.** Use the existing ordinary β-VAE checkpoints as the starting representations for the new experiments; record their paths, hashes, β, and model seeds. Previous classification design was a proposal, not completed training.

The dSprites factors are shape (3), scale (6), rotation (40 nominal states), horizontal X-position (32), and vertical Y-position (32). We now focus on **three separate classification tasks**, trained individually:

- **X-position:** 32 classes.
- **Shape:** 3 classes.
- **Rotation:** nominally 40 classes; see symmetry control below.

Each task uses the same underlying images and the corresponding factor label. No labels are used in the original β-VAE pretraining. Our claim is **task-specific early-prefix accessibility/organization**, not automatically full disentanglement or a uniquely identifiable basis.

## 2. Two main experiments

### Experiment A — Frozen β-VAE + learnable orthogonal task basis + MRL classifiers

For pretrained encoder \(E_{\phi_0}\), use posterior means to define a common frozen reference representation:

\[
 r(x)=\mu_{\phi_0}(x)\in\mathbb R^{10},\qquad
 z_t(x)=Q_t^\top r(x),\quad Q_t^\top Q_t=I_{10}.
\]

The **encoder stays frozen**. Train a separate task-specific orthogonal \(Q_t\) and a set of linear prefix-classification heads \(h_{t,k}\). No decoder is needed for the classifier loss; retain the pretrained decoder untouched for reconstruction diagnostics. An orthogonal transformation preserves Euclidean distances and full-vector information, but can change what information lies along early axes.

\[
\boxed{\mathcal L_{A,t}^{\mathrm{MRL}}=
\frac{1}{|\mathcal K|}\sum_{k\in\mathcal K}
\operatorname{CE}\!\left(h_{t,k}((Q_t^\top r(x))_{1:k}),y_t\right)}.
\]

**Question A:** Can we learn a task-specific ordered *linear basis* in a shared pretrained representation without updating the underlying encoder?

### Experiment B — Joint β-VAE fine-tuning + MRL classifiers

Start from **the same pretrained β-VAE checkpoint** for fair comparison, but now update encoder \(\phi\) and decoder \(\theta\), together with task-specific linear prefix heads. There is **no \(Q_t\) in the main B condition**; the encoder itself is allowed to reorganize the coordinates.

\[
 z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I),
\]
\[
\boxed{\mathcal L_{B,t}^{\mathrm{MRL}}
=R_{10}(x;\phi,\theta)+4K(\phi)
+\lambda\frac{1}{|\mathcal K|}\sum_{k\in\mathcal K}
\operatorname{CE}\!\left(h_{t,k}(z_{1:k}),y_t\right)}.
\]

For initial implementation, classification may use the posterior mean \(\mu_\phi(x)\) rather than sampled \(z\), while the VAE reconstruction uses sampled \(z\). **Choose and lock this convention before training; use the same convention across matched B controls.** If using posterior means, substitute \(\mu_\phi(x)\) in the CE term above. Keep full-latent Bernoulli reconstruction and analytic KL, β=4, and no capacity schedule, as in the prior reproduction.

**Question B:** Does allowing the generative representation to change lead to better task-specific early-prefix performance, and what happens to reconstruction and factor structure?

**Interpretation warning:** Comparing A and B is **not a controlled estimate of the sole effect of freezing the encoder**, because A has an orthogonal transform and B has encoder/decoder fine-tuning. Their comparison is between two distinct mechanisms. Use the within-experiment controls below to isolate MRL.

## 3. Essential non-MRL controls

For *each task* and *each pretrained β-VAE model seed*, train:

| Condition | Encoder | Learnable basis | Classification training | Purpose |
|---|---|---|---|---|
| **A0: Frozen + Q + full-head** | Frozen | Orthogonal \(Q_t\) | One linear 10-D classifier | Control for trainable \(Q\) without nested losses |
| **A1: Frozen + Q + MRL** | Frozen | Orthogonal \(Q_t\) | Linear heads on all prefixes | Main experiment A |
| **B0: Joint β-VAE + full-head** | Trainable | None | Full-vector classification + reconstruction/KL | Control for supervised encoder adaptation |
| **B1: Joint β-VAE + MRL** | Trainable | None | Prefix classification + reconstruction/KL | Main experiment B |

Also evaluate **F0: frozen β-VAE + full-vector linear probe** to establish whether each task is actually decodable from the pretrained representation. This probe is not a substitute for A0/B0.

**Important degeneracy of A0:** With a full-dimensional *unregularized* linear head, \(h(Q^\top r)\) is equivalent to a linear head directly on \(r\). The full-vector prediction loss therefore does not uniquely determine \(Q\), and A0's learned basis should **not** be interpreted geometrically. Use it only as a full-vector performance control. For the ordering comparison, additionally use **validation-chosen coordinate permutations** of the frozen baseline and **matched trainable Q with prefix-specific non-nested baselines** where relevant.

**Important degeneracy of A1:** With independent unconstrained linear prefix heads and only a single task, an orthogonal \(Q\) may rotate one predictive direction into \(z_1\) even without a uniquely identified basis. Check seed stability and use full-vector performance and baseline comparisons before making identifiability claims.

## 4. Locked and proposed defaults

Keep the original dSprites reproduction architecture where it matters: flattened 4096 input; encoder 4096→1200→1200→10 mean/log-variance with ReLU encoder hidden layers; decoder 10→1200→1200→1200→4096 with Tanh hidden layers, Bernoulli logits; β=4; Adagrad initial learning rate 0.01; 10 latent coordinates; no capacity schedule, KL warm-up, or β annealing.

**Suggested starting choices (must be recorded and validated before full runs):**

| Item | Initial choice |
|---|---|
| Pretrained β-VAE model seeds | 0, 1, 2 existing checkpoints; extend later if needed |
| New classifier/Q/fine-tune seeds | 0, 1, 2 *per pretrained checkpoint*; distinguish seed levels |
| Prefixes | \(\mathcal K=\{1,2,\ldots,10\}\), equal weighting |
| Heads | Independent **linear** softmax heads (with bias), appropriately sized for 32/3/rotation classes |
| Orthogonal Q | Initialize identity, then also test random orthogonal initialization; use an exact orthogonality-preserving parameterization or projection/retraction and log \(\|Q^\top Q-I\|\) |
| A optimizer | Start with Adam for Q/heads (separate from fixed pretrained optimizer); tune learning rate on validation |
| B optimizer | Start with Adagrad 0.01 for comparability; tune a smaller fine-tuning rate on validation if unstable |
| Classification CE weighting | Mean over prefixes; compare matched full-latent CE with same \(\lambda\) |
| Fine-tune budget | Pilot first, then lock identical update/epoch budgets for B0/B1 |
| Validation | Fixed train/validation/test image split shared across all new conditions; no test-driven tuning |
| Device | Existing CUDA/PyTorch environment, one process per GPU as memory permits |

**Avoid a false fairness claim:** A and B have different trainable parameter counts and computational costs. Report optimizer settings, updates, elapsed time, and GPU memory. Compare A1 to A0 and B1 to B0 for MRL effects, then compare A1 to B1 as mechanisms.

### Rotation symmetry: resolve before starting full runs

The 40 nominal dSprites rotation labels are not distinguishable from images for symmetric objects (notably squares and ellipses). Exact 40-class prediction over all shapes has irreducible ambiguity. **Primary rotation pilot:** restrict to an asymmetric shape (heart), after verifying any residual periodic equivalences. Alternatively define equivalence-class / symmetry-aware losses by shape. Keep **X and shape tasks on all shapes**; record that the rotation task uses a different evaluation subset if selected. Do not silently treat its accuracy as directly comparable to X/shape chance levels.

## 5. Evaluation, with separation of train heads and probes

1. **Full-latent task feasibility:** Held-out test accuracy for the full-vector linear probe using each frozen pretrained β-VAE encoder. If a task is not represented, an orthogonal \(Q\) cannot recover it.
2. **Prefix accuracy curves:** For every method/task, evaluate accuracy for \(k=1,\ldots,10\) using the learned training heads **and** (for stronger cross-condition comparability) newly fitted, fixed-capacity evaluation probes on frozen final representations. Keep the protocols identical.
3. **Summary of ranking:** Accuracy at \(k=1\); area under the prefix accuracy curve; smallest \(k\) achieving a pre-registered fraction (e.g., 95%) of the corresponding full-10-D performance. Baseline comparisons include native order and validation-based *post-hoc coordinate permutations*; never choose permutations on test data.
4. **Coordinate-factor map:** For each native \(z_i\), use a matched single-coordinate probe (or carefully justified MI estimator) to quantify X, rotation, shape, and optionally Y/scale. Keep columns in actual learned prefix order. Note that a single coordinate can contain multiple factors.
5. **Across-seed stability:** Compare native-coordinate factor profiles and early-\(k\) subspaces, distinguishing variance across classifier/Q seeds *within the same pretrained encoder* from variance across different pretrained β-VAE model seeds. Do not treat nested runs as independent pretrained representations.
6. **Geometry for A:** Compare principal angles between \(\mathrm{span}(Q_X[:,1:k])\), \(\mathrm{span}(Q_R[:,1:k])\), and \(\mathrm{span}(Q_S[:,1:k])\) **within the same pretrained encoder**. Do not compare raw Q matrices across independently pretrained β-VAEs without registration/alignment. A signed/permuted basis or equivalent classifier parameterization may yield the same function; geometry must be interpreted with care.
7. **Generative side effects for B:** Report reconstruction BCE, total/per-coordinate KL, number of active latents, and optional MIG/DCI relative to the pretrained β-VAE. Compare full-vector accuracy so prefix improvements are not produced merely by poorer overall solutions.
8. **Latent traversals:** For B, decode single-coordinate traversals through the corresponding jointly fine-tuned decoder. For A, a coordinate \(z_i\) is in transformed space: vary \(z_i\), map back with \(r=Q_t z\), and decode through the original frozen β-VAE decoder. Use comparable empirical traversal ranges, multiple reference images, and native coordinate labels. Traversal is qualitative; do not interpret it alone as proof of ranking.

## 6. Main figures

- **Fig. 1: Two mechanisms** — frozen encoder→orthogonal Q→MRL heads versus trainable β-VAE→MRL heads.
- **Fig. 2: Prefix accuracy** — X, shape, and rotation panels; A0/A1/B0/B1 and meaningful frozen/permutation reference curves. Show seed means and model-seed spread.
- **Fig. 3: Native-order factor-coordinate maps** — each task, each approach, across representative seeds. No sorting by factor identity or posterior variance.
- **Fig. 4: Task-conditioned latent traversals** — same images and comparable ranges; A traversals decoded via \(Q\), B traversals using task-specific decoder.
- **Fig. 5: Cross-task geometry and seed stability** — principal angles for A; separate pretrained-seed and classifier-seed variation.
- **Fig. 6: Reconstruction/factor-retention tradeoff** — B compared to original pretrained β-VAE.

## 7. Hypotheses and legitimate conclusions

- **H1 (frozen-space organization):** A1 reaches stronger early-prefix held-out prediction than a frozen native-order baseline; assess whether it also beats a post-hoc permutation baseline. A success shows ordered linear access to existing information, *not* creation of new information.
- **H2 (joint adaptation):** B1 improves early-prefix performance relative to B0 at comparable full-vector accuracy, while documenting any reconstruction/disentanglement cost.
- **H3 (objective specificity):** Q-based leading subspaces for X, rotation, and shape differ **in the same pretrained reference space**. Semantic heatmaps/traversals indicate which information each objective prioritizes.
- **H4 (repeatability):** Similar task-conditional leading subspaces are recovered across independent classifier/Q initialization seeds within an encoder; evaluate independently across pretrained seeds only after alignment or via task semantics/performance.

A strong low-\(k\) classification result by itself **does not** establish unique directions, full ordering of all generative factors, disentanglement, or transfer compatibility. If A1 is no better than post-hoc sorting, the benefit may be principally reindexing. If B1 is better than A1, this can reflect nonlinear encoder changes, optimization differences, or greater capacity, not simply 'MRL is better.'

## 8. Implementation phases / checkpoints

**Phase 0 — Input audit:** Locate and load existing β-VAE checkpoints (model seeds 0–2), verify encoder/decoder architecture and dSprites archive, compute frozen full-vector classification baseline for X, shape, rotation, and decide the rotation protocol.

**Phase 1 — Frozen experiment A:** Implement orthogonal Q, independent linear prefix heads, A0/A1, prefix-probe evaluation and native-order heatmaps. Run a tiny smoke test, then one pretrained model seed / several classifier seeds before expanding.

**Phase 2 — Joint experiment B:** Reuse the same checkpoint; implement B0/B1 with identical optimizer/data/β and comparable CE scaling. Confirm non-MRL loss reproduces pretrained β-VAE objective plus full-head CE. Pilot numerical stability, then matched seeds.

**Phase 3 — Analysis:** Prefix curves, post-hoc permutation controls, factor profiles, appropriate traversals, geometric comparisons, reconstruction changes, and seed-stability decomposition. Record failures and all runs; avoid score filtering and post-hoc seed selection.

**Phase 4 — Expand:** Increase model/classifier seeds and test alternate prefix schedules, Q initialization, and λ only if the pilot supports a worthwhile hypothesis. Add the joint multi-target X+shape or X+rotation loss-weight experiment later; this document's **primary tasks are separate single-target classifications**.

## 9. Code/workflow requirements

Reuse existing code rather than replacing the β-VAE reproduction:

- `NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py`
- `NonLinearPB/NonLinearPBExperiments/vae_reproduction/model.py`
- `NonLinearPB/NonLinearPBExperiments/vae_reproduction/data.py`
- `NonLinearPB/NonLinearPBExperiments/scripts/plot_dsprites_figure6.py`

Preserve existing checkpoints and output folders. Put new task-classification scripts and tests alongside the existing experiment code, and timestamp every result root. Save full configs, dataset/checkpoint hashes, source snapshots, task-label policy, model/Q/head states, seeds at both levels, learning curves, per-prefix metrics, native-order matrices, figure source data, and failure logs. Follow the project's detached `tmux` workflow for user-authorized training. No training or unit tests are implicitly authorized by this plan; prepare them for explicit review/execution.

## 10. Decisions to lock before a large run

1. Rotation treatment: heart-only versus shape-dependent symmetry-aware labels/loss.
2. Exact orthogonal Q parameterization, optimization, and initialization test.
3. Whether B classification uses posterior mean or reparameterized samples (prefer mean for deterministic classifiers; document and match across controls).
4. Classification λ and optimization budgets, chosen on validation without test leakage.
5. Classification data split and whether factor-combination extrapolation is a separate challenge evaluation.
6. Which seed comparisons are within one frozen pretrained model and which span different pretrained models.

**Decision criterion:** Proceed from the pilot only after (a) frozen full-vector task information exists, (b) training/control losses are verified, (c) early-prefix curves are reliable across new seeds, and (d) rotation labels have a defensible interpretation.
