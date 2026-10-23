# Experiment Plan: Task-Specific Latent Ranking with MRL–β-VAE on dSprites

**Status:** Research design / discussion draft — not a finalized implementation specification.  
**Central objective:** Determine whether a task-specific Matryoshka objective orders latent coordinates by their usefulness for a chosen generative-factor prediction task.

## First task: Reproduce the VAE and β-VAE disentanglement scores

Before adding task supervision or MRL, train an unsupervised **VAE (β=1)** and **β-VAE (β=4)** on dSprites using the selected 2D-shapes architecture and Adagrad setup in §4. Evaluate the Higgins et al. disentanglement metric and compare with **Figure 5 (left)** of the user-provided paper.

| Model | Paper's reported disentanglement metric score |
|---|---|
| VAE, β=1 | 61.58 ± 0.5% |
| β-VAE, β=4 | 99.23 ± 0.1% |
| Ground-truth factors (reference) | 100% |

**Metric protocol (§3 and Appendix A.4):**

1. Choose one of four factors: X-position, Y-position, scale, or rotation. Shape is varied as a nuisance, not used as a classifier target in this paper's metric.
2. Sample image pairs with the chosen factor held fixed and other factors independently varied.
3. Encode images using posterior means and average the absolute latent-coordinate differences over multiple pairs to form one metric example.
4. Train a linear softmax classifier to identify which of the four factors was held fixed, using negative log likelihood and Adagrad with learning rate `1e-2` until convergence.
5. Report accuracy on **5,000 held-out metric examples**; these are aggregated pair-difference examples, not 5,000 individual images. Four balanced targets give a chance accuracy of 25%.

**Paper's score aggregation:** Train 10 model replicas with different seeds, evaluate each replica with 3 classifier-initialization seeds, then discard the bottom 50% of the resulting 30 scores before reporting the remaining results. For reproduction, report this filtered statistic separately from the mean, variability, and raw scores across **all 30 evaluations**. Do not silently apply this filtering to later MRL comparisons. Confirm the definition of the paper's ± statistic before labeling our uncertainty in the same way.

The detailed reproduction pre-plan below supplies proposed choices for settings omitted from the paper. These are our implementation choices, not recovered paper settings.

**Outcome:** Establish a baseline reproduction record with both scores and supporting training/reconstruction diagnostics. These reported scores are comparison targets, not guaranteed outcomes. Subsequent task-specific MRL experiments remain a separate stage; this first stage measures disentanglement, not latent ordering or nonlinear identifiability.

### R0. Scope, environment, and implementation

- Implement one VAE in **PyTorch from scratch**; the VAE and β-VAE differ only in the fixed KL multiplier (`1` versus `4`). Do not modify the reference `disentanglement_lib` checkout.
- Use the selected architecture in §4, with linear posterior mean/log-variance outputs and linear decoder logits. The three decoder hidden layers use Tanh; stable binary cross-entropy with logits implements the Bernoulli negative log likelihood.
- Use existing conda environment **`mrl_env_cuda12`**. Imports and CPU tensor operations were verified. The user confirmed two idle RTX 3090 GPUs with 24 GB each; CUDA visibility inside the execution session still needs verification before benchmarking.
- Keep all new project code, configuration, metric implementation, and tests under `NonLinearPB/NonLinearPBExperiments/`, separate from the existing linear experiments. Proposed subfolders inside that code folder: `vae_reproduction/` for model/data/metric helpers, `experiments/` for the run entry point, and `tests/` for validation. See the code-location rule in [NonLinearPBREADME.md](../NonLinearPBREADME.md).
- Every entry point declares its environment and usage. The experiment entry point owns a complete CONFIG block and logs every resolved setting. No implementation or training is performed by this pre-plan edit.

### R1. Data and sampling

Use the existing archive:

```text
/home/argha/Mat_embedding/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

Verify the expected 737,280 binary 64×64 images, metadata, factor cardinalities, factor-index lookup, and file checksum. The stored factor columns include a constant color factor; exclude color from the metric targets.

**Proposed reproduction choice:** Train the unsupervised models on the full image archive, shuffling once per epoch. No image-level train/validation/test split is used in this stage. This is a documented choice because the paper does not specify the image split, not a consequence of training being unsupervised. Labels never enter the VAE loss.

Metric training, validation, and test sets are separately sampled collections of aggregated image-pair differences. They may draw images from the same archive used for unsupervised training. This measures factor organization on that archive, not encoder generalization to unseen images. Save the sampling seeds and do not reuse the same metric examples across sets.

Preserve the original orientation grid and shape distribution for this reproduction. The rotation-target changes contemplated for later supervised MRL experiments do not alter this first stage.

Keep images in their compact binary/uint8 representation on the CPU and convert each batch to floating point on transfer. Avoid materializing the entire image archive as float32 or copying it unnecessarily into every worker.

### R2. Proposed VAE training configuration

| Setting | Initial choice | Source/status |
|---|---|---|
| Architecture | §4; 10 latent coordinates | Paper Table 1 |
| β | 1 for VAE; 4 for β-VAE | Paper Fig. 5 / Appendix A.4 |
| Optimizer / learning rate | Adagrad / `0.01` | Paper Table 1 |
| Other optimizer settings | Explicitly record PyTorch defaults, including epsilon and initial accumulator | Our choice |
| Batch size | 256 | Our choice; benchmark memory and throughput |
| Training budget | Initial 100 epochs, equal for both models | Our choice; pilot may revise before the full run |
| Precision | Float32; no mixed precision initially | Our choice |
| Posterior sampling | One reparameterized sample per image per update | Our choice |
| Loss reduction | Sum reconstruction over pixels and KL over coordinates, then average over batch | Explicit implementation of the objective |
| Extra regularization | No weight decay, KL warm-up, β schedule, or extra penalties | Our choice |
| Model seeds | Pilot: 0; full run: 0–9 for each β | Paper replica count; our seed values |
| Checkpoints | Every 10 epochs, plus final; include optimizer and RNG state | Our choice |

For a batch of size $B$, minimize:

$$
\mathcal L = \frac{1}{B}\sum_{b=1}^{B}\left[
\sum_{p=1}^{4096}\operatorname{BCEWithLogits}(\ell_{bp},x_{bp})
+ \frac{\beta}{2}\sum_{j=1}^{10}
\left(\mu_{bj}^{2}+\exp(\log\sigma_{bj}^{2})-1-\log\sigma_{bj}^{2}\right)
\right].
$$

Use the same initialization and data-order seed for the paired β=1 and β=4 runs where possible. Record reconstruction NLL, unweighted KL, β-weighted KL, total loss, per-coordinate KL, elapsed time, and GPU memory. Do not normalize BCE over pixels independently of KL.

**Checkpoint policy:** Evaluate the final checkpoint from the fixed budget. Do not select an epoch by the final metric-test score. Inspect training curves and numerical stability during the pilot; if 100 epochs is inadequate, revise and document a common budget before running the remaining seeds. Preliminary pilot metric scores are exploratory; use fresh metric-test examples for the final report after choices are locked.

### R3. Disentanglement metric implementation

Freeze the encoder, use evaluation mode and posterior means, and cache the 10-dimensional means for all archive images in batches. Construct the metric examples from this cache.

For each metric example:

1. Choose a target uniformly from X-position, Y-position, scale, and rotation.
2. Draw **64 image pairs**. Within each pair, sample one value of the target and use it for both images; independently sample all other varying factors, including shape. The fixed value may differ across pairs in the same aggregate. Other factors may coincide by chance; do not force them to differ.
3. Form the 10-dimensional input

$$
z_{\mathrm{diff}} = \frac{1}{64}\sum_{l=1}^{64}
\left|\mu(x_l^{(a)})-\mu(x_l^{(b)})\right|.
$$

4. Label the aggregate with the identity of the held-fixed factor, not its value.

**Proposed settings for unspecified details:**

| Setting | Choice |
|---|---|
| Pair-averaging count | 64 |
| Metric training examples | 10,000; balanced across four targets |
| Metric validation examples | 2,000; balanced and separately sampled |
| Metric test examples | 5,000; balanced and separately sampled (paper test count) |
| Classifier | Linear 10 → 4 logits with bias; softmax cross-entropy |
| Optimizer | Adagrad, learning rate `0.01` (paper) |
| Batch size | 256 |
| Input preprocessing | Raw averaged absolute differences; no extra standardization initially |
| Maximum classifier epochs | 200 |
| Convergence / checkpoint selection | Best validation cross-entropy; stop after 20 epochs without improvement of at least `1e-4` |
| Classifier initialization seeds | 0, 1, 2 for each trained encoder |

Use the same metric pair-sampling seeds across compared models. Hold metric sets fixed across the three classifier initializations for a given comparison so those repeats isolate classifier initialization. Record sampling randomness separately from initialization randomness. Never stop or tune the classifier using test accuracy.

### R4. Validation, benchmark, and one-seed pilot

1. **Implementation validation:** Prepare tests for factor-index lookup and pair constraints, analytic KL, loss reductions, reparameterization/backpropagation, checkpoint reload, and metric train/test separation. Check that a ground-truth-factor metric control is near 100% and that shuffled metric labels produce chance-level performance. Investigate failures before large runs.
2. **GPU benchmark:** On one RTX 3090, run 50 warm-up updates followed by 300 timed updates at batch size 256. Synchronize CUDA around timing and include data loading and optimizer updates. Log peak memory and updates/second; also measure concurrent throughput with one independent process per GPU before extrapolating the full-run time.
3. **Smoke run:** Use `--fast` for a small image subset, short training, and reduced metric sets. Verify losses are finite, gradients propagate, checkpoints reload, and logs/plots/scores are written. Smoke scores are not reproduction evidence.
4. **Pilot:** Train seed 0 at β=1 on GPU 0 and seed 0 at β=4 on GPU 1, with the full dataset and common proposed budget. Inspect curves, reconstructions, latent traversals, and metric-classifier convergence. Evaluate the preliminary disentanglement scores.
5. **Lock the configuration:** Resolve numerical or convergence problems, document any changes, and choose the final budget before the full seed sweep. If the pilot configuration changes, rerun seed 0 under the locked settings.

Testing follows the shared project workflow: supply and read the validation tests for user execution. This document edit does not run tests or launch experiments.

**Timing:** The earlier 1–4 hours per model and 10–40 hours for 20 runs were rough, unmeasured estimates. Replace them with the benchmark. At batch size 256 there are 2,880 updates per epoch, or 288,000 per 100-epoch run. The full experiment has 20 model runs and 60 classifier fits. Estimate wall time using measured concurrent throughput, then add encoding, metric fitting, plotting, and checkpoint overhead.

### R5. Full reproduction and reporting

Run seeds 0–9 for β=1 and β=4 using the locked configuration. Assign independent model runs to the two GPUs; do not initially use distributed training for a single model. Reuse a completed pilot only if it matches the final configuration exactly.

For each model family, retain all 30 metric accuracies. Report:

- Every model/classifier seed score and per-model averages.
- Unfiltered mean and standard deviation, plus variability across the 10 model-seed means. The three classifier repeats are nested within a model seed, not 30 independent model trainings.
- Mean and explicitly labeled standard deviation of the best 15 scores to mirror the paper's bottom-half filtering; do not assert that this reproduces the paper's unspecified ± convention.
- Comparison against 61.58% for VAE and 99.23% for β-VAE, with absolute percentage-point differences.
- Per-factor classifier performance/confusion matrices, training curves, reconstruction examples, per-coordinate KL, and latent traversals.

The initial success criterion is a valid, reproducible metric pipeline and a clear assessment of whether β=4 improves factor separation over β=1. Numerical agreement with Figure 5 is assessed separately. A discrepancy triggers inspection of convergence, metric construction, and documented protocol differences; it is not a reason to tune on final test scores until the desired number appears.

### R6. Outputs and reproducibility record

Keep results outside the code tree, using the shared timestamp convention under the current project root:

```text
/home/argha/Mat_embedding/files/results/exprmnt_{timestamp}/
/home/argha/Mat_embedding/files/results/test_runs/exprmnt_{timestamp}/  # --fast
```

The shared helper currently assumes the older root name `Mat_embedding_hyperbole`; resolve this explicitly during implementation rather than writing results into the code folder. No shared-helper edits are authorized by this documentation-only step.

Each run records the configuration and paper deviations, dataset checksum, package versions, device/driver information, seeds, checkpoints, raw metric arrays, runtime, and a code snapshot. Include the shared mandatory outputs: timestamped training curves, `experiment_description.log`, `results_summary.txt`, `runtime.txt`, and `code_snapshot/`. Store benchmark measurements separately from scientific scores. Preserve prior run artifacts when resuming or reevaluating.

### R7. Implementation and initial two-epoch trial — 2026-10-12

Implemented [the reproduction entry point](../NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py), model/data/metric helpers, and subprocess-based tests under `NonLinearPBExperiments/`. Four correctness tests passed: factorial lookup/pair constraints, analytic KL and loss units, gradients/checkpoint reload, and ground-truth/shuffled-label metric controls.

Ran both β=1 and β=4 for **two full-data epochs**, seed 0, on GPU 0 in `mrl_env_cuda12`. Each encoder was evaluated with all three classifier seeds and the configured 5,000 test metric examples.

| Model | Unfiltered mean metric accuracy (3 classifier seeds) | Training seconds per epoch |
|---|---|---|
| VAE, β=1 | 96.79% | 10.02, 9.58 |
| β-VAE, β=4 | 95.75% | 9.63, 9.60 |

End-to-end runtime was **72 seconds**, including data verification, training, encoding, evaluation, figures, and snapshotting. Outputs:

```text
/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_12__21_51_46/
```

Both models had finite, decreasing epoch-averaged losses and saved the required artifacts. **This validates execution, not Figure 5 reproduction:** the models are undertrained, all six metric classifiers reached their 200-epoch cap, and the paper's VAE/β-VAE score gap is absent in this trial. Inspect classifier convergence and protocol differences during the pilot; do not treat these scores as final results or tune against them to force the paper's numbers.

The measured epoch times supersede the earlier rough training-time estimate for this setup. They imply about 16 minutes of training for 100 epochs per model if throughput stays similar, excluding evaluation/checkpoint overhead. Two-GPU concurrent throughput and the epochs required for convergence remain unmeasured.

### R8. Untrained control and traversal inspection — 2026-10-12

Added an untrained VAE control, enabled by default once per model seed. It uses the same initialization seed, architecture, raw metric features, metric sets, and classifier settings as trained models. The encoder/decoder receive **no optimization updates**; the implementation verifies all their parameters remain unchanged through evaluation. The metric classifier is trained as usual.

Run the control alone with:

```bash
python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --trial --untrained-only --seeds 0 --device cuda:0
```

Use `--skip-untrained` to omit the control. `--classifier-epochs` overrides the metric classifier cap; retain matching classifier budgets when comparing model conditions.

Seed-0 results with the original **200-epoch classifier cap**:

| Condition | Classifier-seed accuracies | Unfiltered mean |
|---|---|---|
| Untrained VAE | 43.26%, 43.74%, 29.24% | 38.75% |
| VAE after 2 epochs | 96.96%, 96.94%, 96.48% | 96.79% |
| β-VAE after 2 epochs | 95.56%, 95.98%, 95.70% | 95.75% |

The untrained control's best-two-of-three mean is 43.50%. The paper reports 44.14 ± 2.5% for an untrained VAE, but its 30-score aggregation differs; this pilot is not a replication of that statistic. All three control classifier fits reached the epoch cap, so classifier convergence remains unresolved. Runtime was 20 seconds. Artifacts:

```text
/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_12__22_17_14/
```

**Visual inspection:** Checked reconstructions and multiple reference traversals from the two-epoch models, alongside an untrained traversal. The untrained decoder produces gray noise. The trained decoders recover approximate position/size as blurry blobs, with poor shape fidelity. Individual coordinate traversals can jointly change position, size, and elongation; these examples do not establish clean one-factor-per-coordinate disentanglement despite high metric accuracy.

A separate synthetic diagnostic using normalized ground-truth X/Y/scale/rotation coordinates scored 100%, while an orthogonally mixed version scored 94.9% (64-pair aggregation, one classifier initialization, 2,000-epoch cap). This illustrates that the chosen metric can remain high for mixed factors; it is not an experiment on trained VAE encoders. Keep traversals and metric controls alongside the scalar score in subsequent interpretation.

### R9. Detached one-seed, 100-epoch pilot — launched 2026-10-12

Started tmux session **`dsprites-seed0-100ep`** using [the pilot launcher](../NonLinearPBExperiments/scripts/run_dsprites_seed0_tmux.sh). Window `vae` runs β=1 on GPU 0; window `beta-vae` runs β=4 on GPU 1. Both use seed 0, the full image archive, and 100 epochs. The metric classifier cap is increased to **2,000 epochs**, retaining the same validation-based stopping rule, for both models and the untrained control. The control runs once in the VAE window before training; it is skipped in the β-VAE window to avoid duplicate evaluation.

```text
VAE results:      /home/argha/Mat_embedding/files/results/exprmnt_2026_10_12__22_22_49/
β-VAE results:    /home/argha/Mat_embedding/files/results/exprmnt_2026_10_12__22_22_53/
Console logs:    /home/argha/Mat_embedding/files/results/launch_logs/dsprites_seed0_2026_10_12__22_22_46/
```

Attach with `tmux attach -t dsprites-seed0-100ep`. Detach with Ctrl-b, then d; switch windows with Ctrl-b, then n. The session retains completed panes and their exit status. Closing a terminal does not terminate these detached jobs. Results remain pending until both run directories record completion; this launch is not a full 10-seed reproduction.

**Recovery record:** The original VAE pane was interrupted after epoch 17 (exit status 130); β-VAE continued. Added `--resume-checkpoint` support and verified exact agreement with uninterrupted training in a CPU test, including optimizer and RNG restoration (five tests now pass). Resumed the VAE from its epoch-10 checkpoint at epoch 11 in window `vae-resumed`. Recovery results are in `/home/argha/Mat_embedding/files/results/exprmnt_2026_10_12__22_29_14/`; console logs are in `/home/argha/Mat_embedding/files/results/launch_logs/dsprites_seed0_2026_10_12__22_29_11/`. Earlier artifacts are preserved. The target remains 100 total epochs, not 100 additional epochs; the uninterrupted β-VAE run is unchanged.

### R10. Figure 6 A/B-style analysis — 2026-10-13

Generated from the completed **100-epoch seed-0** pilot: β=4 uses
`exprmnt_2026_10_12__22_22_53/beta_4_seed_0/checkpoint_final.pt`; β=1 uses
`exprmnt_2026_10_12__22_29_14/beta_1_seed_0/checkpoint_final.pt`. No new training.
The detached tmux session is `dsprites-figure6` (window `figures`); log:
`/tmp/nonlinearpb_figure6_2026_10_13.log`. Figure generation completed in 15.6 seconds.

The [plotting script](../NonLinearPBExperiments/scripts/plot_dsprites_figure6.py)
computes exact full-archive marginal means for position, scale, and rotation;
uses separate red/green/blue curves for square/ellipse/heart; and decodes each
coordinate at +3, +1.5, 0, −1.5, −3 with all other coordinates fixed to a shared
reference image's inferred mean. Reference image index is 19477, selected with
the same seed-2026 convention as the earlier reconstruction figures. Response
limits are shared across panels. Columns are ordered by **ascending full-dataset
mean posterior variance**, with original coordinate IDs retained. This documents
our aggregation choice where the paper caption does not specify aggregation.
All 10 coordinates are shown separately rather than grouping unused latents.

Standalone A, B, and combined A/B PNG/PDF files, numerical responses and
traversals, provenance, runtime, and relevant plotting-code snapshots are in:
`/tmp/nonlinearpb_figure6_2026_10_13/`. This permitted temporary output destination
keeps results outside the code repository; it is not the long-term results archive.
The focused [factor-aggregation test](../NonLinearPBExperiments/tests/helper_figure6.py)
was created and read for coverage; it was not run, following the shared user-run
test convention. The generated plots were inspected visually.

## 1. Research question and hypothesis


**Question:** Holding the data and architecture fixed, does changing the supervised prediction objective change which generative factor is emphasized in the earliest coordinates of a VAE's learned latent representation?

Use the same dSprites images for three tasks:

1. **X-position** prediction (32 discrete values).
2. **Y-position** prediction (32 discrete values).
3. **Rotation** prediction (40 nominal orientation values; see symmetry caveat below).

**Hypothesis:** Under an X-specific MRL loss, early latent coordinates should predict X well; under a Y-specific loss, they should predict Y well; under a rotation-specific loss, they should predict rotation well. This is **task-specific information prioritization**, not automatically proof of complete factor ordering or disentanglement.

No ground-truth ranking of the generative factors is assumed. Task relevance defines the desired priority.

## 2. Dataset

**dSprites / 2D shapes**: binary 64×64 images formed by varying five generative factors:

- Shape (3 categories)
- Scale (6 values)
- Rotation (40 values)
- X-position (32 values)
- Y-position (32 values)

For the later task-specific MRL experiments, use the *same* train/validation/test image split for every experimental condition. The first reproduction stage uses the full-archive protocol in R1 and is reported separately. Retain generative-factor annotations for the supervised task objectives and evaluations.

**Rotation caution:** Rotation can be ambiguous because certain shapes are rotationally symmetric. Do **not** assume each of the 40 nominal angles is visually identifiable for each shape. Before finalizing this task, consider a symmetry-aware target, or restrict the initial rotation experiment to a suitable asymmetric shape and carefully check identifiability. Retain the full dataset for X/Y experiments.

## 3. Main models and essential controls

Three primary model families:

| Family | Objective | Task label in training? | Nested task heads? |
|---|---|---|---|
| **VAE** | Image reconstruction + KL (β=1) | No | No |
| **β-VAE** | Image reconstruction + β·KL (β>1) | No | No |
| **MRL–β-VAE** | β-VAE objective + task-specific prefix losses | Yes | Yes |

**Essential no-MRL supervision control:** Train a **supervised β-VAE** with the same task and the same full-latent prediction loss, but *without* nested prefix losses. This separates the effect of MRL from the effect of supplying target labels. The experiment therefore has **three model families, plus an important supervised β-VAE control**.

**Additional strong control:** Post-hoc coordinate permutation of the trained supervised β-VAE. Rank coordinates on validation data, then evaluate reordered prefixes on held-out test data. This asks whether jointly trained MRL beats simple after-the-fact sorting.

## 4. Architecture and proposed training objectives

Begin with a common encoder/decoder architecture and **10 latent coordinates** in all conditions. The uploaded β-VAE paper uses 10 latents for its 2D-shapes experiment. Hold model capacity, training data, optimization budget, and evaluation protocol fixed where possible.

**Selected architecture and optimizer:** Use the paper's 2D-shapes VAE setup for all conditions.

| Component | Specification |
|---|---|
| Input | Binary 64×64×1 image, flattened to 4,096 values |
| Encoder | Fully connected 4,096 → 1,200 → 1,200; ReLU hidden activations |
| Posterior | Separate 10-dimensional mean and log-variance outputs |
| Latent | 10 coordinates; sample using reparameterization during VAE training |
| Decoder | Fully connected 10 → 1,200 → 1,200 → 1,200 → 4,096; Tanh hidden activations |
| Observation model | Bernoulli likelihood for the binary pixels |
| Optimizer | Adagrad, learning rate `1e-2` |

The Bernoulli output must parameterize valid pixel probabilities (or equivalent logits for a numerically stable binary cross-entropy loss). Tanh applies to the decoder hidden layers.

The encoder produces `q_φ(z | x)`; use its mean `μ_φ(x)` for deterministic probing and latent traversal reference codes. The decoder reconstructs the image from a sampled latent during VAE training.

### 4.1 Base VAE/β-VAE

Minimize:

```math
\mathcal L_{\beta\text{-VAE}}
= \mathcal L_{\mathrm{recon}}(x,\hat x)
+ \beta D_{\mathrm{KL}}\big(q_\phi(z\mid x)\|\mathcal N(0,I)\big).
```

- VAE: β=1.
- β-VAE: β>1, selected using validation data and a small hyperparameter sweep.

### 4.2 Supervised β-VAE (non-MRL control)

```math
\mathcal L_{\mathrm{sup}}
= \mathcal L_{\beta\text{-VAE}}
+ \lambda\,\mathcal L_t\big(h(z_{1:10}),v_t\big).
```

Use a task prediction head on **all 10** latent coordinates; train separate models for X, Y, and rotation. Keep this supervised loss comparable to the MRL condition.

### 4.3 Task-MRL β-VAE

```math
\mathcal L_{\mathrm{task\text{-}MRL}}
= \mathcal L_{\beta\text{-VAE}}
+ \lambda\sum_{k\in\mathcal K}w_k\,
\mathcal L_t\big(h_k(z_{1:k}),v_t\big),
\qquad \mathcal K=\{1,2,4,6,8,10\}.
```

- A short prefix is forced to predict the selected task, creating pressure for early coordinates to contain relevant information.
- Initial tasks: X-position, Y-position, and (after resolving symmetry) rotation.
- Task heads may be independent or weight-tied. **Decide and document this before implementation.**
- Choose task loss (e.g. cross-entropy for discrete factor values), task weight λ, prefix weights `w_k`, and β using validation. Make sure changing the number of heads does not accidentally increase the effective supervision strength without an appropriate control (e.g. normalize prefix weights).

**Unsettled design decision:** This plan primarily uses **nested task prediction**, not nested reconstruction. An optional *unsupervised nested-reconstruction β-VAE* can be added later to distinguish reconstruction-derived ranking from task-derived ranking.

## 5. Training matrix

| Training run | X-position | Y-position | Rotation |
|---|---|---|---|
| Unsupervised VAE | Evaluate same model | Evaluate same model | Evaluate same model |
| Unsupervised β-VAE | Evaluate same model | Evaluate same model | Evaluate same model |
| Supervised β-VAE, full-head only | Train separately | Train separately | Train separately |
| Task-MRL β-VAE, nested heads | Train separately | Train separately | Train separately |

- Reuse unsupervised models across target evaluations.
- Try a small number of seeds to validate the pipeline; run **at least 5 seeds** for the primary comparison.
- Evaluate on a held-out test set, never use test results to choose coordinate permutations or hyperparameters.

## 6. Primary evaluation: Does the *prefix order* help?

After model training:

1. Freeze each encoder.
2. Extract the latent means `μ_φ(x)`.
3. For each prefix `z_{1:k}`, fit an **identical low-capacity evaluation probe** (on train data) to predict the designated factor.
4. Report test performance for all `k=1,…,10` (not only the trained prefix sizes).
5. Use the same probe family and evaluation protocol for all model conditions.

```math
P_t(k)=\mathrm{Performance}\big(\mathrm{Probe}_k(z_{1:k}),v_t\big).
```

Produce **prefix-performance curves** for X, Y, and rotation. MRL is successful on this criterion if it achieves strong held-out performance using fewer early coordinates than well-controlled non-MRL baselines.

Evaluate the supervised β-VAE both in its **native coordinate order** and with **validation-based post-hoc permutation**. Comparing only to unpermuted baseline coordinates would be too weak.

Potential summaries: test performance at `k=1`, area under the prefix-performance curve, and smallest `k` reaching a specified fraction of full-latent performance. Set definitions before inspecting final test results.

## 7. Interpretability: Latent traversal is a core figure

For each trained model, choose representative held-out images. Encode an image into its latent mean `μ(x)`, then change **one coordinate at a time** while fixing all others:

```math
z^{(i)}(a)=\big(z_1,\ldots,z_{i-1},a,z_{i+1},\ldots,z_{10}\big),
\qquad \hat x^{(i)}(a)=g_\theta\big(z^{(i)}(a)\big).
```

- Traverse reasonable values within the coordinate's **empirical encoded range**, possibly with a standardized interval for consistent displays.
- Show a row for each `z_i` and columns for traversal values.
- Observe whether changing `z_i` shifts X, shifts Y, rotates the shape, changes scale/shape, or changes **several factors at once**.
- Use several reference examples and seeds to avoid overinterpreting a single image.

### Key visual comparison

| Training objective | Question for early coordinates |
|---|---|
| X-specific MRL | Does changing `z₁` move the object horizontally? |
| Y-specific MRL | Does changing `z₁` move the object vertically? |
| Rotation-specific MRL | Does changing an early coordinate rotate the object? Could rotation require multiple coordinates? |

**Important:** This table describes a *hypothesis*. A factor may span more than one coordinate, especially periodic rotation. Traversal is qualitative evidence about a coordinate's effect, **not** a substitute for predictive ranking metrics.

## 8. Factor–coordinate predictability map

For each trained model and each coordinate `z_i`, measure held-out predictability of the factors using consistent **single-coordinate probes** (or a carefully justified mutual-information estimator):

```text
               X     Y    Rotation   [optionally shape, scale]
z1            ...   ...     ...
z2            ...   ...     ...
...
z10           ...   ...     ...
```

Create heatmaps in the **original learned coordinate order**. Compare X-, Y-, and rotation-task models. This tests whether target relevance moves toward the left of the matrix when the objective changes.

Also inspect whether one coordinate predicts multiple unrelated factors, but treat comprehensive disentanglement metrics such as MIG/DCI as **secondary diagnostics**, not the central result.

## 9. Planned figures

1. **Training scheme:** shared encoder/decoder and full-latent versus nested task heads.
2. **Prefix curves:** performance vs. number of first latent coordinates, separated by task; include VAE, β-VAE, supervised β-VAE, MRL–β-VAE, and the post-hoc sorting control.
3. **Latent traversal grids:** compare the meaning of early coordinates between the X, Y, and rotation objectives.
4. **Coordinate–factor heatmaps:** target predictability for each `z_i`, displayed in learned order.
5. **Optional:** seed-to-seed variability and reconstruction/trade-off diagnostics.

## 10. What conclusions would be justified?

**Positive result:** MRL achieves better low-`k` prediction than supervised β-VAE (including post-hoc permutation), and traversal/heatmaps show that the earlier coordinates preferentially reflect the task target. This supports **task-specific latent prioritization**.

**Limited result:** MRL beats the baseline in its native arbitrary coordinate order but not the optimally permuted version. This suggests MRL may primarily learn an ordering rather than fundamentally different latent factors.

**Negative result:** Strong task prediction is only available from larger prefixes, or early dimensions mix multiple factors. This means the selected objective did not reliably produce the intended coordinate-level ranking.

**Do not overclaim:** An accurate `z₁` task predictor does not by itself prove factor disentanglement, complete ordering of all five factors, seed-stable semantics, or a unique objective-specific privileged basis.

## 11. Decisions to resolve together before writing code

1. Confirm **task-supervised nested MRL** as the primary version, with optional reconstruction-nested MRL later.
2. Choose exact rotation treatment for symmetric shapes.
3. Define supervised task head(s): linear vs. small MLP; independent vs. weight-tied prefixes.
4. Choose β, λ, and prefix weights; normalize objective scales.
5. Architecture and optimizer are selected in §4; choose the training budget and use the same encoder/decoder for all models.
6. Choose classification versus regression for discrete positional factors; decide evaluation metric(s).
7. Specify train/validation/test strategy, seed count, and held-out evaluation probes.
8. Agree on the primary success criterion *before looking at test outcomes*.

## 12. Reference and status

- Higgins et al., **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework** (user-provided paper), especially §2, §4.2, Fig. 6 and Appendix A.1.
- Research scratch supplied by the user: **Task-Specific Ranked Disentanglement**, developed further in our discussion.

**This is an experimental proposal, not a claim that MRL has already demonstrated task-specific factor ordering.**
