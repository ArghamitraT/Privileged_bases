# dSprites: VAE / β-VAE versus reconstruction Matryoshka, three model seeds

**Date:** 2026-10-13.  
**Status:** Full training and evaluation completed, verified 2026-10-14: 12 models, 36 classifier fits, six Figure 6 pairs. Detailed cross-seed visual interpretation remains pending. Smoke validation is in §14; full launch/results are in §15.  
**Framework:** PyTorch throughout.  
**Execution:** All training/retraining and subsequent evaluation run inside detached tmux, following [NonLinearPBREADME.md](../NonLinearPBREADME.md#training-and-retraining-always-use-tmux).

## 1. Purpose and scope

Compare ordinary unsupervised VAE and fixed-β β-VAE with reconstruction-based Matryoshka versions, using the same architecture, optimizer, data, budgets, model seeds, and evaluation procedure.

The question is whether requiring every leading latent prefix to reconstruct the image changes the learned latent organization, factor selectivity, and seed variability. This experiment introduces ordering pressure through **reconstruction utility**, without prescribing which generative factor belongs in which coordinate.

All four conditions remain unsupervised. Ground-truth factors are used only for evaluation. No X/Y/rotation prediction heads or factor-label training losses are included. Task-specific factor-supervised MRL remains a later, separate experiment in [the task-specific dSprites plan](dsprites_task_specific_mrl_experiment_plan_AEOLUS.md).

This experiment uses the existing original-style **Adagrad / Tanh / default initialization** setup. It does not use capacity control, Adam, Xavier initialization, ReLU decoder layers, or a β schedule. Existing optional variants remain available for separate experiments; their defaults and past artifacts must be preserved.

## 2. Agreed experiment matrix and counts

| Condition ID | Name | Reconstruction training | Fixed β |
|---|---|---|---:|
| `vae` | Ordinary VAE | Full 10-dimensional latent only | 1 |
| `beta_vae` | Ordinary β-VAE | Full 10-dimensional latent only | 4 |
| `mrl_vae` | Matryoshka VAE | Average of ten prefix reconstruction losses | 1 |
| `mrl_beta_vae` | Matryoshka β-VAE | Average of ten prefix reconstruction losses | 4 |

For **each condition**:

- Model initialization/training seeds: **0, 1, 2**.
- Metric-classifier initialization seeds: **0, 1, 2 for each trained model**.
- Three independently trained encoders, each with three classifier fits: **nine metric scores per condition**.
- Retain all nine scores. No bottom-half filtering, best-half reporting, or selection of the cleanest model seed.

Whole experiment: **12 model trainings and 36 metric-classifier fits**. Classifier repetitions are nested within a trained encoder; they are not nine independently trained encoders.

Train all twelve models fresh under the locked specification. Previous seed-0 pilot runs supply historical context, not substitutes for a new matched comparison. Skip the entry point's automatic untrained-control fits for this experiment so the agreed total remains 36. Additional controls require separate scope and accounting.

## 3. Data and preprocessing

Use the same existing archive as the reproduction experiment:

```text
/home/argha/Mat_embedding/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

| Item | Specification |
|---|---|
| Images | 737,280 binary grayscale images, each 64×64×1 |
| Input | Flatten to 4,096 values; convert to float32 per batch |
| Factors | Shape 3, scale 6, rotation 40, X-position 32, Y-position 32 |
| Color | Constant; not a metric target |
| Training data | Full archive, shuffled each epoch |
| Preprocessing | No augmentation, centering, or additional normalization |
| Labels during training | None |
| Dataset checksum | `857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2` |

Validate image shape, binary values, factor cardinalities, and exact factorial indexing before training. Store images compactly on CPU and avoid full-archive float32 copies.

Preserve the current protocol: no image-level reconstruction train/validation/test split. Independently sampled metric examples can draw from the same image archive used for training. Results describe latent organization on this archive; they do not establish encoder generalization to unseen images or factor combinations.

## 4. Architecture, posterior, and decoder

All conditions use exactly the same parameterized architecture:

```text
Encoder:
  4096 → Linear 1200 → ReLU → Linear 1200 → ReLU
                                             ├→ Linear 10 → posterior mean
                                             └→ Linear 10 → posterior log variance

Decoder:
  10 → Linear 1200 → Tanh → Linear 1200 → Tanh
     → Linear 1200 → Tanh → Linear 4096 → pixel logits
```

Every linear layer includes biases. Posterior heads and final decoder logits have no activation. Use PyTorch's default `nn.Linear` initialization, including its default biases; do not introduce special MRL initialization.

The posterior and prior are:

$$
q_\phi(z\mid x)=\mathcal N\left(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right),
\qquad p(z)=\mathcal N(0,I_{10}).
$$

Predict $\ell(x)=\log\sigma^2(x)$ without positive clipping. For each image and update, draw one noise vector and one full latent sample:

$$
\epsilon\sim\mathcal N(0,I_{10}),\qquad
z=\mu(x)+\exp\left(\tfrac12\ell(x)\right)\odot\epsilon.
$$

The decoder produces 4,096 logits. Sigmoid defines a Bernoulli probability per pixel. Display probabilities as grayscale in reconstructions/traversals; do not sample or threshold pixels for the existing diagnostic plots.

## 5. Ordinary VAE / β-VAE objectives

For a batch of $B$ images, the full-latent reconstruction loss is:

$$
R_{10}=\frac1B\sum_{b=1}^{B}\sum_{j=1}^{4096}
\operatorname{BCEWithLogits}\left(g_\theta(z_b)_j,x_{bj}\right).
$$

The analytic full-posterior KL is:

$$
K=\frac{1}{2B}\sum_{b=1}^{B}\sum_{i=1}^{10}
\left(\mu_{bi}^2+\exp\ell_{bi}-1-\ell_{bi}\right).
$$

Ordinary training minimizes:

$$
\mathcal L_{\mathrm{ordinary}}=R_{10}+\beta K,
\qquad \beta\in\{1,4\}.
$$

Reconstruction sums over pixels, KL sums over coordinates, and both average over images. Do not normalize BCE by the pixel count independently of KL. One sampled latent per image estimates the reconstruction expectation; KL is computed analytically.

## 6. Exact Matryoshka objective

### Prefixes and shared decoder

Train **every prefix**:

$$
\mathcal P=\{1,2,3,4,5,6,7,8,9,10\}.
$$

For each sampled full latent vector, construct:

$$
\tilde z^{(k)}=(z_1,\ldots,z_k,0,\ldots,0).
$$

Use the **same 10-input decoder** for all prefixes:

$$
a^{(k)}=g_\theta(\tilde z^{(k)}),
\qquad \hat x^{(k)}=\operatorname{sigmoid}(a^{(k)}).
$$

Zeros are literal latent-coordinate zeros. Do not use separate prefix decoders, prefix-specific biases/heads, posterior-mean imputation, or sampled prior values for omitted coordinates. All ten prefixes reconstruct the **same original input image**.

### Scalar losses and one KL penalty

Define a scalar reconstruction loss for each prefix:

$$
R_k=\frac1B\sum_{b=1}^{B}\sum_{j=1}^{4096}
\operatorname{BCEWithLogits}(a_{bj}^{(k)},x_{bj}).
$$

The agreed objective is:

$$
\boxed{
\mathcal L_{\mathrm{MRL}}=\frac1{10}\sum_{k=1}^{10}R_k+\beta K
},\qquad \beta\in\{1,4\}.
$$

Requirements:

1. **Average the ten scalar reconstruction losses**, not the reconstructed images, probabilities, or logits.
2. Calculate KL **once**, over the full 10-coordinate posterior; do not multiply KL by ten or compute a different KL for each prefix.
3. Give every prefix equal weight $1/10$; no sampled-prefix approximation, prefix dropout, or learned weighting in this comparison.
4. Sample the full $z$ once per image/update and reuse that same sample across prefixes. Mask the sampled latent after reparameterization; do not resample noise separately for each prefix.
5. Include $k=10$, where the mask is the identity. This is one of ten equally weighted reconstruction terms, not an extra full-vector loss outside the average.

### Gradients and interpretation

An omitted coordinate receives no reconstruction gradient from that prefix, but still receives full-posterior KL gradients. Coordinate $i$ participates in reconstruction terms for $k\geq i$; early coordinates participate in more terms. That asymmetry supplies ordering pressure without factor labels.

Although averaging keeps the aggregate reconstruction coefficient on the same scale as ordinary VAE, it changes the objective and distribution of gradients. Full-latent reconstruction quality may deteriorate because the same model must support small prefixes. Neither useful prefixes nor a high disentanglement metric proves a unique or seed-reproducible basis.

### Memory-conscious implementation requirements

Encoding and sampling occur once per batch. Decode all ten masks and compute all ten losses with mathematically equivalent sequential/chunked accumulation if needed. Avoid retaining an unnecessarily large tensor of all prefixes' 4,096-pixel outputs.

If backpropagation is split across prefixes to reduce memory, accumulate each reconstruction gradient with weight $1/10$, include the full KL gradient once, and call the optimizer **once per original batch**. Preserve the encoder graph where required. No parameter update is allowed between prefixes. Do not detach the encoder/sample or overwrite the shared latent in place in a way that alters later masks.

Equal parameter count does not imply equal computation: MRL requires ten decoder evaluations per update. Record wall time, throughput, and GPU memory separately; do not silently reduce MRL epochs or prefix coverage to match runtime.

## 7. Locked training settings and matched randomness

| Setting | All four conditions |
|---|---|
| Latent / hidden dimensions | 10 / 1200 |
| Model seeds | 0, 1, 2 |
| Optimizer | Adagrad, initial LR 0.01 |
| Optimizer details | `lr_decay=0`, `eps=1e-10`, `weight_decay=0`, `initial_accumulator_value=0` |
| Batch size | 256 |
| Training budget | 100 epochs |
| Updates | 2,880 per epoch; 288,000 optimizer updates/model |
| Precision | Float32; TF32 disabled; no mixed precision initially |
| Decoder activation | Tanh |
| Initialization | Default PyTorch |
| Extra mechanisms | No capacity schedule, KL warm-up, learning-rate scheduler, dropout, batch normalization, or gradient clipping |
| Checkpoints | Every 10 epochs and final |
| Model selection | Final epoch-100 checkpoint; no metric-test-driven checkpoint selection |
| Environment | Existing `mrl_env_cuda12`; record actual package/device/driver versions |

For a given model seed, use the same initial weights and data-order seed in all four conditions. Seed Python, NumPy, Torch, CUDA, and the data-loader generator. MRL must not instantiate additional trainable modules or consume extra random latent draws merely to create masks. Match latent sampling randomness where feasible; do not claim identical trained solutions after gradients diverge.

Record epoch averages of total loss, aggregate reconstruction, full-latent reconstruction, each prefix reconstruction for MRL, total and coordinate KL, β-weighted KL, image count, optimizer-update count, runtime, and peak GPU memory. Clearly label $R_{10}$ versus the average $\frac1{10}\sum_kR_k$; they are different quantities for MRL.

Checkpoints include model/optimizer/config, epoch/update count, and all RNG/data-loader states. Resume into a fresh folder and restore the exact training condition, including MRL flag, prefix schedule, equal weights, masking/sampling convention, β, optimizer, architecture, dataset checksum, and batch size. Reject incompatible resume settings. Preserve interrupted runs and recovery provenance.

## 8. Existing metric and other evaluations: unchanged

### Held-fixed-factor metric

Freeze the final encoder, use evaluation mode, and cache full 10-dimensional **posterior means** for all archive images. Preserve the current pair sampling and classifier calculation:

$$
d=\frac1{64}\sum_{n=1}^{64}
\left|\mu(x_n^{(a)})-\mu(x_n^{(b)})\right|.
$$

Targets are X-position, Y-position, scale, and rotation; shape is a nuisance. Within each pair, the target factor's value matches. The shared value is resampled between pairs, and other factors vary independently. Balanced targets give chance accuracy 25%.

| Metric setting | Locked value |
|---|---|
| Pair count per example | 64 |
| Examples | 10,000 training / 2,000 validation / 5,000 test |
| Sampling seeds | 1729 / 1730 / 1731 respectively |
| Features | Raw averaged absolute mean differences; no standardization |
| Classifier | Linear 10→4 with bias; softmax cross-entropy |
| Optimizer | Adagrad, LR 0.01 |
| Classifier batch size | 256 |
| Classifier seeds | 0, 1, 2 per trained encoder |
| Epoch cap | 2,000 |
| Stopping/selection | Best validation CE; patience 20; minimum improvement $10^{-4}$ |
| Final scoring | Restore selected classifier weights, then evaluate test once |

Use the same metric examples across conditions/model seeds where possible; sampling seeds are independent of model/classifier seeds. Keep examples fixed across the three classifier initializations. Save features, labels, classifier histories/weights, test accuracy, per-factor accuracy, confusion matrix, and whether the epoch cap was reached.

Keep the metric on the full 10-dimensional representation for every condition. No prefix classifiers are added to this agreed experiment; the total remains 36 fits. Historical classifiers reached their caps, so disclose that status if it recurs rather than asserting convergence.

### Reconstructions and empirical traversals

Preserve the existing deterministic selection of four references using seed 2026. Decode the posterior mean for reconstruction grids. Show original images above reconstructed Bernoulli probabilities.

For each of the four references, traverse each coordinate using the existing seven values over that model's empirical 5th–95th percentile range, while keeping other coordinates fixed at the reference mean. Preserve this procedure and identify its range as distinct from Figure 6's prior-range traversal.

Keep full-latent reconstruction diagnostics for all four conditions, including MRL. Additional prefix reconstruction grids/held-out prefix curves could be a later analysis, but are **not added to the fixed evaluation protocol by this plan**. Training prefix-loss logs document optimization behavior without pretending to measure held-out prefix utility.

### Figure 6 A/B for every model seed

For each seed 0, 1, 2, generate two comparisons:

| Comparison | Panel A | Panel B |
|---|---|---|
| Ordinary pair | Ordinary β-VAE, β=4 | Ordinary VAE, β=1 |
| Matryoshka pair | Matryoshka β-VAE, β=4 | Matryoshka VAE, β=1 |

This gives **six combined A/B figures**, plus standalone A/B panels in PNG/PDF. Each of the twelve trained models receives the same factor-response and traversal analysis, with the condition-specific column-order rule below.

Preserve the current procedures:

- Position heatmap: average posterior mean at each X/Y position over shape, scale, and rotation.
- Scale/rotation curves: separate square/ellipse/heart curves, averaging the other factors; existing colors.
- Shared reference image index 19477 under the seed-2026 convention.
- Traverse one coordinate at $+3,+1.5,0,-1.5,-3$, keeping the other nine at the reference posterior mean; decode probabilities.
- Ordinary VAE/β-VAE figures: retain ascending dataset-mean posterior-variance sorting $\mathbb E_x[\exp\ell_i(x)]$, with original coordinate IDs retained.
- **MRL figures: original coordinate order $z_1,\ldots,z_{10}$, left to right, for every model seed. No variance sorting or coordinate matching/permutation.**
- All ten coordinates shown; shared response scales within each A/B pair.
- Label condition, β, model seed, and epoch explicitly; save source checkpoints and numerical/provenance arrays.

For MRL Figure 6, column $i$ always denotes learned coordinate $z_i$ across both β conditions and all three model seeds. This exposes the learned ranking and lets us compare the same coordinate index across seeds without hiding order changes through sorting or matching. Retain posterior variance in the bottom row as a diagnostic, but do not use it to choose the column order. Record `coordinate_order=native` in MRL figure provenance and label the figure accordingly. Native-order inspection is evidence about ranking/seed stability; it does not by itself prove reproducibility.

**Do not reorder MRL coordinates during training or metric evaluation either.** Preserve original coordinate IDs and native-order numerical arrays. Ordinary-model variance sorting remains a display convention and must be labeled distinctly from MRL's learned order.

## 9. Reporting without score discarding

Per condition, report the complete 3×3 score table with model seed as rows and classifier seed as columns. Report:

1. All nine accuracies and their unfiltered mean.
2. Mean classifier accuracy for each of the three trained model seeds.
3. Variability across those three model-seed means, separately from within-model classifier variability.
4. Per-factor accuracies/confusion matrices, classifier convergence/cap status, and reconstruction/KL diagnostics.
5. Qualitative Figure 6 observations across all seeds, including failures and mixed coordinates.

Use explicitly labeled standard deviations where reported. Do not treat the nine nested fits as nine independent model samples or invent independent-sample confidence intervals. For the main comparisons, calculate paired model-seed differences after averaging classifier scores per encoder: `mrl_vae − vae` and `mrl_beta_vae − beta_vae`.

No best-half score, no dropped seeds, and no retrospective clean-seed selection. Failed runs remain in the record with their cause and recovery/retry provenance. Completion requires three successful model seeds in every condition; a failed seed must not disappear from the analysis silently.

Hypotheses, not guaranteed outcomes: MRL may prioritize information useful for reconstruction in earlier coordinates, improve factor selectivity, worsen reconstruction fidelity, or leave coordinates mixed/unstable. Report each separately. This small three-seed comparison does not prove nonlinear basis identifiability or match the paper's ten-model-replica study.

## 10. Planned implementation boundaries and validation

Use and extend the existing [reproduction code](../NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py), [model/loss](../NonLinearPBExperiments/vae_reproduction/model.py), and [Figure 6 script](../NonLinearPBExperiments/scripts/plot_dsprites_figure6.py). Preserve [data loading](../NonLinearPBExperiments/vae_reproduction/data.py) and [metric calculations](../NonLinearPBExperiments/vae_reproduction/metric.py).

The future implementation needs:

- An explicit reconstruction-MRL switch, disabled by default, with all ten prefixes/equal weights logged. Its final CLI name is not implemented by this plan.
- Backward-compatible ordinary fixed-β loss and old-checkpoint loading.
- Condition-aware checkpoint/run labels, summaries, and Figure 6 titles; distinguish MRL from ordinary models even when β matches. Figure generation must select native coordinate order for MRL and variance order for ordinary models, recording the rule in provenance.
- A dedicated four-condition/three-seed workflow. The existing paired `repo`/`paper` launcher does **not** implement this matrix and must not be presented as its launch command.
- Condition-aware resume validation and prefix-loss logging, with one optimizer update per original batch.

Prepare and read validation tests for user execution under [shared project norms](../../MainREADME.md). Coverage should include:

| Check | Required evidence |
|---|---|
| Prefix masks | Exact zeros after $k$; identity mask for $k=10$; all ten prefixes present |
| Shared sampling/decoder | One encoder pass/sample per batch; same decoder parameters for all prefixes |
| Loss arithmetic | Mean of ten scalar BCE losses plus one β-weighted full KL |
| Averaging distinction | A constructed case distinguishes mean-of-losses from loss-of-mean-images/logits |
| Ordinary compatibility | MRL disabled preserves baseline architecture, initialization, and objective |
| Gradients | Omitted coordinates have no reconstruction gradient from that prefix; retained coordinates and shared decoder receive gradients; KL acts on full posterior |
| Optimizer cadence | Prefix gradient accumulation makes exactly one optimizer update per original batch |
| Resume | Resumed ordinary/MRL training agrees with uninterrupted small deterministic runs; changed prefix/condition configs are rejected |
| Evaluation | Existing metric code/settings and factor/traversal calculations remain identical; MRL Figure 6 preserves indices 1–10 across seeds regardless of variances; ordinary figures retain variance sorting; titles/source loading/provenance record the correct condition/order |
| Matrix/reporting | All twelve runs and all 36 fits are accounted for; all nine scores retained per condition; ordinary/MRL folders and summaries cannot collide |

Do not run the unit-test files automatically under the current project convention. Validate implementation before launching full jobs using the user-run checks and explicitly authorized smoke/benchmark steps as appropriate. Such execution is a later step, not part of this plan creation.

## 11. Detached tmux execution and compute planning

Create a fresh detached tmux session for this experiment and keep training, evaluations, aggregation, and Figure 6 generation inside it. Use persistent per-run logs and retained exit status. Closing SSH/terminal must not terminate the workflow. A reboot/process failure still requires checkpoint recovery.

Use one independent model-training process per RTX 3090 GPU, with at most two model runs active at once. Queue the twelve trainings in a documented deterministic order. For example, process β=1 ordinary/MRL pairs for seeds 0–2, then β=4 ordinary/MRL pairs for seeds 0–2, one pair at a time. Preserve completed runs and avoid oversubscribing either GPU during evaluation/figure generation.

Benchmark MRL decoder throughput and peak memory before forecasting duration. It uses ten decoder passes rather than one, so the previous approximately 13-second ordinary epoch measurement cannot be copied as its runtime estimate. Both conditions still receive the same 100 epochs and update count; this is an equal-data/update-budget comparison, not equal-compute training.

The workflow must detect child failures, record an explicit failed stage, and retain all logs/checkpoints. Do not declare success because tmux exists or a checkpoint was produced. Generate a pair's Figure 6 only after both corresponding final models and metric evaluations have completed successfully.

## 12. Output organization and provenance

Create a fresh timestamped comparison root outside the code repository:

```text
/home/argha/Mat_embedding/files/results/exprmnt_{timestamp}/
    config.json
    experiment_description.log
    manifest.json
    status.json
    results_summary.txt
    runtime.txt
    code_snapshot/
    launch_logs/
    conditions/
        vae/seed_0/exprmnt_{timestamp}/...
        vae/seed_1/exprmnt_{timestamp}/...
        vae/seed_2/exprmnt_{timestamp}/...
        beta_vae/seed_{0,1,2}/exprmnt_{timestamp}/...
        mrl_vae/seed_{0,1,2}/exprmnt_{timestamp}/...
        mrl_beta_vae/seed_{0,1,2}/exprmnt_{timestamp}/...
    figure6/
        ordinary/seed_{0,1,2}/...
        matryoshka/seed_{0,1,2}/...
    aggregate/
        all_36_scores.json
        per_condition_model_classifier_tables.csv
        per_model_means.csv
        paired_model_seed_differences.csv
```

The tree is a proposed organization, not a claim that these files or folders already exist. Each leaf run follows the existing timestamped convention with exact config, environment/checksum, training curves, epoch/final checkpoints, runtime, source snapshot, cached means, metric features/labels, classifier histories/weights/confusion matrices, reconstructions, and four empirical traversals. Figure filenames include timestamps and preserve source model/seed/condition provenance.

Root-level manifest maps every condition/model seed to its run directory, final checkpoint, nine-score condition table, and Figure 6 outputs. Record β, MRL flag, prefixes, equal weights, sampling/masking convention, optimizer/initialization/activation, all seeds, training/classifier budgets, dataset identity, and source revision/snapshot.

Use condition keys in summaries rather than grouping only by β; ordinary and MRL share β values and must never be merged. Old runs, snapshots, and figures remain untouched. Do not place these new figures into older trial folders; the new comparison has its own complete provenance.

## 13. Completion criteria

The experiment is complete only when:

- All twelve models finish 100 epochs with recorded final checkpoints and diagnostics.
- All 36 classifier fits finish with validation-selected weights and retained results, including cap-status flags.
- Six combined Figure 6 A/B comparisons and their standalone panels are saved with numerical/provenance support; MRL columns are in original $z_1,\ldots,z_{10}$ order for all seeds.
- All nine scores per condition, model-seed means, and paired ordinary/MRL comparisons are reported without filtering.
- Training curves, reconstruction grids, all reference traversals, factor responses, and classifier results are reviewed across every seed; contradictory evidence is reported.
- The tmux workflow records successful completion and artifact checks pass.
- [NonLinearPBREADME.md](../NonLinearPBREADME.md) and this plan receive the exact launch/run locations and measured results.

A negative or mixed scientific result still satisfies execution completion. Claims about disentanglement, ordering, and stability must follow the evidence rather than the desired outcome.

## 14. Implementation and authorized two-epoch smoke validation

Implemented on 2026-10-13:

- [model.py](../NonLinearPBExperiments/vae_reproduction/model.py): one encoder/sample per batch, immutable zero-prefix masks, mean scalar prefix BCE, and one analytic full KL. Sequential prefix backward accumulates gradients without optimizer updates between prefixes; ordinary forward remains the default.
- [dsprites_vae_reproduction.py](../NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py): optional `--mrl-reconstruction`, explicit classifier seeds, `--no-score-filtering`, prefix/full/aggregate reconstruction logs, and MRL-aware resume validation. MRL is rejected with capacity control.
- [plot_dsprites_figure6.py](../NonLinearPBExperiments/scripts/plot_dsprites_figure6.py): automatic native coordinate order for MRL, variance order for ordinary models, explicit order provenance, and smoke labels for subset-trained checkpoints. An explicit request to variance-sort MRL is rejected.
- [run_dsprites_mrl_matrix.py](../NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix.py): queues the four conditions over model seeds 0–2, three classifier seeds per encoder, one process/GPU, no untrained-control extras; generates all six Figure 6 pairs, retains all 36 scores, and writes nested tables/paired differences.
- [run_dsprites_mrl_matrix_tmux.sh](../NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix_tmux.sh): detached launcher with persistent logs and retained exit status.
- [helper_mrl.py](../NonLinearPBExperiments/tests/helper_mrl.py): mask/sample reuse, scalar-loss arithmetic, sequential versus single-objective gradients, one optimizer update/batch, resume equivalence, native order, and all-score reporting checks. The existing subprocess test runner now includes this helper. Unit-test files were prepared and read, not run under the shared convention; the user explicitly authorized the smoke execution below.

Smoke command (from `code/`):

```bash
bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix_tmux.sh --smoke --epochs 2
```

`--smoke` restricts model training to **exactly two epochs**, retains the full four-condition × three-model-seed × three-classifier-seed matrix, and uses the existing fast settings: 8,192 training images, 16 pairs per metric example, 1,000/400/1,000 metric train/validation/test examples, classifier cap 40 with fast patience 10. Explicit classifier seeds 0–2 override the fast entry point's one-classifier default. Full model widths and all ten MRL prefixes are retained. Each run still encodes the full archive for factorial lookup and Figure 6 response calculations. These are execution diagnostics, not final scientific scores.

The full workflow defaults remain 100 epochs, full image archive, 64 pairs, 10,000/2,000/5,000 metric examples, classifier cap 2,000, and patience 20. It is not launched by this smoke command.

Initial smoke launch:

```text
tmux session: dsprites-mrl-2026_10_13__20_59_08
results root: /home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_13__20_59_08/
workflow log: /home/argha/Mat_embedding/files/results/launch_logs/dsprites-mrl-2026_10_13__20_59_08/workflow.log
```

**Smoke completed successfully in 205.924 seconds (3 minutes 25.924 seconds).** All twelve models ran exactly two epochs on 8192 images (64 optimizer updates/model), all 36 classifier fits completed, and all six A/B comparisons plus standalone panels were saved (36 PNG/PDF figure files). Every condition retained nine scores without filtering.

Artifact validation checked original Adagrad/Tanh/default options, fixed β, epoch/data/update counts, finite losses, ten-prefix scalar averaging, full-prefix logs, three classifier files and four empirical traversal grids per model, no best-half fields, and native MRL coordinate indices in all Figure 6 provenance files. Ordinary figures retain variance ordering. A representative MRL A/B plot was visually inspected and has columns $z_1,\ldots,z_{10}$ in both panels.

Static comparisons against prior source snapshots confirmed unchanged metric pair sampling, metric-example calculation, classifier fitting, frozen encoding, reconstruction diagnostics, and Figure 6 factor-response/plot-panel calculations. Unit-test files were reviewed but not executed under the shared convention; these smoke results do not claim that the separate unit suite has passed.

Smoke artifacts: [results root](/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_13__20_59_08/), [manifest](/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_13__20_59_08/manifest.json), [aggregate summary](/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_13__20_59_08/aggregate/summary.json), and [MRL seed-0 Figure 6](/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_13__20_59_08/figure6/matryoshka/seed_0/figure6_AB_2026_10_13__21_00_00.png).

No full 100-epoch experiment was launched during that smoke-validation step. The reduced-data/short-classifier smoke scores are pipeline diagnostics, not evidence for or against the scientific hypotheses.

## 15. Full experiment launch — 2026-10-13

The user explicitly authorized full training after the smoke test. Both RTX 3090 GPUs were verified idle before launch. Started the detached matrix workflow with:

```bash
bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix_tmux.sh --epochs 100 --seeds 0 1 2 --classifier-seeds 0 1 2 --classifier-epochs 2000
```

```text
tmux session: dsprites-mrl-2026_10_13__21_22_10
results root: /home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__21_22_10/
workflow log: /home/argha/Mat_embedding/files/results/launch_logs/dsprites-mrl-2026_10_13__21_22_10/workflow.log
```

Verified root configuration: `smoke=false`, 100 epochs, model seeds 0–2, classifier seeds 0–2, classifier cap 2000, Adagrad 0.01, default initialization, Tanh decoder, prefixes 1–10, and no score filtering. The first active pair is ordinary VAE beta1/seed0 on GPU0 and ordinary beta4/seed0 on GPU1. For each seed, ordinary and then Matryoshka pairs run sequentially, with at most one model process per GPU. Metrics and Figure 6 generation follow each pair automatically.

At launch, training was running and results were pending. Completion was verified on 2026-10-14 from the root status, all twelve child statuses/epoch histories, the manifest, and aggregate summary. All models finished epoch 100, all 36 classifier fits completed, and six Figure 6 comparisons were generated. Total workflow runtime: **25867.038 seconds (7 hours 11 minutes 7 seconds)**. Detailed cross-seed visual review remains pending; completion of computation does not establish disentanglement or stable ranking.

| Condition | Seed-0 classifier mean | Seed-1 classifier mean | Seed-2 classifier mean | All nine scores, unfiltered mean |
|---|---:|---:|---:|---:|
| VAE, β=1 | 76.93% | 77.69% | 86.83% | 80.48% |
| β-VAE, β=4 | 92.06% | 60.68% | 78.91% | 77.22% |
| MRL VAE, β=1 | 82.29% | 82.50% | 86.76% | 83.85% |
| MRL β-VAE, β=4 | 84.07% | 83.42% | 67.21% | 78.23% |

Every seed-level mean averages its three classifier initializations. All nine scores per condition were retained, with no bottom-half filtering. Sources: [aggregate summary](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__21_22_10/aggregate/summary.json) and [raw scores](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__21_22_10/aggregate/all_36_scores.json). These scalar summaries should be interpreted alongside the native-order MRL Figure 6 panels and reconstruction/traversal evidence.

Attach with `tmux attach -t dsprites-mrl-2026_10_13__21_22_10`; detach with Ctrl-b, then d. The jobs continue after terminal/SSH disconnection. Checkpoints are saved every ten epochs and final. All prior smoke/pilot outputs remain preserved.
