# VAE / β-VAE reproduction on dSprites

**Updated:** 2026-10-13.  
**Status:** PyTorch baseline and optional training variants implemented; two-epoch trial and seed-0, 100-epoch baseline pilots completed. Paper-level reproduction remains incomplete. The variant comparisons below have not been launched.  
**Implementation rule:** All models, objectives, training loops, and evaluation additions use **PyTorch** under `NonLinearPBExperiments/`. External TensorFlow code is a reference for inspection, not our execution framework.

## 1. Scope and source of this plan

This document consolidates the baseline reproduction descriptions from [dSprites task-specific MRL experiment plan, First task and R0–R10](dsprites_task_specific_mrl_experiment_plan_AEOLUS.md). That file continues to describe the later task-supervision/MRL research. This standalone plan focuses on reproducing ordinary VAE and β-VAE results **before** adding supervision or MRL.

The immediate aims are to:

1. Assess the original paper's VAE/β-VAE metric comparison.
2. Reproduce the **analysis procedure** of Figure 6 A/B and assess whether learned coordinates separate position, scale, and rotation.
3. Explain discrepancies through implementation audits and controlled comparisons.
4. Explore a later capacity-controlled objective as a separately labeled extension.

Reference paper: [local Higgins et al. β-VAE PDF](/home/argha/Mat_embedding/files/papers/beta-vae,-higgins,-iclr2017.pdf). Our supplied under-review version uses **Figure 5** for metric results and **Figure 6 A/B** for latent analysis; numbering differs between paper versions.

| Condition | Paper's reported metric score in the supplied version |
|---|---:|
| VAE, β=1 | 61.58 ± 0.5% |
| β-VAE, β=4 | 99.23 ± 0.1% |
| Ground-truth factors | 100% |

These are comparison targets, not guaranteed results. The paper's uncertainty convention needs confirmation. Reproducing a plotting layout does not establish that its learned representations have been reproduced.

## 2. Current data and architecture

### Data

Archive:

```text
/home/argha/Mat_embedding/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

- 737,280 binary grayscale images, each 64×64, flattened to 4,096 values.
- Factors: shape (3), scale (6), rotation (40), X-position (32), Y-position (32); color is constant.
- SHA-256: `857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2`.
- Full archive used for unsupervised training, shuffled each epoch. No image-level reconstruction train/validation/test split. This is our documented reproduction choice.
- No augmentation or additional normalization. Compact binary CPU arrays are converted to float32 per batch.
- Factor annotations validate indexing and support evaluation; they never enter the VAE training loss.
- Independently sampled metric examples may use images from the training archive. Metric results measure organization on that archive, not generalization to unseen images.

### Network

```text
Encoder:
  4096 → Linear 1200 → ReLU → Linear 1200 → ReLU
                                             ├→ Linear 10 → mean μ
                                             └→ Linear 10 → log variance ℓ

Sampling:
  z = μ + exp(ℓ / 2) ⊙ ε,  ε ~ N(0, I),  z has 10 coordinates

Decoder:
  10 → Linear 1200 → Tanh → Linear 1200 → Tanh
     → Linear 1200 → Tanh → Linear 4096 → pixel logits
```

All linear layers include biases. Posterior heads and decoder logits have no final activation. The posterior is a diagonal Gaussian:

$$
q_\phi(z\mid x)=\mathcal N\left(\mu_\phi(x),\operatorname{diag}(\exp\ell_\phi(x))\right),
\qquad p(z)=\mathcal N(0,I).
$$

The decoder defines conditionally independent Bernoulli pixels with probabilities $p_j(z)=\operatorname{sigmoid}(a_j(z))$. Plots display those probabilities as grayscale, not sampled binary pixels. A diagonal conditional posterior does not establish independent or disentangled dataset-level latent means.

## 3. Current loss, optimization, and checkpoints

For batch size $B$, the implemented loss is:

$$
\mathcal L=\frac1B\sum_{b=1}^B\left[
\sum_{j=1}^{4096}\operatorname{BCEWithLogits}(a_{bj},x_{bj})
+\frac\beta2\sum_{i=1}^{10}
\left(\mu_{bi}^2+\exp\ell_{bi}-1-\ell_{bi}\right)
\right].
$$

The reconstruction expectation is estimated using **one reparameterized latent sample per image per update**. KL is analytic. Sum over pixels and latent coordinates, then average over images; do not independently divide BCE by 4,096.

| Setting | Completed 100-epoch pilots |
|---|---|
| β | Fixed 1 or 4 |
| Optimizer | Adagrad, initial learning rate 0.01 |
| Adagrad settings | `lr_decay=0`, `eps=1e-10`, `weight_decay=0`, `initial_accumulator_value=0` |
| Batch size | 256 |
| Training budget | 100 epochs; 2,880 updates/epoch; 288,000 total updates/model |
| Initialization | Default PyTorch `nn.Linear`; model seed 0 for each condition |
| Precision | Float32; no mixed precision; TF32 disabled |
| Hardware | RTX 3090; VAE on GPU 0, β-VAE on GPU 1 |
| Additional mechanisms | No dropout, batch normalization, gradient clipping, KL warm-up, β schedule, or capacity schedule |
| Model selection | Final fixed-budget checkpoint; no VAE early stopping or metric-test selection |
| Checkpoints | Every 10 epochs and final; weights, optimizer, epoch/config, Python/NumPy/Torch/CUDA/data-loader RNG states |
| Environment | `mrl_env_cuda12`: Python 3.10.20, PyTorch 2.5.1+cu124, NumPy 2.2.6 |

Each update encodes the batch, samples latents, decodes, computes loss, backpropagates, and updates encoder/heads/decoder jointly. Fresh sampling noise is used for each forward pass. Adagrad changes effective parameter step sizes through its accumulators even with zero explicit learning-rate decay.

## 4. Current disentanglement metric

The encoder is frozen. Use its means $\mu(x)$, cached for the full archive.

For one balanced four-class metric example, choose X-position, Y-position, scale, or rotation, then draw 64 image pairs. Within each pair the selected factor matches; its shared value is resampled between pairs. Other factors, including shape, vary independently and may coincide by chance. Form:

$$
d=\frac1{64}\sum_{n=1}^{64}\left|\mu(x_n^{(a)})-\mu(x_n^{(b)})\right|.
$$

The label is the **identity of the fixed factor**, not its value. Feed the 10-dimensional raw difference vector to a linear classifier with bias, `Linear(10, 4)`, trained with softmax cross-entropy and Adagrad at 0.01. No hidden layers or feature standardization are used.

| Setting | Current choice |
|---|---|
| Metric examples | 10,000 train / 2,000 validation / 5,000 test |
| Sampling seeds | Train 1729; validation 1730; test 1731 |
| Classifier seeds | 0, 1, 2; fixed examples across classifier initializations and compared models |
| Batch size | 256 |
| Epoch cap | 200 for initial trial/control; 2,000 for 100-epoch pilots |
| Selection/stopping | Best validation CE; patience 20; minimum improvement $10^{-4}$ |
| Test scoring | Once after restoring the selected classifier weights |
| Chance | 25% |

High classification accuracy can coexist with mixed latent coordinates: the classifier combines information across all ten dimensions. Keep response maps, traversals, and metric controls alongside scores.

The source plan records the paper's aggregation as 10 model replicas × 3 classifier seeds, discarding the bottom 15 of 30 scores. Our completed pilot has **one model seed per condition**, not ten. Report raw/unfiltered scores and across-model variability separately from any paper-style best-half summary; classifier repetitions are nested within a trained encoder.

## 5. Code map: what each file does

Paths below are relative to `NonLinearPB/NonLinearPBExperiments/`.

| Script/module | Responsibility |
|---|---|
| [experiments/dsprites_vae_reproduction.py](../NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py) | Complete CONFIG and CLI; seeding, environment/checksum capture, training loop, resume/checkpoints, frozen encoding, metric fitting, reconstruction/traversal diagnostics, score summaries, runtime and snapshots |
| [vae_reproduction/model.py](../NonLinearPBExperiments/vae_reproduction/model.py) | `VAE`, encoder/heads/decoder, reparameterization, Bernoulli reconstruction and analytic KL loss |
| [vae_reproduction/data.py](../NonLinearPBExperiments/vae_reproduction/data.py) | Compact archive loading/validation, SHA-256, exact factor-to-image indexing, optional smoke-training subset |
| [vae_reproduction/metric.py](../NonLinearPBExperiments/vae_reproduction/metric.py) | Held-fixed-factor pair sampling, aggregated features, linear classifier training/validation/test, confusion matrices |
| [scripts/run_dsprites_seed0_tmux.sh](../NonLinearPBExperiments/scripts/run_dsprites_seed0_tmux.sh) | Detached paired 100-epoch GPU pilot, persistent console logs, safe session checks, VAE resume launcher |
| [scripts/plot_dsprites_figure6.py](../NonLinearPBExperiments/scripts/plot_dsprites_figure6.py) | Post-hoc Figure 6 A/B response maps, shape-specific curves, prior-range traversals, posterior-variance ordering; no training |
| [tests/run_tests.py](../NonLinearPBExperiments/tests/run_tests.py) | Subprocess runner for reproduction helper tests; runner itself does not import Torch |
| [tests/helper_reproduction.py](../NonLinearPBExperiments/tests/helper_reproduction.py) | Index/pair constraints, KL/loss units, gradients/checkpoints, ground-truth/shuffled metric controls, exact resume agreement |
| [tests/helper_figure6.py](../NonLinearPBExperiments/tests/helper_figure6.py) | Factor marginalization and five-value traversal convention; separate helper, not included automatically in `tests/run_tests.py` |

The entry point supports `--fast`, `--trial`, `--benchmark`, `--untrained-only`, `--skip-untrained`, `--classifier-epochs`, and `--resume-checkpoint`, alongside beta/seed/epoch/device/data/output overrides. `--fast` reduces training/metric settings but evaluation still encodes the full archive for exact pair lookup. `--trial` changes the output destination without reducing data/metric settings by itself.

Shared [utility.py](../../utility.py) supplies runtime reporting. The reproduction entry point resolves the current project root directly instead of depending on the shared helper's older `Mat_embedding_hyperbole` root-name assumption.

## 6. Experiment record and exact result locations

All folders below are under `/home/argha/Mat_embedding/files/results/`.

| Folder | Purpose/status | Models and evaluation |
|---|---|---|
| `test_runs/exprmnt_2026_10_12__21_51_46` | Completed two-epoch exploratory trial | Seed 0, β=1 and β=4, full archive; 200-epoch classifier cap; runtime 72 seconds |
| `test_runs/exprmnt_2026_10_12__22_17_14` | Completed random-encoder control | Seed 0, no VAE updates; only metric classifiers trained, cap 200; runtime 20 seconds |
| `exprmnt_2026_10_12__22_22_49` | Original VAE pilot, interrupted | VAE reached epoch 17 before interruption; epoch-10 checkpoint supplied recovery. Preserve as historical artifacts, not the completed final run |
| `exprmnt_2026_10_12__22_22_53` | Completed β-VAE pilot | `beta_4_seed_0/checkpoint_final.pt`, 100 epochs, classifier cap 2,000 |
| `exprmnt_2026_10_12__22_29_14` | Completed resumed VAE pilot | `beta_1_seed_0/checkpoint_final.pt`, restored epoch-10 optimizer/RNG states, trained epochs 11–100; 100 total epochs, classifier cap 2,000 |

### Metric scores

These are held-fixed-factor test accuracies. Means are across three classifier initializations of **one encoder**, not three model replicas. Values below are rounded from saved results.

| Condition | Classifier seed 0 | Seed 1 | Seed 2 | Unfiltered mean |
|---|---:|---:|---:|---:|
| Untrained control, original cap 200 | 43.26% | 43.74% | 29.24% | 38.75% |
| VAE, 2 epochs, cap 200 | 96.96% | 96.94% | 96.48% | 96.79% |
| β-VAE, 2 epochs, cap 200 | 95.56% | 95.98% | 95.70% | 95.75% |
| VAE, 100 epochs, cap 2,000 | 77.14% | 76.86% | 76.78% | 76.93% |
| β-VAE, 100 epochs, cap 2,000 | 92.16% | 92.14% | 91.88% | 92.06% |

Sources: each run's `results_summary.txt`, `metric_scores.json`, `metric_summary.json`, and per-classifier JSON files. The initial trial/control classifiers all reached their 200-epoch caps. The two-epoch and 100-epoch evaluations use different classifier caps, so their score changes do not isolate VAE training duration. Inspect the saved classifier histories before claiming convergence for later runs.

Epoch-100 **training averages** (not a separate final-checkpoint evaluation):

| Model | Reconstruction NLL | Unweighted KL | Weighted KL | Total loss |
|---|---:|---:|---:|---:|
| VAE, β=1 | 39.8893 | 17.4505 | 17.4505 | 57.3397 |
| β-VAE, β=4 | 53.8243 | 12.3747 | 49.4987 | 103.3231 |

### Outputs and recovery

Run-level files include `config.json`, `environment.json`, `status.json`, `experiment_description.log`, `results_summary.txt`, metric score/summary JSON, timestamped training plots, `runtime.txt`, and `code_snapshot/`. Model subfolders contain their config, epoch/final checkpoints, `training.jsonl`, `latent_means.npy`, metric feature archives, classifier weights/results/history, reconstructions, and reference traversals, as applicable. Untrained folders contain epoch-zero weights rather than a VAE training history.

The detached training session was `dsprites-seed0-100ep`, including the later `vae-resumed` window. Launch logs:

```text
files/results/launch_logs/dsprites_seed0_2026_10_12__22_22_46/
files/results/launch_logs/dsprites_seed0_2026_10_12__22_29_11/
```

### Figure 6 A/B provenance and current location

The Figure 6 plots use the **100-epoch checkpoints** from `22_22_53` (β=4) and `22_29_14` (β=1). They do not use the two-epoch trial's weights.

At the user's request, the six PNG/PDF files were moved into:

```text
/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_12__21_51_46/
    figure6_AB_2026_10_13__13_24_54.{png,pdf}
    figure6_A_beta_vae_2026_10_13__13_24_54.{png,pdf}
    figure6_B_vae_2026_10_13__13_24_54.{png,pdf}
```

**Storage location does not determine model provenance:** this folder's original summary/config still describe the two-epoch trial. Figure labels show 100 epochs. [Combined Figure 6 PNG](/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_12__21_51_46/figure6_AB_2026_10_13__13_24_54.png).

Generation completed in 15.6 seconds via tmux session `dsprites-figure6`. Remaining numerical arrays, provenance JSON, reference image, and plotting snapshot are currently in `/tmp/nonlinearpb_figure6_2026_10_13/`; console log is `/tmp/nonlinearpb_figure6_2026_10_13.log`. These temporary support files need permanent archival with subsequent analysis outputs; only the six figures were moved.

Plot conventions:

- Position: full-factorial mean latent response at each X/Y location, averaging shape/scale/rotation.
- Scale/rotation: mean response per shape, averaging the other factors; red square, green ellipse, blue heart.
- Reference index 19477; deterministic seed-2026 image selection.
- Traversal values $+3,+1.5,0,-1.5,-3$; other coordinates fixed to that reference's posterior mean.
- Columns ordered by ascending $\mathbb E_x[\exp\ell_i(x)]$, our explicit posterior-variance aggregation choice; original coordinate labels retained.
- Shared response limits across A/B; all ten coordinates shown rather than grouping unused latents.

Current visual conclusion: β-VAE position maps show diagonal/curved responses and traversals mix properties. The clean independent horizontal/vertical position coordinates shown in the paper have **not** been reproduced. Higher metric accuracy alone does not overturn that observation. No post-hoc rotation or selection of attractive coordinates should be presented as native learned disentanglement.

## 7. External repository: Akella17/Beta-VAE

Reference: [Akella17/Beta-VAE](https://github.com/Akella17/Beta-VAE), inspected 2026-10-13. Its README discusses both the original β-VAE and the later capacity-control approach, and presents reconstruction/traversal examples. Its reported example settings, $\gamma=100,C=20$, differ from notebook defaults. Treat those illustrations as qualitative references rather than a verified configuration-matched result.

The [MLP notebook](https://github.com/Akella17/Beta-VAE/blob/master/beta_vae.ipynb) has similar hidden widths, ReLU decoder layers, Xavier initialization, Adam at 0.0005, and batch size 64. It intends to minimize reconstruction plus $\gamma|\mathrm{KL}-C(t)|$, with multiplier 1,000 and capacity rising from 0 to 25 over 100,000 updates. However, sampling treats `log_std_dev` as log variance while KL treats it as log standard deviation. Clipping it to positive values also prevents variances below approximately one under the sampling interpretation. Do not reproduce these inconsistencies in PyTorch.

The [convolutional notebook](https://github.com/Akella17/Beta-VAE/blob/master/conv_beta_vae.ipynb) proposes four convolutional encoder layers followed by fully connected layers. Its supplied decoder reshape and Python variable references have additional inconsistencies. Any convolutional comparison needs a correctly dimensioned PyTorch model and validation, not a direct translation.

The useful experimental ideas are capacity scheduling, optimizer choice, initialization, and activation comparisons. This repository does not establish which change causes better factor separation, and its current source is not an exact original-paper reproduction.

## 8. Proposed investigation, entirely in PyTorch

### Implemented switches — 2026-10-13

The existing entry point now supports these independent options. All previous evaluation calculations and classifier settings are retained.

| Flag | Default (existing baseline) | Optional variant |
|---|---|---|
| `--objective` | `beta` | `capacity` |
| `--optimizer` | `adagrad` | `adam` |
| `--lr` | 0.01 for Adagrad | 0.0005 for Adam unless explicitly overridden |
| `--initialization` | `default` (PyTorch) | `xavier` (Xavier-uniform weights, zero biases) |
| `--decoder-activation` | `tanh` | `relu` |
| `--gamma` | 1000; used only for capacity | Any finite positive multiplier |
| `--capacity-max` | 25 nats; used only for capacity | Finite nonnegative target |
| `--capacity-steps` | 100000; used only for capacity | Positive ramp length in optimizer updates |

All commands below run from `code/` in `mrl_env_cuda12`. They are examples, not jobs launched by this implementation change. Retaining `--classifier-epochs 2000` matches the completed 100-epoch pilots; the original script's classifier default remains 200.

```bash
# Original baseline: these are also the defaults when variant flags are omitted.
python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --betas 1 4 --seeds 0 --epochs 100 --classifier-epochs 2000 --objective beta --optimizer adagrad --lr 0.01 --initialization default --decoder-activation tanh

# Adam only, fixed beta4.
python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --betas 4 --seeds 0 --epochs 100 --classifier-epochs 2000 --optimizer adam

# Capacity objective only, existing optimizer and architecture.
python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --seeds 0 --epochs 100 --classifier-epochs 2000 --objective capacity

# Capacity plus Adam, keeping the existing initialization/Tanh architecture.
python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --seeds 0 --epochs 100 --classifier-epochs 2000 --objective capacity --optimizer adam

# Combined repository-inspired variant, still entirely PyTorch.
python NonLinearPB/NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py --seeds 0 --epochs 100 --classifier-epochs 2000 --objective capacity --optimizer adam --initialization xavier --decoder-activation relu --gamma 1000 --capacity-max 25 --capacity-steps 100000
```

Batch size stays 256 unless `--batch-size 64` is explicitly selected. Capacity mode trains one condition per model seed and rejects `--betas`, which has no role in its penalty. For compatibility with existing score plumbing, its saved `betas`/`beta` fields retain an unused value of 4; `beta_used_in_loss=false`, objective labels, and `capacity_seed_*` folders identify the actual condition. Capacity summaries omit the original fixed-beta paper reference.

The analytic KL and reparameterization use consistent, unrestricted log variance. No repository clipping or inconsistent standard-deviation formulas were copied. Capacity applies absolute value after batch-mean KL. Training histories now also log regularization penalty, capacity target (capacity mode), and cumulative update count.

Checkpoints record the update count and training options; resumed capacity training continues the same schedule rather than restarting it. Resume rejects changed optimizer, objective, initialization, decoder activation, or capacity settings. Older checkpoints without these fields use the original baseline defaults, with update count recovered from completed epochs and data-loader length.

Figure 6 loads decoder architecture options from checkpoint configuration. Its factor aggregation, variance ordering, traversal values/reference selection, and plotting scales are unchanged. Panel A accepts a capacity checkpoint through `--beta-checkpoint` and labels the objective/options accurately; panel B remains a fixed-beta1 VAE. The default source checkpoints remain the existing 100-epoch pilots.

Added test coverage for capacity reduction/gradient signs, ramp limits, option-aware model reloads, baseline defaults, Adam capacity resume, old-checkpoint resume, and capacity reporting. Tests were read but not run, following the shared user-run test convention. Python syntax was checked. No new scientific scores are available yet.

### Phase A — audit before changing the baseline

1. Check the supplied paper's objective, architecture, optimizer, and metric protocol against code and saved snapshots. Record known settings separately from unspecified choices: batch size, training budget, initialization, Adagrad epsilon/accumulator, and classifier stopping.
2. Confirm pixel-sum/batch-mean BCE and coordinate-sum/batch-mean KL; confirm `logvar` semantics consistently in sampling, KL, and variance plots.
3. Inspect loss/per-coordinate-KL and classifier-validation histories. Compare several already-saved epochs with a common classifier budget; do not select checkpoints by final metric-test accuracy.
4. Validate factor-map orientation and traversal indexing. Use multiple predefined reference images, because a single traversal depends on the reference.
5. Keep ground-truth, shuffled-label, random-encoder, and mixed-factor controls. The earlier synthetic rotated-factor score of 94.9% is evidence of metric limitations, not a trained-model result.

Deliverable: an explicit fidelity/deviation table and a stable evaluation specification. Audit changes that fix mistakes are distinguished from experimental hyperparameter changes.

### Phase B — controlled fixed-β comparisons

Retain our current implementation as the reference. Start with paired seeds 0, 1, 2 for β=1 and β=4. Evaluate both scalar scores and Figure 6 structure using a common metric-classifier budget. Change one setting at a time:

| Comparison | Proposed condition | Purpose |
|---|---|---|
| Seed replication | Current configuration, seeds 0–2 | Determine whether coordinate mixing persists across seeds |
| Adagrad details | Paper-compatible settings if independently established | Resolve optimizer implementation differences before tuning |
| Initialization | Xavier weights with explicitly defined biases | Test whether default initialization contributes |
| Optimizer | Adam, learning rate 0.0005 | Compare with repository-inspired optimization |
| Decoder activation | ReLU instead of Tanh | Repository-inspired architecture ablation; not the original selected decoder |
| Batch size | 64 instead of 256 | Test optimization effects; report both update count and images processed |
| Training duration | Inspect checkpoints; predefine any longer budget | Test convergence without metric-test-driven checkpoint selection |

Batch-size changes need separate matched-update and/or matched-image-budget comparisons: 100 epochs at batch 64 gives four times as many updates as batch 256. Do not silently attribute that difference to batch size alone. Keep all settings and budgets explicit.

If individual changes are informative, test a clearly labeled combined Adam/Xavier/ReLU condition. Avoid claiming that a combined improvement identifies a single cause. β sweeps, if later justified, use a predefined pilot grid and fresh final evaluation examples after selection.

### Phase C — capacity-controlled objective as a separate extension

Implement the later method correctly with the same log-variance parameterization:

$$
\mathcal L_C=\mathcal L_{\mathrm{reconstruction}}
+\gamma\left|\overline{\mathrm{KL}}-C(t)\right|,
\qquad
C(t)=C_{\max}\min\left(\frac{t}{T_C},1\right).
$$

Here $\overline{\mathrm{KL}}$ is the analytic KL summed over coordinates and averaged across the batch. Penalize the absolute difference **after** this batch reduction; do not silently replace it with the mean of per-image absolute differences.

Initial repository-inspired candidate: $\gamma=1000$, $C_{\max}=25$ nats, $T_C=100000$ updates. The README's $\gamma=100,C=20$ example is a separate candidate whose schedule/provenance is unresolved. These are proposed comparisons, not validated optimal settings.

- Keep posterior `logvar` unrestricted initially; do not copy the inconsistent positive clipping.
- Track target capacity, achieved KL, reconstruction, per-coordinate KL, and factor responses through training.
- First compare fixed-β and capacity-controlled losses with the same architecture/optimizer/budget. Then assess interaction with Adam if useful.
- Report this as a later-method extension, not as reproduction of original fixed-β Figure 6.

### Phase D — optional convolutional comparison

Only after baseline/audit comparisons, consider a corrected PyTorch convolutional architecture motivated by the repository and later paper. Explicitly specify spatial dimensions, padding, channels, posterior heads, and one-channel 64×64 decoder logits. Preserve loss units and evaluation. Architecture changes are a separate condition; they cannot establish fidelity of the original MLP reproduction.

### Phase E — lock configuration and final seed sweep

After pilot decisions, lock architecture, objective, optimizer, training and classifier budgets, and evaluation conventions. Use fresh metric-test examples for the final report. Run model seeds 0–9 with three classifier initializations each for the original fixed-β comparison. Include a selected extension only as a separately labeled family.

Report all raw scores, unfiltered mean/variability, variability across trained model seeds, and separately labeled best-half summaries. Do not tune against the final test scores or retrospectively choose the cleanest seed to represent a family.

## 9. Evidence and reporting requirements

Reproduction has two separate assessments:

1. **Metric agreement:** compare VAE/β-VAE scores with paper values and explain protocol deviations and uncertainty.
2. **Coordinate separation:** inspect predefined Figure 6 response maps and traversals for X/Y isolation, scale/rotation selectivity, nuisance dependence, and unused coordinates across seeds.

Useful optional quantitative diagnostics include a latent-coordinate × factor response matrix, position-map dependence on X versus Y, and factor changes induced by controlled decoder traversals. Define these before comparing outcomes; identify them as additional diagnostics, not the paper's metric. Decoder interventions do not automatically establish real-world causal meaning.

Every new run uses a fresh timestamped external results directory, exact CONFIG/environment/seeds, dataset checksum, checkpoints, raw metric arrays, runtime, training curves, summaries, and code snapshot. Preserve old runs and resume provenance. Archive figure provenance and numerical support beside new analysis figures, and explicitly label source checkpoints when outputs live elsewhere.

Prepare and read relevant validation tests for user execution according to shared project norms. This document creation launches no tests or training. Keep task-specific supervision/MRL work in the original dSprites plan until the baseline question is resolved.

## 10. Paired repo-inspired retraining — 2026-10-13

Launched in detached tmux session `dsprites-retrain-2026_10_13__14_52_36`, through [the paired launcher](../NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh). The [comparison workflow](../NonLinearPBExperiments/scripts/run_dsprites_comparison.py) automatically runs training, the existing evaluations, then Figure 6 A/B, with persistent logs and an explicit final status.

Configuration: seed 0, 100 epochs, batch 256, Adam at 0.0005, Xavier-uniform weights/zero biases, ReLU decoder for both models. GPU 0 trains fixed-beta1 VAE and evaluates its untrained control. GPU 1 trains the capacity-controlled candidate with gamma 1000 and target 0→25 nats over 100000 updates. Both use the unchanged metric sampling protocol and a 2000-epoch classifier cap. This is a combined repository-inspired comparison, not a one-change ablation or original fixed-beta4 reproduction.

```text
Comparison root:
  /home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__14_52_36/
VAE run:
  vae_runs/exprmnt_2026_10_13__14_52_38/
Capacity candidate run:
  beta_vae_runs/exprmnt_2026_10_13__14_52_38/
Automatic Figure 6 destination:
  figure6/
Workflow launch log:
  /home/argha/Mat_embedding/files/results/launch_logs/dsprites-retrain-2026_10_13__14_52_36/workflow.log
```

The root's `vae.log`, `beta_vae.log`, `figure6.log`, `status.json`, `manifest.json`, and combined `results_summary.txt` track the workflow. Both 100-epoch runs, evaluations, and Figure 6 completed successfully. Total paired workflow runtime was 1468.57 seconds (24 minutes 28.57 seconds). New baseline/comparison test helpers were prepared and read; unit tests were not run under the project convention. Syntax was checked; the full training/evaluation workflow executed successfully.

### Completed results and evaluation review

| Condition | Classifier seed 0 | Seed 1 | Seed 2 | Unfiltered mean | Epoch-100 reconstruction NLL | Epoch-100 KL |
|---|---:|---:|---:|---:|---:|---:|
| New VAE, beta1, Adam/Xavier/ReLU | 83.42% | 83.06% | 82.54% | 83.01% | 17.8782 | 20.7103 |
| New capacity candidate, Adam/Xavier/ReLU | 98.44% | 98.56% | 98.48% | 98.49% | 92.5568 | 25.0004 |

The new untrained Xavier/ReLU control scored 69.54%, 69.66%, and 69.86% (mean 69.69%) with the 2000-epoch classifier cap. Do not directly attribute its difference from the earlier 38.75% control to initialization alone: that earlier control used a 200-epoch classifier cap.

For context, the earlier default 100-epoch VAE/β=4 pilots scored 76.93%/92.06%, with reconstruction NLL 39.8893/53.8243. The new runs change multiple settings together; improved metric accuracy cannot identify which individual change caused it. Capacity control is a different objective from fixed beta4.

Mean held-fixed-factor classifier accuracies across the three fits:

| Model | X-position | Y-position | Scale | Rotation |
|---|---:|---:|---:|---:|
| New VAE | 68.61% | 63.87% | 100.00% | 99.55% |
| Capacity candidate | 100.00% | 100.00% | 98.72% | 95.25% |

**Classifier convergence limitation:** All six trained-model classifier fits reached the 2000-epoch cap. Their selected weights still come from validation CE, but convergence remains unresolved. These are exploratory scores from one model seed per condition, not a ten-replica reproduction.

Inspected both training curves, both reconstruction grids, all four empirical-reference traversals for each trained model, the classifier results/confusion matrices, and combined Figure 6 A/B:

- VAE reconstruction is substantially sharper, with recognizable shape details. Several native coordinates mix position, shape, and orientation.
- The capacity model has mainly three active coordinates ($z_1,z_2,z_8$); the other seven contribute little KL. The active posterior variances are extremely small, approximately $10^{-8}$ for the two position-dominated coordinates and $10^{-6}$ for $z_8$.
- Capacity reconstructions are blurry, often losing shape identity. Position-dominated traversals still move along mixed directions and can change appearance; the scale-dominated coordinate also changes shape/elongation. The clean position/scale/rotation separation in the paper has not been established.
- Figure 6 has stronger position responses concentrated in two capacity coordinates, a scale response in $z_8$, and weak rotation response structure. This does not support treating the 98.49% scalar score as proof of one-factor-per-coordinate disentanglement.
- Capacity training maintained epoch-mean KL near 25 nats, but the final average regularization penalty was 114.2370. This averages $\gamma|\mathrm{KL}_{\mathrm{batch}}-C|$ over batches; it is not $\gamma$ times the absolute difference of the epoch-mean KL and target. The total loss 206.7936 includes that penalty plus reconstruction NLL 92.5568.

Figure outputs (permanent, together with arrays/provenance/snapshot):

```text
files/results/exprmnt_2026_10_13__14_52_36/figure6/
    figure6_AB_2026_10_13__15_16_48.{png,pdf}
    figure6_A_beta_vae_2026_10_13__15_16_48.{png,pdf}
    figure6_B_vae_2026_10_13__15_16_48.{png,pdf}
```

[Combined Figure 6 A/B](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__14_52_36/figure6/figure6_AB_2026_10_13__15_16_48.png), [VAE reconstructions](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__14_52_36/vae_runs/exprmnt_2026_10_13__14_52_38/beta_1_seed_0/reconstructions_2026_10_13__14_52_38.png), [capacity reconstructions](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__14_52_36/beta_vae_runs/exprmnt_2026_10_13__14_52_38/capacity_seed_0/reconstructions_2026_10_13__14_52_38.png).
