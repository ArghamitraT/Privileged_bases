# Experiment 1 — Training, Architecture, Loss, and Evaluation

**Status:** Data already generated. This document specifies the modeling and analysis only; do **not** regenerate or retune the data.

**Data:** use **v2**, `/home/argha/Mat_embedding/data/synthetic_data/hierarchical_images_v2/` (frozen 2026-11-02; uint8, load as `images / 255`; seeds and SHA-256 in its `config.json`). v1 is superseded: two parent pairs had identical fine details. Record the data version and checksums in every run's `config.json`. Details: `NonLinearPB/NonLinearPBMeetingNotes/2026_11_02_hierarchical_dataset_v2_report.md`.

## Scientific question

Does a nonlinear full-prefix Matryoshka autoencoder (MRL-AE) learn representations whose groupings progressively refine the known **3 coarse → 9 mid → 27 leaf** hierarchy, and are those relationships reproducible across seeds?

The hierarchy is a **ground-truth evaluation target**, not a training label. The dataset's reconstruction-cost ladder was designed to favor coarse before mid before fine, but this alone does not prove that the learned *groupings* nest.

## 1. Models and controls

Use the **same small residual CNN autoencoder** with two training objectives:

| Condition | Training objective | Role |
|---|---|---|
| Standard CNN-AE | Full-latent reconstruction MSE | Main objective control |
| CNN-MRL-AE | Average reconstruction MSE over all prefixes k = 1,…,10 | Main experimental model |

- Both models use a **deterministic** 10-dimensional latent (no VAE sampling, no KL term).
- Same input data, preprocessing, architecture, optimizer, training budget, and seed list.
- Train **5 seeds per condition**, for 10 main runs.
- **No labels** (coarse/mid/fine/leaf) in autoencoder training, checkpoint selection, or loss.
- An untrained/random encoder with the same architecture is an additional negative control during evaluation.
- Do **not** add full ConvNeXt, a second architecture, or classifier supervision in v1. Architecture ablations can come later.

## 2. Final architecture — small residual CNN-AE

**Input:** one 64 × 64 grayscale image, scaled to [0,1]. PyTorch tensor shape `[B, 1, 64, 64]`.

### Encoder

| Stage | Operation | Output tensor |
|---|---|---|
| Input | Image | `[B,1,64,64]` |
| 1 | Conv2d(1,32,kernel=3,stride=2,padding=1) + GELU | `[B,32,32,32]` |
| 2 | ResidualBlock(32) | `[B,32,32,32]` |
| 3 | Conv2d(32,64,kernel=3,stride=2,padding=1) + GELU | `[B,64,16,16]` |
| 4 | ResidualBlock(64) | `[B,64,16,16]` |
| 5 | Conv2d(64,128,kernel=3,stride=2,padding=1) + GELU | `[B,128,8,8]` |
| 6 | Flatten + Linear(8192,10) | **`z ∈ R^10`** |

**Residual block** at channel width `C`:

```text
h ------------------------------ (+) ---> output
 |                                  ^
 +--> LayerNorm2d(C) --> Conv3x3(C,C) --> GELU --> Conv3x3(C,C) --+
```

- Use **channelwise LayerNorm** at each spatial position (implement explicitly for channel-first tensors, e.g. permute to NHWC for `nn.LayerNorm(C)` and back). Do not accidentally normalize the entire image and channel tensor together.
- The residual branch has no spatial downsampling, so skip and main path shapes match.
- **No normalization or activation on the final 10 latent values.**
- **No global average pooling**; flattening preserves information about where local details occur.

### Decoder (shared across all MRL prefixes)

| Stage | Operation | Output tensor |
|---|---|---|
| Input | Latent z (10 values) | `[B,10]` |
| 1 | Linear(10,8192) + GELU, reshape | `[B,128,8,8]` |
| 2 | ResidualBlock(128) | `[B,128,8,8]` |
| 3 | Bilinear upsample ×2 + Conv2d(128,64,3,padding=1) + GELU | `[B,64,16,16]` |
| 4 | ResidualBlock(64) | `[B,64,16,16]` |
| 5 | Bilinear upsample ×2 + Conv2d(64,32,3,padding=1) + GELU | `[B,32,32,32]` |
| 6 | Bilinear upsample ×2 + Conv2d(32,16,3,padding=1) + GELU | `[B,16,64,64]` |
| 7 | Conv2d(16,1,3,padding=1), **linear output (no Sigmoid)** | `[B,1,64,64]` |

Use bilinear upsampling followed by convolution rather than transposed convolutions, to limit checkerboard artifacts. The **same decoder weights** are used for every prefix.

**Change 2026-11-02: no final Sigmoid.** With MSE loss a sigmoid output saturated: in the first full launch all 6 models (AE seeds 0–3, MRL seeds 0–1) collapsed to all-black output within 5–20 steps (pre-sigmoid values fell to about −188, gradient ~1e-10; the MSE gradient is multiplied by the sigmoid slope, which is ~0 there, and 88% of pixels are black). Lowering the learning rate to 1e-4 still left 1 of 5 AE seeds collapsed; a linear output at the planned lr 1e-3 learned in 7 of 7 test runs (MSE 0.003–0.008 after one epoch). Consequence: reconstructions can fall slightly outside [0, 1]. Training loss and the reported MSE use the raw output; the frozen reconstruction classifier and the reconstruction grids receive the output clipped to [0, 1]; the clipped MSE and the fraction of out-of-range pixels are also reported. (Sigmoid + BCE would not saturate this way; the plan keeps MSE to match the data's cost calibration.)

### Prefix mechanism

Given `z = encoder(x)` and prefix length `k`, define

`z^(k) = [z_1,...,z_k,0,...,0]` (length 10), then `x_hat^(k) = decoder(z^(k))`.

This **does not retrain a decoder or classifier for each k**. For the standard AE, evaluate the same zero-padded prefixes even though it was only trained on full-width latents.

## 3. Training losses

For a batch of B images and P=64×64 pixels, define per-image mean squared reconstruction error:

`MSE(x,x_hat) = (1/P) Σ_pixels (x - x_hat)^2`.

**Standard AE:**

`L_AE = mean_batch MSE(x, decoder(encoder(x)))`.

**Full-prefix MRL-AE:**

`L_MRL = (1/10) Σ_(k=1)^10 mean_batch MSE(x, decoder(z^(k)))`.

All prefixes k=1…10 have **equal weight** in v1. There is **no KL term, no label loss, no separate prefix head, and no separate prefix decoder**. The reconstruction cost used during training must be MSE, matching the generator validation.

## 4. Training plan

| Setting | Value |
|---|---|
| Dataset | Use already-generated, fixed train/val/test splits |
| Input | float32 images in [0,1], shape `[B,1,64,64]` |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 3e-4 (changed from 1e-3 on 2026-11-02; see note below) |
| Maximum epochs | 200 (changed from 100 on 2026-11-02) |
| Seeds | 5 matched seed identifiers across both objectives |
| Checkpoint | Best validation **own training objective** (AE full MSE; MRL mean-prefix MSE) |
| Training labels | None |

**Change 2026-11-02: lr 3e-4, 200 epochs.** At lr 1e-3, MRL seed 1 collapsed twice (epochs 28 and 19) to a constant output (val MSE 0.072): Adam's update size ran away over ~15 steps and all units of the last decoder layer died; MRL seed 2 survived with 7 loss-spike events (worst 45x). In a replay from the saved state 19 steps before the seed 1 spike (400 steps, same batches, one optimizer change per variant): baseline lr 1e-3 collapsed (twice), gradient clipping at 1.0 also collapsed, Adam eps 1e-6 and beta2 0.99 avoided the collapse but still spiked (worst batch loss 0.042 and 0.011; mean of the last 50 steps 0.0043 and 0.0015), and lr 3e-4 was smooth (worst 0.0004, last 50 steps 0.00033, the healthy level). lr 3e-4 was chosen as the cleanest single change; eps and beta2 were not tested beyond this replay. At 3e-4, MRL seeds 1 (100 ep), 2 and 3 (200 ep) did not collapse: 1, 3 and 4 short spike events (worst 23x), each recovered within 80 steps. 3e-4 learns more slowly; at 200 epochs MRL seed 2 reaches val 0.000143 vs 0.000148 for the best of 100 epochs at 1e-3, with matching per-prefix MSE. AE and MRL use the same setting. Evidence: `files/results/exprmnt_2026_11_02__13_07_00/diagnosis_mrl_seed1_collapse/`, `exprmnt_2026_11_02__14_39_14/` (diagnostics + replay), `exprmnt_2026_11_02__15_03_31/`, `exprmnt_2026_11_02__19_43_27/`. The final run is `exprmnt_2026_11_02__19_43_27` (MRL seeds 2 and 3 trained there first, the other 8 models, classifier and evaluation added by resuming it).

Use identical seed identifiers and deterministic dataset splits, but treat runs as independent. Track training and validation **full-width MSE plus per-prefix MSE** for *both* models, even when these are not the checkpoint-selection objective. Do not tune using the test set.

**Before interpreting hierarchy:** Confirm that the full-width `k=10` reconstructions for *both models* recover the visible coarse, mid, and fine details reasonably well. If full-width reconstruction fails, the bottleneck/model capacity or training is insufficient for the hierarchy test; report this rather than claiming a hierarchy failure.

## 5. Evaluation: three complementary views

Freeze each trained encoder and decoder. Use the same fixed test images for every model, prefix, and seed. Evaluate k=1,...,10; do not assume the three hierarchy levels must appear exactly at k=1,2,3.

### A. Linear probes — what can be recovered?

For every `k`, train **separate simple linear probes** on frozen `z_1:k` to predict:

1. **Coarse label:** 3 classes.
2. **Mid node:** 9 classes (joint coarse+mid identity).
3. **Leaf identity:** 27 classes (joint coarse+mid+fine identity).

Train probes on training embeddings, tune regularization on validation embeddings, and evaluate on held-out test embeddings; report accuracy and balanced accuracy with mean ± SD across 5 encoder seeds. No gradient from the probes reaches the autoencoder.

**Essential nesting-loophole check:** Also evaluate **conditional probes**: mid distinctions *within a fixed coarse parent* (3-way), and fine distinctions *within a fixed mid parent* (3-way). This distinguishes direct recoverability of a child from predicting its ancestor from a child code. Linear probes measure accessible information, **not proof of geometric nesting**.

### B. Decode and inspect — what each prefix reconstructs

For a fixed representative test panel covering **all 27 leaves** (with a few samples per leaf), show columns:

`original | k=1 | k=2 | k=3 | k=5 | k=10`.

Assess whether broad shape appears first, mid refinements next, and parent-anchored fine details last. Look particularly for *redraws*: does the coarse identity in a small-prefix image change substantially when another coordinate is introduced?

**Quantify alongside images:**
- Test MSE versus k, for AE and MRL-AE.
- Evaluate reconstructions against the known hierarchy by a **fixed, independently trained image classifier** on coarse (3), mid (9), and leaf (27) labels; report classification accuracy of reconstructed images as a *secondary* metric. Check this classifier on real test images first and keep it fixed across models and k.

A reconstruction shows what the **decoder uses**. It cannot, by itself, prove that absent details were not encoded but ignored by the decoder.

### C. Representation distances — what is grouped together?

For each k, define `d_k(i,j) = ||z_i[1:k] - z_j[1:k]||_2` on held-out images.

Compare pair distances for:

1. **Same leaf** (only nuisance differs).
2. **Same mid parent, different leaf** (fine distinction).
3. **Same coarse parent, different mid** (mid-level distinction).
4. **Different coarse parents** (coarse distinction).

Use **matched, balanced numbers of pairs** in every category and sample multiple parent branches. Report distance distributions, not just means. Absolute distances grow with k as dimensions are added; compare relative distances, discrimination of pair types, and nearest-neighbor group consistency. Standardize or normalize **only for analyses where justified**, using train-set statistics consistently, and report the choice; don't silently modify the model's latent values.

**Direct test of refinement vs redraw:** For each k, cluster the test embeddings into 3, 9, and 27 groups (e.g. k-means, fixed number of clusters); report adjusted Rand index (ARI) to the corresponding ground-truth level. Also compare partitions at successive k to check whether a 9-group partition stays largely *inside* its 3-group parent partition, rather than cutting across it. ARI alone does not answer nesting, so report the cross-parent mixing or partition-containment rate.

**Across seeds:** Compare pairwise distance rankings / same-group neighborhoods, or compare partitions for the *same test images* across seeds. Do not directly compare individual latent axes across seeds.

### D. Settled evaluation decisions (2026-11-02)

These fix the choices that sections A–C leave open. Settled before any code is written. With this section, architecture, loss, training and evaluation protocol are **final**; what remains is implementation and validation, not design.

| Component | Primary choice |
|---|---|
| Nesting (refine vs redraw) | 9 clusters at a later prefix inside 3 at an earlier prefix (and 27 inside 9): size-weighted containment minus a size-preserving shuffled baseline, reported with ground-truth ARI |
| Stability | Same cluster count at consecutive prefixes: ARI between the two partitions |
| Cross-seed / cross-model | Same test images, identical clustering settings |
| Probes | Global (3), pooled (9, 27), and parent-specific multinomial logistic regression |
| Reconstruction evaluator | Small position-preserving CNN, trained once per experiment on original images and frozen, stored in the experiment folder |
| Distances | Raw Euclidean (primary) + normalized by the mean random-pair distance at each k (sensitivity) |
| Output | `files/results/exprmnt_<timestamp>/` |

**1. Containment (refine vs redraw).** For a finer partition $\mathcal B$ and a coarser partition $\mathcal A$ of the same test images:

$$C(\mathcal B \mid \mathcal A) = \sum_{B\in\mathcal B}\frac{|B|}{N}\,\max_{A\in\mathcal A}\frac{|A\cap B|}{|B|}$$

- 1 means every finer cluster lies inside one coarser cluster; lower means finer clusters mix coarser ones.
- Two comparisons:
  - **Nesting** (primary): $\mathcal B$ = 9 clusters at prefix $k'$, $\mathcal A$ = 3 clusters at prefix $k \le k'$; likewise 27 inside 9. Answers "do finer groups split earlier groups?"
  - **Stability**: $n$ clusters at $k$ vs $n$ clusters at $k+1$, for $n = 3, 9, 27$. Answers "do groups survive when a coordinate is added?" Both partitions have the same number of clusters, so neither is "finer": measure agreement with **ARI between the two partitions** (permutation-invariant, symmetric), **not** with directional containment $C$.
- Nesting and stability are related but different questions (subdivide vs preserve existing groups); report both, alongside ground-truth ARI.
- **Shuffled baseline (nesting):** recompute $C$ after randomly permuting the finer cluster labels (cluster sizes kept), 100 permutations. Report observed $C$, the shuffled mean, and the difference. 9-in-3 is above 1/3 by chance alone, so the raw value is not interpretable without this. (ARI is already chance-corrected, so stability needs no shuffle.)
- $C = 1$ also when nothing changes, so always report it together with ARI to the ground-truth 3 / 9 / 27 labels at the same $k$: refinement means containment stays high **while** ARI to finer labels rises.
- Clustering: k-means on raw $z_{1:k}$, `n_init = 20`, fixed seed, on test images. Sensitivity: repeat with cluster counts ±1 around 3, 9 and 27.
- **Cross-seed and cross-model comparisons** use the **same test images** (all 5,400, same order) and **identical clustering settings** (algorithm, `n_init`, seed, cluster counts), so differences come from the representations, not the evaluation.

**2. Probes.** All are multinomial logistic regression on frozen raw $z_{1:k}$, with regularization $C$ chosen on validation embeddings from a fixed grid, fit on train, scored on test (accuracy and balanced accuracy, mean ± SD over 5 seeds).

| Probe | Task | Purpose |
|---|---|---|
| Global | 3 coarse groups | coarse recovery |
| Pooled | 9 mid groups (joint c, m); 27 leaves | overall recovery |
| Parent-specific | 3 mid siblings within each coarse parent (3 probes); 3 fine siblings within each mid parent (9 probes) | genuine local refinement; fine labels mean different things in different parents |

Report the parent-specific probes **per parent** as well as averaged. Flag dumbbell "straight" and ring "stem below", which have the weakest fine details in v2 (data report, caveat 1).

**3. Reconstruction evaluator.** Train **once per experiment run**, on original v2 training images with ground-truth labels, select on validation, then freeze. Store it **inside the experiment folder** (`exprmnt_<timestamp>/classifier/`), so every model, seed and $k$ in that run is scored by the identical network; its checksum is verified before every use. (Changed 2026-11-02 from a shared folder outside all runs. Consequence: classifier scores from *different* experiment runs come from different classifiers; to share one, pass `--classifier-dir` to the queue.)

- Architecture: `Conv(1→32, 3, stride 2) → ReLU → Conv(32→64, 3, stride 2) → ReLU → Conv(64→64, 3, stride 2) → ReLU → Flatten (64·8·8) → Linear`, three heads (3 coarse, 9 mid, 27 leaf).
- **Deviation from the first proposal:** no global pooling. Pooling discards *where* things are, and several v2 leaves differ mainly by position or larger structure (ring "through" vs "below", boomerang bulge vs inner bump, dumbbell keel vs spike).
- **Gate:** before it is used on any reconstruction, validate it on the original v2 **test** images at **all three levels** (coarse 3, mid 9, leaf 27; target ≥ 98% accuracy on each head) and record those accuracies in `config.json`. Otherwise reconstruction scores are not interpretable.
- Apply it to the reconstructions at every $k$ for both models. Report alongside pixel MSE and reconstruction grids. It was trained on clean images, so blur alone can lower its accuracy; treat it as a secondary metric.

**4. Distances.** Primary: raw $d_k(x_i, x_j) = \lVert z_{1:k}(x_i) - z_{1:k}(x_j)\rVert_2$, the scale the decoder receives. No per-coordinate standardization in the primary analysis: it would erase the coordinate importance MRL may learn.

- Pair types: same leaf; same mid, different leaf; same coarse, different mid; different coarse. Use balanced, matched numbers of pairs per type, sampled over all parent branches.
- Sensitivity (across seeds and models): divide by the mean distance over random test pairs at the same $k$.
- Also report pair-type comparisons **within each family**, because v2 noise differs by family (boomerang about half of the others).

**5. Output layout.**

```text
files/results/exprmnt_<timestamp>/
    config.json          # seeds, architecture, objective, data version + SHA-256, optimizer,
                         # probe grid, clustering settings, evaluator location
    classifier/          # classifier.pt + metrics.json (test gate, checkpoint SHA-256)
    checkpoints/{objective}/seed_{s}/best.pt
    training/            # per-epoch train/val full-width and per-prefix MSE, both models
    probes/              # linear_probes.csv (global, pooled, parent-specific)
    reconstructions/     # prefix_reconstruction.csv (MSE, evaluator scores) + grids
    distances/           # representation_distances.csv
    clustering/          # clustering_and_nesting.csv (ARI, containment, shuffled baseline)
    figures/
    summary.csv
```

The untrained-encoder control is evaluated with the same code and stored as objective `random`.

## 6. Primary claims and controls

| Claim | Evidence needed |
|---|---|
| Coarse-to-fine recovery | Probes and decoded images reveal coarse before mid before fine; representation distances support the same order |
| Genuine refinement | New groups at larger k mostly subdivide prior groups instead of redrawing them |
| Cross-seed reproducibility | Similar partition relationships and distance rankings across independently trained seeds |
| MRL-specific effect | Stronger and/or more consistent refinement and seed agreement for MRL-AE than for standard AE |

**Negative control:** untrained random encoder, evaluated in the same way. **Objective control:** standard AE with the same architecture and training setup. Report **full-width reconstruction quality** so improvements aren't interpreted without regard to model fit.

## 7. Suggested outputs

Superseded by the layout in section 5D.5; the file contents below still apply.

- `models/{objective}/seed_{s}/best.pt` and training/validation loss histories.
- `metrics/prefix_reconstruction.csv`: model, seed, k, MSE, optional reconstructed-image classifier scores.
- `metrics/linear_probes.csv`: model, seed, k, target level, accuracy, balanced accuracy; conditional targets included.
- `metrics/representation_distances.csv`: model, seed, k, pair type, mean/median/quantiles and comparisons.
- `metrics/clustering_and_nesting.csv`: model, seed, k, clusters, ARI, partition-containment / cross-parent mixing.
- `figures/reconstruction_contact_sheet.png`: fixed test images and prefix reconstructions.
- `figures/prefix_curves.png`: probe, reconstruction, and hierarchy summaries versus k (with seed variability).
- `config.json`: architecture, objective, data version/path, optimizer, training seeds, probe settings, and evaluation choices.

## 8. Interpretation guardrails

- Do **not** claim that MRL must recover the semantic tree exactly at k=1,2,3. A one-dimensional latent can encode more than three categories.
- The image cost ladder makes coarse-first learning plausible, but **does not force geometric nesting or seed stability**.
- A strong linear probe is not equivalent to a clean natural grouping, and a visually incomplete reconstruction is not proof that information is absent from the code.
- Standard AE prefix masking is out-of-training-distribution for most k; interpret AE prefix reconstructions cautiously and rely on probes and geometry as complementary comparisons.
- If capacity or decoder artifacts prevent strong full-width reconstruction, fix the **model/training** using validation data before evaluating the hierarchy; do not change the frozen data to favor MRL.
cla