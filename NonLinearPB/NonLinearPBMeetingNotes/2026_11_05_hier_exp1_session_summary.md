# Experiment 1 (hierarchical synthetic data): session summary, 2026-11-02 to 2026-11-05

Data: `hierarchical_images_v2`, 27 leaves (3 coarse × 3 mid × 3 fine), 64×64 images, 1600 / 200 / 200 per leaf.
Model: residual CNN-AE, 10-d latent, linear output. MRL loss = mean of the 10 prefix MSEs; prefix decode
x̂_{1:k} = D([z1..zk, 0..0]). AE and MRL-AE, seeds 0–4.

## HTML pages

All are private claude.ai pages (only you can open them while logged in). Local copies are under
`/home/argha/Mat_embedding/files/results/`.

| Page | Online | Local file |
|---|---|---|
| Experiment 1 report (training + all analyses) | https://claude.ai/artifact/PtNCPeCVLaeQJs7kU8m33S | `exp1_report/index.html` |
| Prefix reconstructions by epoch, final run lr 3e-4 (epochs 0–20 + 200) | https://claude.ai/artifact/6X1PEYsmhDD9utQHzTeDis | `exprmnt_2026_11_02__19_43_27/figures/recon_epochs/recon_epochs.html` |
| Prefix reconstructions by epoch, lr 6e-5 (epochs 0–25) | https://claude.ai/artifact/CdAxuLvf38MhC8P1QTbLCr | `exprmnt_2026_11_03__15_08_36/figures/recon_epochs/recon_epochs.html` |
| MRL prefix changes (epoch slider, 5 views) | https://claude.ai/artifact/6bNqsjbGQLXFY8BYCWncMT | `exprmnt_2026_11_03__15_08_36/figures/prefix_change/slider/mrl.html` |
| AE prefix changes (epoch slider, 5 views) | https://claude.ai/artifact/XafsLH5KZ6wUcsfnTkfaVh | `exprmnt_2026_11_03__15_08_36/figures/prefix_change/slider/ae.html` |
| Seed consistency of brightening (Jaccard, τ slider) | https://claude.ai/artifact/5SbJW89dUs7FJhMQhFWoUR | `exprmnt_2026_11_03__15_08_36/figures/prefix_change/jaccard/seed_jaccard.html` |
| MRL prefix changes, **5-d latent** (epoch slider, 5 views) | https://claude.ai/artifact/NdQWC4uPY7xCf25AHwpLM4 | `exprmnt_2026_11_05__11_44_34/figures/prefix_change/slider/mrl.html` |
| AE prefix changes, **5-d latent** (epoch slider, 5 views) | https://claude.ai/artifact/4DPukzgddNkCwuz9ryRk45 | `exprmnt_2026_11_05__11_44_34/figures/prefix_change/slider/ae.html` |
| Reconstructions + cosine to T_C/T_M/T_F (+ cos vs epoch), **5-d latent**, seed 0, 4 images | https://claude.ai/artifact/CBLfETePCXeAzk2o871mpG | `exprmnt_2026_11_05__11_44_34/figures/prefix_change/recon_cosine/recon_cosine.html` |
| z1 + one coordinate: pair-trained model vs 5-d MRL, cosine by epoch, seed 0, 4 images | https://claude.ai/artifact/B1H2gKeuoVVCT3KDodojmL | `exprmnt_2026_11_05__13_45_22/figures/pair_change/recon_cosine/recon_cosine.html` |
| z1 alone across all 5 seeds (pair model vs 5-d MRL), first-step cosine vs epoch, 4 images, with T_C/T_M/T_F | https://claude.ai/artifact/VspVwGPFhaqFaGPVaY5nve | `exprmnt_2026_11_05__13_45_22/figures/pair_change/z1_seeds/z1_seeds.html` |
| 5-d MRL, every prefix step × all 5 seeds, 5 views, cos vs epoch, 4 images | https://claude.ai/artifact/VqBMgRvyj737d7Rm2Ya1o9 | `exprmnt_2026_11_05__11_44_34/figures/prefix_change/steps_seeds/mrl_steps_seeds.html` |
| CIFAR-100: AE vs MRL-prefix vs MRL-pair reconstructions by epoch, 4 test images | https://claude.ai/artifact/N82GdA3ag5n31asTL6nYj8 | `exprmnt_2026_11_05__20_43_41/figures/recon_epochs/recon_epochs.html` |

## Experiment folders

All under `/home/argha/Mat_embedding/files/results/`; each has a README.md (aim, what was done, what happened).

| Folder | What it is | Status | Pages |
|---|---|---|---|
| `exprmnt_2026_11_02__13_07_00` | first full run, lr 1e-3; MRL seed 1 collapsed (`diagnosis_mrl_seed1_collapse/`) | superseded | — |
| `exprmnt_2026_11_02__14_39_14` | MRL seeds 1, 2 at lr 1e-3 with diagnostics + optimizer replay | diagnostic | — |
| `exprmnt_2026_11_02__15_03_31` | MRL seed 1 at lr 3e-4, 100 epochs | diagnostic | — |
| `exprmnt_2026_11_02__19_43_27` | **final run**: lr 3e-4, 200 epochs, AE + MRL seeds 0–4, classifier, evaluation | final | recon by epoch (6X1PEYsmhDD9utQHzTeDis) |
| `exprmnt_2026_11_03__12_23_52` | noise on the code (σ 0.2 / 0.5 / 2), MRL seed 0 (+ AE σ 0.5), 100 epochs | scouting | — |
| `exprmnt_2026_11_03__15_08_36` | lr 6e-5, 25 epochs, weights every epoch; all prefix-change analyses | done | CdAxuLvf38MhC8P1QTbLCr, 6bNqsjbGQLXFY8BYCWncMT, XafsLH5KZ6wUcsfnTkfaVh, 5SbJW89dUs7FJhMQhFWoUR |
| `exprmnt_2026_11_05__11_44_34` | as lr 6e-5 run but **5-d latent**, AE + MRL seeds 0–4 | done | NdQWC4uPY7xCf25AHwpLM4, 4DPukzgddNkCwuz9ryRk45, CBLfETePCXeAzk2o871mpG, VqBMgRvyj737d7Rm2Ya1o9 |
| `exprmnt_2026_11_05__13_45_22` | 5-d latent, **pair objective** ({z1}, {z1,zj}), seeds 0–4 | done | B1H2gKeuoVVCT3KDodojmL, VspVwGPFhaqFaGPVaY5nve |
| `exprmnt_2026_11_05__20_43_41` | **CIFAR-100** 32×32 colour, 10-d, lr 6e-5, 25 epochs, seed 0: AE, MRL-prefix, MRL-pair | done | N82GdA3ag5n31asTL6nYj8 |
| `exprmnt_2026_11_05__21_01_13` | **CIFAR-100**, same but lr 3e-4, 100 epochs (to convergence) | done | — |
| `exp1_report/` | compiled Experiment 1 report | — | PtNCPeCVLaeQJs7kU8m33S |

Page links: `https://claude.ai/artifact/<id>` (all private).

## Training

- **Final run** `exprmnt_2026_11_02__19_43_27`: lr 3e-4, 200 epochs, stable (lr 1e-3 collapsed through Adam
  update runaway; gradient clipping did not help). MRL packs all 27 leaves into z1 (a codebook: z1 alone
  reconstructs every leaf; coarse families contiguous along z1). Cleanup 2026-11-03: 14 spike dumps (9.3 GB)
  deleted, trigger lines kept in `spike_triggers_deleted_dumps.txt`.
- **Noise on the code** `exprmnt_2026_11_03__12_23_52` (MRL seed 0, 100 epochs; AE also at σ 0.5):
  z_k + σ·N(0,1) during training only. The encoder escapes by scaling z up (z1 SD 8 → 38 / 86 / 278 for
  σ 0.2 / 0.5 / 2) but not fully: at σ 2 k-means on z1 alone recovers the 3 coarse families exactly
  (ARI 1.0), at a cost in k = 1 MSE (0.00043 → 0.00158). Families stay contiguous at every σ.
- **Save-everything run**: deleted (local files) at your request.
- **lr 6e-5 run** `exprmnt_2026_11_03__15_08_36`: 25 epochs, weights saved after every epoch, launched
  3:08 PM, finished 3:51 PM on 2026-11-03, 0 failed, no spikes. Slow-down is about 3–4x, not 5x
  (epoch 25 at 6e-5 ≈ epoch 6–8 at 3e-4). Used for all analyses below.

- **5-d latent run** `exprmnt_2026_11_05__11_44_34` (2026-11-05): same as the lr 6e-5 run but z has 5
  dimensions; 11:44 AM–12:10 PM, 0 failed. Epoch 25, val MSE ×10⁻³: MRL z1 2.6 (10-d: 3.4), full 1.1 (10-d: 0.9);
  AE full 0.8 (10-d: 0.7).

- **Pair objective run** `exprmnt_2026_11_05__13_45_22` (2026-11-05): 5-d, loss = mean MSE of {z1}, {z1,z2} … {z1,z5};
  1:45–2:06 PM, 0 failed. z2…z5 learned as copies of one another (correlation ±1), all pair codes MSE 1.5×10⁻³ at epoch 25,
  full code 7.6×10⁻³ (never trained). z1 as in MRL.

- **CIFAR-100 run** `exprmnt_2026_11_05__20_43_41` (2026-11-05): 10-d, lr 6e-5, 25 epochs, seed 0, AE / MRL / pair; 8:43–8:54 PM, 0 failed.
  Epoch 25 val MSE ×10⁻³: AE full 20.5, MRL full 20.9, MRL z1 46.1, pair {z1,zj} 38.2 (z2…z10 copies again). Data: `data/cifar100/v1`.
- **CIFAR-100 lr 3e-4 run** `exprmnt_2026_11_05__21_01_13`: 100 epochs, 9:01–9:41 PM, 0 failed. Converged: AE best 19.5 (epoch 45, then overfits),
  MRL own 28.4 (plateau from epoch 50; full 20.2, z1 45.6), pair best 36.3 (epoch 79, unstable at the end).

Code changes: `--save-all`, `--epoch-weights`, `--grid-every` flags; lr-aware collapse guard; `--latent-dim`; `pair` objective.

## Analysis (all on the lr 6e-5 run unless noted)

1. **Prefix reconstructions by epoch** (sliders above), for both runs.
2. **Eval plan** written: `NonLinearPB/plans/hierarchical_Exp1_eval.md`.
3. **Prefix-change maps** (`scripts/hier_exp1_prefix_change.py`): D_k = x̂_{1:k} − x̂_{1:k−1}; references from
   training-set conditional means T_C / T_M / T_F; cosine alignment. 4 images, epochs 2–25, 5 seeds.
   MRL: step 1 aligns with coarse (≈0.58) at every epoch; step 2 starts mid + fine and becomes fine-only
   while mid moves into z1. AE: weak alignment, no ordering.
4. **Change-map slider pages** (MRL, AE): views changes per tile / shared / reconstructions /
   brighter per tile / brighter shared. A grey-out of tiles < 0.05 was tried and reverted.
5. **Cross-seed Jaccard of brightening** (`scripts/hier_exp1_seed_jaccard.py`, page with τ slider):
   at τ = 0 the score is chance level (≈0.35, background noise). Above τ ≈ 0.001, MRL is not more
   consistent than AE on this image (its later steps change very few pixels).
6. **Error attribution** (`scripts/hier_exp1_error_attribution.py`): R_k(p) splits the error drop of each
   step over pixels exactly. MRL: almost all of the drop is in z1, then 1→2 fixes the fine detail. AE: drop
   spread over all steps. PNG + SVG for c1m0f0 epochs 3 and 5, c0m2f0 epoch 5, seeds 0–2.
7. **PCA trajectories** (`scripts/hier_exp1_prefix_pca.py`), epoch 2: AE paths are long, start far from the
   input and differ per seed; MRL paths are short and start near the input (zoomed panel added).
8. **Spatial construction / 80% dominant-change sets** (`scripts/hier_exp1_spatial_trajectory.py`), epoch 2:
   MRL work concentrates in the first steps (6→7 … 8→9 nearly inactive); AE spreads it over all steps.
   All-seed grids per model.
9. **σ 0.2 / σ 2 table** filled in for the noise run.
10. **Report** compiled (first page in the table above).

11. **5-d latent pages** (`exprmnt_2026_11_05__11_44_34`): MRL and AE prefix-change sliders; reconstruction + cosine
    page (seed 0, 4 images, cos(D_k, T_C/T_M/T_F) vs epoch per step, `scripts/hier_exp1_recon_cosine_html.py`).
12. **Pair-code change maps** (`scripts/hier_exp1_pair_change.py`, `exprmnt_2026_11_05__13_45_22/figures/pair_change/`):
    D_1 = x̂_{z1} − decode(0), D_j = x̂_{z1,zj} − x̂_{z1}; pair model vs 5-d MRL (AE values also in the npz, not shown).
13. **z1 across seeds** (`scripts/hier_exp1_z1_seeds_html.py`): first step aligns with T_C (0.64–0.89) in every seed unless
    decode(0) already is that image's family (dumbbell in seeds 0, 3; boomerang in seeds 1, 2, 4), then ≈0 or negative.

Findings: MRL packs every leaf into z1; its 1→2 step consistently fixes mid/fine detail across seeds;
AE builds the image late and differently per seed; at σ 2, z1 alone separates the coarse families.

## What was added in W&B

Project `hier_exp1` (entity at3836-columbia-university, forge.coreweave.com/wandb); one run per model
(`<model>_lr<lr>_ep<epochs>`), grouped by exprmnt folder.

What each run logs:
- step log (with `--diagnostics`): loss, per-prefix loss, gradient / update / weight norms; spikes only as
  a chart marker `step/spike_recorded`, no email alerts;
- epoch metrics, per-coordinate latent stats (`epoch/z_{stat}_coord{i}`, `z_saturated`);
- `recon/grid`: one validation image per mid node; columns input, k = 1, 2, 3, 5, 10;
- an alert only when a run ends (finished / crashed / collapsed). No API key stored (uses `wandb login`).

Added this session:
- `--grid-every N` (default: epoch 1 + every 5 epochs). Every epoch for the save-everything run and the
  lr 6e-5 run, so training can be stepped through epoch by epoch in W&B;
- `tags`, with a `save_all` tag on runs that save full checkpoints;
- config now records `save_all` and `epoch_weights`;
- spike alert emails removed (marker only);
- groups: `exprmnt_2026_11_03__12_23_52` (noise), `exprmnt_2026_11_03__15_08_36` (lr 6e-5); the save-everything
  run's W&B entries were not removed.

## Still open

- End state of the lr 6e-5 run vs the 3e-4 run (z1 codebook, coarse contiguity).
- Chance line on the Jaccard page.
- Confirm σ 2 with more seeds and epochs.
- Stricter share (e.g. 50%) for the spatial maps.
- AE mean-fill variant (fill the missing code with the mean instead of 0).
