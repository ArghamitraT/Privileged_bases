# disentanglement_lib — Reproduction Notes

Plan: `plans/CODEX_disentanglement_lib_CPU_reproduction_AEOLUS.md` (§1c execution plan).
Logs: `reproduction_outputs/stageA_logs/`.

---

## Repository

| | |
|---|---|
| Path | `code/NonLinearPB/disentanglement_lib/` |
| Remote | `https://github.com/google-research/disentanglement_lib.git` |
| Commit | `86a644d4ed35c771560dc3360756363d35477357` (2020-08-27, "internal change") |
| Branch | `master` (no branch created; clone is never committed to) |
| Working tree at start | clean |

## Machine

| | |
|---|---|
| OS | Linux 6.8.0-136-generic x86_64 |
| CPU | AMD Ryzen Threadripper 3970X 32-Core (64 threads, AVX2) |
| RAM | 251 GB |
| GPU | 2× NVIDIA RTX 3090, driver 550.144.03 (not used in Stage A) |
| conda | 25.3.1 (`~/miniconda3`, channel `defaults`) |
| Container runtime | none (docker / apptainer / singularity / podman absent) |

---

## Stage A — environment `dlib-tf1-cpu`

### Versions (installed 2026-10-08; full list: `reproduction_outputs/env_dlib-tf1-cpu_freeze.txt`)

```text
Python:                 3.7.16 (conda defaults)
TensorFlow:             1.14.0 (CPU wheel, built without CUDA)
TensorFlow Estimator:   1.14.0
TensorFlow Probability: 0.7.0
TensorFlow Hub:         0.5.0
NumPy:                  1.16.6
SciPy:                  1.4.1
gin-config:             0.3.0
protobuf:               3.20.3
gast / h5py:            0.2.2 / 2.10.0
scikit-learn / pandas:  0.20.4 / 1.0.5   (sklearn was 0.22.2.post1 — see Compatibility issue 2)
matplotlib / seaborn:   3.2.2 / 0.10.1
pillow / imageio:       6.2.2 / 2.9.0   (pillow was 7.2.0 — see Compatibility issue 1)
absl-py:                2.1.0 (unpinned, pulled by TF; imports fine)
pip / setuptools:       23.3.2 / <68
CPU execution confirmed: yes — tf.matmul ran in a tf.Session; is_gpu_available() = False
```

### Deviations from the §7b.3 pins

| Package | Planned | Used | Reason |
|---|---|---|---|
| pillow | 7.2.0 | **6.2.2** | Compatibility issue 1 (installed fine, failed at runtime) |
| scikit-learn | 0.22.2.post1 | **0.20.4** | Compatibility issue 2 (installed fine, failed at runtime in evaluation) |

All other pins installed and work as planned.

### Changes made to the original repo

None to source. `pip install -e .` created the untracked build artifact
`disentanglement_lib/disentanglement_lib.egg-info/` (only `git status` change).

### Compatibility issues

**1. Pillow ≥7 breaks `scream_dsprites` (and Cars3D) — fixed by pinning `pillow==6.2.2`.**
- File: `disentanglement_lib/data/ground_truth/dsprites.py:182` (`ScreamDSprites.__init__`);
  same pattern in `cars3d.py` (`pic.thumbnail((64, 64, 3), ...)`).
- Problem: `scream.thumbnail((350, 274, 3))` passes a 3-tuple. Pillow 7.x `thumbnail()`
  does `x, y = map(math.floor, size)` → `ValueError: too many values to unpack (expected 2)`.
  Pillow 6.x only indexed `size[0]`, `size[1]`, so the 3rd element was ignored.
- Change: environment only — `pip install pillow==6.2.2` (+ re-asserted low-level pins).
  **No library source edited.** `pip check` clean afterwards.
- Logs: `11_data_check.log` (failure), `12_pillow_downgrade.log`, `13_data_check_rerun.log` (pass).

**2. scikit-learn ≥0.22 breaks `downstream_task_logistic_regression` — fixed by pinning `scikit-learn==0.20.4`.**
- Found in: Step 6 smoke test (`reproduction_outputs/smoke_beta_vae_seed0/stdout.log`, run kept as `FAILED`).
- Where: `evaluation/metrics/downstream_task.py:89` → `utils.logistic_regression_cv()` =
  `LogisticRegressionCV(Cs=10, cv=KFold(n_splits=5))`, at `num_train = 10`.
- Problem: with 10 training points, the 5 folds contain different subsets of a factor's classes.
  sklearn 0.22 changed `LogisticRegressionCV` defaults (`multi_class` 'ovr' → 'auto' ⇒ multinomial,
  solver 'liblinear' → 'lbfgs'); the multinomial path builds per-fold coefficient arrays with
  different class counts → `ValueError: could not broadcast input array from shape (10,3,11) into shape (10)`.
- Why 0.20.4: the original study (Locatello et al., 2018–19) ran with sklearn 0.20.x defaults
  ('ovr' + 'liblinear'), so this restores the original metric behaviour — the most faithful fix.
  Also affects defaults of the other sklearn-based metrics (GBT, LR probes) → closer to original.
- Change: environment only — `pip install scikit-learn==0.20.4` (+ re-asserted low-level pins).
  **No library source edited.** `pip check` clean. Log: `stageA_logs/14_sklearn_downgrade.log`.
- Re-validation: full test suite re-run into `reproduction_outputs/stageB_tests_sklearn020/`;
  smoke pipeline re-run (same trained model) into `reproduction_outputs/smoke_beta_vae_seed0_sklearn020/`.

Only warning seen at import: one-time TF XLA:CPU info message (harmless).

---

## Data (Milestone C, 2026-10-08)

`DISENTANGLEMENT_LIB_DATA=/home/argha/Mat_embedding/data`. Downloaded with `wget` per file
(not `dlib_download_data`, see plan §1b). Log: `10_download_data.log`.

| Dataset | File | Size | sha256 |
|---|---|---|---|
| dSprites | `data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` | 19 318 135 B | `857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2` |
| Scream | `data/scream/scream.jpg` | 4 219 156 B | `a9bd171fe0a8da87f90a467e30542be195fb2e2a46d650f5287421869e57d390` |

Not downloaded (decision 2026-10-08): smallNORB, Cars3D (links live; not needed yet);
Shapes3D (loader expects Google-internal `.npz`, public file is `.h5`); MPI3D-toy (URL returns 403).

### Verified with the library's own loaders (`named_data.get_named_ground_truth_data`)

Raw npz: `imgs (737280, 64, 64) uint8`; metadata `latents_names = (color, shape, scale,
orientation, posX, posY)`, `latents_sizes = [1, 3, 6, 40, 32, 32]`.

| Study name | Loads | num_factors | factors_num_values | observation_shape | dtype |
|---|---|---|---|---|---|
| `dsprites_full` | ✅ 5.6 s | 5 | [3, 6, 40, 32, 32] | [64, 64, 1] | float32 |
| `color_dsprites` | ✅ 5.6 s | 5 | [3, 6, 40, 32, 32] | [64, 64, 3] | float64 |
| `noisy_dsprites` | ✅ 5.7 s | 5 | [3, 6, 40, 32, 32] | [64, 64, 3] | float64 |
| `scream_dsprites` | ✅ 5.8 s (after pillow fix) | 5 | [3, 6, 40, 32, 32] | [64, 64, 3] | float32 |

**Factor index → name in the library** (`latent_factor_indices = [1..5]`; the constant
`color` latent of the raw file is dropped):

| idx | factor | #values | note |
|---|---|---|---|
| 0 | shape | 3 | categorical (square, ellipse, heart) |
| 1 | scale | 6 | ordinal |
| 2 | orientation | 40 | **circular** |
| 3 | posX | 32 | ordinal |
| 4 | posY | 32 | ordinal |

Sampling (`sample_factors`, `sample_observations_from_factors`) works; images in [0, 1].
Script: `reproduction_outputs/stageA_logs/check_data.py` (loads each variant, samples 4 factors + observations).

### Milestone A — PASSED (2026-10-08)

- [x] `import tensorflow` → 1.14.0, CPU-only
- [x] `import tensorflow_hub, tensorflow_probability, gin.tf`
- [x] `import disentanglement_lib` (resolves to the clone, editable)
- [x] core submodules import: `methods.unsupervised.{train,vae}`, `evaluation.evaluate`,
      `postprocessing.postprocess`, `visualize.visualize_model`, `config.reproduce`
- [x] CLI on PATH: `dlib_train`, `dlib_reproduce`, `dlib_tests`, `dlib_evaluate`, `dlib_postprocess`
- [x] `pip check`: no broken requirements

### Commands used (logs in `reproduction_outputs/stageA_logs/`)

```bash
conda create -n dlib-tf1-cpu python=3.7 -y                                   # 01_conda_create.log
E=~/miniconda3/envs/dlib-tf1-cpu/bin
$E/python -m pip install --upgrade "pip<24" "setuptools<68" wheel            # 02_pip_tooling.log
$E/pip install numpy==1.16.6 protobuf==3.20.3 gast==0.2.2 h5py==2.10.0       # 03_lowlevel.log
C='numpy==1.16.6 protobuf==3.20.3 gast==0.2.2 h5py==2.10.0 tensorflow==1.14.0 tensorflow-estimator==1.14.0'
$E/pip install tensorflow==1.14.0 tensorflow-estimator==1.14.0 $C            # 04_tensorflow.log
$E/pip install tensorflow-probability==0.7.0 tensorflow-hub==0.5.0 gin-config==0.3.0 $C   # 05_tfp_hub_gin.log
$E/pip install scipy==1.4.1 scikit-learn==0.22.2.post1 pandas==1.0.5 matplotlib==3.2.2 \
               seaborn==0.10.1 pillow==7.2.0 imageio==2.9.0 six future simplejson $C     # 06_scientific.log
cd disentanglement_lib && $E/pip install -e . --no-deps                      # 07_dlib_install.log
$E/pip check                                                                 # 08_pip_check.log
$E/pip freeze > reproduction_outputs/env_dlib-tf1-cpu_freeze.txt
# Milestone A checks (§7b.5), run from a scratch dir with
# DISENTANGLEMENT_LIB_DATA=~/Mat_embedding/data                              # 09_milestoneA.log
```

Re-passing `$C` on every `pip install` re-asserts the low-level pins so a later install
cannot silently upgrade numpy / protobuf / TF.

---

## Pipeline — Milestone E (read from source, 2026-10-08)

Sources: `bin/dlib_reproduce`, `methods/unsupervised/train.py`, `postprocessing/postprocess.py`,
`evaluation/evaluate.py`, `utils/results.py`, `config/unsupervised_study_v1/`.

`dlib_reproduce --model_num=N --output_directory=OUT` (study `unsupervised_study_v1`):

| # | Stage | What happens | Output |
|---|---|---|---|
| 0 | config | `sweep.get_config()[N]` → gin bindings + `model_configs/shared.gin`; `--only_print` stops here | stdout |
| 1 | train | `train.train_with_gin` → TPUEstimator(`use_tpu=False`), `tf_random_seed = model.random_seed`, data-sampling seed drawn from `RandomState(model.random_seed)`; `training_steps` steps; then exports encoder as TF-Hub module; then 1000-batch eval | `OUT/model/tf_checkpoint/`, `OUT/model/tfhub/`, `OUT/model/results/` |
| 2 | visualize | reconstructions, samples, latent traversals (GIFs) | `OUT/visualizations/` |
| 3 | postprocess | for `mean.gin` (`mean_representation`) and `sampled.gin` (`sampled_representation`): wraps the encoder into a representation TF-Hub module | `OUT/postprocessed/{mean,sampled}/tfhub/`, `…/results/` |
| 4 | evaluate | 9 metric configs × 2 representations = **18 evaluations**: `beta_vae_sklearn`, `dci`, `downstream_task_boosted_trees`, `downstream_task_logistic_regression`, `factor_vae_metric`, `mig`, `modularity_explicitness`, `sap_score`, `unsupervised` | `OUT/metrics/{mean,sampled}/<metric>/results/` |

Results layout (every stage, `utils/results.update_result_directory`): `results/gin/<step>.gin`,
`results/json/<step>_config.json`, `results/json/<step>_results.json`,
`results/aggregate/aggregate_results_<step>.json` (aggregates all previous steps).

Seeds:
- model: `model.random_seed` (from the sweep: seed = `model_num mod 50`).
- postprocess + evaluation seeds come from `np.random.RandomState(0)` in `dlib_reproduce`
  → **identical across models** (same eval samples for every model — good for cross-seed comparison).

Metric sample sizes (metric_configs): `num_train = 10 000`, `num_test/num_eval = 5 000`;
downstream tasks at `num_train ∈ {10, 100, 1000, 10000}`.

### Consequences for later steps

1. **No resume.** `train()` raises `ValueError` if `model_dir` exists and `--overwrite` is not
   set; with `--overwrite` it deletes the directory. An interrupted 300k-step run must restart
   from step 0 → launch runs detached (setsid/nohup) and never kill them casually.
   Checkpoints: `keep_checkpoint_max=1`.
2. **Matrices are not saved.** DCI writes only `informativeness_*`, `disentanglement`,
   `completeness`; MIG only `discrete_mig`. The latent × factor importance matrix
   (`dci.compute_importance_gbt`) and the MI matrix (`utils.discrete_mutual_info`) — needed for
   §19 / Step 8 — must be recomputed by our own script (in
   `disentanglement_lib_reproduction/`) by calling these library functions on the exported
   `postprocessed/mean/tfhub` representation. No library edit needed.
3. `--output_directory` must always be given (default is `output/{study}/{model_num}` under cwd).

---

## Model selection — Milestone F (verified with `dlib_reproduce --model_num=N --only_print`, 2026-10-08)

`model_num = dataset_idx × 1800 + model_idx × 50 + seed` (7 datasets × 36 model configs × 50 seeds = 12 600).

| model_num | dataset | model.name | vae.beta | model.random_seed |
|---|---|---|---|---|
| 0 | dsprites_full | beta_vae | 1.0 (= vanilla VAE) | 0 |
| 1 | dsprites_full | beta_vae | 1.0 | 1 |
| 49 | dsprites_full | beta_vae | 1.0 | 49 |
| 50 | dsprites_full | beta_vae | 2.0 | 0 |
| 100 | dsprites_full | beta_vae | 4.0 | 0 |

Common to all (shared.gin): `encoder.num_latent = 10`, `model.batch_size = 64`,
`model.training_steps = 300000`, Adam lr 1e-4 (β1 0.9, β2 0.999, ε 1e-8),
`conv_encoder` / `deconv_decoder`, `bernoulli_loss`.

**Chosen:** vanilla VAE seeds 0,1,2 → model_num **0, 1, 2**; β-VAE β=4 seeds 0,1,2 → model_num **100, 101, 102**.

Pretrained-model bucket (`storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/<n>.zip`)
returns **403** → all models must be trained here.

---

## Tests — Milestone B (2026-10-08) — **34 / 34 modules PASS**

Runner: `disentanglement_lib_reproduction/run_stageB_tests.sh` (launched detached; runs each
module of `bin/dlib_tests` individually with `python -m`, CPU only, `CUDA_VISIBLE_DEVICES=""`).
Logs: `reproduction_outputs/stageB_tests/<module>.log`, table: `summary.tsv`, marker: `DONE`.

- 34 modules, **272 test cases, 0 failures, 0 errors**; total wall time 693 s (11.5 min).
- Skips: 13 skipped cases across 11 modules, all `*.test_session` — TF1's deprecated
  `tf.test.TestCase.test_session` helper, which TF itself always skips. Not library tests.
- Slowest: `strong_downstream_task_test` 195 s, `train_semi_supervised_test` 153 s,
  `evaluate_test` 81 s, `train_test` 54 s.
- No environment changes and no source changes were needed for the tests.

| Area | Modules | Result |
|---|---|---|
| methods.shared / methods.unsupervised (vae, train) | 4 | PASS |
| postprocessing, utils (convolute_hub, results), data util | 4 | PASS |
| evaluation (evaluate + all 12 metric tests) | 13 | PASS |
| visualize (util, dataset, dendrogram, scores) | 4 | PASS |
| abstract_reasoning, UDR, semi-supervised, weak, validation | 9 | PASS |

### Tests re-run after sklearn → 0.20.4 (2026-10-08) — **34 / 34 PASS**

`OUT=reproduction_outputs/stageB_tests_sklearn020 bash disentanglement_lib_reproduction/run_stageB_tests.sh`
→ 34 modules, 272 cases, 0 failures; 565 s. The earlier 34/34 run (sklearn 0.22.2.post1) is kept in
`reproduction_outputs/stageB_tests/`. **Current environment = sklearn 0.20.4.**

---

## Step 6 — Smoke test + timing (2026-10-08) — **PASSED end-to-end**

### SMOKE TEST (pipeline correctness, not a reproduction run)

| | |
|---|---|
| Train | `dlib_train --gin_config=examples/model.gin --model_dir=reproduction_outputs/smoke_beta_vae_seed0/model` — dsprites_full, beta_vae β=4, 10 latents, seed 0, **5 steps**, 32 s |
| Attempt 1 | `dlib_reproduce --model_dir=… --output_directory=reproduction_outputs/smoke_beta_vae_seed0` → **FAILED** at `downstream_task_logistic_regression` (sklearn 0.22; Compatibility issue 2). Kept as record. |
| Attempt 2 | same model, sklearn 0.20.4, `disentanglement_lib_reproduction/run_stage6_reproduce_only.sh` → `reproduction_outputs/smoke_beta_vae_seed0_sklearn020/` — **exit 0, 913 s** |

Outputs confirmed (plan §22-H):

| Output | Path | Present |
|---|---|---|
| checkpoint | `smoke_beta_vae_seed0/model/tf_checkpoint/model.ckpt-5.*` | ✅ |
| representation export (TF-Hub) | `smoke_beta_vae_seed0/model/tfhub/` | ✅ |
| postprocessed representations | `smoke_beta_vae_seed0_sklearn020/postprocessed/{mean,sampled}/tfhub/` | ✅ |
| metrics | 18 × `metrics/{mean,sampled}/<metric>/results/json/evaluation_results.json` | ✅ 18/18 |
| visualization | `visualizations/{reconstructions,sampled,traversals,animated_traversals,interventional_effects}/` | ✅ |

Metric values (mean representation) are near chance, as expected for a 5-step model:
DCI disentanglement 0.017, MIG 0.005, β-VAE metric 0.236, SAP 0.008, **FactorVAE metric 0.0**
(FactorVAE score returns 0 when no latent passes its active-variance threshold — plausible for
an untrained model; re-check on a real run).

### TIMING (CPU, Threadripper 3970X, TF default threading, one job at a time)

| Item | Measured |
|---|---|
| Training speed | **≈ 24.6 steps/s** (2000-step run, `timing_beta_vae_seed0_2000steps/`, 115 s incl. setup; loss 160.7, recon 112.1, KL 12.2 at step 2000) |
| Full evaluation (`dlib_reproduce` after training: visualize + 2 postprocess + 18 metrics) | **913 s ≈ 15 min** (slowest: GBT downstream 121–124 s, DCI 105–110 s, β-VAE metric ≈ 87 s each) |

**Estimate for one study model (300 000 steps):** 300 000 / 24.6 ≈ 12 200 s ≈ **3.4 h training**
+ ≈ 15 min evaluation ≈ **3.7 h per model** (CPU, run alone). 6 models (model_num 0–2, 100–102)
sequentially ≈ **22 h**. Parallel throughput not yet measured — TF uses all cores by default, so
concurrent runs will contend; thread caps per run (`inter/intra_op_parallelism`) would need to be
set via env/config without editing the library, or runs launched with `taskset`.
