# Codex Task: Reproduce `disentanglement_lib` Before Any PyTorch Reimplementation

## 1. Context

We are working on a research project about **disentanglement, identifiability, and privileged bases in learned representations**.

Before implementing anything new in PyTorch, we want to understand and reproduce the behavior of the original disentanglement benchmark library:

- Repository: `google-research/disentanglement_lib`
- Main purpose: reproduce classic disentanglement experiments such as VAE, beta-VAE, FactorVAE, DIP-VAE, etc.
- Initial dataset of interest: **dSprites**
- Immediate scientific goal: understand whether latent *factors* may be recovered while the identity/order of latent coordinates changes across random seeds.

For example, we eventually want to inspect behavior like:

```text
Seed 1:
    z2 -> x-position
    z5 -> scale
    z8 -> orientation

Seed 2:
    z1 -> scale
    z4 -> x-position
    z7 -> orientation
```

The point is **not yet to develop a new model**.

The point is:

1. reproduce the original library,
2. understand exactly how its training/evaluation pipeline works,
3. verify a small baseline experiment,
4. inspect latent-coordinate behavior across seeds,
5. only then consider a modern PyTorch reimplementation.

---

# 1b. Layout decisions (2026-10-08) — these override any path shown later in this file

| What | Location |
|------|----------|
| Cloned library | `~/Mat_embedding/code/NonLinearPB/disentanglement_lib/` |
| Datasets (`DISENTANGLEMENT_LIB_DATA`) | `~/Mat_embedding/data/` (shared project data folder → dSprites lands in `~/Mat_embedding/data/dsprites/`) |
| Run outputs | `~/Mat_embedding/code/NonLinearPB/reproduction_outputs/<model>_seed<N>/` |
| Reproduction log | `~/Mat_embedding/code/NonLinearPB/REPRODUCTION_NOTES.md` |

```bash
export NLPB=~/Mat_embedding/code/NonLinearPB
export DISENTANGLEMENT_LIB_DATA=~/Mat_embedding/data
```

> **Warning — do not run `dlib_download_data` blindly.** It downloads *every*
> dataset (small_norb, cars, dSprites, scream, mpi3d_toy) into the **current
> working directory**, and its cars step runs `mv data/cars . && rm -r data`
> relative to the cwd. Run from `~/Mat_embedding`, that would delete the whole
> project `data/` folder. For dSprites, download the single file directly
> (see Milestone C).

---

# 1c. Execution plan (agreed 2026-10-08)

**Current scope (started 2026-10-08, running inside tmux session `dlib`; long jobs launched detached with logs + DONE/FAILED markers):
Steps 3–6 (Milestone B → smoke test).** Stop after Step 6 with timings, before any
300k-step run. Steps 7–8 need a separate go-ahead.

**Code location:** all code we write for this task (analysis scripts, helpers, runners)
goes in `code/NonLinearPB/disentanglement_lib_reproduction/`. The clone
`disentanglement_lib/` stays untouched; outputs stay in `reproduction_outputs/`.

| Step | What | Pass condition |
|---|---|---|
| 0 — Bookkeeping ✅ | Create `code/NonLinearPB/REPRODUCTION_NOTES.md` (commit `86a644d`, machine info) and `code/NonLinearPB/reproduction_outputs/stageA_logs/`; add `.gitignore` in `NonLinearPB/` ignoring `disentanglement_lib/` and `reproduction_outputs/`. No git branch. | files exist |
| 1 — Environment ✅ | `conda create -n dlib-tf1-cpu python=3.7`; install pinned packages in the order of §7b.4; `pip install -e . --no-deps` | `pip check` clean; `pip freeze` saved to `reproduction_outputs/env_dlib-tf1-cpu_freeze.txt` |
| 2 — Milestone A ✅ (passed 2026-10-08) | Run the import checks of §7b.5 | TF 1.14.x, GPU not visible, hub/tfp/`gin.tf` import, `disentanglement_lib` imports, `dlib_*` on PATH |

Rules while executing:
- If a pin fails, change **only that package**, rerun, and log version + reason in `REPRODUCTION_NOTES.md`. TF 1.15 only if 1.14 cannot install.
- Do **not** edit library source. If a source change appears necessary, stop and ask.
- Do not touch `base` or any existing conda env; install nothing outside `dlib-tf1-cpu`.
- All command output is saved under `reproduction_outputs/stageA_logs/`.

### Steps 3–8 (agreed 2026-10-08)

Facts established while planning (2026-10-08):
- **Pretrained models are gone:** `storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/<n>.zip`
  returns 403 → every model must be trained by us; no exact reference numbers to diff against
  (sanity-check against Locatello et al. 2019 instead).
- **`model_num` mapping verified with `--only_print`:**
  `model_num = dataset_idx × 1800 + model_idx × 50 + seed`
  (7 datasets × 36 model configs × 50 seeds; `dsprites_full` = dataset 0, β-VAE β∈{1,2,4,6,8,16} = model_idx 0–5).
- Shared training config (`unsupervised_study_v1/model_configs/shared.gin`): 10 latents, batch 64,
  Adam lr 1e-4, conv encoder / deconv decoder, Bernoulli recon loss, **300 000 steps**.

| Step | Milestone | What | Pass condition / output |
|---|---|---|---|
| C/D ✅ | Data | dSprites + Scream downloaded; 4 dSprites variants load via library loaders; factor order documented (see `REPRODUCTION_NOTES.md`) | done 2026-10-08 (pillow → 6.2.2) |
| 3 ✅ | B — tests (34/34 PASS) | Run each of the 34 modules in `bin/dlib_tests` individually (`python -m …_test`), one log each in `reproduction_outputs/stageB_tests/`. Classify every failure as dependency/compatibility vs implementation. Fix env-only; if a source change seems needed → stop and ask. Failures in abstract_reasoning / UDR / semi- / weakly-supervised are recorded but non-blocking. | pass/fail table in `REPRODUCTION_NOTES.md`; all tests for `methods.unsupervised`, `methods.shared`, `evaluation.*`, `postprocessing`, `visualize` pass |
| 4 ✅ | E — pipeline | Read `bin/dlib_reproduce`, `config/reproduce.py`, `config/study.py`; document: train → visualize (reconstructions, samples, traversal GIFs) → postprocess (`mean`, `sampled`) → evaluate (9 metrics: β-VAE, FactorVAE, MIG, DCI, SAP, modularity/explicitness, unsupervised, downstream LR, downstream GBT) = 18 evaluations/model; inputs, outputs, directory layout | written section in `REPRODUCTION_NOTES.md` |
| 5 ✅ | F — model selection | Record config of candidate models via `--only_print` | vanilla VAE (β=1) seeds 0,1,2 = **model_num 0, 1, 2**; β-VAE β=4 seeds 0,1,2 = **model_num 100, 101, 102** |
| 6 ✅ | Smoke test (passed; ≈3.7 h/model on CPU) | `dlib_train --gin_config=examples/model.gin --model_dir=reproduction_outputs/smoke_beta_vae_seed0/model` (5 steps, dsprites_full, β=4, 10 latents) then `dlib_reproduce --model_dir=… --output_directory=reproduction_outputs/smoke_beta_vae_seed0`; time training steps/sec and each metric | checkpoint, TFHub export, both postprocessed reps, 18 metric JSONs, visualizations exist; timing table → CPU cost estimate for a 300k-step run. **STOP and report.** |
| 7 | Reproduction runs *(needs go-ahead)* | `dlib_reproduce --model_num=0` first; if clean, 1, 2, then 100–102. CPU: several runs in parallel with capped threads (64 cores). Each in `reproduction_outputs/{vae_seed{0,1,2}, beta_vae_b4_seed{0,1,2}}/` with `command.txt`, `environment.txt`, stdout/stderr | full pipeline outputs per run; metrics plausible vs Locatello et al. 2019 |
| 8 | §19 cross-seed analysis *(needs go-ahead)* | Start from library-native matrices — DCI importance matrix (latent × factor), MIG mutual-information matrix; per seed: strongest factor per latent + strength; across seeds: does the latent ↔ factor assignment permute? Handle categorical `shape` and circular `orientation` explicitly. Code in `code/NonLinearPB/disentanglement_lib_reproduction/` following project conventions (CONFIG block, timestamped run dir, figures pdf/png/svg) | table `seed | latent | strongest factor | strength` + cross-seed comparison |

Then report (§22).

---

# 2. Machine

Current machine:

```text
OS: Linux
GPU 0: NVIDIA GeForce RTX 3090, 24 GB
GPU 1: NVIDIA GeForce RTX 3090, 24 GB
Driver: 550.144.03
CUDA reported by nvidia-smi: 12.4
```

However, **do not use the GPU initially**.

The original library is old and relies on TensorFlow 1.x-era dependencies. RTX 3090 / Ampere compatibility makes an exact legacy GPU environment unnecessarily complicated.

## Decision

### Two-stage execution strategy: CPU validation, then GPU training

**Stage A — CPU first:** Set up an isolated legacy TensorFlow environment; inspect imports, run tests, load dSprites, and complete a short end-to-end VAE smoke test. CPU is chosen for software compatibility, **not** because the RTX 3090 is too slow or unsuitable. The original TensorFlow 1.14-era GPU stack predates Ampere, whereas this machine has Ampere GPUs and a modern driver. Avoid debugging CUDA until basic correctness is established.

**Stage B — GPU for real experiments:** After CPU validation, investigate an isolated GPU environment (prefer a compatible NVIDIA TensorFlow 1.x container, if available and verified to work with Ampere). Confirm that TensorFlow actually detects the RTX 3090 and runs a small operation and model step on it. Preserve the original model, dataset and Gin configuration. Compare a small CPU/GPU sanity result before launching full VAE / beta-VAE runs and three-seed experiments. Do not assume GPU compatibility merely because `nvidia-smi` works.

If legacy GPU support proves disproportionately difficult, report the blocker and the practical CPU fallback; do not rewrite the library in PyTorch or silently modernize its semantics.

---

# 3. High-level rule

## Preserve the original repository as much as possible.

Do **not** immediately:

- port the code to PyTorch,
- rewrite model classes,
- update TensorFlow APIs,
- modernize the entire repository,
- change evaluation metrics,
- change dataset preprocessing,
- replace Gin configuration,
- introduce Hydra,
- introduce Lightning,
- optimize the training loop.

We want to know what the **original implementation actually does**.

Small compatibility fixes are acceptable only when required to make the original code execute.

Every compatibility change must be documented.

---

# 4. Working directory

The parent project is `~/Mat_embedding`; all work for this task lives in
`~/Mat_embedding/code/NonLinearPB/` (see §1b).

Clone the original library there if it is not already present:

```bash
cd ~/Mat_embedding/code/NonLinearPB
git clone https://github.com/google-research/disentanglement_lib.git
cd disentanglement_lib
```

Immediately record:

```bash
git remote -v
git rev-parse HEAD
git status
```

Save the commit hash in the reproduction notes.

**No git branch** (decision 2026-10-08): the clone stays on `master` and is never
committed to. Any compatibility edit to the library is tracked by `git diff` against
commit `86a644d` and documented in `REPRODUCTION_NOTES.md`; our own notes, logs and
outputs live in folders under `code/NonLinearPB/` (see §1b), not in the clone.

---

# 5. First task: inspect before installing

Before changing the environment, inspect the repository.

Determine:

1. Python versions the code appears to expect.
2. TensorFlow version pinned by the repository.
3. TensorFlow Probability version.
4. TensorFlow Hub version.
5. NumPy / SciPy constraints.
6. `gin-config` dependency.
7. Whether `setup.py`, requirements files, or CI files define expected versions.
8. What the CLI entry points are.
9. What `dlib_reproduce` actually executes.
10. Where dSprites is loaded from.
11. Where outputs/checkpoints/metrics are written.
12. How random seeds are passed to the model.
13. Which configs define vanilla VAE and beta-VAE experiments.

Useful inspection commands may include:

```bash
find . -maxdepth 2 -type f | sort | head -200
grep -R "tensorflow" -n setup.py setup.cfg requirements* . 2>/dev/null | head -100
grep -R "entry_points" -n .
grep -R "dlib_reproduce" -n .
grep -R "dsprites" -n disentanglement_lib | head -100
grep -R "random_seed\|seed" -n disentanglement_lib | head -100
```

Do not blindly install packages until this inspection is done.

---

# 6. Environment strategy

Create an isolated environment.

Prefer one of:

```text
conda / mamba
```

Do not install into `(base)`.

A reasonable first attempt is an older Python version compatible with the repository, likely Python 3.6 or 3.7.

Example:

```bash
conda create -n dlib-tf1-cpu python=3.7 -y
conda activate dlib-tf1-cpu
```

But:

## Codex must inspect the repository first and choose the most compatible Python version based on the actual dependency metadata.

---

# 7. CPU-first TensorFlow setup and subsequent GPU validation

The target is an original-style **TensorFlow 1.x CPU environment**.

The repository historically used TensorFlow 1.x.

For **Stage A**, do not install `tensorflow-gpu` or configure CUDA; use CPU TensorFlow only. For **Stage B**, create a **separate** GPU-compatible environment/container. Never replace the working CPU environment in place.

Before installing, inspect the exact versions expected by the repository.

If the exact original package versions cannot install because of modern package-index / Python compatibility issues:

1. preserve TensorFlow 1.x behavior,
2. make the smallest possible version adjustment,
3. document the adjustment,
4. do not jump directly to TensorFlow 2.x.

If TensorFlow 1.15 is required as a compatibility substitute for 1.14, that is acceptable **only if necessary**, and the reason must be recorded.

---

# 7b. Stage A install specification (from inspecting the clone, 2026-10-08)

## 7b.1 What the repository declares

Inspected clone: `code/NonLinearPB/disentanglement_lib/`, branch `master`,
commit **`86a644d4ed35c771560dc3360756363d35477357`** (last commit 2020-08-27, "internal change").
No CI files, `requirements*.txt`, `setup.cfg` or `pyproject.toml` exist — `setup.py` is the
only dependency source. README only says "tested with Python 3 … on Linux".

`setup.py` (version 1.5):

| Package | Declared | Comment |
|---|---|---|
| `tensorflow` | `==1.14` (extra `[tf]`) | `[tf_gpu]` → `tensorflow-gpu==1.14` — **not** used in Stage A |
| `tensorflow_probability` | `==0.7` | matches TF 1.14 |
| `tensorflow_hub` | `>=0.2` | unbounded — modern versions need TF2 |
| `gin-config` | unpinned | |
| `numpy`, `six`, `future`, `simplejson`, `imageio`, `scikit-learn`, `seaborn` | unpinned | |
| `matplotlib>=1.5.2`, `pillow>=5.0.0`, `pandas>=0.23.0`, `scipy>=1.0.0` | lower bound only | |

**Undeclared but imported** (found by grepping all imports): `absl` (53 imports) and
`tensorflow_estimator` (TPUEstimator, 6 imports) — both are pulled in by the TF 1.14 wheel
(`tensorflow-estimator>=1.14,<1.15`, `absl-py`). `pkg_resources` (setuptools) and
`distutils` (stdlib, fine on Python 3.7) are also used.

## 7b.2 Why unpinned packages are the main risk

`pip install .[tf]` today would resolve every unpinned package to the newest release that
still supports Python 3.7 — mostly 2022–2023 versions that break TF 1.14. API usage found
in the code that constrains versions:

| Code usage | Constraint |
|---|---|
| `hub.Module`, `hub.create_module_spec` (TF1 Hub API, 9 imports) | `tensorflow-hub` must be a TF1-era release |
| `tfp.distributions.Bernoulli` | `tensorflow-probability==0.7.0` |
| `gin.tf`, `gin.tf.external_configurables` | `gin-config` with TF1 support |
| `PIL.Image.ANTIALIAS` (cars3d, norb); `Image.thumbnail((w, h, 3))` with a 3-tuple (scream_dsprites, cars3d) | `pillow<7` — Pillow ≥7.0 unpacks exactly 2 values in `thumbnail()` (ValueError); `ANTIALIAS` removed in 10 |
| `np.load(..., allow_pickle=True, encoding="latin1")` (dsprites) | `numpy>=1.10` |
| TF 1.14 C-extensions built against numpy 1.16 | `numpy<1.17` |
| TF 1.14 protobuf descriptors | `protobuf<3.21` (4.x raises "Descriptors cannot be created directly") |
| TF 1.14 autograph | `gast==0.2.2` (newer gast breaks TF1 autograph) |
| keras-applications (TF dep) | `h5py<3` |

## 7b.3 Pinned install list (Stage A, CPU)

Environment: **`dlib-tf1-cpu`**, **Python 3.7** (newest Python with TF 1.14 wheels).

| Package | Pin | Reason |
|---|---|---|
| python | `3.7` | TF 1.14 wheels: cp35–cp37 only |
| tensorflow | `==1.14.0` | as declared (CPU build) |
| tensorflow-estimator | `==1.14.0` | match TF |
| tensorflow-probability | `==0.7.0` | as declared |
| tensorflow-hub | `==0.5.0` | TF1 `hub.Module` API, contemporaneous with TF 1.14 (fallback: `0.6.0`) |
| numpy | `==1.16.6` | TF 1.14 ABI |
| protobuf | `==3.20.3` | last 3.x before TF1-breaking changes |
| gast | `==0.2.2` | TF1 autograph |
| h5py | `==2.10.0` | keras-applications / numpy 1.16 |
| gin-config | `==0.3.0` | 2020 release, has `gin.tf` for TF1 |
| scipy | `==1.4.1` | supports numpy 1.16 |
| scikit-learn | `==0.20.4` | original-study defaults (`LogisticRegressionCV` ovr+liblinear); 0.22.2.post1 crashed in `downstream_task_logistic_regression` (fixed 2026-10-08) |
| pandas | `==1.0.5` | `>=0.23`, numpy 1.16 OK |
| matplotlib | `==3.2.2` | numpy 1.16 OK |
| seaborn | `==0.10.1` | numpy 1.16 / pandas 1.0 OK |
| pillow | `==6.2.2` | `<7` for 3-tuple `thumbnail()` (was 7.2.0 — failed on scream_dsprites, fixed 2026-10-08) |
| imageio | `==2.9.0` | GIF writing in visualize |
| six, future, simplejson | latest for py3.7 | no known constraint |

These pins are the **starting hypothesis**, chosen to match the 2019–2020 era of the code.
If one fails to install or import, change only that package, then record the
version actually used and the reason in `REPRODUCTION_NOTES.md`. TF 1.15 is allowed only
if 1.14 cannot be installed (§7).

## 7b.4 Install commands

```bash
# 1. isolated env (never base)
conda create -n dlib-tf1-cpu python=3.7 -y
conda activate dlib-tf1-cpu
python -m pip install --upgrade "pip<24" "setuptools<68" wheel   # last versions supporting py3.7

# 2. TF1 core first, with pinned low-level deps
pip install "numpy==1.16.6" "protobuf==3.20.3" "gast==0.2.2" "h5py==2.10.0"
pip install "tensorflow==1.14.0" "tensorflow-estimator==1.14.0"
pip install "tensorflow-probability==0.7.0" "tensorflow-hub==0.5.0" "gin-config==0.3.0"

# 3. scientific stack, pinned
pip install "scipy==1.4.1" "scikit-learn==0.20.4" "pandas==1.0.5" \
            "matplotlib==3.2.2" "seaborn==0.10.1" "pillow==6.2.2" "imageio==2.9.0" \
            six future simplejson

# 4. the library itself — editable, NO [tf] extra, NO dependency resolution,
#    so pip cannot silently upgrade numpy/TF
cd ~/Mat_embedding/code/NonLinearPB/disentanglement_lib
pip install -e . --no-deps

# 5. confirm pip did not upgrade anything behind our back
pip check
mkdir -p ~/Mat_embedding/code/NonLinearPB/reproduction_outputs
pip freeze > ~/Mat_embedding/code/NonLinearPB/reproduction_outputs/env_dlib-tf1-cpu_freeze.txt
```

## 7b.5 Post-install checks (= Milestone A)

```bash
export DISENTANGLEMENT_LIB_DATA=~/Mat_embedding/data
python -c "import numpy, tensorflow as tf; print(tf.__version__, numpy.__version__)"
python -c "import tensorflow as tf; print('GPU visible:', tf.test.is_gpu_available())"   # must be False
python -c "import tensorflow_hub, tensorflow_probability, gin.tf; print('hub/tfp/gin OK')"
python -c "import disentanglement_lib; print('disentanglement_lib import OK')"
which dlib_train dlib_reproduce dlib_tests     # bin/ scripts installed on PATH
```

Deprecation / FutureWarnings are expected and ignored; only fatal errors matter.

## 7b.6 Not needed for Stage A

- No CUDA, cuDNN, `tensorflow-gpu`, or container runtime.
- No system packages (`wget` only for the dSprites download in Milestone C).
- No datasets other than dSprites (§1b).
- Note for Stage B: as of 2026-10-08 **no container runtime** (docker / apptainer /
  singularity / podman) is installed on this machine — Stage B is blocked on that, not on
  anything in Stage A.

## 7b.7 Stage A known risks (from inspecting the clone + machine, 2026-10-08)

| # | Risk | Mitigation |
|---|---|---|
| 1 | Unpinned deps resolve to versions that break TF 1.14 (protobuf 4, numpy ≥1.17, gast ≥0.3, TF2-only hub) | §7b.3 pins + `pip install -e . --no-deps` |
| 2 | Pins in §7b.3 (except TF/TFP) are hypotheses | change one package at a time, log in `REPRODUCTION_NOTES.md` |
| 3 | Python 3.7 is EOL: pip ≥24 / setuptools ≥68 dropped it; `defaults` channel may not serve it | pin `pip<24 setuptools<68`; fallback `-c conda-forge` |
| 4 | `dlib_reproduce` defaults `--output_directory` to `output/{study}/{model_num}` **relative to cwd** | always pass `--output_directory ~/Mat_embedding/code/NonLinearPB/reproduction_outputs/<model>_seed<N>` |
| 5 | Study config trains 300 000 steps (`unsupervised_study_v1/model_configs/shared.gin`); `dlib_reproduce` has no step override | smoke test = `dlib_train --gin_config=examples/model.gin` (5 steps, dsprites_full, β-VAE β=4, 10 latents) then `dlib_reproduce --model_dir=…`; measure steps/sec before any full run |
| 6 | Evaluation metrics (DCI gradient boosting, FactorVAE, β-VAE metric) are slow on CPU | expected; time them in the smoke test |
| 7 | No separate "vanilla VAE" in the study — it is `beta_vae` with β=1 (β ∈ {1,2,4,6,8,16}, 50 seeds each) | find matching `model_num` via `--only_print` (§14) |
| 8 | `bin/dlib_tests` runs 34 modules under `set -e` — first failure hides the rest | run each `python -m …_test` module individually, record pass/fail per module |

Machine checks (all OK): 64 cores with AVX2, 251 GB RAM (dSprites ≈ 3 GB in memory), 219 GB free disk,
no existing `dlib-tf1-cpu` env. No `channels_first`/NCHW ops in the code; TPUEstimator always runs with `use_tpu=False`.

---

# 8. Keep a reproduction log

Create:

```text
REPRODUCTION_NOTES.md
```

Record:

```text
Repository commit:
Python version:
TensorFlow version:
TensorFlow Probability version:
TensorFlow Hub version:
NumPy version:
SciPy version:
gin-config version:

Changes made to original repo:
1.
2.
3.

Compatibility issues:
1.
2.
3.

Commands used:
...
```

Every code modification should include:

- file,
- original behavior/problem,
- exact change,
- why it was needed.

---

# 9. Milestone A — import the library

Before training anything, this should work:

```bash
python -c "import disentanglement_lib; print('disentanglement_lib import OK')"
```

Then:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

During Stage A, confirm TensorFlow is running CPU-only. During Stage B, verify TensorFlow detects the GPU and executes a real TensorFlow operation on it.

It is fine if TensorFlow prints warnings caused by legacy APIs.

We care about fatal errors, not deprecation warnings.

---

# 10. Milestone B — run tests

Find the intended test command.

Possibilities may include:

```bash
dlib_tests
```

or repository-specific test invocations.

Run the project's own tests before changing model logic.

Record:

```text
tests passed:
tests failed:
failure reasons:
```

Do not "fix" scientifically meaningful code merely to satisfy tests without understanding the cause.

Separate:

- dependency/compatibility failures,
- actual implementation failures.

---

# 11. Milestone C — dataset download

Determine how the library expects datasets to be stored.

The original library typically uses an environment variable similar to:

```bash
DISENTANGLEMENT_LIB_DATA
```

Use the shared project data folder (§1b):

```bash
export DISENTANGLEMENT_LIB_DATA=~/Mat_embedding/data
```

Do **not** run `dlib_download_data` (see warning in §1b). Download only dSprites,
using the same URL and filename the script uses:

```bash
mkdir -p ~/Mat_embedding/data/dsprites
wget -O ~/Mat_embedding/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
    https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

Primary dataset:

```text
dSprites
```

After downloading, verify:

1. file exists,
2. loader can open it,
3. image shape is correct,
4. factor metadata is available,
5. sampling works.

---

# 12. Milestone D — understand dSprites in this library

Before model training, inspect the dataset implementation.

Document the exact factor ordering used by this repository.

dSprites conceptually contains factors such as:

```text
shape
scale
orientation
x-position
y-position
```

But **do not assume the internal indexing/order**.

Read the implementation and report:

```text
factor index -> factor name
number of values for each factor
observation shape
whether any factor is treated as constant / ignored
sampling mechanism
```

Write a tiny sanity script if helpful that samples several examples and prints their ground-truth factor values.

No fancy visualization is needed yet.

---

# 13. Milestone E — understand the reproduction pipeline

Inspect `dlib_reproduce`.

Explain, from the source code, what sequence it performs.

We expect roughly:

```text
training
    ->
postprocessing / representation extraction
    ->
evaluation
    ->
visualization
```

But Codex should confirm the actual steps.

Report:

```text
input:
output directory:
training config:
checkpoint:
TFHub/exported representation:
postprocessing:
evaluation metrics:
visualization:
```

Also determine how:

```bash
dlib_reproduce --model_num=<N>
```

maps `model_num` to:

- dataset,
- model,
- hyperparameters,
- seed.

---

# 14. Milestone F — inspect experiments before running them

Use any dry-run / print functionality provided by the repository.

For example, if supported:

```bash
dlib_reproduce --model_num=0 --only_print
```

The exact CLI may differ; inspect it first.

We want to identify a **small dSprites experiment**.

For each candidate model number, report:

```text
model_num
dataset
model
latent dimension
beta / regularization
batch size
training steps
seed
other important hyperparameters
```

Do not launch a giant sweep.

---

# 15. First scientific reproduction

Start with only:

```text
dSprites
+
one vanilla VAE
```

If that works, then:

```text
dSprites
+
one beta-VAE
```

Do not begin with FactorVAE, DIP-VAE, or a full benchmark sweep.

For the first run, use the original configuration as closely as possible.

If the full original number of training iterations is unnecessarily expensive for a smoke test:

### First run a short smoke test.

Purpose:

```text
Does the entire pipeline execute end-to-end?
```

Then run one proper baseline using the intended configuration, preferably on the verified GPU environment. If GPU validation is blocked, report it and continue CPU-only only when practical.

Clearly distinguish:

```text
SMOKE TEST
```

from

```text
REPRODUCTION RUN
```

---

# 16. What must be saved from each run

For each experiment, save/record:

```text
repository commit
environment versions
dataset
model
config
random seed
latent dimension
training steps
checkpoint path
representation/export path
evaluation result path
visualization path
wall-clock runtime
```

Also preserve stdout/stderr logs.

Structure (under `~/Mat_embedding/code/NonLinearPB/`, see §1b):

```text
reproduction_outputs/
    vae_seed0/
        command.txt
        environment.txt
        stdout.log
        ...
    beta_vae_seed0/
        ...
```

---

# 17. Multi-seed experiment

After one VAE run works end-to-end:

Run a very small seed study.

Suggested first pass:

```text
3 seeds
```

for one model.

Example:

```text
VAE:
seed 0
seed 1
seed 2
```

Optionally later:

```text
beta-VAE:
seed 0
seed 1
seed 2
```

Do not run 50 seeds yet.

---

# 18. Scientific question we eventually want to inspect

The important question is not simply:

> Does the model learn the factors?

It is:

> Are the *same latent coordinates* associated with the same factors across independently trained runs?

We want to distinguish:

```text
factor recovery
```

from:

```text
coordinate/basis stability
```

A representation may recover the same semantic factors while assigning them to different latent coordinates across seeds.

That is relevant to our later work on:

```text
identifiability
privileged bases
objective-specific basis structure
Matryoshka representation learning
```

---

# 19. Initial latent-coordinate analysis

Do not invent a new complicated disentanglement metric yet.

For each trained model, extract latent representations for a sufficiently large dSprites sample.

We want an initial table resembling:

```text
seed | latent dimension | strongest associated factor | association strength
```

Possible simple associations:

- Pearson correlation for appropriate numerical factors,
- Spearman correlation,
- simple linear probe,
- mutual information if already supported by the library.

Use the simplest reasonable diagnostic first.

Be careful with:

- circular orientation,
- categorical shape.

Do not treat every factor as ordinary Euclidean continuous data without noting limitations.

The immediate purpose is exploratory:

```text
Does latent-coordinate identity permute/change across seeds?
```

---

# 20. Important distinction

If Seed 1 learns:

```text
z2 -> x-position
z5 -> scale
```

and Seed 2 learns:

```text
z7 -> x-position
z1 -> scale
```

that is **not necessarily a failure of disentanglement**.

It can indicate a permutation ambiguity / lack of stable coordinate identity.

This is precisely one of the phenomena we care about.

---

# 21. Do not start MRL yet

Do not implement Matryoshka Representation Learning in this task.

Do not compare MRL yet.

The correct order is:

```text
1. CPU: import/tests/dSprites/small end-to-end smoke test
2. GPU: separate environment and compatibility verification
3. reproduce original VAE/beta-VAE training and evaluate multiple seeds
4. understand baseline latent behavior
5. establish a clean data/model/evaluation interface
6. only later consider a modern PyTorch implementation and MRL
```

---

# 22. What Codex should report back

At the end of the first pass, return a concise report with these sections.

## A. Repository

```text
commit:
original dependency expectations:
important CLI commands:
```

## B. Environment

```text
Python:
TensorFlow:
TFP:
TF Hub:
NumPy:
SciPy:
Gin:
CPU execution confirmed:
```

## C. Compatibility changes

For each change:

```text
file/package:
problem:
change:
reason:
```

## D. Dataset

```text
dSprites successfully loaded: yes/no
path:
observation shape:
factor ordering:
factor cardinalities:
```

## E. Tests

```text
passed:
failed:
remaining issues:
```

## F. Reproduction command

Provide the exact command used.

## G. First model

```text
dataset:
model:
seed:
latent dim:
training steps:
output path:
success/failure:
```

## H. Pipeline outputs

Confirm whether these were produced:

```text
checkpoint
representation export
postprocessed representation
metrics
visualization
```

## I. Next blocker

Only report the **single next technical blocker**, if one exists.

Do not begin a broad refactor without asking.

---

# 23. Success criteria for this task

For the initial CPU milestone, complete the CPU items first. Before declaring the **full experimental reproduction** finished, address the GPU decision as well:

- [ ] repository cloned and commit recorded
- [ ] isolated CPU environment created
- [ ] original library imports
- [ ] TensorFlow 1.x-compatible runtime works
- [ ] dSprites downloads/loads correctly
- [ ] dataset factors are understood and documented
- [ ] repository tests have been attempted and results recorded
- [ ] `dlib_reproduce` pipeline is understood
- [ ] a small dSprites VAE smoke test runs end-to-end
- [ ] one real baseline reproduction run completes (prefer verified GPU; document CPU fallback)
- [ ] outputs and exact configs are preserved
- [ ] no PyTorch rewrite has been started
- [ ] CPU-first correctness established before attempting GPU setup
- [ ] a separate GPU environment was validated for training, **or** its specific blocker and CPU fallback were documented
- [ ] full training/multi-seed experiments use GPU where verified and practical

---

# 24. Working style

When something fails:

1. Read the traceback.
2. Find the source of the incompatibility.
3. Prefer a dependency/environment fix over rewriting model code.
4. Make the smallest possible change.
5. rerun the smallest failing command.
6. document the fix.
7. proceed only after the current milestone works.

Do not make several speculative changes at once.

---

# 25. First commands to execute

Start with:

```bash
cd ~/Mat_embedding/code/NonLinearPB

if [ ! -d disentanglement_lib ]; then
    git clone https://github.com/google-research/disentanglement_lib.git
fi

cd disentanglement_lib

git remote -v
git rev-parse HEAD
git status
```

Then inspect:

```bash
python --version
ls
find . -maxdepth 2 -type f | sort | head -200
```

Then inspect dependency definitions and CLI entry points.

### Stop before modifying source code.

First report what environment the repository expects and propose the minimal CPU validation environment. After the CPU smoke test passes, propose and test a separate Ampere-compatible TF1 GPU environment before expensive experiments.

---

# Final principle

**Reproduce first. Modernize second.**

The value of this exercise is that later, when we implement the same experiment in PyTorch or add MRL, we will know exactly which behavior comes from the original disentanglement setup and which behavior comes from our changes.
