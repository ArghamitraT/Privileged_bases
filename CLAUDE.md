# Project: Matryoshka Embedding Interpretability

## Goal
Explore whether Matryoshka Representation Learning (MRL) increases embedding interpretability.

## Key Hypothesis: "Privileged Bases"
- Original axis-aligned data has privileged bases (certain axes carry semantic meaning)
- Test: random rotations should degrade decision tree accuracy more for embeddings than original data (or vice versa)
- This operationalizes "interpretability" as rotation-sensitivity

---

## Experiment Overview

> **Full details** (idea, inputs/outputs, eval metrics, how to run, expected output) are in **[EXPERIMENTS.md](EXPERIMENTS.md)**.
>
> **Rule: when adding a new experiment**, update BOTH this table in `CLAUDE.md` AND add a full description section in `EXPERIMENTS.md`.

| Exp | Script | One-line summary |
|-----|--------|-----------------|
| 1 | `exp1_prefix_curve.py` | Prefix accuracy curve: Standard vs Mat vs PCA |
| 2 | `exp2_cluster_viz.py` | Cluster geometry (t-SNE/UMAP) + silhouette vs prefix k |
| 5 | `exp5_seed_stability.py` | Cross-seed ordering stability + CKA analysis |
| 6 | `exp6_ortho_mat_ae.py` | Ortho + Mat AE should recover PCA eigenvectors |
| 7 | `exp7_mrl_vs_ff.py` | MRL vs FF vs L1 — ordering vs sparsity |
| 8 | `exp8_dim_importance.py` | Per-dim importance scoring + best-k vs first-k gap |
| 9 | `exp9_dense_prefix.py` | Dense prefix sweep (k=1..64), multi-seed |
| 10 | `exp10_dense_multidim.py` | Exp7 without FF, dense prefix (k=1..embed_dim), multi-dim |
| 11 | `exp11_learned_prefix_lp.py` | MRL vs PrefixLp (fixed p) vs LearnedPrefixLp (p learned) |
| 12 | `exp12_vector_learned_p.py` | MRL vs ScalarLearnedPrefixLp vs VectorLearnedPrefixLp (p per dim) |
| 13 | `exp13_mrl_cd34_supervised.py` | Supervised MRL on CD34 HSPCs — prefix clustering vs SEACells baseline |

### Quick-start (common run commands)
```bash
# Shared infra tests
python tests/run_tests.py --fast

# Per-experiment smoke tests
python experiments/exp7_mrl_vs_ff.py --fast
python experiments/exp8_dim_importance.py --fast
python experiments/exp9_dense_prefix.py --fast
python experiments/exp10_dense_multidim.py --fast
python experiments/exp11_learned_prefix_lp.py --fast
python experiments/exp12_vector_learned_p.py --fast
python experiments/exp13_mrl_cd34_supervised.py --fast

# Per-experiment unit tests
python tests/run_tests_exp7.py --fast
python tests/run_tests_exp8.py --fast
python tests/run_tests_exp9.py --fast
python tests/run_tests_exp10.py --fast
python tests/run_tests_exp11.py --fast
python tests/run_tests_exp12.py --fast
python tests/run_tests_exp13.py --fast

# Multi-dim wrapper (exp10 → exp8 for each of dims 8, 16, 32)
python scripts/run_exp10_8_multidim.py --fast --dims 8
```

---

## Project File Structure
- `config.py` — `ExpConfig` dataclass (schema + validation only; no experiment defaults — see Per-Experiment CONFIG Pattern below)
- `utility.py` — `get_path()`, `create_run_dir()`, `save_runtime()`, `save_code_snapshot()`
- `data/loader.py` — dataset loading (sklearn + fetch_openml for MNIST)
- `models/encoder.py` — MLP encoder: input_dim → hidden → embed_dim
- `models/heads.py` — SharedClassifier (mode A) and MultiHeadClassifier (mode B)
- `models/linear_ae.py` — LinearAutoencoder with QR orthogonalization (exp6)
- `losses/mat_loss.py` — MatryoshkaLoss, L1RegLoss, PrefixL1Loss, `build_loss()`
- `training/trainer.py` — generic training loop
- `evaluation/prefix_eval.py` — prefix sweep evaluation
- `experiments/` — one script per experiment (exp1–exp9)
- `tests/` — test runners and helper scripts (one per experiment)
- `env/` — conda environment yml files
- `figure/` — ad-hoc figures (not experiment outputs)

---

## Results / Output Convention

### Output Rules
- Code lives in `code/` (git-tracked). Results do NOT go in git.
- Full runs go to: `Mat_embedding_hyperbole/files/results/exprmnt_{timestamp}/`
- `--fast` (smoke test) runs go to: `Mat_embedding_hyperbole/files/results/test_runs/exprmnt_{timestamp}/`
- Use `create_run_dir(fast=args.fast)` from `utility.py` — it handles both paths automatically.

### `--use-weights` output convention
**When an experiment is re-run with `--use-weights <folder>`, a new timestamped
subfolder `exprmnt_{timestamp}` is created *inside* the weights folder, and all
outputs go there.**

Example: `--use-weights exprmnt_2026_04_01__22_04_54` produces:
```
files/results/exprmnt_2026_04_01__22_04_54/
└── exprmnt_2026_04_02__10_30_00/   ← new subfolder for this re-run
    ├── linear_accuracy_curve_....png
    ├── results_summary.txt
    ├── runtime.txt
    └── code_snapshot/
```

Implementation pattern (used in exp8, exp10):
```python
sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
run_dir   = os.path.join(weights_dir, sub_stamp)
os.makedirs(run_dir, exist_ok=True)
```
- Applies to: exp8, exp10 (and any future experiment that adds `--use-weights`).
- Do NOT write directly into `weights_dir` — `code_snapshot/` already exists there.

### Every experiment/edit MUST include:
1. **Runtime logging** — `save_runtime(run_dir, elapsed)` at end of `main()`.
2. **Code snapshot** — `save_code_snapshot(run_dir)` copies entire `code/` folder into `run_dir/code_snapshot/`.
3. **Test file validation** — create/update test file; **read it** to verify coverage; do NOT run it (user runs tests).

### Mandatory outputs per run
- `training_curves_{stamp}.png` — loss-vs-epoch for all trained models. Never omit.
- `experiment_description.log` — what/why/expected outcome + full config dump.
- `results_summary.txt` — accuracy tables, per-seed raw values, key metrics.
- `runtime.txt` — total elapsed time (seconds).
- `code_snapshot/` — exact copy of code/ at run time.

### PrefixL1 dimension-reversal convention
**For any model trained with `PrefixL1` loss, dimensions must be reversed before
any prefix sweep or analysis.**  Reason: the loss penalises dim 0 most heavily
(lightest penalty on dim `embed_dim-1`), so dim 0 carries the least information.
Flipping puts the most informative dimension first, making the prefix sweep
read best-first (matching MRL's convention).

- After extracting embeddings: `Z = np.ascontiguousarray(Z[:, ::-1])`
- Use key / legend label `"PrefixL1 (rev)"` everywhere (plots, tables, legends).
- Already implemented in: exp10 (step 9), exp8 (step 7).
- Must be applied in any future experiment that evaluates PrefixL1 with a prefix sweep.

### Figure filename timestamping convention
**All figure filenames must include a `_{YYYY_MM_DD__{HH_MM_SS}` suffix** so that
re-running an experiment (e.g. with `--use-weights`) into a fresh run folder never
overwrites previously saved images.

Implementation pattern (set once at the top of `main()`, pass to every plot call):
```python
fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
# then e.g.:
plot_all_curves(..., fig_stamp=fig_stamp)
plot_training_curves(run_dir, model_tags=[...], fig_stamp=fig_stamp)
```

Applies to: `training_curves`, `linear_accuracy_curve`, `1nn_accuracy_curve`,
`combined_comparison`, and any other per-run figure. The stamp goes **before** `.png`.

---

## Script Docstring Convention

Every script must start with a docstring including a **`Usage:`** section:
```
Usage:
    python experiments/exp7_mrl_vs_ff.py --fast        # smoke test (digits, 5 epochs)
    python experiments/exp7_mrl_vs_ff.py               # full run (MNIST, 20 epochs)
```
Update whenever a new CLI flag is added.

---

## Per-Experiment CONFIG Pattern

Every experiment script owns its complete configuration. **Do not put experiment defaults in `config.py`.**

### Rule
> "What does this experiment run with?" → look at the experiment file's CONFIG block.
> "What fields exist?" → look at `config.py` (schema + validation only).

### `config.py` role
- Typed dataclass container (`ExpConfig`) that shared modules accept instead of long argument lists.
- Validation in `__post_init__` (prefix sizes, head_mode, fractions).
- **No experiment-level defaults** — only `test_size=0.2` and `val_size=0.1` are stable enough to live here.

### Experiment file structure
Each experiment script must have a CONFIG block immediately after the imports:

```python
# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET       = "mnist"
EMBED_DIM     = 64
HIDDEN_DIM    = 256
HEAD_MODE     = "shared_head"
EVAL_PREFIXES = [1, 2, 4, 8, 16, 32, 64]
EPOCHS        = 20
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
L1_LAMBDA     = 0.05          # include only if the experiment uses L1
# any experiment-specific non-ExpConfig params go here too (e.g. MAX_1NN_DB)
# ==============================================================================
```

`main()` builds `ExpConfig` from these constants:
```python
cfg = ExpConfig(
    dataset=DATASET, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
    head_mode=HEAD_MODE, eval_prefixes=EVAL_PREFIXES,
    lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE,
    patience=PATIENCE, weight_decay=WEIGHT_DECAY,
    seed=SEED, l1_lambda=L1_LAMBDA,
    experiment_name="expN_name",
)
```

`--fast` overrides are hardcoded smoke-test values inside `main()` — they are not part of the CONFIG block (they are not something you tune).

---

## Shared Module Safety

Shared modules: `trainer.py`, `prefix_eval.py`, `loader.py`, `encoder.py`,
`heads.py`, `mat_loss.py`, `config.py`, `utility.py`.

**Rules:**
1. **Always run `python tests/run_tests.py --fast` before AND after any edit to a shared module.**
2. **Optional dependencies use try/except with a plain fallback** (e.g. `tqdm`).
3. **No nested tqdm `position=N` arguments** — macOS Terminal.app deadlocks on them.
4. **Keep experiment-specific progress features in the experiment file**, not shared modules.

---

## Known Issues / Gotchas

### Issue 1 — PyTorch + NumPy 2.x pin removed
- **Fix**: removed `numpy<2` pin. Current env runs numpy 2.2.6 + torch 2.2.2 without issues.

### Issue 2 — MNIST loader segfault on macOS
- **Root cause**: `.tolist()` on large tensors exhausts macOS stack.
- **Fix**: use `.numpy()` in `data/loader.py`. **Rule**: never call `.tolist()` on large tensors.

### Issue 3 — `import torch` hangs in test runner on macOS
- **Fix**: set `OMP_NUM_THREADS=1` etc. before imports; move torch-dependent tests to `tests/helper_<name>.py` subprocesses.
- **Rule**: the test runner process must never import torch directly.

### Issue 4 — numpy ufuncs crash on torch-backed arrays
- **Root cause**: numpy 2.x + torch bridge incompatibility when loading weights from disk.
- **Fix**: use `.tolist()` → `np.array(list, dtype=np.float64)` for any tensor going into numpy ufuncs.
- **Rule**: `.numpy().copy()` is NOT sufficient. Use `.tolist()` → `np.array(...)`.

### Issue 5 — umap-learn must be installed via conda-forge
- **Fix**: `conda install -c conda-forge umap-learn` (already in `mrl_env.yml`).

### Issue 6 — torchvision MNIST download segfaults on macOS
- **Fix**: use `fetch_openml("mnist_784")` from sklearn instead of torchvision.
- **Rule**: do not use `torchvision.datasets.MNIST` for downloading.

---

## Conda Environments

### Per-script convention
Every script docstring declares which environment it requires on a line:
```
Conda environment: <env_name>  (short note if non-obvious)
```
This line appears at the top of the `Usage:` block so it is the first thing seen when opening a script.

### Environments in use

| Environment | Scripts | Purpose |
|-------------|---------|---------|
| `mrl_env` | all `exp*.py` | MRL training + evaluation (PyTorch, sklearn, etc.) |
| `seacells` | `seacells_*.py` | SEACells metacell analysis (scanpy, SEACells) |

### mrl_env
- Definition: `env/mrl_env.yml`
- Create: `conda env create -f env/mrl_env.yml`
- Activate: `conda activate mrl_env`

### seacells
- Separate env for SEACells pipeline (incompatible PyTorch/scanpy versions)
- Activate: `conda activate seacells`

## Dependencies
- Core: numpy, pandas, seaborn, matplotlib, sklearn, scipy
- Deep learning: pytorch (added in exp1)
- Optional: umap-learn (exp2, install via conda-forge)
