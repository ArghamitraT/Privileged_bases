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

### Quick-start (common run commands)
```bash
# Shared infra tests
python tests/run_tests.py --fast

# Per-experiment smoke tests
python experiments/exp7_mrl_vs_ff.py --fast
python experiments/exp8_dim_importance.py --fast
python experiments/exp9_dense_prefix.py --fast
python experiments/exp10_dense_multidim.py --fast

# Per-experiment unit tests
python tests/run_tests_exp7.py --fast
python tests/run_tests_exp8.py --fast
python tests/run_tests_exp9.py --fast
python tests/run_tests_exp10.py --fast

# Multi-dim wrapper (exp10 → exp8 for each of dims 8, 16, 32)
python scripts/run_exp10_8_multidim.py --fast --dims 8
```

---

## Project File Structure
- `config.py` — `ExpConfig` dataclass (all training/eval settings)
- `utility.py` — `get_path()`, `create_run_dir()`, `save_runtime()`, `save_code_snapshot()`
- `data/loader.py` — dataset loading (sklearn + fetch_openml for MNIST)
- `models/encoder.py` — MLP encoder: input_dim → hidden → embed_dim
- `models/heads.py` — SharedClassifier (mode A) and MultiHeadClassifier (mode B)
- `models/linear_ae.py` — LinearAutoencoder with QR orthogonalization (exp6)
- `losses/mat_loss.py` — MatryoshkaLoss, L1RegLoss, `build_loss()`
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
- All outputs go to: `Mat_embedding_hyperbole/files/results/`
- Each run creates a timestamped subfolder: `exprmnt_{timestamp}/`

### Every experiment/edit MUST include:
1. **Runtime logging** — `save_runtime(run_dir, elapsed)` at end of `main()`.
2. **Code snapshot** — `save_code_snapshot(run_dir)` copies entire `code/` folder into `run_dir/code_snapshot/`.
3. **Test file validation** — create/update test file; **read it** to verify coverage; do NOT run it (user runs tests).

### Mandatory outputs per run
- `training_curves.png` — loss-vs-epoch for all trained models. Never omit.
- `experiment_description.log` — what/why/expected outcome + full config dump.
- `results_summary.txt` — accuracy tables, per-seed raw values, key metrics.
- `runtime.txt` — total elapsed time (seconds).
- `code_snapshot/` — exact copy of code/ at run time.

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

## Conda Environment
- Current env: `mrl_env` → `env/mrl_env.yml`
- Create: `conda env create -f env/mrl_env.yml`
- Activate: `conda activate mrl_env`

## Dependencies
- Core: numpy, pandas, seaborn, matplotlib, sklearn, scipy
- Deep learning: pytorch (added in exp1)
- Optional: umap-learn (exp2, install via conda-forge)
