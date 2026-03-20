# Project: Matryoshka Embedding Interpretability

## Goal
Explore whether Matryoshka Representation Learning (MRL) increases embedding interpretability.

## Key Hypothesis: "Privileged Bases"
- Original axis-aligned data has privileged bases (certain axes carry semantic meaning)
- Test: random rotations should degrade decision tree accuracy more for embeddings than original data (or vice versa)
- This operationalizes "interpretability" as rotation-sensitivity

## Project Structure
- `utility.py` - helpers: `get_path()` (project-root-relative path resolution), `create_timestamped_filename()`
- `privilegedBases_exp1.py` - single dataset experiment (Iris), Isomap embedding, 500 random rotations, boxplot visualization
- `privilegedBases_exp1_2.py` - multi-dataset version (Iris, Wine, Breast Cancer, Digits), same methodology
- `try.py` - scratch/exploration file
- `figure/` - saved PNG outputs (timestamped)
- `env/` - conda environment yml files (mrl_env.yml, mrl_env_2.yml, ...)

## Experiment 1: Prefix Performance Curve

### Idea
Train three models producing embed_dim-dimensional embeddings:
1. **Standard model** — normal CE loss on full embedding
2. **Matryoshka (Mat) model** — CE loss summed at each prefix scale
3. **PCA baseline** — PCA on training data (natural variance-based ordering)

At test time, keep only the first k dims (zero out the rest) and measure
classification accuracy. Mat embeddings should degrade gracefully; standard
embeddings should not. PCA serves as a sanity check.

### Key Design Decisions
- **Dataset**: flexible (default MNIST 784 -> 64). Use config to swap.
- **Embedding dim**: configurable (default 64)
- **Eval prefixes**: configurable list (default [1, 2, 4, 8, 16, 32, 64])
- **Classifier head modes** (both implemented, selectable via config):
  - **Mode A (shared_head)**: single Linear(embed_dim, n_classes) head.
    During Mat training AND eval, zero-pad the truncated prefix back to
    embed_dim before feeding into the head. Simpler, matches eval protocol.
  - **Mode B (multi_head)**: separate Linear(k, n_classes) head per prefix
    scale. During Mat training each head sees only its k dims. During eval,
    slice embedding[:, :k] and use the matching head. Cleaner gradients,
    more parameters.
- **Config**: use a Python dataclass (`ExpConfig`) for type safety and easy swapping
- **Training**: include learning rate, epochs, batch size, early stopping,
  validation split in config. Set seeds for numpy, torch, sklearn.
- **Evaluation fairness**: the standard model has never seen truncated inputs
  during training — this is intentional and is the point of the experiment.

### File Structure
- `config.py` - ExpConfig dataclass with all settings
- `data/loader.py` - flexible dataset loading (sklearn + torchvision), returns tensors
- `models/encoder.py` - MLP encoder: input_dim -> hidden -> embed_dim
- `models/heads.py` - SharedClassifier (mode A) and MultiHeadClassifier (mode B)
- `losses/mat_loss.py` - MatryoshkaLoss (sums CE at each prefix) + standard CE
- `training/trainer.py` - generic training loop (works for both standard and Mat)
- `evaluation/prefix_eval.py` - prefix sweep: zero out / slice, measure accuracy
- `experiments/exp1_prefix_curve.py` - main script: train all models, evaluate, plot

### Expected Output
- Plot: x-axis = prefix k (log scale), y-axis = accuracy
- Three lines: Standard, Mat, PCA
- Mat line stays high at small k; Standard drops off; PCA somewhere in between

## Experiment 2: Cluster Visualization & Geometry-Performance Analysis

### Idea
For each prefix dimension k, project the k-dimensional prefix embeddings z_{1:k}
from Standard, Matryoshka, and PCA models to 2D (via t-SNE and UMAP) and inspect
how class cluster structure evolves as k grows. Quantify with silhouette score,
intra-class distance, inter-class centroid distance, and separation ratio. Link
cluster geometry to classification accuracy at the same k.

### Key Design Decisions
- **Reuses saved exp1 weights** via `--use-exp1` flag (no retraining needed).
- **Same subsample indices** across all models for scatter plots — fair visual comparison.
- **Metrics computed on full test set** (subsampled to 2000 for speed); viz uses 3000.
- **t-SNE** always generated; **UMAP** generated only if `umap-learn` is installed.
- **k=1 edge case**: rendered as strip plot (x = embedding value, y = jitter).
- **k=2 edge case**: plotted directly, no dim reduction.

### File Structure
- `experiments/exp2_cluster_viz.py` — main script
- `run_tests_exp2.py` — test runner

### How to Run
```bash
python experiments/exp2_cluster_viz.py --use-exp1         # load exp1 weights, full MNIST
python experiments/exp2_cluster_viz.py --use-exp1 --fast  # load exp1 weights, subsampled
python experiments/exp2_cluster_viz.py --fast             # train from scratch on digits
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── tsne_grid.png        # rows=k in [1,2,4,8], cols=[Standard, Matryoshka, PCA]
├── umap_grid.png        # same structure with UMAP (if umap-learn installed)
├── cluster_metrics.png  # silhouette, intra distance, separation ratio vs k
├── combined_summary.png # accuracy (top) + silhouette (bottom) vs k — geometry<->perf
├── results_summary.txt  # full table: k × model × accuracy + all cluster metrics
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 5: Seed Stability

### Idea
Train Standard and Mat models multiple times with different random seeds (fixed
data split, varying model init). Measure whether the learned coordinate ordering
is reproducible.

### Two stability metrics
1. **Prefix variance** — mean ± std of accuracy at each prefix k across seeds.
   Mat should have lower std (stable ordering).
2. **Cross-seed dimension correlation** — for each pair of seed runs, compute the
   pairwise Pearson correlation matrix between embedding dimensions on the same
   test set. If dimension d encodes the same information across runs, the diagonal
   of this matrix will be large. Mat should have higher diagonal correlation.

### Key Design Decisions
- **Fixed data split** (`data_seed=42`): only model init + training randomness varies.
  This isolates the question "does the ordering change?" from "does the data change?"
- **5 model seeds** by default: `[100, 200, 300, 400, 500]`
- PCA baseline computed once (deterministic given fixed data).

### File Structure
- `experiments/exp5_seed_stability.py` — main script

### Expected Output (always generated)
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── seed_100/                     # per-seed subfolder
│   ├── standard_train.log
│   ├── mat_train.log
│   └── *_best.pt
├── seed_200/ ...
├── results_summary.txt           # mean ± std accuracy table + per-seed raw values
├── prefix_stability.png          # mean ± shaded std prefix curves
├── variance_comparison.png       # std bar chart: Standard vs Mat per k
├── correlation_heatmaps.png      # avg |corr| matrices (annotated if embed_dim≤16)
├── correlation_summary.txt       # mean diagonal correlation + per-pair breakdown
├── training_curves.png           # loss-vs-epoch for every seed
├── runtime.txt                   # wall-clock elapsed time
└── code_snapshot/                # full copy of code/ at run time
```

### Low-dim mode: `--low-dim` flag

Use `--low-dim` to set `embed_dim=10` and `eval_prefixes=[1,2,...,10]` for interpretable
per-dimension visualizations. Works with `--fast` for quick testing.

```bash
python experiments/exp5_seed_stability.py --low-dim          # full run
python experiments/exp5_seed_stability.py --fast --low-dim   # quick smoke test
```

**Note:** Set `dataset="digits"` in `config.py` for best results (10 classes match embed_dim=10).

### Additional plots when `embed_dim <= 16`
When in low-dim mode, the following bonus analyses are generated (gated on `embed_dim <= 16`):

1. **`cosine_similarity.png`** — Per-dimension bar chart (Std vs Mat).
   For each dim d, shows mean cosine similarity of that dimension's activations
   across seed pairs. Mat should show higher values (more stable encoding).

2. **`spearman_correlation.png`** — Per-dimension Spearman rank correlation.
   Rank-based alternative to Pearson, more robust to nonlinear scaling differences.

3. **`per_dim_correlation.png`** — Per-dimension Pearson correlation.
   Extracts diagonal of the cross-seed correlation matrix, decomposes the scalar
   "mean diag corr" into per-dimension detail.

4. **`cka_summary.png` + `cka_summary.txt`** — Linear Centered Kernel Alignment.
   CKA is rotation-invariant. Key diagnostic: high CKA + low per-dim corr = info
   present but scrambled (no privileged basis). High CKA + high per-dim corr =
   stable privileged basis.

5. **`tsne_dim_coloring_standard.png` + `tsne_dim_coloring_matryoshka.png`** —
   t-SNE of test set with grid of panels (rows=seeds, cols=dimensions). Each panel
   colors points by that dimension's value. Stable dimensions show consistent
   color patterns across seed rows.

## Results / Output Convention & Critical Checklist

### Output Rules
- Code lives in `code/` (git-tracked). Results do NOT go in git.
- All outputs go to: `Mat_embedding_hyperbole/files/results/`

### Every experiment/edit MUST include:
1. **Runtime logging** — `save_runtime(run_dir, elapsed)` at the end of `main()`.
   Captures wall-clock time for the full run in `runtime.txt`.
2. **Code snapshot** — `save_code_snapshot(run_dir)` copies the entire `code/` folder
   (minus `__pycache__`, `.git`, `figure/`) into `run_dir/code_snapshot/`.
   This guarantees full reproducibility — any result folder is self-contained and
   exactly reproducible without needing to know the git state.
3. **Test file validation** — When you edit or create an experiment script,
   **create or update a test file to verify it runs** (e.g. `run_tests_exp5.py`).
   Do NOT run the full test — that wastes tokens. Instead, **read the test file to
   verify it covers your changes**, and trust that the user will run tests later.

### Mandatory outputs
- **`training_curves.png`** — loss-vs-epoch for all trained models. Never omit.
- **`experiment_description.log`** — what/why/expected outcome + full config dump.
- **`results_summary.txt`** — accuracy tables, per-seed raw values, key metrics.
- **`runtime.txt`** — total elapsed time (seconds).
- **`code_snapshot/`** — exact copy of code/ at run time.
- Each run creates a timestamped subfolder always named `exprmnt_{timestamp}`, e.g.:
  ```
  files/results/exprmnt_2026_03_06__14_30_00/
  ├── experiment_description.log  # what/why/expected outcome + full config
  ├── standard_train.log          # per-epoch training log (standard model)
  ├── mat_train.log               # per-epoch training log (Mat model)
  ├── standard_encoder_best.pt    # best standard encoder weights
  ├── standard_head_best.pt       # best standard head weights
  ├── mat_encoder_best.pt         # best Mat encoder weights
  ├── mat_head_best.pt            # best Mat head weights
  ├── results_summary.txt         # accuracy table: k vs Standard / Mat / PCA
  ├── prefix_curve.png            # main result plot (accuracy vs prefix k)
  ├── training_curves.png         # train + val loss per epoch for both models
  ├── runtime.txt                 # total wall-clock time for the run
  └── code_snapshot/              # full copy of code/ at the time of the run
  ```
- `utility.py` provides:
  - `create_run_dir()` — create and return the timestamped output folder
  - `save_runtime(run_dir, elapsed_seconds)` — write runtime.txt
  - `save_code_snapshot(run_dir)` — copy code/ into code_snapshot/

## Experiment 6: Orthogonal Matryoshka Autoencoder ≈ PCA

### Idea
A linear autoencoder with orthogonal encoder columns + Matryoshka reconstruction
loss should recover PCA eigenvectors **and their ordering** exactly.

PCA = orthogonal directions ordered by variance explained. That is exactly what
Ortho + Mat + Reconstruction gives:
- **Ortho constraint** → forces independent directions (like eigenvectors)
- **Mat loss** → forces ordering by importance (like eigenvalue order)
- **Reconstruction** → finds directions that preserve maximum variance (like PCA)

### 2×2 Ablation Design
| | Standard Recon Loss | Matryoshka Recon Loss |
|---|---|---|
| **No ortho** | Vanilla AE | Mat AE |
| **Ortho** | Ortho AE | **Ortho Mat AE ← should ≈ PCA** |

Key prediction: Only Ortho + Mat recovers both eigenvectors AND ordering.

### File Structure
- `models/linear_ae.py` — LinearAutoencoder (tied weights, `orthogonalize()` via QR)
- `experiments/exp6_ortho_mat_ae.py` — main script with own training loop
- `run_tests_exp6.py` — test runner

### How to Run
```bash
python experiments/exp6_ortho_mat_ae.py           # full run (digits, embed_dim=10)
python experiments/exp6_ortho_mat_ae.py --fast    # smoke test (5 epochs)
python run_tests_exp6.py --fast                   # unit tests only
python run_tests_exp6.py                          # unit tests + e2e smoke
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── column_alignment.png       # |cos(W[i], PC[i])| per dim — 4 models
├── reconstruction_curve.png   # MSE vs prefix k — 5 lines (4 models + PCA)
├── explained_variance.png     # variance explained vs k
├── subspace_angle.png         # principal angle between learned vs PCA subspace
├── training_curves.png        # loss vs epoch for all 4 models (MANDATORY)
├── results_summary.txt        # column alignment, MSE, subspace angle tables
├── vanilla_ae.pt / mat_ae.pt / ortho_ae.pt / ortho_mat_ae.pt
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 7: MRL vs Fixed-Feature (FF) vs L1-Regularized Models

### Idea
Replicate the MRL paper's core comparison on MNIST, extended with an L1 ablation.

Five model families compared at every prefix k:
1. **Standard** — plain CE loss on full embed_dim embedding
2. **L1** — CE + λ·‖z‖₁ on embedding activations (sparsity, no ordering)
3. **MRL (Mat)** — CE summed at every prefix scale (sparsity + ordering)
4. **FF-k** — one dedicated model per k, trained with embed_dim=k
5. **PCA** — analytical baseline (variance-ordered components)

Two evaluation metrics:
- **Linear accuracy** — logistic regression probe on k-dim prefix
- **1-NN accuracy** — 1-nearest-neighbor (train set = database, test set = queries)

Key hypothesis: L1 ≈ Standard at small k (sparsity alone doesn't help prefix eval).
MRL beats L1 at small k → proves ORDERING is essential, not just sparsity.

### Shared module changes
- `config.py` — added `l1_lambda: float = 0.05` field
- `losses/mat_loss.py` — added `L1RegLoss` class and `"l1"` case in `build_loss`

### File Structure
- `experiments/exp7_mrl_vs_ff.py` — main script
- `run_tests_exp7.py` — test runner

### How to Run
```bash
python experiments/exp7_mrl_vs_ff.py --fast        # smoke test (digits, 5 epochs)
python experiments/exp7_mrl_vs_ff.py               # full run (MNIST, 20 epochs)
python experiments/exp7_mrl_vs_ff.py --use-exp1    # load MRL from exp1, train rest
python run_tests_exp7.py --fast                    # unit tests only
python run_tests_exp7.py                           # unit tests + e2e smoke
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── standard_encoder_best.pt / standard_head_best.pt
├── l1_encoder_best.pt       / l1_head_best.pt
├── mat_encoder_best.pt      / mat_head_best.pt
├── ff_k1_encoder_best.pt ... ff_k64_encoder_best.pt
├── training_curves.png         # loss vs epoch for Standard / L1 / MRL
├── linear_accuracy_curve.png   # linear accuracy: 5 lines vs k
├── 1nn_accuracy_curve.png      # 1-NN accuracy:   5 lines vs k
├── combined_comparison.png     # 2-panel (linear top, 1-NN bottom)
├── results_summary.txt
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 8: Per-Dimension Importance Scoring

### Idea
After Exp7 showed MRL beats L1 at prefix eval, Exp8 asks *why* by scoring each
embedding dimension's importance three ways and comparing "first-k" (standard
prefix eval) vs "best-k" (oracle selection of most informative k dims).

Three analyses on Standard, L1, MRL, PCA embeddings:
1. **Importance scoring (3 methods):**
   - `mean_abs[d]`  = mean(|z[:, d]|) — activation magnitude (free)
   - `variance[d]`  = var(z[:, d])    — spread of activations (free)
   - `probe_acc[d]` = logistic regression accuracy using only dim d
2. **Best-k vs First-k:** rank dims by each method → compare oracle-selected-k
   accuracy against prefix-k. Gap ≈ 0 for MRL (ordering enforced); Gap > 0 for L1/Standard.
3. **Method agreement:** Spearman rank correlation between the 3 methods per model.

### Key Reuse
- `train_single_model` and `get_embeddings_np` imported from `exp7_mrl_vs_ff.py` (no copy).
- No shared modules are modified.

### File Structure
- `experiments/exp8_dim_importance.py` — main script
- `run_tests_exp8.py` — test runner (6 unit tests + e2e smoke)

### How to Run
```bash
python experiments/exp8_dim_importance.py --fast        # smoke test (digits, 5 epochs)
python experiments/exp8_dim_importance.py               # full run (MNIST, 20 epochs)
python experiments/exp8_dim_importance.py --use-exp7 PATH  # load exp7 weights, no retraining
python run_tests_exp8.py --fast                         # unit tests only
python run_tests_exp8.py                                # unit tests + e2e smoke
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── training_curves.png         # MANDATORY (placeholder if --use-exp7)
├── importance_scores.png       # per-dim bar charts: 3 methods x 4 models
├── dim_importance_heatmap.png  # heatmap (models x dims), one panel per method
├── best_vs_first_k.png         # first-k vs best-k accuracy curves per model
├── method_agreement.png        # scatter plots + Spearman rho per model
├── results_summary.txt         # gap table, agreement table, top-5 dims table
├── runtime.txt
└── code_snapshot/
```

---

## Method (existing experiments — privileged bases)
1. Load dataset, train/test split (stratified)
2. StandardScaler on training stats
3. Isomap embedding (n_components = original dim, n_neighbors=10)
4. Train DecisionTree (max_depth=3) on: original, embedding
5. Apply 500 random orthogonal rotations, re-evaluate
6. Compare accuracy drop: original vs embedding

## How to Run

### Test files (one per experiment)
Each experiment has its own test runner. Run from `code/`:

```bash
# Shared infrastructure (modules used by all experiments)
python run_tests.py           # all infra tests (includes slow loader + trainer)
python run_tests.py --fast    # quick smoke — skips MNIST download and training
python run_tests.py --module config   # single module only

# Experiment 5: Seed Stability
python run_tests_exp5.py          # infra + unit tests + end-to-end smoke test
python run_tests_exp5.py --fast   # infra + unit tests only (skip e2e smoke)

# Experiment 2: Cluster Visualization
python run_tests_exp2.py          # unit tests + end-to-end smoke test
python run_tests_exp2.py --fast   # unit tests only (skip e2e smoke)

# Experiment 8: Dimension Importance Scoring
python run_tests_exp8.py          # unit tests + end-to-end smoke test
python run_tests_exp8.py --fast   # unit tests only (skip e2e smoke)
```

**Workflow**: Run `--fast` after every non-trivial edit; run the full suite before committing.

### Individual module tests (still work directly)
```bash
python utility.py
python config.py
python data/loader.py
python models/encoder.py
python models/heads.py
python losses/mat_loss.py
python training/trainer.py
python evaluation/prefix_eval.py
```

### Full experiments
```bash
python experiments/exp1_prefix_curve.py

# Experiment 5: Seed Stability
python experiments/exp5_seed_stability.py                    # full run (5 seeds, MNIST)
python experiments/exp5_seed_stability.py --fast            # smoke test (2 seeds, digits)
python experiments/exp5_seed_stability.py --low-dim         # low-dim full run (10D, digits)
python experiments/exp5_seed_stability.py --fast --low-dim  # low-dim smoke test
```

# Experiment 6: Ortho Mat AE ≈ PCA
python experiments/exp6_ortho_mat_ae.py           # full run
python experiments/exp6_ortho_mat_ae.py --fast    # smoke test

# Experiment 7: MRL vs FF vs L1
python experiments/exp7_mrl_vs_ff.py --fast        # smoke test
python experiments/exp7_mrl_vs_ff.py               # full run
python run_tests_exp7.py --fast                    # unit tests
python run_tests_exp7.py                           # unit tests + e2e smoke

# Experiment 2: Cluster Visualization (uses saved exp1 weights)
python experiments/exp2_cluster_viz.py --use-exp1         # full run (MNIST, loads exp1 weights)
python experiments/exp2_cluster_viz.py --use-exp1 --fast  # smoke test (subsampled MNIST)
python experiments/exp2_cluster_viz.py --fast             # fast from scratch (digits, 5 epochs)
python run_tests_exp2.py --fast                           # unit tests only
python run_tests_exp2.py                                  # unit tests + e2e smoke
```

To change settings, edit the `ExpConfig(...)` block in `main()` of that file.
Full instructions also in README.md.

## Shared Module Safety

Shared modules are: `trainer.py`, `prefix_eval.py`, `loader.py`, `encoder.py`,
`heads.py`, `mat_loss.py`, `config.py`, `utility.py`.

**Rules (enforced by convention):**

1. **Always run `python run_tests.py --fast` before AND after any edit to a shared module.**
   A broken shared module silently breaks every experiment that imports it.

2. **Optional dependencies use try/except with a plain fallback.** If a package is
   nice-to-have (e.g. `tqdm`), import it defensively so the module still works without it:
   ```python
   try:
       from tqdm import tqdm as _tqdm
   except ImportError:
       _tqdm = None
   ```

3. **No nested tqdm `position=N` arguments in terminal code.** macOS Terminal.app
   does not render multiple positioned bars correctly — inner bars become invisible and
   runs appear frozen. Use a single epoch-level bar with no `position=` argument.
   `leave=False` is fine for inner bars if you ever need them; explicit `position=` is not.

4. **Keep experiment-specific progress/display features in the experiment file**, not in
   shared modules. (e.g. the seed-loop bar lives in `exp5_seed_stability.py`.)

## Known Issues / Gotchas
- **PyTorch + NumPy 2.x**: previously required `numpy<2`, but the current env
  (mrl_env) runs numpy 2.2.6 + torch 2.2.2 without issues. The `numpy<2` pin
  has been removed from `mrl_env.yml`. Code still uses `.tolist()` instead of
  `.numpy()` as a safe bridge — this is harmless and can stay.
- **umap-learn must be installed via conda-forge**: `pip install umap-learn`
  fails on macOS because `llvmlite` (a dependency) requires building from C++
  source. Use `conda-forge::umap-learn` as in `mrl_env.yml`.

## Conda Environment
- Current env: `mrl_env` -> `env/mrl_env.yml`
- Create: `conda env create -f env/mrl_env.yml`
- Activate: `conda activate mrl_env`

## Dependencies
- Existing: numpy, pandas, seaborn, matplotlib, sklearn, scipy
- Experiment 1 adds: pytorch, torchvision
