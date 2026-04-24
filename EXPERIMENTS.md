# Experiment Reference

Detailed descriptions for all experiments in this project.
For a high-level overview, see [CLAUDE.md](CLAUDE.md).

> **Rule: when adding a new experiment**, add a full description section here in `EXPERIMENTS.md`
> AND add a one-line row to the experiment table in `CLAUDE.md`. Both files must stay in sync.

---

## Data

### Directory Convention (all clusters)

All datasets live under:
```
$HOME/Mat_embedding_hyperbole/data/<dataset_name>/
```
`$HOME` varies per cluster; the rest of the path is fixed. This means the same
relative data paths work everywhere without editing any code.

### PBMC 10k Multiome (used in Exp 11)

**Source:** 10x Genomics — PBMC from a healthy donor, granulocytes removed, 10k cells, Multiome ATAC + GEX, v2.0.0

**Download:**
```bash
bash scripts/bash/download_pbmc.sh
# saves to: $HOME/Mat_embedding_hyperbole/data/pbmc_10k_multiome/
```

**Files downloaded:**
| File | Size | Purpose |
|------|------|---------|
| `pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5` | ~192 MB | GEX + ATAC count matrix (cells × features) |
| `pbmc_granulocyte_sorted_10k_analysis.tar.gz` | ~485 MB | Cluster assignments, UMAP coords (cell type labels for supervised MRL) |

**After download, extract analysis outputs:**
```bash
tar -xzf $HOME/Mat_embedding_hyperbole/data/pbmc_10k_multiome/pbmc_granulocyte_sorted_10k_analysis.tar.gz \
    -C $HOME/Mat_embedding_hyperbole/data/pbmc_10k_multiome/
```

**Note:** Verify URLs in `scripts/bash/download_pbmc.sh` against the 10x dataset page before running on a new cluster.

---

## Environment Management (across clusters)

Conda env: `mrl_env` — defined in `env/mrl_env.yml` (source of truth).

### Scripts

| Script | When to run |
|--------|------------|
| `scripts/bash/create_env.sh` | Fresh cluster — env does not exist yet |
| `scripts/bash/sync_push.sh` | Instead of `git_push.sh` — auto-updates yml with new pip packages then pushes |
| `scripts/bash/git_pull.sh` | Instead of `git pull` — pulls then auto-updates env if yml changed |

### Workflow

**Adding a new package on any cluster:**
```bash
pip install <package>
bash scripts/bash/sync_push.sh "your commit message"
# auto-detects new package → adds to mrl_env.yml → commits → pushes
```

**On another cluster after a push:**
```bash
bash scripts/bash/git_pull.sh
# pulls → detects if mrl_env.yml changed → runs conda env update automatically
```

**Fresh cluster (env doesn't exist yet):**
```bash
git clone <repo>
conda activate base
bash scripts/bash/create_env.sh
conda activate mrl_env
```

**Rule:** never run plain `git push` or `git pull` — always use `sync_push.sh` and `git_pull.sh` so the env stays in sync across clusters.

---

## Experiment 1: Prefix Performance Curve

### Idea
Train three models producing embed_dim-dimensional embeddings:
1. **Standard model** — normal CE loss on full embedding
2. **Matryoshka (Mat) model** — CE loss summed at each prefix scale
3. **PCA baseline** — PCA on training data (natural variance-based ordering)

At test time, keep only the first k dims (zero out the rest) and measure
classification accuracy. Mat embeddings should degrade gracefully; standard
embeddings should not. PCA serves as a sanity check.

### Inputs / Outputs
- **Input**: MNIST (default), embed_dim=64, eval_prefixes=[1,2,4,8,16,32,64]
- **Output**: `prefix_curve.png` — accuracy vs prefix k for all three models

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

### Eval Metrics
- Classification accuracy at each prefix k (0..1)

### File Structure
- `config.py` — ExpConfig dataclass with all settings
- `data/loader.py` — flexible dataset loading (sklearn + torchvision), returns tensors
- `models/encoder.py` — MLP encoder: input_dim -> hidden -> embed_dim
- `models/heads.py` — SharedClassifier (mode A) and MultiHeadClassifier (mode B)
- `losses/mat_loss.py` — MatryoshkaLoss (sums CE at each prefix) + standard CE
- `training/trainer.py` — generic training loop (works for both standard and Mat)
- `evaluation/prefix_eval.py` — prefix sweep: zero out / slice, measure accuracy
- `experiments/exp1_prefix_curve.py` — main script: train all models, evaluate, plot

### How to Run
```bash
python experiments/exp1_prefix_curve.py          # full run (MNIST, embed_dim=64)
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── standard_encoder_best.pt / standard_head_best.pt
├── mat_encoder_best.pt      / mat_head_best.pt
├── results_summary.txt         # accuracy table: k vs Standard / Mat / PCA
├── prefix_curve.png            # main result plot (accuracy vs prefix k)
├── training_curves.png         # train + val loss per epoch for both models
├── runtime.txt
└── code_snapshot/
```

**Expected result**: Mat line stays high at small k; Standard drops off; PCA somewhere in between.

---

## Experiment 2: Cluster Visualization & Geometry-Performance Analysis

### Idea
For each prefix dimension k, project the k-dimensional prefix embeddings z_{1:k}
from Standard, Matryoshka, and PCA models to 2D (via t-SNE and UMAP) and inspect
how class cluster structure evolves as k grows. Quantify with silhouette score,
intra-class distance, inter-class centroid distance, and separation ratio. Link
cluster geometry to classification accuracy at the same k.

### Inputs / Outputs
- **Input**: Saved exp1 weights (via `--use-exp1`) or trains from scratch
- **Output**: t-SNE/UMAP grids, cluster metric plots, interactive HTML visualizations

### Key Design Decisions
- **Reuses saved exp1 weights** via `--use-exp1` flag (no retraining needed).
- **Same subsample indices** across all models for scatter plots — fair visual comparison.
- **Metrics computed on full test set** (subsampled to 2000 for speed); viz uses 3000.
- **t-SNE** always generated; **UMAP** generated only if `umap-learn` is installed.
- **k=1 edge case**: rendered as strip plot (x = embedding value, y = jitter).
- **k=2 edge case**: plotted directly, no dim reduction.

### Eval Metrics
- Silhouette score, intra-class distance, inter-class centroid distance, separation ratio
- All metrics computed at each prefix k and linked to classification accuracy

### File Structure
- `experiments/exp2_cluster_viz.py` — main script
- `tests/run_tests_exp2.py` — test runner

### How to Run
```bash
python experiments/exp2_cluster_viz.py --use-exp1         # load exp1 weights, full MNIST
python experiments/exp2_cluster_viz.py --use-exp1 --fast  # load exp1 weights, subsampled
python experiments/exp2_cluster_viz.py --fast             # train from scratch on digits
python tests/run_tests_exp2.py --fast                     # unit tests only
python tests/run_tests_exp2.py                            # unit tests + e2e smoke
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── tsne_grid.png          # rows=k in [1,2,4,8], cols=[Standard, Matryoshka, PCA]
├── umap_grid.png          # same structure with UMAP (if umap-learn installed)
├── cluster_metrics.png    # silhouette, intra distance, separation ratio vs k
├── combined_summary.png   # accuracy (top) + silhouette (bottom) vs k
├── dim4_animation.gif     # animated GIF: 6 frames, all C(4,2) pairwise 2D scatters
│                          #   NOTE: use dim4_slideshow.html for reliable frame pacing
├── dim4_interactive.html  # interactive HTML: 3D scatter subplots, rotatable
├── dim4_slideshow.html    # button-controlled HTML slideshow, fully self-contained
├── results_summary.txt    # full table: k × model × accuracy + all cluster metrics
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 5: Seed Stability

### Idea
Train Standard and Mat models multiple times with different random seeds (fixed
data split, varying model init). Measure whether the learned coordinate ordering
is reproducible.

### Inputs / Outputs
- **Input**: MNIST or digits (configurable), 5 model seeds [100..500], fixed data_seed=42
- **Output**: prefix stability plots, cross-seed correlation matrices

### Two stability metrics
1. **Prefix variance** — mean ± std of accuracy at each prefix k across seeds.
   Mat should have lower std (stable ordering).
2. **Cross-seed dimension correlation** — for each pair of seed runs, compute the
   pairwise Pearson correlation matrix between embedding dimensions on the same
   test set. If dimension d encodes the same information across runs, the diagonal
   of this matrix will be large. Mat should have higher diagonal correlation.

### Key Design Decisions
- **Fixed data split** (`data_seed=42`): only model init + training randomness varies.
  This isolates "does the ordering change?" from "does the data change?"
- **5 model seeds** by default: `[100, 200, 300, 400, 500]`
- PCA baseline computed once (deterministic given fixed data).

### Eval Metrics
- Mean ± std accuracy across seeds at each prefix k
- Mean diagonal Pearson correlation of cross-seed embedding matrices
- CKA (Centered Kernel Alignment) — rotation-invariant similarity

### File Structure
- `experiments/exp5_seed_stability.py` — main script

### How to Run
```bash
python experiments/exp5_seed_stability.py                    # full run (5 seeds, MNIST)
python experiments/exp5_seed_stability.py --fast            # smoke test (2 seeds, digits)
python experiments/exp5_seed_stability.py --low-dim         # embed_dim=10, digits
python experiments/exp5_seed_stability.py --fast --low-dim  # low-dim smoke test
python tests/run_tests_exp5.py                              # unit tests + e2e smoke
python tests/run_tests_exp5.py --fast                       # infra + unit tests only
```

**Note:** Set `dataset="digits"` in `config.py` for best results with `--low-dim`
(10 classes match embed_dim=10).

### Expected Output
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
├── correlation_heatmaps.png      # avg |corr| matrices
├── correlation_summary.txt       # mean diagonal correlation + per-pair breakdown
├── training_curves.png           # loss-vs-epoch for every seed (MANDATORY)
├── runtime.txt
└── code_snapshot/
```

### Additional plots when `embed_dim <= 16` (low-dim mode)
1. **`cosine_similarity.png`** — per-dim mean cosine similarity across seed pairs
2. **`spearman_correlation.png`** — per-dim Spearman rank correlation
3. **`per_dim_correlation.png`** — per-dim Pearson (diagonal of cross-seed matrix)
4. **`cka_summary.png` + `cka_summary.txt`** — Linear CKA.
   Key diagnostic: high CKA + low per-dim corr = info present but scrambled
   (no privileged basis). High CKA + high per-dim corr = stable privileged basis.
5. **`tsne_dim_coloring_standard.png` + `tsne_dim_coloring_matryoshka.png`** —
   t-SNE grid (rows=seeds, cols=dims), colored by dimension value.

---

## Experiment 6: Orthogonal Matryoshka Autoencoder ≈ PCA

### Idea
A linear autoencoder with orthogonal encoder columns + Matryoshka reconstruction
loss should recover PCA eigenvectors **and their ordering** exactly.

PCA = orthogonal directions ordered by variance explained. That is exactly what
Ortho + Mat + Reconstruction gives:
- **Ortho constraint** → forces independent directions (like eigenvectors)
- **Mat loss** → forces ordering by importance (like eigenvalue order)
- **Reconstruction** → finds directions that preserve maximum variance (like PCA)

### Inputs / Outputs
- **Input**: digits dataset (default), embed_dim=10
- **Output**: column alignment scores, reconstruction curves, subspace angle plots

### 2×2 Ablation Design
| | Standard Recon Loss | Matryoshka Recon Loss |
|---|---|---|
| **No ortho** | Vanilla AE | Mat AE |
| **Ortho** | Ortho AE | **Ortho Mat AE ← should ≈ PCA** |

Key prediction: Only Ortho + Mat recovers both eigenvectors AND ordering.

### Eval Metrics
- Column alignment: `|cos(W[i], PC[i])|` for each dimension i
- Reconstruction MSE at each prefix k
- Principal angle between learned subspace and PCA subspace

### File Structure
- `models/linear_ae.py` — LinearAutoencoder (tied weights, `orthogonalize()` via QR)
- `experiments/exp6_ortho_mat_ae.py` — main script with own training loop
- `tests/run_tests_exp6.py` — test runner

### How to Run
```bash
python experiments/exp6_ortho_mat_ae.py           # full run (digits, embed_dim=10)
python experiments/exp6_ortho_mat_ae.py --fast    # smoke test (5 epochs)
python tests/run_tests_exp6.py --fast             # unit tests only
python tests/run_tests_exp6.py                    # unit tests + e2e smoke
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

Key hypothesis: L1 ≈ Standard at small k (sparsity alone doesn't help prefix eval).
MRL beats L1 at small k → proves ORDERING is essential, not just sparsity.

### Inputs / Outputs
- **Input**: MNIST (full run) or digits (fast), embed_dim=64
- **Output**: linear accuracy curves, 1-NN accuracy curves, combined comparison plot

### Shared Module Changes
- `config.py` — added `l1_lambda: float = 0.05` field
- `losses/mat_loss.py` — added `L1RegLoss` class and `"l1"` case in `build_loss`

### Eval Metrics
- **Linear accuracy** — logistic regression probe on k-dim prefix
- **1-NN accuracy** — 1-nearest-neighbor (train set = database, test set = queries)

### File Structure
- `experiments/exp7_mrl_vs_ff.py` — main script
- `tests/run_tests_exp7.py` — test runner

### How to Run
```bash
python experiments/exp7_mrl_vs_ff.py --fast        # smoke test (digits, 5 epochs)
python experiments/exp7_mrl_vs_ff.py               # full run (MNIST, 20 epochs)
python experiments/exp7_mrl_vs_ff.py --use-exp1    # load MRL from exp1, train rest
python tests/run_tests_exp7.py --fast              # unit tests only
python tests/run_tests_exp7.py                     # unit tests + e2e smoke
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── standard_encoder_best.pt / standard_head_best.pt
├── l1_encoder_best.pt       / l1_head_best.pt
├── mat_encoder_best.pt      / mat_head_best.pt
├── ff_k1_encoder_best.pt ... ff_k64_encoder_best.pt
├── training_curves.png         # loss vs epoch for Standard / L1 / MRL (MANDATORY)
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

### Inputs / Outputs
- **Input**: Trains Standard, L1, MRL models (or loads from exp7 via `--use-exp7`)
- **Output**: importance bar charts, best-k vs first-k curves, method agreement scatter plots

### Three Analyses
1. **Importance scoring (3 methods):**
   - `mean_abs[d]`  = mean(|z[:, d]|) — activation magnitude
   - `variance[d]`  = var(z[:, d])    — spread of activations
   - `probe_acc[d]` = logistic regression accuracy using only dim d
2. **Best-k vs First-k:** rank dims by each method → compare oracle-selected-k
   accuracy against prefix-k. Gap ≈ 0 for MRL (ordering enforced); Gap > 0 for L1/Standard.
3. **Method agreement:** Spearman rank correlation between the 3 methods per model.

### Key Reuse
- `train_single_model` and `get_embeddings_np` imported from `exp7_mrl_vs_ff.py` (no copy).
- No shared modules are modified.

### Eval Metrics
- Gap = best_k_accuracy − first_k_accuracy (lower is better for MRL)
- Spearman rank correlation between importance scoring methods

### File Structure
- `experiments/exp8_dim_importance.py` — main script
- `tests/run_tests_exp8.py` — test runner (6 unit tests + e2e smoke)

### How to Run
```bash
python experiments/exp8_dim_importance.py --fast                    # smoke test (digits, 5 epochs)
python experiments/exp8_dim_importance.py                           # full run (MNIST, 20 epochs)
python experiments/exp8_dim_importance.py --use-weights PATH        # load weights from exp7 or exp10 dir
python experiments/exp8_dim_importance.py --embed-dim 8 --use-weights PATH   # at dim=8
python experiments/exp8_dim_importance.py --embed-dim 16 --use-weights PATH  # at dim=16
python experiments/exp8_dim_importance.py --embed-dim 32 --use-weights PATH  # at dim=32
python tests/run_tests_exp8.py --fast                               # unit tests only
python tests/run_tests_exp8.py                                      # unit tests + e2e smoke
```

> **Note**: `--use-weights` accepts output directories from **both Exp7 and Exp10** — weight
> filenames are identical (`standard_encoder_best.pt`, etc.). The flag was formerly `--use-exp7`.

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

## Experiment 9: Dense Prefix Evaluation + Multi-Seed Best-k vs First-k

### Idea
Train Standard, L1, MRL, and PCA models and evaluate at every dimension
k = 1, 2, ..., embed_dim (dense prefix sweep, not just powers of 2).
Run over two data seeds [42, 123] to confirm robustness across splits.

Key question: does MRL's first-k accuracy closely match oracle best-k at all k,
while Standard and L1 show a persistent gap? Dense eval makes this visible
continuously instead of at just 7 checkpoints.

### Inputs / Outputs
- **Input**: MNIST (full) or digits (fast), embed_dim=64, data_seeds=[42, 123]
- **Output**: dense prefix accuracy curves, per-seed best-k vs first-k plots, gap comparison

### Key Design Decisions
- No FF models — focus is on ordering, not capacity matching.
- Dense eval_prefixes = list(range(1, embed_dim+1)) for smooth curves.
- data_seeds = [42, 123] hardcoded (42 = canonical, 123 = fresh split).
- LR fits for best-k subsampled to max 10k train samples for speed.
- Imports `train_single_model`, `get_embeddings_np` from exp7 (no copy).
- Imports all analysis functions from exp8 (no copy).

### Eval Metrics
- Accuracy at every k from 1 to embed_dim (dense sweep)
- Gap = best_k − first_k per model per seed

### File Structure
- `experiments/exp9_dense_prefix.py` — main script
- `tests/run_tests_exp9.py` — test runner (5 unit tests + e2e smoke)

### How to Run
```bash
python experiments/exp9_dense_prefix.py --fast   # smoke test (digits, 5 epochs, 1 seed)
python experiments/exp9_dense_prefix.py          # full run (MNIST, 20 epochs, 2 seeds)
python tests/run_tests_exp9.py --fast            # unit tests only
python tests/run_tests_exp9.py                   # unit tests + e2e smoke
```

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── training_curves.png          # loss vs epoch for all models x both seeds (MANDATORY)
├── seed_42/
│   ├── standard_encoder_best.pt / standard_head_best.pt
│   ├── l1_encoder_best.pt       / l1_head_best.pt
│   ├── mat_encoder_best.pt      / mat_head_best.pt
│   ├── prefix_accuracy.png      # accuracy vs k (1..64) for all 4 models
│   ├── best_vs_first_k.png      # first-k vs best-k per model (3 importance methods)
│   └── method_agreement.png     # Spearman rho scatter plots
├── seed_123/ (same structure)
├── gap_comparison.png           # gap(best_k - first_k) for both seeds per model
├── results_summary.txt
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 10: Dense Prefix Sweep — MRL vs Standard vs L1 vs PCA (No FF)

### Idea
Exp7 without FF models, evaluated at every dimension k=1..embed_dim (dense sweep).
Supports multiple embedding sizes via `--embed-dim` flag (8, 16, 32, 64).

The goal is smooth, continuous accuracy curves showing exactly where each model's
performance plateaus — without the training overhead of FF models.

**vs Exp7**: no FF models; dense eval instead of sparse powers-of-2.
**vs Exp9**: adds L1 model; adds 1-NN metric; no multi-seed; no best-k analysis.

### Inputs / Outputs
- **Input**: MNIST (full) or digits (fast), embed_dim configurable via `--embed-dim`
- **Output**: dense linear accuracy + 1-NN accuracy curves for Standard, L1, MRL, PCA

### Key Design Decisions
- `eval_prefixes = list(range(1, embed_dim + 1))` — always dense, derived from embed_dim
- No FF models — focus is on ordering, not capacity matching
- 1-NN database subsampled to 10k for speed (same as exp7)
- Imports `train_single_model`, `get_embeddings_np`, `evaluate_prefix_1nn`, `evaluate_pca_1nn`
  from `exp7_mrl_vs_ff` (no copy)
- No shared module changes

### Eval Metrics
- **Linear accuracy** — logistic regression probe on k-dim prefix at every k
- **1-NN accuracy** — 1-nearest-neighbor at every k

### File Structure
- `experiments/exp10_dense_multidim.py` — main script
- `tests/run_tests_exp10.py` — test runner (5 unit tests + e2e smoke)

### How to Run
```bash
# Run exp10 standalone
python experiments/exp10_dense_multidim.py                       # full run (MNIST, embed_dim=64)
python experiments/exp10_dense_multidim.py --fast                # smoke test (digits, 5 epochs)
python experiments/exp10_dense_multidim.py --embed-dim 8         # full run, embed_dim=8
python experiments/exp10_dense_multidim.py --embed-dim 16        # full run, embed_dim=16
python experiments/exp10_dense_multidim.py --embed-dim 32        # full run, embed_dim=32
python experiments/exp10_dense_multidim.py --embed-dim 8 --fast  # smoke test at dim=8

# Run exp10 + exp8 together for all dims via wrapper (recommended)
python scripts/run_exp10_8_multidim.py              # full run, dims=[8,16,32]
python scripts/run_exp10_8_multidim.py --fast       # smoke test, all 3 dims
python scripts/run_exp10_8_multidim.py --dims 8 16  # run only specified dims

# Tests
python tests/run_tests_exp10.py --fast              # unit tests only
python tests/run_tests_exp10.py                     # unit tests + e2e smoke
```

> **Workflow**: The wrapper runs Exp10 for a given dim, then automatically feeds its output
> directory into `exp8 --use-weights` to produce the importance scoring analysis — no manual
> path copying needed.

### Expected Output
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── standard_encoder_best.pt / standard_head_best.pt
├── l1_encoder_best.pt       / l1_head_best.pt
├── mat_encoder_best.pt      / mat_head_best.pt
├── standard_train.log / l1_train.log / mat_train.log
├── training_curves.png         # loss vs epoch — Standard, L1, MRL (MANDATORY)
├── linear_accuracy_curve.png   # linear accuracy: 4 lines vs k=1..embed_dim
├── 1nn_accuracy_curve.png      # 1-NN accuracy:   4 lines vs k=1..embed_dim
├── combined_comparison.png     # 2-panel (linear top, 1-NN bottom)
├── results_summary.txt         # table: k × model × linear_acc × 1nn_acc
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 11: LearnedPrefixLp — MRL vs PrefixLp vs LearnedPrefixLp

### Idea

Exp10 showed that PrefixLp with different fixed p values produces different
ordering quality. The natural next question: can the model learn p itself?

Three loss families are compared on the dense prefix sweep:
1. **MRL** — Matryoshka loss: CE summed at every prefix scale k=1..embed_dim
2. **PrefixLp** — CE + front-loaded weighted Lp penalty, p fixed as hyperparameter
3. **LearnedPrefixLp** — same penalty as PrefixLp, but p is a scalar `nn.Parameter`
   optimised jointly with the encoder and head via gradient descent

Scientific question: does gradient descent converge to a stable p, and does the
learned p produce better prefix ordering than any fixed p?

### Key Design Decisions

- `p = 1 + softplus(p_raw).clamp(max=P_MAX)` — guarantees p > 1, finite, differentiable
- At `p_raw=0.0`, effective p ≈ 1.69 (neutral start between L1 and L2)
- `P_MAX=10.0` clamps p ∈ (1, 11] — safety guardrail, noted in code comment
- Optimizer for LearnedPrefixLp includes `loss_fn.parameters()` to give `p_raw` gradients
- `train_learned_p()` — local training loop (copy of `trainer.train()` + p tracking);
  does NOT modify the shared trainer
- Dimension reversal applied to both PrefixLp and LearnedPrefixLp before eval
  (same convention as exp10 PrefixLp)
- Standard and PCA excluded — isolates the loss family comparison

### Inputs / Outputs

- **Input**: MNIST (full) or digits (fast), embed_dim=8 (CONFIG)
- **Output**: same prefix sweep plots as exp10 for 3 models, plus two new plots

### Eval Metrics

- **Linear accuracy** — logistic regression probe on k-dim prefix at every k
- **1-NN accuracy** — 1-nearest-neighbor at every k

### New outputs (exp11-specific)

| File | Contents |
|------|----------|
| `p_trajectory_{stamp}.png` | learned p vs epoch with reference lines at p=1,2,3,P_FIXED |
| `p_and_val_acc_{stamp}.png` | dual-axis: p (left, blue) and full-embedding val accuracy (right, red) vs epoch |

`results_summary.txt` includes a `LEARNED P SUMMARY` section with `p_init`, `p_final`, and the full `p_trajectory` list.

### File Structure

- `experiments/exp11_learned_prefix_lp.py` — main script
- `losses/mat_loss.py` — `LearnedPrefixLpLoss` class + `"prefix_lp_learned"` in `build_loss()`
- `tests/run_tests_exp11.py` — test runner (4 unit tests + e2e smoke)
- `tests/helper_exp11_loss.py` — subprocess helper for torch-dependent loss test

### How to Run

```bash
python experiments/exp11_learned_prefix_lp.py        # full run (MNIST, embed_dim=8)
python experiments/exp11_learned_prefix_lp.py --fast # smoke test (digits, 3 epochs)
python tests/run_tests_exp11.py --fast               # unit tests only
python tests/run_tests_exp11.py                      # unit tests + e2e smoke
```

### Expected Output

```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── mat_encoder_best.pt      / mat_head_best.pt
├── pl5_encoder_best.pt      / pl5_head_best.pt
├── lp_learned_encoder_best.pt / lp_learned_head_best.pt
├── mat_train.log / pl5_train.log / lp_learned_train.log
├── training_curves_{stamp}.png       # loss vs epoch — all 3 models (MANDATORY)
├── p_trajectory_{stamp}.png          # learned p vs epoch with reference lines
├── p_and_val_acc_{stamp}.png         # dual-axis p + val accuracy vs epoch
├── linear_accuracy_curve_{stamp}.png # linear accuracy: 3 lines vs k=1..embed_dim
├── 1nn_accuracy_curve_{stamp}.png    # 1-NN accuracy:   3 lines vs k=1..embed_dim
├── combined_comparison_{stamp}.png   # 2-panel (linear top, 1-NN bottom)
├── results_summary.txt               # LEARNED P SUMMARY + accuracy table
├── runtime.txt
└── code_snapshot/
```

---

## Experiment 12: VectorLearnedPrefixLp — MRL vs ScalarLearnedPrefixLp vs VectorLearnedPrefixLp

### Idea

Exp11 introduced a scalar learned p. Here p becomes a vector of shape `(embed_dim,)` —
each embedding dimension has its own independently learned exponent.

Three models compared:
1. **MRL** — Matryoshka loss
2. **ScalarLearnedPrefixLp** — one shared p for all dims (from exp11)
3. **VectorLearnedPrefixLp** — one p per dim, all learned jointly

Scientific questions:
- Do different dims converge to different p values?
- Do high-penalty dims (large `dim_weight`) learn higher or lower p than low-penalty dims?
- Does per-dim p improve prefix ordering over scalar p?
- Do the three models agree on which dims are most important?

### Key Design Decisions

- `p_raw: nn.Parameter` shape `(embed_dim,)` — one unconstrained value per dim
- `p = 1 + softplus(p_raw).clamp(max=P_MAX)` — same constraint as scalar, applied per-dim
- `(|z| + 1e-8).pow(p)` broadcasts correctly: `(batch, embed_dim) ^ (embed_dim,)`
- Single `train_learned_p()` handles both scalar and vector by detecting `loss_fn.p.ndim`
- `p_trajectory` stored as list of numpy arrays (shape `()` scalar or `(embed_dim,)` vector)
- Dimension reversal applied to both Lp models (same convention as exp10/exp11)
- Importance scores + method agreement imported from exp8 — `MODEL_COLORS` patched locally

### Eval Metrics

- **Linear accuracy** — logistic regression probe on k-dim prefix at every k
- **1-NN accuracy** — 1-nearest-neighbor at every k
- **Per-dim importance** — mean |z|, variance, 1D probe accuracy (from exp8)
- **Method agreement** — Spearman rho between importance scoring methods (from exp8)

### File Structure

- `experiments/exp12_vector_learned_p.py` — main script
- `losses/mat_loss.py` — `VectorLearnedPrefixLpLoss` + `"prefix_lp_vector_learned"` in `build_loss()`
- `tests/run_tests_exp12.py` — test runner (4 unit tests + e2e smoke)
- `tests/helper_exp12_loss.py` — subprocess helper for torch-dependent vector loss test

### How to Run

```bash
python experiments/exp12_vector_learned_p.py        # full run (MNIST, embed_dim=8)
python experiments/exp12_vector_learned_p.py --fast # smoke test (digits, 3 epochs)
python tests/run_tests_exp12.py --fast              # unit tests only
python tests/run_tests_exp12.py                     # unit tests + e2e smoke
```

### Expected Output

```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── mat_encoder_best.pt / mat_head_best.pt
├── sc_learned_encoder_best.pt / sc_learned_head_best.pt
├── vec_learned_encoder_best.pt / vec_learned_head_best.pt
├── mat_train.log / sc_learned_train.log / vec_learned_train.log
├── training_curves_{stamp}.png          # loss vs epoch — all 3 models (MANDATORY)
├── scalar_p_trajectory_{stamp}.png      # single p line vs epoch
├── scalar_p_and_val_acc_{stamp}.png     # dual-axis scalar p + val acc
├── vector_p_trajectory_{stamp}.png      # embed_dim colored lines vs epoch
├── vector_p_and_val_acc_{stamp}.png     # dual-axis mean p + val acc, min/max band
├── importance_scores_{stamp}.png        # 3-method x 3-model bar charts (from exp8)
├── method_agreement_{stamp}.png         # Spearman rho scatter per model (from exp8)
├── linear_accuracy_curve_{stamp}.png
├── 1nn_accuracy_curve_{stamp}.png
├── combined_comparison_{stamp}.png
├── results_summary.txt                  # scalar p, vector p, agreement, accuracy table
├── runtime.txt
└── code_snapshot/
```

---

## Exp 13 — Supervised MRL on CD34 vs SEACells

**Script:** `experiments/exp13_mrl_cd34_supervised.py`
**Conda env:** `mrl_env`

### Idea
Train multiple loss variants on the CD34 HSPC gene expression dataset using known
cell type labels as supervision. Evaluate each model with a dense prefix sweep
(k = 1..EMBED_DIM) using k-means clustering and three metrics — cell type purity,
compactness, separation — and compare against the SEACells metacell baseline stored
in the h5ad file (`adata.obs['SEACell']`).

This is the **supervised upper-bound** experiment in the MRL-SEACells roadmap
(see `plans/mrl_seacells.md`).

### Inputs
- `$HOME/Mat_embedding_hyperbole/data/cd34_multiome/GSE200046_cd34_multiome_rna.h5ad`
  - 6,881 CD34+ HSPCs, 8 cell types (HSC, HMP, MEP, Ery, Mono, cDC, pDC, CLP)
  - Contains: `adata.X` (log-normalised), `adata.var['highly_variable']`,
    `adata.obsm['X_pca']` (30 PCs), `adata.obsm['X_umap']`,
    `adata.obs['celltype']`, `adata.obs['SEACell']` (paper's metacell IDs)

### Models trained (controlled by `MODELS_TO_RUN` in CONFIG)
| Tag | Loss class | Notes |
|---|---|---|
| `pca` | sklearn PCA | No training; first-k dims = max variance |
| `ce` | `StandardLoss` | Plain cross-entropy on full embedding |
| `mrl` | `MatryoshkaLoss` | CE at each prefix in `MRL_TRAIN_PREFIXES` |
| `fixed_lp` | `PrefixLpLoss(p=FIXED_LP_P)` | Ordered Lp; dims reversed before eval |
| `learned_lp` | `LearnedPrefixLpLoss` | Scalar p learned jointly with encoder |
| `learned_lp_vec` | `VectorLearnedPrefixLpLoss` | Per-dim p learned jointly |

### Evaluation
- **Dense prefix sweep:** k = 1..EMBED_DIM applied to all models
- **MRL training prefixes** (`MRL_TRAIN_PREFIXES`): sparse subset used in loss only
- **k-means:** `N_CLUSTERS` clusters per model per k (default 100, match SEACells)
- **Metrics per cluster set:**
  - Cell type purity: median dominant-cell-type fraction per cluster (↑ better)
  - Compactness: median mean distance to cluster centroid (↓ better)
  - Separation: median nearest-centroid distance (↑ better)
- **SEACells reference:** purity/compactness/separation from paper's assignments,
  plotted as a dashed horizontal line on each curve

### How to run
```bash
conda activate mrl_env
python experiments/exp13_mrl_cd34_supervised.py          # full run
python experiments/exp13_mrl_cd34_supervised.py --fast   # smoke test (500 cells, embed_dim=8)
python experiments/exp13_mrl_cd34_supervised.py --use-weights exprmnt_XYZ  # plots only
python tests/run_tests_exp13.py --fast                   # unit tests only
python tests/run_tests_exp13.py                          # unit tests + e2e smoke
```

### Expected outputs
```
exprmnt_{timestamp}/
├── training_curves_{stamp}.png           # loss vs epoch for all trained models
├── prefix_purity_curve_{stamp}.png       # purity vs k, all models + SEACells line
├── prefix_compactness_curve_{stamp}.png  # compactness vs k
├── prefix_separation_curve_{stamp}.png   # separation vs k
├── umap_comparison_{stamp}.png           # 3-panel: cell types | MRL clusters | SEACells
├── cd34_embeddings.npz                   # embeddings per model tag (n_cells, embed_dim)
├── {tag}_encoder.pt                      # saved weights per trained model
├── {tag}_head.pt
├── results_summary.txt                   # table: model × k × purity × compact × sep
├── experiment_description.log
├── runtime.txt
└── code_snapshot/
```

### Key CONFIG parameters
```python
EMBED_DIM          = 30     # match SEACells PCA dim
N_HVG              = 2000   # input features
MRL_TRAIN_PREFIXES = [2, 4, 8, 16, 30]
N_CLUSTERS         = 100    # k-means clusters (match SEACells n_metacells)
FIXED_LP_P         = 1      # exponent for fixed_lp (1 = PrefixL1)
MODELS_TO_RUN      = ["pca", "ce", "mrl", "fixed_lp", "learned_lp", "learned_lp_vec"]
```

---

## Exp 14 — Two-Evaluation Comparison: Dense MRL vs PrefixL1

**Script:** `experiments/exp14_two_eval_compare.py`
**Conda env:** `mrl_env`

### Idea
Train the same encoder+head pair two different ways (Dense MRL, MRL-E, and three
PrefixL1 α variants), then evaluate each at every prefix `k = 1..d` **two ways**:

| Eval | Linear accuracy | 1-NN accuracy |
|---|---|---|
| **Eval 1** (exp9/10 standard) | Fresh logistic regression refit on ≤`MAX_LR_SAMPLES` subsample of `z_train[:, :k]` | Subsampled (≤`MAX_1NN_DB_E1`) database |
| **Eval 2** (no refit) | Trained `W` used directly: `logits_k = z_test[:, :k] @ W[:, :k].T + b` | Full `z_train` as database |

The gap between Eval 1 and Eval 2 for each model quantifies how much of the
prefix-ordering property is encoded in `W` itself versus being recovered by the
fresh LR. Dense MRL is expected to show a small gap; PrefixL1 a larger one at
small `k` because `W[:, :k]` was never trained for prefix `k`.

### Datasets supported
Primary dataset is controlled by the `DATASET` constant in the CONFIG block
(default `"cd34"`). CLI `--dataset {mnist,cd34}` overrides it.

| `DATASET` | Dataset | Default embed dims | Epochs | Source |
|---|---|---|---|---|
| `"cd34"` (default) | CD34 multiome HSPCs (6,881 cells × 2k HVGs, 8 cell types) | `EMBED_DIMS = [8, 16]` | `EPOCHS = 15` | exp13's `load_cd34_data` + `make_data_split` |
| `"mnist"` | MNIST (digits with `--fast`) | `MNIST_EMBED_DIMS = [16]` | `MNIST_EPOCHS = 20` | `data/loader.py` |

**CD34 caveat:** the train split (~4.8k cells) is smaller than `MAX_1NN_DB_E1`
and `MAX_LR_SAMPLES`, so the subsampling caps don't trigger — Eval 1 and Eval 2
1-NN databases collapse to the same set. The **linear-accuracy gap** is the
meaningful signal on CD34.

### Models trained
| Tag / Legend | Loss | Notes |
|---|---|---|
| `dense_mrl` / "Dense MRL" | `MatryoshkaLoss` (all prefixes) | Skipped with `--no-dense-mrl` |
| `mrl_e` / "MRL-E" | Custom: `(1/d) Σₖ CE(z[:,:k] @ W[:,:k].T, y)` | No bias, direct weight slicing |
| `prefix_l1` / "PrefixL1 (rev)" | `PrefixLpLoss(p=1)`, dim_weights `(d-j)^1.0` | Dims reversed before eval |
| `prefix_l1_a075` / "PrefixL1 α=0.75 (rev)" | Same, dim_weights `(d-j)^0.75` | |
| `prefix_l1_a05` / "PrefixL1 α=0.5 (rev)" | Same, dim_weights `(d-j)^0.5` | |

### How to run
```bash
conda activate mrl_env
python experiments/exp14_two_eval_compare.py --fast                        # smoke test (digits, embed_dim=8)
python experiments/exp14_two_eval_compare.py                               # full MNIST run
python experiments/exp14_two_eval_compare.py --embed-dim 32                # MNIST, single dim
python experiments/exp14_two_eval_compare.py --dataset cd34 --fast         # CD34 smoke test (500 cells)
python experiments/exp14_two_eval_compare.py --dataset cd34                # CD34 full run, sweep [8, 16]
python experiments/exp14_two_eval_compare.py --dataset cd34 --embed-dim 8  # CD34, single dim
python experiments/exp14_two_eval_compare.py --dataset cd34 --no-dense-mrl # CD34 without Dense MRL
python tests/run_tests_exp14.py --fast                                     # unit tests only
```

### Expected outputs
```
exprmnt_{timestamp}/
└── embed_{d}/                                   # one subdir per embed_dim in the sweep
    ├── training_curves_{stamp}.png              # loss vs epoch for all trained models
    ├── combined_comparison_eval1_{stamp}.png    # linear + 1-NN vs k (fresh LR)
    ├── combined_comparison_eval2_{stamp}.png    # linear + 1-NN vs k (trained W)
    ├── importance_scores_{stamp}.png            # mean|z| / variance / probe_acc per dim
    ├── method_agreement_{stamp}.png             # Spearman ρ between importance methods
    ├── results_summary.txt                      # tables for both evals + gap
    ├── {tag}_encoder_best.pt, {tag}_head_best.pt
    └── {tag}_train.log
exprmnt_{timestamp}/
├── runtime.txt
└── code_snapshot/
```

### Key CONFIG parameters
```python
# Primary CONFIG — current defaults match CD34 full run
DATASET        = "cd34"                  # "cd34" or "mnist"  (CLI --dataset overrides)
EMBED_DIMS     = [8, 16]                 # sweep; --embed-dim overrides
HIDDEN_DIM     = 256
EPOCHS         = 15                      # 15 for CD34; MNIST uses MNIST_EPOCHS
L1_ALPHAS      = [0.75, 0.5]             # additional PrefixL1 dim-weight exponents
MAX_LR_SAMPLES = 10_000                  # Eval 1 LR subsample cap
MAX_1NN_DB_E1  = 10_000                  # Eval 1 1-NN database cap

# CD34 data loading (used when DATASET == "cd34")
CD34_DATA_PATH = "$HOME/Mat_embedding_hyperbole/data/cd34_multiome/GSE200046_cd34_multiome_rna.h5ad"
CD34_N_HVG     = 2000                    # HVGs as input features → input_dim=2000

# MNIST overrides (used when DATASET == "mnist")
MNIST_EMBED_DIMS = [16]
MNIST_EPOCHS     = 20
```
