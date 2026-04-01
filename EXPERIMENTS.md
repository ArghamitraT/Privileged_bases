# Experiment Reference

Detailed descriptions for all experiments in this project.
For a high-level overview, see [CLAUDE.md](CLAUDE.md).

> **Rule: when adding a new experiment**, add a full description section here in `EXPERIMENTS.md`
> AND add a one-line row to the experiment table in `CLAUDE.md`. Both files must stay in sync.

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
