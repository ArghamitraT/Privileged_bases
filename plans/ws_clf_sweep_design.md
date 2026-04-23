# Weight Symmetry — Classification Experiment Design

---

## Part 0: Two-Evaluation Comparison — exp14 in code/experiments/

A new standalone script `code/experiments/exp14_two_eval_compare.py`.
Training is **identical to exp9/10** (SharedClassifier, Dense MRL + PrefixL1, same
encoder, same hyperparameters). No changes to exp9/exp10.

The experiment adds a second evaluation mode on the same trained models, producing
two sets of results for comparison.

---

### 0.1 Training — unchanged from exp9/10

Models trained:

| Tag | Head | Loss |
|---|---|---|
| `dense_mrl` | SharedClassifier | $\frac{1}{d}\sum_{k=1}^{d}\mathrm{CE}(z_{1:k}W_{:,1:k}^\top + b,\,y)$ |
| `prefix_l1` | plain Linear | $\mathrm{CE}(Wz,\,y) + \lambda\sum_j(d-j)\overline{\|z_j\|}$ |

Encoder: `input → Linear → BN → ReLU → Drop → Linear → BN → ReLU → Drop → Linear → L2-norm`
(exact exp9/10 architecture, identical for both models).

PrefixL1 dims are **reversed** before all evaluation: `Z = Z[:, ::-1]`.

Config (exact exp9):
```
DATASET=mnist, EMBED_DIM=64, HIDDEN_DIM=256, EPOCHS=20, PATIENCE=5,
LR=1e-3, BATCH_SIZE=128, WEIGHT_DECAY=1e-4, L1_LAMBDA=0.05, SEED=42
```

---

### 0.2 Evaluation 1 — exactly exp9/10, nothing changed

At every prefix $k = 1, \ldots, d$:

- **Linear accuracy**: fresh logistic regression (SAGA) fitted on $\leq 10\,000$
  randomly subsampled $z_\text{train}[:, :k]$, evaluated on $z_\text{test}[:, :k]$
- **1-NN accuracy**: fixed random subsample of $\leq 10\,000$ training embeddings as
  database (drawn once, reused across all $k$); every test point compared against
  all database points using $z[:, :k]$

Per-dimension importance (on test set):
- **mean\_abs**$[j]$: $\text{mean}(|z_\text{test}[:, j]|)$
- **variance**$[j]$: $\text{var}(z_\text{test}[:, j])$
- **probe\_acc**$[j]$: fresh 1D logistic regression fitted on $\leq 2\,000$ subsampled
  $z_\text{train}[:, j]$, evaluated on $z_\text{test}[:, j]$

---

### 0.3 Evaluation 2 — no new fitting (except probe\_acc)

At every prefix $k = 1, \ldots, d$:

- **Linear accuracy**: use the **trained $W$ directly** — compute
  $z_\text{test}[:, :k]\,W_{:, :k}^\top + b$, take argmax, compute accuracy.
  No new LR is fitted.
- **1-NN accuracy**: use the **full $z_\text{train}$** as database (all $\sim 56\,000$
  training points). Every test point compared against all of them. No fitting.

Per-dimension importance (same as Eval 1):
- **mean\_abs**$[j]$ and **variance**$[j]$: identical to Eval 1 (no fitting needed)
- **probe\_acc**$[j]$: **same as Eval 1** — fresh 1D LR per dim $j$, required because
  $W[:, j]$ was trained jointly and cannot isolate dim $j$'s contribution

#### PrefixL1 discrepancy in Eval 2 — known limitation

PrefixL1's $W \in \mathbb{R}^{C \times d}$ was trained only on the **full** $d$-dim
embedding; it never saw prefix $k < d$ during training. In Eval 2 we apply
$z_\text{test}[:, :k]\,W_{:,:k}^\top$ directly — valid computation, but $W_{:,:k}$
was not optimized for prefix $k$. Accuracy at small $k$ will likely be lower than
the fresh-LR baseline from Eval 1. This is expected and acceptable — it is an
inherent property of PrefixL1 (it is not a prefix model). **Both eval results are
reported; the gap is interpretable.**

Dense MRL does not have this issue — its $W$ was explicitly trained at every prefix.

---

### 0.4 Outputs

```
exprmnt_{timestamp}/
├── training_curves_{stamp}.png
├── combined_comparison_eval1_{stamp}.png   ← linear acc (top) + 1-NN (bottom), Eval 1
├── combined_comparison_eval2_{stamp}.png   ← same metrics, Eval 2
├── importance_scores_{stamp}.png           ← bar charts: mean_abs, variance, probe_acc
├── method_agreement_{stamp}.png            ← Spearman ρ scatter
├── results_summary.txt
├── experiment_description.log
├── runtime.txt
└── code_snapshot/
```

**Script:** `code/experiments/exp14_two_eval_compare.py`
Self-contained. Training identical to exp9/10. After this experiment passes, the
same pattern moves to `weight_symmetry/experiments/classification/` for the 9-combo
sweep.

---

## Part 1: Smoke Test — SharedClassifier vs MRL-E

Before the full sweep, run a single comparison to understand whether the head
architecture changes the ordering story.

### 1.1 What is being compared

The encoder (everything up to L2-normalisation) is **identical** for both variants.
Only the head changes.

**Variant A — SharedClassifier (exp9/10 unchanged)**

One weight matrix $W \in \mathbb{R}^{C \times d}$ with a shared bias $b \in \mathbb{R}^C$.
At prefix $k$, the embedding is zero-masked before being passed to $W$:

$$\text{logits}_k = W \hat{z}^{(k)} + b, \qquad \hat{z}^{(k)}_j = \begin{cases} z_j & j < k \\ 0 & j \geq k \end{cases}$$

Expanding the zero-masked multiply:

$$\text{logits}_k = z_{1:k}\, W_{:,\,1:k}^{\top} + b$$

The bias $b$ is always added, regardless of prefix size $k$.

**Variant B — MRL-E (efficient MRL, new)**

Same single $W \in \mathbb{R}^{C \times d}$, but implemented via direct weight slicing
with **no bias**. Taken directly from the MRL paper (Algorithm 2, efficient flag):

```python
efficient_logit = torch.matmul(x[:, :num_feat],
                               (self.nesting_classifier_0.weight[:, :num_feat]).t())
```

$$\text{logits}_k = z_{1:k}\, W_{:,\,1:k}^{\top}$$

No bias term. The `.weight` tensor is accessed directly, bypassing `nn.Linear`'s
forward (which would add `.bias`).

### 1.2 What is actually different

Mathematically, both compute $z_{1:k}\, W_{:,\,1:k}^{\top}$. The **only difference is
the bias term**:

| | SharedClassifier (A) | MRL-E (B) |
|---|---|---|
| Formula | $z_{1:k} W_{:,1:k}^{\top} + b$ | $z_{1:k} W_{:,1:k}^{\top}$ |
| $W$ | one $W \in \mathbb{R}^{C \times d}$ | same |
| Bias | one shared $b \in \mathbb{R}^{C}$, applied at every $k$ | **none** |
| Implementation | `nn.Linear` forward on zero-masked $z$ | `torch.matmul` on sliced `.weight` |
| From | exp9/exp10 unchanged | MRL paper Algorithm 2 (efficient) |

The shared bias in Variant A is calibrated once but applied at every prefix scale —
the same $b$ is added whether $k=1$ or $k=d$. Variant B has no such term.

### 1.3 PrefixL1 — unaffected by head architecture

PrefixL1 is a **non-uniform L1 regularizer** on the embedding dimensions. It has no
prefix structure in the head — only a plain CE loss on the full embedding:

$$\mathcal{L}_\text{PL1} = \mathrm{CE}(W z,\, y) \;+\; \lambda \sum_{j=0}^{d-1} (d-j)\,\overline{|z_j|}$$

Dimension $j=0$ gets the largest weight $(d)$, so it is penalised most heavily and
carries the least information after training. Dimension $j=d-1$ gets weight $1$.

**Nothing about PrefixL1 changes** between Variant A and B. It uses a single full
classifier $W \in \mathbb{R}^{C \times d}$ applied to the complete $d$-dimensional
embedding (no prefix summing, no zero-masking, no slicing during training).

After training, **dimensions are reversed** — $z \leftarrow z[:, ::-1]$ — so that
the most informative dimension is first. Only then is the dense prefix sweep applied.
Legend label everywhere: `PrefixL1 (rev)`.

PrefixL1 is trained once and appears as the same reference line on both plots.

### 1.4 Models in the smoke test

| Tag | Head | Loss | Note |
|---|---|---|---|
| `mrl_shared` | SharedClassifier | $\frac{1}{d}\sum_{k=1}^d \mathrm{CE}(z_{1:k}W_{:,1:k}^\top + b,\, y)$ | exp9/10 unchanged |
| `mrl_e` | MRL-E | $\frac{1}{d}\sum_{k=1}^d \mathrm{CE}(z_{1:k}W_{:,1:k}^\top,\, y)$ | no bias |
| `prefix_l1` | plain Linear | $\mathrm{CE}(Wz,\, y) + \lambda\sum_j(d-j)\overline{\|z_j\|}$ | same for both variants |

### 1.5 Encoder architecture (exact exp9/10)

```
input (784 for MNIST)
  → Linear(input_dim, hidden_dim) → BatchNorm1d → ReLU → Dropout(0.1)
  → Linear(hidden_dim, hidden_dim) → BatchNorm1d → ReLU → Dropout(0.1)
  → Linear(hidden_dim, embed_dim)
  → L2-normalize (unit hypersphere)
```

Encoder is **identical** for all three models. Only the head and loss differ.

### 1.6 Smoke test config (exact exp9)

```python
DATASET      = "mnist"
EMBED_DIM    = 64
HIDDEN_DIM   = 256
EPOCHS       = 20
PATIENCE     = 5
LR           = 1e-3
BATCH_SIZE   = 128
WEIGHT_DECAY = 1e-4
L1_LAMBDA    = 0.05
SEED         = 42
EVAL_PREFIXES = list(range(1, 65))   # dense: 1..64

# --fast: digits, embed_dim=16, 5 epochs
```

### 1.7 Evaluation (same for all models)

All evaluation operates on the encoder output $z$ — the head is never used at
eval time. The computation is **identical regardless of which head was used during
training**.

**Dense prefix sweep** — at every $k = 1, \ldots, d$:

- **Linear accuracy**: fresh logistic regression (SAGA) fitted on 10k random subsample
  of $z_\text{train}[:, :k]$, evaluated on full $z_\text{test}[:, :k]$
- **1-NN accuracy**: fixed random subsample of 10k training points as database
  (drawn once, reused across all $k$); every test point compared against all 10k

**Importance scoring** — per dimension, on test set embeddings:

- **mean\_abs**$[j]$ = $\text{mean}(|z_\text{test}[:, j]|)$
- **variance**$[j]$ = $\text{var}(z_\text{test}[:, j])$
- **probe\_acc**$[j]$ = accuracy of fresh 1D logistic regression fitted on
  2k random subsample of $z_\text{train}[:, j]$, evaluated on $z_\text{test}[:, j]$

**Method agreement** — Spearman $\rho$ between each pair of the 3 importance methods.

### 1.8 Outputs

```
exprmnt_{timestamp}/
├── training_curves_{stamp}.png         ← 3 subplots (mrl_shared, mrl_e, prefix_l1)
├── combined_comparison_{stamp}.png     ← linear acc (top) + 1-NN (bottom)
│                                           mrl_shared     solid orange
│                                           mrl_e          dashed orange
│                                           prefix_l1(rev) solid crimson
├── importance_scores_{stamp}.png       ← bar charts: 3 methods × 3 models
├── method_agreement_{stamp}.png        ← Spearman ρ scatter: 3 models
├── results_summary.txt
├── experiment_description.log
├── runtime.txt
└── code_snapshot/
```

**Script:** `weight_symmetry/experiments/classification/smoke_arch_compare.py`
Self-contained — defines both head classes internally, no imports from `code/`.

### 1.9 What we learn

- If `mrl_shared` and `mrl_e` curves are near-identical → bias makes no practical
  difference; either architecture works for the sweep
- If `mrl_e` shows better ordering at small $k$ → removing the shared bias helps;
  use MRL-E for the sweep
- If they diverge in importance scores → bias affects how information is distributed
  across dimensions

---

## Part 2: Full Sweep — exp_classification

After the smoke test resolves which MRL head to use, run the full 9-combo sweep.

### 2.1 Sweep configuration

| Axis | Values |
|---|---|
| Dataset | MNIST only |
| embed\_dim | 8, 16, 32 |
| hidden\_dim | 64, 128, 256 |
| **Total combos** | **9** |

Models per combo: Dense MRL (winning head from smoke test), PrefixL1 (rev).

### 2.2 Models

**Dense MRL** — full-prefix CE loss, head TBD by smoke test result.
Every prefix $m = 1, \ldots, d$ contributes equally to the loss.

**PrefixL1 (rev)** — plain CE on full embedding + non-uniform L1 regularizer.
Identical to smoke test Section 1.3. Dims reversed before all evaluation.

### 2.3 Training hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| LR | $10^{-3}$ |
| Epochs | 20 |
| Patience | 5 |
| Batch size | 128 |
| Weight decay | $10^{-4}$ |
| Dropout | 0.1 |
| L1\_lambda | 0.05 |
| Seed | 42 |

### 2.4 Evaluation

Same as smoke test Section 1.7, at every $k = 1, \ldots, d$ for each combo.

### 2.5 Output structure

```
exprmnt_{timestamp}/
├── experiment_description.log
├── results_summary.txt          ← 9 rows + ranking table
├── runtime.txt
├── code_snapshot/
├── mnist__8e__64h/
│   ├── dense_mrl_encoder_best.pt
│   ├── dense_mrl_head_best.pt
│   ├── prefix_l1_encoder_best.pt
│   ├── prefix_l1_head_best.pt
│   ├── training_curves_{stamp}.png
│   ├── combined_comparison_{stamp}.png
│   ├── importance_scores_{stamp}.png
│   ├── method_agreement_{stamp}.png
│   └── results_summary.txt
├── mnist__8e__128h/
...
└── mnist__32e__256h/
```

### 2.6 Aggregate ranking

After all 9 combos, root `results_summary.txt` includes a table ranked by:

$$\text{mean gap} = \frac{1}{d}\sum_{k=1}^{d}\left(\text{MRL linear acc}(k) - \text{PrefixL1 linear acc}(k)\right)$$

Top-5 combos highlighted.

### 2.7 CONFIG block

```python
DATASET       = "mnist"
EMBED_DIMS    = [8, 16, 32]
HIDDEN_DIMS   = [64, 128, 256]
EPOCHS        = 20
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
L1_LAMBDA     = 0.05
MAX_1NN_DB        = 10_000   # fixed random subsample of train as 1-NN database
MAX_PROBE_SAMPLES = 2_000    # random subsample of z_train for 1D LR probes
```

### 2.8 File structure

```
weight_symmetry/experiments/classification/
├── __init__.py
├── models.py                ← MLPEncoder, SharedClassifier, MRL-E head (self-contained)
├── losses.py                ← imports from weight_symmetry/losses/losses.py
├── trainer.py               ← training loop
├── analysis.py              ← importance scoring, method agreement, prefix sweep eval
├── smoke_arch_compare.py    ← Part 1: SharedClassifier vs MRL-E
└── exp_classification.py    ← Part 2: full 9-combo sweep
```

### 2.9 CLI

```bash
# Smoke test
python weight_symmetry/experiments/classification/smoke_arch_compare.py
python weight_symmetry/experiments/classification/smoke_arch_compare.py --fast

# Full sweep
python weight_symmetry/experiments/classification/exp_classification.py
python weight_symmetry/experiments/classification/exp_classification.py --fast
python weight_symmetry/experiments/classification/exp_classification.py \
    --embed-dim 16 --hidden-dim 128   # single combo
```
