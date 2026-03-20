# Experiment 7: MRL vs Fixed-Feature (FF) vs L1-Regularized Models

## Context

Replicating the core comparison from the Matryoshka Representation Learning (MRL) paper,
extended with an L1-regularization baseline.

**Paper claims:**
- MRL accuracy ≥ FF model accuracy at every dimension k (linear classification)
- MRL up to 2% better than FF at lower dimensions on 1-NN accuracy
- 1-NN is an excellent no-cost proxy for downstream utility

**L1 regularization as ablation:**
L1 regularization on the embedding (CE + λ·||z||₁) promotes sparsity — some dimensions go to
zero — but imposes **no ordering**. A sparse dimension could be anywhere in the vector.
MRL explicitly forces the first k dimensions to carry the most discriminative information.

Key question: Is it the *sparsity* of MRL that helps at small k, or the *ordering*?
Expected result: L1 produces poor prefix performance (sparse but unordered), MRL stays high →
proves the ordering property is essential, not just sparsity.

---

## Models Compared (5 total)

| Model | Loss | embed_dim | What it tests |
|---|---|---|---|
| **Standard** | CE on full embedding | 64 | baseline — no structure |
| **L1** | CE + λ·‖z‖₁ on full embedding | 64 | sparsity without ordering |
| **MRL (Mat)** | CE summed at every prefix k | 64 | sparsity + ordering |
| **FF-k** | CE on fixed k-dim embedding | k (one per k) | dedicated model per size |
| **PCA** | none (analytical) | 64 | variance-based ordering |

The L1 and Standard models use the same shared head and same `embed_dim=64`.
FF-k models are trained independently at `embed_dim=k`.

---

## Evaluation Metrics (at each k)

1. **Linear classification accuracy** — logistic regression probe on k-dim prefix/embedding
2. **1-NN accuracy** — 1-nearest-neighbor on k-dim embedding
   - train set = database, test set = queries
   - `KNeighborsClassifier(n_neighbors=1, metric='euclidean')`
   - Embeddings are L2-normalised → L2 distance ≡ cosine distance ranking
   - Subsample database to `max_1nn_db` points in fast mode (default 2000)

---

## L1 Loss Implementation

Add `L1RegLoss` to **`losses/mat_loss.py`** (shared module, reused by exp7):

```python
class L1RegLoss(nn.Module):
    """
    Cross-entropy on the full embedding + L1 regularization on embedding activations.
    Promotes sparse embeddings (many dimensions near zero) but does NOT enforce any
    ordering. Used as an ablation against MRL to isolate the ordering property.

    loss = CE(head(z), y) + lambda_l1 * mean(|z|)
    """
    def __init__(self, lambda_l1: float = 0.05):
        ...
    def forward(self, embedding, labels, head) -> Tensor:
        ce_loss = CE(head(embedding), labels)
        l1_reg  = lambda_l1 * embedding.abs().mean()
        return ce_loss + l1_reg
```

Add `"l1"` case to `build_loss(cfg, model_type)` factory.
Add `l1_lambda: float = 0.05` field to `ExpConfig` for configurability.

---

## File Structure

```
losses/mat_loss.py                 ← add L1RegLoss + update build_loss + update ExpConfig
config.py                          ← add l1_lambda field
experiments/exp7_mrl_vs_ff.py     ← main script
run_tests_exp7.py                  ← test runner
```

---

## Implementation Plan

### Changes to shared modules

**`config.py`** — add one field:
```python
l1_lambda: float = 0.05   # L1 regularization strength for the L1 model
```

**`losses/mat_loss.py`** — add `L1RegLoss` class and update `build_loss`:
```python
def build_loss(cfg, model_type):
    # existing: "standard", "matryoshka"
    # new:      "l1"  → L1RegLoss(lambda_l1=cfg.l1_lambda)
```

### `experiments/exp7_mrl_vs_ff.py`

**Reused from existing code (no changes needed):**
- `config.ExpConfig`
- `data.loader.load_data`
- `models.encoder.MLPEncoder`
- `models.heads.build_head`
- `losses.mat_loss.build_loss` (after adding "l1" type)
- `training.trainer.train`
- `evaluation.prefix_eval.evaluate_prefix_sweep` (for Standard, L1, MRL prefix eval)
- `evaluation.prefix_eval.evaluate_pca_baseline`
- `utility.create_run_dir / save_runtime / save_code_snapshot`

**New functions in exp7:**

```python
def train_ff_models(cfg_base, eval_prefixes, data, run_dir):
    """Train one FF model per k, save weights, return dict {k: (encoder, head)}."""

def evaluate_1nn(encoder, data, cfg, prefix_k=None, max_db_samples=None):
    """
    Extract k-dim prefix embeddings from train (database) and test (queries).
    Fit KNeighborsClassifier(n_neighbors=1).
    If prefix_k is None: use full embed_dim (for FF models).
    Returns: {k: accuracy} dict.
    """

def evaluate_ff_linear(ff_models, data, eval_prefixes):
    """For each k, evaluate FF-k model's linear accuracy (full k-dim embedding)."""

def evaluate_ff_1nn(ff_models, data, eval_prefixes, max_db_samples=None):
    """For each k, compute 1-NN accuracy of FF-k model."""
```

**`main()` flow:**
1. Parse args (`--fast`, `--use-exp1`)
2. Config + seeds + run_dir + experiment_description.log
3. Load data
4. Train Standard model (`model_tag="standard"`)
5. Train L1 model (`model_tag="l1"`)
6. Train MRL model (or load from `--use-exp1`)
7. Train FF-k models for each k in eval_prefixes
8. **Linear accuracy**: prefix sweep for Standard / L1 / MRL + FF-k direct eval + PCA
9. **1-NN accuracy**: prefix sweep for Standard / L1 / MRL + FF-k + PCA
10. `training_curves.png` — loss vs epoch for Standard, L1, MRL (FF models in separate subplot)
11. `linear_accuracy_curve.png` — 5 lines (Standard, L1, MRL, FF, PCA)
12. `1nn_accuracy_curve.png` — same 5 lines
13. `combined_comparison.png` — 2-panel: linear top, 1-NN bottom
14. `results_summary.txt`
15. `save_runtime`, `save_code_snapshot`

**`--fast` mode:**
- dataset="digits", embed_dim=16, eval_prefixes=[1,2,4,8,16], epochs=5, patience=3
- max_1nn_db=500

**Full mode:**
- dataset="mnist", embed_dim=64, eval_prefixes=[1,2,4,8,16,32,64]
- max_1nn_db=None (full training set)

---

## Expected Output Structure

```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── standard_encoder_best.pt / standard_head_best.pt
├── l1_encoder_best.pt       / l1_head_best.pt
├── mat_encoder_best.pt      / mat_head_best.pt
├── ff_k1_encoder_best.pt    / ff_k1_head_best.pt
├── ...
├── ff_k64_encoder_best.pt   / ff_k64_head_best.pt
├── standard_train.log  /  l1_train.log  /  mat_train.log
├── ff_k1_train.log  ...  ff_k64_train.log
├── training_curves.png         ← loss vs epoch (mandatory)
├── linear_accuracy_curve.png   ← linear acc: 5 lines vs k
├── 1nn_accuracy_curve.png      ← 1-NN acc:   5 lines vs k
├── combined_comparison.png     ← both metrics (2-panel)
├── results_summary.txt
├── runtime.txt
└── code_snapshot/
```

---

## Expected Results

| k | Standard | L1 | MRL | FF-k |
|---|---|---|---|---|
| 1 | low | low (sparse but unordered) | **high** | medium |
| 4 | low | low-medium | **high** | medium |
| 64| high | high | high | high |

- **MRL ≥ FF** at every k (replicating paper claim)
- **L1 << MRL** at small k → proves ordering matters, not just sparsity
- **L1 ≈ Standard** at small k (same story as exp1 but with sparsity)

---

## Test File: `run_tests_exp7.py`

Unit tests:
- `test_l1_reg_loss` — verify L1RegLoss computes finite scalar, gradient flows
- `test_train_ff_models` — trains FF models for k=1,4 on digits, checks weights saved
- `test_evaluate_1nn` — runs 1-NN on synthetic embeddings, checks shape/type
- `test_evaluate_1nn_subsample` — checks subsampling logic works

Smoke test (skipped with `--fast` flag):
- End-to-end: `python experiments/exp7_mrl_vs_ff.py --fast`
- Verify all expected output files exist in run folder

---

## Run Commands

```bash
python experiments/exp7_mrl_vs_ff.py --fast        # smoke test (digits, 5 epochs)
python experiments/exp7_mrl_vs_ff.py               # full run  (MNIST, all models)
python experiments/exp7_mrl_vs_ff.py --use-exp1    # load MRL from exp1, train rest
python run_tests_exp7.py --fast                    # unit tests only
python run_tests_exp7.py                           # unit tests + smoke
```
