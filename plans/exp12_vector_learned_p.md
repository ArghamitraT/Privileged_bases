# Plan: Experiment 12 — VectorLearnedPrefixLp (per-dim learned p)

**Goal**: Extend exp11's scalar learned p to a vector — each embedding dimension
has its own `p_j` learned independently via gradient descent.

Compare three loss families:
- `MRL` — Matryoshka loss
- `ScalarLearnedPrefixLp` — one shared p for all dims (reuses exp11's class)
- `VectorLearnedPrefixLp` — one p per dim (new class)

---

## Motivation

Exp11 showed that a single scalar p can be learned. But different dimensions
may have different "optimal" ordering pressure — dim 0 (highest penalty weight)
might benefit from a different exponent than dim 7 (lowest weight).

Key questions:
- Do dims converge to different p values, or all to the same?
- Do high-penalty dims (large `dim_weight`) learn higher or lower p?
- Does per-dim p produce better prefix ordering than scalar p?
- Do the three models agree on which dims are most important?

---

## Part 1: New Loss Class — `VectorLearnedPrefixLpLoss` (in `losses/mat_loss.py`)

```
p_raw : nn.Parameter  shape (embed_dim,)   ← one per dim
p     = 1 + softplus(p_raw).clamp(max=P_MAX)   shape (embed_dim,)

forward:
  (|z| + 1e-8).pow(p)  →  (batch, embed_dim) ^ (embed_dim,)  ← broadcasts
  dim_weights * (...)  →  element-wise as before
```

All dims initialised to the same `p_init=0.0` → eff p ≈ 1.69.
Registered in `build_loss("prefix_lp_vector_learned")`.

---

## Part 2: New Script — `experiments/exp12_vector_learned_p.py`

### CONFIG block

```python
DATASET       = "mnist"
EMBED_DIM     = 8
HIDDEN_DIM    = 256
HEAD_MODE     = "shared_head"
EVAL_PREFIXES = list(range(1, EMBED_DIM + 1))
EPOCHS        = 10
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
L1_LAMBDA     = 0.05
P_INIT        = 0.0     # raw init for all dims → eff p ≈ 1.69
P_MAX         = 10.0    # clamp; p ∈ (1, 11]
MAX_1NN_DB    = 10_000
MAX_PROBE     = 10_000  # cap for per-dim logistic probe (exp8)
```

### Single `train_learned_p()` for both scalar and vector

Detects shape at runtime:
```python
p_np = loss_fn.p.detach().cpu().numpy()   # () for scalar, (embed_dim,) for vector
p_trajectory.append(p_np.copy())
```
Returns `history`, `p_trajectory`, `val_accs`.

### Optimizer (same pattern as exp11, both models)

```python
opt = Adam(
    list(encoder.parameters()) + list(head.parameters()) + list(loss_fn.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY,
)
```

---

## Part 3: Outputs

### Standard (same as exp11)
- `training_curves_{stamp}.png`
- `linear_accuracy_curve_{stamp}.png`
- `1nn_accuracy_curve_{stamp}.png`
- `combined_comparison_{stamp}.png`
- `experiment_description.log`, `results_summary.txt`, `runtime.txt`, `code_snapshot/`

### p trajectory plots
- `scalar_p_trajectory_{stamp}.png` — single line, reference lines at p=1,2,3
- `scalar_p_and_val_acc_{stamp}.png` — dual-axis
- `vector_p_trajectory_{stamp}.png` — embed_dim lines, one per dim, tab10/viridis coloured
- `vector_p_and_val_acc_{stamp}.png` — mean p ± min/max band (left), val acc (right)

### Importance analysis (imported from exp8)
- `importance_scores_{stamp}.png` — 3 methods × 3 models bar charts
- `method_agreement_{stamp}.png` — Spearman rho scatter per model

### `results_summary.txt` sections
- `SCALAR LEARNED P SUMMARY` — p_init, p_final, trajectory
- `VECTOR LEARNED P SUMMARY` — p_final per dim, mean/min/max
- `METHOD AGREEMENT` — Spearman rho per model × method pair
- Accuracy table (k × model × linear × 1-NN)

---

## Part 4: Exp8 integration

Import directly (no subprocess):
```python
from experiments.exp8_dim_importance import (
    compute_importance_scores, compute_method_agreement,
    plot_importance_scores, plot_method_agreement,
)
```

Monkey-patch `exp8.MODEL_COLORS` temporarily to add exp12 model colours, then restore.

---

## Files Created / Modified

| File | Change |
|------|--------|
| `losses/mat_loss.py` | Add `VectorLearnedPrefixLpLoss` + `"prefix_lp_vector_learned"` in `build_loss()` |
| `experiments/exp12_vector_learned_p.py` | New script |
| `tests/run_tests_exp12.py` | New test runner (4 unit + e2e) |
| `tests/helper_exp12_loss.py` | Subprocess helper for torch-dependent vector loss test |
| `CLAUDE.md` | Add exp12 row + quick-start commands |
| `EXPERIMENTS.md` | Add exp12 section |

## Implementation Order

1. `VectorLearnedPrefixLpLoss` in `mat_loss.py` + verify forward/backward
2. `exp12_vector_learned_p.py`
3. `tests/helper_exp12_loss.py` + `tests/run_tests_exp12.py`
4. `CLAUDE.md` + `EXPERIMENTS.md`
