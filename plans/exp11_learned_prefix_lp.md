# Plan: Experiment 11 — LearnedPrefixLp (Learned p in Prefix Lp Loss)

**Goal**: Compare three loss families on the dense prefix sweep task:
`MRL`, `PrefixLp` (fixed p as hyperparameter), and `LearnedPrefixLp`
(p is a scalar `nn.Parameter` optimized jointly with the encoder).

The key scientific question: does gradient descent converge to a stable p,
and does the learned p produce better prefix ordering than any fixed p?

---

## Motivation

Exp10 showed that `PrefixLp` with different fixed p values produces different
ordering quality. The natural next question: can the model learn p itself?

- If p converges to a consistent value → the data has a preferred exponent.
- If p converges to a value near our best hand-tuned p → validates exp10.
- If p diverges or oscillates → the loss landscape is flat in p direction.
- The trajectory of p during training reveals how ordering pressure evolves.

---

## Models Trained

| Model | Loss | Notes |
|-------|------|-------|
| MRL | `MatryoshkaLoss` | ordering via multi-scale CE |
| PrefixLp | `PrefixLpLoss(p=P_FIXED)` | fixed p, hyperparameter |
| LearnedPrefixLp | `LearnedPrefixLpLoss` | p is a learned scalar parameter |

Standard and PCA are excluded — this experiment isolates the loss family comparison.

---

## Part 1: New Loss Class — `LearnedPrefixLpLoss` (in `losses/mat_loss.py`)

### Parameterization

```
p_raw  : nn.Parameter  (scalar, unconstrained, init = P_INIT)
p      = 1.0 + softplus(p_raw).clamp(max=P_MAX)

At P_INIT=0.0: p ≈ 1.69  (between L1 and L2 — neutral start)
p ∈ (1.0, 1.0 + P_MAX]  with P_MAX=10.0  → p ∈ (1.0, 11.0]
```

The clamp is a safety guardrail — prevents p from drifting to extreme values
that cause numerical instability (noted in a comment in the class).

### Forward pass

```
ce_loss    = CE(head(z), y)
penalty    = (dim_weights * (|z| + 1e-8).pow(p)).mean(dim=0).sum()
total_loss = ce_loss + lambda_l1 * penalty
```

`1e-8` epsilon guards the gradient `d/dp (|z|+ε)^p = (|z|+ε)^p * log(|z|+ε)`
from the `0 * (-inf)` case when activations are exactly zero.

`dim_weights` is `[embed_dim, embed_dim-1, ..., 1]` — same as `PrefixLpLoss`.

### `build_loss()` registration

Add `"prefix_lp_learned"` as a new model_type key.

---

## Part 2: New Script — `experiments/exp11_learned_prefix_lp.py`

### CONFIG block

```python
DATASET         = "mnist"
EMBED_DIM       = 8
HIDDEN_DIM      = 256
HEAD_MODE       = "shared_head"
EVAL_PREFIXES   = list(range(1, EMBED_DIM + 1))
EPOCHS          = 10
PATIENCE        = 5
LR              = 1e-3
BATCH_SIZE      = 128
WEIGHT_DECAY    = 1e-4
SEED            = 42
L1_LAMBDA       = 0.05
P_FIXED         = 5          # exponent for the fixed PrefixLp baseline
P_INIT          = 0.0        # p_raw initialisation → p ≈ 1.69 at epoch 0
P_MAX           = 10.0       # p is clamped to (1, 1+P_MAX]; safety cap
MAX_1NN_DB      = 10_000
```

### Custom training loop: `train_learned_p()`

**Not** using `trainer.train()` — written locally in exp11 to avoid touching
the shared module. Structurally identical to `trainer.train()`, with two additions:

1. After each epoch: `p_trajectory.append(loss_fn.p.item())`
2. After val step: `val_accs.append(val_accuracy_this_epoch)`

Returns: `history`, `p_trajectory` (list, length = epochs_run), `val_accs`.

Used only for `LearnedPrefixLp`. MRL and PrefixLp are trained with the
standard `trainer.train()` unchanged.

### Optimizer for LearnedPrefixLp

```python
opt = Adam(
    list(encoder.parameters()) +
    list(head.parameters()) +
    list(loss_fn.parameters()),   # includes p_raw
    lr=LR, weight_decay=WEIGHT_DECAY
)
```

MRL and PrefixLp use the standard optimizer (encoder + head only).

### Dimension reversal

`LearnedPrefixLp` penalises dim 0 most heavily → flip embeddings before eval,
same as PrefixLp in exp10. Label in plots/tables: `"LearnedPrefixLp (rev)"`.

---

## Part 3: Outputs

### Standard outputs (same as exp10)
- `training_curves_{stamp}.png` — loss vs epoch for all three models
- `linear_accuracy_curve_{stamp}.png` — linear probe prefix sweep
- `1nn_accuracy_curve_{stamp}.png` — 1-NN prefix sweep
- `combined_comparison_{stamp}.png` — 2-panel linear + 1-NN
- `experiment_description.log`
- `results_summary.txt`
- `runtime.txt`
- `code_snapshot/`

### New outputs (exp11-specific)

**`p_trajectory_{stamp}.png`**
- x: epoch, y: learned p value
- Horizontal dashed reference lines at p=1, p=2, p=3, p=P_FIXED
- Single curve showing how p evolves during training

**`p_and_val_acc_{stamp}.png`** (dual-axis)
- Left y-axis (blue): learned p vs epoch
- Right y-axis (red): full-embedding val accuracy vs epoch
- Shows whether accuracy improvements correlate with p rising or settling

### `results_summary.txt` additions
```
LEARNED P SUMMARY
  p_init (epoch 0)  = X.XX
  p_final           = X.XX
  p_trajectory      = [1.69, ..., X.XX]
```

---

## Part 4: Tests — `tests/run_tests_exp11.py`

- `--fast` smoke test: dataset=digits, embed_dim=8, epochs=3
- Checks: both new plots are produced, `p_trajectory` has length == epochs_run,
  `final_p` is in the valid range (1, 1+P_MAX]

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `losses/mat_loss.py` | Add `LearnedPrefixLpLoss`; register `"prefix_lp_learned"` in `build_loss()` |
| `experiments/exp11_learned_prefix_lp.py` | New script |
| `tests/run_tests_exp11.py` | New smoke test runner |
| `CLAUDE.md` | Add exp11 row to experiment table |
| `EXPERIMENTS.md` | Add exp11 description section |

---

## Implementation Order

1. `LearnedPrefixLpLoss` in `mat_loss.py` + run `tests/run_tests.py --fast`
2. `exp11_learned_prefix_lp.py` (CONFIG, training, evaluation, plots)
3. `tests/run_tests_exp11.py`
4. Update `CLAUDE.md` and `EXPERIMENTS.md`
