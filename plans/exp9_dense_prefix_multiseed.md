# Experiment 9: Dense Prefix Evaluation + Multi-Seed Best-k vs First-k

## Goal

Extend the MRL prefix analysis from exp7/exp8 in two ways:
1. **Dense prefix sweep** — evaluate at every dimension k = 1, 2, ..., embed_dim
   instead of only powers of 2. This gives a smooth, high-resolution curve.
2. **Multiple data seeds** — vary the train/test split across two seeds to confirm
   the findings are not artifacts of a particular data split.

The key question: **does MRL's first-k accuracy closely match the oracle best-k
accuracy, while Standard and L1 show a large gap?** With dense eval this gap is
visible at every single dimension, not just at a few checkpoints.

---

## Models

| Model      | Loss                          | Notes                        |
|------------|-------------------------------|------------------------------|
| Standard   | CE on full embed_dim          | No ordering pressure         |
| L1         | CE + λ·‖z‖₁                  | Sparsity, no ordering        |
| MRL        | CE summed at every prefix k   | Ordering enforced            |
| PCA        | Analytical (no training)      | Variance-ordered baseline    |

No FF models — the experiment focuses on the ordering question, not capacity matching.

---

## Data Seeds

Two seeds: `[42, 123]`

- Seed 42 is the canonical seed used in all prior experiments (allows direct
  comparison with exp7/exp8 results).
- Seed 123 is a fresh split to test robustness.
- For each seed: the full dataset is re-split into train/val/test. Standard, L1,
  and MRL are trained from scratch on that split. PCA is fitted on the same
  train set.

---

## Config

- Dataset: MNIST
- embed_dim: 64
- hidden_dim: 256
- eval_prefixes: list(range(1, 65))  — all 64 dims (dense)
- head_mode: shared_head
- epochs: 20, patience: 5
- lr: 1e-3, batch_size: 256
- l1_lambda: 0.05
- data_seeds: [42, 123]

---

## Analyses (both inherited from exp8, run per seed then aggregated)

### 1. Prefix Accuracy Curves
- For each seed: plot accuracy vs k (1..64) for all 4 models.
- Dense x-axis makes the shape of the curve clear (not just 7 points).

### 2. Best-k vs First-k
- **First-k**: use embedding[:, :k] directly (prefix eval — what MRL optimises).
- **Best-k**: rank dims by importance score, select the top-k dims (oracle).
- Three importance scoring methods (same as exp8):
  - `mean_abs[d]` = mean(|z[:, d]|)
  - `variance[d]` = var(z[:, d])
  - `probe_acc[d]` = logistic regression accuracy using only dim d
- Plot first-k and best-k accuracy on the same axis per model.
- Gap = best_k_acc - first_k_acc. Near-zero for MRL (ordering enforced),
  large for Standard and L1.

### 3. Gap Summary Across Seeds
- For each model, plot the gap curve (best_k - first_k) for both seeds on the
  same axes. Shaded band if seeds agree; spread visible if not.
- Confirms whether the ordering advantage of MRL is stable across data splits.

### 4. Method Agreement (Spearman rank correlation)
- Pairwise Spearman rho between the 3 importance methods per model.
- Tells us whether mean_abs, variance, and probe_acc agree on which dims matter.

---

## File Structure

```
experiments/exp9_dense_prefix.py     — main script
tests/run_tests_exp9.py              — test runner
```

---

## Output Structure

```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── training_curves.png              # loss vs epoch for all trained models (both seeds)
├── seed_42/
│   ├── standard_encoder_best.pt / standard_head_best.pt
│   ├── l1_encoder_best.pt       / l1_head_best.pt
│   ├── mrl_encoder_best.pt      / mrl_head_best.pt
│   ├── prefix_accuracy.png      # accuracy vs k (1..64) for all 4 models
│   ├── best_vs_first_k.png      # best-k vs first-k per model (3 importance methods)
│   └── method_agreement.png     # Spearman rho between importance methods
├── seed_123/
│   └── (same structure as seed_42/)
├── gap_comparison.png               # best_k - first_k gap per model, both seeds overlaid
├── results_summary.txt              # accuracy tables + gap tables for both seeds
├── runtime.txt
└── code_snapshot/
```

---

## Key Predictions

- **MRL**: first-k ≈ best-k across all k → gap ≈ 0. Ordering enforced by loss.
- **Standard**: large gap at small k, converges at full embed_dim. No ordering.
- **L1**: gap smaller than Standard (sparsity helps) but larger than MRL (no ordering).
- **PCA**: gap ≈ 0 by construction (variance ordering = best variance ordering).
- Both seeds should show the same qualitative pattern, confirming robustness.

---

## Reuse from Existing Code

- `config.py` — ExpConfig (add `data_seeds` field, or pass via CLI)
- `data/loader.py` — load_data (already accepts seed in cfg)
- `models/encoder.py` — MLPEncoder (unchanged)
- `models/heads.py` — build_head (unchanged)
- `losses/mat_loss.py` — build_loss with "standard", "l1", "matryoshka" (unchanged)
- `training/trainer.py` — train() (unchanged)
- `evaluation/prefix_eval.py` — evaluate_prefix_sweep, evaluate_pca_baseline (unchanged)
- `utility.py` — create_run_dir, save_runtime, save_code_snapshot (unchanged)

Import helpers from exp8 (no copy):
- `compute_importance_scores`
- `compute_best_vs_first_k`
- `compute_method_agreement`
- `plot_best_vs_first_k`
- `plot_method_agreement`

---

## CLI Flags

```
python experiments/exp9_dense_prefix.py               # full run (MNIST, 2 seeds, dense)
python experiments/exp9_dense_prefix.py --fast        # smoke test (digits, 1 seed, 5 epochs)
```

No `--use-exp7` flag — exp9 always trains its own models (different eval_prefixes
means weights from exp7 can't be fairly compared).

---

## Open Questions / Decisions Logged

- FF models excluded by design — focus is on ordering, not capacity.
- Dense eval (all 64 dims) is fast per-forward-pass; total runtime dominated by training.
  Expected runtime: ~10-15 min per seed on CPU for MNIST (same as exp7).
- data_seeds hardcoded to [42, 123] in exp9 (not in ExpConfig, to keep config clean).
