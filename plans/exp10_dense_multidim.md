# Plan: Experiment 10 — Dense Prefix Sweep (MRL vs Standard vs L1 vs PCA, No FF)
# + Wrapper: run_exp10_8_multidim.py (Exp10 → Exp8 for each embed_dim)

**Goal**: Exp7 without FF models, evaluated at every dimension k=1..embed_dim
(dense sweep). Supports multiple embedding sizes via `--embed-dim` flag.
A wrapper script runs Exp10 then Exp8 sequentially for each dim, so a single
command produces both the dense accuracy curves and the per-dimension importance
analysis.

---

## Motivation

Exp7 evaluates at sparse powers-of-2 checkpoints and includes FF models for
capacity matching. Here we want:
1. Smooth, continuous accuracy curves (every k) — Exp10
2. Per-dimension importance scoring and best-k vs first-k gap — Exp8 (reusing weights)

Key differences from existing experiments:
- **vs Exp7**: no FF models; dense eval (k=1..embed_dim) instead of sparse
- **vs Exp9**: adds L1 model; adds 1-NN metric; no multi-seed; no best-k analysis

---

## Part 1: Experiment 10

### New script: `experiments/exp10_dense_multidim.py`

**Models trained**: Standard, L1, MRL
**Analytical baseline**: PCA (no training)
**Evaluation**: at every k from 1 to embed_dim
**Metrics**: linear accuracy + 1-NN accuracy

**CLI flags:**
```bash
python experiments/exp10_dense_multidim.py                       # full run (MNIST, embed_dim=64)
python experiments/exp10_dense_multidim.py --fast                # smoke test (digits, 5 epochs)
python experiments/exp10_dense_multidim.py --embed-dim 8         # full run, embed_dim=8
python experiments/exp10_dense_multidim.py --embed-dim 16        # full run, embed_dim=16
python experiments/exp10_dense_multidim.py --embed-dim 32        # full run, embed_dim=32
python experiments/exp10_dense_multidim.py --embed-dim 8 --fast  # smoke test at dim=8
```

**main() flow:**
1. Parse args; build config:
   - `--fast`: dataset="digits", embed_dim=16 (or --embed-dim override), epochs=5, patience=3
   - Full: dataset="mnist", embed_dim=64 (or --embed-dim override), epochs=20, patience=5
   - `eval_prefixes = list(range(1, embed_dim + 1))` — always dense
2. Create run_dir, save experiment_description.log
3. Load data
4. Train Standard, L1, MRL via `train_single_model()` (imported from exp7, no copy)
5. Plot training_curves.png (MANDATORY)
6. Extract embeddings via `get_embeddings_np()` (imported from exp7, no copy)
7. Evaluate at every k:
   - Linear accuracy: LR probe on z[:, :k]
   - 1-NN accuracy via `evaluate_prefix_1nn()` (imported from exp7, no copy)
   - PCA via `evaluate_pca_1nn()` (imported from exp7, no copy)
8. Plot:
   - `linear_accuracy_curve.png` — 4 lines (Standard, L1, MRL, PCA) vs k
   - `1nn_accuracy_curve.png`    — 4 lines vs k
   - `combined_comparison.png`   — 2-panel (linear top, 1-NN bottom)
9. Save results_summary.txt, runtime.txt, code_snapshot/

**`--embed-dim` logic:**
```python
if args.embed_dim is not None:
    cfg.embed_dim = args.embed_dim
cfg.eval_prefixes = list(range(1, cfg.embed_dim + 1))  # always dense
```

**Key reuse (no copy):**
- `train_single_model`, `get_embeddings_np` — from `exp7_mrl_vs_ff`
- `evaluate_prefix_1nn`, `evaluate_pca_1nn` — from `exp7_mrl_vs_ff`

**No shared module changes.**

**Output structure:**
```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── standard_encoder_best.pt / standard_head_best.pt
├── l1_encoder_best.pt       / l1_head_best.pt
├── mat_encoder_best.pt      / mat_head_best.pt
├── standard_train.log / l1_train.log / mat_train.log
├── training_curves.png         # MANDATORY
├── linear_accuracy_curve.png   # 4 lines vs k=1..embed_dim
├── 1nn_accuracy_curve.png      # 4 lines vs k=1..embed_dim
├── combined_comparison.png     # 2-panel
├── results_summary.txt
├── runtime.txt
└── code_snapshot/
```

---

## Part 2: Exp8 changes — rename `--use-exp7` → `--use-weights`

Exp10 saves weights with the same filenames as Exp7
(`standard_encoder_best.pt`, etc.), so Exp8 can load them directly.
The flag `--use-exp7 PATH` is renamed `--use-weights PATH` to make it clear
it accepts output directories from any experiment (Exp7 or Exp10).

**Changes to `experiments/exp8_dim_importance.py`:**
- Rename `--use-exp7` argument to `--use-weights`
- Rename internal variable `exp7_dir` → `weights_dir`
- Rename `load_models_from_exp7()` → `load_models_from_dir()`
- Update docstring `Usage:` section

**Also add `--embed-dim` flag to Exp8** (same powers-of-2 derivation as exp7):
- `--embed-dim N` overrides `cfg.embed_dim` and derives `eval_prefixes`
- Must match the embed_dim used in the weights directory being loaded
- The wrapper enforces consistency automatically

**New Exp8 CLI invocations:**
```bash
python experiments/exp8_dim_importance.py --embed-dim 8 --use-weights PATH
python experiments/exp8_dim_importance.py --embed-dim 16 --use-weights PATH
python experiments/exp8_dim_importance.py --embed-dim 32 --use-weights PATH
python experiments/exp8_dim_importance.py --embed-dim 8 --fast   # smoke test
```

---

## Part 3: New wrapper script: `scripts/run_exp10_8_multidim.py`

Runs Exp10 then Exp8 sequentially for each of the three dims.

**Logic:**
```
for dim in [8, 16, 32]:
    1. Run: python experiments/exp10_dense_multidim.py --embed-dim {dim}
       Stream output live. Parse "[utility] Run directory created: /path/..."
       to extract exp10_output_dir.
    2. Run: python experiments/exp8_dim_importance.py --embed-dim {dim}
                --use-weights {exp10_output_dir}
       Stream output live. Parse output dir the same way.
    3. Record (dim, exp10_dir, exp8_dir).

Print final summary table on success.
Stop immediately on any subprocess failure.
```

**Summary table printed at the end:**
```
┌────────────┬──────────────────────────────────┬──────────────────────────────────┐
│ embed_dim  │ exp10 output dir                 │ exp8 output dir                  │
├────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ 8          │ files/results/exprmnt_..._A/      │ files/results/exprmnt_..._B/     │
│ 16         │ files/results/exprmnt_..._C/      │ files/results/exprmnt_..._D/     │
│ 32         │ files/results/exprmnt_..._E/      │ files/results/exprmnt_..._F/     │
└────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

**CLI:**
```bash
python scripts/run_exp10_8_multidim.py              # full run, dims=[8,16,32]
python scripts/run_exp10_8_multidim.py --fast       # smoke test, all 3 dims
python scripts/run_exp10_8_multidim.py --dims 8 16  # run only specified dims
```

**Implementation details:**
- Parse output dir using: `re.search(r'\[utility\] Run directory created: (.+)', line)`
- Stream each subprocess line-by-line to terminal so progress is visible
- Set `OMP_NUM_THREADS=1` etc. on child processes (per Known Issue #3 in CLAUDE.md)
- `--fast` is forwarded to both exp10 and exp8 subprocesses

---

## Part 4: Test file updates

- `tests/run_tests_exp10.py` — new file:
  1. Config derivation: dense prefixes for each embed_dim
  2. `get_embeddings_np` output shape check
  3. Linear eval output shape check
  4. 1-NN eval output shape check
  5. E2E smoke: `python experiments/exp10_dense_multidim.py --fast --embed-dim 8`

- `tests/run_tests_exp8.py` — add:
  1. Unit test for `--use-weights` flag (renamed from `--use-exp7`)
  2. Unit test for `--embed-dim 8` config derivation (powers-of-2 prefixes)

---

## Implementation Order

1. Edit `exp8_dim_importance.py` — rename `--use-exp7` → `--use-weights`, add `--embed-dim`
2. Create `experiments/exp10_dense_multidim.py`
3. Create `scripts/run_exp10_8_multidim.py`
4. Create `tests/run_tests_exp10.py`
5. Update `tests/run_tests_exp8.py`
6. Update `EXPERIMENTS.md` — exp8 flag rename + exp10 How to Run + wrapper CLI
7. Smoke-test: `python scripts/run_exp10_8_multidim.py --fast --dims 8`

---

## Risks / Notes

- **embed_dim consistency**: `--embed-dim` passed to exp8 must match the dim used
  in the weights dir. The wrapper enforces this; manual runs require care.
- **1-NN at every k on MNIST**: subsample database to 10k (same as exp7) for speed.
- **Dense eval_prefixes**: up to 64 LR fits per model at dim=64; only 8 at dim=8.
- **`--use-exp7` rename**: backwards-incompatible CLI change — any existing shell
  scripts using `--use-exp7` must be updated. No shared module changes.
