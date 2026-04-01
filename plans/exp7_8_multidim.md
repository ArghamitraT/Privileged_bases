# Plan: Exp7 with embed_dim ∈ {8, 16, 32}

**Goal**: Add `--embed-dim` flag to `exp7_mrl_vs_ff.py` so it can be run at
embedding sizes 8, 16, and 32 (in addition to the default 64).
Exp7 is run standalone — no wrapper needed.
The wrapper pairing (dense sweep + importance scoring) lives in `plans/exp10_dense_multidim.md`.

---

## Motivation

All prior full runs used `embed_dim=64`. Smaller dims let us check whether MRL's
FF-vs-MRL advantage holds at lower capacity, and compare sparse-eval (exp7)
against dense-eval (exp10) at the same embed_dim.

---

## Changes Required

### 1. `experiments/exp7_mrl_vs_ff.py` — add `--embed-dim` flag

- Add `parser.add_argument("--embed-dim", type=int, default=None)`
- In `main()`, after building the base config, if `--embed-dim N` is provided:
  - Override `cfg.embed_dim = N`
  - Derive `eval_prefixes` as all powers-of-2 up to N:
    - `N=8`  → `[1, 2, 4, 8]`
    - `N=16` → `[1, 2, 4, 8, 16]`
    - `N=32` → `[1, 2, 4, 8, 16, 32]`
    - `N=64` → `[1, 2, 4, 8, 16, 32, 64]` (same as current default)
  - Override `cfg.eval_prefixes = derived_list`
- All other settings (dataset=mnist, epochs=20, patience=5) unchanged.
- FF models are trained for each k in the derived prefix list.
- The run directory printout `[utility] Run directory created: /path/...` already
  exists — the wrapper will parse this line from stdout if needed.
- Update the script docstring `Usage:` section.

**New CLI invocations:**
```bash
python experiments/exp7_mrl_vs_ff.py --embed-dim 8    # embed_dim=8,  prefixes=[1,2,4,8]
python experiments/exp7_mrl_vs_ff.py --embed-dim 16   # embed_dim=16, prefixes=[1,2,4,8,16]
python experiments/exp7_mrl_vs_ff.py --embed-dim 32   # embed_dim=32, prefixes=[1,2,4,8,16,32]
python experiments/exp7_mrl_vs_ff.py --embed-dim 8 --fast   # smoke test at dim=8
```

### 2. Update test file: `tests/run_tests_exp7.py`

- Add a unit test that verifies `--embed-dim 8` sets `embed_dim=8` and
  `eval_prefixes=[1,2,4,8]` correctly (no training needed).

### 3. Update `EXPERIMENTS.md`

- Add `--embed-dim` invocations to Exp7's How to Run section.

---

## Implementation Order

1. Edit `exp7_mrl_vs_ff.py` — add `--embed-dim` flag + docstring update
2. Update `tests/run_tests_exp7.py`
3. Update `EXPERIMENTS.md`
4. Smoke-test: `python experiments/exp7_mrl_vs_ff.py --fast --embed-dim 8`

---

## Risks / Notes

- **FF models**: for dim=8 there are only 4 FF models (k=1,2,4,8); training time
  scales roughly linearly with the number of FF models.
- **No shared module changes** — override happens entirely inside `main()`.
- Exp7 is run independently per dim. The Exp10+Exp8 wrapper (see
  `plans/exp10_dense_multidim.md`) is the primary multi-dim workflow going forward.
