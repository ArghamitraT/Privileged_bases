# Experiment 8: Dimension Importance Scoring

## Context
After Exp7 showed that MRL beats L1 at prefix eval, the next question is: *why*?
The hypothesis is that MRL enforces front-loading — dim 0 carries the most info,
dim 1 the next, etc. L1 creates sparse embeddings but scatters the informative
dims randomly. This experiment makes that concrete by scoring each dimension's
importance three ways, comparing "first-k" (prefix eval) vs "best-k" (oracle
selection), and checking whether the three scoring methods agree with each other.

---

## New Files

| File | Purpose |
|------|---------|
| `experiments/exp8_dim_importance.py` | Main experiment script |
| `run_tests_exp8.py` | Unit tests + e2e smoke test |

No shared modules are modified.

---

## Three Analyses

### Analysis 1 — Importance Scoring (3 methods)
For Standard, L1, MRL, PCA: compute per-dimension scores on the test set:
1. `mean_abs[d]`  = `mean(|Z_test[:, d]|)` — one line, free
2. `variance[d]`  = `var(Z_test[:, d])` — one line, free
3. `probe_acc[d]` = logistic regression accuracy using only dim d (train → test)

### Analysis 2 — Best-k vs First-k
For each model and each k in eval_prefixes:
- **first_k**: logistic regression on `Z[:, :k]`
- **best_k_<method>**: sort dims by importance desc, take top-k, run LR
- **gap** = best_k_acc − first_k_acc

Key diagnostic: MRL gap ≈ 0 (ordering enforced). L1 / Standard gap > 0.

### Analysis 3 — Method Agreement
For each model, Spearman rank correlation between each pair of the 3 methods:
- (mean_abs, variance), (mean_abs, probe_acc), (variance, probe_acc)
- Scatter plot per pair, rho annotated in corner
- If methods agree → importance signal is robust; if not → metrics capture different things

---

## Function Signatures

```python
# Core analysis
def compute_importance_scores(
    Z_test, Z_train, y_train, y_test,
    max_probe_samples, seed, model_tag
) -> dict[str, np.ndarray]:
    # returns {"mean_abs": arr, "variance": arr, "probe_acc": arr}
    # probe_acc: loop over dims, wrap LR fit in try/except, skip degenerate dims

def compute_best_vs_first_k(
    Z_train, Z_test, y_train, y_test,
    importance_scores, eval_prefixes, seed, model_tag
) -> dict[str, dict[int, float]]:
    # returns {"first_k": {k: acc}, "best_k_mean_abs": ..., "best_k_variance": ...,
    #          "best_k_probe_acc": ...}

def compute_method_agreement(importance_scores, model_tag) -> dict[tuple, float]:
    # returns {("mean_abs", "variance"): rho, ...} via scipy.stats.spearmanr

# PCA baseline
def get_pca_embeddings_np(data, cfg) -> tuple[np.ndarray, np.ndarray]:
    # n_comp = min(embed_dim, n_train, n_features)
    # zero-pad if n_comp < embed_dim

# Weight loading
def load_models_from_exp7(exp7_dir, cfg, data):
    # loads standard/l1/mat encoder+head .pt files, returns 6 modules in eval mode

# Plotting
def plot_importance_scores(all_scores, run_dir, cfg) -> None
def plot_dim_importance_heatmap(all_scores, run_dir, cfg) -> None
def plot_best_vs_first_k(all_gap_results, run_dir, cfg) -> None
def plot_method_agreement(all_scores, all_agreement, run_dir, cfg) -> None
def plot_training_curves(run_dir, model_tags) -> None  # placeholder if --use-exp7

def save_results_summary(all_gap_results, all_agreement, all_scores,
                         eval_prefixes, run_dir) -> None
```

---

## Imports — What to Reuse vs New

**Import from existing modules (do NOT copy):**
```python
from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot, get_path
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss
from training.trainer import train
from experiments.exp7_mrl_vs_ff import train_single_model, get_embeddings_np
```

**New imports:**
```python
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
try:
    import seaborn as sns; _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
```

---

## Plot Layouts

| File | Layout | Description |
|------|--------|-------------|
| `importance_scores.png` | 3 rows × 4 cols | rows=methods, cols=models; horizontal bar charts per dim |
| `dim_importance_heatmap.png` | 3 rows × 1 col | rows=methods; heatmap (models × dims), `RdYlGn`, normalized |
| `best_vs_first_k.png` | 1 row × 4 cols | cols=models; 4 lines per plot: first-k (black) + 3 best-k (colored dashed) |
| `method_agreement.png` | 4 rows × 3 cols | rows=models, cols=method pairs; scatter + Spearman rho annotation |

Annotate heatmap cells and scatter dim labels only when `embed_dim <= 16`.

---

## `main()` — Order of Operations

```
run_start = time.time()
1.  Parse args (--fast, --use-exp7 PATH)
2.  Build cfg; set max_probe_samples (500 fast / 2000 full)
3.  set_seeds(cfg.seed)
4.  run_dir = create_run_dir()
5.  save_experiment_description(cfg, run_dir, exp7_dir, fast)
6.  data = load_data(cfg)
7.  Train Standard/L1/MRL  OR  load_models_from_exp7()
8.  plot_training_curves(run_dir, ["standard","l1","mat"])  # placeholder if --use-exp7
9.  Extract embeddings: Z_train/Z_test for Standard, L1, MRL, PCA
10. compute_importance_scores() for each model  [Analysis 1]
11. compute_best_vs_first_k() for each model    [Analysis 2]
12. compute_method_agreement() for each model   [Analysis 3]
13. plot_importance_scores()
14. plot_dim_importance_heatmap()
15. plot_best_vs_first_k()
16. plot_method_agreement()
17. save_results_summary()
18. save_runtime(run_dir, time.time() - run_start)
19. save_code_snapshot(run_dir)
```

---

## Config Settings

| Mode | dataset | embed_dim | eval_prefixes | epochs | max_probe_samples |
|------|---------|-----------|---------------|--------|-------------------|
| `--fast` | digits | 16 | [1,2,4,8,16] | 5 | 500 |
| full | mnist | 64 | [1,2,4,8,16,32,64] | 20 | 2000 |

---

## Edge Cases

| Case | Handling |
|------|---------|
| Degenerate dim (all zeros) | Wrap LR fit in try/except → assign probe_acc=0.0 |
| PCA n_comp < embed_dim | Zero-pad transform output to embed_dim |
| k=1 single dim | LR handles 1 feature; no special case |
| k=embed_dim best-k | All dims used → best_k ≈ first_k (sanity check) |
| --use-exp7 training curves | Write placeholder PNG with text message |
| seaborn missing | Fallback to ax.imshow + plt.colorbar |
| scipy.stats.spearmanr API | `result.statistic if hasattr(result,"statistic") else result.correlation` |

---

## `run_tests_exp8.py` — Tests

| Test | What it checks |
|------|---------------|
| `test_compute_importance_scores` | Shape, range, informative dim ranks higher on probe_acc |
| `test_compute_best_vs_first_k` | Keys present, all accs in [0,1], best_k >= first_k, at k=embed_dim both equal |
| `test_compute_method_agreement` | 3 pairs, rho in [-1,1], perfectly correlated = 1.0 |
| `test_get_pca_embeddings_np` | Shape correct, no NaN, train mean ≈ 0 |
| `test_importance_scores_degenerate_dim` | All-zero column → no crash, valid output |
| `test_plot_functions_no_crash` | All 4 plot functions run on tiny synthetic data, PNGs created |
| `test_e2e_fast` (slow) | `python experiments/exp8_dim_importance.py --fast` → all mandatory files exist |

BLAS env vars must be set at the very top of `run_tests_exp8.py` (before any imports):
```python
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
```

---

## CLAUDE.md Updates

After implementation, add Experiment 8 section to CLAUDE.md with:
- Idea, key analyses, file structure, how to run, expected output
- Note about importing `train_single_model` and `get_embeddings_np` from exp7

---

## Mandatory Output Files

```
files/results/exprmnt_{timestamp}/
├── experiment_description.log
├── training_curves.png          ← MANDATORY (placeholder if --use-exp7)
├── importance_scores.png
├── dim_importance_heatmap.png
├── best_vs_first_k.png
├── method_agreement.png
├── results_summary.txt
├── runtime.txt
└── code_snapshot/
```

---

## Critical Files for Reference

- `experiments/exp7_mrl_vs_ff.py` — import `train_single_model`, `get_embeddings_np`; follow structure
- `run_tests_exp7.py` — exact test file pattern to replicate
- `evaluation/prefix_eval.py` — `evaluate_pca_baseline` pattern for `get_pca_embeddings_np`
- `config.py` — `ExpConfig` fields, set `experiment_name="exp8_dim_importance"`
- `utility.py` — `create_run_dir`, `save_runtime`, `save_code_snapshot`
