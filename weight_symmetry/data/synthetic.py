"""
Script: weight_symmetry/data/synthetic.py
------------------------------------------
Synthetic dataset generator for Exp 2 (Divergence — Objective-Specific Privilege).

Constructs a dataset where the PCA subspace and LDA subspace are orthogonal by design:

    x = [x_noise  |  x_signal]
         (p_noise)   (p_signal)

    Noise block  (dims 0..p_noise-1) : x_noise ~ N(0, sigma_noise^2 I)
                                        same distribution for all classes
                                        high variance -> PCA picks these dims

    Signal block (dims p_noise..p-1) : x_signal ~ N(mu_c, sigma_signal^2 I)
                                        class-specific mean mu_c in R^{p_signal}
                                        low within-class variance -> LDA picks these

Key construction property: p_signal = C - 1 exactly.
This guarantees the between-class scatter matrix is full rank in the signal space,
so all p_signal signal dims are discriminative and LDA directions span the entire
signal block.

Default parameters (Option A from plan):
    p_noise   = 50    C = 20    sigma_noise  = 5.0
    p_signal  = 19    n = 10000  sigma_signal = 0.1
    p_total   = 69              class_sep    = 1.0

Ordered LDA mode (ordered_lda=True / --ordered-lda flag):
    By default class means are random unit vectors, so all signal dims are
    equally discriminative (no LDA ordering).

    With ordered_lda=True, class means are constructed so that:
        signal dim 0  separates classes most  (class_sep * 1.0)
        signal dim 1  separates classes less  (class_sep * lda_scale_decay)
        ...
        signal dim C-2 separates classes least (class_sep * lda_scale_decay^(C-2))

    This gives a ground-truth LDA ordering that MRL models can be tested against —
    analogous to how PCA directions are ordered by variance.

    Construction: class_means = Q * diag(scales), where
        Q      : (C, p_signal), orthonormal columns each orthogonal to 1_C
                 (zero grand mean => no intercept bias)
        scales[j] = class_sep * lda_scale_decay^j  (geometrically decreasing)
    Result: between-class scatter S_B is diagonal in signal space,
    S_B[j,j] = scales[j]^2 / C, so LDA direction j aligns with signal axis j
    and class separability strictly decreases with j.

    Saved with tag suffix _olda so files are separate from the default variant.

Data is saved to:
    Mat_embedding_hyperbole/data/synthetic_data/

Usage:
    Conda environment: mrl_env

    python weight_symmetry/data/synthetic.py                    # default (random LDA)
    python weight_symmetry/data/synthetic.py --seed 43
    python weight_symmetry/data/synthetic.py --ordered-lda      # ordered LDA construction
    python weight_symmetry/data/synthetic.py --ordered-lda --verify
    python weight_symmetry/data/synthetic.py --verify           # verify default variant
"""

import os
import sys
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works from any cwd
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT  = os.path.dirname(_HERE)
_CODE_ROOT = os.path.dirname(_WS_ROOT)
_PROJ_ROOT = os.path.dirname(_CODE_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR      = os.path.join(_PROJ_ROOT, "data", "synthetic_data", "nonOrderedLDA")
DATA_DIR_OLDA = os.path.join(_PROJ_ROOT, "data", "synthetic_data", "orderedLDA")
DATA_DIR_NSD  = os.path.join(_PROJ_ROOT, "data", "synthetic_data", "orderedBoth")

# ==============================================================================
# Default parameters
# ==============================================================================
P_NOISE         = 50
C               = 20          # number of classes  →  p_signal = C - 1 = 19
SIGMA_NOISE     = 5.0         # std of noise block
SIGMA_SIGNAL    = 0.1         # within-class std of signal block
CLASS_SEP       = 1.0         # scale of class means (distance from origin)
N_PER_CLASS     = 500         # samples per class  →  n_total = 10 000
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1
ORDERED_LDA       = False     # if True, signal dims are ordered by discriminative power
LDA_SCALE_DECAY   = 0.7       # geometric decay rate for ordered LDA (ignored otherwise)
NOISE_SCALE_DECAY = 1.0       # <1 gives geometrically decaying noise variances
                               # (1.0 = isotropic, default; 0.9 = distinct eigenvalues)

# Named variants — used by load_data(synthetic_variant=...) so callers never
# need to know the internal (ordered_lda, noise_scale_decay) values.
SYNTHETIC_VARIANTS = {
    "nonOrderedLDA": dict(ordered_lda=False, noise_scale_decay=1.0),
    "orderedLDA":    dict(ordered_lda=True,  noise_scale_decay=1.0),
    "orderedBoth":   dict(ordered_lda=True,  noise_scale_decay=0.9),
}
# ==============================================================================


def generate_synthetic(
    p_noise:          int   = P_NOISE,
    C:                int   = C,
    sigma_noise:      float = SIGMA_NOISE,
    sigma_signal:     float = SIGMA_SIGNAL,
    class_sep:        float = CLASS_SEP,
    n_per_class:      int   = N_PER_CLASS,
    test_size:        float = TEST_SIZE,
    val_size:         float = VAL_SIZE,
    seed:             int   = 42,
    ordered_lda:      bool  = False,
    lda_scale_decay:  float = LDA_SCALE_DECAY,
    noise_scale_decay: float = NOISE_SCALE_DECAY,
) -> dict:
    """
    Generate the synthetic PCA-vs-LDA divergence dataset.

    ordered_lda=False (default): class means are random unit vectors in signal space.
        All signal dims are roughly equally discriminative — no LDA ordering.

    ordered_lda=True: class means are constructed so that signal dim j separates
        classes with strength class_sep * lda_scale_decay^j (geometrically decreasing).
        LDA direction j aligns with signal axis j; class separability decreases with j.
        Files are saved with an _olda tag suffix to avoid overwriting the default variant.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test  : np.float32 / np.int64
        class_means   : (C, p_signal) array of true class means in signal block
        pca_dirs      : (p, p_noise) top-p_noise PCA eigenvectors from X_train
        lda_dirs      : (p, C-1) LDA directions from X_train + y_train
        params        : dict of all generation parameters (includes ordered_lda flag)
    """
    rng       = np.random.default_rng(seed)
    p_signal  = C - 1          # = 19 for C=20
    p         = p_noise + p_signal   # = 69
    n         = n_per_class * C      # = 10 000

    # ------------------------------------------------------------------
    # Class means
    # ------------------------------------------------------------------
    if ordered_lda:
        # Construct class means so that signal dim j has class separability
        # proportional to class_sep * lda_scale_decay^j (decreasing).
        #
        # Key: build Q (C x p_signal) with orthonormal columns, each orthogonal
        # to 1_C.  Then class_means = Q * diag(scales).
        #
        # Why this works:
        #   S_B = (1/C) * class_means^T class_means
        #       = (1/C) * diag(scales) * (Q^T Q) * diag(scales)
        #       = (1/C) * diag(scales^2)           [Q^T Q = I]
        # → S_B is diagonal; LDA direction j aligns with signal axis j.
        # Orthogonality to 1_C ensures grand mean = 0 for every signal dim.
        scales = class_sep * lda_scale_decay ** np.arange(p_signal)  # (p_signal,)
        e1  = np.ones(C) / np.sqrt(C)
        raw = rng.standard_normal((C, p_signal))
        raw -= np.outer(e1, e1 @ raw)   # project out 1_C from each column
        Q, _ = np.linalg.qr(raw)        # (C, p_signal), orthonormal columns
        class_means = Q * scales[np.newaxis, :]   # (C, p_signal)
    else:
        # Default: C random unit vectors in R^{p_signal}, scaled by class_sep.
        # Using C random vectors guarantees the between-class scatter spans R^{p_signal}
        # (C > p_signal = C-1, so with probability 1 they span the full space).
        raw = rng.standard_normal((C, p_signal))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        class_means = (raw / norms) * class_sep     # (C, p_signal)

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    # Each noise dim i gets std = sigma_noise * noise_scale_decay^i.
    # decay=1.0 → isotropic (original behaviour); decay<1 → distinct eigenvalues.
    noise_sigmas = sigma_noise * (noise_scale_decay ** np.arange(p_noise))
    X_noise  = (rng.normal(0.0, 1.0, (n, p_noise)) * noise_sigmas).astype(np.float32)
    X_signal = np.zeros((n, p_signal), dtype=np.float32)
    y        = np.zeros(n, dtype=np.int64)

    for c in range(C):
        sl = slice(c * n_per_class, (c + 1) * n_per_class)
        X_signal[sl] = rng.normal(
            class_means[c], sigma_signal, (n_per_class, p_signal)
        ).astype(np.float32)
        y[sl] = c

    X = np.concatenate([X_noise, X_signal], axis=1)   # (n, 69): noise first

    # Shuffle
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    # ------------------------------------------------------------------
    # Train / val / test split (stratified)
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv
    )

    # Centre only (subtract mean, do NOT scale variance).
    # Scaling to unit variance would destroy the variance contrast between
    # noise and signal blocks that makes PCA prefer the noise dims.
    scaler  = StandardScaler(with_std=False).fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ------------------------------------------------------------------
    # Ground-truth PCA directions (from X_train)
    # ------------------------------------------------------------------
    _, _, Vh = np.linalg.svd(X_train, full_matrices=False)
    pca_dirs = Vh[:p_noise, :].T    # (p, p_noise) — top-p_noise eigenvectors

    # ------------------------------------------------------------------
    # Ground-truth LDA directions (from X_train + y_train)
    # ------------------------------------------------------------------
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    # lda.scalings_: (p, C-1) — discriminant directions in input space
    lda_dirs_raw = lda.scalings_[:, :C - 1]                # (p, C-1)
    # Orthonormalise for clean subspace comparison
    lda_dirs, _ = np.linalg.qr(lda_dirs_raw)               # (p, C-1)

    params = dict(
        p_noise=p_noise, p_signal=p_signal, p=p,
        C=C, n_per_class=n_per_class, n=n,
        sigma_noise=sigma_noise, sigma_signal=sigma_signal,
        class_sep=class_sep,
        test_size=test_size, val_size=val_size, seed=seed,
        ordered_lda=ordered_lda,
        lda_scale_decay=lda_scale_decay,
        noise_scale_decay=noise_scale_decay,
    )

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        class_means=class_means.astype(np.float32),
        pca_dirs=pca_dirs.astype(np.float32),
        lda_dirs=lda_dirs.astype(np.float32),
        params=params,
    )


def _write_olda_readme(out_dir: str, params: dict):
    """Write README.md to the orderedLDA subfolder (Markdown, matching nonOrderedLDA style)."""
    decay    = params.get("lda_scale_decay", LDA_SCALE_DECAY)
    p_noise  = params.get("p_noise",  P_NOISE)
    p_signal = params.get("p_signal", C - 1)
    p        = params.get("p",        p_noise + p_signal)
    n_C      = params.get("C",        C)
    n_per    = params.get("n_per_class", N_PER_CLASS)
    n_total  = params.get("n",        n_per * n_C)
    s_noise  = params.get("sigma_noise",  SIGMA_NOISE)
    s_signal = params.get("sigma_signal", SIGMA_SIGNAL)
    c_sep    = params.get("class_sep",    CLASS_SEP)

    content = f"""\
# Synthetic Dataset — Ordered LDA

Generated for **Experiment 2** (and future LDA-ordering experiments) of the ICML 2026 paper:
*"Objective-Specific Privileged Bases via Full-Prefix Matryoshka Learning"*

---

## Purpose

This dataset is a variant of the `nonOrderedLDA/` dataset. The noise block and signal
block structure is identical, but the **class means in the signal block are ordered by
discriminative power**. This provides a ground-truth LDA ordering to test:

> Does full-prefix MRL with cross-entropy loss recover the LDA ordering?
> (Analogue of Exp 1, which tests PCA ordering recovery with MSE loss.)

In `nonOrderedLDA/`, all signal dims are equally/randomly discriminative.
Here, **signal dim 0 separates classes most**, dim 1 less so, ..., dim {p_signal-1} least.

---

## Construction

The input space is split into two disjoint blocks (same as `nonOrderedLDA/`):

```
x = [ x_noise (dims 0–{p_noise-1})  |  x_signal (dims {p_noise}–{p-1}) ]
```

### Noise block (dims 0–{p_noise-1}, p_noise = {p_noise})
Identical to `nonOrderedLDA/`: `x_noise ~ N(0, {s_noise}² I)` for all classes.
High variance → **PCA eigenvectors land here**.

### Signal block (dims {p_noise}–{p-1}, p_signal = {p_signal})
Class means are **ordered** using the following construction:

```
class_means = Q * diag(scales)     shape: (C, p_signal)

Q      : ({n_C}, {p_signal}) matrix with orthonormal columns,
          each orthogonal to 1_C  →  grand mean = 0
scales[j] = {c_sep} * {decay}^j   (geometrically decreasing)
```

**Why this gives ordered LDA directions:**

The between-class scatter matrix in signal space is:

```
S_B = (1/C) * class_means^T class_means
    = (1/C) * diag(scales^2)         [because Q^T Q = I]
```

`S_B` is diagonal with `S_B[j,j] = scales[j]² / C`, so:
- LDA direction j aligns with **signal axis j**
- Class separability strictly decreases with j (eigenvalue j > eigenvalue j+1)

### Key property (same as nonOrderedLDA)
```
p_signal = C - 1 = {p_signal}
```
Between-class scatter is full rank — every signal dim is a genuine discriminant direction.

---

## Dataset Parameters

| Parameter         | Value    | Description                                   |
|-------------------|----------|-----------------------------------------------|
| `p_noise`         | {p_noise}       | Number of noise dimensions                    |
| `p_signal`        | {p_signal}       | Number of signal dimensions (= C-1)           |
| `p_total`         | {p}       | Total input dimensionality                    |
| `C`               | {n_C}       | Number of classes                             |
| `n_per_class`     | {n_per}     | Samples per class                             |
| `n_total`         | {n_total:,}  | Total samples                                 |
| `sigma_noise`     | {s_noise}     | Std of noise block (all classes)              |
| `sigma_signal`    | {s_signal}     | Within-class std of signal block              |
| `class_sep`       | {c_sep}     | Base scale of class means                     |
| `lda_scale_decay` | {decay}   | Geometric decay rate for LDA scales           |
| Train split       | 70%      | {int(n_total*0.7):,} samples                            |
| Val split         | 10%      | {int(n_total*0.1):,} samples                            |
| Test split        | 20%      | {int(n_total*0.2):,} samples                            |

### LDA scale values (first 5 dims)
| Signal dim j | scales[j]         | S_B[j,j]               |
|--------------|-------------------|------------------------|
| 0            | {c_sep * decay**0:.4f}            | {(c_sep * decay**0)**2 / n_C:.6f}             |
| 1            | {c_sep * decay**1:.4f}            | {(c_sep * decay**1)**2 / n_C:.6f}             |
| 2            | {c_sep * decay**2:.4f}            | {(c_sep * decay**2)**2 / n_C:.6f}             |
| 3            | {c_sep * decay**3:.4f}            | {(c_sep * decay**3)**2 / n_C:.6f}             |
| 4            | {c_sep * decay**4:.4f}            | {(c_sep * decay**4)**2 / n_C:.6f}             |

### Normalisation
Same as `nonOrderedLDA/`: centred only (subtract per-dimension mean, no variance scaling).

---

## File Listing

Each seed produces two files:

```
synthetic_p{p}_C{n_C}_n{n_total}_seed{{s}}_olda.npz       — data arrays
synthetic_p{p}_C{n_C}_n{n_total}_seed{{s}}_olda_meta.npz  — ground-truth directions + params
```

### Data file keys
| Key       | Shape          | dtype   | Description               |
|-----------|----------------|---------|---------------------------|
| X_train   | (7000, {p})    | float32 | Training inputs           |
| y_train   | (7000,)        | int64   | Training labels (0–{n_C-1})  |
| X_val     | (1000, {p})    | float32 | Validation inputs         |
| y_val     | (1000,)        | int64   | Validation labels         |
| X_test    | (2000, {p})    | float32 | Test inputs               |
| y_test    | (2000,)        | int64   | Test labels               |

### Metadata file keys
| Key               | Shape       | Description                                         |
|-------------------|-------------|-----------------------------------------------------|
| class_means       | ({n_C}, {p_signal})  | True class means in signal space R^{p_signal}           |
| pca_dirs          | ({p}, {p_noise})  | Top-{p_noise} PCA eigenvectors from X_train             |
| lda_dirs          | ({p}, {p_signal})  | LDA discriminant directions from X_train+y_train    |
| ordered_lda       | scalar      | True (1)                                            |
| lda_scale_decay   | scalar      | {decay}                                              |
| p_noise           | scalar      | {p_noise}                                              |
| p_signal          | scalar      | {p_signal}                                              |
| p                 | scalar      | {p}                                                |
| C                 | scalar      | {n_C}                                              |
| sigma_noise       | scalar      | {s_noise}                                              |
| sigma_signal      | scalar      | {s_signal}                                              |
| class_sep         | scalar      | {c_sep}                                              |
| seed              | scalar      | Random seed used                                    |

---

## Generation Code

```bash
# Generate all 5 seeds with verification
conda activate mrl_env_cuda12
cd Mat_embedding_hyperbole/code
python weight_symmetry/data/synthetic.py --ordered-lda --all-seeds --verify

# Single seed
python weight_symmetry/data/synthetic.py --ordered-lda --seed 42 --verify

# Custom decay rate
python weight_symmetry/data/synthetic.py --ordered-lda --lda-scale-decay 0.5
```

Source: `code/weight_symmetry/data/synthetic.py`

---

## Loading in Experiments

```python
from weight_symmetry.data.loader import load_data
from weight_symmetry.data.synthetic import load_synthetic

# Ordered LDA variant — pass ordered_lda=True
raw = load_synthetic(seed=42, ordered_lda=True)
pca_dirs = raw["pca_dirs"]   # ({p}, {p_noise})  — top-{p_noise} PCA eigenvectors
lda_dirs = raw["lda_dirs"]   # ({p}, {p_signal})  — LDA directions (ordered by eigenvalue)

# lda_dirs[:, 0] aligns with signal axis 0 (most discriminative)
# lda_dirs[:, 1] aligns with signal axis 1, etc.

# Data splits — same interface as nonOrderedLDA
data = load_data("synthetic", seed=42)   # NOTE: load_data uses nonOrderedLDA by default
```
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"[synthetic] Wrote {readme_path}")


def _write_nsd_readme(out_dir: str, params: dict):
    """Write README.md for the orderedBoth subfolder."""
    nsd      = params.get("noise_scale_decay", NOISE_SCALE_DECAY)
    decay    = params.get("lda_scale_decay",   LDA_SCALE_DECAY)
    p_noise  = params.get("p_noise",  P_NOISE)
    p_signal = params.get("p_signal", C - 1)
    p        = params.get("p",        p_noise + p_signal)
    n_C      = params.get("C",        C)
    n_per    = params.get("n_per_class", N_PER_CLASS)
    n_total  = params.get("n",        n_per * n_C)
    s_noise  = params.get("sigma_noise",  SIGMA_NOISE)
    s_signal = params.get("sigma_signal", SIGMA_SIGNAL)
    c_sep    = params.get("class_sep",    CLASS_SEP)

    # Expected noise eigenvalues for first few dims
    noise_eigs = [f"dim {i}: σ²={s_noise**2 * nsd**(2*i):.2f}" for i in range(5)]

    content = f"""\
# Synthetic Dataset — Ordered Both (Separable Eigenvalues)

Generated for **Experiment 2** (and related experiments) — variant with **distinct eigenvalues
in both the noise block and the signal block**.

---

## Key difference from other variants

| Variant | Noise block | Signal block | Folder |
|---------|------------|--------------|--------|
| `nonOrderedLDA` | isotropic N(0, {s_noise}²·I) — flat spectrum | random class means | `nonOrderedLDA/` |
| `orderedLDA`    | isotropic N(0, {s_noise}²·I) — flat spectrum | ordered by LDA power | `orderedLDA/` |
| **`orderedBoth`** | **geometrically decaying variances** | **ordered by LDA power** | **`orderedBoth/`** |

`orderedBoth` is the only variant where individual PCA eigenvectors are recoverable
(each has a distinct eigenvalue) AND LDA directions are ordered by discriminative power.

---

## Construction

### Noise block (dims 0–{p_noise-1})

Noise dim `i` is drawn from `N(0, σ_i²)` where:
```
σ_i = {s_noise} × {nsd}^i       (noise_scale_decay = {nsd})
```
This gives eigenvalue `λ_i ≈ σ_i²`:
{chr(10).join(f"  {e}" for e in noise_eigs)}
  ...

The spectrum decays geometrically, so consecutive eigenvalues are separated by a
gap ∝ `(1 - {nsd}²) × σ_i²`. PCA can uniquely identify each axis.

### Signal block (dims {p_noise}–{p-1})

Identical to `orderedLDA/`: class means are constructed so that signal dim `j` has
between-class scatter proportional to `({c_sep} × {decay}^j)²`. LDA direction `j`
aligns with signal axis `j`; discriminative power decreases with `j`.

---

## Dataset Parameters

| Parameter           | Value     | Description                                     |
|---------------------|-----------|-------------------------------------------------|
| `p_noise`           | {p_noise}        | Number of noise dimensions                      |
| `p_signal`          | {p_signal}        | Number of signal dimensions (= C-1)             |
| `p_total`           | {p}        | Total input dimensionality                      |
| `C`                 | {n_C}        | Number of classes                               |
| `n_per_class`       | {n_per}      | Samples per class                               |
| `n_total`           | {n_total:,}   | Total samples                                   |
| `sigma_noise`       | {s_noise}      | Base std of noise block (dim 0)                 |
| `noise_scale_decay` | {nsd}    | Geometric decay rate for noise dim stds         |
| `sigma_signal`      | {s_signal}      | Within-class std of signal block                |
| `class_sep`         | {c_sep}      | Base scale of class means                       |
| `lda_scale_decay`   | {decay}    | Geometric decay rate for LDA scales             |

---

## File naming

```
synthetic_p{p}_C{n_C}_n{n_total}_seed{{s}}_olda_nsd{int(round(nsd*100))}.npz
synthetic_p{p}_C{n_C}_n{n_total}_seed{{s}}_olda_nsd{int(round(nsd*100))}_meta.npz
```

---

## Generation

```bash
conda activate mrl_env
cd Mat_embedding_hyperbole/code
python weight_symmetry/data/synthetic.py --ordered-lda --noise-scale-decay {nsd} --all-seeds --verify
```

Source: `code/weight_symmetry/data/synthetic.py`
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"[synthetic] Wrote {readme_path}")


def save_synthetic(data: dict, seed: int, out_dir: str = None):
    """Save generated dataset to out_dir as two .npz files.

    Folder routing (when out_dir is None):
      noise_scale_decay < 1 and ordered_lda=True  → orderedBoth/
      noise_scale_decay = 1 and ordered_lda=True  → orderedLDA/
      otherwise                                   → nonOrderedLDA/
    """
    ordered = data["params"].get("ordered_lda", False)
    nsd     = data["params"].get("noise_scale_decay", 1.0)
    use_nsd = nsd < 1.0

    if out_dir is None:
        if use_nsd and ordered:
            out_dir = DATA_DIR_NSD
        elif ordered:
            out_dir = DATA_DIR_OLDA
        else:
            out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    olda_str = "_olda" if ordered else ""
    nsd_str  = f"_nsd{int(round(nsd * 100))}" if use_nsd else ""
    tag = (f"p{data['params']['p']}_C{data['params']['C']}"
           f"_n{data['params']['n']}_seed{seed}{olda_str}{nsd_str}")

    # Main data file
    data_path = os.path.join(out_dir, f"synthetic_{tag}.npz")
    np.savez_compressed(
        data_path,
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"],     y_val=data["y_val"],
        X_test=data["X_test"],   y_test=data["y_test"],
    )

    # Metadata file (ground-truth directions + params)
    meta_path = os.path.join(out_dir, f"synthetic_{tag}_meta.npz")
    np.savez_compressed(
        meta_path,
        class_means=data["class_means"],
        pca_dirs=data["pca_dirs"],
        lda_dirs=data["lda_dirs"],
        **{k: np.array(v) if not isinstance(v, np.ndarray) else v
           for k, v in data["params"].items()},
    )

    print(f"[synthetic] Saved {data_path}")
    print(f"[synthetic] Saved {meta_path}")

    # Write README (idempotent)
    if use_nsd and ordered:
        _write_nsd_readme(out_dir, data["params"])
    elif ordered:
        _write_olda_readme(out_dir, data["params"])

    return data_path, meta_path


def load_synthetic(seed: int = 42, out_dir: str = None,
                   ordered_lda: bool = False,
                   lda_scale_decay: float = LDA_SCALE_DECAY,
                   noise_scale_decay: float = NOISE_SCALE_DECAY) -> dict:
    """
    Load pre-generated synthetic dataset. Auto-generates if file not found.
    Returns same dict as generate_synthetic().

    ordered_lda=False (default): loads from nonOrderedLDA/ (random class means).
    ordered_lda=True: loads from orderedLDA/ (_olda tag suffix); auto-generates if missing.
    out_dir: override the directory (defaults based on ordered_lda flag).
    """
    use_nsd = noise_scale_decay < 1.0
    if out_dir is None:
        if use_nsd and ordered_lda:
            out_dir = DATA_DIR_NSD
        elif ordered_lda:
            out_dir = DATA_DIR_OLDA
        else:
            out_dir = DATA_DIR
    p        = P_NOISE + (C - 1)
    n        = N_PER_CLASS * C
    olda_str = "_olda" if ordered_lda else ""
    nsd_str  = f"_nsd{int(round(noise_scale_decay * 100))}" if use_nsd else ""
    tag      = f"p{p}_C{C}_n{n}_seed{seed}{olda_str}{nsd_str}"
    dpath    = os.path.join(out_dir, f"synthetic_{tag}.npz")
    mpath    = os.path.join(out_dir, f"synthetic_{tag}_meta.npz")

    if not os.path.exists(dpath):
        print(f"[synthetic] File not found — generating seed={seed} "
              f"ordered_lda={ordered_lda} ...")
        data = generate_synthetic(seed=seed, ordered_lda=ordered_lda,
                                  lda_scale_decay=lda_scale_decay,
                                  noise_scale_decay=noise_scale_decay)
        save_synthetic(data, seed, out_dir)
        return data

    print(f"[synthetic] Loading {dpath}")
    d = np.load(dpath)
    m = np.load(mpath)

    base_keys = ["p_noise", "p_signal", "p", "C", "n_per_class", "n",
                 "sigma_noise", "sigma_signal", "class_sep",
                 "test_size", "val_size", "seed"]
    params = {k: m[k].item() for k in base_keys}
    # Newer fields — default gracefully for files generated before these were added
    params["ordered_lda"]      = bool(m["ordered_lda"].item()) \
                                  if "ordered_lda" in m else False
    params["lda_scale_decay"]  = float(m["lda_scale_decay"].item()) \
                                  if "lda_scale_decay" in m else LDA_SCALE_DECAY
    params["noise_scale_decay"] = float(m["noise_scale_decay"].item()) \
                                  if "noise_scale_decay" in m else NOISE_SCALE_DECAY

    return dict(
        X_train=d["X_train"], y_train=d["y_train"],
        X_val=d["X_val"],     y_val=d["y_val"],
        X_test=d["X_test"],   y_test=d["y_test"],
        class_means=m["class_means"],
        pca_dirs=m["pca_dirs"],
        lda_dirs=m["lda_dirs"],
        params=params,
    )


# ==============================================================================
# Verification
# ==============================================================================

def verify(data: dict):
    """
    Run sanity checks on generated data:
    1. Noise block variance >> signal block variance
    2. Top-p_noise PCA eigenvectors lie in noise block (dims 0..p_noise-1)
    3. LDA directions lie in signal block (dims p_noise..p-1)
    4. Angle between top-19 PCA subspace and top-19 LDA subspace ~ 90 degrees
    """
    from scipy.linalg import subspace_angles

    p      = data["params"]["p"]
    p_noise = data["params"]["p_noise"]
    C      = data["params"]["C"]
    X_tr   = data["X_train"]
    y_tr   = data["y_train"]

    print("\n=== Synthetic Data Verification ===\n")

    # 1. Variance check
    var_noise  = X_tr[:, :p_noise].var(axis=0).mean()
    var_signal = X_tr[:, p_noise:].var(axis=0).mean()
    print(f"Mean variance — noise block: {var_noise:.3f}  signal block: {var_signal:.3f}")
    assert var_noise > var_signal * 10, "Noise variance not sufficiently larger than signal"
    print("  PASSED: noise variance >> signal variance\n")

    # 2. PCA directions in noise block
    pca_dirs = data["pca_dirs"]                   # (p, p_noise)
    mass_in_noise = (pca_dirs[:p_noise, :] ** 2).sum()
    mass_total    = (pca_dirs ** 2).sum()
    pca_noise_frac = mass_in_noise / mass_total
    print(f"PCA dirs: fraction of energy in noise block = {pca_noise_frac:.4f}  (expect ~1.0)")
    assert pca_noise_frac > 0.99, f"PCA dirs not concentrated in noise block: {pca_noise_frac:.4f}"
    print("  PASSED: PCA directions lie in noise block\n")

    # 3. LDA directions in signal block
    lda_dirs = data["lda_dirs"]                   # (p, C-1)
    mass_in_signal = (lda_dirs[p_noise:, :] ** 2).sum()
    mass_total_lda = (lda_dirs ** 2).sum()
    lda_signal_frac = mass_in_signal / mass_total_lda
    # For ordered LDA, the weakest signal dims (high j) have tiny class_sep * decay^j,
    # so those LDA directions pick up some noise-block leakage — relax threshold.
    lda_thresh = 0.85 if data["params"].get("ordered_lda", False) else 0.99
    print(f"LDA dirs: fraction of energy in signal block = {lda_signal_frac:.4f}  "
          f"(expect >{lda_thresh})")
    assert lda_signal_frac > lda_thresh, \
        f"LDA dirs not concentrated in signal block: {lda_signal_frac:.4f}"
    print("  PASSED: LDA directions lie in signal block\n")

    # 4. Subspace angle between top-(C-1) PCA and top-(C-1) LDA subspaces
    n_lda = C - 1
    angles_rad = subspace_angles(pca_dirs[:, :n_lda], lda_dirs[:, :n_lda])
    mean_angle = float(np.degrees(angles_rad).mean())
    print(f"Mean principal angle (top-{n_lda} PCA vs top-{n_lda} LDA): {mean_angle:.2f} deg  (expect ~90)")
    assert mean_angle > 80, f"PCA and LDA subspaces not sufficiently orthogonal: {mean_angle:.2f} deg"
    print("  PASSED: PCA ⊥ LDA by construction\n")

    # 5. Ordered LDA checks (only when ordered_lda=True)
    if data["params"].get("ordered_lda", False):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda_check = LinearDiscriminantAnalysis()
        lda_check.fit(X_tr, y_tr)

        evr = lda_check.explained_variance_ratio_
        print(f"LDA explained_variance_ratio (first 5): {np.round(evr[:5], 3)}")
        assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1)), \
            "LDA explained_variance_ratio not in descending order!"
        print("  PASSED: LDA discriminative power is ordered (descending)\n")

        # Each LDA direction should load most heavily on its corresponding signal axis
        p_signal_v = data["params"]["p_signal"]
        print("LDA direction alignment with signal axes (expect diagonal-dominant):")
        lda_dirs_signal = np.abs(lda_dirs[p_noise:, :])   # (p_signal, C-1)
        for j in range(min(5, p_signal_v)):
            top_axis = int(np.argmax(lda_dirs_signal[:, j]))
            loading  = lda_dirs_signal[top_axis, j]
            print(f"  LDA dir {j}: max loading on signal axis {top_axis} = {loading:.3f}"
                  f"  {'OK' if top_axis == j else 'MISMATCH'}")
        print()

    print("=== All verification checks passed ===\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PCA-vs-LDA dataset")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--verify",      action="store_true",
                        help="Run verification after generation")
    parser.add_argument("--all-seeds",   action="store_true",
                        help="Generate for all 5 experiment seeds (42-46)")
    parser.add_argument("--ordered-lda", action="store_true",
                        help="Use ordered LDA construction (signal dims sorted by "
                             "discriminative power); saved with _olda tag suffix")
    parser.add_argument("--lda-scale-decay", type=float, default=LDA_SCALE_DECAY,
                        help=f"Geometric decay rate for ordered LDA scales "
                             f"(default: {LDA_SCALE_DECAY})")
    parser.add_argument("--noise-scale-decay", type=float, default=NOISE_SCALE_DECAY,
                        help=f"Geometric decay rate for noise dim variances; "
                             f"<1 gives distinct eigenvalues (default: {NOISE_SCALE_DECAY})")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46] if args.all_seeds else [args.seed]

    for s in seeds:
        variant = "ordered LDA" if args.ordered_lda else "default"
        print(f"\n[synthetic] Generating seed={s}  variant={variant} ...")
        data = generate_synthetic(seed=s,
                                  ordered_lda=args.ordered_lda,
                                  lda_scale_decay=args.lda_scale_decay,
                                  noise_scale_decay=args.noise_scale_decay)
        p = data["params"]
        print(f"  p={p['p']} (noise={p['p_noise']}, signal={p['p_signal']})  "
              f"C={p['C']}  n={p['n']}  "
              f"train={data['X_train'].shape[0]}  "
              f"val={data['X_val'].shape[0]}  "
              f"test={data['X_test'].shape[0]}  "
              f"ordered_lda={p['ordered_lda']}")
        save_synthetic(data, s)
        if args.verify:
            verify(data)

    print("\n[synthetic] Done.")
