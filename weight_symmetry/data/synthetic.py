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

Data is saved to:
    Mat_embedding_hyperbole/data/synthetic_data/

Usage:
    Conda environment: mrl_env

    python weight_symmetry/data/synthetic.py                   # default params, seed 42
    python weight_symmetry/data/synthetic.py --seed 43
    python weight_symmetry/data/synthetic.py --verify           # generate + run verification
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

DATA_DIR = os.path.join(_PROJ_ROOT, "data", "synthetic_data")

# ==============================================================================
# Default parameters
# ==============================================================================
P_NOISE      = 50
C            = 20          # number of classes  →  p_signal = C - 1 = 19
SIGMA_NOISE  = 5.0         # std of noise block
SIGMA_SIGNAL = 0.1         # within-class std of signal block
CLASS_SEP    = 1.0         # scale of class means (distance from origin)
N_PER_CLASS  = 500         # samples per class  →  n_total = 10 000
TEST_SIZE    = 0.2
VAL_SIZE     = 0.1
# ==============================================================================


def generate_synthetic(
    p_noise:      int   = P_NOISE,
    C:            int   = C,
    sigma_noise:  float = SIGMA_NOISE,
    sigma_signal: float = SIGMA_SIGNAL,
    class_sep:    float = CLASS_SEP,
    n_per_class:  int   = N_PER_CLASS,
    test_size:    float = TEST_SIZE,
    val_size:     float = VAL_SIZE,
    seed:         int   = 42,
) -> dict:
    """
    Generate the synthetic PCA-vs-LDA divergence dataset.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test  : np.float32 / np.int64
        class_means   : (C, p_signal) array of true class means in signal block
        pca_train_dirs: (p, p_noise) top-p_noise PCA eigenvectors from X_train
        lda_dirs      : (p, C-1) LDA directions from X_train + y_train
        params        : dict of all generation parameters
    """
    rng       = np.random.default_rng(seed)
    p_signal  = C - 1          # = 19 for C=20
    p         = p_noise + p_signal   # = 69
    n         = n_per_class * C      # = 10 000

    # ------------------------------------------------------------------
    # Class means: C random unit vectors in R^{p_signal}, scaled by class_sep
    # Using C random vectors guarantees the between-class scatter spans R^{p_signal}
    # (C > p_signal = C-1, so with probability 1 they span the full space)
    # ------------------------------------------------------------------
    raw = rng.standard_normal((C, p_signal))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    class_means = (raw / norms) * class_sep     # (C, p_signal)

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    X_noise  = rng.normal(0.0, sigma_noise, (n, p_noise)).astype(np.float32)
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


def save_synthetic(data: dict, seed: int, out_dir: str = DATA_DIR):
    """Save generated dataset to out_dir as two .npz files."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"p{data['params']['p']}_C{data['params']['C']}_n{data['params']['n']}_seed{seed}"

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
    return data_path, meta_path


def load_synthetic(seed: int = 42, out_dir: str = DATA_DIR) -> dict:
    """
    Load pre-generated synthetic dataset. Auto-generates if file not found.
    Returns same dict as generate_synthetic().
    """
    p      = P_NOISE + (C - 1)
    n      = N_PER_CLASS * C
    tag    = f"p{p}_C{C}_n{n}_seed{seed}"
    dpath  = os.path.join(out_dir, f"synthetic_{tag}.npz")
    mpath  = os.path.join(out_dir, f"synthetic_{tag}_meta.npz")

    if not os.path.exists(dpath):
        print(f"[synthetic] File not found — generating seed={seed} ...")
        data = generate_synthetic(seed=seed)
        save_synthetic(data, seed, out_dir)
        return data

    print(f"[synthetic] Loading {dpath}")
    d = np.load(dpath)
    m = np.load(mpath)

    params = {k: m[k].item() for k in
              ["p_noise", "p_signal", "p", "C", "n_per_class", "n",
               "sigma_noise", "sigma_signal", "class_sep",
               "test_size", "val_size", "seed"]}

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
    print(f"LDA dirs: fraction of energy in signal block = {lda_signal_frac:.4f}  (expect ~1.0)")
    assert lda_signal_frac > 0.99, f"LDA dirs not concentrated in signal block: {lda_signal_frac:.4f}"
    print("  PASSED: LDA directions lie in signal block\n")

    # 4. Subspace angle between top-(C-1) PCA and top-(C-1) LDA subspaces
    n_lda = C - 1
    angles_rad = subspace_angles(pca_dirs[:, :n_lda], lda_dirs[:, :n_lda])
    mean_angle = float(np.degrees(angles_rad).mean())
    print(f"Mean principal angle (top-{n_lda} PCA vs top-{n_lda} LDA): {mean_angle:.2f} deg  (expect ~90)")
    assert mean_angle > 80, f"PCA and LDA subspaces not sufficiently orthogonal: {mean_angle:.2f} deg"
    print("  PASSED: PCA ⊥ LDA by construction\n")

    print("=== All verification checks passed ===\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PCA-vs-LDA dataset")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="Run verification after generation")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Generate for all 5 experiment seeds (42-46)")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46] if args.all_seeds else [args.seed]

    for s in seeds:
        print(f"\n[synthetic] Generating seed={s} ...")
        data = generate_synthetic(seed=s)
        p = data["params"]
        print(f"  p={p['p']} (noise={p['p_noise']}, signal={p['p_signal']})  "
              f"C={p['C']}  n={p['n']}  "
              f"train={data['X_train'].shape[0]}  "
              f"val={data['X_val'].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
        save_synthetic(data, s)
        if args.verify:
            verify(data)

    print("\n[synthetic] Done.")
