"""
Script: tests/run_tests_exp13.py
----------------------------------
Test runner for Experiment 13 — Supervised MRL on CD34.

Tests:
  1. test_metric_functions     — purity / compactness / separation on synthetic data
  2. test_data_split           — DataSplit shape, dtype, stratification
  3. test_extract_embeddings   — embedding extraction + fixed_lp reversal (subprocess)
  4. test_prefix_sweep         — prefix sweep returns correct structure and value ranges
  5. test_plots_no_crash       — prefix curve plots + training curves produce PNGs
  6. test_e2e_fast (slow)      — full --fast run, checks all output files are present
                                 (skipped with --fast flag to this runner)

NOTE: the h5ad data file is NOT required for tests 1–5 (synthetic data used).
      test_e2e_fast requires the real data file at DATA_PATH.

Usage:
    python tests/run_tests_exp13.py           # unit tests + e2e smoke test
    python tests/run_tests_exp13.py --fast    # unit tests only (skip e2e smoke)

Conda environment: mrl_env
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import tempfile
import argparse
import subprocess

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

from experiments.exp13_mrl_cd34_supervised import (
    _celltype_purity,
    _compactness,
    _separation,
    make_data_split,
    plot_prefix_curves,
    plot_training_curves,
    LEGEND,
    METRIC_LABELS,
)


# ==============================================================================
# Test 1: Metric functions on synthetic data
# ==============================================================================

def test_metric_functions():
    """Purity / compactness / separation on known synthetic clusters."""
    print("--- test_metric_functions ---")

    rng = np.random.default_rng(0)
    n_cells     = 400   # must be divisible by both n_clusters and n_classes
    n_clusters  = 4
    n_classes   = 4

    # Perfect clusters: each cluster has only one cell type
    cluster_labels = np.repeat(np.arange(n_clusters), n_cells // n_clusters)
    y_int          = cluster_labels.copy()   # cluster == class → purity = 1.0

    purity = _celltype_purity(cluster_labels, y_int)
    assert purity == 1.0, f"Expected purity=1.0 for perfect clusters, got {purity}"

    # Mixed clusters: uniform cell type distribution → purity = 1/n_classes
    y_mixed = np.tile(np.arange(n_classes), n_cells // n_classes)
    purity_mixed = _celltype_purity(cluster_labels, y_mixed)
    assert purity_mixed <= 1.0 / n_classes + 1e-6, (
        f"Mixed purity should be ~1/{n_classes}, got {purity_mixed}"
    )

    # Compactness: tight clusters (points at centroid) → should be 0
    Z_k = np.zeros((n_cells, 3), dtype=np.float32)
    compact = _compactness(Z_k, cluster_labels)
    assert compact == 0.0, f"Compactness of zero-matrix should be 0, got {compact}"

    # Compactness: random data → positive value
    Z_rand = rng.standard_normal((n_cells, 3)).astype(np.float32)
    compact_rand = _compactness(Z_rand, cluster_labels)
    assert compact_rand > 0.0, f"Compactness of random data should be >0"

    # Separation: centroids at known positions → check is positive
    Z_sep = np.zeros((n_cells, 2), dtype=np.float32)
    for c in range(n_clusters):
        mask = cluster_labels == c
        Z_sep[mask] = np.array([c * 10.0, 0.0])
    sep = _separation(Z_sep, cluster_labels)
    assert sep > 0.0, f"Separation should be positive for spread centroids"
    # Nearest-centroid distance should be ~10.0
    assert abs(sep - 10.0) < 1e-3, f"Expected sep≈10.0, got {sep}"

    print("  purity (perfect)  : PASSED")
    print("  purity (mixed)    : PASSED")
    print("  compactness (zero): PASSED")
    print("  compactness (rand): PASSED")
    print("  separation        : PASSED")
    print("  PASSED\n")


# ==============================================================================
# Test 2: DataSplit shape and dtype
# ==============================================================================

def test_data_split():
    """make_data_split returns correctly shaped tensors with expected dtypes."""
    print("--- test_data_split ---")

    import torch

    rng     = np.random.default_rng(1)
    n       = 400
    n_feat  = 50
    n_cls   = 5
    X       = rng.standard_normal((n, n_feat)).astype(np.float32)
    y       = (np.arange(n) % n_cls).astype(np.int64)

    data = make_data_split(X, y, test_size=0.2, val_size=0.1, seed=42)

    total = len(data.X_train) + len(data.X_val) + len(data.X_test)
    assert total == n, f"Total samples mismatch: {total} != {n}"
    assert data.X_train.dtype == torch.float32, "X_train should be float32"
    assert data.y_train.dtype == torch.long,    "y_train should be long"
    assert data.input_dim == n_feat,            "input_dim mismatch"
    assert data.n_classes == n_cls,             "n_classes mismatch"

    # All classes should appear in train split (stratified)
    assert len(torch.unique(data.y_train)) == n_cls, (
        "All classes should appear in train split"
    )

    print(f"  train={len(data.X_train)}  val={len(data.X_val)}  "
          f"test={len(data.X_test)}")
    print("  PASSED\n")


# ==============================================================================
# Test 3: Embedding extraction (subprocess, keeps torch out of runner)
# ==============================================================================

def test_extract_embeddings():
    """Run embedding extraction in a subprocess; check shapes + fixed_lp reversal."""
    print("--- test_extract_embeddings ---")

    script = os.path.join(CODE_DIR, "tests", "helper_exp13_embed.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  FAILED\n{result.stdout}\n{result.stderr}")
        raise AssertionError("helper_exp13_embed.py exited non-zero")

    output = result.stdout
    assert "shape: OK"    in output, f"shape check missing:\n{output}"
    assert "reversal: OK" in output, f"reversal check missing:\n{output}"
    print("  PASSED\n")


# ==============================================================================
# Test 4: Prefix sweep structure and value ranges
# ==============================================================================

def test_prefix_sweep():
    """run_prefix_sweep returns correct keys, list lengths, and value ranges."""
    print("--- test_prefix_sweep ---")

    from experiments.exp13_mrl_cd34_supervised import run_prefix_sweep

    rng = np.random.default_rng(2)
    n, d, n_cl, n_cls = 300, 8, 10, 4

    Z = rng.standard_normal((n, d)).astype(np.float32)
    y = (np.arange(n) % n_cls).astype(np.int64)

    embeddings    = {"mrl": Z, "pca": Z * 0.5}
    eval_prefixes = list(range(1, d + 1))

    results = run_prefix_sweep(embeddings, y, eval_prefixes, n_clusters=n_cl, seed=0)

    assert set(results.keys()) == {"mrl", "pca"}, f"Unexpected keys: {results.keys()}"

    for tag in ["mrl", "pca"]:
        for metric in ["purity", "compactness", "separation"]:
            vals = results[tag][metric]
            assert len(vals) == len(eval_prefixes), (
                f"[{tag}][{metric}] length {len(vals)} != {len(eval_prefixes)}"
            )
            assert all(0.0 <= v <= 1.0 for v in vals) or metric != "purity", (
                f"purity values out of [0,1]: {vals}"
            )
            assert all(v >= 0.0 for v in vals), (
                f"[{tag}][{metric}] negative values: {vals}"
            )

    print("  Key structure  : PASSED")
    print("  Value ranges   : PASSED")
    print("  PASSED\n")


# ==============================================================================
# Test 5: Plots produce PNG files without crashing
# ==============================================================================

def test_plots_no_crash():
    """plot_prefix_curves and plot_training_curves write PNG files."""
    print("--- test_plots_no_crash ---")

    rng = np.random.default_rng(3)
    d, n_tags = 8, 3
    eval_prefixes = list(range(1, d + 1))
    models = ["pca", "ce", "mrl"]

    results = {
        tag: {
            "purity":      rng.uniform(0.2, 0.9, size=d).tolist(),
            "compactness": rng.uniform(0.1, 0.5, size=d).tolist(),
            "separation":  rng.uniform(0.5, 2.0, size=d).tolist(),
        }
        for tag in models
    }
    seacell_ref = {"purity": 0.85, "compactness": 0.20, "separation": 1.50}

    histories = {
        "ce":  {"train_losses": [1.0, 0.8, 0.6], "val_losses": [1.1, 0.9, 0.7]},
        "mrl": {"train_losses": [1.2, 0.9, 0.7], "val_losses": [1.3, 1.0, 0.8]},
    }

    with tempfile.TemporaryDirectory() as tmp:
        stamp = "_test"

        plot_prefix_curves(results, seacell_ref, eval_prefixes, models, tmp, stamp)
        for metric in ["purity", "compactness", "separation"]:
            path = os.path.join(tmp, f"prefix_{metric}_curve{stamp}.png")
            assert os.path.exists(path), f"Missing: {path}"

        plot_training_curves(histories, models, tmp, stamp)
        tc_path = os.path.join(tmp, f"training_curves{stamp}.png")
        assert os.path.exists(tc_path), f"Missing: {tc_path}"

    print("  prefix_purity_curve    : PASSED")
    print("  prefix_compactness_curve: PASSED")
    print("  prefix_separation_curve : PASSED")
    print("  training_curves         : PASSED")
    print("  PASSED\n")


# ==============================================================================
# Test 6: End-to-end --fast run (slow; skipped with --fast to this runner)
# ==============================================================================

def test_e2e_fast():
    """
    Run the full experiment with --fast and verify all required output files exist.
    Requires the real CD34 h5ad data file to be present at DATA_PATH.
    """
    print("--- test_e2e_fast ---")

    from experiments.exp13_mrl_cd34_supervised import DATA_PATH
    if not os.path.exists(DATA_PATH):
        print(f"  SKIPPED: data file not found: {DATA_PATH}")
        return

    script = os.path.join(CODE_DIR, "experiments", "exp13_mrl_cd34_supervised.py")
    result = subprocess.run(
        [sys.executable, script, "--fast"],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"  FAILED\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}")
        raise AssertionError("exp13 --fast exited non-zero")

    # Find the run directory from stdout
    run_dir = None
    for line in result.stdout.splitlines():
        if "[exp13] Output:" in line:
            run_dir = line.split("[exp13] Output:")[-1].strip()
            break

    assert run_dir and os.path.isdir(run_dir), (
        f"Could not find run_dir in output:\n{result.stdout[-2000:]}"
    )

    required = [
        "training_curves",
        "prefix_purity_curve",
        "prefix_compactness_curve",
        "prefix_separation_curve",
        "umap_comparison",
        "cd34_embeddings.npz",
        "results_summary.txt",
        "experiment_description.log",
        "runtime.txt",
        "code_snapshot",
    ]

    missing = []
    for fname in required:
        # Accept both exact name and timestamped variants
        found = any(
            f.startswith(fname) or f == fname
            for f in os.listdir(run_dir)
        )
        if not found:
            missing.append(fname)

    if missing:
        print(f"  Missing output files: {missing}")
        print(f"  Found: {os.listdir(run_dir)}")
        raise AssertionError(f"Missing outputs: {missing}")

    print(f"  Run dir: {run_dir}")
    print(f"  All required outputs present.")
    print("  PASSED\n")


# ==============================================================================
# Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Run unit tests only; skip e2e smoke test")
    args = parser.parse_args()

    tests = [
        test_metric_functions,
        test_data_split,
        test_extract_embeddings,
        test_prefix_sweep,
        test_plots_no_crash,
    ]
    if not args.fast:
        tests.append(test_e2e_fast)

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}\n")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
