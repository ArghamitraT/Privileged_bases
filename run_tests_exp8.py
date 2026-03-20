"""
Script: run_tests_exp8.py
--------------------------
Test runner for Experiment 8 — Per-Dimension Importance Scoring.

Tests:
  1. test_compute_importance_scores     — shape, range, informative dim ranks higher
  2. test_compute_best_vs_first_k       — keys, accs in [0,1], best_k >= first_k
  3. test_compute_method_agreement      — 3 pairs, rho in [-1,1], identical = 1.0
  4. test_get_pca_embeddings_np         — correct shape, no NaN, centered mean
  5. test_importance_scores_degenerate  — all-zero column does not crash
  6. test_plot_functions_no_crash       — all 4 plots run, PNGs created
  7. test_e2e_fast (slow)               — full --fast run, all output files present

Usage:
    python run_tests_exp8.py           # unit tests + e2e smoke test
    python run_tests_exp8.py --fast    # unit tests only (skip e2e smoke)

Inputs:  None (uses internal synthetic data / digits dataset)
Outputs: PASS / FAIL messages printed to stdout; non-zero exit on failure.
"""

import os

# Must be set before any numpy/scipy/sklearn imports to prevent BLAS deadlocks on macOS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import tempfile
import argparse
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExpConfig
from data.loader import load_data

from experiments.exp8_dim_importance import (
    compute_importance_scores,
    compute_best_vs_first_k,
    compute_method_agreement,
    get_pca_embeddings_np,
    plot_importance_scores,
    plot_dim_importance_heatmap,
    plot_best_vs_first_k,
    plot_method_agreement,
    MODEL_NAMES,
    IMPORTANCE_METHODS,
    METHOD_PAIRS,
)


# ==============================================================================
# Unit tests
# ==============================================================================

def test_compute_importance_scores():
    """
    compute_importance_scores returns correct shapes and value ranges.
    For perfectly separable data, probe_acc[0] > probe_acc[1]
    (informative dim should rank higher than noise dim).
    """
    print("--- test_compute_importance_scores ---")

    # Build 2D embeddings: dim 0 = class * 10 (separating), dim 1 = noise
    rng = np.random.default_rng(0)
    n_cls, n_per = 5, 30
    embed_dim    = 2

    Z_train = np.vstack([
        np.column_stack([
            np.full(n_per, i * 10.0),
            rng.standard_normal(n_per) * 0.1,
        ])
        for i in range(n_cls)
    ]).astype(np.float32)
    y_train = np.repeat(np.arange(n_cls), n_per)

    Z_test = np.vstack([
        np.column_stack([
            np.full(10, i * 10.0),
            rng.standard_normal(10) * 0.1,
        ])
        for i in range(n_cls)
    ]).astype(np.float32)
    y_test = np.repeat(np.arange(n_cls), 10)

    scores = compute_importance_scores(
        Z_test=Z_test, Z_train=Z_train,
        y_train=y_train, y_test=y_test,
        max_probe_samples=500, seed=42, model_tag="test",
    )

    assert scores["mean_abs"].shape  == (embed_dim,), "mean_abs wrong shape"
    assert scores["variance"].shape  == (embed_dim,), "variance wrong shape"
    assert scores["probe_acc"].shape == (embed_dim,), "probe_acc wrong shape"

    assert np.all(scores["mean_abs"]  >= 0), "mean_abs must be >= 0"
    assert np.all(scores["variance"]  >= 0), "variance must be >= 0"
    assert np.all(scores["probe_acc"] >= 0), "probe_acc must be >= 0"
    assert np.all(scores["probe_acc"] <= 1), "probe_acc must be <= 1"

    # Informative dim (0) should rank higher than noise dim (1)
    assert scores["probe_acc"][0] > scores["probe_acc"][1], (
        f"Expected probe_acc[0] > probe_acc[1], "
        f"got {scores['probe_acc'][0]:.4f} vs {scores['probe_acc'][1]:.4f}"
    )

    print(f"  mean_abs={scores['mean_abs'].tolist()}")
    print(f"  probe_acc={scores['probe_acc'].tolist()}")
    print("  PASSED\n")


def test_compute_best_vs_first_k():
    """
    compute_best_vs_first_k returns dict with correct keys and valid accuracies.
    best_k_acc >= first_k_acc always (oracle selection never worse than prefix).
    At k=embed_dim, all dims are used in both cases -> accs should be equal.
    """
    print("--- test_compute_best_vs_first_k ---")

    rng = np.random.default_rng(1)
    embed_dim       = 4
    n_train, n_test = 100, 50
    eval_prefixes   = [1, 2, 4]

    Z_train = rng.standard_normal((n_train, embed_dim)).astype(np.float32)
    Z_test  = rng.standard_normal((n_test,  embed_dim)).astype(np.float32)
    y_train = rng.integers(0, 3, n_train).astype(np.int64)
    y_test  = rng.integers(0, 3, n_test).astype(np.int64)

    importance_scores = {
        m: rng.random(embed_dim).astype(np.float32) for m in IMPORTANCE_METHODS
    }

    results = compute_best_vs_first_k(
        Z_train=Z_train, Z_test=Z_test,
        y_train=y_train, y_test=y_test,
        importance_scores=importance_scores,
        eval_prefixes=eval_prefixes,
        seed=42, model_tag="test",
    )

    expected_keys = {"first_k", "best_k_mean_abs", "best_k_variance", "best_k_probe_acc"}
    assert set(results.keys()) == expected_keys, \
        f"Missing keys: {set(results.keys())} != {expected_keys}"

    for key, acc_dict in results.items():
        assert set(acc_dict.keys()) == set(eval_prefixes), \
            f"Wrong prefixes for {key}"
        for k, acc in acc_dict.items():
            assert 0.0 <= acc <= 1.0, f"Accuracy out of range for {key}[{k}]: {acc}"

    # Oracle (best_k) should never be worse than first_k
    for method in IMPORTANCE_METHODS:
        for k in eval_prefixes:
            first = results["first_k"][k]
            best  = results[f"best_k_{method}"][k]
            assert best >= first - 1e-6, (
                f"best_k_{method}[{k}]={best:.4f} < first_k[{k}]={first:.4f}"
            )

    # At k=embed_dim all dims used in both -> should be equal
    for method in IMPORTANCE_METHODS:
        first_all = results["first_k"][embed_dim]
        best_all  = results[f"best_k_{method}"][embed_dim]
        assert abs(first_all - best_all) < 1e-4, (
            f"At k=embed_dim: first_k={first_all:.4f} != best_k_{method}={best_all:.4f}"
        )

    print(f"  result keys: {list(results.keys())}")
    print("  PASSED\n")


def test_compute_method_agreement():
    """
    compute_method_agreement returns Spearman rho in [-1, 1] for all 3 pairs.
    Identical scores -> rho = 1.0. Reversed scores -> rho = -1.0.
    """
    print("--- test_compute_method_agreement ---")

    d           = 10
    scores_base = np.arange(d, dtype=np.float32)

    # mean_abs == variance -> rho = 1.0
    # probe_acc = reversed -> rho = -1.0 with both
    importance_scores = {
        "mean_abs":  scores_base.copy(),
        "variance":  scores_base.copy(),
        "probe_acc": scores_base[::-1].copy(),
    }

    agreement = compute_method_agreement(importance_scores, model_tag="test")

    assert set(agreement.keys()) == set(METHOD_PAIRS), \
        f"Wrong pairs: {set(agreement.keys())}"

    for pair, rho in agreement.items():
        assert -1.0 - 1e-4 <= rho <= 1.0 + 1e-4, f"rho={rho} out of [-1,1] for {pair}"

    rho_same = agreement[("mean_abs", "variance")]
    assert abs(rho_same - 1.0) < 1e-4, f"Expected rho=1.0, got {rho_same:.6f}"

    rho_anti = agreement[("mean_abs", "probe_acc")]
    assert abs(rho_anti - (-1.0)) < 1e-4, f"Expected rho=-1.0, got {rho_anti:.6f}"

    print(f"  agreement={agreement}")
    print("  PASSED\n")


def test_get_pca_embeddings_np():
    """
    get_pca_embeddings_np returns arrays of correct shape with no NaN.
    Train embeddings should be approximately zero-centered (PCA centers data).
    """
    print("--- test_get_pca_embeddings_np ---")

    cfg = ExpConfig(
        dataset="digits", embed_dim=10,
        eval_prefixes=[1, 2, 4, 8, 10], epochs=1,
    )
    data = load_data(cfg)

    Z_train, Z_test = get_pca_embeddings_np(data, cfg)

    assert Z_train.shape[1] == cfg.embed_dim, \
        f"Expected {cfg.embed_dim} cols, got {Z_train.shape[1]}"
    assert Z_test.shape[1]  == cfg.embed_dim, \
        f"Expected {cfg.embed_dim} cols, got {Z_test.shape[1]}"
    assert not np.any(np.isnan(Z_train)), "NaN in Z_train"
    assert not np.any(np.isnan(Z_test)),  "NaN in Z_test"

    train_mean = Z_train.mean(axis=0)
    assert np.max(np.abs(train_mean)) < 0.5, \
        f"Train mean not near zero: max|mean|={np.max(np.abs(train_mean)):.4f}"

    print(f"  Z_train.shape={Z_train.shape}  Z_test.shape={Z_test.shape}")
    print(f"  max|train_mean|={np.max(np.abs(train_mean)):.4f}")
    print("  PASSED\n")


def test_importance_scores_degenerate_dim():
    """
    If a dimension is all-zero, compute_importance_scores must not crash.
    mean_abs[2] and variance[2] should be 0; probe_acc[2] should be in [0,1].
    """
    print("--- test_importance_scores_degenerate_dim ---")

    rng       = np.random.default_rng(2)
    embed_dim = 4
    n         = 60

    Z       = rng.standard_normal((n, embed_dim)).astype(np.float32)
    Z[:, 2] = 0.0  # degenerate column
    y       = rng.integers(0, 3, n).astype(np.int64)

    scores = compute_importance_scores(
        Z_test=Z[:30], Z_train=Z[30:],
        y_train=y[30:], y_test=y[:30],
        max_probe_samples=50, seed=42, model_tag="degenerate_test",
    )

    assert scores["mean_abs"][2]  == 0.0, \
        f"mean_abs[2] should be 0, got {scores['mean_abs'][2]}"
    assert scores["variance"][2]  == 0.0, \
        f"variance[2] should be 0, got {scores['variance'][2]}"
    assert 0.0 <= scores["probe_acc"][2] <= 1.0, \
        f"probe_acc[2] out of [0,1]: {scores['probe_acc'][2]}"

    print(f"  mean_abs[2]={scores['mean_abs'][2]}  "
          f"variance[2]={scores['variance'][2]}  "
          f"probe_acc[2]={scores['probe_acc'][2]:.4f}")
    print("  PASSED\n")


def test_plot_functions_no_crash():
    """
    All 4 plot functions complete without error on minimal synthetic data.
    Verifies each expected PNG is created in a temp directory.
    """
    print("--- test_plot_functions_no_crash ---")

    cfg = ExpConfig(
        dataset="digits", embed_dim=4,
        eval_prefixes=[1, 2, 4], epochs=1,
    )

    rng       = np.random.default_rng(3)
    embed_dim = 4

    all_scores = {
        model: {
            "mean_abs":  rng.random(embed_dim).astype(np.float32),
            "variance":  rng.random(embed_dim).astype(np.float32),
            "probe_acc": rng.random(embed_dim).astype(np.float32),
        }
        for model in MODEL_NAMES
    }

    all_gap_results = {
        model: {
            "first_k":          {k: float(rng.random()) for k in [1, 2, 4]},
            "best_k_mean_abs":  {k: float(rng.random()) for k in [1, 2, 4]},
            "best_k_variance":  {k: float(rng.random()) for k in [1, 2, 4]},
            "best_k_probe_acc": {k: float(rng.random()) for k in [1, 2, 4]},
        }
        for model in MODEL_NAMES
    }

    all_agreement = {
        model: {pair: float(rng.uniform(-1, 1)) for pair in METHOD_PAIRS}
        for model in MODEL_NAMES
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_importance_scores(all_scores, tmpdir, cfg)
        plot_dim_importance_heatmap(all_scores, tmpdir, cfg)
        plot_best_vs_first_k(all_gap_results, tmpdir, cfg)
        plot_method_agreement(all_scores, all_agreement, tmpdir, cfg)

        for fname in [
            "importance_scores.png",
            "dim_importance_heatmap.png",
            "best_vs_first_k.png",
            "method_agreement.png",
        ]:
            fpath = os.path.join(tmpdir, fname)
            assert os.path.isfile(fpath), f"Missing: {fname}"
            print(f"  Created: {fname}")

    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_fast():
    """
    Run the full experiment with --fast flag and verify all output files exist.
    Trains digits models for ~5 epochs then runs all three analyses.
    """
    print("--- test_e2e_fast (end-to-end smoke test) ---")
    print("  Running: python experiments/exp8_dim_importance.py --fast")
    print("  (trains digits models for 5 epochs + runs analyses) ...")

    result = subprocess.run(
        [sys.executable, "experiments/exp8_dim_importance.py", "--fast"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-2000:])
        print("  STDERR:", result.stderr[-2000:])
        raise AssertionError(
            f"exp8 --fast failed with return code {result.returncode}"
        )

    # Find the most recently created run folder
    from utility import get_path
    results_dir = get_path("files/results")
    runs = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith("exprmnt_") and
        os.path.isdir(os.path.join(results_dir, d))
    ])
    assert runs, "No output folder found after e2e run"
    run_dir = os.path.join(results_dir, runs[-1])

    required_files = [
        "experiment_description.log",
        "training_curves.png",
        "importance_scores.png",
        "dim_importance_heatmap.png",
        "best_vs_first_k.png",
        "method_agreement.png",
        "results_summary.txt",
        "runtime.txt",
        "code_snapshot",
    ]
    for fname in required_files:
        fpath = os.path.join(run_dir, fname)
        assert os.path.exists(fpath), f"Missing output: {fname}"
        print(f"  Found: {fname}")

    print(f"  Run folder: {run_dir}")
    print("  PASSED\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="run_tests_exp8.py")
    parser.add_argument(
        "--fast", action="store_true",
        help="Run unit tests only; skip the end-to-end smoke test.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("run_tests_exp8.py — Experiment 8 test suite")
    print("=" * 60)

    print("\nUnit tests")
    print("-" * 40)
    test_compute_importance_scores()
    test_compute_best_vs_first_k()
    test_compute_method_agreement()
    test_get_pca_embeddings_np()
    test_importance_scores_degenerate_dim()
    test_plot_functions_no_crash()

    if not args.fast:
        print("\nEnd-to-end smoke test")
        print("-" * 40)
        test_e2e_fast()

    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
