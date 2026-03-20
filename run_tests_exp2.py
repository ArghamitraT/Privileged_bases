"""
Script: run_tests_exp2.py
--------------------------
Test runner for Experiment 2 — Cluster Visualization.

Tests:
  1. Unit tests for each helper function (embedding extraction, cluster metrics,
     dimensionality reduction, subsampling).
  2. End-to-end smoke test of the full experiment using the --fast flag
     (digits dataset, 5 epochs, small subsample).

Usage:
    python run_tests_exp2.py           # unit tests + e2e smoke test
    python run_tests_exp2.py --fast    # unit tests only (skip e2e smoke)

Inputs:  None (uses internal test data)
Outputs: PASS / FAIL messages printed to stdout; non-zero exit on failure.
"""

import os
import sys
import argparse
import numpy as np
import torch

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExpConfig
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head

# Import all helpers we want to test from the experiment module
from experiments.exp2_cluster_viz import (
    subsample_indices,
    extract_prefix,
    reduce_to_2d,
    compute_cluster_metrics,
    get_neural_embeddings,
    get_pca_full_embeddings,
)


# ==============================================================================
# Unit tests
# ==============================================================================

def test_subsample_indices():
    """subsample_indices returns correct length and valid range."""
    print("--- test_subsample_indices ---")

    # Case: n > max_n — should subsample
    idx = subsample_indices(n=1000, max_n=200, seed=0)
    assert len(idx) == 200, f"Expected 200, got {len(idx)}"
    assert idx.min() >= 0 and idx.max() < 1000, "Indices out of range"

    # Case: n <= max_n — should return all
    idx2 = subsample_indices(n=50, max_n=200, seed=0)
    assert len(idx2) == 50, f"Expected 50, got {len(idx2)}"
    assert list(idx2) == list(range(50)), "Should return 0..49"

    print("  PASSED\n")


def test_extract_prefix():
    """extract_prefix slices correct columns; guards against k > embed_dim."""
    print("--- test_extract_prefix ---")

    emb = np.random.randn(100, 64).astype(np.float32)

    # Normal slice
    p = extract_prefix(emb, 8)
    assert p.shape == (100, 8), f"Expected (100, 8), got {p.shape}"
    assert np.allclose(p, emb[:, :8]), "Values should match first 8 cols"

    # k equals embed_dim — return all
    p_full = extract_prefix(emb, 64)
    assert p_full.shape == (100, 64)

    # k > embed_dim — should be clamped to embed_dim
    p_big = extract_prefix(emb, 100)
    assert p_big.shape == (100, 64), "Should clamp to embed_dim"

    print("  PASSED\n")


def test_reduce_to_2d_k1():
    """reduce_to_2d with k=1 adds jitter column and returns (n, 2)."""
    print("--- test_reduce_to_2d k=1 ---")

    emb = np.random.randn(50, 1).astype(np.float32)
    out = reduce_to_2d(emb, method="t-SNE", seed=0)
    assert out.shape == (50, 2), f"Expected (50, 2), got {out.shape}"
    # x-column should match original embedding
    assert np.allclose(out[:, 0], emb[:, 0]), "x col should be original embedding"

    print("  PASSED\n")


def test_reduce_to_2d_k2():
    """reduce_to_2d with k=2 returns a copy without any reduction."""
    print("--- test_reduce_to_2d k=2 ---")

    emb = np.random.randn(50, 2).astype(np.float32)
    out = reduce_to_2d(emb, method="t-SNE", seed=0)
    assert out.shape == (50, 2), f"Expected (50, 2), got {out.shape}"
    assert np.allclose(out, emb), "k=2 should return values unchanged"

    print("  PASSED\n")


def test_reduce_to_2d_tsne():
    """reduce_to_2d with t-SNE produces (n, 2) output for k>2."""
    print("--- test_reduce_to_2d t-SNE k=8 ---")

    np.random.seed(0)
    emb = np.random.randn(60, 8).astype(np.float32)
    out = reduce_to_2d(emb, method="t-SNE", seed=42, n_iter_tsne=250, perplexity=10)
    assert out.shape == (60, 2), f"Expected (60, 2), got {out.shape}"
    assert np.isfinite(out).all(), "t-SNE output contains non-finite values"

    print("  PASSED\n")


def test_compute_cluster_metrics_shape():
    """compute_cluster_metrics returns correct keys and in-range values."""
    print("--- test_compute_cluster_metrics ---")

    np.random.seed(0)
    # Two well-separated clusters
    emb = np.vstack([
        np.random.randn(100, 8) + np.array([5] * 8),
        np.random.randn(100, 8) - np.array([5] * 8),
    ]).astype(np.float32)
    labels = np.array([0] * 100 + [1] * 100)

    m = compute_cluster_metrics(emb, labels, max_samples=200, seed=0)

    assert set(m.keys()) == {"silhouette", "intra", "inter", "separation"}, \
        f"Unexpected keys: {m.keys()}"
    assert -1.0 <= m["silhouette"] <= 1.0, f"Silhouette out of range: {m['silhouette']}"
    assert m["intra"] >= 0, f"Intra distance should be non-negative: {m['intra']}"
    assert m["inter"] >= 0, f"Inter distance should be non-negative: {m['inter']}"
    assert m["separation"] >= 0, f"Separation ratio should be non-negative"

    # Well-separated clusters should yield positive silhouette
    assert m["silhouette"] > 0.5, \
        f"Expected high silhouette for well-separated clusters, got {m['silhouette']:.4f}"

    print(f"  silhouette={m['silhouette']:.4f}  separation={m['separation']:.3f}")
    print("  PASSED\n")


def test_compute_cluster_metrics_degenerate():
    """compute_cluster_metrics handles single-class and tiny inputs gracefully."""
    print("--- test_compute_cluster_metrics degenerate cases ---")

    emb  = np.random.randn(10, 4).astype(np.float32)
    labs = np.zeros(10, dtype=np.int64)  # all same class

    m = compute_cluster_metrics(emb, labs, max_samples=100, seed=0)
    assert m["silhouette"] == 0.0, "Single class should give silhouette=0"

    print("  PASSED\n")


def test_get_neural_embeddings():
    """get_neural_embeddings returns correct shape from a small config."""
    print("--- test_get_neural_embeddings ---")

    cfg  = ExpConfig(dataset="digits", embed_dim=16, eval_prefixes=[1, 2, 4, 8, 16], epochs=1)
    data = load_data(cfg)
    encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)

    emb = get_neural_embeddings(encoder, data)
    assert emb.shape == (len(data.y_test), cfg.embed_dim), \
        f"Expected ({len(data.y_test)}, {cfg.embed_dim}), got {emb.shape}"
    assert isinstance(emb, np.ndarray), "Should return numpy array"

    # L2 norm should be ≈ 1 (encoder applies F.normalize)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"Embeddings should be unit-normalised; norms range: {norms.min():.4f}–{norms.max():.4f}"

    print(f"  Embeddings shape: {emb.shape}  norms: {norms.min():.4f}–{norms.max():.4f}")
    print("  PASSED\n")


def test_get_pca_embeddings():
    """get_pca_full_embeddings returns correct shape and finite values."""
    print("--- test_get_pca_full_embeddings ---")

    cfg  = ExpConfig(dataset="digits", embed_dim=16, eval_prefixes=[1, 2, 4, 8, 16], epochs=1)
    data = load_data(cfg)

    pca_emb = get_pca_full_embeddings(data, cfg)
    n_test  = len(data.y_test)
    n_components = min(cfg.embed_dim, data.input_dim)

    assert pca_emb.shape == (n_test, n_components), \
        f"Expected ({n_test}, {n_components}), got {pca_emb.shape}"
    assert np.isfinite(pca_emb).all(), "PCA embeddings contain non-finite values"

    print(f"  PCA embeddings shape: {pca_emb.shape}")
    print("  PASSED\n")


# ==============================================================================
# End-to-end smoke test
# ==============================================================================

def test_e2e_smoke():
    """
    Full end-to-end smoke test using --fast flag (digits, 5 epochs, small subsample).
    Verifies the experiment runs to completion and produces expected output files.
    """
    print("--- e2e smoke test (--fast, no weights) ---")
    import subprocess
    import glob

    script = os.path.join(os.path.dirname(__file__), "experiments", "exp2_cluster_viz.py")

    result = subprocess.run(
        [sys.executable, script, "--fast"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout[-3000:])
        print("STDERR:\n", result.stderr[-3000:])
        raise AssertionError(f"exp2 --fast exited with code {result.returncode}")

    # Find the most-recently created run directory
    from utility import get_path
    results_base = get_path("files/results")
    run_dirs = sorted(glob.glob(os.path.join(results_base, "exprmnt_*")))
    assert run_dirs, "No run directory was created"
    latest = run_dirs[-1]

    # Verify mandatory output files
    required = [
        "experiment_description.log",
        "tsne_grid.png",
        "cluster_metrics.png",
        "combined_summary.png",
        "results_summary.txt",
        "runtime.txt",
    ]
    for fname in required:
        path = os.path.join(latest, fname)
        assert os.path.isfile(path), f"Missing output file: {fname}"
        assert os.path.getsize(path) > 0, f"Output file is empty: {fname}"

    print(f"  Output directory: {latest}")
    print(f"  All required files present: {required}")
    print("  PASSED\n")


# ==============================================================================
# Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast", action="store_true",
        help="Run unit tests only — skip the slow end-to-end smoke test.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("run_tests_exp2.py — Experiment 2 test suite")
    print("=" * 60 + "\n")

    # --- Unit tests (always run) ---
    print("Unit tests\n" + "-" * 40)
    test_subsample_indices()
    test_extract_prefix()
    test_reduce_to_2d_k1()
    test_reduce_to_2d_k2()
    test_reduce_to_2d_tsne()
    test_compute_cluster_metrics_shape()
    test_compute_cluster_metrics_degenerate()
    test_get_neural_embeddings()
    test_get_pca_embeddings()

    if args.fast:
        print("=" * 60)
        print("All unit tests PASSED  (e2e smoke test skipped with --fast)")
        print("=" * 60)
        return

    # --- End-to-end smoke test ---
    print("End-to-end smoke test\n" + "-" * 40)
    test_e2e_smoke()

    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
