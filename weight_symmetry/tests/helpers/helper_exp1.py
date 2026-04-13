"""
Helper: weight_symmetry/tests/helpers/helper_exp1.py
------------------------------------------------------
Subprocess-based unit tests for Exp 1 modules.
Run by run_tests_exp1.py via subprocess to avoid torch import issues in the
test runner process (Issue 3 in CLAUDE.md).

Tests:
  1. LinearAE forward / prefix / orthogonalize
  2. All four loss functions (shapes, values, gradients)
  3. train_ae completes 2 epochs on digits
  4. compute_pca_directions + metrics shapes
  5. --fast smoke test: exp1 runs end-to-end

Usage:
    python weight_symmetry/tests/helpers/helper_exp1.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch
import tempfile

from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.losses.losses import (
    MSELoss, StandardMRLLoss, FullPrefixMRLLoss, OftadehLoss
)
from weight_symmetry.training.trainer import train_ae
from weight_symmetry.data.loader import load_data
from weight_symmetry.evaluation.metrics import (
    compute_pca_directions, compute_all_prefix_metrics
)

PASS = "PASSED"
FAIL = "FAILED"


def test_linear_ae():
    print("--- Test: LinearAE ---")
    p, d, b = 64, 8, 16
    model = LinearAE(p, d)

    # Forward
    x     = torch.randn(b, p)
    x_hat = model(x)
    assert x_hat.shape == (b, p), f"forward shape: {x_hat.shape}"

    # Prefix encode/decode for every m
    for m in range(1, d + 1):
        z_m = model.encode_prefix(x, m)
        assert z_m.shape == (b, m)
        xr  = model.decode_prefix(z_m, m)
        assert xr.shape == (b, p)

    # Orthogonality after init
    A   = model.get_decoder_matrix().numpy()
    err = np.abs(A.T @ A - np.eye(d)).max()
    assert err < 1e-5, f"Init ortho err: {err}"

    # Orthogonalize after corruption
    model.decoder.weight.data = torch.randn(p, d)
    model.orthogonalize()
    A   = model.get_decoder_matrix().numpy()
    err = np.abs(A.T @ A - np.eye(d)).max()
    assert err < 1e-5, f"Post-ortho err: {err}"

    print(f"  {PASS}")


def test_losses():
    print("--- Test: Loss functions ---")
    p, d, b = 64, 8, 16
    model   = LinearAE(p, d)
    x       = torch.randn(b, p, requires_grad=False)

    for name, loss_fn in [
        ("MSELoss",           MSELoss()),
        ("StandardMRLLoss",   StandardMRLLoss([2, 4, 6, 8])),
        ("FullPrefixMRLLoss", FullPrefixMRLLoss()),
        ("OftadehLoss",       OftadehLoss()),
    ]:
        val = loss_fn(x, model)
        assert val.item() > 0, f"{name} loss is not positive"

        # Gradient flows
        model2 = LinearAE(p, d)
        opt    = torch.optim.SGD(model2.parameters(), lr=0.01)
        opt.zero_grad()
        loss_fn(x, model2).backward()
        opt.step()
        assert model2.encoder.weight.grad is not None, f"{name} no grad on encoder"

    print(f"  {PASS}")


def test_trainer():
    print("--- Test: train_ae (2 epochs) ---")
    data = load_data("digits", seed=42)
    cfg  = dict(epochs=2, lr=1e-3, batch_size=32, patience=10, seed=42,
                embed_dim=8, weight_decay=0.0)

    with tempfile.TemporaryDirectory() as run_dir:
        model = LinearAE(data.input_dim, 8)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        h     = train_ae(model, MSELoss(), opt, data, cfg, run_dir, "test_mse", False)
        assert len(h["train_losses"]) == 2
        assert h["best_epoch"] in [0, 1]

        # With orthogonalize
        model2 = LinearAE(data.input_dim, 8)
        opt2   = torch.optim.Adam(model2.parameters(), lr=1e-3)
        h2     = train_ae(model2, FullPrefixMRLLoss(), opt2, data, cfg,
                          run_dir, "test_fp_ortho", True)
        A  = model2.get_decoder_matrix().numpy()
        err = np.abs(A.T @ A - np.eye(8)).max()
        assert err < 1e-4, f"Ortho not enforced after training: {err}"

    print(f"  {PASS}")


def test_metrics():
    print("--- Test: PCA metrics ---")
    data = load_data("digits", seed=42)
    d    = 8
    p    = data.input_dim

    U = compute_pca_directions(data.X_train, d)
    assert U.shape == (p, d)
    err = np.abs(U.T @ U - np.eye(d)).max()
    assert err < 1e-6, f"PCA dirs not orthonormal: {err}"

    model = LinearAE(p, d)
    results = compute_all_prefix_metrics(model, U)
    assert len(results["prefix_sizes"])      == d
    assert len(results["subspace_angles"])   == d
    assert len(results["column_alignments"]) == d
    assert all(0 <= a <= 90 for a in results["subspace_angles"])
    assert all(0 <= a <= 1  for a in results["column_alignments"])

    print(f"  {PASS}")


def test_fast_smoke():
    print("--- Test: --fast smoke test ---")
    import subprocess
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "experiments", "exp1_pca_recovery.py"
    )
    result = subprocess.run(
        [sys.executable, script, "--fast"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"--fast smoke test failed (returncode={result.returncode})")
    print(f"  {PASS}")


if __name__ == "__main__":
    test_linear_ae()
    test_losses()
    test_trainer()
    test_metrics()
    if os.environ.get("SKIP_SMOKE") != "1":
        test_fast_smoke()
    print("\n=== All helper_exp1 tests passed ===")
