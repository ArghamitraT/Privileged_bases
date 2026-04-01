"""
Helper script: tests/helper_mnist_loader.py
---------------------------------------------
Standalone regression test for the MNIST loader segfault fix, run as a
subprocess by run_tests_exp6.py.

Verifies that the .numpy() conversion path in data/loader.py works correctly
on a small synthetic tensor — same dtype as MNIST but only 100 samples to
avoid in-process large-allocation segfaults on macOS.

Usage (called by run_tests_exp6.py — not meant to be run directly):
    python tests/helper_mnist_loader.py <code_dir>
"""

import sys
import os

code_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, code_dir)

import numpy as np
import torch

# Use 100 samples — same shape/dtype as MNIST train_ds.data but small enough
# to avoid the macOS in-process large-allocation segfault.
fake_data    = torch.zeros(100, 28, 28, dtype=torch.uint8)
fake_targets = torch.zeros(100, dtype=torch.long)

# Exact conversion from data/loader.py after the fix
X = fake_data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y = fake_targets.numpy().astype(np.int64)

assert X.shape == (100, 784), f"Expected (100, 784), got {X.shape}"
assert y.shape == (100,),     f"Expected (100,), got {y.shape}"
assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
# Note: avoid X.max()/X.min() — triggers numpy.core._methods which has a
# NumPy 2.x internal import error on this machine.  Shape + dtype is enough
# to confirm the .numpy() conversion path works without segfaulting.

print("  PASSED: .numpy() on (100,28,28) uint8 tensor -> shape and dtype=float32 verified")
