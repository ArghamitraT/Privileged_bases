"""
Helper: tests/helper_exp12_loss.py
------------------------------------
Subprocess helper for test_vector_learned_p_loss in run_tests_exp12.py.

Checks:
  - p is a vector of shape (embed_dim,)
  - all values > 1 and <= 1 + p_max
  - forward pass produces finite scalar loss
  - backward: p_raw.grad is a vector, all entries non-zero
"""

import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

import torch
import torch.nn as nn
from losses.mat_loss import VectorLearnedPrefixLpLoss

embed_dim  = 8
batch_size = 16
n_classes  = 10
p_init     = 0.0
p_max      = 10.0

loss_fn = VectorLearnedPrefixLpLoss(
    embed_dim=embed_dim, lambda_l1=0.05, p_init=p_init, p_max=p_max
)
head = nn.Linear(embed_dim, n_classes)

z = torch.randn(batch_size, embed_dim, requires_grad=True)
y = torch.randint(0, n_classes, (batch_size,))

# p shape and range
p_val = loss_fn.p
assert p_val.shape == (embed_dim,), f"p shape wrong: {p_val.shape}"
assert (p_val > 1.0).all(),              f"some p <= 1: {p_val}"
assert (p_val <= 1.0 + p_max).all(),     f"some p > {1+p_max}: {p_val}"
print(f"p_shape: OK  (shape={tuple(p_val.shape)}, all in (1, {1+p_max}])")

# Forward
loss = loss_fn(z, y, head)
assert loss.ndim == 0,          f"Loss not scalar: {loss.shape}"
assert torch.isfinite(loss),    f"Loss not finite: {loss.item()}"
print(f"forward: OK  (loss={loss.item():.4f})")

# Backward
loss.backward()
assert z.grad is not None,                      "No grad on z"
assert loss_fn.p_raw.grad is not None,          "No grad on p_raw"
assert loss_fn.p_raw.grad.shape == (embed_dim,), \
    f"p_raw.grad shape wrong: {loss_fn.p_raw.grad.shape}"
assert (loss_fn.p_raw.grad != 0).all(), \
    f"Some p_raw gradients are zero: {loss_fn.p_raw.grad}"
print(f"backward: OK  (p_raw.grad shape={tuple(loss_fn.p_raw.grad.shape)}, all nonzero)")

print("All vector loss helper checks passed.")
