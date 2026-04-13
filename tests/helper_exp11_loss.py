"""
Helper: tests/helper_exp11_loss.py
------------------------------------
Subprocess helper for test_learned_prefix_lp_loss in run_tests_exp11.py.

Run by the test runner as a subprocess to keep torch out of the main process
(avoids macOS hang — see Known Issues in CLAUDE.md).

Checks:
  - forward pass produces a valid scalar loss
  - backward pass runs without error
  - p property returns value > 1 and <= 1 + p_max
  - p_raw.grad is non-zero after backward (gradient flows to p)
"""

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)

import torch
import torch.nn as nn
from losses.mat_loss import LearnedPrefixLpLoss

embed_dim  = 8
batch_size = 16
n_classes  = 10
p_init     = 0.0
p_max      = 10.0

loss_fn = LearnedPrefixLpLoss(
    embed_dim=embed_dim, lambda_l1=0.05, p_init=p_init, p_max=p_max
)
head = nn.Linear(embed_dim, n_classes)

z = torch.randn(batch_size, embed_dim, requires_grad=True)
y = torch.randint(0, n_classes, (batch_size,))

# Forward
loss = loss_fn(z, y, head)
assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
print("forward: OK")

# Backward
loss.backward()
assert z.grad is not None, "No gradient on z"
assert torch.any(torch.isfinite(z.grad)), "z.grad contains non-finite values"
print("backward: OK")

# p property: must be in (1.0, 1.0 + p_max]
p_val = loss_fn.p.item()
assert p_val > 1.0,         f"p must be > 1.0, got {p_val}"
assert p_val <= 1.0 + p_max, f"p must be <= {1.0+p_max}, got {p_val}"
print(f"p_property: OK  (p={p_val:.4f})")

# p_raw must have received a gradient
assert loss_fn.p_raw.grad is not None, "p_raw.grad is None — gradient did not flow to p"
assert loss_fn.p_raw.grad.item() != 0.0, \
    f"p_raw.grad is zero — loss may not depend on p: {loss_fn.p_raw.grad.item()}"
print(f"p_grad: OK  (p_raw.grad={loss_fn.p_raw.grad.item():.6f})")

print("All loss helper checks passed.")
