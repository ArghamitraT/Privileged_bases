"""
Helper script: tests/helper_linear_ae.py
-----------------------------------------
Standalone unit test for LinearAutoencoder run as a subprocess by
run_tests_exp6.py. Tests forward pass, encode/decode shapes, prefix ops,
orthogonalize(), and gradient flow.

Usage (called by run_tests_exp6.py — not meant to be run directly):
    python tests/helper_linear_ae.py <code_dir>
"""

import sys
import os

# code_dir is passed as the first argument so the subprocess can import modules
code_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, code_dir)

import numpy as np
import torch
import torch.nn.functional as F
from models.linear_ae import LinearAutoencoder

np.random.seed(0)
torch.manual_seed(0)

input_dim  = 64
embed_dim  = 10
batch_size = 32

model = LinearAutoencoder(input_dim=input_dim, embed_dim=embed_dim)
x     = torch.randn(batch_size, input_dim)

# Case 1: forward pass shape
x_hat = model(x)
assert x_hat.shape == (batch_size, input_dim), \
    f"Expected ({batch_size},{input_dim}), got {x_hat.shape}"
print(f"  PASSED: forward pass shape {x_hat.shape}")

# Case 2: encode / decode shapes
z     = model.encode(x)
assert z.shape == (batch_size, embed_dim)
x_rec = model.decode(z)
assert x_rec.shape == (batch_size, input_dim)
print(f"  PASSED: encode {x.shape} -> {z.shape}, decode -> {x_rec.shape}")

# Case 3: prefix shapes
for k in [1, 5, 10]:
    zk = model.encode_prefix(x, k)
    assert zk.shape == (batch_size, k), f"encode_prefix k={k}: wrong shape"
    xk = model.decode_prefix(zk, k)
    assert xk.shape == (batch_size, input_dim), f"decode_prefix k={k}: wrong shape"
print("  PASSED: encode_prefix/decode_prefix shapes for k in [1,5,10]")

# Case 4: orthogonalize makes W W^T ≈ I
model.encoder.weight.data = torch.randn(embed_dim, input_dim)
model.orthogonalize()
WWT      = model.encoder.weight @ model.encoder.weight.T
residual = (WWT - torch.eye(embed_dim)).abs().mean().item()
assert residual < 1e-5, f"orthogonalize residual too large: {residual:.6f}"
print(f"  PASSED: orthogonalize — W W^T residual = {residual:.2e}")

# Case 5: gradient flow
x2     = torch.randn(batch_size, input_dim)
x_hat2 = model(x2)
loss   = F.mse_loss(x_hat2, x2)
loss.backward()
assert model.encoder.weight.grad is not None, "No gradient on encoder weight"
print(f"  PASSED: gradient flows "
      f"(grad norm={model.encoder.weight.grad.norm().item():.4f})")
