"""
Script: models/linear_ae.py
-----------------------------
Linear Autoencoder with optional orthogonality constraint.

Used in Experiment 6 to test whether Orthogonal + Matryoshka reconstruction
loss recovers PCA eigenvectors and their ordering.

Architecture:
    Encoder: W ∈ R^{embed_dim × input_dim}  (nn.Linear, no bias)
    Decoder: W^T  (tied weights — no separate decoder layer)
    Forward: x̂ = W^T (W x)

Orthogonality enforcement (call after each optimizer.step()):
    QR decomposition of W^T: W^T = QR
    Set W = Q^T  so that W W^T = I  (rows of W are orthonormal)

Matryoshka prefix support:
    encode_prefix(x, k): project onto first k rows of W only
    decode_prefix(z, k): reconstruct from first k dims using first k rows of W^T

Inputs:
    input_dim (int) : number of raw input features
    embed_dim (int) : number of embedding dimensions (latent size)

Outputs:
    torch.Tensor of shape (batch_size, input_dim) — reconstructed input

Usage:
    from models.linear_ae import LinearAutoencoder
    model = LinearAutoencoder(input_dim=784, embed_dim=64)
    python models/linear_ae.py   # smoke test (forward pass, orthogonalize, prefix ops)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAutoencoder(nn.Module):
    """
    Linear autoencoder with tied weights and optional Stiefel-manifold projection.

    The encoder weight W has shape (embed_dim, input_dim). Rows of W are the
    learned directions in input space. After orthogonalize(), W W^T = I
    (rows are orthonormal), making the model analogous to a PCA projection.

    Args:
        input_dim (int): Dimensionality of the input data.
        embed_dim (int): Dimensionality of the latent embedding.
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()

        # Encoder weight W: shape (embed_dim, input_dim)
        # No bias — PCA has no bias (data should be pre-centered)
        self.encoder = nn.Linear(input_dim, embed_dim, bias=False)

        # Store dims for reference
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Initialise with orthonormal rows via QR decomposition
        with torch.no_grad():
            W = self.encoder.weight.data  # (embed_dim, input_dim)
            Q, _ = torch.linalg.qr(W.T)  # Q: (input_dim, embed_dim)
            self.encoder.weight.data = Q.T  # (embed_dim, input_dim)

        print(f"[linear_ae] LinearAutoencoder: input_dim={input_dim}, embed_dim={embed_dim}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input onto all embed_dim directions.

        Args:
            x (torch.Tensor): Shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Shape (batch_size, embed_dim).
        """
        # x @ W^T  =  x @ encoder.weight.T  (standard nn.Linear forward)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from full embedding using tied weights (W^T).

        Args:
            z (torch.Tensor): Shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Shape (batch_size, input_dim).
        """
        # z @ W  =  F.linear(z, encoder.weight)  — note: no .T needed here
        # because F.linear computes z @ weight.T, and weight is (embed_dim, input_dim),
        # so weight.T is (input_dim, embed_dim), giving z @ W.T = (batch, input_dim). Wait—
        # We want z @ W where W is (embed_dim, input_dim), giving (batch, input_dim).
        # F.linear(z, A) computes z @ A.T. So F.linear(z, encoder.weight.T) gives z @ W.
        # Simpler: just use torch.mm directly.
        return z @ self.encoder.weight   # (batch, embed_dim) @ (embed_dim, input_dim)

    def encode_prefix(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Project input onto only the first k directions (Matryoshka prefix).

        Args:
            x (torch.Tensor): Shape (batch_size, input_dim).
            k (int)          : Number of prefix dimensions to use. Must be <= embed_dim.

        Returns:
            torch.Tensor: Shape (batch_size, k).
        """
        assert k <= self.embed_dim, f"k={k} exceeds embed_dim={self.embed_dim}"
        # Use only first k rows of W: shape (k, input_dim)
        return x @ self.encoder.weight[:k].T   # (batch, input_dim) @ (input_dim, k)

    def decode_prefix(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """
        Reconstruct input from a k-dimensional prefix embedding.

        Args:
            z (torch.Tensor): Shape (batch_size, k) — prefix embedding.
            k (int)          : Number of prefix dimensions. Must match z.shape[1].

        Returns:
            torch.Tensor: Shape (batch_size, input_dim).
        """
        assert z.shape[1] == k, f"z has {z.shape[1]} dims but k={k}"
        # Use only first k rows of W: shape (k, input_dim)
        return z @ self.encoder.weight[:k]     # (batch, k) @ (k, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode then decode using all embed_dim dimensions.

        Args:
            x (torch.Tensor): Shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Reconstructed input, shape (batch_size, input_dim).
        """
        z = self.encode(x)
        return self.decode(z)

    def orthogonalize(self):
        """
        Project encoder weight rows onto the Stiefel manifold via QR decomposition.

        After this call, W W^T = I  (rows of W are orthonormal).
        Sign ambiguity is resolved by ensuring the diagonal of R is positive.

        This should be called after every optimizer.step() when using the
        orthogonality constraint.
        """
        with torch.no_grad():
            W = self.encoder.weight.data   # (embed_dim, input_dim)

            # QR decomposition of W^T: shape (input_dim, embed_dim)
            Q, R = torch.linalg.qr(W.T)   # Q: (input_dim, embed_dim), R: (embed_dim, embed_dim)

            # Fix sign ambiguity: make diagonal of R positive
            # so the decomposition is unique and consistent across steps
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1.0        # avoid multiplying by 0
            Q = Q * signs.unsqueeze(0)     # broadcast: (input_dim, embed_dim)

            # Set W = Q^T: rows of W are now orthonormal
            self.encoder.weight.data = Q.T  # (embed_dim, input_dim)


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.insert(0, "..")

    input_dim = 64
    embed_dim = 10
    batch_size = 32

    print("--- Building LinearAutoencoder ---")
    model = LinearAutoencoder(input_dim=input_dim, embed_dim=embed_dim)
    print(f"  encoder.weight shape: {model.encoder.weight.shape}")
    print()

    # Checkpoint 1: forward pass shapes
    print("--- Checkpoint 1: forward pass shapes ---")
    x = torch.randn(batch_size, input_dim)
    x_hat = model(x)
    assert x_hat.shape == (batch_size, input_dim), f"Expected ({batch_size}, {input_dim}), got {x_hat.shape}"
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {x_hat.shape}")
    print("  PASSED\n")

    # Checkpoint 2: encode/decode shapes
    print("--- Checkpoint 2: encode / decode shapes ---")
    z = model.encode(x)
    assert z.shape == (batch_size, embed_dim)
    x_rec = model.decode(z)
    assert x_rec.shape == (batch_size, input_dim)
    print(f"  encode: {x.shape} -> {z.shape}")
    print(f"  decode: {z.shape} -> {x_rec.shape}")
    print("  PASSED\n")

    # Checkpoint 3: encode_prefix / decode_prefix shapes
    print("--- Checkpoint 3: prefix encode/decode shapes ---")
    for k in [1, 3, 5, 10]:
        zk = model.encode_prefix(x, k)
        assert zk.shape == (batch_size, k), f"encode_prefix k={k}: expected ({batch_size},{k}), got {zk.shape}"
        xk = model.decode_prefix(zk, k)
        assert xk.shape == (batch_size, input_dim), f"decode_prefix k={k}: wrong shape"
    print(f"  encode_prefix + decode_prefix for k in [1,3,5,10]: PASSED\n")

    # Checkpoint 4: orthogonalize makes W W^T ≈ I
    print("--- Checkpoint 4: orthogonalize → W W^T ≈ I ---")
    # Corrupt the weights first to make it non-orthogonal
    model.encoder.weight.data = torch.randn(embed_dim, input_dim)
    WWT_before = model.encoder.weight @ model.encoder.weight.T
    off_diag_before = (WWT_before - torch.eye(embed_dim)).abs().mean().item()

    model.orthogonalize()

    WWT_after = model.encoder.weight @ model.encoder.weight.T
    off_diag_after = (WWT_after - torch.eye(embed_dim)).abs().mean().item()

    print(f"  Off-diagonal |WW^T - I| before: {off_diag_before:.4f}")
    print(f"  Off-diagonal |WW^T - I| after:  {off_diag_after:.6f}")
    assert off_diag_after < 1e-5, f"Orthogonalize failed: residual={off_diag_after}"
    print("  PASSED\n")

    # Checkpoint 5: gradient flows through forward pass
    print("--- Checkpoint 5: gradient flow ---")
    x_grad = torch.randn(batch_size, input_dim)
    x_hat_grad = model(x_grad)
    loss = F.mse_loss(x_hat_grad, x_grad)
    loss.backward()
    assert model.encoder.weight.grad is not None, "No gradient on encoder weight"
    print(f"  Gradient norm: {model.encoder.weight.grad.norm().item():.4f}")
    print("  PASSED\n")

    print("All LinearAutoencoder checks passed.")
