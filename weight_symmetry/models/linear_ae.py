"""
Script: weight_symmetry/models/linear_ae.py
--------------------------------------------
Linear Autoencoder with separate (untied) encoder B and decoder A.

Matches the paper formulation:
    Encoder  B ∈ R^{d×p}  :  z = Bx
    Decoder  A ∈ R^{p×d}  :  x̂ = Az

Orthogonality constraint A^T A = I is enforced via QR projection:
    call model.orthogonalize() after each optimizer.step()

Prefix operations for MRL losses:
    encode_prefix(x, m)  -> z_{1:m}   using first m rows of B
    decode_prefix(z_m, m)-> x̂_m       using first m columns of A

Usage:
    from weight_symmetry.models.linear_ae import LinearAE
    model = LinearAE(input_dim=784, embed_dim=32)
    model.orthogonalize()   # call after each optimizer step if ortho=True
"""

import torch
import torch.nn as nn


class LinearAE(nn.Module):
    """
    Linear autoencoder with separate encoder B and decoder A (untied weights).

    Args:
        input_dim (int): Input dimensionality p.
        embed_dim (int): Latent dimensionality d.
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # B ∈ R^{d×p}: encoder weight shape (d, p) — nn.Linear(p, d) convention
        self.encoder = nn.Linear(input_dim, embed_dim, bias=False)
        # A ∈ R^{p×d}: decoder weight shape (p, d) — nn.Linear(d, p) convention
        self.decoder = nn.Linear(embed_dim, input_dim, bias=False)

        # Initialise decoder with orthonormal columns (A^T A = I)
        self._init_orthonormal()

    def _init_orthonormal(self):
        with torch.no_grad():
            W = self.decoder.weight.data   # shape (p, d)
            Q, R = torch.linalg.qr(W)     # Q: (p, d), R: (d, d)  [reduced QR]
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1.0
            self.decoder.weight.data = Q * signs.unsqueeze(0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """z = Bx.  x: (batch, p) -> z: (batch, d)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """x̂ = Az.  z: (batch, d) -> x̂: (batch, p)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode. x: (batch, p) -> x̂: (batch, p)"""
        return self.decode(self.encode(x))

    def encode_prefix(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """
        Encode using only first m rows of B.
        x: (batch, p) -> z_{1:m}: (batch, m)
        """
        assert m <= self.embed_dim
        # encoder.weight shape: (d, p); first m rows: (m, p)
        return x @ self.encoder.weight[:m].T   # (batch, p) @ (p, m) -> (batch, m)

    def decode_prefix(self, z_m: torch.Tensor, m: int) -> torch.Tensor:
        """
        Decode from m-dimensional prefix embedding using first m columns of A.
        z_m: (batch, m) -> x̂_m: (batch, p)
        """
        assert z_m.shape[1] == m
        # decoder.weight shape: (p, d); first m columns: (p, m)
        return z_m @ self.decoder.weight[:, :m].T  # (batch, m) @ (m, p) -> (batch, p)

    def orthogonalize(self):
        """
        Project decoder A onto Stiefel manifold via QR so that A^T A = I_d.
        Call after every optimizer.step() when using the orthogonality constraint.
        Sign is fixed so diagonal of R is positive (consistent across steps).
        """
        with torch.no_grad():
            W = self.decoder.weight.data       # (p, d)
            Q, R = torch.linalg.qr(W)         # Q: (p, d), R: (d, d)
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1.0
            self.decoder.weight.data = Q * signs.unsqueeze(0)

    def orthogonalize_encoder(self):
        """
        Project encoder B onto row-orthonormal form via QR so that B B^T = I_d.
        Equivalent to orthonormalizing the columns of B^T, then transposing back.
        Call after every optimizer.step() when using the encoder orthogonality constraint.

        Why: if the target is AB = UU^T with A = U (orthonormal columns), then the
        optimal encoder is B = U^T, which has orthonormal rows (B B^T = U^T U = I_d).
        """
        with torch.no_grad():
            W = self.encoder.weight.data       # (d, p)
            Q, R = torch.linalg.qr(W.T)       # Q: (p, d), R: (d, d)  [QR on B^T]
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1.0
            self.encoder.weight.data = (Q * signs.unsqueeze(0)).T  # back to (d, p)

    def get_decoder_matrix(self) -> torch.Tensor:
        """Return A = decoder.weight, shape (p, d). Detached numpy-ready."""
        return self.decoder.weight.detach()

    def get_encoder_matrix(self) -> torch.Tensor:
        """Return B = encoder.weight, shape (d, p). Detached numpy-ready."""
        return self.encoder.weight.detach()


# ==============================================================================
# Sanity check
# ==============================================================================

if __name__ == "__main__":
    import torch.nn.functional as F

    p, d, batch = 784, 32, 64
    model = LinearAE(input_dim=p, embed_dim=d)
    x = torch.randn(batch, p)

    # Forward pass shapes
    x_hat = model(x)
    assert x_hat.shape == (batch, p), f"Expected ({batch},{p}), got {x_hat.shape}"
    print(f"Forward pass: {x.shape} -> {x_hat.shape}  PASSED")

    # Prefix shapes
    for m in [1, 4, 16, 32]:
        z_m = model.encode_prefix(x, m)
        x_m = model.decode_prefix(z_m, m)
        assert z_m.shape == (batch, m)
        assert x_m.shape == (batch, p)
    print("Prefix encode/decode shapes  PASSED")

    # Orthogonality after init
    A = model.get_decoder_matrix()
    orth_err = (A.T @ A - torch.eye(d)).abs().max().item()
    assert orth_err < 1e-5, f"Init orthogonality error: {orth_err}"
    print(f"Init orthogonality (A^T A = I): max_err={orth_err:.2e}  PASSED")

    # Orthogonalize after corruption
    model.decoder.weight.data = torch.randn(p, d)
    model.orthogonalize()
    A = model.get_decoder_matrix()
    orth_err = (A.T @ A - torch.eye(d)).abs().max().item()
    assert orth_err < 1e-5, f"Post-orthogonalize error: {orth_err}"
    print(f"After orthogonalize: max_err={orth_err:.2e}  PASSED")

    # Gradient flow
    x.requires_grad_(False)
    x2 = torch.randn(batch, p, requires_grad=False)
    model2 = LinearAE(p, d)
    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = F.mse_loss(model2(x2), x2)
    loss.backward()
    opt.step()
    assert model2.encoder.weight.grad is not None
    print("Gradient flow  PASSED")

    print("\nAll LinearAE checks passed.")
