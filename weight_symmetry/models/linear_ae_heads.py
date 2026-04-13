"""
Script: weight_symmetry/models/linear_ae_heads.py
--------------------------------------------------
Linear AE encoder + per-prefix classifier heads for CE-loss experiments.

Architecture:
    Encoder  B ∈ R^{d×p}  :  z = Bx
    Decoder  A ∈ R^{p×d}  :  x̂ = Az  (kept for compatibility; unused in CE loss)
    Heads    W_m ∈ R^{C×m}:  logits_m = W_m z_{1:m},  m = 1..d

The encoder learns prefix-ordered directions that are optimal for classification
at every prefix size simultaneously (when trained with FullPrefixCELoss).

Usage:
    from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads
    model = LinearAEWithHeads(input_dim=69, embed_dim=32, n_classes=20)
    logits = model.classify_prefix(x, m=8)  # (batch, n_classes)
"""

import torch
import torch.nn as nn


class LinearAEWithHeads(nn.Module):
    """
    Linear autoencoder encoder + per-prefix classifier heads.

    Args:
        input_dim (int): Input dimensionality p.
        embed_dim (int): Latent dimensionality d.
        n_classes (int): Number of target classes C.
    """

    def __init__(self, input_dim: int, embed_dim: int, n_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        # B ∈ R^{d×p}
        self.encoder = nn.Linear(input_dim, embed_dim, bias=False)
        # A ∈ R^{p×d} (unused in CE loss but kept for API compatibility)
        self.decoder = nn.Linear(embed_dim, input_dim, bias=False)
        # Per-prefix heads: W_m ∈ R^{C×m} for m = 1..d
        self.heads = nn.ModuleList([
            nn.Linear(m, n_classes, bias=True)
            for m in range(1, embed_dim + 1)
        ])

        self._init_orthonormal()

    def _init_orthonormal(self):
        with torch.no_grad():
            W = self.decoder.weight.data       # (p, d)
            Q, R = torch.linalg.qr(W)
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1.0
            self.decoder.weight.data = Q * signs.unsqueeze(0)

    def encode_prefix(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """z_{1:m} = B_{1:m} x.  x: (batch, p) -> (batch, m)"""
        return x @ self.encoder.weight[:m].T

    def classify_prefix(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Logits from m-dim prefix.  x: (batch, p) -> (batch, n_classes)"""
        z_m = self.encode_prefix(x, m)
        return self.heads[m - 1](z_m)

    def get_encoder_matrix(self) -> torch.Tensor:
        """Return B = encoder.weight, shape (d, p). Detached."""
        return self.encoder.weight.detach()

    def get_decoder_matrix(self) -> torch.Tensor:
        """Return A = decoder.weight, shape (p, d). Detached."""
        return self.decoder.weight.detach()


# ==============================================================================
# Sanity check
# ==============================================================================

if __name__ == "__main__":
    p, d, C, batch = 69, 32, 20, 16
    model = LinearAEWithHeads(p, d, C)
    x = torch.randn(batch, p)
    y = torch.randint(0, C, (batch,))

    # Prefix encode shapes
    for m in [1, 8, 19, 32]:
        z   = model.encode_prefix(x, m)
        log = model.classify_prefix(x, m)
        assert z.shape   == (batch, m),  f"encode_prefix shape: {z.shape}"
        assert log.shape == (batch, C),  f"classify_prefix shape: {log.shape}"
    print("Shapes  PASSED")

    # Gradient flows through encoder + heads
    import torch.nn.functional as F
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = sum(F.cross_entropy(model.classify_prefix(x, m), y)
               for m in range(1, d + 1)) / d
    loss.backward()
    opt.step()
    assert model.encoder.weight.grad is not None
    print("Gradient flow  PASSED")

    print("\nAll LinearAEWithHeads checks passed.")
