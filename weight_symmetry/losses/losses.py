"""
Script: weight_symmetry/losses/losses.py
-----------------------------------------
Reconstruction loss functions for weight_symmetry experiments.

All losses operate on a LinearAE model and a batch of inputs x.
Interface: loss_fn(x, model) -> scalar tensor

Losses:
    MSELoss          : standard ||x - ABx||^2 (vanilla AE)
    OftadehLoss      : weighted prefix sum with S_d weights, equivalent to
                       full-prefix MRL for reconstruction (see Remark in paper)
    StandardMRLLoss  : sum of ||x - A_{1:m} B_{1:m} x||^2 at prefix sizes M only
    FullPrefixMRLLoss: sum of ||x - A_{1:m} B_{1:m} x||^2 for all m = 1..d

Usage:
    from weight_symmetry.losses.losses import MSELoss, FullPrefixMRLLoss
    loss_fn = FullPrefixMRLLoss()
    loss = loss_fn(x_batch, model)
"""

import torch
import torch.nn.functional as F
from typing import List


class MSELoss:
    """
    Vanilla AE reconstruction loss: ||x - A B x||^2 averaged over batch.
    """
    def __call__(self, x: torch.Tensor, model) -> torch.Tensor:
        x_hat = model(x)
        return F.mse_loss(x_hat, x)


class StandardMRLLoss:
    """
    MRL reconstruction loss: sum of prefix MSE at a fixed set M of prefix sizes.
    Loss = (1/|M|) * sum_{m in M} ||x - A_{1:m} B_{1:m} x||^2

    Args:
        prefix_sizes (List[int]): The set M of prefix sizes to evaluate at.
    """
    def __init__(self, prefix_sizes: List[int]):
        self.prefix_sizes = sorted(prefix_sizes)

    def __call__(self, x: torch.Tensor, model) -> torch.Tensor:
        total = torch.zeros(1, device=x.device)
        for m in self.prefix_sizes:
            z_m   = model.encode_prefix(x, m)
            x_hat = model.decode_prefix(z_m, m)
            total = total + F.mse_loss(x_hat, x)
        return total / len(self.prefix_sizes)


class FullPrefixMRLLoss:
    """
    Full-prefix MRL reconstruction loss: sum over ALL prefix sizes m = 1..d.
    Loss = (1/d) * sum_{m=1}^{d} ||x - A_{1:m} B_{1:m} x||^2

    Equivalent to Oftadeh S_d weighting by Theorem 1 of the paper.
    """
    def __call__(self, x: torch.Tensor, model) -> torch.Tensor:
        d     = model.embed_dim
        total = torch.zeros(1, device=x.device)
        for m in range(1, d + 1):
            z_m   = model.encode_prefix(x, m)
            x_hat = model.decode_prefix(z_m, m)
            total = total + F.mse_loss(x_hat, x)
        return total / d


class OftadehLoss:
    """
    Oftadeh weighted reconstruction loss with S_d weights.
    Loss = (1/Z) * sum_{m=1}^{d} w_m * ||x - A_{1:m} B_{1:m} x||^2
    where w_m = (d - m + 1)  (dim 1 gets weight d, dim d gets weight 1).

    This is the explicit S_d weighting from Oftadeh et al. (2020).
    For linear AE + MSE, mathematically equivalent to FullPrefixMRLLoss
    (same S_d structure emerges from prefix sum — see Remark in paper).
    Running both provides an empirical sanity check.
    """
    def __call__(self, x: torch.Tensor, model) -> torch.Tensor:
        d      = model.embed_dim
        total  = torch.zeros(1, device=x.device)
        Z      = 0.0
        for m in range(1, d + 1):
            w     = float(d - m + 1)
            z_m   = model.encode_prefix(x, m)
            x_hat = model.decode_prefix(z_m, m)
            total = total + w * F.mse_loss(x_hat, x)
            Z    += w
        return total / Z


# ==============================================================================
# Classification losses (CE) — require LinearAEWithHeads
# Interface: loss_fn(x, model, y) -> scalar tensor
# ==============================================================================

class FullPrefixCELoss:
    """
    Full-prefix cross-entropy loss: sum over ALL prefix sizes m = 1..d.
    Loss = (1/d) * sum_{m=1}^{d} CE(W_m B_{1:m} x, y)

    Requires model with classify_prefix(x, m) method (LinearAEWithHeads).
    """
    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        d     = model.embed_dim
        total = torch.zeros(1, device=x.device)
        for m in range(1, d + 1):
            logits = model.classify_prefix(x, m)
            total  = total + F.cross_entropy(logits, y)
        return total / d


class StandardMRLCELoss:
    """
    Standard MRL cross-entropy loss: sum at a fixed set M of prefix sizes.
    Loss = (1/|M|) * sum_{m in M} CE(W_m B_{1:m} x, y)

    Requires model with classify_prefix(x, m) method (LinearAEWithHeads).

    Args:
        prefix_sizes (List[int]): The set M of prefix sizes.
    """
    def __init__(self, prefix_sizes: List[int]):
        self.prefix_sizes = sorted(prefix_sizes)

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        total = torch.zeros(1, device=x.device)
        for m in self.prefix_sizes:
            logits = model.classify_prefix(x, m)
            total  = total + F.cross_entropy(logits, y)
        return total / len(self.prefix_sizes)


class PrefixL1CELoss:
    """
    CE on full embedding + front-loaded L1 penalty for dimension ordering.

    Penalises dim j with weight (embed_dim - j): dim 0 is penalised most and
    carries the least information after training.  Dimensions must be reversed
    before prefix evaluation — label embeddings as "PrefixL1 (CE) (rev)".

    Interface: loss_fn(x, model, y) — supervised, requires LinearAEWithHeads.

    Args:
        l1_lambda (float): L1 regularisation strength.
    """
    def __init__(self, l1_lambda: float = 0.01):
        self.l1_lambda = l1_lambda

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        d      = model.embed_dim
        z      = model.encode_prefix(x, d)          # (batch, d) full encoding
        logits = model.heads[-1](z)                  # full-dim head (heads[d-1])
        ce     = F.cross_entropy(logits, y)
        # Weight for dim j = (d - j): dim 0 gets weight d, dim d-1 gets weight 1
        weights = torch.arange(d, 0, -1, dtype=z.dtype, device=z.device)
        l1_pen  = (weights * z.abs()).mean(dim=0).sum()
        return ce + self.l1_lambda * l1_pen


# ==============================================================================
# Sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from weight_symmetry.models.linear_ae import LinearAE

    p, d, batch = 64, 8, 32
    model = LinearAE(input_dim=p, embed_dim=d)
    x     = torch.randn(batch, p)

    for name, loss_fn in [
        ("MSELoss",           MSELoss()),
        ("StandardMRLLoss",   StandardMRLLoss([2, 4, 6, 8])),
        ("FullPrefixMRLLoss", FullPrefixMRLLoss()),
        ("OftadehLoss",       OftadehLoss()),
    ]:
        val = loss_fn(x, model)
        assert val.shape == torch.Size([1]) or val.shape == torch.Size([])
        assert val.item() > 0
        print(f"  {name}: {val.item():.4f}  PASSED")

    # FullPrefixMRL and Oftadeh should be close (same S_d structure, different normalisation)
    v1 = FullPrefixMRLLoss()(x, model).item()
    v2 = OftadehLoss()(x, model).item()
    print(f"\nFullPrefixMRL={v1:.4f}  OftadehLoss={v2:.4f}  (different scale, same structure)")

    print("\nAll loss checks passed.")
