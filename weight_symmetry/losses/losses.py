"""
Script: weight_symmetry/losses/losses.py
-----------------------------------------
Reconstruction loss functions for weight_symmetry experiments.

All losses operate on a LinearAE model and a batch of inputs x.
Interface: loss_fn(x, model) -> scalar tensor

Losses:
    MSELoss             : standard ||x - ABx||^2 (vanilla AE)
    OftadehLoss         : weighted prefix sum with S_d weights, equivalent to
                          full-prefix MRL for reconstruction (see Remark in paper)
    StandardMRLLoss     : sum of ||x - A_{1:m} B_{1:m} x||^2 at prefix sizes M only
    FullPrefixMRLLoss   : sum of ||x - A_{1:m} B_{1:m} x||^2 for all m = 1..d
    FisherLoss          : -Tr((S_W + εI)^{-1} S_B) on full embedding (supervised)
    FullPrefixFisherLoss: sum of FisherLoss over all prefix sizes k=1..d (supervised)

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


class NonUniformL2Loss:
    """
    Bao et al. (NeurIPS 2020) non-uniform L2-regularised LAE loss.

        L = ||x - ABx||²_F + Σ_i λ_i (||B[i,:]||² + ||A[:,i]||²)

    With 0 < λ_1 < λ_2 < ... < λ_d and λ_d < σ²_d (the d-th PCA eigenvalue),
    Theorem 3 guarantees that the global minimum has ordered axis-aligned
    principal directions: row i of B = ±u_i (i-th PCA eigenvector).
    Reference: "Regularized linear autoencoders recover the principal
    components, eventually", NeurIPS 2020.

    Args:
        lambdas: 1-D tensor/array of length d, strictly increasing.
                 Calibrate λ_d < σ²_d from the data's PCA spectrum.
    """
    def __init__(self, lambdas):
        self.lambdas = torch.as_tensor(lambdas, dtype=torch.float32)

    def __call__(self, x: torch.Tensor, model) -> torch.Tensor:
        x_hat = model(x)
        recon = F.mse_loss(x_hat, x)
        A   = model.decoder.weight                 # (p, d)
        B   = model.encoder.weight                 # (d, p)
        lam = self.lambdas.to(B.device)            # (d,)
        reg_B = (lam * B.pow(2).sum(dim=1)).sum()  # Σ_i λ_i ||B[i,:]||²
        reg_A = (lam * A.pow(2).sum(dim=0)).sum()  # Σ_i λ_i ||A[:,i]||²
        return recon + reg_B + reg_A


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


class PrefixL1MSELoss:
    """
    MSE reconstruction + front-loaded L1 penalty on latent dimensions.

    Reconstruction analogue of PrefixL1CELoss:
        L = ||x - ABx||² + λ * sum_j (d - j) * |z_j|   (mean over batch)

    Dim 0 is penalised most heavily → carries least information after training.
    Dimensions must be reversed before prefix evaluation — label "PrefixL1 (rev)".

    Interface: loss_fn(x, model) — unsupervised, works with plain LinearAE.

    Args:
        l1_lambda (float): L1 regularisation strength.
    """
    def __init__(self, l1_lambda: float = 0.01):
        self.l1_lambda = l1_lambda

    def __call__(self, x: torch.Tensor, model) -> torch.Tensor:
        d     = model.embed_dim
        x_hat = model(x)
        recon = F.mse_loss(x_hat, x)
        z       = model.encode(x)                                          # (batch, d)
        weights = torch.arange(d, 0, -1, dtype=z.dtype, device=z.device)  # d, d-1, ..., 1
        l1_pen  = (weights * z.abs()).mean(dim=0).sum()
        return recon + self.l1_lambda * l1_pen


# ==============================================================================
# Classification losses (CE) — require LinearAEWithHeads
# Interface: loss_fn(x, model, y) -> scalar tensor
# ==============================================================================

class CELoss:
    """
    Plain cross-entropy on full embedding — no ordering mechanism.

    L = CE(W_d B x, y)

    Baseline for CE group: encoder directions are arbitrary (same role as
    MSE LAE in the reconstruction group).

    Requires model with classify_prefix(x, m) method (LinearAEWithHeads).
    """
    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        d      = model.embed_dim
        logits = model.classify_prefix(x, d)   # full-dim head
        return F.cross_entropy(logits, y)


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
# Fisher / LDA losses (supervised, plain LinearAE)
# Interface: loss_fn(x, model, y) -> scalar tensor
# ==============================================================================

class FisherLoss:
    """
    Fisher (LDA) criterion on the full d-dimensional encoder output.

    Maximises the ratio of between-class to within-class scatter in embedding space:
        L = -Tr((S_W + ε I)^{-1} S_B)

    where S_B and S_W are the between- and within-class scatter matrices of
    the batch embeddings z = B x.

    Recovers the span of the top-d LDA directions but imposes no ordering —
    any rotation within the optimal subspace is equally valid.

    Interface: loss_fn(x, model, y) — supervised, requires plain LinearAE.
    The model's decoder is unused during training (only encoder B is optimised).

    Args:
        eps (float): regularisation added to S_W for numerical stability.
    """
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def _fisher_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device  = z.device
        classes = y.unique()
        n       = z.shape[0]
        mean_all = z.mean(dim=0)              # (d,)

        S_B = torch.zeros(z.shape[1], z.shape[1], device=device)
        S_W = torch.zeros(z.shape[1], z.shape[1], device=device)

        for c in classes:
            mask  = (y == c)
            n_c   = mask.sum().float()
            z_c   = z[mask]
            mu_c  = z_c.mean(dim=0)
            diff  = (mu_c - mean_all).unsqueeze(1)   # (d, 1)
            S_B   = S_B + (n_c / n) * (diff @ diff.T)
            z_c_c = z_c - mu_c.unsqueeze(0)
            S_W   = S_W + (z_c_c.T @ z_c_c) / n

        reg = self.eps * torch.eye(z.shape[1], device=device)
        return -torch.trace(torch.linalg.solve(S_W + reg, S_B))

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        z = model.encode(x)    # (batch, d) — full embedding
        return self._fisher_loss(z, y)


class ExplicitFisherLoss:
    """
    Fisher loss with B appearing explicitly in the computation.

    Computes scatter matrices S_B(x) and S_W(x) in input space first,
    then sandwiches with B to get the d-dimensional versions:

        S_B(z) = B S_B(x) B^T
        S_W(z) = B S_W(x) B^T
        L = -Tr( (B S_W(x) B^T + εI)^{-1}  (B S_B(x) B^T) )

    Mathematically identical to FisherLoss but makes B explicit.
    Gradients flow through B in the sandwich products, not through z.

    Args:
        eps (float): regularisation added to B S_W B^T for numerical stability.
    """
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        device   = x.device
        B        = model.encoder.weight          # (d, p) — explicit B
        classes  = y.unique()
        n        = x.shape[0]
        mean_all = x.mean(dim=0)                 # (p,)

        # Scatter matrices in input space — (p, p) each
        S_B_x = torch.zeros(x.shape[1], x.shape[1], device=device)
        S_W_x = torch.zeros(x.shape[1], x.shape[1], device=device)

        for c in classes:
            mask  = (y == c)
            n_c   = mask.sum().float()
            x_c   = x[mask]                      # (n_c, p)
            mu_c  = x_c.mean(dim=0)              # (p,)
            diff  = (mu_c - mean_all).unsqueeze(1)  # (p, 1)
            S_B_x = S_B_x + (n_c / n) * (diff @ diff.T)
            x_c_c = x_c - mu_c.unsqueeze(0)     # (n_c, p)
            S_W_x = S_W_x + (x_c_c.T @ x_c_c) / n

        # Sandwich with B: project scatter matrices into embedding space
        B_SB_Bt = B @ S_B_x @ B.T               # (d, d)
        B_SW_Bt = B @ S_W_x @ B.T               # (d, d)

        reg = self.eps * torch.eye(B.shape[0], device=device)
        return -torch.trace(torch.linalg.solve(B_SW_Bt + reg, B_SB_Bt))


class FullPrefixFisherLoss:
    """
    Full-prefix Fisher loss: sum of Fisher criterion over ALL prefix sizes k = 1..d.

    L = (1/d) * sum_{k=1}^{d} FisherLoss(B_{1:k} x, y)

    Forces each prefix to maximally separate classes using only its k dimensions.
    Expected to recover LDA directions in prefix order: dim 1 ≈ top LDA direction,
    dim 2 ≈ second LDA direction, etc.

    Interface: loss_fn(x, model, y) — supervised, requires plain LinearAE.

    Args:
        eps (float): regularisation added to S_W for numerical stability.
    """
    def __init__(self, eps: float = 1e-4):
        self.eps       = eps
        self._base     = FisherLoss(eps=eps)

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        d     = model.embed_dim
        total = torch.zeros(1, device=x.device)
        for k in range(1, d + 1):
            z_k   = model.encode_prefix(x, k)   # (batch, k)
            total = total + self._base._fisher_loss(z_k, y)
        return total / d


class PrefixL1FisherLoss:
    """
    Fisher (LDA) criterion on full embedding + front-loaded L1 penalty.

        L = FisherLoss(z_full, y) + λ * sum_j (d - j) * mean(|z_j|)

    Dim 0 is penalised most → carries least discriminative information after
    training.  Reverse dims before prefix eval — label "PrefixL1 Fisher (rev)".

    Interface: loss_fn(x, model, y) — supervised, requires plain LinearAE.

    Args:
        l1_lambda (float): L1 regularisation strength.
        eps (float): regularisation added to S_W in Fisher criterion.
    """
    def __init__(self, l1_lambda: float = 0.01, eps: float = 1e-4):
        self.l1_lambda = l1_lambda
        self._fisher   = FisherLoss(eps=eps)

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        d       = model.embed_dim
        z       = model.encode(x)                                           # (batch, d)
        fl      = self._fisher._fisher_loss(z, y)
        weights = torch.arange(d, 0, -1, dtype=z.dtype, device=z.device)   # d, d-1, …, 1
        l1_pen  = (weights * z.abs()).mean(dim=0).sum()
        return fl + self.l1_lambda * l1_pen


class StandardMRLFisherLoss:
    """
    Standard MRL Fisher loss: sum of FisherLoss at a fixed set M of prefix sizes.

        L = (1/|M|) * sum_{m in M} FisherLoss(B_{1:m} x, y)

    Analogue of StandardMRLLoss using the Fisher criterion instead of MSE.
    Recovers LDA subspace at evaluated prefix sizes; no ordering guarantee between them.

    Interface: loss_fn(x, model, y) — supervised, requires plain LinearAE.

    Args:
        prefix_sizes (List[int]): The set M of prefix sizes to evaluate at.
        eps (float): regularisation added to S_W in Fisher criterion.
    """
    def __init__(self, prefix_sizes: List[int], eps: float = 1e-4):
        self.prefix_sizes = sorted(prefix_sizes)
        self._base        = FisherLoss(eps=eps)

    def __call__(self, x: torch.Tensor, model, y: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(1, device=x.device)
        for m in self.prefix_sizes:
            z_m   = model.encode_prefix(x, m)
            total = total + self._base._fisher_loss(z_m, y)
        return total / len(self.prefix_sizes)


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
        ("NonUniformL2Loss",  NonUniformL2Loss(lambdas=torch.linspace(0.1, 0.9, d))),
        ("PrefixL1MSELoss",   PrefixL1MSELoss(l1_lambda=0.01)),
        ("StandardMRLLoss",   StandardMRLLoss([2, 4, 6, 8])),
        ("FullPrefixMRLLoss", FullPrefixMRLLoss()),
        ("OftadehLoss",       OftadehLoss()),
    ]:
        val = loss_fn(x, model)
        assert val.shape == torch.Size([1]) or val.shape == torch.Size([])
        assert val.item() > 0
        print(f"  {name}: {val.item():.4f}  PASSED")

    # CE losses need LinearAEWithHeads
    from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads
    n_classes = 5
    model_h = LinearAEWithHeads(input_dim=p, embed_dim=d, n_classes=n_classes)
    y = torch.randint(0, n_classes, (batch,))
    for name, loss_fn in [
        ("CELoss",             CELoss()),
        ("FullPrefixCELoss",   FullPrefixCELoss()),
        ("StandardMRLCELoss",  StandardMRLCELoss([2, 4, 6, 8])),
        ("PrefixL1CELoss",     PrefixL1CELoss(l1_lambda=0.01)),
    ]:
        val = loss_fn(x, model_h, y)
        assert val.shape == torch.Size([1]) or val.shape == torch.Size([])
        assert val.item() > 0
        print(f"  {name}: {val.item():.4f}  PASSED")

    # FullPrefixMRL and Oftadeh should be close (same S_d structure, different normalisation)
    v1 = FullPrefixMRLLoss()(x, model).item()
    v2 = OftadehLoss()(x, model).item()
    print(f"\nFullPrefixMRL={v1:.4f}  OftadehLoss={v2:.4f}  (different scale, same structure)")

    # Fisher losses need labels
    n_classes = 5
    y = torch.randint(0, n_classes, (batch,))
    for name, loss_fn in [
        ("FisherLoss",            FisherLoss(eps=1e-4)),
        ("FullPrefixFisherLoss",  FullPrefixFisherLoss(eps=1e-4)),
        ("PrefixL1FisherLoss",    PrefixL1FisherLoss(l1_lambda=0.01, eps=1e-4)),
        ("StandardMRLFisherLoss", StandardMRLFisherLoss([2, 4, 6, 8], eps=1e-4)),
    ]:
        val = loss_fn(x, model, y)
        assert val.shape == torch.Size([1]) or val.shape == torch.Size([])
        assert val.item() < 0, f"Fisher loss should be negative (maximising): {val.item()}"
        print(f"  {name}: {val.item():.4f}  PASSED")

    print("\nAll loss checks passed.")
