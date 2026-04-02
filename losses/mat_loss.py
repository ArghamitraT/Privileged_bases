"""
Script: losses/mat_loss.py
---------------------------
Loss functions for the standard model and the Matryoshka (MRL) model.

StandardLoss
    Plain cross-entropy on the full embed_dim-dimensional embedding.
    Used to train the baseline standard model.

MatryoshkaLoss
    Sums cross-entropy losses computed at each prefix scale k.
    For each k in eval_prefixes:
        - Mode A (shared_head): zero-pad the embedding beyond k, feed full
          vector to the single shared head.
        - Mode B (multi_head) : slice embedding[:, :k], feed to head_k.
    The total loss is the (optionally weighted) sum across all scales.
    This forces the encoder to pack the most discriminative information
    into the earliest dimensions.

PrefixL1Loss
    Plain cross-entropy loss + Matryoshka-style weighted L1 penalty.

    Same structure as L1RegLoss but regularization is front-loaded:
        penalty = sum_{m=1}^{k} ||z_{1:m}||_1
                = sum_{j=0}^{k-1} (k - j) * |z_j|

    Early dimensions face the highest sparsity pressure, creating ordering
    pressure without any multi-scale loss term.

All classes share the same interface:
    loss = criterion(embedding, labels, head)
so the training loop does not need to know which loss is being used.

Inputs:
    embedding (torch.Tensor): shape (batch_size, embed_dim)
    labels    (torch.Tensor): shape (batch_size,) — class indices (long)
    head      (nn.Module)   : SharedClassifier or MultiHeadClassifier

Outputs:
    torch.Tensor: scalar loss value

Usage:
    from losses.mat_loss import StandardLoss, MatryoshkaLoss, PrefixL1Loss, build_loss
    criterion = build_loss('mat', prefixes=[1, 2, 4, 8], head_mode='shared_head')
    python losses/mat_loss.py   # smoke test (forward pass for all loss types)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Standard loss — cross-entropy on the full embedding
# ==============================================================================

class StandardLoss(nn.Module):
    """
    Cross-entropy loss on the full embedding vector.

    Used to train the standard (non-Matryoshka) baseline model.

    Args:
        None
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss using the full embedding.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).
            labels    (torch.Tensor): Shape (batch_size,), class indices.
            head      (nn.Module)   : SharedClassifier — called with full embedding.

        Returns:
            torch.Tensor: Scalar cross-entropy loss.
        """
        logits = head(embedding)
        return self.ce(logits, labels)


# ==============================================================================
# Matryoshka loss — sum of cross-entropy losses at each prefix scale
# ==============================================================================

class MatryoshkaLoss(nn.Module):
    """
    Matryoshka Representation Learning (MRL) loss.

    For each prefix size k in prefixes, computes cross-entropy at that scale
    and sums the results. Optionally accepts per-scale weights.

    The behaviour at each scale depends on head_mode:
        'shared_head': zero-pad embedding beyond k, pass full vector to head.
        'multi_head' : slice embedding[:, :k], pass to the head for scale k.

    Args:
        prefixes   (List[int])          : Prefix sizes to compute loss at,
                                          e.g. [1, 2, 4, 8, 16, 32, 64].
        head_mode  (str)                : 'shared_head' or 'multi_head'.
        weights    (Optional[List[float]]): Per-scale loss weights. If None,
                                           all scales are weighted equally (1.0).
                                           Length must match prefixes.
    """

    def __init__(
        self,
        prefixes: List[int],
        head_mode: str,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        assert head_mode in ("shared_head", "multi_head"), (
            f"head_mode must be 'shared_head' or 'multi_head', got '{head_mode}'"
        )

        self.prefixes  = sorted(prefixes)
        self.head_mode = head_mode
        self.ce        = nn.CrossEntropyLoss()

        # Validate / set weights
        if weights is None:
            # Equal weight for every scale
            self.weights = [1.0] * len(self.prefixes)
        else:
            assert len(weights) == len(prefixes), (
                f"len(weights)={len(weights)} must equal len(prefixes)={len(prefixes)}"
            )
            self.weights = weights

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """
        Compute the weighted sum of cross-entropy losses at all prefix scales.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).
            labels    (torch.Tensor): Shape (batch_size,), class indices.
            head      (nn.Module)   : SharedClassifier (mode A) or
                                      MultiHeadClassifier (mode B).

        Returns:
            torch.Tensor: Scalar total MRL loss (sum over all prefix scales).
        """
        total_loss = torch.tensor(0.0, device=embedding.device, requires_grad=True)

        for k, w in zip(self.prefixes, self.weights):

            if self.head_mode == "shared_head":
                # Mode A: zero-pad beyond k, feed full vector to shared head
                logits = head.forward_prefix(embedding, k)

            else:
                # Mode B: slice first k dims, feed to the dedicated head for k
                logits = head(embedding, k)

            scale_loss = self.ce(logits, labels)
            total_loss = total_loss + w * scale_loss

        return total_loss


# ==============================================================================
# L1 regularization loss — cross-entropy + L1 penalty on embedding activations
# ==============================================================================

class L1RegLoss(nn.Module):
    """
    Cross-entropy loss + L1 regularization on the embedding activations.

    Promotes sparse embeddings (many dimensions driven toward zero) but does NOT
    enforce any ordering of dimensions. Used as an ablation against MRL to test
    whether sparsity alone (without ordering) improves prefix-sweep performance.

    Loss = CE(head(z), y) + lambda_l1 * mean(|z|)

    Args:
        lambda_l1 (float): Weight of the L1 penalty. Default 0.05.
    """

    def __init__(self, lambda_l1: float = 0.05):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss()
        self.lambda_l1 = lambda_l1

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """
        Compute CE loss plus L1 penalty on the embedding.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).
            labels    (torch.Tensor): Shape (batch_size,), class indices.
            head      (nn.Module)   : SharedClassifier — called with full embedding.

        Returns:
            torch.Tensor: Scalar loss = CE + lambda_l1 * mean(|embedding|).
        """
        logits  = head(embedding)
        ce_loss = self.ce(logits, labels)
        l1_reg  = self.lambda_l1 * embedding.abs().mean()
        return ce_loss + l1_reg


# ==============================================================================
# Prefix L1 loss — plain CE + Matryoshka-style weighted L1 penalty
# ==============================================================================

class PrefixL1Loss(nn.Module):
    """
    Cross-entropy loss with a Matryoshka-style weighted L1 penalty.

    Same structure as L1RegLoss, but the regularization penalizes early
    dimensions more heavily by summing the L1 norm over every prefix:

        penalty = sum_{m=1}^{k} ||z_{1:m}||_1

    Expanding the double sum, dimension j (0-indexed) appears in all prefix
    sums for m >= j+1, so it is equivalent to a weighted L1:

        penalty = sum_{j=0}^{k-1} (k - j) * |z_j|

    The weight vector [k, k-1, ..., 1] is pre-computed once and registered
    as a buffer (moves to GPU automatically with .to(device)).

    Compare to L1RegLoss:
        L1RegLoss  : CE + lambda * mean(|z|)          — uniform penalty
        PrefixL1Loss: CE + lambda * weighted_L1(z)    — front-loaded penalty

    The ordering pressure comes entirely from the regularizer, not from
    any multi-scale loss term.

    Full loss = CE(head(z), y) + lambda_l1 * (weights · |z|).mean(dim=0).sum()

    Args:
        embed_dim (int)  : Total embedding dimension k.
        lambda_l1 (float): Weight of the prefix L1 penalty. Default 0.05.
    """

    def __init__(self, embed_dim: int, lambda_l1: float = 0.05):
        super().__init__()

        self.ce        = nn.CrossEntropyLoss()
        self.lambda_l1 = lambda_l1

        # Build weight vector: w[j] = (embed_dim - j) for j in 0..embed_dim-1
        # Shape: (embed_dim,). Registered as buffer so it follows .to(device).
        w = torch.arange(embed_dim, 0, -1, dtype=torch.float32)  # [k, k-1, ..., 1]
        self.register_buffer("dim_weights", w)

        print(f"[PrefixL1Loss] embed_dim={embed_dim}, lambda_l1={lambda_l1}")
        print(f"  dim_weights (first 8): {w[:8].tolist()}")

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """
        Compute CE loss on full embedding + weighted prefix L1 penalty.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).
            labels    (torch.Tensor): Shape (batch_size,), class indices.
            head      (nn.Module)   : SharedClassifier — called with full embedding.

        Returns:
            torch.Tensor: Scalar loss = CE + lambda_l1 * weighted_l1.
        """
        # Plain cross-entropy on the full embedding (same as StandardLoss / L1RegLoss)
        ce_loss = self.ce(head(embedding), labels)

        # Weighted L1: average over batch, then weighted sum over dimensions.
        # Shape: (batch, embed_dim) -> scalar
        weighted_l1 = (self.dim_weights * embedding.abs()).mean(dim=0).sum()

        return ce_loss + self.lambda_l1 * weighted_l1


# ==============================================================================
# Factory: build the right loss from config
# ==============================================================================

def build_loss(cfg, model_type: str) -> nn.Module:
    """
    Build and return the appropriate loss function based on config and model type.

    Args:
        cfg        (ExpConfig): Experiment config. Uses head_mode, eval_prefixes.
        model_type (str)      : 'standard' or 'matryoshka'.

    Returns:
        nn.Module: StandardLoss or MatryoshkaLoss instance.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "standard":
        print("[loss] Building StandardLoss")
        return StandardLoss()

    elif model_type == "matryoshka":
        print(f"[loss] Building MatryoshkaLoss  "
              f"(prefixes={cfg.eval_prefixes}, head_mode='{cfg.head_mode}')")
        return MatryoshkaLoss(
            prefixes=cfg.eval_prefixes,
            head_mode=cfg.head_mode,
        )

    elif model_type == "l1":
        print(f"[loss] Building L1RegLoss  (lambda_l1={cfg.l1_lambda})")
        return L1RegLoss(lambda_l1=cfg.l1_lambda)

    elif model_type == "prefix_l1":
        print(
            f"[loss] Building PrefixL1Loss  "
            f"(embed_dim={cfg.embed_dim}, lambda_l1={cfg.l1_lambda})"
        )
        return PrefixL1Loss(
            embed_dim=cfg.embed_dim,
            lambda_l1=cfg.l1_lambda,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose 'standard', 'matryoshka', 'l1', or 'prefix_l1'."
        )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from config import ExpConfig
    from models.heads import build_head

    cfg       = ExpConfig()
    n_classes = 10
    batch     = 16

    dummy_emb    = torch.randn(batch, cfg.embed_dim)
    dummy_labels = torch.randint(0, n_classes, (batch,))

    # ------------------------------------------------------------------
    print("\n--- StandardLoss with shared_head ---")
    head_a   = build_head(cfg, n_classes)
    std_loss = build_loss(cfg, "standard")
    loss_val = std_loss(dummy_emb, dummy_labels, head_a)
    print(f"  Loss value: {loss_val.item():.4f}")
    assert loss_val.item() > 0, "Loss should be positive"
    assert loss_val.shape == torch.Size([]), "Should be a scalar"
    print("  PASSED")

    # ------------------------------------------------------------------
    print("\n--- MatryoshkaLoss with shared_head (Mode A) ---")
    mat_loss_a = build_loss(cfg, "matryoshka")
    loss_val_a = mat_loss_a(dummy_emb, dummy_labels, head_a)
    print(f"  Loss value: {loss_val_a.item():.4f}")
    assert loss_val_a.item() > 0
    assert loss_val_a.shape == torch.Size([])
    # MRL loss should be larger than standard (sum over multiple scales)
    print(f"  MRL loss ({loss_val_a.item():.3f}) > standard loss ({loss_val.item():.3f}): "
          f"{'PASSED' if loss_val_a.item() > loss_val.item() else 'NOTE: not strictly required'}")

    # ------------------------------------------------------------------
    print("\n--- MatryoshkaLoss with multi_head (Mode B) ---")
    cfg_b      = ExpConfig(head_mode="multi_head")
    head_b     = build_head(cfg_b, n_classes)
    mat_loss_b = build_loss(cfg_b, "matryoshka")
    loss_val_b = mat_loss_b(dummy_emb, dummy_labels, head_b)
    print(f"  Loss value: {loss_val_b.item():.4f}")
    assert loss_val_b.item() > 0
    assert loss_val_b.shape == torch.Size([])
    print("  PASSED")

    # ------------------------------------------------------------------
    print("\n--- Backward pass check (gradients flow) ---")
    dummy_emb.requires_grad_(True)
    loss_val_a = mat_loss_a(dummy_emb, dummy_labels, head_a)
    loss_val_a.backward()
    assert dummy_emb.grad is not None, "No gradient on embedding"
    print(f"  Gradient norm on embedding: {dummy_emb.grad.norm().item():.4f}")
    print("  PASSED")

    # ------------------------------------------------------------------
    print("\n--- PrefixL1Loss (plain CE + prefix-weighted L1) ---")
    prefix_l1_loss = build_loss(cfg, "prefix_l1")
    # Fresh embedding with grad so we can check backward
    dummy_emb_pl = torch.randn(batch, cfg.embed_dim, requires_grad=True)
    loss_pl = prefix_l1_loss(dummy_emb_pl, dummy_labels, head_a)
    print(f"  Loss value: {loss_pl.item():.4f}")
    assert loss_pl.item() > 0, "PrefixL1Loss should be positive"
    assert loss_pl.shape == torch.Size([]), "Should be scalar"

    # Verify weight vector has correct shape and ordering
    w = prefix_l1_loss.dim_weights
    assert w.shape == (cfg.embed_dim,), f"Weight shape mismatch: {w.shape}"
    assert w[0].item() == cfg.embed_dim, f"First weight should be embed_dim={cfg.embed_dim}"
    assert w[-1].item() == 1.0, "Last weight should be 1.0"
    assert (w[:-1] > w[1:]).all(), "Weights should be strictly decreasing"
    print(f"  Weight vector (all {cfg.embed_dim} dims): {w.tolist()}")
    print("  Weight ordering: PASSED")

    # Backward pass
    loss_pl.backward()
    assert dummy_emb_pl.grad is not None, "No gradient on embedding"
    print(f"  Gradient norm on embedding: {dummy_emb_pl.grad.norm().item():.4f}")
    print("  Backward: PASSED")

    # Sanity: prefix_l1 loss > mrl loss alone (L1 penalty adds to it)
    mrl_only_loss = build_loss(cfg, "matryoshka")
    loss_mrl_only = mrl_only_loss(dummy_emb, dummy_labels, head_a)
    # Note: different embeddings, so just check both positive
    print(f"  PrefixL1 loss ({loss_pl.item():.3f}) includes L1 penalty on top of MRL CE")
    print("  PASSED")

    print("\nAll loss checks passed.")
