"""
Script: models/heads.py
------------------------
Classifier head(s) that sit on top of the encoder embedding.

Two modes are implemented, selectable via ExpConfig.head_mode:

  Mode A — SharedClassifier ('shared_head')
      A single Linear(embed_dim, n_classes) layer.
      At each prefix scale k during training and evaluation, the embedding
      is zero-padded: dimensions k..embed_dim are set to zero, and the
      full embed_dim-length vector is fed to the head.
      Pros: simple, directly matches the evaluation protocol.
      Cons: the head must learn to handle sparse/zero-padded inputs.

  Mode B — MultiHeadClassifier ('multi_head')
      One separate Linear(k, n_classes) layer per prefix scale k.
      During training, embedding[:, :k] is sliced (no padding) and fed
      to the head for scale k. During evaluation, the matching head is
      selected for each k.
      Pros: each head sees only clean, non-zero features; cleaner gradients.
      Cons: more parameters; heads do not share learned knowledge.

Inputs:
    embed_dim   (int)      : full embedding size (from ExpConfig.embed_dim)
    n_classes   (int)      : number of target classes
    prefixes    (List[int]): list of prefix sizes k (from ExpConfig.eval_prefixes)
    head_mode   (str)      : 'shared_head' or 'multi_head'

Outputs:
    Logit tensor of shape (batch_size, n_classes)

Usage:
    from models.heads import SharedClassifier, MultiHeadClassifier
    head = SharedClassifier(embed_dim=64, n_classes=10)
    python models/heads.py   # smoke test (random forward pass for both head modes)
"""

from typing import List
import torch
import torch.nn as nn


# ==============================================================================
# Mode A — single shared head with zero-padding
# ==============================================================================

class SharedClassifier(nn.Module):
    """
    Single linear classifier that accepts a full-length (embed_dim) vector.

    To evaluate at prefix k, zero out dimensions k..embed_dim-1 before
    calling forward(). The zero-masking can be done here via the
    forward_prefix() helper, or externally.

    Args:
        embed_dim (int): Dimension of the full embedding vector.
        n_classes (int): Number of output classes.
    """

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim
        self.n_classes = n_classes

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify a full (or pre-masked) embedding.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, n_classes).
        """
        return self.head(embedding)

    def forward_prefix(self, embedding: torch.Tensor, k: int) -> torch.Tensor:
        """
        Classify using only the first k dimensions; zero out the rest.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).
            k         (int)         : Number of prefix dimensions to keep.

        Returns:
            torch.Tensor: Logits of shape (batch_size, n_classes).
        """
        assert k <= self.embed_dim, f"k={k} exceeds embed_dim={self.embed_dim}"

        # Build a zero-padded copy — do NOT modify the original tensor
        masked = embedding.clone()
        masked[:, k:] = 0.0

        return self.head(masked)


# ==============================================================================
# Mode B — separate head per prefix scale
# ==============================================================================

class MultiHeadClassifier(nn.Module):
    """
    One independent linear classifier per prefix scale k.

    Each head maps a k-dimensional slice of the embedding directly to class
    logits, with no zero-padding involved.

    Args:
        prefixes  (List[int]): The set of prefix sizes, e.g. [1,2,4,8,16,32,64].
        n_classes (int)      : Number of output classes.
    """

    def __init__(self, prefixes: List[int], n_classes: int):
        super().__init__()
        self.prefixes  = sorted(prefixes)
        self.n_classes = n_classes

        # Create one Linear head per prefix size, stored in a ModuleDict
        # so PyTorch registers them as trainable parameters.
        # Keys are strings because ModuleDict requires string keys.
        self.heads = nn.ModuleDict({
            str(k): nn.Linear(k, n_classes)
            for k in self.prefixes
        })

    def forward(self, embedding: torch.Tensor, k: int) -> torch.Tensor:
        """
        Classify using the first k dimensions of the embedding.

        Args:
            embedding (torch.Tensor): Shape (batch_size, embed_dim).
                                      Only the first k columns are used.
            k         (int)         : Which prefix head to use. Must be in self.prefixes.

        Returns:
            torch.Tensor: Logits of shape (batch_size, n_classes).

        Raises:
            KeyError: If k is not in the set of registered prefix sizes.
        """
        assert str(k) in self.heads, (
            f"No head registered for k={k}. Available: {self.prefixes}"
        )

        # Slice only the first k dimensions — no padding needed
        prefix_embedding = embedding[:, :k]
        return self.heads[str(k)](prefix_embedding)


# ==============================================================================
# Factory: build the right head from config
# ==============================================================================

def build_head(cfg, n_classes: int) -> nn.Module:
    """
    Build and return the appropriate classifier head based on cfg.head_mode.

    Args:
        cfg       (ExpConfig): Experiment config. Uses head_mode, embed_dim,
                               eval_prefixes.
        n_classes (int)      : Number of target classes.

    Returns:
        nn.Module: Either a SharedClassifier or MultiHeadClassifier instance.

    Raises:
        ValueError: If cfg.head_mode is not recognised.
    """
    if cfg.head_mode == "shared_head":
        print(f"[heads] Building SharedClassifier  (embed_dim={cfg.embed_dim}, n_classes={n_classes})")
        return SharedClassifier(cfg.embed_dim, n_classes)

    elif cfg.head_mode == "multi_head":
        print(f"[heads] Building MultiHeadClassifier (prefixes={cfg.eval_prefixes}, n_classes={n_classes})")
        return MultiHeadClassifier(cfg.eval_prefixes, n_classes)

    else:
        raise ValueError(
            f"Unknown head_mode '{cfg.head_mode}'. "
            f"Choose 'shared_head' or 'multi_head'."
        )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from config import ExpConfig

    cfg = ExpConfig()   # defaults: embed_dim=64, prefixes=[1,2,4,8,16,32,64]
    n_classes = 10
    batch_size = 16

    dummy_embedding = torch.randn(batch_size, cfg.embed_dim)

    # ------------------------------------------------------------------
    print("\n--- Mode A: SharedClassifier ---")
    head_a = build_head(cfg, n_classes)

    # Full embedding
    logits = head_a(dummy_embedding)
    print(f"  Full forward  shape: {logits.shape}")
    assert logits.shape == (batch_size, n_classes)
    print("  PASSED")

    # Prefix forward for each k
    for k in cfg.eval_prefixes:
        logits_k = head_a.forward_prefix(dummy_embedding, k)
        assert logits_k.shape == (batch_size, n_classes), f"Wrong shape for k={k}"

        # Verify original embedding is unchanged.
        # Only check when k < embed_dim: when k == embed_dim, embedding[:, k:]
        # is an empty slice (0 columns), so sum() == 0 and the check is vacuous.
        if k < cfg.embed_dim:
            assert dummy_embedding[:, k:].abs().sum() > 0, \
                "forward_prefix should NOT modify the original embedding"
    print(f"  forward_prefix for all prefixes {cfg.eval_prefixes}: PASSED")

    # ------------------------------------------------------------------
    print("\n--- Mode B: MultiHeadClassifier ---")
    cfg_b = ExpConfig(head_mode="multi_head")
    head_b = build_head(cfg_b, n_classes)

    for k in cfg_b.eval_prefixes:
        logits_k = head_b(dummy_embedding, k)
        assert logits_k.shape == (batch_size, n_classes), f"Wrong shape for k={k}"
    print(f"  forward for all prefixes {cfg_b.eval_prefixes}: PASSED")

    # ------------------------------------------------------------------
    print("\n--- Parameter counts ---")
    params_a = sum(p.numel() for p in head_a.parameters())
    params_b = sum(p.numel() for p in head_b.parameters())
    print(f"  SharedClassifier   params: {params_a:,}")
    print(f"  MultiHeadClassifier params: {params_b:,}")

    print("\nAll heads checks passed.")
