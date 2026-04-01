"""
Script: models/encoder.py
--------------------------
MLP encoder that maps raw input features to a fixed-size embedding vector.

Both the standard model and the Matryoshka model use this same encoder
architecture. The difference between the two lies entirely in the loss
function used during training (see losses/mat_loss.py), not the architecture.

Architecture:
    input (input_dim)
        -> Linear(input_dim, hidden_dim) -> BatchNorm -> ReLU -> Dropout
        -> Linear(hidden_dim, hidden_dim) -> BatchNorm -> ReLU -> Dropout
        -> Linear(hidden_dim, embed_dim)
        -> L2-normalised embedding

The L2 normalisation ensures all embeddings lie on the unit hypersphere,
making the prefix truncation / zero-padding more principled (no single
dimension can dominate via scale).

Inputs:
    input_dim (int) : number of raw input features
    hidden_dim (int): width of the hidden layers (from ExpConfig.hidden_dim)
    embed_dim (int) : size of the output embedding (from ExpConfig.embed_dim)

Outputs:
    torch.Tensor of shape (batch_size, embed_dim), L2-normalised

Usage:
    from models.encoder import MLPEncoder
    encoder = MLPEncoder(input_dim=784, hidden_dim=256, embed_dim=64)
    python models/encoder.py   # smoke test (random forward pass, checks output shape)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    Two-hidden-layer MLP that produces an L2-normalised embedding.

    Args:
        input_dim  (int): Number of input features.
        hidden_dim (int): Width of both hidden layers.
        embed_dim  (int): Dimensionality of the output embedding.
        dropout    (float): Dropout probability applied after each hidden layer.
                            Default 0.1. Set to 0.0 to disable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- Layer definitions ---

        # First hidden block
        self.fc1   = nn.Linear(input_dim,  hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        # Second hidden block
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        # Output projection to embedding space (no activation — raw logits)
        self.fc_out = nn.Linear(hidden_dim, embed_dim)

        # Store dims for reference / checkpoints
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input features -> L2-normalised embedding.

        Args:
            x (torch.Tensor): Input of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: L2-normalised embedding of shape (batch_size, embed_dim).
        """

        # First hidden block
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        # Second hidden block
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        # Project to embedding space
        x = self.fc_out(x)

        # L2 normalise so all embeddings lie on the unit hypersphere
        x = F.normalize(x, p=2, dim=1)

        return x


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from config import ExpConfig

    cfg = ExpConfig()

    print("--- Building MLPEncoder ---")
    encoder = MLPEncoder(
        input_dim=784,           # MNIST
        hidden_dim=cfg.hidden_dim,
        embed_dim=cfg.embed_dim,
    )
    print(encoder)
    print()

    # Checkpoint 1: forward pass with a random batch
    print("--- Checkpoint 1: forward pass ---")
    batch = torch.randn(32, 784)   # batch of 32 MNIST-like inputs
    embeddings = encoder(batch)
    print(f"  Input shape:     {batch.shape}")
    print(f"  Embedding shape: {embeddings.shape}")
    assert embeddings.shape == (32, cfg.embed_dim), "Wrong embedding shape"
    print("  PASSED\n")

    # Checkpoint 2: L2 normalisation — each row should have unit norm
    print("--- Checkpoint 2: L2 normalisation ---")
    norms = embeddings.norm(dim=1)
    print(f"  Embedding norms (first 5): {norms[:5].detach().tolist()}")
    assert torch.allclose(norms, torch.ones(32), atol=1e-5), \
        "Embeddings are not unit-normalised"
    print("  PASSED\n")

    # Checkpoint 3: parameter count
    print("--- Checkpoint 3: parameter count ---")
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print("  PASSED\n")

    print("All encoder checks passed.")
