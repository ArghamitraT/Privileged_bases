"""Paper architecture and per-image Bernoulli/KL loss.

Conda environment: mrl_env_cuda12
Usage:
    Import VAE and vae_loss from the reproduction entry point.
"""

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """Higgins et al. 2D-shapes MLP, with diagonal Gaussian posterior."""

    def __init__(self, latent_dim=10, hidden_dim=1200,
                 decoder_activation="tanh", initialization="default"):
        super().__init__()
        if decoder_activation not in ("tanh", "relu"):
            raise ValueError("decoder_activation must be tanh or relu")
        if initialization not in ("default", "xavier"):
            raise ValueError("initialization must be default or xavier")
        activation = nn.Tanh if decoder_activation == "tanh" else nn.ReLU
        self.encoder = nn.Sequential(
            nn.Linear(4096, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, 4096),
        )
        if initialization == "xavier":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def encode(self, x):
        hidden = self.encoder(x.reshape(-1, 4096))
        return self.mean(hidden), self.logvar(hidden)

    def forward(self, x):
        z, mean, logvar = self.sample_posterior(x)
        return self.decoder(z), mean, logvar

    def sample_posterior(self, x):
        """One encoder pass and one full latent draw, shared by MRL prefixes."""
        mean, logvar = self.encode(x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z, mean, logvar


def posterior_kl(mean, logvar):
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean(dim=0)


def masked_prefix(z, k):
    if not 1 <= k <= z.shape[1]:
        raise ValueError("Prefix must fit the latent dimension")
    return z * (torch.arange(z.shape[1], device=z.device) < k).to(z.dtype)


def vae_loss(logits, x, mean, logvar, beta, *, objective="beta",
             capacity=0.0, gamma=1000.0):
    """Sum over pixels/coordinates; average over images, never over pixels."""
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, x.reshape_as(logits), reduction="none"
    ).sum(dim=1).mean()
    kl_per_coordinate = posterior_kl(mean, logvar)
    kl = kl_per_coordinate.sum()
    if objective == "beta":
        penalty = beta * kl
    elif objective == "capacity":
        # Apply absolute value after batch-mean KL, as in capacity-control training.
        penalty = gamma * (kl - capacity).abs()
    else:
        raise ValueError("objective must be beta or capacity")
    return reconstruction + penalty, reconstruction, kl, kl_per_coordinate


def mrl_vae_loss(model, x, beta, *, backward=False):
    """Mean of all prefix BCE scalars + one full KL, reusing one sampled z.

    backward=True accumulates prefix gradients sequentially to bound decoder
    activation memory. It performs no optimizer step and returns detached logs.
    backward=False returns a differentiable objective for checks/small batches.
    """
    z, mean, logvar = model.sample_posterior(x)
    coordinates = posterior_kl(mean, logvar)
    kl = coordinates.sum()
    if not torch.isfinite(kl):
        raise RuntimeError("Nonfinite MRL posterior KL")
    prefixes = []
    for k in range(1, z.shape[1] + 1):
        logits = model.decoder(masked_prefix(z, k))
        recon = F.binary_cross_entropy_with_logits(
            logits, x.reshape_as(logits), reduction="none"
        ).sum(dim=1).mean()
        if not torch.isfinite(recon):
            raise RuntimeError(f"Nonfinite MRL reconstruction at prefix {k}")
        if backward:
            (recon / z.shape[1]).backward(retain_graph=True)
        prefixes.append(recon.detach() if backward else recon)
    if backward:
        (beta * kl).backward()
        kl, coordinates = kl.detach(), coordinates.detach()
    prefix_losses = torch.stack(prefixes)
    reconstruction = prefix_losses.mean()
    return reconstruction + beta * kl, reconstruction, kl, coordinates, prefix_losses


def model_from_config(config):
    """Restore architecture options; old checkpoints retain the original defaults."""
    return VAE(config["latent_dim"], config["hidden_dim"],
               decoder_activation=config.get("decoder_activation", "tanh"),
               initialization=config.get("initialization", "default"))
