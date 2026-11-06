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


class PrefixHeads(nn.Module):
    """One linear head R^k -> R^C per prefix k = 1..d (supervised MRL)."""

    def __init__(self, latent_dim, classes):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(k, classes) for k in range(1, latent_dim + 1))

    def forward(self, z):
        """(B, d) -> (d, B, C): prefix-k logits read only z[:, :k]."""
        return torch.stack([head(z[:, :k]) for k, head in enumerate(self.heads, 1)])


def prefix_masks(latent_dim, device):
    """(d, d) 0/1 rows; row k-1 keeps the first k coordinates."""
    arange = torch.arange(latent_dim, device=device)
    return (arange[None, :] < arange[:, None] + 1).float()


def mrl_pretrain_loss(model, x, beta, heads=None, labels=None, lambda_sup=0.0, ln_classes=1.0):
    """Batched all-prefix objective: mean prefix BCE + beta * full KL + lambda * mean CE / ln C.

    Same reconstruction term as mrl_vae_loss (one shared sample, every masked
    prefix, mean of per-prefix scalar BCE), but all ten prefixes go through the
    decoder as one (d * B) batch. beta == 0 is deterministic: z = mu, no KL.
    """
    mean, logvar = model.encode(x)
    if beta > 0:
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        coordinates = posterior_kl(mean, logvar)
    else:
        z, coordinates = mean, torch.zeros_like(mean[0])
    d = z.shape[1]
    logits = model.decoder((z[None] * prefix_masks(d, z.device)[:, None]).reshape(-1, d))
    bce = F.binary_cross_entropy_with_logits(
        logits, x.repeat(d, 1), reduction="none").sum(dim=1)
    prefix_recon = bce.reshape(d, -1).mean(dim=1)
    kl = coordinates.sum()
    loss = prefix_recon.mean() + beta * kl
    out = {"recon": prefix_recon.mean(), "prefix_recon": prefix_recon, "kl": kl,
           "coordinate_kl": coordinates}
    if heads is not None:
        head_logits = heads(z)
        ce = F.cross_entropy(head_logits.reshape(-1, head_logits.shape[-1]),
                             labels.repeat(d), reduction="none").reshape(d, -1).mean(dim=1)
        sup = ce.mean() / ln_classes
        loss = loss + lambda_sup * sup
        out.update(sup=sup, prefix_ce=ce,
                   prefix_acc=(head_logits.argmax(-1) == labels[None]).float().mean(dim=1))
    out["loss"] = loss
    return out


def model_from_config(config):
    """Restore architecture options; old checkpoints retain the original defaults."""
    return VAE(config["latent_dim"], config["hidden_dim"],
               decoder_activation=config.get("decoder_activation", "tanh"),
               initialization=config.get("initialization", "default"))
