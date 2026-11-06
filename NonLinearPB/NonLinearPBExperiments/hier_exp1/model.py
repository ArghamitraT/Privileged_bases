"""Small residual CNN autoencoder (standard and full-prefix MRL) and the reconstruction classifier.

Conda environment: mrl_env_cuda12
Usage:
    from hier_exp1.model import CNNAE, objective_loss, prefix_mse, ReconClassifier

Spec: NonLinearPB/plans/hierarchical_Exp1.md sections 2-3 and 5D.3. One
architecture for both objectives; the decoder is shared by every prefix, and a
prefix keeps z_1..z_k and zeroes the rest. Images are [B, 1, 64, 64] in [0, 1].
The decoder output is linear (no final Sigmoid): with MSE, a sigmoid saturates
and every model collapsed to all-black output (plan section 2, 2026-11-02), so
reconstructions can fall slightly outside [0, 1]; clip only for display/classifier.
"""

import torch
from torch import nn
from torch.nn import functional as F

LATENT_DIM = 10


class LayerNorm2d(nn.Module):
    """LayerNorm over channels at each spatial position (NCHW -> NHWC -> NCHW)."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    """h + Conv3x3(GELU(Conv3x3(LayerNorm2d(h)))); no downsampling."""

    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(LayerNorm2d(channels), nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, h):
        return h + self.branch(h)


def _up(c_in, c_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         nn.Conv2d(c_in, c_out, 3, padding=1), nn.GELU())


class CNNAE(nn.Module):
    """Deterministic 10-d latent; no global pooling. Default: no normalization or activation on z.
    bounded=True appends tanh to the encoder, so z lies in [-1, 1] (noise experiment, 2026-11-03);
    tanh has no parameters, so the initial weights are the same as the unbounded model with the same seed."""

    def __init__(self, latent_dim=LATENT_DIM, bounded=False, channels=1, size=64):
        """size 64 (synthetic, 1 channel): the original network, unchanged. size 32 (CIFAR, 3 channels): same layers
        except the third encoder conv has stride 1 and the last decoder upsampling stage is dropped, so the
        bottleneck is still 128 x 8 x 8."""
        super().__init__()
        if size not in (32, 64):
            raise ValueError(size)
        self.latent_dim, self.bounded, self.channels, self.size = latent_dim, bounded, channels, size
        s3 = 2 if size == 64 else 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1), nn.GELU(), ResidualBlock(32),  # 32 x 32 (16 x 16)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), ResidualBlock(64),      # 16 x 16 (8 x 8)
            nn.Conv2d(64, 128, 3, stride=s3, padding=1), nn.GELU(),                        # 8 x 8
            nn.Flatten(), nn.Linear(128 * 8 * 8, latent_dim), *([nn.Tanh()] if bounded else []))
        self.decoder_in = nn.Sequential(nn.Linear(latent_dim, 128 * 8 * 8), nn.GELU())
        if size == 64:
            self.decoder = nn.Sequential(
                ResidualBlock(128), _up(128, 64), ResidualBlock(64), _up(64, 32), _up(32, 16),
                nn.Conv2d(16, channels, 3, padding=1))
        else:
            self.decoder = nn.Sequential(
                ResidualBlock(128), _up(128, 64), ResidualBlock(64), _up(64, 32),
                nn.Conv2d(32, channels, 3, padding=1))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(self.decoder_in(z).view(-1, 128, 8, 8))


def prefix_masks(latent_dim, device):
    """Row k-1 keeps the first k coordinates."""
    return torch.tril(torch.ones(latent_dim, latent_dim, device=device))


def pair_masks(latent_dim, device):
    """Row 0 keeps z1 only; row j-1 (j = 2..d) keeps z1 and zj. The 'pair' objective's codes."""
    m = torch.zeros(latent_dim, latent_dim, device=device)
    m[:, 0] = 1
    idx = torch.arange(1, latent_dim, device=device)
    m[idx, idx] = 1
    return m


def masked_mse(model, x, masks, z=None):
    """Per-image MSE for each code mask (rows of masks [K, d]): tensor [K, B]. One batched decoder pass."""
    z = model.encode(x) if z is None else z
    zk = (z[None] * masks[:, None]).reshape(-1, model.latent_dim)
    xhat = model.decode(zk).view(len(masks), *x.shape)
    return ((xhat - x[None]) ** 2).mean(dim=(2, 3, 4))


def pair_mse(model, x, z=None):
    """Per-image MSE for {z1}, {z1, z2}, ..., {z1, zd}: tensor [d, B]."""
    z = model.encode(x) if z is None else z
    return masked_mse(model, x, pair_masks(model.latent_dim, z.device), z)


def prefix_mse(model, x, z=None, ks=None, noise_sigma=0.0):
    """Per-image MSE for each prefix k in ks: tensor [len(ks), B]. One batched decoder pass.
    noise_sigma > 0 (training only): fresh Gaussian noise on the kept first k entries of each prefix
    code, never on the zeroed tail; each prefix gets its own draw."""
    z = model.encode(x) if z is None else z
    ks = list(range(1, model.latent_dim + 1)) if ks is None else list(ks)
    masks = prefix_masks(model.latent_dim, z.device)[[k - 1 for k in ks]]          # [K, d]
    zk = z[None] * masks[:, None]                                                    # [K, B, d]
    if noise_sigma > 0:
        zk = zk + noise_sigma * torch.randn_like(zk) * masks[:, None]
    zk = zk.reshape(-1, model.latent_dim)                                            # [K*B, d]
    xhat = model.decode(zk).view(len(ks), *x.shape)
    return ((xhat - x[None]) ** 2).mean(dim=(2, 3, 4))


def objective_loss(model, x, objective, noise_sigma=0.0):
    """'ae': full-width MSE. 'mrl': equal-weight mean over k = 1..10 of the prefix MSE.
    'pair': equal-weight mean of the MSE from {z1}, {z1, z2}, ..., {z1, zd} (rest of the code zero).
    noise_sigma > 0: Gaussian noise on the code before decoding (all 10 entries for ae, the kept
    first k for each mrl prefix)."""
    if objective == "ae":
        z = model.encode(x)
        if noise_sigma > 0:
            z = z + noise_sigma * torch.randn_like(z)
        return F.mse_loss(model.decode(z), x)
    if objective == "mrl":
        return prefix_mse(model, x, noise_sigma=noise_sigma).mean()
    if objective == "pair":                       # equal-weight mean over {z1}, {z1, z2}, ..., {z1, zd}; full code not used
        if noise_sigma > 0:
            raise ValueError("noise is not implemented for the pair objective")
        return pair_mse(model, x).mean()
    raise ValueError(objective)


class ReconClassifier(nn.Module):
    """3 stride-2 convs + flatten (keeps position), heads for coarse (3), mid node (9), leaf (27)."""

    HEADS = {"coarse": 3, "mid_node": 9, "leaf_id": 27}

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.heads = nn.ModuleDict({k: nn.Linear(64 * 8 * 8, n) for k, n in self.HEADS.items()})

    def forward(self, x):
        h = self.features(x)
        return {k: head(h) for k, head in self.heads.items()}
