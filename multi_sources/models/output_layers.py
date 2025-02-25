"""Implements layers to project the output of a ViT in latent space to the output space."""

import torch.nn as nn
from multi_sources.models.icnr import ICNR
from multi_sources.models.conv import ResNet
from einops.layers.torch import Rearrange


class SourcetypeProjection2d(nn.Module):
    """Receives the embedded values and coordinates of a source and projects them to
    that source's original space. Meant to be shared across all sources of the same type.
    """

    def __init__(self, values_dim, coords_dim, out_channels, patch_size):
        """
        Args:
            values_dim (int): Dimension of the values embeddings.
            coords_dim (int): Dimension of the coordinates embeddings.
            out_channels (int): Number of channels in the output space.
            patch_size (int): Size of the embedding patches.
        """
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(values_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(coords_dim, 2 * values_dim))
        # Subpixel convolution to project the latent space to the output space
        self.conv = nn.Conv2d(
            in_channels=values_dim,
            out_channels=out_channels * patch_size**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        # Apply the ICNR initialization to the deconvolution, to reduce checkerboard artifacts
        weight = ICNR(
            self.conv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=patch_size
        )
        self.conv.weight.data.copy_(weight)

    def forward(self, values, coords, tokens_shape, **unused_kwargs):
        """
        Args:
            values (torch.Tensor): Embedded values of shape (B, L, D).
            coords (torch.Tensor): Embedded coordinates of shape (B, L, D).
            tokens_shape (tuple of int): Shape for rearranging tokens, (width, height).
        Returns:
            torch.Tensor of shape (B, channels, H, W) containing the projected output.
        """
        # Apply the modulation to the values embeddings
        shift, scale = self.modulation(coords).chunk(2, dim=-1)
        v = (1 + scale) * self.norm(values) + shift
        # Transpose x from (B, L, D) to (B, D, w, h)
        v = v.transpose(1, 2).view(v.size(0), -1, *tokens_shape)
        # Deconvolve the latent space using subpixel convolutions
        v = self.conv(v)
        v = self.pixel_shuffle(v)  # (B, C, H, W)
        if hasattr(self, "rearrange"):
            v = self.rearrange(v)
        return v


class SourcetypeProjection0d(nn.Module):
    """Receives the embedded values and coordinates of a 0D source and projects them to
    that source's original space. Meant to be shared across all sources of the same type."""

    def __init__(self, values_dim, coords_dim, out_channels, patch_size=None):
        """Args:
        values_dim (int): Dimension of the values embeddings.
        coords_dim (int): Dimension of the coordinates embeddings.
        out_channels (int): Number of channels in the output space.
        patch_size (int, optional): Not used for 0D sources, kept for API consistency.
        """
        super().__init__()
        self.norm = nn.LayerNorm(values_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(coords_dim, 2 * values_dim))
        self.output_proj = nn.Linear(values_dim, out_channels)

    def forward(self, values, coords, **unused_kwargs):
        """Args:
            values (torch.Tensor): Embedded values of shape (B, L=1, D).
            coords (torch.Tensor): Embedded coordinates of shape (B, L=1, D).
        Returns:
            torch.Tensor of shape (B, C) containing the projected output.
        """
        # Apply the modulation to the values embeddings
        shift, scale = self.modulation(coords).chunk(2, dim=-1)
        v = (1 + scale) * self.norm(values) + shift
        # Project to output space
        v = self.output_proj(v)[:, 0]  # (B, C)
        return v
