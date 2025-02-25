"""Implements layers to project the output of a ViT in latent space to the output space."""

import torch.nn as nn
from multi_sources.models.icnr import ICNR
from multi_sources.models.conv import ResNet
from einops.layers.torch import Rearrange


class SourcetypeProjection2d(nn.Module):
    """Projects the ViT output in latent space to the output space for multiple
    2D sources sharing the same source type.
    """

    def __init__(self, channels, patch_size, latent_dim):
        """
        Args:
            channels (int): Number of channels in the 2D source type.
            patch_size (int): The size of each patch used in the embedding.
            latent_dim (int): Dimension of the latent tokens from the ViT.
        """
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(latent_dim)
        self.conv = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=channels * patch_size**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        # Apply the ICNR initialization to the deconvolution
        weight = ICNR(
            self.conv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=patch_size
        )
        self.conv.weight.data.copy_(weight)

    def forward(self, x, tokens_shape):
        """
        Args:
            x (torch.Tensor): Latent tokens of shape (B, L, D).
            tokens_shape (tuple of int): Shape for rearranging tokens, (width, height).
        Returns:
            torch.Tensor of shape (B, channels, H, W) containing the projected output.
        """
        x = self.norm(x)
        # Transpose x from (B, L, D) to (B, D, w, h)
        x = x.transpose(1, 2).view(x.size(0), -1, *tokens_shape)
        # Deconvolve the latent space using subpixel convolutions
        x = self.conv(x)
        x = self.pixel_shuffle(x)  # (B, C, H, W)
        if hasattr(self, "rearrange"):
            x = self.rearrange(x)
        return x


class SourcetypeProjection0d(nn.Module):
    """Projects the ViT output in latent space to the output space for multiple
    0D sources sharing the same source type.
    """

    def __init__(self, channels, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.linear = nn.Linear(latent_dim, channels)

    def forward(self, x, **kwargs):
        x = self.norm(x[:, 0])  # (B, D)
        x = self.linear(x)  # (B, C)
        return x
