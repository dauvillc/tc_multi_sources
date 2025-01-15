"""Implements layers to project the output of a ViT in latent space to the output space."""

import torch.nn as nn
from multi_sources.models.icnr import ICNR
from multi_sources.models.conv import ResNet


class SourceSpecificProjection2d(nn.Module):
    """For a single 2d source, receives the output of the ViT in latent space as
    a tensor of shape (B, L, D) where B is the batch size, L is the number of tokens
    and D is the dimension of the latent space.
    The module applies a deconvolution to project the tokens to the output space as
    a tensor (B, C, H, W).
    """

    def __init__(self, channels, spatial_shape, patch_size, latent_dim):
        """
        Args:
            channels (int): The number of channels in the source.
            spatial_shape (tuple of int): The spatial shape (H, W) of the source.
            patch_size (int): The size of the patches when they were embedded.
            latent_dim (int): The dimension of the latent space.
        """
        super().__init__()
        self.patch_size = patch_size
        self.spatial_shape = spatial_shape
        self.norm = nn.LayerNorm(latent_dim)
        # We'll deconvolve the latent space using subpixel convolutions
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
        # Final CNN to reduce the artifacts
        self.final_conv = ResNet(channels, 64, n_blocks=2)

    def forward(self, x, tokens_shape):
        """
        Args:
            x (torch.Tensor): The tensor of shape (B, L, D) containing the latent space tokens.
            tokens_shape (tuple of int): Number of patches along each dimension in the original
                image, i.e. (ceil(W / patch_size), ceil(H / patch_size)).
        Returns:
            torch.Tensor: The tensor of shape (B, C, H, W) containing the projected tokens.
        """
        x = self.norm(x)
        # Transpose x from (B, L, D) to (B, D, w, h)
        x = x.transpose(1, 2).view(x.size(0), -1, *tokens_shape)
        # Apply the subpixel deconvolution
        x = self.conv(x)  # (B, C * patch_size ** 2, w, h)
        x = self.pixel_shuffle(x)  # (B, C, H, W)
        # Apply the final Conv2d to reduce the artifacts
        x = self.final_conv(x)
        # Remove the potential padding that was added to the image
        x = x[:, :, : self.spatial_shape[0], : self.spatial_shape[1]]
        return x


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
        # Final Conv2d to reduce the artifacts
        self.final_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

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
        x = self.final_conv(x)
        return x


class SourceSpecificProjection0d(nn.Module):
    """For a single 0d source, receives the output of the ViT in latent space as
    a tensor of shape (B, 1, D) where B is the batch size,
    and D is the dimension of the latent space.
    The module applies a linear layer to project the tokens to the output space as
    a tensor (B, C).
    """

    def __init__(self, channels, latent_dim):
        """
        Args:
            channels (int): The number of channels in the source.
            latent_dim (int): The dimension of the latent space.
        """
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.linear = nn.Linear(latent_dim, channels)

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): The tensor of shape (B, 1, D) containing the latent space tokens.
            kwargs: Additional arguments for compatibility with other forms of projection.
        Returns:
            torch.Tensor: The tensor of shape (B, C) containing the projected tokens.
        """
        x = self.norm(x[:, 0])  # (B, D)
        # Apply the linear layer
        x = self.linear(x)  # (B, C)
        return x


class SourcetypeProjection0d(nn.Module):
    """Projects the ViT output in latent space to the output space for multiple
    0D sources sharing the same source type.
    """

    def __init__(self, channels, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.linear = nn.Linear(latent_dim, channels)

    def forward(self, x):
        x = self.norm(x[:, 0])  # (B, D)
        x = self.linear(x)      # (B, C)
        return x