"""Implements layers to project the output of a ViT in latent space to the output space."""
import torch
import torch.nn as nn


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
        # We'll deconvolve the latent space using subpixel convolutions
        self.conv = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=channels * patch_size ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
    
    def forward(self, x, tokens_shape):
        """
        Args:
            x (torch.Tensor): The tensor of shape (B, L, D) containing the latent space tokens.
            tokens_shape (tuple of int): Number of patches along each dimension in the original
                image, i.e. (ceil(W / patch_size), ceil(H / patch_size)).
        Returns:
            torch.Tensor: The tensor of shape (B, C, H, W) containing the projected tokens.
        """
        # Transpose x from (B, L, D) to (B, D, w, h)
        x = x.transpose(1, 2).view(x.size(0), -1, *tokens_shape)
        # Apply the subpixel deconvolution
        x = self.conv(x)  # (B, C * patch_size ** 2, w, h)
        x = self.pixel_shuffle(x)  # (B, C, H, W)
        # Remove the potential padding that was added to the image
        x = x[:, :, :self.spatial_shape[0], :self.spatial_shape[1]]
        return x