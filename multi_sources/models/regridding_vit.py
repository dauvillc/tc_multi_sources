"""Implements class for a specific ViT architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_sources.models.regridding_vit_classes import EntryConvBlock
from multi_sources.models.regridding_vit_classes import AttentionRegriddingBlock


def remove_dots_in_keys(d):
    """Replaces the character '.' in the keys of a dict with '`'."""
    return {k.replace(".", "`"): v for k, v in d.items()}


def restore_dots_in_keys(d):
    """Replaces the character '`' in the keys of a dict with '.'."""
    return {k.replace("`", "."): v for k, v in d.items()}


def pair(x):
    """Ensures that x is a pair of integers."""
    if isinstance(x, tuple):
        return x
    return (x, x)


class RegriddingTransformer(nn.Module):
    """Implements a ViT-like architecture which:
    - Computes an attention map between the patches' coordinates;
    - Based on the attention map, selects a common number of patches
      from each source;
    - Reconstructs the images from the selected patches, and concatenates
      them along the channel dimension;
    - Applies a convolutional block to the concatenated images.
    """

    def __init__(
        self,
        img_sizes,
        channels,
        patch_size,
        c_dim,
        inner_channels,
        depth,
        downsample,
        heads,
        dim_head,
        output_resnet_depth=3,
        output_resnet_inner_channels=8,
        output_resnet_kernel_size=3,
    ):
        """
        Args:
            img_sizes (dict of str to int or tuple of int): Size of the images for each source.
            channels (dict of str to int): Number of channels in the values (V) for each source.
            patch_size (int or tuple of int): Size of the patches to be extracted from the image.
            c_dim (int): Dimension of the internal embeddings for the coordinates.
            inner_channels (int): Number of channels in the internal convolutions.
            depth (int): Number of layers.
            downsample (bool): Whether to downsample the images by 2 at the beginning.
            heads (int): Number of heads for the attention mechanism.
            dim_head (int): Dimension of each head.
            output_resnet_depth (int): Depth of the output ResNet.
                set to 0 to disable any convolutional layer at the output.
            output_resnet_inner_channels (int): Number of channels in the inner layers
                of the output
            output_resnet_kernel_size (int): Kernel size of the output ResNet.
        """
        super().__init__()
        # ModuleDict objects keys can't contain the character '.'; so replace it with '_'
        # in the keys of img_sizes and channels.
        img_sizes = img_sizes
        self.n_sources = len(img_sizes)
        self.c_dim = c_dim
        self.depth = depth
        self.heads = heads
        self.channels = channels
        patch_size = pair(patch_size)
        self.patch_size = patch_size
        # Create the input convolutions for the images and the coordinates.
        self.entry_block = EntryConvBlock(self.channels, inner_channels, downsample)
        self.coord_entry_block = (
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()
        )
        # Create the regridding blocks.
        self.regridding_blocks = nn.ModuleList(
            [
                AttentionRegriddingBlock(
                    self.n_sources,
                    1,
                    inner_channels,
                    1,
                    patch_size,
                    3,  # context: availability flag, dt, source index
                    c_dim,
                    heads,
                    dim_head,
                )
                for _ in range(depth)
            ]
        )
        # Create an output transposed convolution to upsample the image back to the original size.
        self.output_transposed_conv = nn.ModuleList(
            [
                nn.Sequential(
                    (
                        nn.ConvTranspose2d(
                            inner_channels,
                            channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
                        if downsample
                        else nn.Conv2d(inner_channels, channels, kernel_size=3, padding=1)
                    ),
                )
                for channels in self.channels
            ]
        )

    def forward(self, inputs):
        """
        Args:
            inputs (dict of str to tuple of tensors): Inputs to the model, as a dict
                {source_name: (A, S, DT, C, D, V)}.
        Returns:
            output (dict of str to tensor): Reconstructed values for each source, as a dict
                {source_name: tensor of shape (batch_size, channels, H, W)}.
        """
        images, coords, contexts = [], [], []
        for source_name, (A, S, DT, C, _, V) in inputs.items():
            images.append(V)
            coords.append(C[:, :2])  # 0 and 1 are lat/lon
            contexts.append(torch.stack([A, S, DT], dim=1))
        # For each source, retrieve the values tensor V and pad it to the size stored in img_sizes.
        # Also pad the coordinates tensor C to the same size.
        padded_images, padded_coords = [], []
        for (H, W), V, C in zip(self.img_sizes, images, coords):
            padded_images.append(F.pad(V, (0, W - V.shape[-1], 0, H - V.shape[-2])))
            padded_coords.append(F.pad(C, (0, W - C.shape[-1], 0, H - C.shape[-2])))
        # Apply the input convolutions.
        images = self.entry_block(images)
        coords = [self.coord_entry_block(coord) for coord in coords]
        # Apply the regridding blocks.
        for regridding_block in self.regridding_blocks:
            padded_images = regridding_block(padded_images, padded_coords, contexts)
        # Remove the padding.
        output = {}
        for k, (source_name, (H, W)) in enumerate(self.original_img_sizes.items()):
            output[source_name] = padded_images[k][:, :, :H, :W]
        return output
