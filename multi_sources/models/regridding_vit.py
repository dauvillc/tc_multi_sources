"""Implements class for a specific ViT architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Store the original image sizes for later reconstruction.
        self.original_img_sizes = {
            source_name: pair(img_size) for source_name, img_size in img_sizes.items()
        }
        # For each source, compute the next multiple of patch_size (for both height and width).
        # This is the size to which the image will be padded.
        self.img_sizes = []
        for source_name, (H, W) in self.original_img_sizes.items():
            self.img_sizes.append(
                (
                    H + patch_size[0] - H % patch_size[0],
                    W + patch_size[1] - W % patch_size[1],
                )
            )
        # Create a simple input Conv2d layer for each source, to project it to the
        # common number of channels.
        self.input_convs = nn.ModuleList(
            [
                nn.Conv2d(channels[source_name], 1, kernel_size=1)
                for source_name in img_sizes.keys()
            ]
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
        padded_images = [
            input_conv(padded_image)
            for input_conv, padded_image in zip(self.input_convs, padded_images)
        ]
        # Apply the regridding blocks.
        for regridding_block in self.regridding_blocks:
            padded_images = regridding_block(padded_images, padded_coords, contexts)
        # Remove the padding.
        output = {}
        for k, (source_name, (H, W)) in enumerate(self.original_img_sizes.items()):
            output[source_name] = padded_images[k][:, :, :H, :W]
        return output
