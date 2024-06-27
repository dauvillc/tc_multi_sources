"""Implements class for a specific ViT architecture."""

import torch
import torch.nn as nn
from multi_sources.models.regridding_vit_classes import AttentionRegriddingBlock
from multi_sources.models.regridding_vit_classes import EntryConvBlock
from multi_sources.models.utils import pad_to_next_multiple_of, pair
from multi_sources.models.utils import normalize_coords_across_sources


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
        kernel_size=3,
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
            downsample (bool): Whether to downsample the image at the input by 2.
            heads (int): Number of heads for the attention mechanism.
            dim_head (int): Dimension of each head.
            kernel_size (int or tuple of int): Kernel size of the internal convolutions.
            output_resnet_depth (int): Depth of the output ResNet.
                set to 0 to disable any convolutional layer at the output.
            output_resnet_inner_channels (int): Number of channels in the inner layers
                of the output
            output_resnet_kernel_size (int): Kernel size of the output ResNet.
        """
        super().__init__()
        # ModuleDict objects keys can't contain the character '.'; so replace it with '_'
        # in the keys of img_sizes and channels.
        self.n_sources = len(img_sizes)
        self.c_dim = c_dim
        self.depth = depth
        self.downsample = downsample
        self.heads = heads
        self.channels = list(channels.values())
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
                    inner_channels,
                    inner_channels,
                    inner_channels,
                    patch_size,
                    3,  # context: availability flag, dt, source index
                    c_dim,
                    heads,
                    dim_head,
                    kernel_size=kernel_size,
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
        images, coords, contexts, missing_values = [], [], [], []
        for source_name, (A, S, DT, C, D, V) in inputs.items():
            images.append(V)
            coords.append(C[:, :2])  # 0 and 1 are lat/lon
            contexts.append(torch.stack([A, S, DT], dim=1))
            # Missing values have been filled beforehand, but they can be located via
            # infinite values in D.
            missing_values.append(D == float("inf"))
        # Normalize the coordinates to be in the range [-1, 1].
        coords = normalize_coords_across_sources(coords, ignore_mask=missing_values)
        # Store the original image sizes.
        original_img_sizes = [image.shape[-2:] for image in images]
        if self.downsample:
            # Pad the images and coords, to have a size divisible by 2.
            images = [pad_to_next_multiple_of(image, 2) for image in images]
            coords = [pad_to_next_multiple_of(coord, 2) for coord in coords]
        # Apply the input convolutions.
        images = self.entry_block(images)
        coords = [self.coord_entry_block(coord) for coord in coords]
        # Apply the regridding blocks.
        for regridding_block in self.regridding_blocks:
            images = regridding_block(images, coords, contexts)
        # Apply the output transposed convolutions.
        images = [
            output_transposed_conv(image)
            for output_transposed_conv, image in zip(self.output_transposed_conv, images)
        ]
        # Remove the padding.
        output = {}
        for k, (source_name, (H, W)) in enumerate(zip(inputs.keys(), original_img_sizes)):
            output[source_name] = images[k][:, :, :H, :W]
        return output
