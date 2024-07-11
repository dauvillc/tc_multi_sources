"""Implements a transformer model with an encoder-decoder architecture,
similarly to the original transformer model.
"""

import torch.nn as nn
from multi_sources.models.input_output_blocks import MultiSourceEntryConvBlock
from multi_sources.models.input_output_blocks import MultiSourceExitConvBlock
from multi_sources.models.utils import normalize_coords_across_sources


class Encoder(nn.Module):
    """The encoder part of the transformer, which computes features
    from the non-masked input sources.
    """

    def __init__(self, n_sources, channels, n_blocks, **block_kwargs):
        """
        Args:
            n_sources (int): Number of input sources.
            channels (int): Number of input and output channels.
            n_blocks (int): Number of blocks to use in the encoder.
            block_kwargs (dict): Keyword arguments to pass to the block class,
                of the form {block_name: {block_class: class, **block_kwargs}}.
        """
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            for block_name, kwargs in block_kwargs.items():
                bclass = kwargs["block_class"]
                args = {k: v for k, v in kwargs.items() if k != "block_class"}
                block = bclass(n_sources, n_sources, channels, **args)
                self.blocks.append(block)

    def forward(self, input_pixels, input_coords):
        x = input_pixels
        outputs = []
        for block in self.blocks:
            # In the encoder, the keys and queries are the same
            # and come from the input (non-masked) sources
            y = block(x, x, input_coords, input_coords)
            # Residual connection
            x = [x_ + y_ for x_, y_ in zip(x, y)]
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    """Decoder part of the transformer, which computes features
    from the masked input sources and the encoder outputs.
    """

    def __init__(self, n_encoded_sources, n_decoded_sources, channels, n_blocks, **block_kwargs):
        """
        Args:
            n_encoded_sources (int): Number of sources given to the encoder.
            n_decoded_sources (int): Number of sources in the decoder.
            channels (int): Number of input and output channels.
            block_classes (nn.Module): Classes of the blocks to use in the decoder.
            n_blocks (int): Number of blocks to use in the decoder.
            block_kwargs (dict): Keyword arguments to pass to the block classes,
                of the form {block_name: {block_class: class, **block_kwargs}}.
        """
        super().__init__()
        self.blocks = nn.ModuleList([])
        # Each decoder block will be made of two blocks from each class:
        # one with cross-attention to the encoder outputs and one with
        # self-attention to the decoder outputs.
        for _ in range(n_blocks):
            for block_name, kwargs in block_kwargs.items():
                bclass = kwargs["block_class"]
                args = {k: v for k, v in kwargs.items() if k != "block_class"}
                cross_att = bclass(n_encoded_sources, n_decoded_sources, channels, **args)
                self_att = bclass(n_decoded_sources, n_decoded_sources, channels, **args)
                self.blocks.append(nn.ModuleList([cross_att, self_att]))

    def forward(self, encoder_outputs, input_coords, masked_pixels, masked_coords):
        x = masked_pixels
        for blocks, output in zip(self.blocks, encoder_outputs):
            cross_att, self_att = blocks
            y = cross_att(output, x, input_coords, masked_coords)
            x = [x_ + y_ for x_, y_ in zip(x, y)]
            y = self_att(x, x, masked_coords, masked_coords)
            x = [x_ + y_ for x_, y_ in zip(x, y)]
        return x


class EncoderDecoderTransformer(nn.Module):
    """Transformer model with an encoder-decoder architecture.
    The model receives two types of inputs:
    - input sources, which are used to compute the encoder outputs.
    - masked sources coordinates, which are the coordinates of the sources
        that should be reconstructed by the decoder.
    """

    def __init__(
        self,
        n_input_sources,
        n_masked_sources,
        input_sources_channels,
        output_channels,
        inner_channels,
        n_blocks,
        **block_kwargs
    ):
        """
        Args:
            n_input_sources (int): Number of input sources.
            n_masked_sources (int): Number of masked sources.
            input_sources_channels (list of int): Number of channels in each input source.
            output_channels (list of int): Number of channels in each output source.
            inner_channels (int): Number of channels in the intermediate layers of the model.
            n_blocks (int): Number of blocks to use in the encoder and decoder.
            block_kwargs (dict): Keyword arguments which define the blocks, of the form
                {block_name: {block_class: class, **block_kwards}} where class is a subclass
                of nn.Module.
                Each block in the encoder and decoder will include one block of each
                class in the list, in the specified order.
                The constructors must be of the form
                `block_class(n_keys, n_queries, channels, **block_kwargs)`.
                The forward method must be of the form
                `forward(key_pixels, query_pixels, key_coords, query_coords)`.
        """
        if len(input_sources_channels) != n_input_sources:
            raise ValueError("len(input_sources_channels) must be equal to n_input_sources")
        if len(output_channels) != n_masked_sources:
            raise ValueError("len(output_channels) must be equal to n_masked_sources")
        super().__init__()
        # Create an entry convolutional block to project all input sources to a
        # common number of channels
        self.entry_input_sources = MultiSourceEntryConvBlock(
            input_sources_channels, inner_channels, downsample=False
        )
        self.entry_masked_sources = MultiSourceEntryConvBlock(
            [1] * n_masked_sources, inner_channels, downsample=False
        )
        self.encoder = Encoder(n_input_sources, inner_channels, n_blocks, **block_kwargs)
        self.decoder = Decoder(
            n_input_sources, n_masked_sources, inner_channels, n_blocks, **block_kwargs
        )
        # Create an exit convolutional block to project the output features to the
        # number of channels of the output sources
        self.exit_block = MultiSourceExitConvBlock(inner_channels, output_channels, upsample=False)

    def forward(self, input_sources, masked_sources_coords):
        """
        Args:
            input_sources: list of tuples (a, s, dt, c, v) where:
                - a is a tensor of shape (b,) whose value is 1 if the source is available
                    and -1 if it is not.
                - s (b,) gives the source index of each sample.
                - dt (b,) gives the relative time delta of each sample.
                - c (b, 3, H, W) gives the coordinates (lat lon land_mask);
                - v (b, c, H, W) gives the values of each sample.
            masked_sources_coords: list of tuples (s, dt, c), same as above.
                All masked sources are available and have no values since those
                are the targets.
        Returns:
            list of tensors of shape (b, c, H, W).
        """
        input_pixels = [v for _, _, _, _, v in input_sources]
        # Since there are no pixels for the masked sources, we'll start with
        # the land mask only.
        masked_pixels = [c[:, 2:3] for _, _, c in masked_sources_coords]
        # Use only the lat/lon coordinates
        input_coords = [c[:, :2] for _, _, _, c, _ in input_sources]
        masked_coords = [c[:, :2] for _, _, c in masked_sources_coords]
        input_coords = normalize_coords_across_sources(input_coords)
        masked_coords = normalize_coords_across_sources(masked_coords)
        # Project the input pixels to inner_channel
        input_pixels = self.entry_input_sources(input_pixels)
        masked_pixels = self.entry_masked_sources(masked_pixels)
        # Compute the encoder outputs
        encoder_outputs = self.encoder(input_pixels, input_coords)
        # Compute the decoder outputs
        x = self.decoder(encoder_outputs, input_coords, masked_pixels, masked_coords)
        # Project the output features to the number of channels of the output sources
        return self.exit_block(x)
