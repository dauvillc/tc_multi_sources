"""Implements a transformer model with an encoder-decoder architecture,
similarly to the original transformer model.
"""

import torch.nn as nn
from multi_sources.models.input_output_blocks import MultiSourceEntryConvBlock
from multi_sources.models.input_output_blocks import MultiSourceExitConvBlock


class Encoder(nn.Module):
    """The encoder part of the transformer, which computes features
    from the non-masked input sources.
    """

    def __init__(self, n_sources, channels, block_class, n_blocks, **block_kwargs):
        """
        Args:
            n_sources (int): Number of input sources.
            channels (int): Number of input and output channels.
            block_class (nn.Module): Class of the block to use in the encoder.
            n_blocks (int): Number of blocks to use in the encoder.
            block_kwargs (dict): Keyword arguments to pass to the block class.
                The block class constructor must be of the form
                `block_class(n_keys, n_queries, channels, **block_kwargs)`.
                Its forward method must be of the form `forward(keys, queries)`.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [block_class(n_sources, n_sources, channels, **block_kwargs) for _ in range(n_blocks)]
        )

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x, x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    """Decoder part of the transformer, which computes features
    from the masked input sources and the encoder outputs.
    """

    def __init__(
        self, n_encoded_sources, n_decoded_sources, channels, block_class, n_blocks, **block_kwargs
    ):
        """
        Args:
            n_encoded_sources (int): Number of sources given to the encoder.
            n_decoded_sources (int): Number of sources in the decoder.
            channels (int): Number of input and output channels.
            block_class (nn.Module): Class of the block to use in the decoder.
            n_blocks (int): Number of blocks to use in the decoder.
            block_kwargs (dict): Keyword arguments to pass to the block class.
                The block class constructor must be of the form
                `block_class(n_keys, n_queries, channels, **block_kwargs)`.
                Its forward method must be of the form `forward(keys, queries)`.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                block_class(n_encoded_sources, n_decoded_sources, channels, **block_kwargs)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, encoder_outputs, x):
        for block, output in zip(self.blocks, encoder_outputs):
            x = block(output, x)
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
        block_class,
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
            block_class (nn.Module): Class of the block to use in the encoder and decoder.
               Its constructor must be of the form
               `block_class(n_keys, n_queries, channels, **block_kwargs)`.
               Its forward method must be of the form `forward(keys, queries)`.
            n_blocks (int): Number of blocks to use in the encoder and decoder.
            block_kwargs (dict): Keyword arguments to pass to the block class.
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
                [3] * n_masked_sources, inner_channels, downsample=False
        )
        self.encoder = Encoder(
            n_input_sources, inner_channels, block_class, n_blocks, **block_kwargs
        )
        self.decoder = Decoder(
            n_input_sources,
            n_masked_sources,
            inner_channels,
            block_class,
            n_blocks,
            **block_kwargs
        )
        # Create an exit convolutional block to project the output features to the
        # number of channels of the output sources
        self.exit_block = MultiSourceExitConvBlock(inner_channels, output_channels, upsample=False)

    def forward(self, input_sources, masked_sources_coords):
        """
        Args:
            input_sources: list of tensors of shape (b, c, H, W).
            masked_sources_coords: list of tensors of shape (b, 3, H, W).
        Returns:
            list of tensors of shape (b, c, H, W).
        """
        # Project the input sources and the masked sources coordinates to
        # inner_channels
        x = self.entry_input_sources(input_sources)
        masked_sources_coords = self.entry_masked_sources(masked_sources_coords)
        # Compute the encoder outputs
        encoder_outputs = self.encoder(x)
        # Compute the decoder outputs
        x = self.decoder(encoder_outputs, masked_sources_coords)
        # Project the output features to the number of channels of the output sources
        return self.exit_block(x)
