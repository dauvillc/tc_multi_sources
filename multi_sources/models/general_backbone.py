"""Implements a general backbone for the mask-autoencoding task,
which can be used with custom blocks."""

import torch
import torch.nn as nn
from multi_sources.models.utils import pair
from multi_sources.models.embedding_layers import LinearEmbedding


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for the multisource mask-autoencoding task.
    The model receives a list of tuples (s, dt, c, lm, v, m) as input, where:
    - s (b,) is the index of the source;
    - dt (b,) is the relative time delta between the latest source and the current source;
    - c (b, n, 2 * ph * pw) is the geographical coordinates of the tokens;
    - lm (b, n, ph * pw) is the land mask;
    - v (b, n, ph * pw) is the values of the tokens.
        Some tokens may be masked.
    - m (b, n, ph * pw) is a binary mask indicating which pixels are available.
    The model returns a list of tensors of shape (b, n, ph * pw), which are the
    predicted values of the tokens.
    """

    def __init__(self, patch_size, n_blocks, *, coords_dim=None, pixels_dim=None, **layer_kwargs):
        """
        Args:
            patch_size (int or tuple): size of the patches.
            n_blocks (int): number of blocks in the backbone.
            coords_dim (int): Embedding dimension for the geographical coordinates.
                Defaults to patch_height * patch_width * 2.
            pixels_dim (int): Embedding dimension for the pixel values.
                Defaults to patch_height * patch_width.
            **layer_kwargs: Dict defining the successive layers that compose the backbone,
                as {layer_name: layer_kwargs}.
                For each layer, the kwargs must include the key 'layer_class',
                which should be a nn.Module class. The other keys are the arguments
                to this class's constructor.
                The class's constructor must be of the form
                `layer_class(pixels_dim, coords_dim, **kwargs)`.
                The class's forward method must be of the form
                `forward(pixels_seq, coords_seq) -> pixels_seq`.
                Each block in the backbone will be composed of these layers, in the order
                they appear in the dict.
        """
        super().__init__()
        self.patch_size = pair(patch_size)
        ph, pw = self.patch_size
        if coords_dim is None:
            self.coords_dim = ph * pw * 2  # 2 for the latitude and longitude
        if pixels_dim is None:
            self.pixels_dim = ph * pw
        # Embedding layers for the coordinates
        self.coord_embedding = LinearEmbedding(ph * pw * 2, self.coords_dim)
        # Embedding layers for the pixel values
        self.pixel_embedding = LinearEmbedding(ph * pw, self.pixels_dim)
        # Embedding layers for the land mask
        self.landmask_embedding = LinearEmbedding(ph * pw, self.pixels_dim)
        # Embedding layers for the mask
        self.mask_embedding = LinearEmbedding(ph * pw, self.pixels_dim)
        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleList()
            for layer_name, layer_kwargs in layer_kwargs.items():
                layer_class = layer_kwargs.pop("layer_class")
                block.append(layer_class(self.pixels_dim, self.coords_dim, **layer_kwargs))
            self.blocks.append(block)
        # Output embedding layer to project the predicted pixels back to their
        # original dimension
        self.output_embedding = LinearEmbedding(self.pixels_dim, ph * pw)

    def forward(self, inputs):
        """Forward pass. See the class docstring for the input/output format."""
        # Concatenate the inputs along the token dimension to form the following tensors:
        # coords_seq, pixels_seq, landmask_seq, mask_seq
        s_seq, dt_seq, coords_seq, landmask_seq, pixels_seq, mask_seq = zip(*inputs)
        # Save the number of tokens in each sequence
        n_tokens_seq = [coords.shape[1] for coords in coords_seq]
        coords_seq = torch.cat(coords_seq, dim=1)
        pixels_seq = torch.cat(pixels_seq, dim=1)
        landmask_seq = torch.cat(landmask_seq, dim=1)
        mask_seq = torch.cat(mask_seq, dim=1)
        # Apply the embedding layers
        coords_seq = self.coord_embedding(coords_seq)
        pixels_seq = self.pixel_embedding(pixels_seq)
        landmask_seq = self.landmask_embedding(landmask_seq)
        mask_seq = self.mask_embedding(mask_seq)
        # Apply the blocks
        for block in self.blocks:
            for layer in block:
                pixels_seq = layer(pixels_seq, coords_seq)
        # Apply the output embedding layer
        pixels_seq = self.output_embedding(pixels_seq)
        # Split the sequences back into the original sequences
        pixels_seq = pixels_seq.split(n_tokens_seq, dim=1)
        return pixels_seq
