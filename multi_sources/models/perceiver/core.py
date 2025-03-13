"""Implements a Perceiver model (Jaegle et al., 2022) adapted to process multiple
tasks.
"""

import torch.nn as nn

from multi_sources.models.perceiver.transformer import ValuesCoordinatesTransformer
from multi_sources.models.perceiver.cross_att import (
    MultisourcePerceiverEncoder,
    MultisourcePerceiverDecoder,
)


class MultisourcePerceiver(nn.Module):
    """Implements a Perceiver model (Jaegle et al., 2022) adapted to process
    multiple tasks. The perceiver performs the following steps:
    - Create two learned latent arrays Lv and Lc of shapes (N, Dv) and (M, Dc),
        for the values and coordinates, respectively.
    - Perform cross-attention the input values and coordinates with the latent arrays.
    - Feed the updated latent arrays through a transformer.
    - Use the reconstructed sources' coordinates as queries in a final attention
        layer to reconstruct the sources' values.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        latent_size,
        num_blocks,
        att_inner_ratio=2,
        num_heads=8,
        mlp_inner_ratio=4,
        mlp_hidden_layers=2,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            latent_size (int): Index size of the latent arrays.
            num_blocks (int): Number of transformer blocks.
            att_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the self-attention layer.
            num_heads (int): Number of heads in the attention block.
            mlp_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the MLP.
            mlp_hidden_layers (int): Number of layers in the MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.encoder = MultisourcePerceiverEncoder(
            values_dim, coords_dim, latent_size, att_inner_ratio, num_heads, dropout
        )
        self.transformer = ValuesCoordinatesTransformer(
            values_dim,
            coords_dim,
            num_blocks,
            att_inner_ratio,
            num_heads,
            mlp_inner_ratio,
            mlp_hidden_layers,
            dropout,
        )
        self.decoder = MultisourcePerceiverDecoder(
            values_dim, coords_dim, att_inner_ratio, num_heads, dropout
        )

    def forward(self, x):
        """
        Args:
            x (dict of str to dict of str to tensor): Dictionary of inputs, such that
                x[src] contains at least the entries "embedded_values" and "embedded_coords". 
        
        Returns:
            dict of str to tensor: Dict Y such that Y[src] is the output for source src,
                of same shape as x[src]["embedded_values"].
        """
        # Encode the data within the latent values and coordinate arrays.
        Lv, Lc = self.encoder(x)

        # Apply the transformer.
        Lv, Lc = self.transformer(Lv, Lc)

        # Decode the sources.
        return self.decoder(x, Lv, Lc)
