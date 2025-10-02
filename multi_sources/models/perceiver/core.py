"""Implements a Perceiver model (Jaegle et al., 2022) adapted to process multiple
tasks.
"""

import torch
import torch.nn as nn

from multi_sources.models.perceiver.ada_ln import CrossAttAdaLN, DecoderAdaLN, LatentAdaLN
from multi_sources.models.perceiver.cross_att import (
    MultisourcePerceiverCrossAttention,
    MultisourcePerceiverDecoder,
)
from multi_sources.models.perceiver.self_att import ValuesCoordinatesSelfAttention
from multi_sources.models.perceiver.small_layers import VCMLP


class MultisourcePerceiver(nn.Module):
    """Implements a Perceiver model (Jaegle et al., 2022) adapted to process
    multiple tasks. The perceiver performs the following steps:
    - Create three learned latent arrays Lv, Lc and Ld of shapes (N, Dv), (N, Dc) and (N, Dv)
        for the values, coordinates and conditioning respectively.
    - Perform cross-attention between the input data and the latent arrays, to fill the
        latter with latent representations of the input values, coords and conditioning.
    - Feed the updated latent arrays through a transformer.
    - Use the reconstructed sources' coordinates as queries in a final attention
        layer to reconstruct the sources' values.
    Every block is wrapped in an Adaptive LayerNorm (AdaLN).
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        latent_seq_len,
        num_blocks,
        att_inner_ratio=2,
        num_heads=8,
        mlp_inner_ratio=4,
        mlp_hidden_layers=2,
        cross_att_every_n_blocks=1,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            latent_seq_len (int): Sequence length of the latent arrays.
            num_blocks (int): Number of transformer blocks.
            att_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the self-attention layer.
            num_heads (int): Number of heads in the attention block.
            mlp_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the MLP.
            mlp_hidden_layers (int): Number of layers in the MLP.
            cross_att_every_n_blocks (int): Number of transformer blocks between two
                cross-attention layers. If None, only the encoding cross-attention is used.
            dropout (float): Dropout rate.
        """
        super().__init__()
        # Create the latent arrays.
        self.latent_values = nn.Parameter(torch.randn(latent_seq_len, values_dim))
        self.latent_coords = nn.Parameter(torch.randn(latent_seq_len, coords_dim))
        self.latent_condit = nn.Parameter(torch.randn(latent_seq_len, values_dim))

        # Create the successive blocks of the transformer.
        self.blocks = nn.ModuleList()
        for k in range(num_blocks):
            block = nn.ModuleList()

            # Optional cross-attention layer.
            include_cross_att = (
                cross_att_every_n_blocks is not None and k % cross_att_every_n_blocks == 0
            ) or (k == 0)
            if include_cross_att:
                block.append(
                    CrossAttAdaLN(
                        MultisourcePerceiverCrossAttention(
                            values_dim, coords_dim, att_inner_ratio, num_heads, dropout
                        ),
                        values_dim,
                        coords_dim,
                        modulate_latents=k != 0,  # Ld doesn't contain info at first cross-att
                    )
                )

            # Self-attention
            block.append(
                LatentAdaLN(
                    ValuesCoordinatesSelfAttention(
                        values_dim, coords_dim, att_inner_ratio, num_heads, dropout=dropout
                    ),
                    values_dim,
                    coords_dim,
                )
            )

            # MLP
            block.append(
                LatentAdaLN(
                    VCMLP(
                        values_dim,
                        coords_dim,
                        mlp_hidden_layers,
                        mlp_inner_ratio,
                        dropout=dropout,
                    ),
                    values_dim,
                    coords_dim,
                )
            )

            self.blocks.append(block)

        # Final decoder
        self.decoder = DecoderAdaLN(
            MultisourcePerceiverDecoder(
                values_dim,
                coords_dim,
                att_inner_ratio=att_inner_ratio,
                num_heads=num_heads,
                mlp_hidden_layers=mlp_hidden_layers,
                mlp_inner_ratio=mlp_inner_ratio,
                dropout=dropout,
            ),
            values_dim,
            coords_dim,
        )

    def forward(self, x):
        """
        Args:
            x (dict of str to dict of str to tensor): Dictionary of inputs, such that
                x[src] contains at least the entries "values", "coords" and
                "conditioning".

        Returns:
            dict of str to tensor: Dict Y such that Y[src] is the output for source src,
                of same shape as x[src]["values"].
        """
        batch_size = next(iter(x.values()))["values"].shape[0]

        # Expand the latent arrays to the batch size.
        Lv = self.latent_values.unsqueeze(0).expand(batch_size, -1, -1)
        Lc = self.latent_coords.unsqueeze(0).expand(batch_size, -1, -1)
        Ld = self.latent_condit.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply the successive blocks of the transformer.
        for block in self.blocks:
            if len(block) == 3:
                cross_att, self_att, mlp = block
                Lv, Lc, Ld = cross_att(x, Lv, Lc, Ld)
            else:
                self_att, mlp = block
            Lv, Lc, Ld = self_att(Lv, Lc, Ld)
            Lv, Lc, Ld = mlp(Lv, Lc, Ld)

        # Decode the output values using the reconstructed latent arrays.
        Y = self.decoder(x, Lv, Lc, Ld)

        return Y
