"""Implements a general backbone for the mask-autoencoding task
which can be used with custom blocks."""

import torch
import torch.nn as nn

from multi_sources.models.motif_single_embed.cross_attention import (
    MultisourcesWindowedCrossAttention,
)
from multi_sources.models.motif_single_embed.self_attention import (
    SeparateWindowedValuesCoordinatesAttention,
)
from multi_sources.models.motif_single_embed.small_layers import FeedForward


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for the multisource mask-autoencoding task.
    Each block consists of three layers:
    1. Multi-source anchored cross-attention;
    2. Separate windowed attention;
    3. Feed-forward network.
    Each layer is wrapped in an adaptive conditional normalization module.
    In this version, the embedded coordinates are fused with the values,
    either by summing or concatenating them.
    """

    def __init__(
        self,
        n_blocks,
        values_dim,
        coords_dim,
        cond_dim,
        sum_or_concat_coords="sum",
        att_inner_ratio=1.0,
        cross_att_inner_ratio_v=1.0,
        num_heads=8,
        cross_att_window_size=4,
        iwsa_window_size=8,
        mlp_inner_ratio=2,
        dropout=0.0,
    ):
        """
        Args:
            n_blocks (int): number of blocks in the backbone.
            values_dim (int): Embedding dimension for the values of each source.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            sum_or_concat_coords (str): Whether to sum or concatenate the coordinates
                with the values. Must be either "sum" or "concat".
                The coordinates embeddings are projected to the values dimension
                with a linear layer before summing or concatenating.
        """
        super().__init__()
        self.values_dim, self.coords_dim, self.cond_dim = values_dim, coords_dim, cond_dim
        self.sum_or_concat_coords = sum_or_concat_coords
        if sum_or_concat_coords == "concat":
            self.values_dim += coords_dim
            self.coords_proj = nn.Linear(coords_dim, coords_dim)
        elif sum_or_concat_coords == "sum":
            self.coords_proj = nn.Linear(coords_dim, values_dim)
        else:
            raise ValueError('sum_or_concat_coords must be either "sum" or "concat"')

        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            shifted = bool(block_idx % 2)  # Shifting in both attention modules
            block = nn.ModuleList(
                [
                    AdapativeConditionalNormalization(
                        MultisourcesWindowedCrossAttention(
                            values_dim,
                            inner_ratio_qk=att_inner_ratio,
                            window_size=cross_att_window_size,
                            inner_ratio_v=cross_att_inner_ratio_v,
                        ),
                        values_dim,
                        cond_dim,
                    ),
                    AdapativeConditionalNormalization(
                        SeparateWindowedValuesCoordinatesAttention(
                            values_dim,
                            att_inner_ratio,
                            num_heads=num_heads,
                            window_size=iwsa_window_size,
                            dropout=dropout,
                            shifted=shifted,
                        ),
                        values_dim,
                        cond_dim,
                    ),
                    AdapativeConditionalNormalization(
                        FeedForward(
                            values_dim,
                            inner_ratio=mlp_inner_ratio,
                            dropout=dropout,
                        ),
                        values_dim,
                        cond_dim,
                    ),
                ]
            )
            self.blocks.append(block)

    def forward(self, x):
        """
        Args:
            x (dict): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys "values", "coords" and
                optionally "conditioning",
                where (source_name, index) is a tuple containing the source name
                and the index of the observation (0 = most recent).
        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the predicted values of the tokens.
        """
        # Fuse the embedded coordinates with the values
        for source_index_pair, source_data in x.items():
            V, C = source_data["values"], source_data["coords"]
            C = self.coords_proj(C)
            if self.sum_or_concat_coords == "sum":
                x[source_index_pair]["values"] = V + C
            else:  # concat
                x[source_index_pair]["values"] = torch.cat([V, C], dim=-1)

        for block in self.blocks:
            # Update the values and coords via the successive layers
            for layer in block:
                # Apply the layer and update the values and coordinates
                layer_otp = layer(x)
                # Check that all sources are present in the output
                assert set(x.keys()) == set(layer_otp.keys())
                # Update the values
                for source_index_pair, source_output in layer_otp.items():
                    x[source_index_pair]["values"] = source_output["values"]

        # Return only the predicted values, like the original backbone
        return {
            source_index_pair: x[source_index_pair]["values"] for source_index_pair in x.keys()
        }


class AdapativeConditionalNormalization(nn.Module):
    """Wraps a torch module to apply adaptive conditional normalization, as in DiT.
    The module expects the data to include a key "conditioning", of same shape
    as the values. If that key is absent, no conditioning is applied and the inputs
    are just passed through LayerNorms, and residual connections are applied to
    the wrapped module's output.
    """

    def __init__(self, module, values_dim, cond_dim):
        """
        Args:
            module (nn.Module): The module to wrap.
            values_dim (int): Embedding dimension for the values of each source.
            cond_dim (int): Embedding dimension for the conditioning of each source.
        """
        super().__init__()
        self.module = module

        # Normalization and conditioning for values
        self.values_norm = nn.LayerNorm(values_dim)
        self.values_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, values_dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.values_cond_proj[1].weight)
        nn.init.zeros_(self.values_cond_proj[1].bias)

    def forward(self, data, *args, **kwargs):
        """Args:
            data (dict): Dictionary of inputs, such that
                data[(source_name, index)] contains the keys
                "values" and "conditioning", where
                (source_name, index) is a tuple containing the source name
                and observation index (0 = most recent).
            args, kwargs: Additional arguments for the wrapped module.
        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] is a dictionary containing:
                - "values": the predicted values of the tokens.
        """
        values_skips, values_gates = {}, {}
        modulated_data = {}

        for source_index_pair, source_data in data.items():
            # Create a new dict to avoid modifying the input one in-place
            modulated_data[source_index_pair] = {k: v for k, v in source_data.items()}

            # Process values
            values_skip = source_data["values"]
            values_x = self.values_norm(values_skip)
            values_skips[source_index_pair] = values_skip

            # Apply conditioning if available
            cond = source_data.get("conditioning", None)
            if cond is not None:
                # Apply conditioning to values
                values_shift, values_scale, values_gate = self.values_cond_proj(cond).chunk(
                    3, dim=-1
                )
                values_x = values_x * (values_scale + 1) + values_shift
                values_gates[source_index_pair] = values_gate

            # Save the module's inputs for that source
            modulated_data[source_index_pair]["values"] = values_x

        # Apply the wrapped module with the updated inputs
        module_output = self.module(modulated_data, *args, **kwargs)

        # Apply gates and skip connections
        output = {}
        for source_index_pair, source_output in module_output.items():
            # For the values
            values_out = source_output["values"]
            # Process values with gates and skip connections
            if source_index_pair in values_gates:
                values_out = (
                    values_out * values_gates[source_index_pair] + values_skips[source_index_pair]
                )
            else:
                values_out = values_out + values_skips[source_index_pair]

            output[source_index_pair] = {"values": values_out}

        return output
