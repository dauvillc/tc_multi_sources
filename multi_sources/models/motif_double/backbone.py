"""Implements a general backbone for the mask-autoencoding task that also updates coordinates,
which can be used with custom blocks."""

import torch.nn as nn

from multi_sources.models.motif_double.cross_attention import MultisourcesWindowedCrossAttention
from multi_sources.models.motif_double.self_attention import (
    SeparateWindowedValuesCoordinatesAttention,
)
from multi_sources.models.motif_double.small_layers import FeedForward


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for the multisource mask-autoencoding task that also updates coordinates.
    Each block consists of three layers:
    1. Multi-source ancnhored cross-attention;
    2. Separate windowed attention;
    3. Feed-forward network.
    Each layer is wrapped in an adaptive conditional normalization module,
    which applies conditional normalization based on the values and coordinates of each source.
    Within the backbone, each layer outputs a dictionary D such that D[source_index_pair]
    is itself a dict with keys "values" and "coords", containing the updated values and coordinates
    for the source identified by source_index_pair.
    """

    def __init__(
        self,
        n_blocks,
        coords_dim,
        values_dim,
        cond_dim=None,
        update_coords=False,
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
            coords_dim (int): Embedding dimension for the coordinates of each source.
            values_dim (int): Embedding dimension for the values of each source.
            cond_dim (int): Dimension of the conditioning vector for each source.
                if None, defaults to values_dim.
            update_coords (bool): Whether to update the coordinates in the backbone.
                If False, the embedded coordinates remain unchanged throughout the backbone.
        """
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else values_dim
        self.values_dim, self.coords_dim, self.cond_dim = values_dim, coords_dim, cond_dim
        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            # In the last block, we do not update the coordinates,
            # since they aren't used in the loss function.
            update_coords = update_coords and (block_idx < n_blocks - 1)
            modulate_coords = update_coords
            shifted = bool(block_idx % 2)  # Shifting in both attention modules
            block = nn.ModuleList(
                [
                    AdapativeConditionalNormalization(
                        MultisourcesWindowedCrossAttention(
                            values_dim,
                            coords_dim,
                            inner_ratio_qk=att_inner_ratio,
                            window_size=cross_att_window_size,
                            inner_ratio_v=cross_att_inner_ratio_v,
                            update_coords=update_coords,
                        ),
                        values_dim,
                        coords_dim,
                        cond_dim,
                        modulate_coords=modulate_coords,
                        expect_coords_in_output=update_coords,
                    ),
                    AdapativeConditionalNormalization(
                        SeparateWindowedValuesCoordinatesAttention(
                            values_dim,
                            coords_dim,
                            att_inner_ratio,
                            update_coords,
                            num_heads=num_heads,
                            window_size=iwsa_window_size,
                            dropout=dropout,
                            shifted=shifted,
                        ),
                        values_dim,
                        coords_dim,
                        cond_dim,
                        modulate_coords=modulate_coords,
                        expect_coords_in_output=update_coords,
                    ),
                    AdapativeConditionalNormalization(
                        FeedForward(
                            values_dim,
                            coords_dim,
                            update_coords,
                            inner_ratio=mlp_inner_ratio,
                            dropout=dropout,
                        ),
                        values_dim,
                        coords_dim,
                        cond_dim,
                        modulate_coords=modulate_coords and (block_idx < n_blocks - 1),
                        expect_coords_in_output=update_coords,
                    ),
                ]
            )
            self.blocks.append(block)

    def forward(self, x):
        """
        Args:
            x (dict): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys "coords" and "values".
                where (source_name, index) is a tuple containing the source name
                and the index of the observation (0 = most recent).
        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the predicted values of the tokens.
        """
        for block in self.blocks:
            # Update the values and coords via the successive layers
            for layer in block:
                # Apply the layer and update the values and coordinates
                layer_otp = layer(x)
                # Check that all sources are present in the output
                assert set(x.keys()) == set(layer_otp.keys())
                # Update the values and coordinates
                for source_index_pair, source_output in layer_otp.items():
                    x[source_index_pair]["values"] = source_output["values"]
                    x[source_index_pair]["coords"] = source_output["coords"]

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

    def __init__(
        self, module, values_dim, coords_dim, cond_dim, modulate_coords, expect_coords_in_output
    ):
        """
        Args:
            module (nn.Module): The module to wrap.
            values_dim (int): Embedding dimension for the values of each source.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            cond_dim (int): Dimension of the conditioning vector for each source.
            modulate_coords (bool): Whether to modulate the coordinates with the conditioning.
            expect_coords_in_output (bool): Whether the wrapped module is expected to output
                coordinates in addition to values. If True, the output will contain both
                "values" and "coords" keys. If False, only expects "values" key.
                If modulate_coords is False, this argument is ignored.
        """
        super().__init__()
        self.module = module
        self.expect_coords_in_output = expect_coords_in_output
        self.cond_dim = cond_dim
        self.modulate_coords = modulate_coords
        if not self.modulate_coords:
            expect_coords_in_output = False

        # Normalization and conditioning for values
        self.values_norm = nn.LayerNorm(values_dim)
        self.values_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, values_dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.values_cond_proj[1].weight)
        nn.init.zeros_(self.values_cond_proj[1].bias)

        if self.modulate_coords:
            # Normalization and conditioning for coordinates. If the output module does not
            # output coordinates, no gate projection is applied for the coordinates.
            cond_proj_dim = coords_dim * 3 if expect_coords_in_output else coords_dim * 2
            self.coords_norm = nn.LayerNorm(coords_dim)
            self.coords_cond_proj = nn.Sequential(
                nn.SiLU(), nn.Linear(self.cond_dim, cond_proj_dim)
            )
            # Initialize the weights of the conditional normalization to zero (no effect)
            nn.init.zeros_(self.coords_cond_proj[1].weight)
            nn.init.zeros_(self.coords_cond_proj[1].bias)

    def forward(self, data, *args, **kwargs):
        """Args:
            data (dict): Dictionary of inputs, such that
                data[(source_name, index)] contains the keys
                "values", "coords", and "conditioning", where
                (source_name, index) is a tuple containing the source name
                and observation index (0 = most recent).
            args, kwargs: Additional arguments for the wrapped module.
        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] is a dictionary containing:
                - "values": the predicted values of the tokens.
                - "coords": the predicted coordinates of the tokens (if expect_coords_in_output
                   is True).
        """
        values_skips, values_gates = {}, {}
        coords_skips, coords_gates = {}, {}
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

                # Apply conditioning to coordinates
                if not self.modulate_coords:
                    # If not modulating coordinates, don't even apply the layer norm.
                    coords_x = source_data["coords"]
                    coords_skips[source_index_pair] = coords_x
                else:
                    coords_x = self.coords_norm(source_data["coords"])
                    coords_skips[source_index_pair] = coords_x
                    coords_proj = self.coords_cond_proj(cond)
                    if self.expect_coords_in_output:
                        coords_shift, coords_scale, coords_gate = coords_proj.chunk(3, dim=-1)
                        # Only produce a gate projection if the wrapped module outputs
                        # new coordinates.
                        coords_gates[source_index_pair] = coords_gate
                    else:
                        coords_shift, coords_scale = coords_proj.chunk(2, dim=-1)
                    coords_x = coords_x * (coords_scale + 1) + coords_shift

            # Save the module's inputs for that source
            modulated_data[source_index_pair]["values"] = values_x
            modulated_data[source_index_pair]["coords"] = coords_x

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

            # Process coordinates with gates and skip connections
            if self.expect_coords_in_output:
                # If the wrapped module outputs coordinates, apply the gates and skips
                # (or just the skips if no gates are present).
                coords_out = source_output["coords"]
                if source_index_pair in coords_gates:
                    coords_out = (
                        coords_out * coords_gates[source_index_pair]
                        + coords_skips[source_index_pair]
                    )
                else:
                    coords_out = coords_out + coords_skips[source_index_pair]
            else:
                # If the wrapped module does not output coordinates, just forward the
                # unchanged coordinates.
                coords_out = coords_skips[source_index_pair]

            output[source_index_pair] = {"values": values_out, "coords": coords_out}

        return output
