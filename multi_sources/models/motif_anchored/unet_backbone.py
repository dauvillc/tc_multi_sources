"""Implements a general backbone for the mask-autoencoding task that also updates coordinates,
which can be used with custom blocks."""

import torch.nn as nn

from multi_sources.models.motif_double.cross_attention import MultisourcesWindowedCrossAttention
from multi_sources.models.motif_double.patch_merging import (
    MultiSourcePatchMerging,
    MultiSourcePatchSplitting,
)
from multi_sources.models.motif_double.self_attention import (
    SeparateWindowedValuesCoordinatesAttention,
)
from multi_sources.models.motif_double.small_layers import FeedForward


class MultisourceGeneralBackboneUNet(nn.Module):
    """General backbone for the multisource mask-autoencoding task.
    Each block consists of three layers:
    1. Multi-source ancnhored cross-attention;
    2. Separate windowed attention;
    3. Feed-forward network.
    4. Patch merging (downsampling) or patch splitting (upsampling), except in the middle
        and last blocks.
    The outputs of downsampling layers are added to the inputs of the corresponding
    upsampling layers, like in a UNet.
    Each layer is wrapped in an adaptive conditional normalization module,
    which applies conditional normalization based on the embedded conditioning of each source.
    Within the backbone, each layer outputs a dictionary D such that D[source_index_pair]
    is itself a dict with keys "values" and "coords", containing the updated values and coordinates
    for the source identified by source_index_pair.
    """

    def __init__(
        self,
        n_blocks_per_stage,
        coords_dim,
        values_dim,
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
            n_blocks_per_stage (list of int): Number of blocks in each stage of the UNet,
                for the downsampling half. The first value is the number of blocks before
                the first downsampling, and the last value is the number of blocks
                in the bottleneck. The upsampling half won't add more blocks in the bottleneck,
                but will have the same number of blocks as the downsampling half in each stage.
                For instance, [1, 2, 3] means 1 block before the first downsampling,
                2 blocks after the first downsampling, and 3 blocks in the bottleneck.
                The upsampling half will then have 2 blocks after the first upsampling,
                and 1 block after the second upsampling.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            values_dim (int): Embedding dimension for the values of each source.
        """
        super().__init__()
        if values_dim % (2 ** (len(n_blocks_per_stage) - 1)) != 0:
            raise ValueError("values_dim must be divisible by 2^(number of downsampling stages).")
        self.values_dim, self.coords_dim = values_dim, coords_dim

        def create_block(block_idx, update_coords=False, modulate_coords=False):
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
                        ),
                        values_dim,
                        coords_dim,
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
                        modulate_coords=modulate_coords,
                        expect_coords_in_output=update_coords,
                    ),
                ]
            )
            return block

        # Create the downsampling stages, as a list of (stage blocks, merging layer)
        self.downsampling_blocks = nn.ModuleList()
        block_cnt = 0
        for stage_idx, n_blocks in enumerate(n_blocks_per_stage[:-1]):
            stage_blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                stage_blocks.append(create_block(block_idx + block_cnt))
                block_cnt += 1
            # Add a patch merging layer after each stage except the last one
            merging = MultiSourcePatchMerging(values_dim, coords_dim)
            values_dim *= 2
            coords_dim *= 2
            self.downsampling_blocks.append(nn.ModuleList([stage_blocks, merging]))

        # Create the bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList()
        n_bottleneck_blocks = n_blocks_per_stage[-1]
        for block_idx in range(n_bottleneck_blocks):
            self.bottleneck_blocks.append(create_block(block_idx + block_cnt))
            block_cnt += 1

        # Create the upsampling stages as a list of (splitting layer, stage blocks)
        self.upsampling_blocks = nn.ModuleList()
        for stage_idx, n_blocks in reversed(list(enumerate(n_blocks_per_stage[:-1]))):
            # Add a patch splitting layer before each stage
            splitting = MultiSourcePatchSplitting(values_dim, coords_dim)
            values_dim //= 2
            coords_dim //= 2
            stage_blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                stage_blocks.append(create_block(block_idx + block_cnt))
                block_cnt += 1
            self.upsampling_blocks.append(nn.ModuleList([splitting, stage_blocks]))

    def forward(self, x):
        """
        Args:
            x (dict): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys "coords",
                "values" and "conditioning".
        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the predicted values of the tokens.
        """
        # Downsampling half
        skips = []
        for stage_blocks, merging in self.downsampling_blocks:
            for block in stage_blocks:
                for layer in block:
                    layer_output = layer(x)
                    for src, output in layer_output.items():
                        x[src]["values"] = output["values"]
            skips.append(x)
            x = merging(x)

        # Bottleneck
        for block in self.bottleneck_blocks:
            for layer in block:
                layer_output = layer(x)
                for src, output in layer_output.items():
                    x[src]["values"] = output["values"]

        # Upsampling half
        for splitting, stage_blocks in self.upsampling_blocks:
            x = splitting(x)
            skip = skips.pop()
            # Add skip connections from the downsampling half
            for src in x.keys():
                # The upsampled data may be spatially larger than the skip connection data,
                # due to odd sizes. In that case, crop the upsampled data to the skip size.
                skip_values = skip[src]["values"]
                upsampled_values = x[src]["values"]
                upsampled_coords = x[src]["coords"]
                upsampled_cond = x[src].get("conditioning", None)
                if upsampled_values.shape[1:-1] != skip_values.shape[1:-1]:
                    crop_slices = (
                        (slice(None),)
                        + tuple(slice(0, s) for s in skip_values.shape[1:-1])
                        + (slice(None),)
                    )  # Keep batch and feature dimensions
                    upsampled_values = upsampled_values[crop_slices]
                    upsampled_coords = upsampled_coords[crop_slices]
                    if upsampled_cond is not None:
                        upsampled_cond = upsampled_cond[crop_slices]
                x[src]["values"] = upsampled_values + skip_values
                x[src]["coords"] = upsampled_coords
                if upsampled_cond is not None:
                    x[src]["conditioning"] = upsampled_cond
            for block in stage_blocks:
                for layer in block:
                    layer_output = layer(x)
                    for src, output in layer_output.items():
                        x[src]["values"] = output["values"]

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

    def __init__(self, module, values_dim, coords_dim, modulate_coords, expect_coords_in_output):
        """
        Args:
            module (nn.Module): The module to wrap.
            values_dim (int): Embedding dimension for the values of each source.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            modulate_coords (bool): Whether to modulate the coordinates with the conditioning.
            expect_coords_in_output (bool): Whether the wrapped module is expected to output
                coordinates in addition to values. If True, the output will contain both
                "values" and "coords" keys. If False, only expects "values" key.
                If modulate_coords is False, this argument is ignored.
        """
        super().__init__()
        self.module = module
        self.expect_coords_in_output = expect_coords_in_output
        self.modulate_coords = modulate_coords
        if not self.modulate_coords:
            expect_coords_in_output = False

        # Normalization and conditioning for values
        self.values_norm = nn.LayerNorm(values_dim)
        self.values_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, values_dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.values_cond_proj[1].weight)
        nn.init.zeros_(self.values_cond_proj[1].bias)

        if self.modulate_coords:
            # Normalization and conditioning for coordinates. If the output module does not
            # output coordinates, no gate projection is applied for the coordinates.
            cond_proj_dim = coords_dim * 3 if expect_coords_in_output else coords_dim * 2
            self.coords_norm = nn.LayerNorm(coords_dim)
            self.coords_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, cond_proj_dim))
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
