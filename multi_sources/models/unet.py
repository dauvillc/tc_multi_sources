import torch
import torch.nn as nn

from multi_sources.models.adaptive_normalization import AdaCondLN
from multi_sources.models.downsample import PatchMerging, PatchSplitting
from multi_sources.models.factorized_attention import (
    MultisourcesAnchoredCrossAttention,
    SeparateWindowedValuesCoordinatesAttention,
)
from multi_sources.models.small_layers import FeedForward


class MultisourceDownsamplingBlockAttention(nn.Module):
    """Implements a downsampling block for multisource data that uses
    factorized attention layers."""

    def __init__(
        self,
        values_dim,
        coords_dim,
        att_inner_ratio,
        att_heads,
        anchor_points_spacing,
        window_size,
        window_shifted,
        mlp_hidden_layers,
        mlp_inner_ratio,
        dropout,
        resizing="downsample",
    ):
        """
        Args:
            values_dim (int): Dimension of the embedded values in the input and output.
            coords_dim (int): Dimension of the embedded coordinates in the input and output.
            att_inner_ratio (int): Inner ratio for the attention layers.
            att_heads (int): Number of heads for the attention layers.
            anchor_points_spacing (int): Spacing between anchor points in the cross-sources
                attention.
            window_size (int): Size of the window for the self-attention.
            window_shifted (bool): Whether the window is shifted by half the window size.
            mlp_hidden_layers (int): Number of hidden layers in the MLPs.
            mlp_inner_ratio (int): Inner ratio for the MLPs.
            dropout (float): Dropout rate.
            resizing (str): Type of resizing:
                - "downsample" to downsample the input,
                - "upsample" to upsample the input,
                - "none" to keep the input as is.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.att_inner_ratio = att_inner_ratio
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_inner_ratio = mlp_inner_ratio

        # Patch merging or splitting
        self.values_input_norm = nn.LayerNorm(values_dim)
        self.coords_input_norm = nn.LayerNorm(coords_dim)
        if resizing == "downsample":
            self.values_resizing = PatchMerging(values_dim)
            self.coords_resizing = PatchMerging(coords_dim)
            # The patch merging doubles the features
            values_dim *= 2
            coords_dim *= 2
        elif resizing == "upsample":
            self.values_resizing = PatchSplitting(values_dim)
            self.coords_resizing = PatchSplitting(coords_dim)
            # The patch splitting halves the features
            values_dim //= 2
            coords_dim //= 2
        elif resizing == "none":
            self.values_resizing = nn.Identity()
            self.coords_resizing = nn.Identity()

        # Factorized attention
        self.cross_att = AdaCondLN(
            MultisourcesAnchoredCrossAttention(
                values_dim, coords_dim, att_inner_ratio, anchor_points_spacing, att_heads, dropout
            ),
            values_dim,
            coords_dim,
        )
        self.self_att = AdaCondLN(
            SeparateWindowedValuesCoordinatesAttention(
                values_dim,
                coords_dim,
                att_inner_ratio,
                att_heads,
                dropout,
                window_size,
                window_shifted,
            ),
            values_dim,
            coords_dim,
        )

        # MLP
        self.mlp = AdaCondLN(
            FeedForward(values_dim, coords_dim, dropout, inner_ratio=mlp_inner_ratio),
            values_dim,
            coords_dim,
        )

    def forward(self, x):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x['source_name'] contains at least the keys "embedded_values" and
                "embedded_coords".
        Returns:
            dict of str: dict of str: tensor: Dictionary of outputs, such that
                outputs['source_name'] contains the keys "embedded_values" and
                "embedded_coords".
        """
        # Copy the input dictionary to avoid modifying the input
        x = {src: {key: val for key, val in data.items()} for src, data in x.items()}

        # Patch merging or splitting
        for src, data in x.items():
            V, C = data["embedded_values"], data["embedded_coords"]
            V = self.values_input_norm(V)
            C = self.coords_input_norm(C)
            V = self.values_resizing(V)
            C = self.coords_resizing(C)
            x[src] = {"embedded_values": V, "embedded_coords": C}

        # Factorized attention
        y = self.cross_att(x)
        for src, y_s in y.items():
            x[src] = {"embedded_values": y_s, "embedded_coords": x[src]["embedded_coords"]}
        y = self.self_att(x)
        for src, y_s in y.items():
            x[src] = {"embedded_values": y_s, "embedded_coords": x[src]["embedded_coords"]}

        # MLP
        y = self.mlp(x)
        for src, y_s in y.items():
            x[src]["embedded_values"] = y_s

        return x


class MultisourceUNet(nn.Module):
    """Implements a UNet model for multisource data, with factorized attention
    at the bottleneck layer.
    The module receives as input a dict {source_name: source_data}, where
    D[s] contains the following keys:
    * 'embedded_values': tensor of shape (B, ..., Dv) where ... are the spatial
        dimensions, e.g. (h, w) for 2D sources and (,) for 0D sources.
    * 'embedded_coords': tensor of the same shape as the embedded values.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        n_downsamplings,
        n_bottleneck_blocks,
        att_inner_ratio=1,
        att_heads=8,
        anchor_points_spacing=4,
        window_size=8,
        mlp_hidden_layers=2,
        mlp_inner_ratio=2,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Dimension of the embedded values in the input and output.
            coords_dim (int): Dimension of the embedded coordinates in the input and output.
            n_downsamplings (int): Number of downsampling (and thus upsampling) blocks.
            n_bottleneck_blocks (int): Number of blocks in the bottleneck layer.
            att_inner_ratio (int): Inner ratio for the attention layers.
            att_heads (int): Number of heads for the attention layers.
            anchor_points_spacing (int): Spacing between anchor points in the cross-sources
                attention.
            window_size (int): Size of the window for the self-attention.
            mlp_hidden_layers (int): Number of hidden layers in the MLPs.
            mlp_inner_ratio (int): Inner ratio for the MLPs.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim

        # Downsampling blocks. The windows are shifted every other block.
        self.downsamplings = nn.ModuleList(
            [
                MultisourceDownsamplingBlockAttention(
                    values_dim * 2**i,  # As the downsampling doubles the features.
                    coords_dim * 2**i,
                    att_inner_ratio,
                    att_heads,
                    anchor_points_spacing,
                    window_size,
                    window_shifted=(i % 2 == 1),
                    mlp_hidden_layers=mlp_hidden_layers,
                    mlp_inner_ratio=mlp_inner_ratio,
                    dropout=dropout,
                    resizing="downsample",
                )
                for i in range(n_downsamplings)
            ]
        )

        # Bottleneck blocks
        self.bottleneck = nn.ModuleList(
            [
                MultisourceDownsamplingBlockAttention(
                    values_dim * 2**n_downsamplings,
                    coords_dim * 2**n_downsamplings,
                    att_inner_ratio,
                    att_heads,
                    anchor_points_spacing,
                    window_size,
                    window_shifted=(i % 2 == 1),
                    mlp_hidden_layers=mlp_hidden_layers,
                    mlp_inner_ratio=mlp_inner_ratio,
                    dropout=dropout,
                    resizing="none",
                )
                for i in range(n_bottleneck_blocks)
            ]
        )

        # Upsampling blocks
        self.upsamplings = nn.ModuleList(
            [
                MultisourceDownsamplingBlockAttention(
                    values_dim * 2 ** (n_downsamplings - i + 1),  # Account for the skip.
                    coords_dim * 2 ** (n_downsamplings - i + 1),
                    att_inner_ratio,
                    att_heads,
                    anchor_points_spacing,
                    window_size,
                    window_shifted=(i % 2 == 1),
                    mlp_hidden_layers=mlp_hidden_layers,
                    mlp_inner_ratio=mlp_inner_ratio,
                    dropout=dropout,
                    resizing="upsample",
                )
                for i in range(n_downsamplings)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x['source_name'] contains at least the keys "embedded_values" and
                "embedded_coords".
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs['source_name'] contains the predicted values of the tokens.
        """
        # Copy the input dictionary to avoid modifying the input
        x = {src: {key: val for key, val in data.items()} for src, data in x.items()}

        # Downsampling: apply the downsampling blocks and save the outputs for the
        # skip connections.
        skip_connections = []
        for block in self.downsamplings:
            x = block(x)
            skip_connections.append(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # Upsampling: apply the upsampling blocks and add the skip connections.
        for block, skip in zip(self.upsamplings, reversed(skip_connections)):
            # Concatenate the skip connection with the upsampled features
            for src, data in x.items():
                V, C = data["embedded_values"], data["embedded_coords"]
                V_skip, C_skip = skip[src]["embedded_values"], skip[src]["embedded_coords"]
                V = torch.cat([V, V_skip], dim=-1)
                C = torch.cat([C, C_skip], dim=-1)
                x[src] = {"embedded_values": V, "embedded_coords": C}
            # Apply the upsampling block
            x = block(x)

        # Return only the predicted values
        return {source_name: x[source_name]["embedded_values"] for source_name in x.keys()}
