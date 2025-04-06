"""Implements the FactorizedMultisourcesAttention class."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from multi_sources.models.attention import ValuesCoordinatesAttentionInternal
from multi_sources.models.small_layers import RMSNorm


class MultisourcesAnchoredCrossAttention(nn.Module):
    """Computes attention across the sources using an anchor points system.
    For each source, a set of anchor points is selected from the values and coordinates.
    Those tokens are then concatenated into a single sequence,
    on which the attention is computed. The updated tokens are then summed back
    to the original tokens of each source.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_ratio,
        anchor_points_spacing,
        num_heads=8,
        dropout=0.0,
        **kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_ratio (float): Ratio of the inner dimension to the values dimension.
            anchor_points_spacing (int): Spacing between the anchor points. For example,
                3 means that every third token along each axis is selected as an anchor point.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = inner_ratio * values_dim
        self.anchor_points_spacing = anchor_points_spacing
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, self.inner_dim, num_heads, dropout
        )

    def forward(self, inputs):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_coords", "embedded_values".
                The values are expected of shape (B, ..., Dv) and the coordinates of shape
                (B, ..., Dc), where ... is the spatial dimensions of the embedded source,
                e.g. (h, w) for 2D sources.

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # For each source:
        # - Read the length of that source's sequence
        # - Gather the anchor tokens from the values and coordinates
        # - Save the indices of the anchor tokens
        anchor_values, anchor_coords = {}, {}
        anchor_indices, n_anchors_list = {}, []
        for source_name, source_inputs in inputs.items():
            V, C = source_inputs["embedded_values"], source_inputs["embedded_coords"]
            spatial_dims = V.shape[1:-1]

            # For 1D sources, simply select the anchor points at regular intervals. 0D sources
            # are just 1D sources of length 1, so we can handle them here as well.
            if len(spatial_dims) == 1:
                n = spatial_dims[0]  # Number of tokens in the sequence
                n_anchor_points = n // self.anchor_points_spacing
                indices = torch.linspace(0, n - 1, n_anchor_points).long().to(V.device)
                anchor_values[source_name] = V[:, indices]
                anchor_coords[source_name] = C[:, indices]
                # Save the indices for later
                anchor_indices[source_name] = indices

            # For 2D sources, we want to select the anchor points in a grid pattern.
            elif len(spatial_dims) == 2:
                h, w = spatial_dims
                anchor_cols = torch.arange(0, w, self.anchor_points_spacing)
                anchor_cols += ((w - 1) % self.anchor_points_spacing) // 2  # Centering
                anchor_rows = torch.arange(0, h, self.anchor_points_spacing)
                anchor_rows += ((h - 1) % self.anchor_points_spacing) // 2

                anchor_v_s = V[:, anchor_rows[:, None], anchor_cols]
                anchor_v_s = rearrange(anchor_v_s, "b h w d -> b (h w) d")
                anchor_c_s = C[:, anchor_rows[:, None], anchor_cols]
                anchor_c_s = rearrange(anchor_c_s, "b h w d -> b (h w) d")
                anchor_values[source_name] = anchor_v_s
                anchor_coords[source_name] = anchor_c_s

                # Save the indices of the rows and columns for later
                anchor_indices[source_name] = (anchor_rows, anchor_cols)
                n_anchor_points = len(anchor_rows) * len(anchor_cols)

            # Save the number of anchor points for later
            n_anchors_list.append(n_anchor_points)

        # Concatenate the anchor tokens from all sources
        anchor_values = torch.cat([anchor_values[src] for src in inputs], dim=1)
        anchor_coords = torch.cat([anchor_coords[src] for src in inputs], dim=1)
        # Compute the attention across the anchor tokens
        anchor_values = self.attention(anchor_values, anchor_coords)
        # Split back the sequence of anchor tokens to the sources
        anchor_values = torch.split(anchor_values, n_anchors_list, dim=1)

        # Split the updated anchor tokens back to the sources and sum them to the original tokens
        outputs = {}
        for i, (source_name, source_inputs) in enumerate(inputs.items()):
            V = source_inputs["embedded_values"].clone()
            spatial_dims = V.shape[1:-1]

            if len(spatial_dims) == 1:
                indices = anchor_indices[source_name]
                anchor_values_i = anchor_values[i]
                V[:, indices] += anchor_values_i

            elif len(spatial_dims) == 2:
                anchor_rows, anchor_cols = anchor_indices[source_name]
                anchor_v_s = anchor_values[i]
                anchor_v_s = rearrange(anchor_v_s, "b (h w) d -> b h w d", h=len(anchor_rows))
                V[:, anchor_rows[:, None], anchor_cols] += anchor_v_s

            outputs[source_name] = V
        return outputs


# Adapted from
# https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical
# -vision-transformer-using-shifted-91cbf6abc678
class SeparateWindowedValuesCoordinatesAttention(nn.Module):
    """Attention block that computes the attention over each source independently,
    using a spatial window over the tokens as in the Swin Transformer.
    For 0D sources, does nothing."""

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_ratio,
        num_heads=8,
        dropout=0.0,
        window_size=8,
        shifted=0,
        block_idx=None,
        **kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_ratio (float): Ratio of the inner dimension to the values dimension.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            window_size (int): Number of tokens included in each window.
            shifted (bool): Whether to shift the windows by half the window size.
            block_idx (int): Index of the block in the model. If specified, the shifting
                will be determined by block_idx's parity: odd blocks shift the windows.
                If not None, it overrides the shifted argument.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = inner_ratio * values_dim
        self.head_dim = self.inner_dim // num_heads
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        if block_idx is not None:  # Override the shifted argument if block_idx is specified
            self.shifted = bool(block_idx % 2)

        # Projection to values queries, keys and values.
        # For the queries and keys, we apply an RMSNorm to stabilize the training.
        self.values_qk = nn.Sequential(
            nn.Linear(values_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.values_v = nn.Linear(values_dim, self.inner_dim)
        # Projection to coordinates queries and keys.
        self.coords_qk = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )

        self.dropout = nn.Dropout(dropout)

        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        self.indices = torch.tensor(
            np.array([[x, y] for x in range(window_size) for y in range(window_size)])
        )
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

        self.output_proj = nn.Sequential(
            nn.Linear(self.inner_dim, values_dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x[source_name] contains the keys "embedded_coords" and "embedded_values".

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, source_inputs in x.items():
            spatial_dims = source_inputs["embedded_values"].shape[1:-1]
            # If the source is 0D or 1D, do nothing
            if len(spatial_dims) < 2:
                outputs[source_name] = source_inputs["embedded_values"]
                continue
            h, w = spatial_dims
            # Project the values to queries, keys and values
            v = source_inputs["embedded_values"]
            v = torch.cat([self.values_qk(v), self.values_v(v)], dim=-1)
            v = rearrange(v, "b h w (d k) -> b h w d k", k=3)
            # Project the coordinates to queries and keys
            c = self.coords_qk(source_inputs["embedded_coords"])
            c = rearrange(c, "b h w (d k) -> b h w d k", k=2)
            # --> k=3 for queries, keys and values (no values for the coordinates)
            # Pad the values and coordinates so that h and w are multiples of the window size
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size
            v = F.pad(v, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
            c = F.pad(c, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
            # Roll the windows if needed
            if self.shifted:
                v = torch.roll(v, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
                c = torch.roll(c, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            # Reshape to windows and separate the heads
            v = rearrange(
                v,
                "b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            c = rearrange(
                c,
                "b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            # Compute the queries, keys and values
            qv, kv, vv = v.chunk(3, dim=6)
            qv, kv, vv = qv.squeeze(6), kv.squeeze(6), vv.squeeze(6)  # (b H Wh Ww w**2 d)
            qc, kc = c.chunk(2, dim=6)
            qc, kc = qc.squeeze(6), kc.squeeze(6)
            # Matrix product for the queries and keys from the values and coordinates
            dots = (qv @ kv.transpose(4, 5) + qc @ kc.transpose(4, 5)) / self.head_dim**0.5
            # (b H Wh Ww w**2 w**2)
            # Add the positional embeddings
            dots += self.pos_embeddings[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]
            # For shifted windows, compute the attention mask
            if self.shifted:
                row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
                row_mask[
                    -self.window_size * (self.window_size // 2) :,
                    0 : -self.window_size * (self.window_size // 2),
                ] = float("-inf")
                row_mask[
                    0 : -self.window_size * (self.window_size // 2),
                    -self.window_size * (self.window_size // 2) :,
                ] = float("-inf")
                column_mask = rearrange(
                    row_mask,
                    "(r w1) (c w2) -> (w1 r) (w2 c)",
                    w1=self.window_size,
                    w2=self.window_size,
                )
                dots[:, :, -1, :] += row_mask
                dots[:, :, :, -1] += column_mask
            # Deduce the attention weights
            y = F.softmax(dots, dim=-1) @ vv  # (b H Wh Ww w**2 d)
            # Reshape back to the original spatial layout of the tokens
            y = rearrange(
                y,
                "b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            # Remove the padding
            y = y[:, :h, :w, :]
            outputs[source_name] = self.output_proj(y)  # Back to values_dim
        return outputs
