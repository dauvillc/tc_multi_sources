"""Implements the FactorizedMultisourcesAttention class."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from multi_sources.models.attention import ValuesCoordinatesAttentionInternal


class MultisourcesAnchoredCrossAttention(nn.Module):
    """Computes attention across the sources using an anchor points system.
    K evenly spaced anchor tokens are gathered from each source. Those tokens are then
    concatenated into a single sequence, on which the attention is computed. The updated
    tokens are then summed back to the original tokens of each source.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_dim,
        num_anchor_points,
        num_heads=8,
        dropout=0.0,
        **kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_dim (int): Inner dimension of the attention block.
            num_anchor_points (int): Number of anchor points.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = inner_dim
        self.num_anchor_points = num_anchor_points
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_coords"
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # For each source:
        # - Read the length of that source's sequence;
        # - Gather the anchor tokens from the values and coordinates (and
        #   attention masks if provided);
        # - Save the indices of the anchor tokens;
        anchor_values, anchor_coords, anchor_masks = {}, {}, {}
        anchor_indices, n_anchors_list = {}, []
        for source_name, source_inputs in inputs.items():
            _, n = source_inputs["embedded_values"].shape[:2]
            # Don't take more anchor points than the number of tokens
            n_anchor_points = min(self.num_anchor_points, n)
            indices = (
                torch.linspace(0, n - 1, n_anchor_points)
                .long()
                .to(source_inputs["embedded_values"].device)
            )
            anchor_values[source_name] = source_inputs["embedded_values"][:, indices]
            anchor_coords[source_name] = source_inputs["embedded_coords"][:, indices]
            anchor_indices[source_name] = indices
            n_anchors_list.append(n_anchor_points)
        # Concatenate the anchor tokens from all sources;
        anchor_values = torch.cat([anchor_values[source_name] for source_name in inputs], dim=1)
        anchor_coords = torch.cat([anchor_coords[source_name] for source_name in inputs], dim=1)
        # Compute the attention across the anchor tokens;
        anchor_values = self.attention(anchor_values, anchor_coords)
        # Split back the sequence of anchor tokens to the sources;
        anchor_values = torch.split(anchor_values, n_anchors_list, dim=1)
        # Split the updated anchor tokens back to the sources and sum them to the original tokens;
        outputs = {}
        for i, (source_name, source_inputs) in enumerate(inputs.items()):
            indices = anchor_indices[source_name]
            bs, _, values_dim = source_inputs["embedded_values"].shape
            outputs[source_name] = torch.scatter(
                source_inputs["embedded_values"],
                1,
                indices.view(1, -1, 1).expand(bs, -1, values_dim),
                anchor_values[i] + source_inputs["embedded_values"][:, indices],
            )

        return outputs


class MultisourcesAnchoredTemporalAttention(nn.Module):
    """Computes attention across the sources using an anchor points system.
    K evenly spaced anchor tokens are gathered from each source. The ith anchor token
    of every source is concatenated into a single sequence, forming K sequences. The
    attention is computed separately for each of these sequences, and the updated tokens
    are then summed back to the original tokens of each source.
    """

    def __init__(
        self, values_dim, coords_dim, inner_dim, num_anchor_points, num_heads=8, dropout=0.0
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_dim (int): Inner dimension of the attention block.
            num_anchor_points (int): Number of anchor points.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = inner_dim
        self.num_anchor_points = num_anchor_points
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_coords"
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # For each source:
        # - Read the length of that source's sequence;
        # - Gather the anchor tokens from the values and coordinates (and
        #   attention masks if provided);
        # - Save the indices of the anchor tokens;
        anchor_values, anchor_coords, anchor_masks = {}, {}, {}
        anchor_indices = {}
        for source_name, source_inputs in inputs.items():
            _, n = source_inputs["embedded_values"].shape[:2]
            indices = (
                torch.linspace(0, n - 1, self.num_anchor_points)
                .long()
                .to(source_inputs["embedded_values"].device)
            )
            anchor_values[source_name] = source_inputs["embedded_values"][:, indices]
            anchor_coords[source_name] = source_inputs["embedded_coords"][:, indices]
            if attention_mask is not None:
                anchor_masks[source_name] = attention_mask[source_name][:, indices]
            anchor_indices[source_name] = indices
        # Stack the anchor tokens from all sources; then split them into K sequences
        # of anchor tokens.
        anchor_values = torch.stack(
            [anchor_values[source_name] for source_name in inputs], dim=1
        )  # (b, S, K, d)
        anchor_values = rearrange(anchor_values, "b S K d -> (b K) S d")
        anchor_coords = torch.stack([anchor_coords[source_name] for source_name in inputs], dim=1)
        anchor_coords = rearrange(anchor_coords, "b S K d -> (b K) S d")
        if attention_mask is not None:
            anchor_masks = torch.stack(
                [anchor_masks[source_name] for source_name in inputs], dim=1
            )
            anchor_masks = rearrange(anchor_masks, "b S K -> (b K) S")
        else:
            anchor_masks = None
        # Compute the attention across the anchor tokens;
        out = self.attention(anchor_values, anchor_coords, anchor_masks)
        # Split back the sequences of anchor tokens to the sources;
        out = rearrange(out, "(b K) S d -> b K S d", K=self.num_anchor_points)
        out = torch.split(out, 1, dim=2)  # Tuple of S tensors of shape (b, K, 1, d)
        # Sum the output to the original tokens of each source;
        outputs = {}
        for i, (source_name, source_inputs) in enumerate(inputs.items()):
            indices = anchor_indices[source_name]
            outputs[source_name] = source_inputs["embedded_values"].clone()
            bs, _, values_dim = outputs[source_name].shape
            outputs[source_name].scatter_(
                1,
                indices.view(1, -1, 1).expand(bs, -1, values_dim),
                out[i].squeeze(2) + outputs[source_name][:, indices],
            )
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
        inner_dim,
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
            inner_dim (int): Inner dimension of the attention block.
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
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // num_heads
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        if block_idx is not None:  # Override the shifted argument if block_idx is specified
            self.shifted = bool(block_idx % 2)

        self.values_qkv = nn.Linear(values_dim, inner_dim * 3)
        self.coords_qkv = nn.Linear(coords_dim, inner_dim * 2)  # No values for the coordinates
        self.dropout = nn.Dropout(dropout)

        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        self.indices = torch.tensor(
            np.array([[x, y] for x in range(window_size) for y in range(window_size)])
        )
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x[source_name] contains the keys "embedded_coords" and "embedded_values",
                "tokens_shape".

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, source_inputs in x.items():
            # If the source isn't 2D, skip the spatial attention
            if len(source_inputs["tokens_shape"]) != 2:
                outputs[source_name] = source_inputs["embedded_values"]
                continue
            # Apply the linear transformations to the values and coordinates
            v = self.values_qkv(source_inputs["embedded_values"])
            c = self.coords_qkv(source_inputs["embedded_coords"])
            # Reshape to the original spatial layout of the tokens
            h, w = source_inputs["tokens_shape"]
            v = rearrange(v, "b (h w) (d k) -> b h w d k", h=h, w=w, k=3, d=self.values_dim)
            c = rearrange(c, "b (h w) (d k) -> b h w d k", h=h, w=w, k=2, d=self.coords_dim)
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
            # Reshape back to the original sequence layout
            y = rearrange(y, "b h w d -> b (h w) d")
            outputs[source_name] = self.output_proj(y)  # Back to values_dim
        return outputs


class SeparateValuesCoordinatesAttention(nn.Module):
    """Attention block that computes the attention over each source independently."""

    def __init__(self, values_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0, **kwargs):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_dim (int): Inner dimension of the attention block.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.attention = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_coords" and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, source_inputs in inputs.items():
            mask = attention_mask[source_name] if attention_mask is not None else None
            outputs[source_name] = self.attention(
                source_inputs["embedded_values"], source_inputs["embedded_coords"], mask
            )
        return outputs
