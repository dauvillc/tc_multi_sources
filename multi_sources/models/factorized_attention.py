"""Implements the FactorizedMultisourcesAttention class."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from multi_sources.models.attention import ValuesMetadataAttentionInternal


class MultisourcesAnchoredCrossAttention(nn.Module):
    """Computes attention across the sources using an anchor points system.
    K evenly spaced anchor tokens are gathered from each source. Those tokens are then
    concatenated into a single sequence, on which the attention is computed. The updated
    tokens are then summed back to the original tokens of each source.
    """

    def __init__(
        self, values_dim, metadata_dim, inner_dim, num_anchor_points, num_heads=8, dropout=0.0
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            metadata_dim (int): Embedding dimension of the metadata.
            inner_dim (int): Inner dimension of the attention block.
            num_anchor_points (int): Number of anchor points.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.metadata_dim = metadata_dim
        self.inner_dim = inner_dim
        self.num_anchor_points = num_anchor_points
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = ValuesMetadataAttentionInternal(
            values_dim, metadata_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_metadata"
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # For each source:
        # - Read the length of that source's sequence;
        # - Gather the anchor tokens from the values and metadata (and
        #   attention masks if provided);
        # - Save the indices of the anchor tokens;
        anchor_values, anchor_metadata, anchor_masks = {}, {}, {}
        anchor_indices = {}
        for source_name, source_inputs in inputs.items():
            _, n = source_inputs["embedded_values"].shape[:2]
            indices = (
                torch.linspace(0, n - 1, self.num_anchor_points)
                .long()
                .to(source_inputs["embedded_values"].device)
            )
            anchor_values[source_name] = source_inputs["embedded_values"][:, indices]
            anchor_metadata[source_name] = source_inputs["embedded_metadata"][:, indices]
            if attention_mask is not None:
                anchor_masks[source_name] = attention_mask[source_name][:, indices]
            anchor_indices[source_name] = indices
        # Concatenate the anchor tokens from all sources;
        anchor_values = torch.cat([anchor_values[source_name] for source_name in inputs], dim=1)
        anchor_metadata = torch.cat(
            [anchor_metadata[source_name] for source_name in inputs], dim=1
        )
        if attention_mask is not None:
            anchor_masks = torch.cat([anchor_masks[source_name] for source_name in inputs], dim=1)
        else:
            anchor_masks = None
        # Compute the attention across the anchor tokens;
        anchor_values = self.attention(anchor_values, anchor_metadata, anchor_masks)
        # Split back the sequence of anchor tokens to the sources;
        anchor_values = torch.split(anchor_values, self.num_anchor_points, dim=1)
        # Split the updated anchor tokens back to the sources and sum them to the original tokens;
        outputs = {}
        for i, (source_name, source_inputs) in enumerate(inputs.items()):
            indices = anchor_indices[source_name]
            outputs[source_name] = source_inputs["embedded_values"].clone()
            bs, _, values_dim = outputs[source_name].shape
            outputs[source_name].scatter_(
                1,
                indices.view(1, -1, 1).expand(bs, -1, values_dim),
                anchor_values[i] + outputs[source_name][:, indices],
            )

        return outputs


class FactorizedMultisourcesAttention(nn.Module):
    """Computes attention in two steps:
    - First across the sources, using an anchor points system.
    - Then within each source independently.
    The attention is computed using both the values and the metadata embeddings.
    """

    def __init__(
        self,
        values_dim,
        metadata_dim,
        inner_dim,
        num_anchor_points,
        num_heads=8,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            metadata_dim (int): Embedding dimension of the metadata.
            inner_dim (int): Inner dimension of the attention block.
            num_anchor_points (int): Number of anchor points.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.metadata_dim = metadata_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        if inner_dim % num_heads != 0:
            raise ValueError("Inner dimension must be divisible by the number of heads.")
        self.dim_head = inner_dim // num_heads
        self.dropout = dropout

        self.cross_attention = MultisourcesAnchoredCrossAttention(
            values_dim, metadata_dim, inner_dim, num_anchor_points, num_heads, dropout
        )
        self.self_attention = ValuesMetadataAttentionInternal(
            values_dim, metadata_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_metadata"
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Compute the attention across the sources;
        updated_values = self.cross_attention(inputs, attention_mask)
        # Compute the self-attention over each source;
        outputs = {}
        for source_name, source_inputs in inputs.items():
            values = updated_values[source_name]
            metadata = source_inputs["embedded_metadata"]
            attn_mask = attention_mask[source_name] if attention_mask is not None else None
            outputs[source_name] = self.self_attention(values, metadata, attn_mask)
        return outputs
