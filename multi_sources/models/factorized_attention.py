"""Implements the FactorizedMultisourcesAttention class."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from multi_sources.models.attention import ValuesMetadataAttentionInternal
from multi_sources.models.window_utils import source_to_2D_windows, windows_2D_to_source


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


class SeparateWindowedValuesMetadataAttention(nn.Module):
    """Attention block that computes the attention over each source independently,
    using a spatial window over the tokens as in the Swin Transformer."""

    def __init__(
        self,
        values_dim,
        metadata_dim,
        inner_dim,
        num_heads=8,
        dropout=0.0,
        window_size=8,
        use_shifted_windows=False,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            metadata_dim (int): Embedding dimension of the metadata.
            inner_dim (int): Inner dimension of the attention block.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            window_size (int): Number of tokens included in each window.
            use_shifted_windows (bool): Whether to shift the windows by half the window size.
        """
        super().__init__()
        self.window_size = window_size
        self.use_shifted_windows = use_shifted_windows

        self.attention = ValuesMetadataAttentionInternal(
            values_dim, metadata_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_metadata" and "embedded_values",
                "tokens_shape".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        shift = self.window_size // 2 if self.use_shifted_windows else 0
        # For each source, partition the values, metadata and attention mask into windows.
        values, metadata, masks = [], [], []
        for source_name, data in inputs.items():
            source_values = data["embedded_values"]
            source_metadata = data["embedded_metadata"]
            source_mask = attention_mask[source_name] if attention_mask is not None else None
            tokens_shape = data["tokens_shape"]
            # Partitions the values, metadata and the mask, and also sets mask = True
            # where padding is added for the partitioning.
            (source_values, source_metadata), source_mask = source_to_2D_windows(
                [source_values, source_metadata],
                tokens_shape,
                self.window_size,
                shift,
                source_mask,
            )
            values.append(source_values)
            metadata.append(source_metadata)
            masks.append(source_mask)
        # Save the size of each source along the batch dimension.
        source_sizes = [v.shape[0] for v in values]
        # Concatenate the windows from all sources.
        values = torch.cat(values, dim=0)
        metadata = torch.cat(metadata, dim=0)
        masks = torch.cat(masks, dim=0)
        # Feed the windows to the attention block.
        att_out = self.attention(values, metadata, masks)
        # Split the attention output back to the sources.
        att_out = torch.split(att_out, source_sizes, dim=0)
        # Reconstruct the source from the windows.
        out = {}
        for source_name, source_out in zip(inputs.keys(), att_out):
            # Reconstruct the source from the windows.
            tokens_shape = inputs[source_name]["tokens_shape"]
            out[source_name] = windows_2D_to_source(
                [source_out], tokens_shape, self.window_size, shift
            )[0]  # Takes in and returns a list of tensors
        return out
