import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from multi_sources.models.motif_anchored.small_layers import RMSNorm


class AttentionMap(nn.Module):
    """Computes the attention map from the embedded keys and queries.
    Using this intermediary Module allows to retrieve the attention maps
    with register_forward_hook."""

    def __init__(self, dim_head, relative_pos=False, rel_pos_dim_head=None):
        super().__init__()
        self.scale = dim_head**-0.5

        self.relative_pos = relative_pos
        if self.relative_pos:
            self.rel_pos_scale = rel_pos_dim_head**-0.5
            self.rel_pos_weight = nn.Parameter(torch.randn(1))

    def forward(self, keys, queries, pos_key=None, pos_query=None, mask=None):
        """
        Args:
            keys: Tensor of shape (batch_size, num_keys, embed_dim), or
                (batch_size, heads, num_keys, dim_head). The same is true for
                all other arguments.
            queries: Tensor of shape (batch_size, num_queries, embed_dim)
            pos_key: Tensor of shape (batch_size, num_keys, rel_pos_dim)
            pos_query: Tensor of shape (batch_size, num_queries, rel_pos_dim)
            mask: Tensor of shape (batch_size, num_keys) or (batch_size, heads, num_keys),
                or None. Keys for which the mask is False will not be attended to.
            coords_weight: float, optional, default=1.0
                Weight to apply to the coordinates in the attention map.
        Returns:
            Tensor of shape (batch_size, num_queries, num_keys)
        """
        dots = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Optional relative positional encodings
        if self.relative_pos:
            rel_pos_dots = torch.matmul(pos_query, pos_key.transpose(-2, -1)) * self.rel_pos_scale
            dots = dots + self.rel_pos_weight * rel_pos_dots
        # Mask the columns of the attention map that correspond to the masked tokens.
        if mask is not None:
            # Expand the mask in the middle to reach the same number of dimensions as dots.
            if dots.dim() == 3:
                mask = mask.unsqueeze(1)
            elif dots.dim() == 4:
                mask = mask.unsqueeze(1).unsqueeze(1)
            dots = dots.masked_fill(~mask, float("-inf"))

        return F.softmax(dots, dim=-1)


class ValuesCoordinatesAttentionInternal(nn.Module):
    """Self-attention block that uses both the values and coordinates
    to compute the attention weights, as Softmax(QpKp^T + QcKc^T)."""

    def __init__(self, values_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0):
        """Args:
        values_dim (int): Dimension of the values.
        coords_dim (int): Dimension of the coordinates.
        inner_dim (int): Dimension of the inner representations.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate to apply to the attention map.
        """
        super().__init__()
        self.num_heads = num_heads

        self.values_to_qkv = nn.Linear(values_dim, inner_dim * 3, bias=False)
        # RMSNorm on the queries and keys to stabilize the training (cf Stable Diff. 3)
        self.values_qnorm = RMSNorm(inner_dim)
        self.values_knorm = RMSNorm(inner_dim)

        self.coords_to_qk = nn.Sequential(
            nn.Linear(coords_dim, inner_dim * 2, bias=False), RMSNorm(inner_dim * 2)
        )

        dim_head = inner_dim // num_heads
        self.coords_weight = nn.Parameter(torch.tensor(1.0))
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

        # Dropout on the attention map
        self.att_dropout = nn.Dropout(dropout)

    def forward(self, values, coords, coords_weight=1.0, attention_mask=None):
        """
        Args:
            values (tensor): Tensor of shape (batch_size, seq_len, values_dim).
            coords (tensor): Tensor of shape (batch_size, seq_len, coords_dim).
            coords_weight (float): Weight to apply to the coordinates in the attention map.
            attention_mask (tensor): Tensor of shape (batch_size, seq_len), or None.
        Returns:
            Tensor of shape (batch_size, seq_len, values_dim), the updated values.
        """
        # Project the values and coordinates to the query, key and value spaces.
        qv, kv, vv = self.values_to_qkv(values).chunk(3, dim=-1)
        qv, kv = self.values_qnorm(qv), self.values_knorm(kv)  # RMSNorm
        qc, kc = self.coords_to_qk(coords).chunk(2, dim=-1)

        # Reshape to split into multiple heads
        qv, kv, vv = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (qv, kv, vv)
        )
        qc, kc = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (qc, kc))

        # Compute the attention map using two sets of keys and queries, one from the values
        # and one from the coordinates.
        attn = self.attention_map(kv, qv, kc, qc, mask=attention_mask, coords_weight=coords_weight)
        # Apply dropout to the attention map
        attn = self.att_dropout(attn)
        out = torch.matmul(attn, vv)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.output_proj(out)
        return out


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
        shifted=False,
        num_heads=8,
        dropout=0.0,
        attention_coords_weight=1.0,
        **kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_ratio (float): Ratio of the inner dimension to the values dimension.
            anchor_points_spacing (int): Spacing between the anchor points. For example,
                3 means that every third token along each axis is selected as an anchor point.
            shifted (bool): Whether to shift the anchor points by half the spacing.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            attention_coords_weight (float): Weight to apply to the coordinates
                in the attention map.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = inner_ratio * values_dim
        self.anchor_points_spacing = anchor_points_spacing
        self.shifted = shifted
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_coords_weight = attention_coords_weight

        self.attention = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, self.inner_dim, num_heads, dropout
        )

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary of inputs, such that
                inputs[(source_name, index)] contains the keys "embedded_coords", "embedded_values".
                where (source_name, index) is a tuple with the source name and the observation index
                (0 = most recent).
                The values are expected of shape (B, ..., Dv) and the coordinates of shape
                (B, ..., Dc), where ... is the spatial dimensions of the embedded source,
                e.g. (h, w) for 2D sources.

        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the predicted values of the tokens.
        """
        # For each source:
        # - Read the length of that source's sequence
        # - Gather the anchor tokens from the values and coordinates
        # - Save the indices of the anchor tokens
        anchor_values, anchor_coords = {}, {}
        anchor_indices, n_anchors_list = {}, []
        for source_index_pair, source_inputs in inputs.items():
            V, C = source_inputs["embedded_values"], source_inputs["embedded_coords"]
            spatial_dims = V.shape[1:-1]

            # For 0D sources, there's a single token, which will be the anchor point.
            if len(spatial_dims) == 0:
                anchor_values[source_index_pair] = V.unsqueeze(1)  # "Sequence" of 1 token
                anchor_coords[source_index_pair] = C.unsqueeze(1)
                n_anchor_points = 1

            # For 2D sources, we want to select the anchor points in a grid pattern.
            elif len(spatial_dims) == 2:
                h, w = spatial_dims
                anchor_cols = torch.arange(0, w, self.anchor_points_spacing)
                anchor_rows = torch.arange(0, h, self.anchor_points_spacing)
                if not self.shifted:
                    # By default, the anchor points are centered on the grid.
                    anchor_cols += ((w - 1) % self.anchor_points_spacing) // 2
                    anchor_rows += ((h - 1) % self.anchor_points_spacing) // 2

                anchor_v_s = V[:, anchor_rows[:, None], anchor_cols]
                anchor_v_s = rearrange(anchor_v_s, "b h w d -> b (h w) d")
                anchor_c_s = C[:, anchor_rows[:, None], anchor_cols]
                anchor_c_s = rearrange(anchor_c_s, "b h w d -> b (h w) d")
                anchor_values[source_index_pair] = anchor_v_s
                anchor_coords[source_index_pair] = anchor_c_s

                # Save the indices of the rows and columns for later
                anchor_indices[source_index_pair] = (anchor_rows, anchor_cols)
                n_anchor_points = len(anchor_rows) * len(anchor_cols)

            # Save the number of anchor points for later
            n_anchors_list.append(n_anchor_points)

        # Concatenate the anchor tokens from all sources
        anchor_values = torch.cat([anchor_values[src] for src in inputs], dim=1)
        anchor_coords = torch.cat([anchor_coords[src] for src in inputs], dim=1)
        # Compute the attention across the anchor tokens
        anchor_values = self.attention(
            anchor_values, anchor_coords, coords_weight=self.attention_coords_weight
        )
        # Split back the sequence of anchor tokens to the sources
        anchor_values = torch.split(anchor_values, n_anchors_list, dim=1)

        # Split the updated anchor tokens back to the sources and sum them to the original tokens
        outputs = {}
        for i, (source_index_pair, source_inputs) in enumerate(inputs.items()):
            V = source_inputs["embedded_values"].clone()
            spatial_dims = V.shape[1:-1]

            if len(spatial_dims) == 0:
                # For 0D sources, just sum the updated anchor token to the original token
                V += anchor_values[i].squeeze(1)  # (B, 1, Dv) to (B, Dv)

            elif len(spatial_dims) == 2:
                anchor_rows, anchor_cols = anchor_indices[source_index_pair]
                anchor_v_s = anchor_values[i]
                anchor_v_s = rearrange(anchor_v_s, "b (h w) d -> b h w d", h=len(anchor_rows))
                V[:, anchor_rows[:, None], anchor_cols] += anchor_v_s

            outputs[source_index_pair] = V
        return outputs
