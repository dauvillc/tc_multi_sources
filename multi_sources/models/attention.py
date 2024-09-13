"""Defines attention mechanisms for the multisource masked autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
                or None. Keys for which the mask is True will not be attended to.
        Returns:
            Tensor of shape (batch_size, num_queries, num_keys)
        """
        # Clamp the keys and queries to avoid numerical instability in the case where
        # they are perfectly correlated.
        keys, queries = keys.clamp(-5, 5), queries.clamp(-5, 5)
        dots = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Optional relative positional encodings
        if self.relative_pos:
            pos_key, pos_query = pos_key.clamp(-5, 5), pos_query.clamp(-5, 5)
            rel_pos_dots = torch.matmul(pos_query, pos_key.transpose(-2, -1)) * self.rel_pos_scale
            dots = dots + rel_pos_dots
        # Mask the columns of the attention map that correspond to the masked tokens.
        if mask is not None:
            # Expand the mask in the middle to reach the same number of dimensions as dots.
            if dots.dim() == 3:
                mask = mask.unsqueeze(1)
            elif dots.dim() == 4:
                mask = mask.unsqueeze(1).unsqueeze(1)
            dots = dots.masked_fill(mask, float("-inf"))

        return F.softmax(dots, dim=-1)


class SelfAttention(nn.Module):
    """A self-attention block that can use values that are different from the keys/queries."""

    def __init__(self, key_dim, value_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"Inner dimension {inner_dim} must be divisible by the number of heads {num_heads}"
            )
        self.qk_norm = nn.LayerNorm(key_dim)
        self.norm_value = nn.LayerNorm(value_dim)

        self.to_qk = nn.Linear(key_dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.num_heads = num_heads

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head)

        self.output_proj = nn.Sequential(nn.Linear(inner_dim, value_dim), nn.Dropout(dropout))

    def forward(self, keys, values, mask=None):
        keys = self.qk_norm(keys)
        values = self.norm_value(values)

        q, k = self.to_qk(keys).chunk(2, dim=-1)
        q, k = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (q, k))

        v = self.to_v(values)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = self.attention_map(k, q, mask=mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.output_proj(out)


class ValuesCoordinatesAttention(nn.Module):
    """Self-attention block that uses the values values as well as the
    coordinates to compute the attention weights,
    as Softmax(QpKp^T + QcKc^T)."""

    def __init__(self, values_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert inner_dim % num_heads == 0
        self.num_heads = num_heads

        self.values_to_qkv = nn.Linear(values_dim, inner_dim * 3, bias=False)
        self.coords_to_qk = nn.Linear(coords_dim, inner_dim * 2, bias=False)

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "dt", "embedded_dt", "embedded_coords",
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Concatenate the sequences from all sources.
        values = torch.cat([data["embedded_values"] for data in inputs.values()], dim=1)
        coords = torch.cat([data["embedded_coords"] for data in inputs.values()], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([mask for mask in attention_mask.values()], dim=1)
        # Project the values and coordinates to the query, key and value spaces.
        qkv_values = self.values_to_qkv(values).chunk(3, dim=-1)
        qk_coords = self.coords_to_qk(coords).chunk(2, dim=-1)
        qp, kp, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qkv_values
        )
        qc, kc = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qk_coords)

        # Compute the attention map using two sets of keys and queries, one from the values
        # and one from the coordinates.
        attn = self.attention_map(kp, qp, kc, qc, mask=attention_mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.output_proj(out)
        # Split the updated values sequence back into sequences for each source.
        source_seqs = torch.split(out, [data["embedded_values"].shape[1] for data in inputs.values()], dim=1)
        return {source_name: seq for source_name, seq in zip(inputs.keys(), source_seqs)}


class TemporalWindowedAttention(nn.Module):
    """Attention block that uses windowed attention over the temporal dimension,
    as well as coordinates embeddings as relative positional encodings."""

    def __init__(self, values_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0, window_size=4):
        super().__init__()
        assert inner_dim % num_heads == 0
        assert window_size % 2 == 0
        self.num_heads = num_heads
        self.window_size = window_size

        self.values_to_qkv = nn.Linear(values_dim, inner_dim * 3, bias=False)
        self.coords_to_qk = nn.Linear(coords_dim, inner_dim * 2, bias=False)

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

    def forward(self, values, coords, mask=None):
        """
        Args:
            values (list of torch.Tensor): Embedded sequence of tokens from the values,
                for each source. Each tensor should have shape (bs, n_tokens, values_dim).
                The sources are assumed to be sorted in increasing order of time.
            coords (list of torch.Tensor): Embedded sequence of tokens from the coordinates,
                for each source.
                Each tensor should have shape (bs, n_tokens, coords_dim).
            mask (list of torch.Tensor): Mask for the attention map, for each source.
                Each tensor should have shape (bs, n_tokens).
        Returns:
            list of torch.Tensor: The updated values.
        """
        # TODO
        pass
