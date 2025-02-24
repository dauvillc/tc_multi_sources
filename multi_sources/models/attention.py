"""Defines attention mechanisms for the multisource masked autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from multi_sources.models.small_layers import RMSNorm

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
                or None. Keys for which the mask is False will not be attended to.
        Returns:
            Tensor of shape (batch_size, num_queries, num_keys)
        """
        dots = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Optional relative positional encodings
        if self.relative_pos:
            rel_pos_dots = torch.matmul(pos_query, pos_key.transpose(-2, -1)) * self.rel_pos_scale
            dots = dots + rel_pos_dots
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
        super().__init__()
        assert inner_dim % num_heads == 0
        self.num_heads = num_heads

        self.values_to_qkv = nn.Sequential(
            nn.Linear(values_dim, inner_dim * 3, bias=False),
            RMSNorm(inner_dim * 3),
        )
        self.coords_to_qk = nn.Sequential(
            nn.Linear(coords_dim, inner_dim * 2, bias=False),
            RMSNorm(inner_dim * 2),
        )

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

    def forward(self, values, coords, attention_mask=None):
        """
        Args:
            values (tensor): Tensor of shape (batch_size, seq_len, values_dim).
            coords (tensor): Tensor of shape (batch_size, seq_len, coords_dim).
            attention_mask (tensor): Tensor of shape (batch_size, seq_len), or None.
        Returns:
            Tensor of shape (batch_size, seq_len, values_dim), the updated values.
        """
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
        return out
