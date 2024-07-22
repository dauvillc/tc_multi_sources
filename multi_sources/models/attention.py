"""Defines attention mechanisms for the multisource masked autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionMap(nn.Module):
    """Computes the attention map from the embedded keys and queries.
    Using this intermediary Module allows to retrieve the attention maps
    with register_forward_hook."""

    def __init__(self, key_dim, query_dim):
        super().__init__()
        self.scale = key_dim**-0.5

    def forward(self, keys, queries):
        dots = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
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
        # To avoid the keys and queries becoming too large, we normalize them after
        # the linear transformation.
        self.key_norm = nn.LayerNorm(inner_dim)
        self.query_norm = nn.LayerNorm(inner_dim)

        self.to_qk = nn.Linear(key_dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.num_heads = num_heads

        self.attention_map = AttentionMap(inner_dim, inner_dim)

        self.output_proj = nn.Sequential(nn.Linear(inner_dim, value_dim), nn.Dropout(dropout))

    def forward(self, keys, values):
        keys = self.qk_norm(keys)
        values = self.norm_value(values)

        q, k = self.to_qk(keys).chunk(2, dim=-1)
        q, k = self.query_norm(q), self.key_norm(k)
        q, k = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (q, k))

        v = self.to_v(values)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = self.attention_map(k, q)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.output_proj(out)


class CoordinatesAttention(nn.Module):
    """Self-attention block that uses the spatio-temporal coordinates of the
    tokens to compute the attention weights."""

    def __init__(self, pixel_dim, coords_dim, **attention_kwargs):
        super().__init__()
        self.attention = SelfAttention(coords_dim, pixel_dim, **attention_kwargs)

    def forward(self, pixel_values, coords):
        return self.attention(coords, pixel_values)


class PixelsAttention(nn.Module):
    """Self-attention block that uses the pixel values of the tokens to compute
    the attention weights."""

    def __init__(self, pixel_dim, coords_dim, **attention_kwargs):
        # coords_dim is in the arguments for compatibility within the general backbone.
        super().__init__()
        self.attention = SelfAttention(pixel_dim, pixel_dim, **attention_kwargs)

    def forward(self, pixel_values, coords):
        return self.attention(pixel_values, pixel_values)
