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
        self.scale = dim_head ** -0.5

        self.relative_pos = relative_pos
        if self.relative_pos:
            self.rel_pos_scale = rel_pos_dim_head ** -0.5

    def forward(self, keys, queries, pos_key=None, pos_query=None):
        # Clamp the keys and queries to avoid numerical instability in the case where
        # they are perfectly correlated.
        keys, queries = keys.clamp(-5, 5), queries.clamp(-5, 5)
        dots = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Optional relative positional encodings
        if self.relative_pos:
            pos_key, pos_query = pos_key.clamp(-5, 5), pos_query.clamp(-5, 5)
            rel_pos_dots = torch.matmul(pos_query, pos_key.transpose(-2, -1)) * self.rel_pos_scale
            dots = dots + rel_pos_dots
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

    def forward(self, keys, values):
        keys = self.qk_norm(keys)
        values = self.norm_value(values)

        q, k = self.to_qk(keys).chunk(2, dim=-1)
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


class PixelsCoordinatesAttention(nn.Module):
    """Self-attention block that uses the pixel values as well as the
    coordinates to compute the attention weights,
    as Softmax(QpKp^T + QcKc^T)."""

    def __init__(self, pixel_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert inner_dim % num_heads == 0
        self.num_heads = num_heads

        self.pixels_to_qkv = nn.Linear(pixel_dim, inner_dim * 3, bias=False)
        self.coords_to_qk = nn.Linear(coords_dim, inner_dim * 2, bias=False)

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, pixel_dim), nn.Dropout(dropout))

    def forward(self, pixels, coords):
        # Project the pixel values and coordinates to the query, key and value spaces.
        # The values come from the pixels.
        qkv_pixels = self.pixels_to_qkv(pixels).chunk(3, dim=-1)
        qk_coords = self.coords_to_qk(coords).chunk(2, dim=-1)
        qp, kp, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qkv_pixels
        )
        qc, kc = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qk_coords)

        # Compute the attention map using two sets of keys and queries, one from the pixels
        # and one from the coordinates.
        attn = self.attention_map(kp, qp, kc, qc)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.output_proj(out)
