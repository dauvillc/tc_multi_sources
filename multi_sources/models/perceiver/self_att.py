import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from multi_sources.models.perceiver.small_layers import RMSNorm


class ValuesCoordinatesSelfAttention(nn.Module):
    """Attention block that computes the attention over the values and coordinates.
    The layer performs attention in the following manner:
    - The values and coords are projected to two sets of queries, keys and values
        (qv, kv, vv) and (qc, kc, vc). /!\ there's "values" in the sense of the values sequence,
        and values in the sense of the attention mechanism.
    - A common attention map is computed using the queries and keys from the values
        and coordinates using Softmax(QvKv^T + QcKc^T).
    - Two outputs are computed: one using vv and one using vc, using the computed attention maps.
    - The outputs are projected back to the values and coordinates spaces.
    """

    def __init__(
        self, values_dim, coords_dim, inner_ratio, num_heads, dropout=0.0, **unused_kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_ratio (float): Ratio of the inner dimension to the values dimension.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = inner_ratio * values_dim
        self.head_dim = self.inner_dim // num_heads
        self.num_heads = num_heads

        self.v_norm = nn.LayerNorm(values_dim)
        self.c_norm = nn.LayerNorm(coords_dim)

        self.values_qk = nn.Sequential(
            nn.Linear(values_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.coords_qk = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.values_v = nn.Linear(values_dim, self.inner_dim)
        self.coords_v = nn.Linear(coords_dim, self.inner_dim)

        self.dropout = nn.Dropout(dropout)

        self.output_proj_v = nn.Sequential(
            nn.Linear(self.inner_dim, values_dim), nn.Dropout(dropout)
        )
        self.output_proj_c = nn.Sequential(
            nn.Linear(self.inner_dim, coords_dim), nn.Dropout(dropout)
        )

    def forward(self, V, C):
        """
        Args:
            V (tensor): Values tensor of shape (batch_size, num_values, values_dim).
            C (tensor): Coordinates tensor of shape (batch_size, num_values, coords_dim).

        Returns:
            tensor: Updated values tensor of shape (batch_size, num_values, values_dim).
            tensor: Updated coordinates tensor of shape (batch_size, num_values, coords_dim).
        """
        # Normalize the values and coordinates.
        V = self.v_norm(V)
        C = self.c_norm(C)

        # Project the values and coordinates to the query, key and value spaces.
        qv, kv = self.values_qk(V).chunk(2, dim=-1)
        qc, kc = self.coords_qk(C).chunk(2, dim=-1)
        vv = self.values_v(V)
        vc = self.coords_v(C)

        # Split the queries, keys and values in the heads dimension.
        qv, kv, vv = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (qv, kv, vv)
        )
        qc, kc, vc = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (qc, kc, vc)
        )

        # Compute the common attention map.
        attn_map = qc @ kc.transpose(-2, -1) + qv @ kv.transpose(-2, -1)
        attn_map /= np.sqrt(self.head_dim)
        attn_map = F.softmax(attn_map, dim=-1)
        attn_map = self.dropout(attn_map)

        # Compute the output values and coords.
        out_v = attn_map @ vv
        out_v = rearrange(out_v, "b h n d -> b n (h d)")
        out_v = self.output_proj_v(out_v)
        out_c = attn_map @ vc
        out_c = rearrange(out_c, "b h n d -> b n (h d)")
        out_c = self.output_proj_c(out_c)

        return out_v, out_c
