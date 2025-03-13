import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from multi_sources.models.perceiver.small_layers import RMSNorm


class MultisourcePerceiverEncoder(nn.Module):
    """Implements the encoding part of the Perceiver model (Jaegle et al., 2022) for
    multiple sources. Each source s includes two sequences: a values sequence Vs and a
    coordinates sequence Cs.
    When instantiated, the encoder creates two learned latent arrays Lv and Lc of shapes
    (N, Dv) and (N, Dc), for the values and coordinates, respectively.
    The encoder performs the following steps during the forward pass:
    - Project Lv and c to queries q_v and q_c.
    - Concatenante the values and coords sequences of all sources to create a single
        values sequence V and a single coordinates sequence C.
    - Project V to keys and values k_v and v_v.
    - Project C to keys and values k_c and v_c.
    - Compute a single attention map A = Softmax(q_v k_v^T + q_c k_c^T).
    - Update Lv and Lc via Lv = A @ v_v and Lc = A @ v_c.
    - Project the updated latents to the values and coordinates spaces.
    - Output Lv and Lc.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        latent_size,
        inner_ratio,
        num_heads,
        dropout=0.0,
        **unused_kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            latent_size (int): Index size of the latent arrays.
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

        self.latent_values = nn.Parameter(torch.randn(1, latent_size, values_dim))
        self.latent_coords = nn.Parameter(torch.randn(1, latent_size, coords_dim))

        self.values_q = nn.Sequential(
            nn.Linear(values_dim, self.inner_dim), RMSNorm(self.inner_dim)
        )
        self.coords_q = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim), RMSNorm(self.inner_dim)
        )
        self.values_kv = nn.Sequential(
            nn.Linear(values_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.coords_kv = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.output_proj_v = nn.Sequential(
            nn.Linear(self.inner_dim, values_dim), nn.Dropout(dropout)
        )
        self.output_proj_c = nn.Sequential(
            nn.Linear(self.inner_dim, coords_dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (dict of str to tensor): Dictionary of inputs for each source, such
                that x[src] contains at least the entries "embedded_values" and
                "embedded_coords".

        Returns:
            Lv (tensor): Updated latent array for the values.
            Lc (tensor): Updated latent array for the coordinates.
        """
        # Project the latents to queries.
        q_v = self.values_q(self.latent_values)
        q_c = self.coords_q(self.latent_coords)

        # Concatenate the values and coords sequences of all sources.
        V = torch.cat([x[src]["embedded_values"] for src in x], dim=1)
        C = torch.cat([x[src]["embedded_coords"] for src in x], dim=1)

        # Project the values and coordinates to keys and values.
        kv, vv = self.values_kv(V).chunk(2, dim=-1)
        kc, vc = self.coords_kv(C).chunk(2, dim=-1)

        # Reshape to use parallel heads.
        qv = rearrange(q_v, "b n (h d) -> b h n d", h=self.num_heads)
        qc = rearrange(q_c, "b n (h d) -> b h n d", h=self.num_heads)
        kv = rearrange(kv, "b n (h d) -> b h n d", h=self.num_heads)
        kc = rearrange(kc, "b n (h d) -> b h n d", h=self.num_heads)
        vv = rearrange(vv, "b n (h d) -> b h n d", h=self.num_heads)
        vc = rearrange(vc, "b n (h d) -> b h n d", h=self.num_heads)

        # Compute the common attention map.
        attn_map = qc @ kc.transpose(-2, -1) + qv @ kv.transpose(-2, -1)
        attn_map /= np.sqrt(self.head_dim)
        attn_map = F.softmax(attn_map, dim=-1)
        attn_map = self.dropout(attn_map)

        # Update the latent arrays.
        Lv = attn_map @ vv
        Lc = attn_map @ vc

        # Project back to the original dimensions.
        Lv = rearrange(Lv, "b h n d -> b n (h d)")
        Lv = self.output_proj_v(Lv)
        Lc = rearrange(Lc, "b h n d -> b n (h d)")
        Lc = self.output_proj_c(Lc)

        return Lv, Lc


class MultisourcePerceiverDecoder(nn.Module):
    """Implements the decoding part of the Perceiver model (Jaegle et al., 2022) for
    multiple sources. Each source s includes a sequence of coordinates Cs.
    The decoder receives the latent arrays Lv and Lc, and the coordinates
    sequences of all sources. It performs the following steps during the forward pass:
    - Project Lv to values v_v.
    - Project Lc to keys k_c.
    - For each source s:
        - Project Cs to queries q_c.
        - Compute the attention map A = Softmax(q_c k_c^T).
        - Compute the output values Vs = A @ v_v.
        - Project Vs to the original values space.
    - Output the values sequences Vs.
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

        self.coords_k = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim), RMSNorm(self.inner_dim)
        )
        self.coords_q = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim), RMSNorm(self.inner_dim)
        )
        self.values_v = nn.Sequential(
            nn.Linear(values_dim, self.inner_dim), RMSNorm(self.inner_dim)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.inner_dim, values_dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Lv, Lc):
        """
        Args:
            x (dict of str to tensor): Dictionary of inputs for each source, such
                that x[src] contains at least the entry "embedded_coords".
            Lv (tensor): Latent array for the values.
            Lc (tensor): Latent array for the coordinates.

        Returns:
            dict of str to tensor: Dictionary {src: Vs} of output values sequences.
        """
        # Project the latents to values and keys.
        v_v = self.values_v(Lv)
        k_c = self.coords_k(Lc)
        k_c = rearrange(k_c, "b n (h d) -> b h n d", h=self.num_heads)
        v_v = rearrange(v_v, "b n (h d) -> b h n d", h=self.num_heads)

        outputs = {}
        for src in x:
            C = x[src]["embedded_coords"]
            q_c = self.coords_q(C)
            
            # Reshape to use parallel heads.
            q_c = rearrange(q_c, "b n (h d) -> b h n d", h=self.num_heads)

            # Compute the attention map.
            attn_map = q_c @ k_c.transpose(-2, -1)
            attn_map /= np.sqrt(self.head_dim)
            attn_map = F.softmax(attn_map, dim=-1)
            attn_map = self.dropout(attn_map)

            # Compute the output values.
            Vs = attn_map @ v_v
            Vs = rearrange(Vs, "b h n d -> b n (h d)")
            Vs = self.output_proj(Vs)
            outputs[src] = Vs

        return outputs
