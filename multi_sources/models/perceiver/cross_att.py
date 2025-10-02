import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from multi_sources.models.perceiver.small_layers import MLP, RMSNorm


class MultisourcePerceiverCrossAttention(nn.Module):
    """Implements the encoding part of the Perceiver model (Jaegle et al., 2022) for
    multiple sources. Each source s includes three embedded sequences:
    - a values sequence Vs
    - a coordinates sequence Cs
    - a conditioning sequence Ds
    Besides, the encoder receives three latent arrays Lv, Lc and Ld of shapes
    (N, Dv), (N, Dc) and (N, Dv), for the values, coordinates and conditioning, respectively.
    The encoder performs the following steps during the forward pass:
    - Project Lv, Lc and Ld to queries q_v, q_c and q_d.
    - Concatenante the values and coords sequences of all sources to create a single
    values sequence V, coordinates sequence C and conditioning sequence D.
    - Project V to keys and values k_v and v_v.
    - Project C to keys and values k_c and v_c.
    - Project D to keys and values k_d and v_d.
    - Compute a single attention map W = Softmax(q_v k_v^T + lambda * q_c k_c^T).
    - Update via Lv = W @ v_v , Lc = W @ v_c , Ld = W @ v_d.
    - Project Lv, Lc and Ld back to the native dimensions.
    - Output Lv, Lc and Ld.
    """

    def __init__(
        self, values_dim, coords_dim, att_inner_ratio, num_heads, dropout=0.0, **unused_kwargs
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            att_inner_ratio (float): Ratio of the inner dimension to the values dimension
                in the attention layer.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.att_inner_dim = int(att_inner_ratio * values_dim)
        self.head_dim = self.att_inner_dim // num_heads
        self.num_heads = num_heads
        self.att_lambda = nn.Parameter(torch.randn(1), requires_grad=True)

        self.values_q = nn.Sequential(
            nn.Linear(values_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.coords_q = nn.Sequential(
            nn.Linear(coords_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.values_k = nn.Sequential(
            nn.Linear(values_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.coords_k = nn.Sequential(
            nn.Linear(coords_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.values_v = nn.Linear(values_dim, self.att_inner_dim)
        self.coords_v = nn.Linear(coords_dim, self.att_inner_dim)
        self.condit_v = nn.Linear(values_dim, self.att_inner_dim)

        self.output_proj_v = nn.Sequential(
            nn.Linear(self.att_inner_dim, values_dim), nn.Dropout(dropout)
        )
        self.output_proj_c = nn.Sequential(
            nn.Linear(self.att_inner_dim, coords_dim), nn.Dropout(dropout)
        )
        self.output_proj_d = nn.Sequential(
            nn.Linear(self.att_inner_dim, values_dim), nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Lv, Lc):
        """
        Args:
            x (dict of str to tensor): Dictionary of inputs for each source, such
                that x[src] contains at least the entries "values", "coords' and
                "conditioning".
            Lv (tensor): Latent array for the values.
            Lc (tensor): Latent array for the coordinates.
            Ld (tensor): Latent array for the conditioning.

        Returns:
            Lv (tensor): Updated latent array for the values.
            Lc (tensor): Updated latent array for the coordinates.
        """
        B = next(iter(x.values()))["coords"].shape[0]  # Batch size
        Dv, Dc = self.values_dim, self.coords_dim

        # Project the latents to queries.
        qv = self.values_q(Lv)
        qc = self.coords_q(Lc)

        # Concatenate the sequences of all sources.
        V = torch.cat([x[src]["values"].view(B, -1, Dv) for src in x], dim=1)
        C = torch.cat([x[src]["coords"].view(B, -1, Dc) for src in x], dim=1)
        D = torch.cat([x[src]["conditioning"].view(B, -1, Dv) for src in x], dim=1)

        # Project the values and coordinates to keys and values.
        kv = self.values_k(V)
        kc = self.coords_k(C)
        vv = self.values_v(V)
        vc = self.coords_v(C)
        vd = self.condit_v(D)

        # Reshape to use parallel heads.
        qv = rearrange(qv, "b n (h d) -> b h n d", h=self.num_heads)
        qc = rearrange(qc, "b n (h d) -> b h n d", h=self.num_heads)
        kv = rearrange(kv, "b n (h d) -> b h n d", h=self.num_heads)
        kc = rearrange(kc, "b n (h d) -> b h n d", h=self.num_heads)
        vv = rearrange(vv, "b n (h d) -> b h n d", h=self.num_heads)
        vc = rearrange(vc, "b n (h d) -> b h n d", h=self.num_heads)
        vd = rearrange(vd, "b n (h d) -> b h n d", h=self.num_heads)

        # Compute the common attention map.
        attn_map = qc @ kc.transpose(-2, -1) + self.att_lambda * (qv @ kv.transpose(-2, -1))
        attn_map /= np.sqrt(self.head_dim)
        attn_map = F.softmax(attn_map, dim=-1)
        attn_map = self.dropout(attn_map)

        # Update the latent arrays.
        Lv = attn_map @ vv
        Lc = attn_map @ vc
        Ld = attn_map @ vd

        # Project back to the original dimensions.
        Lv = rearrange(Lv, "b h n d -> b n (h d)")
        Lv = self.output_proj_v(Lv)
        Lc = rearrange(Lc, "b h n d -> b n (h d)")
        Lc = self.output_proj_c(Lc)
        Ld = rearrange(Ld, "b h n d -> b n (h d)")
        Ld = self.output_proj_d(Ld)

        return Lv, Lc, Ld


class MultisourcePerceiverDecoder(nn.Module):
    """Implements the decoding part of the Perceiver IO model (Jaegle et al., 2022) for
    multiple sources. The decoding query is the sources' embedded coordinates.
    Each source s includes:
    - a values sequence Vs
    - a coordinates sequence Cs
    - a conditioning sequence Ds
    The decoder receives the latent arrays Lv and Lc, and the coordinates
    sequences of all sources. It performs the following steps during the forward pass:
    - Project Lv to keys k_v and values v_v.
    - Project Lc to keys k_c.
    - For each source s:
        - Project Vs to queries q_v and Cs to queries q_c.
        - Compute the attention map A = Softmax(q_c k_c^T + lambda * q_v k_v^T).
        - Compute the output values Vs_out = A @ v_v
        - Project Vs_out to the original values space.
        - Run Vs_out through a (LayerNorm, MLP) with residual connection.
    - Output the values sequences Vs.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        att_inner_ratio,
        num_heads,
        mlp_hidden_layers,
        mlp_inner_ratio,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            att_inner_ratio (float): Ratio of the inner dimension to the values dimension.
            num_heads (int): Number of heads in the attention block.
            mlp_hidden_layers (int): Number of hidden layers in the MLP.
            mlp_inner_ratio (float): Ratio of the inner dimension to the values dimension
                in the MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.att_inner_dim = int(att_inner_ratio * values_dim)
        self.head_dim = self.att_inner_dim // num_heads
        self.num_heads = num_heads
        self.att_lambda = nn.Parameter(torch.randn(1), requires_grad=True)

        # Normalization layers for the latent arrays
        self.lv_norm = nn.LayerNorm(values_dim)
        self.lc_norm = nn.LayerNorm(coords_dim)

        self.coords_k = nn.Sequential(
            nn.Linear(coords_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.coords_q = nn.Sequential(
            nn.Linear(coords_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.values_q = nn.Sequential(
            nn.Linear(values_dim, self.att_inner_dim), RMSNorm(self.att_inner_dim)
        )
        self.values_vk = nn.Linear(values_dim, self.att_inner_dim * 2)

        self.output_proj = nn.Sequential(
            nn.Linear(self.att_inner_dim, values_dim), nn.Dropout(dropout)
        )

        self.output_mlp = nn.Sequential(
            nn.LayerNorm(values_dim), MLP(values_dim, mlp_hidden_layers, mlp_inner_ratio)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Lv, Lc):
        """
        Args:
            x (dict of str to tensor): Dictionary of inputs for each source, such
                that x[src] contains at least the entry "coords".
                The values are expected to have shape (B, ..., Dv) where ... are the
                spatial dimensions, e.g. (B, h, w, Dv) for 2D sources.
            Lv (tensor): Latent array for the values.
            Lc (tensor): Latent array for the coordinates.

        Returns:
            dict of str to tensor: Dictionary {src: Vs} of output values sequences.
        """
        # Normalize the latents.
        Lv = self.lv_norm(Lv)
        Lc = self.lc_norm(Lc)

        # Project the latents to values and keys.
        v_v, k_v = self.values_vk(Lv).chunk(2, dim=-1)
        k_c = self.coords_k(Lc)
        k_c = rearrange(k_c, "b n (h d) -> b h n d", h=self.num_heads)
        v_v = rearrange(v_v, "b n (h d) -> b h n d", h=self.num_heads)
        k_v = rearrange(k_v, "b n (h d) -> b h n d", h=self.num_heads)

        outputs = {}
        for src in x:
            # The original embeddings are already normalized.
            C = x[src]["coords"]
            V = x[src]["values"]
            spatial_shape = C.shape[1:-1]  # e.g. (h, w) for 2D sources

            # Flatten the spatial dimensions and project to queries.
            V = rearrange(V, "b ... d -> b (...) d")
            q_v = self.values_q(V)
            q_v = rearrange(q_v, "b n (h d) -> b h n d", h=self.num_heads)
            C = rearrange(C, "b ... d -> b (...) d")
            q_c = self.coords_q(C)
            q_c = rearrange(q_c, "b n (h d) -> b h n d", h=self.num_heads)

            # Compute the attention map.
            attn_map = q_c @ k_c.transpose(-2, -1) + self.att_lambda * (
                q_v @ k_v.transpose(-2, -1)
            )
            attn_map /= np.sqrt(self.head_dim)
            attn_map = F.softmax(attn_map, dim=-1)
            attn_map = self.dropout(attn_map)

            # Compute the output values.
            V_out = attn_map @ v_v
            V_out = rearrange(V_out, "b h n d -> b n (h d)")
            V_out = self.output_proj(V_out)

            # Reshape the values to their original dimensionality.
            V_out = V_out.view(V.shape[0], *spatial_shape, self.values_dim)

            # Apply the MLP.
            V_out = self.output_mlp(V_out)

            outputs[src] = V_out

        return outputs
