from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from multi_sources.models.motif_single_embed.small_layers import RMSNorm


class MultisourcesWindowedCrossAttention(nn.Module):
    """Computes attention across the sources using a windowed system.
    The attention weights are computed using the averages of the windows.
    while the re-weighted values are computed using all vectors in the sources.
    - Each source is divided into windows, and every vector is projected to a query and key.
    - The queries and keys are averaged within each window to obtain one set of queries and keys
        per window.
    - The keys and queries of all windows are concatenated into (Q, K).
    - The attention weights A are computed as
      A = softmax(Q @ K^T / sqrt(D))
    - Within each window, all vectors' values are projected to a (potentially smaller) dimension,
        then concatenated along the feature dimension to form a single vector per window.
    - All windows' values are concatenated into a single sequence of vectors V.
    - The re-weighted values are computed as V' = A @ V.
    - V' is split back into windows, projected back to the original values dimension,
        and summed back to the original vectors.

    For 2D sources, the windows are square patches of size window_size x window_size.
    """

    def __init__(
        self,
        values_dim,
        inner_ratio_qk,
        window_size,
        inner_ratio_v=1.0,
        num_heads=8,
        mask_self_attention=True,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Dimension of the values.
            inner_ratio_qk (float): Ratio of the inner dimensions to the original dimensions,
                used for the Q and K projections.
            window_size (int): Size of the window for attention.
            inner_ratio_v (float, optional): Ratio of the inner dimension to the original
                dimension used for the values projection. Defaults to 1.0.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            mask_self_attention (bool, optional): Whether to mask out attention weights
                between elements of the same source.
                based on attention. Defaults to False.
            dropout (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.values_dim = values_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask_self_attention = mask_self_attention

        self.inner_qk_dim = int(values_dim * inner_ratio_qk)
        # Find the next multiple of num_heads for the inner value dimension
        self.inner_v_dim = int(values_dim * inner_ratio_v)
        if self.inner_v_dim % num_heads != 0:
            self.inner_v_dim += num_heads - (self.inner_v_dim % num_heads)

        # Projections to Q and K
        self.values_qk_proj = nn.Sequential(
            nn.Linear(values_dim, self.inner_qk_dim * 2), RMSNorm(self.inner_qk_dim * 2)
        )

        # Projection (compression) of values to V
        self.v_proj = nn.Linear(
            values_dim, self.inner_v_dim, bias=False
        )  # We don't normalize the projected values
        # Projection back to the original values dimension
        self.v_back_proj = nn.Linear(self.inner_v_dim, values_dim, bias=False)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary of inputs, such that
                inputs[(source_name, index)] contains at least the key "values".
                where (source_name, index) is a tuple with the source name and the observation index
                (0 = most recent).
                The values are expected of shape (B, ..., Dv).

        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the key "values" with the
                updated values of shape (B, ..., Dv).
        """
        keys, queries, values = {}, {}, {}
        windowed_shapes, n_windows = {}, []
        for source, source_data in inputs.items():
            V = source_data["values"]
            _, *spatial_dims, _ = V.shape

            # First step: reshape each source into windows
            if len(spatial_dims) == 2:
                # Pad the spatial dimensions to be divisible by the window size
                pad_h = (self.window_size - spatial_dims[0] % self.window_size) % self.window_size
                pad_w = (self.window_size - spatial_dims[1] % self.window_size) % self.window_size
                V = F.pad(V, (0, 0, 0, pad_w, 0, pad_h))
                # Reshape to windows.
                V = rearrange(
                    V,
                    "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d",
                    w1=self.window_size,
                    w2=self.window_size,
                )
            else:
                raise NotImplementedError("Only 2D sources are supported.")

            # Average the values within each window
            V_avg = V.mean(dim=-2)  # (b, Wh, Ww, Dv)
            # Store the shape of the windowed source and the number of windows
            windowed_shapes[source] = V.shape[1:3]
            n_windows.append(prod(windowed_shapes[source]))

            # Project to Q and K
            qk = self.values_qk_proj(V_avg)
            # Separate the attention heads and reshape to a sequence
            qk = rearrange(qk, "b Wh Ww (H e) -> b H (Wh Ww) e", H=self.num_heads)
            # Split into queries and keys
            queries[source], keys[source] = qk.chunk(2, dim=-1)

            # Project values to V and stack the vectors of each window along the
            # feature dimension to form a single vector per window.
            V = self.v_proj(V)
            values[source] = rearrange(
                V,
                "b Wh Ww n (e H) -> b H (Wh Ww) (n e)",
                H=self.num_heads,
            )

        # Concatenate all sequences across sources
        queries = torch.cat(list(queries.values()), dim=-2)  # (B, H, N, Dv)
        keys = torch.cat(list(keys.values()), dim=-2)  # (B, H, N, Dv)

        # Compute attention weights
        attn_weights = (queries @ keys.transpose(-2, -1)) / self.inner_qk_dim**0.5  # (B, H, N, N)

        if self.mask_self_attention:
            # For each source, the attention weights contain a block centered on the diagonal that
            # correspond to key/query pairs from the same source. In order to prioritize attention
            # between different sources only in this layer, we'll mask out those blocks.
            blocks = [torch.full((n, n), True) for n in n_windows]
            mask = torch.block_diag(*blocks).to(attn_weights.device).unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Concatenate the values across sources
        # (B, H, N, Dv' * n // H) where n is the number of elements in a window
        values = torch.cat(list(values.values()), dim=-2)
        # Compute the re-weighted values
        reweighted_values = attn_weights @ values  # (B, H, N, Dv' * n // H)
        # Project back to the original values dimension
        reweighted_values = rearrange(
            reweighted_values,
            "b H N (n e) -> b N n (e H)",
            n=self.window_size**2,
            e=self.inner_v_dim // self.num_heads,
        )  # (B, N, n, Dv')
        reweighted_values = self.v_back_proj(reweighted_values)  # (B, N, n, Dv)
        # Split back into the sources
        reweighted_values = torch.split(reweighted_values, n_windows, dim=1)

        # Re-insert the updated values back into the windows
        outputs = {}
        for i, (source, (Wh, Ww)) in enumerate(windowed_shapes.items()):
            if len(windowed_shapes[source]) == 2:
                V = rearrange(
                    reweighted_values[i],
                    "b (Wh Ww) (w1 w2) d -> b (Wh w1) (Ww w2) d",
                    Wh=Wh,
                    Ww=Ww,
                    w1=self.window_size,
                    w2=self.window_size,
                )
                # Remove the padding if it was added
                h, w = inputs[source]["values"].shape[1:3]
                V = V[:, :h, :w, :]

            outputs[source] = {
                "values": V,
            }
        return outputs
