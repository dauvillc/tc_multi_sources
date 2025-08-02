from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from multi_sources.models.motif_double.small_layers import RMSNorm


class MultisourcesWindowedCrossAttention(nn.Module):
    """Computes attention across the sources using a windowed system.
    The attention weights are computed using the averages of the windows' values and coordinates,
    while the re-weighted values are computed using all values in the sources.
    - Each source is divided into windows, and every vector is projected to a query and key.
    - The queries and keys are averaged within each window to obtain one set of queries and keys
        per window.
    - The keys and queries of all windows are concatenated into two sets of sequences,
        (Qv, Kv) for values and (Qc, Kc) for coordinates.
    - The attention weights A are computed as
      A = softmax((Qv @ Kv^T / sqrt(Dv)) + (Qc @ Kc^T / sqrt(Dc)))
    - Within each window, all vectors' values are projected to a (potentially smaller) dimension,
        then concatenated along the feature dimension to form a single vector per window.
    - All windows' values are concatenated into a single sequence of vectors Vv.
    - The re-weighted values are computed as V' = A @ Vv.
    - V' is split back into windows, projected back to the original values dimension,
        and summed back to the original vectors.

    For 2D sources, the windows are square patches of size window_size x window_size.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_ratio_qk,
        window_size,
        inner_ratio_v=0.25,
        num_heads=8,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Dimension of the values.
            coords_dim (int): Dimension of the coordinates.
            inner_ratio_qk (float): Ratio of the inner dimensions to the original dimensions,
                used for the Q and K projections.
            window_size (int): Size of the window for attention.
            update_coords (bool): Whether to update the coordinates based on attention.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.inner_qk_v_dim = int(values_dim * inner_ratio_qk)
        self.inner_qk_c_dim = int(coords_dim * inner_ratio_qk)
        # Find the next multiple of num_heads for the inner value dimension
        self.inner_v_v_dim = int(values_dim * inner_ratio_v)
        if self.inner_v_v_dim % num_heads != 0:
            self.inner_v_v_dim += num_heads - (self.inner_v_v_dim % num_heads)

        # Projections to Qv, Kv, Qc, Kc
        self.values_qk_proj = nn.Sequential(
            nn.Linear(values_dim, self.inner_qk_v_dim * 2), RMSNorm(self.inner_qk_v_dim * 2)
        )
        self.coords_qk_proj = nn.Sequential(
            nn.Linear(coords_dim, self.inner_qk_c_dim * 2), RMSNorm(self.inner_qk_c_dim * 2)
        )

        # Projection (compression) of values to Vv
        self.values_v_proj = nn.Linear(
            values_dim, self.inner_v_v_dim, bias=False
        )  # We don't normalize the projected values
        # Projection back to the original values dimension
        self.values_v_back_proj = nn.Linear(self.inner_v_v_dim, values_dim, bias=False)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary of inputs, such that
                inputs[(source_name, index)] contains the keys "coords", "values".
                where (source_name, index) is a tuple with the source name and the observation index
                (0 = most recent).
                The values are expected of shape (B, ..., Dv) and the coordinates of shape
                (B, ..., Dc), where ... is the spatial dimensions of the embedded source,
                e.g. (h, w) for 2D sources.

        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the keys "values" with the updated values.
        """
        keys_v, queries_v, keys_c, queries_c = {}, {}, {}, {}
        values_v = {}
        windowed_shapes, n_windows = {}, []
        for source, source_data in inputs.items():
            C, V = source_data["coords"], source_data["values"]
            _, *spatial_dims, _ = V.shape

            # First step: reshape each source into windows
            if len(spatial_dims) == 2:
                # Pad the spatial dimensions to be divisible by the window size
                pad_h = (self.window_size - spatial_dims[0] % self.window_size) % self.window_size
                pad_w = (self.window_size - spatial_dims[1] % self.window_size) % self.window_size
                V = F.pad(V, (0, 0, 0, pad_w, 0, pad_h))
                C = F.pad(C, (0, 0, 0, pad_w, 0, pad_h))
                # Reshape to windows.
                V = rearrange(
                    V,
                    "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d",
                    w1=self.window_size,
                    w2=self.window_size,
                )
                C = rearrange(
                    C,
                    "b (Wh w1) (Ww w2) d -> b Wh Ww (w1 w2) d",
                    w1=self.window_size,
                    w2=self.window_size,
                )
            else:
                raise NotImplementedError("Only 2D sources are supported.")

            # Average the values and coordinates within each window
            V_avg = V.mean(dim=-2)  # (b, Wh, Ww, Dv)
            C_avg = C.mean(dim=-2)
            # Store the shape of the windowed source and the number of windows
            windowed_shapes[source] = V.shape[1:3]
            n_windows.append(prod(windowed_shapes[source]))

            # Project to Qv, Kv, Qc, Kc
            qk_v = self.values_qk_proj(V_avg)
            qk_c = self.coords_qk_proj(C_avg)
            # Separate the attention heads and reshape to a sequence
            qk_v = rearrange(qk_v, "b Wh Ww (H e) -> b H (Wh Ww) e", H=self.num_heads)
            qk_c = rearrange(qk_c, "b Wh Ww (H e) -> b H (Wh Ww) e", H=self.num_heads)
            # Split into queries and keys
            queries_v[source], keys_v[source] = qk_v.chunk(2, dim=-1)
            queries_c[source], keys_c[source] = qk_c.chunk(2, dim=-1)

            # Project values to Vv and stack the vectors of each window along the
            # feature dimension to form a single vector per window.
            Vv = self.values_v_proj(V)
            values_v[source] = rearrange(
                Vv,
                "b Wh Ww n (e H) -> b H (Wh Ww) (n e)",
                H=self.num_heads,
            )

        # Concatenate all sequences across sources
        queries_v = torch.cat(list(queries_v.values()), dim=-2)  # (B, H, N, Dv)
        keys_v = torch.cat(list(keys_v.values()), dim=-2)  # (B, H, N, Dv)
        # (B, H, N, Dv' * n // H) where n is the number of elements in a window
        values_v = torch.cat(list(values_v.values()), dim=-2)
        queries_c = torch.cat(list(queries_c.values()), dim=-2)  # (B, H, N, Dc)
        keys_c = torch.cat(list(keys_c.values()), dim=-2)  # (B, H, N, Dc)

        # Compute attention weights
        attn_weights = (queries_v @ keys_v.transpose(-2, -1)) / self.inner_qk_v_dim**0.5 + (
            queries_c @ keys_c.transpose(-2, -1)
        ) / self.inner_qk_c_dim**0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute the re-weighted values
        reweighted_values = attn_weights @ values_v  # (B, H, N, Dv' * n // H)
        # Project back to the original values dimension
        reweighted_values = rearrange(
            reweighted_values,
            "b H N (n e) -> b N n (e H)",
            n=self.window_size**2,
            e=self.inner_v_v_dim // self.num_heads,
        )  # (B, N, n, Dv')
        reweighted_values = self.values_v_back_proj(reweighted_values)  # (B, N, n, Dv)
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
