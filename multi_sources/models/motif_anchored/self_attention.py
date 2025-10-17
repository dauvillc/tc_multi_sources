"""Implements versions of the factorized attention layers that also update coordinates."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from multi_sources.models.motif_double.small_layers import RMSNorm


class SeparateWindowedValuesCoordinatesAttention(nn.Module):
    """Attention block that computes the attention over each source independently,
    using a spatial window over the tokens as in the Swin Transformer.
    For 0D sources, does nothing.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_ratio,
        update_coords,
        num_heads=8,
        dropout=0.0,
        window_size=8,
        shifted=False,
        **kwargs,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_ratio (float): Ratio of the inner dimension to the values dimension.
            update_coords (bool): Whether to update the coordinates alongside the values.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            window_size (int): Number of tokens included in each window.
            shifted (bool): Whether to shift the windows by half the window size.
        """
        super().__init__()
        self.values_dim = values_dim
        self.coords_dim = coords_dim
        self.inner_dim = int(inner_ratio * values_dim)
        self.head_dim = self.inner_dim // num_heads
        self.update_coords = update_coords
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.att_lambda = nn.Parameter(torch.randn(1), requires_grad=True)

        # Projection of values to Q, K and V.
        # For the queries and keys, we apply an RMSNorm to stabilize the training.
        self.values_qk = nn.Sequential(
            nn.Linear(values_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        self.values_v = nn.Linear(values_dim, self.inner_dim)
        # Projection of coordinates to Q, K and V.
        self.coords_qk = nn.Sequential(
            nn.Linear(coords_dim, self.inner_dim * 2), RMSNorm(self.inner_dim * 2)
        )
        if update_coords:
            self.coords_v = nn.Linear(coords_dim, self.inner_dim)

        # Dropout layer for the attention map
        self.dropout = nn.Dropout(dropout)

        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        self.indices = torch.tensor(
            np.array([[x, y] for x in range(window_size) for y in range(window_size)])
        )
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

        self.output_proj_v = nn.Sequential(
            nn.Linear(self.inner_dim, values_dim), nn.Dropout(dropout)
        )
        if update_coords:
            self.output_proj_c = nn.Sequential(
                nn.Linear(self.inner_dim, coords_dim), nn.Dropout(dropout)
            )

    def forward(self, x):
        """
        Args:
            x (dict): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys "coords" and "values",
                where (source_name, index) is a tuple containing the source name and observation
                index (0 = most recent).

        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the keys "values" and "coords" with
                the updated values and coordinates.
                If update_coords is False, only "values" is present.
        """
        outputs = {}
        for source_index_pair, source_inputs in x.items():
            spatial_dims = source_inputs["values"].shape[1:-1]
            # If the source is 0D or 1D, do nothing
            if len(spatial_dims) < 2:
                outputs[source_index_pair] = {
                    "values": source_inputs["values"],
                    "coords": source_inputs["coords"],
                }
                continue
            h, w = spatial_dims
            # Project the values to queries, keys and values
            v = source_inputs["values"]
            v = torch.cat([self.values_qk(v), self.values_v(v)], dim=-1)
            v = rearrange(v, "b h w (d k) -> b h w d k", k=3)
            # Project the coordinates to queries, keys and values
            c = source_inputs["coords"]
            if self.update_coords:
                c = torch.cat([self.coords_qk(c), self.coords_v(c)], dim=-1)
                c_k = 3  # Q, K, V for coordinates
            else:
                c = self.coords_qk(c)
                c_k = 2  # Q, K only for coordinates
            c = rearrange(c, "b h w (d k) -> b h w d k", k=c_k)
            # Pad the values and coordinates so that h and w are multiples of the window size
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size
            v = F.pad(v, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
            c = F.pad(c, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
            # Roll the windows if needed
            if self.shifted:
                v = torch.roll(v, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
                c = torch.roll(c, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            # Reshape to windows and separate the heads
            v = rearrange(
                v,
                "b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            c = rearrange(
                c,
                "b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            # Compute the queries, keys and values
            qv, kv, vv = v.chunk(3, dim=6)
            qv, kv, vv = qv.squeeze(6), kv.squeeze(6), vv.squeeze(6)  # (b H Wh Ww w**2 d)
            if self.update_coords:
                qc, kc, vc = c.chunk(3, dim=6)
                qc, kc, vc = qc.squeeze(6), kc.squeeze(6), vc.squeeze(6)
            else:
                qc, kc = c.chunk(2, dim=6)
                qc, kc = qc.squeeze(6), kc.squeeze(6)
            # Matrix product for the queries and keys from the values and coordinates
            dots = (
                qv @ kv.transpose(4, 5) + self.att_lambda * (qc @ kc.transpose(4, 5))
            ) / self.head_dim**0.5
            # (b H Wh Ww w**2 w**2)
            # Add the positional embeddings
            dots += self.pos_embeddings[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]
            # For shifted windows, compute the attention mask
            if self.shifted:
                row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
                row_mask[
                    -self.window_size * (self.window_size // 2) :,
                    0 : -self.window_size * (self.window_size // 2),
                ] = float("-inf")
                row_mask[
                    0 : -self.window_size * (self.window_size // 2),
                    -self.window_size * (self.window_size // 2) :,
                ] = float("-inf")
                column_mask = rearrange(
                    row_mask,
                    "(r w1) (c w2) -> (w1 r) (w2 c)",
                    w1=self.window_size,
                    w2=self.window_size,
                )
                dots[:, :, -1, :] += row_mask
                dots[:, :, :, -1] += column_mask
            # Deduce the attention weights
            att_weights = F.softmax(dots, dim=-1)
            att_weights = self.dropout(att_weights)

            # Dot product to get the updated values
            yv = att_weights @ vv  # (b H Wh Ww w**2 d)
            # Reshape back to the original spatial layout of the tokens
            yv = rearrange(
                yv,
                "b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)",
                w1=self.window_size,
                w2=self.window_size,
                H=self.num_heads,
            )
            # Remove the padding
            yv = yv[:, :h, :w, :]
            outputs[source_index_pair] = {
                "values": self.output_proj_v(yv),
            }

            if self.update_coords:
                # Same treatment for the coordinates
                yc = att_weights @ vc  # (b H Wh Ww w**2 d)
                yc = rearrange(
                    yc,
                    "b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)",
                    w1=self.window_size,
                    w2=self.window_size,
                    H=self.num_heads,
                )
                yc = yc[:, :h, :w, :]
                outputs[source_index_pair]["coords"] = self.output_proj_c(yc)

        return outputs
