"""Implements class for a specific ViT architecture."""

from multi_sources.models.vit_classes import FeedForward
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class CoordinatesEmbedding(nn.Module):
    def __init__(self, patch_h, patch_w, dim):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.dim = dim
        self.patch_dim = 2 * patch_h * patch_w
        self.embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_h, p2=patch_w),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.embedding(x)


class CoordinatesAttention(nn.Module):
    """Applies relative self-attention using the spatio-temporal coordinates
    as relative positioning.
    """

    def __init__(self, x_dim, c_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == x_dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.x_norm = nn.LayerNorm(x_dim)
        self.c_norm = nn.LayerNorm(c_dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(c_dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(x_dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, x_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, coords):
        """
        Args:
            x: (b, n, d) tensor, values;
            coords: (b, n, c) tensor, [embedded] spatio-temporal coordinates.
        """
        x = self.x_norm(x)
        coords = self.c_norm(coords)

        q, k = self.to_qk(coords).chunk(2, dim=-1)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2))  # (b, h, n, n)

        attn = self.attend(dots * self.scale)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CoordinatesTransformer(nn.Module):
    """Implements a transformer that receives that on top of its usual patches input,
    receives other embeddings, which are the coordinates of the patches.
    The model first computes the self-attention between the coordinate embeddings, and
    then the traditional transformer encoding of the values. Each time an attention
    map is computed over the values, it is multiplied by the attention map computed
    over the coordinates, so that the model can learn to attend to the patches based
    on them.
    """

    def __init__(self, x_dim, c_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(x_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CoordinatesAttention(
                            x_dim, c_dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        FeedForward(x_dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, coords):
        # Compute the attention map for the values
        for attn, ff in self.layers:
            attn_map = attn(x, coords)
            x = attn_map + x
            x = ff(x) + x

        return self.norm(x)
