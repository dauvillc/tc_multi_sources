"""Implements classes that are part of a larger VIT model.
Inspired / Taken from lucidrain's vit-pytorch repository.
"""

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_h, patch_w, channels, dim):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.dim = dim
        self.patch_dim = channels * patch_h * patch_w
        self.embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_h, p2=patch_w),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
        )

    def forward(self, x):
        return self.embedding(x)


class ResNet(nn.Module):
    """Implements a ResNet FCN model to place at the end of the VIT model,
    mostly to correct inconsistencies at the border of the patches.
    """

    def __init__(self, channels, inner_channels, n_blocks, kernel_size=3):
        """
        Args:
            channels (int): Number of input AND output channels.
            inner_channels (int): Number of channels in the intermediate layers.
            n_blocks (int): Number of residual blocks.
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        self.input_channels = channels
        # Create an input convolutional layer to match the number of channels
        self.input_layer = nn.Conv2d(
            channels, inner_channels, kernel_size, padding=kernel_size // 2
        )
        self.layers = nn.ModuleList([])
        for i in range(n_blocks):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(inner_channels),
                    nn.Conv2d(
                        inner_channels, inner_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        inner_channels, inner_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.GELU(),
                )
            )
            channels = inner_channels
        # Create an output convolutional layer for the regression task
        self.output_layer = nn.Conv2d(
            inner_channels, self.input_channels, kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x) + x
        return self.output_layer(x)
