"""Implements small layers that can be shared across different models."""
import torch.nn as nn


class NonLinearEmbedding(nn.Module):
    """Non-linear patch embedding layer."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.LayerNorm(dim_out),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, x):
        return self.embedding(x)
