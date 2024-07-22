"""Implements embedding layers."""

import torch.nn as nn


class LinearEmbedding(nn.Module):
    """Embeds a vector using a linear layer."""
    def __init__(self, input_dim, output_dim, norm=True):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        self.use_norm = norm
        if norm:
            self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.embedding(x)
        if self.use_norm:
            x = self.ln(x)
        return x
