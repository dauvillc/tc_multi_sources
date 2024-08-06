"""Implements embedding layers."""

import torch
import torch.nn as nn


class LinearEmbedding(nn.Module):
    """Embeds a vector using a linear layer."""

    def __init__(self, input_dim, output_dim, norm=False):
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


class SourceEmbedding(nn.Module):
    """Stores an embedding for each source."""

    def __init__(self, source_names, dim):
        super().__init__()
        # Create embeddings for all sources, and associate each source name with an index.
        self.embedding = nn.Embedding(len(source_names), dim)
        self.source_indices = {source_name: i for i, source_name in enumerate(source_names)}

    def forward(self, source_name, shape):
        """
        Returns:
            embed: torch.Tensor of shape (*shape, dim).
        """
        idx = torch.full(shape, self.source_indices[source_name], dtype=torch.long)
        return self.embedding(idx.to(self.embedding.weight.device))
