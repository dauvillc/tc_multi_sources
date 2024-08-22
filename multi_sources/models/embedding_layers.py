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


class SharedSourceEmbedding(nn.Module):
    """A module that shares embeddings across sources of the same type.
    A source type is supposed to be associated with a fixed number of context variables,
    e.g. 'passive_microwave' is associated with 'frequency' and 'IFOV'. This module
    maps those context variables to a shared embedding space.
    """
    def __init__(self, context_variables, dim):
        """
        Args:
            context_variables (dict of str to list of str): A mapping from source type
                to the context variables associated with that source type.
            dim (int): The dimension of the shared embedding space.
        """
        super().__init__()
        self.source_types = list(context_variables.keys())
        self.context_variables = context_variables
        self.embeddings = nn.ModuleDict()
        for source_type, variables in context_variables.items():
            self.embeddings[source_type] = nn.Linear(len(variables), dim)

    def forward(self, source_type, context):
        """
        Args:
            source_type (str): The type of the source.
            context (torch.Tensor): A tensor of shape (batch_size, num_context_variables)
                containing the context variables for each source.

        Returns:
            embed: torch.Tensor of shape (batch_size, dim).
        """
        return self.embeddings[source_type](context)
