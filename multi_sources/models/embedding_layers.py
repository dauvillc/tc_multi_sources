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


class ConvPatchEmbedding2d(nn.Module):
    """A module that embeds an image into a sequence of patches using
    a 2D convolutional layer.
    """

    def __init__(self, channels, spatial_shape, patch_size, emb_dim):
        """
        Args:
            channels (int): The number of channels in the image.
            spatial_shape (tuple of int): The spatial shape (H, W) of the image.
            patch_size (int): The size of the patches.
            emb_dim (int): The dimension of the embedding space.
        """
        super().__init__()
        self.patch_size = patch_size
        # Pad the image so that its spatial dimensions are multiples of the patch size
        self.pad = nn.ZeroPad2d(
            (
                0,
                (patch_size - spatial_shape[1] % patch_size) % patch_size,
                0,
                (patch_size - spatial_shape[0] % patch_size) % patch_size,
            )
        )
        self.embedding = nn.Conv2d(
            channels, emb_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
        Returns:
            embedded_image: torch.Tensor of shape (B, num_patches, emb_dim).
        """
        image = self.pad(image)
        embedded_image = self.embedding(image)  # (B, emb_dim, h, w)
        # Flatten the spatial dimensions into a sequence of patches and transpose
        # the tensor to (B, num_patches, emb_dim)
        embedded_image = embedded_image.flatten(2).transpose(1, 2)
        embedded_image = self.norm(embedded_image)
        return embedded_image


class SourceSpecificEmbedding2d(nn.Module):
    """A module that embeds a single source's pixels, land-sea mask and
    spatiotemporal coordinates into a latent space of fixed dimension.
    Made for 2D sources, which means that
    * the pixels tensor has shape (B, C, H, W)
    * the availability mask is (B, H, W)
    * the land-sea mask is (B, H, W)
    * the spatial coordinates are (B, 3, H, W) (for lat sin, lon sin and lon cos)
    * the temporal coordinate is a scalar (B,).
    Patches of the input are separately embedded into tokens of shape (B, emb_dim)
    and concatenated into a sequence of shape (B, num_patches, emb_dim).
    """

    def __init__(self, channels, spatial_shape, patch_size, values_dim, meta_dim):
        """
        Args:
            channels (int): The number of channels in the source.
            spatial_shape (tuple of int): The spatial shape (H, W) of the source.
            patch_size (int): The size of the patches to be embedded.
            values_dim (int): The dimension of the embedding space for the values.
            meta_dim (int): The dimension of the embedding space for the metadata.
        """
        super().__init__()
        self.patch_size = patch_size
        # Values embedding: embeds the pixels, the land-sea mask and the availability mask using
        # a strided convolution.
        self.values_embedding = ConvPatchEmbedding2d(
            channels + 2, spatial_shape, patch_size, values_dim
        )
        # Metadata embedding: same strategy as for the values embedding, but with
        # 4 channels: lat sin, lon sin, lon cos and time.
        self.meta_embedding = ConvPatchEmbedding2d(4, spatial_shape, patch_size, meta_dim)

    def forward(self, data):
        """
        Args:
            data (dict of str to torch.Tensor): A dictionary containing the following keys:
                * 'values': torch.Tensor of shape (B, C, H, W) containing the pixels of the source.
                * 'avail_mask': torch.Tensor of shape (B, H, W) containing the availability mask.
                * 'landmask': torch.Tensor of shape (B, H, W) containing the land-sea mask.
                * 'coords': torch.Tensor of shape (B, 3, H, W) containing the spatial coordinates.
                * 'dt': torch.Tensor of shape (B,) containing the temporal coordinate.
        Returns:
            embedded_values: torch.Tensor of shape (B, num_patches, values_dim).
            embedded_meta: torch.Tensor of shape (B, num_patches, meta_dim).
        """
        values = data["values"]
        avail_mask = data["avail_mask"]
        landmask = data["landmask"]
        coords = data["coords"]
        dt = data["dt"]
        # Embed the values
        values = torch.cat([values, avail_mask.unsqueeze(1), landmask.unsqueeze(1)], dim=1)
        embedded_values = self.values_embedding(values)
        # Embed the metadata
        # - Repeat dt to match the spatial dimensions
        dt = dt.view(-1, 1, 1, 1).repeat(1, 1, *coords.shape[2:])  # (B, 1, H, W)
        # - Concatenate the spatial and temporal coordinates
        meta = torch.cat([coords, dt], dim=1)
        embedded_meta = self.meta_embedding(meta)

        return embedded_values, embedded_meta
