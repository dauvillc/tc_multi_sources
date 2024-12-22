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


class ConvPatchEmbedding2d(nn.Module):
    """A module that embeds an image into a sequence of patches using
    a 2D convolutional layer.
    """

    def __init__(self, channels, patch_size, emb_dim, norm=True):
        """
        Args:
            channels (int): The number of channels in the image.
            patch_size (int): The size of the patches.
            emb_dim (int): The dimension of the embedding space.
            norm (bool): Whether to apply layer normalization after the embedding.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Conv2d(
            channels, emb_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        if norm:
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
        Returns:
            embedded_image: torch.Tensor of shape (B, num_patches, emb_dim).
        """
        # Compute padding dynamically
        H, W = image.shape[2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad = nn.ZeroPad2d((0, pad_w, 0, pad_h))

        image = pad(image)
        embedded_image = self.embedding(image)  # (B, emb_dim, h, w)
        # Flatten the spatial dimensions to (B, emb_dim, num_patches) then transpose
        embedded_image = embedded_image.flatten(2).transpose(1, 2)
        if hasattr(self, "norm"):
            embedded_image = self.norm(embedded_image)
        return embedded_image


class CoordinatesEmbedding2d(nn.Module):
    """A module that embeds the coordinates of a sequence of patches.
    Made for 2D coordinates, which means that:
    * the coordinates tensor has shape (B, 3, H, W),
        for latitude, longitude sin and longitude cos.
    * the time delta tensor has shape (B,)
    Patches of the input are separately embedded into tokens of shape (B, emb_dim)
    and concatenated into a sequence of shape (B, num_patches, emb_dim).
    The time coordinates are embedded using a linear layer and summed to the
    embedded spatial coordinates.
    """

    def __init__(self, patch_size, emb_dim):
        """
        Args:
            patch_size (int): The size of the patches.
            emb_dim (int): The dimension of the embedding space.
        """
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.coords_embedding = ConvPatchEmbedding2d(3, patch_size, emb_dim, norm=False)
        self.time_embedding = nn.Linear(1, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, data):
        """
        Args:
            data (dict of str to torch.Tensor): A dictionary containing at least the following keys:
            * coords : A tensor of shape (B, 3, H, W) containing the coordinates.
            * dt : A tensor of shape (B,) containing the time delta.
        Returns:
            embedded_coords: torch.Tensor of shape (B, num_patches, emb_dim).
        """
        coords, dt = data["coords"], data["dt"]
        B = coords.size(0)
        dt = dt.view(B, 1, 1)
        embedded_dt = self.time_embedding(dt)
        embedded_coords = self.coords_embedding(coords)
        embedded_coords += embedded_dt
        embedded_coords = self.norm(embedded_coords)
        return embedded_coords


class SupplementaryValuesEmbedding2d(nn.Module):
    """A module that embeds masks into a sequence of patches using
    a 2D convolutional layer. Takes as input a land-sea mask and
    embeds it into a sequence of patches.
    """

    def __init__(self, patch_size, emb_dim):
        """
        Args:
            patch_size (int): The size of the patches.
            emb_dim (int): The dimension of the embedding space.
        """
        super().__init__()
        self.embedding = ConvPatchEmbedding2d(1, patch_size, emb_dim)

    def forward(self, data):
        """
        Args:
            data (dict of str to torch.Tensor): A dictionary containing at least:
                * 'landmask': torch.Tensor of shape (B, H, W) containing the land-sea mask
        Returns:
            embedded_masks: torch.Tensor of shape (B, num_patches, emb_dim).
        """
        masks = data["landmask"].unsqueeze(1)  # (B, 1, H, W)
        return self.embedding(masks)


class SourceSpecificEmbedding2d(nn.Module):
    """A module that embeds a single source into a sequence of patches.
    Made for 2D sources, which means that the pixels tensor has shape (B, C, H, W).
    An additional channel containing the availability mask is added to the source.
    Patches of the input are separately embedded into tokens of shape (B, emb_dim)
    and concatenated into a sequence of shape (B, num_patches, emb_dim).
    """

    def __init__(self, channels, patch_size, values_dim):
        """
        Args:
            channels (int): The number of channels in the source, not counting
                the availability mask.
            patch_size (int): The size of the patches to be embedded.
            values_dim (int): The dimension of the embedding space for the values.
        """
        super().__init__()
        self.patch_size = patch_size
        # Values embedding: embeds the pixels using a strided convolution
        self.values_embedding = ConvPatchEmbedding2d(channels + 1, patch_size, values_dim)

    def forward(self, data):
        """
        Args:
            data (dict of str to torch.Tensor): A dictionary containing at least the following keys:
                * 'values': torch.Tensor of shape (B, C, H, W) containing the pixels of the source.
                * 'avail_mask': torch.Tensor of shape (B, H, W) containing the availability mask.
        Returns:
            embedded_values: torch.Tensor of shape (B, num_patches, values_dim).
        """
        # Disable cudNN here if the patch size is superior to 4,
        # as it raises an error for large patch sizes.
        with torch.backends.cudnn.flags(enabled=self.patch_size <= 4):
            # Add the availability mask as an additional channel
            values = data["values"]
            avail_mask = data["avail_mask"].unsqueeze(1)
            values = torch.cat([values, avail_mask], dim=1)
            embedded_values = self.values_embedding(values)

        return embedded_values


class SourcetypeEmbedding2d(nn.Module):
    """A module that embeds 2d sources from a common source type into a sequence of patches.
    Made for 2D sources, which means that the pixels tensor has shape (B, C, H, W).
    An additional channel containing the availability mask is added to the source.
    Patches of the input are separately embedded into tokens of shape (B, emb_dim)
    and concatenated into a sequence of shape (B, num_patches, emb_dim).

    A source type is a set of sources that supposedly contain the same information
    but from different sources that may have different characteristics. For example,
    microwave brightness temperatures from different satellites. A source type is
    associated with a set of context variables, the same for all sources of the type.
    Those describe more precisely the characteristics of the sources within the type,
    for example the frequency of the microwave sensor or its ground resolution.

    All sources in a given type must have the same channels, in the same order.
    """

    def __init__(self, channels, n_context_vars, patch_size, values_dim):
        """Args:
            channels (int): The number of channels in the source, not counting
                the availability mask.
            n_context_vars (int): The number of context variables for the source type.
            patch_size (int): The size of the patches to be embedded.
            values_dim (int): The dimension of the embedding space for the values.
        """
        super().__init__()
        self.patch_size = patch_size
        # Values embedding: embeds the pixels using a strided convolution
        self.values_embedding = ConvPatchEmbedding2d(
            channels + 1, patch_size, values_dim, norm=False
        )
        self.context_embedding = LinearEmbedding(n_context_vars, values_dim)
        self.norm = nn.LayerNorm(values_dim)
    
    def forward(self, data):
        """Args:
            data (dict of str to torch.Tensor): A dictionary containing at least the following
                keys:
                * 'values': torch.Tensor of shape (B, C, H, W) containing the pixels of the source.
                * 'avail_mask': torch.Tensor of shape (B, H, W) containing the availability mask.
                * 'context_vars': torch.Tensor of shape (B, n_context_vars) containing the context
                    variables for the source type.
        Returns:
            embedded_values: torch.Tensor of shape (B, num_patches, values_dim).
        """
        # Disable cudNN here if the patch size is superior to 4,
        # as it raises an error for large patch sizes.
        with torch.backends.cudnn.flags(enabled=self.patch_size <= 4):
            # Add the availability mask as an additional channel
            values = data["values"]
            avail_mask = data["avail_mask"].unsqueeze(1)
            values = torch.cat([values, avail_mask], dim=1)
            embedded_values = self.values_embedding(values)
            context_vars = data["context_vars"]
            embedded_context = self.context_embedding(context_vars)
            embedded_values += embedded_context
            embedded_values = self.norm(embedded_values)
        return embedded_values
