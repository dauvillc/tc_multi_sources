"""Implements embedding layers."""

import torch
import torch.nn as nn


class LinearEmbedding(nn.Module):
    """Embeds a vector using a linear layer."""

    def __init__(self, input_dim, output_dim, norm=False):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.use_norm = norm
        if norm:
            self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.act(x)
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

        self.embedding = nn.Sequential(
            nn.Conv2d(channels, emb_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
        )
        if norm:
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
        Returns:
            embedded_image: torch.Tensor of shape (B, h, w, emb_dim).
        """
        # Compute padding dynamically
        H, W = image.shape[2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        pad = nn.ZeroPad2d((0, pad_w, 0, pad_h))

        image = pad(image)
        embedded_image = self.embedding(image)  # (B, emb_dim, h, w)
        embedded_image = embedded_image.permute(0, 2, 3, 1)  # (B, h, w, emb_dim)

        if hasattr(self, "norm"):
            embedded_image = self.norm(embedded_image)
        return embedded_image


class SourcetypeEmbedding2d(nn.Module):
    """A class that embeds the values and coordinates of a 2D source, including optional
    characteristic variables.

    This module handles both coordinate embeddings (latitude, longitude,
    and time) and values embeddings (channels, masks, diffusion timestep),
    then outputs their embedded representations.
    Additionally, a conditioning tensor is computed that embeds the conditioning
    that isn't the values or the spatio-temporal coordinates. This includes:
    - The characteristic variables.
    - The diffusion timestep.
    If those elements aren't given, the conditioning tensor is set to None.
    """

    def __init__(
        self,
        channels,
        patch_size,
        values_dim,
        coords_dim,
        n_charac_vars=0,
        use_diffusion_t=True,
        pred_mean_channels=0,
        include_coords_in_conditioning=False,
    ):
        """
        Args:
            channels (int): Number of channels for the source data, excluding
                land-sea and availability masks.
            patch_size (int): Size of the patches to be used for convolution.
            values_dim (int): Dimension of the values embedding space.
            coords_dim (int): Dimension of the coordinate embedding space.
            n_charac_vars (int): Number of optional characteristic variables.
            use_diffusion_t (bool): Whether to include a diffusion timestep embedding.
            pred_mean_channels (int): Number of channels for the predicted mean. If zero,
                the predicted mean is not used.
            include_coords_in_conditioning (bool): Whether to include the coordinates
                in the conditioning tensor.
        """
        super().__init__()

        # Values embedding layers
        self.values_patch_size = patch_size
        self.use_diffusion_t = use_diffusion_t
        self.use_predicted_mean = pred_mean_channels > 0
        self.include_coords_in_conditioning = include_coords_in_conditioning

        ch = channels + 2 + pred_mean_channels  # landmask + availability + predicted mean
        self.values_embedding = ConvPatchEmbedding2d(ch, patch_size, values_dim, norm=True)

        # Coords embedding layers
        self.coords_patch_size = patch_size
        self.coords_emb_dim = coords_dim
        self.coords_embedding = ConvPatchEmbedding2d(3, patch_size, coords_dim, norm=False)
        self.time_embedding = nn.Linear(1, coords_dim)
        self.coords_norm = nn.LayerNorm(coords_dim)

        # Conditioning embedding layers
        # - spatial conditioning
        ch_spatial_cond = 1  # landmask
        ch_spatial_cond += int(self.include_coords_in_conditioning) * 3  # coords
        self.spatial_cond_embedding = ConvPatchEmbedding2d(
            ch_spatial_cond, patch_size, values_dim, norm=False
        )
        # - 0D / 1D embeddings
        # there's always at least 1 channel: the availability flag
        ch_cond = 1 + n_charac_vars + int(self.use_diffusion_t)
        ch_cond += int(self.include_coords_in_conditioning)  # time coordinate
        self.cond_embedding = LinearEmbedding(ch_cond, values_dim, norm=False)
        # final layer normalization for the conditioning
        self.cond_norm = nn.LayerNorm(values_dim)

    def forward(self, data):
        """
        Args:
            data (dict): Must contain:
                - "coords": (B, 3, H, W) coordinate tensor.
                - "dt": (B,) time delta tensor.
                - "values": (B, C, H, W) source pixel data.
                - "avail_mask": (B, H, W) data availability mask.
                - "landmask": (B, H, W) land-sea mask.
                - "avail": (B,) tensor, valued 1 if the sample is available, -1 if missing,
                    and 0 if masked (and to be reconstructed).
                - Optionally "characs": (B, n_characs_vars) characteristic variables.
                - Optionally "diffusion_t": (B,) diffusion timestep.
                - Optionally "pred_mean": (B, C_out, H, W), predicted mean.
        Returns:
            embedded_values: (B, h, w, values_dim) tensor of embedded values.
            embedded_coords: (B, h, w, coords_dim) tensor of embedded coordinates.
            conditioning: (B, h, w, values_dim) tensor containing the conditioning, or
                None if no conditioning is given.
        """

        # Embed values
        with torch.backends.cudnn.flags(enabled=self.values_patch_size <= 4):
            values = data["values"]
            avail_mask = data["avail_mask"].unsqueeze(1)
            landmask = data["landmask"].unsqueeze(1)
            values = torch.cat([values, avail_mask, landmask], dim=1)
            if self.use_predicted_mean:
                predicted_mean = data["pred_mean"]
                values = torch.cat([values, predicted_mean], dim=1)
            embedded_values = self.values_embedding(values)

        # Embed coords
        coords = data["coords"]
        dt = data["dt"].view(coords.size(0), 1, 1, 1)
        embedded_coords = self.coords_embedding(coords)
        embedded_coords += self.time_embedding(dt)
        embedded_coords = self.coords_norm(embedded_coords)

        # Conditioning tensor
        available_conditioning = []
        b = values.size(0)
        # Spatial conditionings (embedded together via patch embedding)
        # - Land-sea mask
        spatial_cond = [landmask]
        # - optionally, the spatial coordinates
        if self.include_coords_in_conditioning:
            spatial_cond.append(coords)
        spatial_cond = torch.cat(spatial_cond, dim=1)  # (B, C_spatial_cond, H, W)
        spatial_cond = self.spatial_cond_embedding(spatial_cond)
        available_conditioning.append(spatial_cond)
        # Non-spatial conditionings: we'll concatenate the ones that are available
        # and embed them with a single linear embedding.
        conds = []
        # - Availability flag (always used)
        conds.append(data["avail"].float().view(-1, 1))  # (B, 1)
        # - Characteristic variables
        if "characs" in data and data["characs"] is not None:
            conds.append(data["characs"])  # (B, n_characs_vars)
        # - Diffusion timestep
        if self.use_diffusion_t:
            diffusion_t = data["diffusion_t"].view(b, 1)
            conds.append(diffusion_t)
        # - Optionally, the time coordinate (dt)
        if self.include_coords_in_conditioning:
            conds.append(data["dt"].view(b, 1))
        # Concatenate the conditioning tensors
        conds = torch.cat(conds, dim=1)  # (B, ch_cond)
        conds = self.cond_embedding(conds)  # (B, values_dim)
        # Reshape to sum to the spatial dimensions
        conds = conds.view(b, 1, 1, -1)
        available_conditioning.append(conds)
        # Finally, sum the available conditioning tensors
        # and apply layer normalization.
        if len(available_conditioning) > 0:
            conditioning = sum(available_conditioning)
            conditioning = self.cond_norm(conditioning)
        else:
            conditioning = None

        return embedded_values, embedded_coords, conditioning


class SourcetypeEmbedding0d(nn.Module):
    """A class that embeds the values and time of a 0D source, including optional
    characteristic variables.

    This module handles both time embeddings and values embeddings (channels, masks),
    then outputs their embedded representations.
    Additionally, a conditioning tensor is computed that embeds the conditioning
    that isn't the values or the time. This includes:
    - The characteristic variables.
    - The diffusion timestep.
    If those elements aren't given, the conditioning tensor is set to None.
    """

    def __init__(
        self,
        channels,
        values_dim,
        coords_dim,
        n_charac_vars=0,
        use_diffusion_t=True,
        pred_mean_channels=0,
    ):
        """
        Args:
            channels (int): Number of channels for the source data, excluding
                land-sea and availability masks.
            values_dim (int): Dimension of the values embedding space.
            coords_dim (int): Dimension of the coordinate (time) embedding space.
            n_charac_vars (int): Number of optional characteristic variables.
            use_diffusion_t (bool): Whether to include a diffusion timestep embedding.
            pred_mean_channels (int): Number of channels for the predicted mean. If zero,
                the predicted mean is not used.
        """
        super().__init__()

        # Values embedding layers
        self.use_diffusion_t = use_diffusion_t
        self.use_predicted_mean = pred_mean_channels > 0

        ch = channels + 2 + pred_mean_channels  # landmask + availability + predicted mean
        self.values_embedding = nn.Sequential(
            nn.Linear(ch, values_dim), nn.GELU(), nn.LayerNorm(values_dim)
        )

        # Coords embedding layers
        self.coords_emb_dim = coords_dim
        self.coords_embedding = nn.Linear(3, coords_dim)
        self.time_embedding = nn.Linear(1, coords_dim)
        self.coords_norm = nn.LayerNorm(coords_dim)

        # Conditioning embedding layers
        # there are always at least 2 channels: landmask and availability flag
        ch_cond = 2 + pred_mean_channels + n_charac_vars + int(self.use_diffusion_t)
        self.cond_embedding = LinearEmbedding(ch_cond, values_dim, norm=True)

    def forward(self, data):
        """
        Args:
            data (dict): Must contain:
                - "coords": (B, 3) coordinate tensor.
                - "dt": (B,) time delta tensor.
                - "values": (B, C) source pixel data.
                - "avail_mask": (B,) data availability mask.
                - "landmask": (B,) land-sea mask.
                - "avail": (B,) tensor, valued 1 if the sample is available, -1 if missing,
                    and 0 if masked (and to be reconstructed).
                - Optionally "characs": (B, n_characs_vars) characteristic variables.
                - Optionally "diffusion_t": (B,) diffusion timestep.
                - Optionally "pred_mean": (B, C_out), predicted mean.
        Returns:
            embedded_values: (B, values_dim) tensor of embedded values.
            embedded_coords: (B, coords_dim) tensor of embedded coordinates.
            conditioning: (B, values_dim) tensor containing the conditioning, or
                None if no conditioning is given.
        """

        # Embed values
        values = data["values"]
        avail_mask = data["avail_mask"].unsqueeze(1)
        landmask = data["landmask"].unsqueeze(1)
        values = torch.cat([values, avail_mask, landmask], dim=1)
        if self.use_predicted_mean:
            predicted_mean = data["pred_mean"]
            values = torch.cat([values, predicted_mean], dim=1)
        embedded_values = self.values_embedding(values)

        # Embed coords
        coords = data["coords"]
        dt = data["dt"].view(coords.size(0), 1)
        embedded_coords = self.coords_embedding(coords)
        embedded_coords += self.time_embedding(dt)
        embedded_coords = self.coords_norm(embedded_coords)

        # Conditioning tensor
        available_conditioning = [data["landmask"].view(-1, 1), data["avail"].view(-1, 1)]
        if "characs" in data and data["characs"] is not None:
            available_conditioning.append(data["characs"])
        if self.use_diffusion_t:
            diffusion_t = data["diffusion_t"].view(-1, 1)
            available_conditioning.append(diffusion_t)
        if self.use_predicted_mean:
            available_conditioning.append(predicted_mean)
        # Concatenate the conditioning tensors and embed them at once
        if len(available_conditioning) > 0:
            conditioning = torch.cat(available_conditioning, dim=1)
            conditioning = self.cond_embedding(conditioning)
        else:
            conditioning = None

        return embedded_values, embedded_coords, conditioning
