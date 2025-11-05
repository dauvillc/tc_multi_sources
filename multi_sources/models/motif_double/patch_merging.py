"""Implements the PatchMerging and PatchSplitting modules."""

import torch
from torch import nn
from torch.nn import functional as F


class PatchMerging(nn.Module):
    """Implements the patch merging of Swin Transformer."""

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * input_dim)
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).
        Returns:
            torch.Tensor: Output tensor of shape (B, H//2, W//2, C*2).
        """
        # Pad the input tensor to have spatial dims divisible by 2
        B, H, W, C = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2), mode="constant", value=0)

        # Stack the corners of each 4x4 patch
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H//2, W//2, 4*C)

        # Reduce the dimensionality and normalize
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchSplitting(nn.Module):
    """Inverse operation of the patch merging."""

    def __init__(self, input_dim):
        super().__init__()
        self.expansion = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(2 * input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H//2, W//2, C).
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C//2).
        """
        # Expand the dimensionality
        x = self.expansion(x)
        x = self.norm(x)

        # Reshape and split the channels
        B, H_half, W_half, C4 = x.shape
        C = C4 // 4
        x0 = x[:, :, :, 0 * C : 1 * C]
        x1 = x[:, :, :, 1 * C : 2 * C]
        x2 = x[:, :, :, 2 * C : 3 * C]
        x3 = x[:, :, :, 3 * C : 4 * C]

        # Rearrange to reconstruct the original spatial dimensions
        H, W = H_half * 2, W_half * 2
        x = torch.zeros((B, H, W, C), device=x.device, dtype=x.dtype)
        x[:, 0::2, 0::2, :] = x0
        x[:, 1::2, 0::2, :] = x1
        x[:, 0::2, 1::2, :] = x2
        x[:, 1::2, 1::2, :] = x3

        return x


class MultiSourcePatchMerging(nn.Module):
    """Implements the patch merging of Swin Transformer for multi-source inputs.
    Downsamples the values, coordinates and the conditioning of each source by a factor 2,
    while doubling the feature dimension.
    """

    def __init__(self, values_dim, coords_dim, cond_dim):
        super().__init__()
        self.values_merging = PatchMerging(values_dim)
        self.coords_merging = PatchMerging(coords_dim)
        self.conditioning_merging = PatchMerging(cond_dim)

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary of inputs, such that
                inputs[(source_name, index)] contains the keys "coords", "values"
                and "conditioning",
                where (source_name, index) is a tuple with the source name and the observation index
                (0 = most recent).
                The values are expected of shape (B, ..., Dv),
                the coordinates of shape (B, ..., Dc),
                the conditioning of shape (B, ..., Dcond),
                where ... is the spatial dimensions of
                the embedded source, e.g. (h, w) for 2D sources.

        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the keys "values", "coords"
                and "conditioning".
        """
        # Apply patch merging to each source independently
        outputs = {}
        for source, src_data in inputs.items():
            coords = src_data["coords"]
            values = src_data["values"]
            conditioning = src_data["conditioning"]

            # Merge patches
            merged_coords = self.coords_merging(coords)
            merged_values = self.values_merging(values)
            merged_conditioning = self.conditioning_merging(conditioning)

            outputs[source] = {
                "coords": merged_coords,
                "values": merged_values,
                "conditioning": merged_conditioning,
            }
        return outputs


class MultiSourcePatchSplitting(nn.Module):
    """Inverse operation of the patch merging for multi-source inputs.
    Upsamples the values, coordinates and the conditioning of each source by a factor 2,
    while halving the feature dimension.
    """

    def __init__(self, values_dim, coords_dim, cond_dim):
        super().__init__()
        self.values_splitting = PatchSplitting(values_dim)
        self.coords_splitting = PatchSplitting(coords_dim)
        self.conditioning_splitting = PatchSplitting(cond_dim)

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary of inputs, such that
                inputs[(source_name, index)] contains the keys "coords", "values"
                and "conditioning",
                where (source_name, index) is a tuple with the source name and the observation index
                (0 = most recent).
                The values are expected of shape (B, ..., Dv),
                the coordinates of shape (B, ..., Dc),
                the conditioning of shape (B, ..., Dcond),
                where ... is the spatial dimensions of
                the embedded source, e.g. (h, w) for 2D sources.

        Returns:
            dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the keys "values", "coords"
                and "conditioning".
        """
        # Apply patch splitting to each source independently
        outputs = {}
        for source, src_data in inputs.items():
            coords = src_data["coords"]
            values = src_data["values"]
            conditioning = src_data["conditioning"]

            # Split patches
            split_coords = self.coords_splitting(coords)
            split_values = self.values_splitting(values)
            split_conditioning = self.conditioning_splitting(conditioning)

            outputs[source] = {
                "coords": split_coords,
                "values": split_values,
                "conditioning": split_conditioning,
            }
        return outputs
