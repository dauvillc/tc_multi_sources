"""Implements small utility functions for models."""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def pair(x):
    """Ensures that x is a pair of integers."""
    if isinstance(x, tuple):
        return x
    return (x, x)


def remove_dots(strs):
    """Replaces dots in a list of strings or dicts with underscores."""
    if isinstance(strs, str):
        return strs.replace(".", "_")
    if isinstance(strs, list):
        return [s.replace(".", "_") for s in strs]
    if isinstance(strs, dict) or isinstance(strs, DictConfig):
        return {k.replace(".", "_"): v for k, v in strs.items()}


def pad_to_next_multiple_of(tensor, multiple_of, **kwargs):
    """Pad an image or a batch of images to the next multiple of a number.
    Args:
        tensor (torch.Tensor): tensor of shape (..., H, W).
        multiple_of (int or tuple of int): if int, the number to which the
            height and width should be padded. If tuple, the first element
            determines the height padding and the second element the width
            padding.
        kwargs: additional arguments to the F.pad function.
    Returns:
        torch.Tensor: padded tensor of shape (..., H_padded, W_padded).
    """
    H, W = tensor.shape[-2:]
    mo_h, mo_w = pair(multiple_of)
    H_padded = H + (-H) % mo_h
    W_padded = W + (-W) % mo_w
    padding = (0, W_padded - W, 0, H_padded - H)
    return F.pad(tensor, padding, **kwargs)


def normalize_coords_across_sources(coords):
    """Normalizes the coordinates across multiple sources, by setting the min
    and max of each channel to 0 and 1 respectively, across all sources.
    Args:
        coords (list of torch.Tensor): list of tensors of shape (B, C, ...).
            For example, if lat/lon then C=2; if lat sin/lon sin/lon cos then C=3.
    Returns:
        list of torch.Tensor: the normalized coordinates, as list of tensors
            of shape (B, C, ...). For each channel, its min across all sources is 0
            and its max is 1.
    """
    shapes = [c.shape for c in coords]
    B, C = coords[0].shape[:2]
    flat_coords = [c.view(c.shape[0], C, -1) for c in coords]
    # Compute the min and max of each source. The coordinates may contain NaNs.
    # to avoid them, we'll replace them with +inf for min and -inf for max.
    min_vals = [torch.nan_to_num(c, float("inf")).min(dim=-1)[0] for c in flat_coords]
    max_vals = [torch.nan_to_num(c, float("-inf")).max(dim=-1)[0] for c in flat_coords]
    cross_source_min = torch.stack(min_vals, dim=0).min(dim=0)[0].view(B, C, -1)
    cross_source_max = torch.stack(max_vals, dim=0).max(dim=0)[0].view(B, C, -1)
    # Normalize the latitudes and longitudes. NaNs remain NaNs.
    coords = [(c - cross_source_min) / (cross_source_max - cross_source_min) for c in flat_coords]
    # Reshape the coordinates.
    coords = [c.view(*s) for c, s in zip(flat_coords, shapes)]
    return coords


def embed_coords_to_sincos(coords_list):
    """Embeds geographical coordinates (latitudes and longitudes) into
    sine and cosine functions.
    Args:
        coords_list (list of torch.Tensor): list of tensors of shape (B, 2, H, W),
            where the first channel is the latitude and the second channel is the
            longitude, between -180 and 180.
    Returns:
        list of torch.Tensor: the embedded coordinates, as tensors of shape
            (B, 3, H, W). The first is the sine of the latitude, the second is
            the sine of the longitude, the third is the cosine of the longitude.
    """
    embedded_coords = []
    for coords in coords_list:
        lat, lon = coords.unbind(dim=1)
        # Ensure the longitudes are in the range [-180, 180].
        lon = torch.where(lon > 180, lon - 360, lon)
        lat_sin = torch.sin((lat * 3.141592653589793) / 180)
        lon_sin = torch.sin((lon * 3.141592653589793) / 180)
        lon_cos = torch.cos((lon * 3.141592653589793) / 180)
        embedded_coords.append(torch.stack([lat_sin, lon_sin, lon_cos], dim=1))
    return embedded_coords
