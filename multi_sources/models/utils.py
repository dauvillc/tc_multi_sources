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
        return strs.replace('.', '_')
    if isinstance(strs, list):
        return [s.replace('.', '_') for s in strs]
    if isinstance(strs, dict) or isinstance(strs, DictConfig):
        return {k.replace('.', '_'): v for k, v in strs.items()}


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
    """Normalizes the coordinates across multiple sources, by setting
    their min to 0 and their max to 1.
    The coordinates are assumed to be in the range [-180, 180] for
    longitudes and [-90, 90] for latitudes. NaNs are ignored in the
    computation of the min and max and are left so in the output.
    Args:
        coords (list of torch.Tensor): list of tensors of shape (B, 2, H, W).
    Returns:
        list of torch.Tensor: the normalized coordinates.
    """
    shapes = [c.shape for c in coords]
    flat_coords = [c.view(c.shape[0], 2, -1) for c in coords]
    # Compute the min and max of each source. The coordinates may contain NaNs.
    # to avoid them, we'll replace them with +190 in the min computation (which
    # is higher than any valid latitude or longitude), and with -190 in the max
    # computation.
    min_vals = [torch.nan_to_num(c, 190.).min(dim=-1)[0] for c in flat_coords]  # list of (B, 2)
    max_vals = [torch.nan_to_num(c, -190.).max(dim=-1)[0] for c in flat_coords]
    cross_source_min = torch.stack(min_vals, dim=0).min(dim=0)[0].view(-1, 2, 1, 1)
    cross_source_max = torch.stack(max_vals, dim=0).max(dim=0)[0].view(-1, 2, 1, 1)
    # Normalize the latitudes and longitudes. NaNs remain NaNs.
    coords = [(c - cross_source_min) / (cross_source_max - cross_source_min) for c in coords]
    # Reshape the coordinates.
    coords = [c.view(*s) for c, s in zip(coords, shapes)]
    return coords
