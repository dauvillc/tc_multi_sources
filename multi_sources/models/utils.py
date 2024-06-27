"""Implements small utility functions for models."""

import torch
import torch.nn.functional as F


def pair(x):
    """Ensures that x is a pair of integers."""
    if isinstance(x, tuple):
        return x
    return (x, x)


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


def normalize_coords_across_sources(coords, ignore_mask=None):
    """Normalizes the coordinates across multiple sources, by setting
    their min to 0 and their max to 1.
    Args:
        coords (list of torch.Tensor): list of tensors of shape (B, 2, H, W).
        ignore_mask (list of torch.Tensor): list of boolean tensor of shape (B, H, W),
            where True indicates that the coordinates at that pixel should be ignored
            during normalization.
    Returns:
        list of torch.Tensor: the normalized coordinates.
    """
    shapes = [c.shape for c in coords]
    coords = [c.view(c.shape[0], 2, -1) for c in coords]
    if ignore_mask is None:
        ignore_mask = [torch.zeros_like(c[:, 0, :]).bool() for c in coords]
    else:
        ignored_mask = [m.view(m.shape[0], 1, -1).expand(-1, 2, -1) for m in ignore_mask]
    min_vals = [
        c.masked_fill(mask, float("inf")).min(dim=-1)[0] for c, mask in zip(coords, ignored_mask)
    ]
    max_vals = [
        c.masked_fill(mask, -float("inf")).max(dim=-1)[0] for c, mask in zip(coords, ignored_mask)
    ]
    # Compute the min and max over each channel (lat and lon)
    # Compute the min and max over all sources
    min_vals = torch.stack(min_vals, dim=0).min(dim=0)[0]  # (B, 2)
    max_vals = torch.stack(max_vals, dim=0).max(dim=0)[0]
    # Normalize the coordinates
    norm_coords = [
        (c.clone() - min_vals[:, :, None]) / (max_vals - min_vals)[:, :, None] for c in coords
    ]
    # Where the mask is True, set the normalized coordinates to the original ones
    for norm_c, c, mask in zip(norm_coords, coords, ignored_mask):
        norm_c[mask] = c[mask]
    return [c.view(s) for c, s in zip(norm_coords, shapes)]
