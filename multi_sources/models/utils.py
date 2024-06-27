"""Implements small utility functions for models."""

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
