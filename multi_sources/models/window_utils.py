"""Implements utility functions to deal with windows in tensors."""

import torch
import torch.nn.functional as F
from einops import rearrange


def source_to_2D_windows(sequences, tokens_shape, window_size, shift, mask=None):
    """Given a list of flattened sequence of tokens coming from a 2D source,
    partitions each into windows of a given size.
    If along a dimension the number of tokens is not a multiple of the window size,
    the last window is padded.
    Args:
        seq (list of torch.Tensor): Flattened sequences of tokens of common shape (B, N, D).
        tokens_shape (tuple): Shape of the original 2D source after patchifying
            but before flattening, i.e. (H // patch_size, W // patch_size).
        window_size (int): Size of the window.
        shift (int): Shift of the first window from the top-left corner.
        mask (torch.Tensor, optional): Boolean mask of shape (B, N), which will be
            reshaped to windows as well. Tokens corresponding to the new padding will
            be set to True. If not given, a new mask will be created.
    Returns:
        out (list of torch.Tensor): List of partitioned tensors, each of shape
            (B * Nw, W * W, D), where Nw is the number of windows and W is the window size.
        mask (torch.Tensor): Boolean mask of shape (B * Nw, W * W) where mask[b, i, j] is True
            if the token at position (i, j) in window b is a padding token or mask[b, i, j] was
            True in the original mask.
    """
    # First, identify the required padding.
    H, W = tokens_shape
    # Compute the amount of padding needed to make h and w multiples of the window size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if mask is None:
        B, N, _ = sequences[0].shape
        mask = torch.zeros((B, N, 1), dtype=torch.bool, device=sequences[0].device)
    else:
        # Add an extra dim to the mask so that it can be treated like the other sequences
        mask = mask.unsqueeze(-1)  # (B, N, 1)
    out = []
    for seq in sequences + [mask]:
        # Reshape the sequence to its original 2D shape
        seq = rearrange(seq, "b (h w) d -> b h w d", h=H, w=W)
        # Pad the sequence
        seq = F.pad(seq, (0, 0, 0, pad_w, 0, pad_h), value=1)
        # Roll the sequence towards the bottom-right corner to account for the shift
        seq = torch.roll(seq, shifts=(shift, shift), dims=(1, 2))
        # Reshape the sequence to windows
        out.append(
            rearrange(
                seq,
                "b (wh w1) (ww w2) d -> (b wh ww) (w1 w2) d",
                wh=(H + pad_h) // window_size,
                ww=(W + pad_w) // window_size,
            )
        )
    mask = out.pop().squeeze(-1)  # (B * Nw, W * W)
    return out, mask


def windows_2D_to_source(sequences, tokens_shape, window_size, shift):
    """Inverse of source_to_2D_windows.
    Args:
        sequences (list of torch.Tensor): List of partitioned tensors, each of shape
            (B * Nw, W * W, D), where Nw is the number of windows and W is the window size.
        tokens_shape (tuple): Shape of the original 2D source.
        window_size (int): Size of the window.
        shift (int): Shift of the first window from the top-left corner.

    Returns:
        out (list of torch.Tensor): Flattened sequences of tokens of common shape (B, N, D).
    """
    H, W = tokens_shape
    # Compute the amount of padding needed to make h and w multiples of the window size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    out = []
    for seq in sequences:
        # Reshape the sequence to windows
        seq = rearrange(
            seq,
            "(b wh ww) (w1 w2) d -> b (wh w1) (ww w2) d",
            wh=(H + pad_h) // window_size,
            ww=(W + pad_w) // window_size,
            w1=window_size,
            w2=window_size,
        )
        # Roll the sequence back to the top-left corner
        seq = torch.roll(seq, shifts=(-shift, -shift), dims=(1, 2))
        # Remove the padding
        seq = seq[:, :H, :W]
        # Flatten the sequence
        out.append(rearrange(seq, "b h w d -> b (h w) d"))
    return out
