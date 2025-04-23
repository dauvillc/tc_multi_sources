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
