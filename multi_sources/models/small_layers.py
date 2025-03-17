import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from .utils import pair


class FeedForward(nn.Module):
    def __init__(
        self, values_dim, coords_dim, dropout=0.0, act_layer=nn.GELU, inner_ratio=4, **kwargs
    ):
        super().__init__()
        inner_dim = values_dim * inner_ratio
        self.net = nn.Sequential(
            nn.Linear(values_dim, inner_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, values_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x['source_name'] contains at least the key "embedded_values".
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs['source_name'] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, data in x.items():
            values = data["embedded_values"]
            outputs[source_name] = self.net(values)
        return outputs


class SwiGLU(nn.Module):
    """SwiGLU mlp, adapted from the timm package."""

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_dim=None,
        act_layer=nn.SiLU,
        bias=True,
        dropout=0.0,
    ):
        super().__init__()
        bias = pair(bias)
        drop_probs = pair(dropout)

        self.fc1_g = nn.Linear(values_dim, inner_dim, bias=bias[0])
        self.fc1_v = nn.Linear(values_dim, inner_dim, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.LayerNorm(inner_dim)
        self.fc2 = nn.Linear(inner_dim, values_dim, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x, **kwargs):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x['source_name'] contains at least the key "embedded_values".
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs['source_name'] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, data in x.items():
            values = data["embedded_values"]
            v_gate = self.fc1_g(values)
            v = self.fc1_v(values)
            v = self.act(v_gate) * v
            v = self.drop1(v)
            v = self.norm(v)
            v = self.fc2(v)
            v = self.drop2(v)
            outputs[source_name] = v
        return outputs


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn


# Taken from torchtune
class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        )
        return x_normed * self.scale


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
