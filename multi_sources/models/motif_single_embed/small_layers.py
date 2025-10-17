import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, values_dim, dropout=0.0, act_layer=nn.GELU, inner_ratio=4, **kwargs):
        """Args:
        values_dim (int): Dimension of the values.
        dropout (float): Dropout rate.
        act_layer (nn.Module): Activation layer to use.
        inner_ratio (int): Ratio for the inner dimension compared to the input dimension.
        """
        super().__init__()
        values_inner_dim = int(values_dim * inner_ratio)

        # Network for values
        self.values_net = nn.Sequential(
            nn.Linear(values_dim, values_inner_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(values_inner_dim, values_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x[(source_name, index)] contains the key "values".
        Returns:
            dict of str: dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the key "values" with
                the updated values.
        """
        outputs = {}
        for source_name, data in x.items():
            values = data["values"]

            # Process values and coordinates through their respective networks
            updated_values = self.values_net(values)

            # Return both updated values and coordinates
            outputs[source_name] = {"values": updated_values}
        return outputs


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
