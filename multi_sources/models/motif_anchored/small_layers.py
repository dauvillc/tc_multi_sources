import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        values_dim,
        coords_dim,
        update_coords,
        dropout=0.0,
        act_layer=nn.GELU,
        inner_ratio=4,
        **kwargs
    ):
        """Args:
        values_dim (int): Dimension of the values.
        coords_dim (int): Dimension of the coordinates.
        update_coords (bool): Whether to update the coordinates.
        dropout (float): Dropout rate.
        act_layer (nn.Module): Activation layer to use.
        inner_ratio (int): Ratio for the inner dimension compared to the input dimension.
        """
        super().__init__()
        self.update_coords = update_coords
        values_inner_dim = int(values_dim * inner_ratio)
        coords_inner_dim = int(coords_dim * inner_ratio)

        # Network for values
        self.values_net = nn.Sequential(
            nn.Linear(values_dim, values_inner_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(values_inner_dim, values_dim),
            nn.Dropout(dropout),
        )

        # Network for coordinates
        if update_coords:
            self.coords_net = nn.Sequential(
                nn.Linear(coords_dim, coords_inner_dim),
                act_layer(),
                nn.Dropout(dropout),
                nn.Linear(coords_inner_dim, coords_dim),
                nn.Dropout(dropout),
            )
        else:
            self.coords_net = nn.Identity()

    def forward(self, x, **kwargs):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys "values" and "coords".
        Returns:
            dict of str: dict: Dictionary of outputs, such that
                outputs[(source_name, index)] contains the keys "values" and "coords" with
                the updated values and coordinates.
        """
        outputs = {}
        for source_name, data in x.items():
            values = data["values"]
            coords = data["coords"]

            # Process values and coordinates through their respective networks
            updated_values = self.values_net(values)
            updated_coords = self.coords_net(coords)

            # Return both updated values and coordinates
            outputs[source_name] = {"values": updated_values, "coords": updated_coords}
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
