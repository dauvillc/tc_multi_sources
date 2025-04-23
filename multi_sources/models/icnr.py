"""Taken from https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
Thanks to A03ki for the implementation.
"""

import torch


def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {tensor.shape[0]}"
        " is not divisible by square of upscale_factor: "
        f"upscale_factor = {upscale_factor}"
    )
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)
