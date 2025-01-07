import torch.nn.functional as F
from einops import rearrange


def pair(x):
    """Ensures that x is a pair of integers."""
    if isinstance(x, tuple):
        return x
    return (x, x)


def img_to_patches(img, patch_size):
    """Converts an image tensor to patches.
    Args:
        img (torch.Tensor): Image tensor of shape (B, C, H, W).
        patch_size (int or tuple of int): Size of the patch.
    Returns:
        torch.Tensor: Patch tensor of shape (B, n_tokens, dim).
    """
    patch_H, patch_W = pair(patch_size)
    img = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_H, p2=patch_W)
    return img


def pad_to_next_multiple_of(tensor, multiple_of, **kwargs):
    """Pad an image or a batch of images to the next multiple of a number.
    Args:
        tensor (torch.Tensor): If of shape (..., H, W), will be padded to
            (..., H_padded, W_padded). Otherwise, skips the padding.
        multiple_of (int or tuple of int): if int, the number to which the
            height and width should be padded. If tuple, the first element
            determines the height padding and the second element the width
            padding.
        kwargs: additional arguments to the F.pad function.
    Returns:
        torch.Tensor: padded tensor of shape (..., H_padded, W_padded).
    """
    # Skip padding if the tensor is not an image.
    if tensor.ndim < 4:
        return tensor
    H, W = tensor.shape[-2:]
    mo_h, mo_w = pair(multiple_of)
    H_padded = H + (-H) % mo_h
    W_padded = W + (-W) % mo_w
    padding = (0, W_padded - W, 0, H_padded - H)
    return F.pad(tensor, padding, **kwargs)


def patches_to_img(patches, H, W, patch_size):
    """Converts patches to an image tensor.
    Args:
        patches (torch.Tensor): Patch tensor of shape (B, n_tokens, dim).
        H (int): Original height of the image.
        W (int): Original width of the image.
        patch_size (int or tuple of int): Size of the patch.
    Returns:
        torch.Tensor: Image tensor of shape (B, C, H, W).
    """
    img = rearrange(
        patches,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=H // patch_size[0],
        w=W // patch_size[1],
        p1=patch_size[0],
        p2=patch_size[1],
    )
    return img


def remove_padding(img, H, W):
    """Removes padding from an image or a batch of images.
    Args:
        img (torch.Tensor): tensor of shape (..., H_padded, W_padded).
        H (int): Original height of the image.
        W (int): Original width of the image.
    Returns:
        torch.Tensor: tensor of shape (..., H, W).
    """
    return img[..., :H, :W]
