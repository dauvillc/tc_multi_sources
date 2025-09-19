"""Implements layers to project the output of a ViT in latent space to the output space."""

from torch import nn

from multi_sources.models.conv import ResNet
from multi_sources.models.icnr import ICNR


class SourcetypeProjection2d(nn.Module):
    """Receives the embedded values and conditioning of a source and projects them to
    that source's original space. Meant to be shared across all sources of the same type.
    """

    def __init__(
        self,
        values_dim,
        out_channels,
        patch_size,
        use_modulation=False,
        resnet_channels=None,
        resnet_blocks=None,
    ):
        """
        Args:
            values_dim (int): Dimension of the values embeddings.
            out_channels (int): Number of channels in the output space.
            patch_size (int): Size of the embedding patches.
            use_modulation (bool): If True, applies modulation to the values embeddings.
            resnet_channels (int): Number of channels in the output ResNet.
            resnet_blocks (int): Number of blocks in the output ResNet.
        """
        super().__init__()
        self.patch_size = patch_size

        self.norm = nn.LayerNorm(values_dim)
        if use_modulation:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, 2 * values_dim))
        # Subpixel convolution to project the latent space to the output space
        self.conv = nn.Conv2d(
            in_channels=values_dim,
            out_channels=out_channels * patch_size**2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        # Apply the ICNR initialization to the deconvolution, to reduce checkerboard artifacts
        weight = ICNR(
            self.conv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=patch_size
        )
        self.conv.weight.data.copy_(weight)

        # Final ResNet to correct the artifacts
        if resnet_channels is None or resnet_blocks is None:
            self.resnet = nn.Identity()
        else:
            self.resnet = ResNet(out_channels, resnet_channels, resnet_blocks)

    def forward(self, values, cond, **unused_kwargs):
        """
        Args:
            values (torch.Tensor): Embedded values of shape (B, h, w, Dv).
            cond (torch.Tensor): Embedded conditioning of shape (B, h, w, Dv).
        Returns:
            torch.Tensor of shape (B, channels, H, W) containing the projected output.
        """
        if hasattr(self, "modulation"):
            # Apply the modulation to the values embeddings
            shift, scale = self.modulation(cond).chunk(2, dim=-1)
            v = (1 + scale) * self.norm(values) + shift
        else:
            # Normalize the values embeddings
            v = self.norm(values)
        # Transpose x from (B, h, w, Dv) to (B, Dv, h, w)
        v = v.permute(0, 3, 1, 2)
        # Deconvolve the latent space using subpixel convolutions
        v = self.conv(v)
        v = self.pixel_shuffle(v)  # (B, C, H, W)

        # Correct the artifacts with a ResNet
        v = self.resnet(v)
        return v


class SourcetypeProjection0d(nn.Module):
    """Receives the embedded values and conditioning of a 0D source and projects them to
    that source's original space. Meant to be shared across all sources of the same type."""

    def __init__(self, values_dim, out_channels, use_modulation=False, **unused_kwargs):
        """Args:
        values_dim (int): Dimension of the values embeddings.
        out_channels (int): Number of channels in the output space.
        use_modulation (bool): If True, applies modulation to the values embeddings.
        **unused_kwargs: Unused arguments.
        """
        super().__init__()
        self.norm = nn.LayerNorm(values_dim)
        if use_modulation:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, 2 * values_dim))
        else:
            self.modulation = nn.Identity()
        self.output_proj = nn.Linear(values_dim, out_channels)

    def forward(self, values, cond, **unused_kwargs):
        """Args:
            values (torch.Tensor): Embedded values of shape (B, D).
            cond (torch.Tensor): Embedded conditioning of shape (B, D).
        Returns:
            torch.Tensor of shape (B, C) containing the projected output.
        """
        # Apply the modulation to the values embeddings
        v = values
        if hasattr(self, "modulation"):
            shift, scale = self.modulation(cond).chunk(2, dim=-1)
            v = (1 + scale) * self.norm(values) + shift
        else:
            # Normalize the values embeddings
            v = self.norm(values)
        # Project to output space
        v = self.output_proj(v)
        return v
