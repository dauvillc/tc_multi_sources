"""Implements subclasses for the RegriddingVit class."""

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from multi_sources.models.utils import pad_to_next_multiple_of


class VectorEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.layers(x).unsqueeze(1)  # Add a dummy dimension for the patches


class CoordinatesEmbedding(nn.Module):
    def __init__(self, patch_h, patch_w, dim):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.dim = dim
        self.patch_dim = 2 * patch_h * patch_w
        self.embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_h, p2=patch_w),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.embedding(x)


class CoordinatesAttention(nn.Module):
    """Computes attention weights based on the spatial coordinates of the input data."""

    def __init__(self, c_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5

        self.c_norm = nn.LayerNorm(c_dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(c_dim, inner_dim * 2, bias=False)

    def forward(self, coords):
        """
        Args:
            coords: (b, n, c) tensor, embedded spatio-temporal coordinates.
        Returns:
            attn: (b, h, n, n) tensor
        """
        coords = self.c_norm(coords)

        q, k = self.to_qk(coords).chunk(2, dim=-1)  # (b, n, 2 * dim_head)
        q, k = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k))

        dots = torch.matmul(q, k.transpose(-1, -2))  # (b, h, k, n)

        attn = self.attend(dots * self.scale)
        return attn


class EntryConvBlock(nn.Module):
    """Entry block of the RegriddingVit model."""

    def __init__(self, in_channels, out_channels, downsample):
        """
        Args:
            in_channels (list of int): number of channels of the input images;
            out_channels (int): number of channels of the output images;
            downsample (bool): whether to downsample the images by a factor of 2.
        """
        super().__init__()
        self.downsample = downsample
        self.layers = nn.ModuleList()
        for c in in_channels:
            if downsample:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(c, out_channels, kernel_size=3, stride=2, padding=1),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(c, out_channels, kernel_size=1),
                    )
                )

    def forward(self, x):
        """
        Args:
            x (list of torch.Tensor): list of input images.
        Returns:
            y (list of torch.Tensor): list of output images.
        """
        y = [layer(img) for layer, img in zip(self.layers, x)]
        return y


class AttentionRegriddingBlock(nn.Module):
    """Regridding attention mechanism:
    Given a a set of images from different sources and their corresponding coordinates,
    - divide the images into patches;
    - embed the patches and the coordinates;
    - compute attention weights based on the coordinates;
    - selects a common number of patches from each image;
    - reconstruct the images based on the selected patches.
    """

    def __init__(
        self,
        n_sources,
        channels,
        inner_channels,
        output_channels,
        patch_size,
        context_size,
        c_dim,
        heads,
        dim_head,
    ):
        """
        Args:
            n_sources (int): number of sources;
            channels (int): number of channels in each image; Must be the same for all sources.
            inner_channels (int): number of channels of the intermediate convolutions;
            output_channels (int): number of channels in the output images;
            patch_size (int or tuple of int): size of the patches;
                Must divide the image sizes;
            context_size (int): size of the contextual vector (input vector
                that gives additional information about each source).
            c_dim (int): dimension of the coordinates embeddings;
            heads (int): number of attention heads;
            dim_head (int): dimension of the attention heads.
        """
        super().__init__()
        self.n_sources = n_sources
        self.channels = channels
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.context_size = context_size
        ph, pw = self.patch_size
        self.c_dim = c_dim
        self.heads = heads
        self.dim_head = dim_head
        # Compute the size of the flattened patch for each source.
        self.patch_dim = ph * pw * channels
        # Create an embedding for the coordinates (lat/lon).
        self.coord_embedding = CoordinatesEmbedding(ph, pw, c_dim)
        # Create a patchening layer for the images.
        self.image_to_patches = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw)
        # Create an embedding for the context vector.
        # which will be summed to the coordinates embeddings.
        self.context_embedding = VectorEmbedding(context_size, c_dim)
        # Create an attention layer
        self.attention = CoordinatesAttention(c_dim, heads, dim_head)
        # Create a 3D Conv block that will be applied to the reconstructed images
        # Using 3D convs with a kernel size of 1 over the depth dimension allows
        # to apply 2D convs with different heads in parallel.
        input_channels = channels * n_sources
        self.conv_heads = nn.Sequential(
            nn.BatchNorm3d(input_channels),
            nn.Conv3d(input_channels, inner_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.BatchNorm3d(inner_channels),
            nn.Conv3d(inner_channels, output_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

    def forward(self, x, coords, context):
        """
        Args:
            x (list of torch.Tensor): list of images from different sources;
            coords (list of torch.Tensor): list of coordinates for each source.
                Each element should be a tensor of shape (B, 2, H, W) (the channels
                being the latitude and longitude).
            context (list of torch.Tensor): list of context vectors for each source.
        Returns:
            y (list of torch.Tensor): list of output images,
                of shape (B, output_channels, H, W).
        """
        # Store the sizes of the images.
        original_img_sizes = [img.shape[-2:] for img in x]
        # Pad the images and coords so that their sizes are multiples of the patch size.
        x = [pad_to_next_multiple_of(img, self.patch_size) for img in x]
        coords = [pad_to_next_multiple_of(coord, self.patch_size) for coord in coords]
        # Store the sizes of the padded images.
        img_sizes = [img.shape[-2:] for img in x]
        # Compute the number of patches for each source.
        num_patches = [H * W // (self.patch_size[0] * self.patch_size[1]) for H, W in img_sizes]
        # Patch the images.
        patches = [self.image_to_patches(img) for img in x]
        # Embed the coordinates into tokens.
        coords = [self.coord_embedding(c) for c in coords]
        # Embed the context vectors into tokens, and sum them to the coordinates.
        context = [self.context_embedding(c) for c in context]
        coords = [c + ctx for c, ctx in zip(coords, context)]
        # Concatenate the embedded coordinates and the patches along the patch dimension, to form
        # the sequences.
        coords = torch.cat(coords, dim=1)  # (B, N, dim_c)
        # Compute the attention weights.
        attn = self.attention(coords)  # (B, head, N, N)
        # Split the columns into the different sources.
        attn_blocks = torch.split(attn, num_patches, dim=-1)  # list of (B, head, N, n_source)
        # For each row of the attention matrix, compute the weighted sum of the patches.
        patches = [
            torch.matmul(attn_block, patch.unsqueeze(1))
            for attn_block, patch in zip(attn_blocks, patches)
        ]
        # At this point, patches[i] has shape (B, head, N, patch_dim).
        # patches[i][., ., j] is the sum of the patches of source i weighted by the attention that
        # the jth patch gives to each of the patches of source i.
        # We'll call that weighted sum the "selected patch".
        patches = torch.stack(patches, dim=2)  # (B, head, N, n_source, patch_dim)
        patches = torch.split(patches, num_patches, dim=3)
        # list of (B, head, n_source, K_source, patch_dim) K_source = number of patches of source k
        # Now, patches[i][., ., k, j] is the selected patch of source i
        # for the jth patch of source k.
        # Reshape each of them into (B, C, head, H, W).
        images = [
            rearrange(
                patch,
                "b hd s (h w) (ph pw c) -> b (c s) hd (h ph) (w pw)",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
                h=H // self.patch_size[0],
                w=W // self.patch_size[1],
            )
            for patch, (H, W) in zip(patches, img_sizes)
        ]
        # Apply the 3D convolutions
        images = [self.conv_heads(img) for img in images]
        # list of (B, output_channels, hd, H, W)
        # Sum over the heads.
        images = [img.sum(dim=2) for img in images]  # list of (B, output_channels, H, W)
        # Remove the padding
        images = [img[..., :H, :W] for img, (H, W) in zip(images, original_img_sizes)]
        return images
