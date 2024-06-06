"""Implements the MultiSourceVIT class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_sources.models.vit_classes import PatchEmbedding, Transformer, ResNet
from einops import rearrange


def pair(x):
    """Ensures that x is a pair of integers."""
    if isinstance(x, tuple):
        return x
    return (x, x)


def remove_dots_in_keys(d):
    """Replaces the character '.' in the keys of a dict with '`'."""
    return {k.replace(".", "`"): v for k, v in d.items()}


def restore_dots_in_keys(d):
    """Replaces the character '`' in the keys of a dict with '.'."""
    return {k.replace("`", "."): v for k, v in d.items()}


class MultiSourceVIT(nn.Module):
    """Uses a ViT backbone to reconstruct images from multiple sources, with a subset
    of them being masked out.
    Specifically, the model takes inputs a Mapping from source name
    to a tuple (A, S, DT, C, V) where:
    - A is a tensor of shape (batch_size, 1) whose value is 1 if the source is present,
        0 if it is missing, and -1 if it is masked.
    - S is a tensor of shape (batch_size) containing the source indices.
    - DT is a tensor of shape (batch_size) containing the time delta.
    - C is a tensor of shape (batch_size, 3, H, W) containing the (lat, lon, land_mask) info
        at each pixel.
    - V is a tensor of shape (batch_size, K, H, W) containing the values to be reconstructed.
    For one or multiple sources, DT, C and V can be masked out.
    """

    def __init__(
        self,
        img_sizes,
        channels,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        output_resnet_depth=3,
        output_resnet_inner_channels=16,
        output_resnet_kernel_size=3,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        """
        Args:
            img_sizes (dict of str to int or tuple of int): Size of the images for each source.
            channels (dict of str to int): Number of channels in the values (V) for each source.
            patch_size (int or tuple of int): Size of the patches to be extracted from the image.
            dim (int): Dimension of the model.
            depth (int): Number of layers.
            heads (int): Number of heads for the attention mechanism.
            mlp_dim (int): Dimension of the feedforward layers.
            dim_head (int): Dimension of each head.
            output_resnet_depth (int): Depth of the output ResNet.
                set to 0 to disable any convolutional layer at the output.
            output_resnet_inner_channels (int): Number of channels in the inner layers
                of the output
            output_resnet_kernel_size (int): Kernel size of the output ResNet.
            dropout (float): Dropout rate.
            emb_dropout (float): Dropout rate for the embeddings.
        """
        super().__init__()
        # ModuleDict objects keys can't contain the character '.'; so replace it with '_'
        # in the keys of img_sizes and channels.
        img_sizes = remove_dots_in_keys(img_sizes)
        channels = remove_dots_in_keys(channels)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        patch_size = pair(patch_size)
        self.patch_size = patch_size
        # Store the original image sizes for later reconstruction.
        self.original_img_sizes = {
            source_name: pair(img_size) for source_name, img_size in img_sizes.items()
        }
        # For each source, compute the next multiple of patch_size (for both height and width).
        # This is the size to which the image will be padded.
        self.img_sizes = {}
        for source_name, (H, W) in self.original_img_sizes.items():
            self.img_sizes[source_name] = (
                H + patch_size[0] - H % patch_size[0],
                W + patch_size[1] - W % patch_size[1],
            )
        # Deduce the total number of patches across all sources.
        self.num_patches = {
            source_name: (H // patch_size[0]) * (W // patch_size[1])
            for source_name, (H, W) in self.img_sizes.items()
        }
        self.total_num_patches = sum(self.num_patches.values())
        # Compute the size of the flattened patch for each source.
        self.patch_dim = {
            source_name: channels[source_name] * patch_size[0] * patch_size[1]
            for source_name in channels
        }
        # Create a patch embedding for each source.
        self.patch_embeddings = nn.ModuleDict(
            {
                source_name: PatchEmbedding(
                    patch_size[0], patch_size[1], channels[source_name], dim
                )
                for source_name in channels
            }
        )
        # Create an embedding for the land mask, which will be summed to the patch embeddings.
        self.land_mask_embeddings = nn.ModuleDict(
            {
                source_name: PatchEmbedding(
                    patch_size[0], patch_size[1], 1, dim
                )  # 1 channel for the land mask
                for source_name in channels
            }
        )
        # Instead of a traditional positional embedding, we'all append the latitude and longitude
        # coordinates to the patch embeddings. The idea is to allow each patch to understand
        # which patches in the other sources correspond to the same approximate location.
        # Note: while C gives the lat/lon for each pixel, we'll only use the coordinates
        # of the four corners of each patch, to limit the additional patch dimension.
        # We'll also append the time delta and the source index, which are scalar values.
        self.dim += 10  # 4 corners * 2 coordinates + time delta + source_index

        # Create the transformer layers.
        self.transformer = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout)

        # Create a final linear layer to go from the transformer dim to the original patch dim.
        self.output_projections = nn.ModuleDict(
            {
                source_name: nn.Sequential(
                    nn.LayerNorm(self.dim), nn.Linear(self.dim, self.patch_dim[source_name])
                )
                for source_name in channels
            }
        )
        # Create a ResNet module for each source at the end to correct artifacts at the borders
        # between patches.
        self.output_resnets = nn.ModuleDict(
            {
                source_name: ResNet(
                    self.channels[source_name],
                    output_resnet_inner_channels,
                    output_resnet_depth,
                    output_resnet_kernel_size,
                )
                for source_name in channels
            }
        )

    def forward(self, inputs):
        """
        Args:
            inputs (dict of str to tuple of tensors): Inputs to the model.
        Returns:
            output (dict of str to tensor): Reconstructed values for each source, as a dict
                {source_name: tensor of shape (batch_size, K, H, W)}.
        """
        inputs = remove_dots_in_keys(inputs)
        # For each source, retrieve the values tensor V and pad it to the size stored in img_sizes.
        # Also pad the coordinates tensor C to the same size.
        padded_values, padded_coords = {}, {}
        for source_name, (H, W) in self.img_sizes.items():
            A, S, DT, C, D, V = inputs[source_name]
            padded_values[source_name] = F.pad(V, (0, W - V.shape[-1], 0, H - V.shape[-2]))
            padded_coords[source_name] = F.pad(C, (0, W - C.shape[-1], 0, H - C.shape[-2]))
        # For each source, compute the patch embeddings.
        embeddings = {
            source_name: self.patch_embeddings[source_name](V)
            for source_name, V in padded_values.items()
        }
        # For each source, compute the land mask embeddings.
        land_mask_embeddings = {
            source_name: self.land_mask_embeddings[source_name](C[:, 2:3])
            for source_name, C in padded_coords.items()
        }
        # Add the land mask embeddings to the patch embeddings.
        embeddings = {
            source_name: emb + land_emb
            for (source_name, emb), land_emb in zip(
                embeddings.items(), land_mask_embeddings.values()
            )
        }
        # For each source, retrieve the coordinates of the corners of each patch.
        coords = {}
        for source_name, C in padded_coords.items():
            lat, lon = C[:, 0], C[:, 1]
            # Rearrange each from (bs, H, W) to (bs, n_patch, patch_h, patch_w).
            lat = rearrange(
                lat,
                "bs (h ph) (w pw) -> bs (h w) ph pw",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
            )
            lon = rearrange(
                lon,
                "bs (h ph) (w pw) -> bs (h w) ph pw",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
            )
            # Fetch the coordinates of the four corners of each patch.
            coords[source_name] = torch.stack(
                [
                    lat[:, :, 0, 0],  # top left
                    lat[:, :, 0, -1],  # top right
                    lat[:, :, -1, 0],  # bottom left
                    lat[:, :, -1, -1],  # bottom right
                    lon[:, :, 0, 0],  # top left
                    lon[:, :, 0, -1],  # top right
                    lon[:, :, -1, 0],  # bottom left
                    lon[:, :, -1, -1],  # bottom right
                ],
                dim=-1,
            )
        # Stack the time delta and the source indices
        dt_and_source = {
            source_name: torch.stack([s, dt], dim=-1)
            .unsqueeze(1)
            .expand(-1, self.num_patches[source_name], -1)
            for source_name, (_, s, dt, _, _, _) in inputs.items()
        }
        # Concatenate the embeddings with the coordinates and the time delta and source indices.
        embeddings = {
            source_name: torch.cat([emb, coords[source_name], dt_and_source[source_name]], dim=-1)
            for source_name, emb in embeddings.items()
        }
        # Concatenate the embeddings for all sources to form the sequence.
        sequence = torch.cat([emb for emb in embeddings.values()], dim=1)
        # Apply the transformer layers.
        transformer_output = self.transformer(sequence)
        # Split the output back to the sources.
        output, cnt = {}, 0
        for source_name, num_patches in self.num_patches.items():
            output[source_name] = transformer_output[:, cnt: cnt + num_patches]
            cnt += num_patches
        # Apply the final projection layer to go back to the original patch dim
        output = {
            source_name: self.output_projections[source_name](otp).view(
                otp.shape[0], otp.shape[1], -1
            )
            for source_name, otp in output.items()
        }
        # For each source, reshape the output from (bs, n_patch, dim) to (bs, K, H, W).
        output = {
            source_name: rearrange(
                otp,
                "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
                c=self.channels[source_name],
                h=self.img_sizes[source_name][0] // self.patch_size[0],
            )
            for source_name, otp in output.items()
        }
        # For each source, remove the padding.
        output = {
            source_name: otp[:, :, :H, :W]
            for source_name, otp, (H, W) in zip(
                output.keys(), output.values(), self.original_img_sizes.values()
            )
        }
        # For each source, apply the output ResNet.
        output = {
            source_name: self.output_resnets[source_name](otp)
            for source_name, otp in output.items()
        }
        # Restore the original keys.
        output = restore_dots_in_keys(output)
        return output
