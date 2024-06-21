"""Implements the MultiSourceVIT class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_sources.models.vit_classes import PatchEmbedding, ResNet
from multi_sources.models.custom_vit import CoordinatesTransformer
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
        x_dim,
        c_dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        output_resnet_depth=3,
        output_resnet_inner_channels=8,
        output_resnet_kernel_size=3,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        """
        Args:
            img_sizes (dict of str to int or tuple of int): Size of the images for each source.
            channels (dict of str to int): Number of channels in the values (V) for each source.
            patch_size (int or tuple of int): Size of the patches to be extracted from the image.
            x_dim (int): Dimension of the internal embeddings for the pixel values.
            c_dim (int): Dimension of the internal embeddings for the coordinates.
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
        self.x_dim = x_dim
        self.c_dim = c_dim
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
                    patch_size[0], patch_size[1], channels[source_name], x_dim
                )
                for source_name in channels
            }
        )
        # Create an embedding for the land mask for each source.
        self.land_mask_embedding = PatchEmbedding(patch_size[0], patch_size[1], 1, x_dim)
        # Create an embedding for the coordinates (lat/lon) for each source.
        self.coord_embedding = PatchEmbedding(patch_size[0], patch_size[1], 2, c_dim)

        # We'll also append the time delta and the source index, which are scalar values, as
        # well as the availability flag.
        self.x_dim = self.x_dim + 3  # Value emb. + [time delta, source_index, avail.]

        # Create the transformer layers.
        self.transformer = CoordinatesTransformer(
            self.x_dim, self.c_dim, depth, heads, dim_head, mlp_dim, dropout
        )

        # Create a final linear layer to go from the transformer dim to the original patch dim.
        self.output_projections = nn.ModuleDict(
            {
                source_name: nn.Sequential(
                    nn.LayerNorm(self.x_dim), nn.Linear(self.x_dim, self.patch_dim[source_name])
                )
                for source_name in channels
            }
        )
        # Create a ResNet module for each source at the end to correct artifacts at the borders
        # between patches.
        if output_resnet_depth > 0:
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
        else:
            self.output_resnets = nn.ModuleDict(
                {source_name: nn.Identity() for source_name in channels}
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
        land_mask_embed = {
            source_name: self.land_mask_embedding(C[:, 2:3])
            for source_name, C in padded_coords.items()
        }
        # For each source, compute the coordinate embeddings.
        coords_embed = {
            source_name: self.coord_embedding(C[:, :2])
            for source_name, C in padded_coords.items()
        }
        # Stack the source indices, the time delta and the availability flag.
        info_tensor = {
            source_name: torch.stack([a, s, dt], dim=-1)
            .unsqueeze(1)
            .expand(-1, self.num_patches[source_name], -1)
            for source_name, (a, s, dt, _, _, _) in inputs.items()
        }
        # Sum the embeddings with the land mask embeddings.
        embeddings = {
            source_name: emb + mask
            for source_name, emb, mask in zip(
                embeddings.keys(), embeddings.values(), land_mask_embed.values()
            )
        }
        # Concatenate the embeddings with the info tensor for each source
        embeddings = {
            source_name: torch.cat([emb, info_tensor[source_name]], dim=-1)
            for source_name, emb in embeddings.items()
        }
        # Concatenate the embeddings for all sources to form the sequence.
        sequence = torch.cat([emb for emb in embeddings.values()], dim=1)
        # Concatenate the coordinates embeddings for all sources to form the sequence.
        coords_sequence = torch.cat([emb for emb in coords_embed.values()], dim=1)
        # Apply the transformer.
        transformer_output = self.transformer(sequence, coords_sequence)
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
                "b (h w) (c pw ph) -> b c (h ph) (w pw)",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
                c=self.channels[source_name],
                h=self.img_sizes[source_name][0] // self.patch_size[0],
            )
            for source_name, otp in output.items()
        }
        # Apply the ResNet to correct artifacts at the borders between patches.
        output = {
            source_name: self.output_resnets[source_name](otp)
            for source_name, otp in output.items()
        }
        # For each source, remove the padding.
        output = {
            source_name: otp[:, :, :H, :W]
            for source_name, otp, (H, W) in zip(
                output.keys(), output.values(), self.original_img_sizes.values()
            )
        }
        # Restore the original keys.
        output = restore_dots_in_keys(output)
        return output
