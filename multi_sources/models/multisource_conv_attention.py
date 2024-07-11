"""Implements the Multi-source Convolutional Attention module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from multi_sources.models.utils import pad_to_next_multiple_of, pair
from multi_sources.models.mbconv import MBConv
from multi_sources.models.small_layers import NonLinearEmbedding


class MultisourceConvAttention(nn.Module):
    """This module receives two lists of images (keys-values and queries), and
    performs a multi-source convolutional attention operation.
    """

    def __init__(
        self,
        n_keys,
        n_queries,
        keys_channels,
        values_channels,
        queries_channels,
        patch_size,
        attention_dim,
        num_heads,
        kernel_size=7,
    ):
        """
        Args:
            n_keys (int): Number of keys, which is also the number of values.
            n_queries (int): Number of queries.
            keys_channels (int): Number of channels in the keys.
            values_channels (int): Number of channels in the values.
            queries_channels (int): Number of channels in the queries.
            patch_size (int or tuple of ints): Size of the patches.
            attention_dim (int): Dimension of the embeddings used in the attention mechanism.
            num_heads (int): Number of heads in the attention mechanism.
                Must be a divisor of attention_dim and values_channels.
            kernel_size (int or tuple of ints): Size of the convolutional kernel.
        """
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("The attention dimension must be divisible by the number of heads.")
        if values_channels % num_heads != 0:
            raise ValueError("The values channels must be divisible by the number of heads.")
        self.n_keys, self.n_values = n_keys, n_keys
        self.keys_channels, self.queries_channels = keys_channels, queries_channels
        self.values_channels = values_channels
        self.patch_size = pair(patch_size)
        ph, pw = self.patch_size
        self.num_heads = num_heads
        self.dim_head = attention_dim // num_heads
        self.kernel_size = pair(kernel_size)
        kh, kw = self.kernel_size
        # Compute the size of the flattened patch for each source.
        self.keys_patch_dim = (ph * pw * keys_channels)
        self.values_patch_dim = (ph * pw * values_channels)
        self.queries_patch_dim = (ph * pw * queries_channels)
        # Create a layernorm for the values
        self.values_ln = nn.LayerNorm(self.values_patch_dim)
        # Create an embedding for the keys and queries.
        self.key_embedding = NonLinearEmbedding(self.keys_patch_dim, attention_dim)
        self.query_embedding = NonLinearEmbedding(self.queries_patch_dim, attention_dim)
        # Rearrange layer to go from images to patches.
        self.to_patches = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)
        # Rearrange layer to add a heads dimension to a sequence.
        self.to_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        # MBConv block to process the output of the attention mechanism.
        # The mbconv block is modified to use grouped convolutions, so that the
        # heads are processed in parallel and separately.
        self.mbconv = MBConv(
            self.values_channels * self.n_keys,
            self.values_channels,
            downsample=False,
            groups=self.num_heads,
        )

    def forward(self, keys, values, queries):
        """
        Args:
            keys (list of torch.Tensor): List of keys images of shape (B, Ck, H, W).
                H and W may vary, while Ck must be the same for all keys.
            values (list of torch.Tensor): List of values images of shape (B, Cv, H, W).
                H and W may vary, while Cv must be the same for all values.
                There must be one value for each key, and those two must have the same size.
            queries (list of torch.Tensor): List of query images of shape (B, Cq, H, W).
                H and W may vary, while Cq must be the same for all queries.
        Returns:
            list of torch.Tensor: List of output images of shape (B, Cv, H, W).
                H and W will have the same shape as their corresponding queries.
        """
        if len(keys) != len(values):
            raise ValueError("The number of keys and values must be the same.")
        for key, value in zip(keys, values):
            if key.shape[-2:] != value.shape[-2:]:
                raise ValueError("The keys and values must have the same spatial size.")
        # Store the original sizes of the queries.
        original_query_sizes = [query.shape[-2:] for query in queries]
        # Pad the queries and keys to the next multiple of the patch size.
        ph, pw = self.patch_size
        keys = [pad_to_next_multiple_of(key, (ph, pw)) for key in keys]
        values = [pad_to_next_multiple_of(value, (ph, pw)) for value in values]
        queries = [pad_to_next_multiple_of(query, (ph, pw)) for query in queries]
        # Store the spatial sizes of the queries after padding.
        query_sizes = [query.shape[-2:] for query in queries]
        # Patchify the keys, queries and values.
        keys_patches = [self.to_patches(key) for key in keys]
        values_patches = [self.to_patches(value) for value in values]
        queries_patches = [self.to_patches(query) for query in queries]
        # Store the number of patches for each source.
        n_keys_patches = [key_patches.size(1) for key_patches in keys_patches]
        n_queries_patches = [query_patches.size(1) for query_patches in queries_patches]
        # Form the sequences of patches.
        keys_seq = torch.cat(keys_patches, dim=1)  # (b, n_keys_seq, k_patch_dim)
        values_seq = torch.cat(values_patches, dim=1)  # (b, n_keys_seq, v_patch_dim)
        queries_seq = torch.cat(queries_patches, dim=1)  # (b, n_queries_seq, q_patch_dim)
        # Apply layer normalization
        values_seq = self.values_ln(values_seq)
        # Embed the keys and queries.
        keys_emb = self.key_embedding(keys_seq)  # (b, hd, n_keys_seq, attention_dim)
        queries_emb = self.query_embedding(queries_seq)  # (b, hd, n_queries_seq, attention_dim)
        # Reshape to add the heads dimension.
        keys_emb = self.to_heads(keys_emb)  # (b, hd, n_keys_seq, attention_dim)
        values_seq = self.to_heads(values_seq)  # (b, hd, n_keys_seq, v_patch_dim)
        queries_emb = self.to_heads(queries_emb)  # (b, hd, n_queries_seq, attention_dim)
        # Compute the attention weights.
        attn = torch.einsum("b h i d, b h j d -> b h i j", queries_emb, keys_emb)
        attn = F.softmax(attn / (self.dim_head**0.5), dim=-1)  # (b, hd, n_queries_seq, n_keys_seq)
        # Split the columns of the attention matrix by key image.
        attn_grouped_keys = torch.split(attn, n_keys_patches, dim=3)
        # Compute the weighted sum of the values with each group of keys.
        values = torch.split(values_seq, n_keys_patches, dim=2)
        weighted_v = [torch.matmul(a, v) for a, v in zip(attn_grouped_keys, values)]
        # weighted_v is a list of (b, hd, n_queries_seq, v_patch_dim) tensors.
        # Stack back the weighted values
        weighted_v = torch.stack(weighted_v, dim=1)  # (n, n_keys, hd, n_queries_seq, v_patch_dim)
        # Split the weighted values by query image.
        weighted_v_grouped_queries = torch.split(weighted_v, n_queries_patches, dim=3)
        # Reshape each group of weighted values into an image of the size of the query.
        weighted_v_grouped_queries = [
            rearrange(
                v,
                "b k hd (h w) (ph pw c) -> b (hd k c) (h ph) (w pw)",
                h=qh // ph,
                w=qw // pw,
                ph=ph,
                pw=pw,
            )
            for v, (qh, qw) in zip(weighted_v_grouped_queries, query_sizes)
        ]  # list of (b, (hd k c), H, W) tensors
        # Apply the MBConv block to each group of weighted values.
        outputs = [self.mbconv(v) for v in weighted_v_grouped_queries]
        # Remove the padding from the outputs.
        outputs = [
            output[..., :h, :w] for output, (h, w) in zip(outputs, original_query_sizes)
        ]
        return outputs


class MultisourceConvAttBlock(nn.Module):
    """Wrapper class that integrates the Multi-source Convolutional Attention module
    into the encoder-decoder transformer.
    """
    def __init__(self, n_keys, n_queries, channels, use_coordinates_attention=False, **kwargs):
        """
        Args:
            n_keys (int): Number of key sources.
            n_queries (int): Number of query sources.
            channels (int): Number of channels in the images.
            use_coordinates_attention (bool): Whether to use the coordinates attention.
                If True, the attention map is computed using the coordinates of the patches
                (latitude, longitude), instead of their content (pixels).
                The values are always the pixels.
            **kwargs: Additional arguments to pass to the MultisourceConvAttention constructor.
        """
        super().__init__()
        self.use_coordinates_attention = use_coordinates_attention
        if use_coordinates_attention:
            key_channels, query_channels = 2, 2
        else:
            key_channels, query_channels = channels, channels
        self.attention = MultisourceConvAttention(
            n_keys=n_keys,
            n_queries=n_queries,
            keys_channels=key_channels,
            values_channels=channels,
            queries_channels=query_channels,
            **kwargs,
        )

    def forward(self, key_pixels, query_pixels, key_coords, query_coords):
        """
        Args:
            key_pixels (list of torch.Tensor): List of key images of shape (B, Ck H, W).
            query_pixels (list of torch.Tensor): List of query images of shape (B, Cq, H, W).
            key_coords (list of torch.Tensor): List of key coordinates of shape (B, 2, H, W).
            query_coords (list of torch.Tensor): List of query coordinates of shape (B, 2, H, W).
        Returns:
            list of torch.Tensor: List of output images of shape (B, Ck, H, W).
        """
        values = key_pixels
        if self.use_coordinates_attention:
            keys = key_coords
            queries = query_coords
        else:
            keys = key_pixels
            queries = query_pixels
        return self.attention(keys, values, queries)
