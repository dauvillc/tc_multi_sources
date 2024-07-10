"""Implements the Multi-source Convolutional Attention module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from multi_sources.models.utils import pad_to_next_multiple_of, pair
from multi_sources.models.mbconv import MBConv


class MultisourceConvAttention(nn.Module):
    """This module receives two lists of images (keys-values and queries), and
    performs a multi-source convolutional attention operation.
    """

    def __init__(
        self,
        n_keys,
        n_queries,
        channels,
        patch_size,
        attention_dim,
        num_heads,
        kernel_size=7,
    ):
        """
        Args:
            n_keys (int): Number of keys.
            n_queries (int): Number of queries.
            channels (int): Number of input and output channels.
            patch_size (int or tuple of ints): Size of the patches.
            attention_dim (int): Dimension of the embeddings used in the attention mechanism.
            num_heads (int): Number of heads in the attention mechanism.
                Must be a divisor of attention_dim and of channels.
            kernel_size (int or tuple of ints): Size of the convolutional kernel.
        """
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("The attention dimension must be divisible by the number of heads.")
        if channels % num_heads != 0:
            raise ValueError("The channels must be divisible by the number of heads.")
        self.n_keys = n_keys
        self.channels, self.channels = channels, channels
        self.patch_size = pair(patch_size)
        ph, pw = self.patch_size
        self.num_heads = num_heads
        self.dim_head = attention_dim // num_heads
        self.kernel_size = pair(kernel_size)
        kh, kw = self.kernel_size
        # Compute the size of the flattened patch for each source.
        self.patch_dim = (ph * pw * channels) // self.num_heads
        # Create an embedding for the keys and queries.
        self.key_embedding = nn.Linear(self.patch_dim, attention_dim)
        self.query_embedding = nn.Linear(self.patch_dim, attention_dim)
        # Rearrange layer to go from images to patches.
        self.to_patches = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)
        # Rearrange layer to add a heads dimension to a sequence.
        self.to_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        # MBConv block to process the output of the attention mechanism.
        # The mbconv block is modified to use grouped convolutions, so that the
        # heads are processed in parallel and separately.
        self.mbconv = MBConv(
            self.channels * self.num_heads * self.n_keys,
            self.channels,
            downsample=False,
            groups=self.num_heads,
        )

    def forward(self, keys, queries):
        """
        Args:
            keys (list of torch.Tensor): List of keys images of shape (B, C, H, W).
                H and W may vary, while C must be the same for all keys.
                The keys also provide the values.
            queries (list of torch.Tensor): List of query images of shape (B, C, H, W).
                H and W may vary, while C must be the same as for the keys.
        Returns:
            list of torch.Tensor: List of output images of shape (B, channels, H, W).
                H and W will have the same shape as their corresponding queries.
        """
        # Store the original sizes of the queries.
        original_query_sizes = [query.shape[-2:] for query in queries]
        # Pad the queries and keys to the next multiple of the patch size.
        ph, pw = self.patch_size
        keys = [pad_to_next_multiple_of(key, (ph, pw)) for key in keys]
        queries = [pad_to_next_multiple_of(query, (ph, pw)) for query in queries]
        # Store the spatial sizes of the queries after padding.
        query_sizes = [query.shape[-2:] for query in queries]
        # Patchify the keys and queries.
        keys_patches = [self.to_patches(key) for key in keys]
        queries_patches = [self.to_patches(query) for query in queries]
        # Store the number of patches for each source.
        n_keys_patches = [key_patches.size(1) for key_patches in keys_patches]
        n_queries_patches = [query_patches.size(1) for query_patches in queries_patches]
        # Form the sequences of patches.
        keys_seq = torch.cat(keys_patches, dim=1)  # (b, n_keys_seq, patch_dim)
        queries_seq = torch.cat(queries_patches, dim=1)  # (b, n_queries_seq, patch_dim)
        # Reshape to add the heads dimension.
        keys_seq = self.to_heads(keys_seq)  # (b, hd, n_keys_seq, patch_dim)
        queries_seq = self.to_heads(queries_seq)  # (b, hd, n_queries_seq, patch_dim)
        # Embed the keys and queries.
        keys_emb = self.key_embedding(keys_seq)  # (b, hd, n_keys_seq, attention_dim)
        queries_emb = self.query_embedding(queries_seq)  # (b, hd, n_queries_seq, attention_dim)
        # Compute the attention weights.
        attn = torch.einsum("b h i d, b h j d -> b h i j", queries_emb, keys_emb)
        attn = F.softmax(attn / (self.dim_head**0.5), dim=-1)  # (b, hd, n_queries_seq, n_keys_seq)
        # Split the columns of the attention matrix by key image.
        attn_grouped_keys = torch.split(attn, n_keys_patches, dim=3)
        # Compute the weighted sum of the values with each group of keys.
        weighted_v = [torch.matmul(a, v) for a, v in zip(attn_grouped_keys, keys_patches)]
        # weighted_v is a list of (b, hd, n_queries_seq, patch_dim) tensors.
        # Stack back the weighted values
        weighted_v = torch.stack(weighted_v, dim=1)  # (n, n_keys, hd, n_queries_seq, patch_dim)
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
