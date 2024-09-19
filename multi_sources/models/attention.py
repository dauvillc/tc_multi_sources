"""Defines attention mechanisms for the multisource masked autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionMap(nn.Module):
    """Computes the attention map from the embedded keys and queries.
    Using this intermediary Module allows to retrieve the attention maps
    with register_forward_hook."""

    def __init__(self, dim_head, relative_pos=False, rel_pos_dim_head=None):
        super().__init__()
        self.scale = dim_head**-0.5

        self.relative_pos = relative_pos
        if self.relative_pos:
            self.rel_pos_scale = rel_pos_dim_head**-0.5

    def forward(self, keys, queries, pos_key=None, pos_query=None, mask=None):
        """
        Args:
            keys: Tensor of shape (batch_size, num_keys, embed_dim), or
                (batch_size, heads, num_keys, dim_head). The same is true for
                all other arguments.
            queries: Tensor of shape (batch_size, num_queries, embed_dim)
            pos_key: Tensor of shape (batch_size, num_keys, rel_pos_dim)
            pos_query: Tensor of shape (batch_size, num_queries, rel_pos_dim)
            mask: Tensor of shape (batch_size, num_keys) or (batch_size, heads, num_keys),
                or None. Keys for which the mask is True will not be attended to.
        Returns:
            Tensor of shape (batch_size, num_queries, num_keys)
        """
        # Clamp the keys and queries to avoid numerical instability in the case where
        # they are perfectly correlated.
        keys, queries = keys.clamp(-5, 5), queries.clamp(-5, 5)
        dots = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Optional relative positional encodings
        if self.relative_pos:
            pos_key, pos_query = pos_key.clamp(-5, 5), pos_query.clamp(-5, 5)
            rel_pos_dots = torch.matmul(pos_query, pos_key.transpose(-2, -1)) * self.rel_pos_scale
            dots = dots + rel_pos_dots
        # Mask the columns of the attention map that correspond to the masked tokens.
        if mask is not None:
            # Expand the mask in the middle to reach the same number of dimensions as dots.
            if dots.dim() == 3:
                mask = mask.unsqueeze(1)
            elif dots.dim() == 4:
                mask = mask.unsqueeze(1).unsqueeze(1)
            dots = dots.masked_fill(mask, float("-inf"))

        return F.softmax(dots, dim=-1)


class SelfAttention(nn.Module):
    """A self-attention block that can use values that are different from the keys/queries."""

    def __init__(self, key_dim, value_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"Inner dimension {inner_dim} must be divisible by the number of heads {num_heads}"
            )
        self.qk_norm = nn.LayerNorm(key_dim)
        self.norm_value = nn.LayerNorm(value_dim)

        self.to_qk = nn.Linear(key_dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.num_heads = num_heads

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head)

        self.output_proj = nn.Sequential(nn.Linear(inner_dim, value_dim), nn.Dropout(dropout))

    def forward(self, keys, values, mask=None):
        keys = self.qk_norm(keys)
        values = self.norm_value(values)

        q, k = self.to_qk(keys).chunk(2, dim=-1)
        q, k = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), (q, k))

        v = self.to_v(values)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = self.attention_map(k, q, mask=mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.output_proj(out)


class ValuesCoordinatesAttentionInternal(nn.Module):
    """Self-attention block that uses the values values as well as the
    coordinates to compute the attention weights,
    as Softmax(QpKp^T + QcKc^T).
    This is an internal module used for factorizing the attention code."""

    def __init__(self, values_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert inner_dim % num_heads == 0
        self.num_heads = num_heads

        self.values_to_qkv = nn.Linear(values_dim, inner_dim * 3, bias=False)
        self.coords_to_qk = nn.Linear(coords_dim, inner_dim * 2, bias=False)

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

    def forward(self, values, coords, attention_mask=None):
        """
        Args:
            values (tensor): Tensor of shape (batch_size, seq_len, values_dim).
            coords (tensor): Tensor of shape (batch_size, seq_len, coords_dim).
            attention_mask (tensor): Tensor of shape (batch_size, seq_len), or None.
        Returns:
            Tensor of shape (batch_size, seq_len, values_dim), the updated values.
        """
        # Project the values and coordinates to the query, key and value spaces.
        qkv_values = self.values_to_qkv(values).chunk(3, dim=-1)
        qk_coords = self.coords_to_qk(coords).chunk(2, dim=-1)
        qp, kp, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qkv_values
        )
        qc, kc = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qk_coords)

        # Compute the attention map using two sets of keys and queries, one from the values
        # and one from the coordinates.
        attn = self.attention_map(kp, qp, kc, qc, mask=attention_mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.output_proj(out)
        return out


class ValuesCoordinatesAttention(nn.Module):
    """Self-attention block that uses the values values as well as the
    coordinates to compute the attention weights,
    as Softmax(QpKp^T + QcKc^T)."""

    def __init__(self, values_dim, coords_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.module = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "dt", "embedded_dt", "embedded_coords",
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Concatenate the sequences from all sources.
        values = torch.cat([data["embedded_values"] for data in inputs.values()], dim=1)
        coords = torch.cat([data["embedded_coords"] for data in inputs.values()], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([mask for mask in attention_mask.values()], dim=1)

        out = self.module(values, coords, attention_mask)

        # Split the updated values sequence back into sequences for each source.
        source_seqs = torch.split(
            out, [data["embedded_values"].shape[1] for data in inputs.values()], dim=1
        )
        return {source_name: seq for source_name, seq in zip(inputs.keys(), source_seqs)}


class WindowedValuesCoordinatesAttention(nn.Module):
    """Attention block that uses a window over the source dimension so that the cost isn't
    quadratic over that dimension:
    - The sources are first sorted in chronological order.
    - The sources are then grouped in windows of size window_size.
    - The attention is computed over the sources in each window.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_dim,
        shifted_windows,
        num_heads=8,
        dropout=0.0,
        window_size=4,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            inner_dim (int): Inner dimension of the attention block.
            shifted_windows (bool): Whether to shift the windows by half the window size.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            window_size (int): Number of sources included in each window. Must be even.
        """
        super().__init__()
        if window_size % 2 != 0:
            raise ValueError("Window size must be even.")
        self.window_size = window_size
        self.shifted_windows = shifted_windows

        self.attention_block = ValuesCoordinatesAttentionInternal(
            values_dim, coords_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "dt", "embedded_dt", "embedded_coords",
                and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Concatenate the dt tensors from all sources to obtain a tensor
        # of shape (b, n_sources).
        dts = torch.cat([data["dt"] for data in inputs.values()], dim=1)
        # Sort the sources in chronological order (ascending dt).
        sorted_indices = dts.argsort(dim=1)  # (b, n_sources, 1)
        # In order for all windows to have the same size, we need all sources to have
        # the same number of tokens. We'll pad the sources to the maximum number of tokens.
        max_tokens = max(data["embedded_values"].shape[1] for data in inputs.values())
        padded_values, padded_coords = [], []
        padded_attn_masks = []
        for source_name, data in inputs.items():
            values = data["embedded_values"]  # (b, n_tokens, values_dim)
            coords = data["embedded_coords"]  # (b, n_tokens, coords_dim)
            padding = max_tokens - values.shape[1]
            padded_values.append(F.pad(values, (0, 0, 0, padding)))
            padded_coords.append(F.pad(coords, (0, 0, 0, padding)))
            # We'll use the attention mask to mask the padding tokens.
            if attention_mask is None:
                source_attn_mask = torch.full(
                    (values.shape[0], values.shape[1]), False, dtype=torch.bool
                ).to(values.device)
            else:
                source_attn_mask = attention_mask[source_name]
            padded_attn_masks.append(F.pad(source_attn_mask, (0, padding), value=True))
        padded_values = torch.stack(padded_values, dim=1)  # (b, n_sources, max_tokens, values_dim)
        padded_coords = torch.stack(padded_coords, dim=1)  # (b, n_sources, max_tokens, coords_dim)
        padded_attn_masks = torch.stack(padded_attn_masks, dim=1)  # (b, n_sources, max_tokens)
        # We now need to order the sources in the same way as the sorted_indices.
        sorted_indices = sorted_indices.unsqueeze(-1).expand(padded_values.shape)
        padded_values = torch.gather(padded_values, 1, sorted_indices)
        padded_coords = torch.gather(padded_coords, 1, sorted_indices)
        padded_attn_masks = torch.gather(padded_attn_masks, 1, sorted_indices[:, :, :, 0])
        # At this point, padded_values has shape (b, n_sources, max_tokens, values_dim).
        # In order to group the sources in windows, we need n_sources to be divisible by
        # the window size. We'll pad the sources to reach the next multiple of the window size.
        ws, n_sources = self.window_size, padded_values.shape[1]
        padding = (ws - n_sources % ws) % ws
        padded_values = F.pad(padded_values, (0, 0, 0, 0, 0, padding))
        padded_coords = F.pad(padded_coords, (0, 0, 0, 0, 0, padding))
        padded_attn_masks = F.pad(padded_attn_masks, (0, 0, 0, padding), value=True)
        # If requested, shift the windows by half the window size.
        if self.shifted_windows:
            padded_values = torch.roll(padded_values, shifts=ws // 2, dims=1)
            padded_coords = torch.roll(padded_coords, shifts=ws // 2, dims=1)
            padded_attn_masks = torch.roll(padded_attn_masks, shifts=ws // 2, dims=1)

        # Form the windows by grouping the sources along the batch dimension.
        padded_values = rearrange(padded_values, "b (n w) t d -> (b n) (w t) d", w=ws)
        padded_coords = rearrange(padded_coords, "b (n w) t d -> (b n) (w t) d", w=ws)
        padded_attn_masks = rearrange(padded_attn_masks, "b (n w) t -> (b n) (w t)", w=ws)

        # Apply the attention block to the windows.
        x = self.attention_block(padded_values, padded_coords, padded_attn_masks)
        # If the windows were shifted, we need to shift the output back.
        if self.shifted_windows:
            x = torch.roll(x, shifts=-ws // 2, dims=1)

        # Go back from windows to sources.
        bs = dts.shape[0]  # batch size
        x = rearrange(x, "(b n) (w t) d -> b (n w) t d", w=ws, b=bs)

        # Remove the padding over the sources dimension.
        x = x[:, :n_sources, :, :]
        # Unsort the sources to restore the original order.
        unsorted_indices = sorted_indices.argsort(dim=1)
        unsorted_indices = unsorted_indices
        x = torch.gather(x, 1, unsorted_indices)
        # Reform the dictionary of outputs.
        result = {source_name: x[:, i] for i, source_name in enumerate(inputs.keys())}
        # Remove the padding over the tokens dimension.
        for source_name, data in result.items():
            original_tokens = inputs[source_name]["embedded_values"].shape[1]
            result[source_name] = data[:, :original_tokens, :]
        return result
