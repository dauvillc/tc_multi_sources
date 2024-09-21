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


class ValuesMetadataAttentionInternal(nn.Module):
    """Self-attention block that uses the values values as well as the
    metadata to compute the attention weights,
    as Softmax(QpKp^T + QmKm^T).
    This is an internal module used to factorize the attention code."""

    def __init__(self, values_dim, metadata_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert inner_dim % num_heads == 0
        self.num_heads = num_heads

        self.values_to_qkv = nn.Linear(values_dim, inner_dim * 3, bias=False)
        self.meta_to_qk = nn.Linear(metadata_dim, inner_dim * 2, bias=False)

        dim_head = inner_dim // num_heads
        self.attention_map = AttentionMap(dim_head, relative_pos=True, rel_pos_dim_head=dim_head)
        self.output_proj = nn.Sequential(nn.Linear(inner_dim, values_dim), nn.Dropout(dropout))

    def forward(self, values, metadata, attention_mask=None):
        """
        Args:
            values (tensor): Tensor of shape (batch_size, seq_len, values_dim).
            metadata (tensor): Tensor of shape (batch_size, seq_len, metadata_dim).
            attention_mask (tensor): Tensor of shape (batch_size, seq_len), or None.
        Returns:
            Tensor of shape (batch_size, seq_len, values_dim), the updated values.
        """
        # Project the values and metadata to the query, key and value spaces.
        qkv_values = self.values_to_qkv(values).chunk(3, dim=-1)
        qk_meta = self.meta_to_qk(metadata).chunk(2, dim=-1)
        qp, kp, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qkv_values
        )
        qm, km = map(lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.num_heads), qk_meta)

        # Compute the attention map using two sets of keys and queries, one from the values
        # and one from the metadata.
        attn = self.attention_map(kp, qp, km, qm, mask=attention_mask)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.output_proj(out)
        return out


class ValuesMetadataAttention(nn.Module):
    """Self-attention block that uses the values values as well as the
    metadata to compute the attention weights,
    as Softmax(QpKp^T + QmKm^T)."""

    def __init__(self, values_dim, metadata_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.module = ValuesMetadataAttentionInternal(
            values_dim, metadata_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_metadata" and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Concatenate the sequences from all sources.
        values = torch.cat([data["embedded_values"] for data in inputs.values()], dim=1)
        metadata = torch.cat([data["embedded_metadata"] for data in inputs.values()], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([mask for mask in attention_mask.values()], dim=1)

        out = self.module(values, metadata, attention_mask)

        # Split the updated values sequence back into sequences for each source.
        source_seqs = torch.split(
            out, [data["embedded_values"].shape[1] for data in inputs.values()], dim=1
        )
        return {source_name: seq for source_name, seq in zip(inputs.keys(), source_seqs)}


class WindowedValuesMetadataAttention(nn.Module):
    """Attention block that uses a window over the source dimension so that the cost isn't
    quadratic over that dimension:
    - The sources are first sorted in chronological order.
    - The sources are then grouped in windows of size window_size.
    - The attention is computed over the sources in each window.
    """

    def __init__(
        self,
        values_dim,
        metadata_dim,
        inner_dim,
        shifted_windows,
        num_heads=8,
        dropout=0.0,
        window_size=4,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            metadata_dim (int): Embedding dimension of the metadata.
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

        self.attention_block = ValuesMetadataAttentionInternal(
            values_dim, metadata_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "dt", "embedded_metadata",
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
        values, meta = [], []
        attn_masks = []
        for source_name, data in inputs.items():
            source_values = data["embedded_values"]  # (b, n_tokens, values_dim)
            source_meta = data["embedded_metadata"]  # (b, n_tokens, metadata_dim)
            padding = max_tokens - source_values.shape[1]
            values.append(F.pad(source_values, (0, 0, 0, padding)))
            meta.append(F.pad(source_meta, (0, 0, 0, padding)))
            # We'll use the attention mask to mask the padding tokens.
            if attention_mask is None:
                source_attn_mask = torch.full(
                    (source_values.shape[0], source_values.shape[1]), False, dtype=torch.bool
                ).to(source_values.device)
            else:
                source_attn_mask = attention_mask[source_name]
            attn_masks.append(F.pad(source_attn_mask, (0, padding), value=True))
        values = torch.stack(values, dim=1)  # (b, n_sources, max_tokens, values_dim)
        meta = torch.stack(meta, dim=1)  # (b, n_sources, max_tokens, metadata_dim)
        attn_masks = torch.stack(attn_masks, dim=1)  # (b, n_sources, max_tokens)
        # We now need to order the sources in the same way as the sorted_indices.
        si_expanded = sorted_indices.unsqueeze(-1).expand(values.shape)
        values = torch.gather(values, 1, si_expanded)
        meta = torch.gather(meta, 1, si_expanded)
        attn_masks = torch.gather(attn_masks, 1, sorted_indices.expand(attn_masks.shape))
        # At this point, values has shape (b, n_sources, max_tokens, values_dim).
        # In order to group the sources in windows, we need n_sources to be divisible by
        # the window size. We'll pad the sources to reach the next multiple of the window size.
        ws, n_sources = self.window_size, values.shape[1]
        padding = (ws - n_sources % ws) % ws
        values = F.pad(values, (0, 0, 0, 0, 0, padding))
        meta = F.pad(meta, (0, 0, 0, 0, 0, padding))
        attn_masks = F.pad(attn_masks, (0, 0, 0, padding), value=True)
        # If requested, shift the windows by half the window size.
        if self.shifted_windows:
            values = torch.roll(values, shifts=ws // 2, dims=1)
            meta = torch.roll(meta, shifts=ws // 2, dims=1)
            attn_masks = torch.roll(attn_masks, shifts=ws // 2, dims=1)

        # Form the windows by grouping the sources along the batch dimension.
        values = rearrange(values, "b (n w) t d -> (b n) (w t) d", w=ws)
        meta = rearrange(meta, "b (n w) t d -> (b n) (w t) d", w=ws)
        attn_masks = rearrange(attn_masks, "b (n w) t -> (b n) (w t)", w=ws)

        # Apply the attention block to the windows.
        x = self.attention_block(values, meta, attn_masks)
        # If the windows were shifted, we need to shift the output back.
        if self.shifted_windows:
            x = torch.roll(x, shifts=-ws // 2, dims=1)

        # Go back from windows to sources.
        bs = dts.shape[0]  # batch size
        x = rearrange(x, "(b n) (w t) d -> b (n w) t d", w=ws, b=bs)

        # Remove the padding over the sources dimension.
        x = x[:, :n_sources, :, :]
        # Unsort the sources to restore the original order.
        unsorted_indices = sorted_indices.argsort(dim=1).unsqueeze(-1).expand(x.shape)
        x = torch.gather(x, 1, unsorted_indices)
        # Reform the dictionary of outputs.
        result = {source_name: x[:, i] for i, source_name in enumerate(inputs.keys())}
        # Remove the padding over the tokens dimension.
        for source_name, data in result.items():
            original_tokens = inputs[source_name]["embedded_values"].shape[1]
            result[source_name] = data[:, :original_tokens, :]
        return result


class AdaptiveValuesMetadataAttention(nn.Module):
    """Attention block that uses a window over the source dimension so that the cost isn't
    quadratic over that dimension. The window size is fixed, but the sources chosen in each
    window are adaptively selected based on the metadata.
    - For each source, the metadata tokens are aggregated into a single metadata vector.
    - An attention map is computed between the metadata vectors.
    - For each source, a window is defined as the sources with the top-W attention weights.
    - The attention is computed over the sources in each window.
    """

    def __init__(
        self,
        values_dim,
        metadata_dim,
        inner_dim,
        num_heads=8,
        dropout=0.0,
        window_size=3,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            metadata_dim (int): Embedding dimension of the metadata.
            inner_dim (int): Inner dimension of the attention block.
            num_heads (int): Number of heads in the attention block.
            dropout (float): Dropout rate.
            window_size (int): Number of sources included in each window.
        """
        super().__init__()
        self.window_size = window_size

        # Layers to compute the attention map between the metadata vectors.
        # We'll only use one head for this attention map.
        self.meta_to_qk = nn.Linear(metadata_dim, inner_dim * 2, bias=False)
        self.attention_map = AttentionMap(inner_dim)

        self.attention_block = ValuesMetadataAttentionInternal(
            values_dim, metadata_dim, inner_dim, num_heads, dropout
        )

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_metadata" and "embedded_values".
            attention_mask (dict of str: tensor): Dictionary of attention masks for each source,
                such that attention_mask[source_name] has shape (b, n).

        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Average the metadata tokens for each source, so that each source can be represented
        # by a single metadata vector.
        metadata = {
            source_name: data["embedded_metadata"].mean(dim=1)
            for source_name, data in inputs.items()
        }
        # Concatenate the metadata vectors into a single tensor.
        metadata = torch.stack(list(metadata.values()), dim=1)  # (b, S, D_meta)
        # If there are less sources than the window size, we'll use all sources.
        ws = self.window_size
        if ws >= metadata.shape[1]:
            ws = metadata.shape[1]
        # Compute the attention map between the metadata vectors.
        qm, km = self.meta_to_qk(metadata).chunk(2, dim=-1)
        meta_attn = self.attention_map(km, qm)  # (b, S, S)
        # For each source S, we want S to be included in its own window. To
        # ensure this, we'll set the diagonal of the attention map to over 1.
        meta_attn = meta_attn + 2 * torch.eye(meta_attn.shape[1]).to(meta_attn.device)
        # Select the top-W sources for each source.
        _, top_indices = torch.topk(meta_attn, self.window_size, dim=2, sorted=True)
        # We can't use gahter directly as the sources have varying sequence lengths.
        # First, we need to pad the sequences of each source to the same length.
        max_tokens = max(data["embedded_values"].shape[1] for data in inputs.values())
        values, meta = [], []
        attn_masks = []
        for source_name, data in inputs.items():
            source_values = data["embedded_values"]
            source_meta = data["embedded_metadata"]
            padding = max_tokens - source_values.shape[1]
            values.append(F.pad(source_values, (0, 0, 0, padding)))
            meta.append(F.pad(source_meta, (0, 0, 0, padding))
            )
            if attention_mask is None:
                source_attn_mask = torch.full(
                    (source_values.shape[0], source_values.shape[1]), False, dtype=torch.bool
                ).to(source_values.device)
            else:
                source_attn_mask = attention_mask[source_name]
            attn_masks.append(F.pad(source_attn_mask, (0, padding), value=True))
        values = torch.stack(values, dim=1)  # (b, S, max_tokens, values_dim)
        meta = torch.stack(meta, dim=1) # (b, S, max_tokens, metadata_dim)
        attn_masks = torch.stack(attn_masks, dim=1) # (b, S, max_tokens)
        # Select the top-W sources for each source.
        bs, S, N, D_v = values.shape
        D_m = meta.shape[-1]
        top_indices = top_indices.view(bs, S, ws, 1, 1)
        values = values.view(bs, 1, S, N, -1).expand(-1, S, -1, -1, -1)
        meta = meta.view(bs, 1, S, N, -1).expand(-1, S, -1, -1, -1)
        attn_masks = attn_masks.view(bs, 1, S, N).expand(-1, S, -1, -1)
        values = torch.gather(values, 2, top_indices.expand((bs, S, ws, N, D_v)))
        meta = torch.gather(meta, 2, top_indices.expand((bs, S, ws, N, D_m)))
        attn_masks = torch.gather(attn_masks, 2, top_indices[..., 0].expand((bs, S, ws, N)))
        # Group the tokens of the same window together.
        values = rearrange(values, "b s w n d -> (b s) (w n) d")
        meta = rearrange(meta, "b s w n d -> (b s) (w n) d")
        attn_masks = rearrange(attn_masks, "b s w n -> (b s) (w n)")

        # Apply the attention block to the windows.
        x = self.attention_block(values, meta, attn_masks)

        # Go back from windows to sources.
        x = rearrange(x, "(b s) (w n) d -> b s w n d", s=S, w=ws)
        # For each source, the first element along the window dimension is the source itself.
        x = x[:, :, 0]
        # Reform the dictionary of outputs.
        result = {source_name: x[:, i] for i, source_name in enumerate(inputs.keys())}
        # Remove the padding over the tokens dimension.
        for source_name, data in result.items():
            original_tokens = inputs[source_name]["embedded_values"].shape[1]
            result[source_name] = data[:, :original_tokens, :]
        return result


