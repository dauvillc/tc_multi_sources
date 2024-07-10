import torch
import numpy as np
from multi_sources.models.multisource_conv_attention import MultisourceConvAttention


if __name__ == "__main__":
    n_keys = 3
    n_queries = 5
    n_channels = 4
    n_heads = 2
    patch_size = 16
    attention_dim = 256
    possible_side_lens = [32, 75, 128, 243]
    rng = np.random.default_rng(42)

    module = MultisourceConvAttention(
        n_keys, n_queries, n_channels, patch_size, attention_dim, n_heads
    )

    # Create a list of len n_keys with random tensors of shape (1, n_channels, H, W)
    # where H and W are randomly chosen from possible_side_lens
    keys = [
        torch.randn(1, n_channels, H, W) for H, W in rng.choice(possible_side_lens, (n_keys, 2))
    ]
    queries = [
        torch.randn(1, n_channels, H, W) for H, W in rng.choice(possible_side_lens, (n_queries, 2))
    ]

    print(f"{len(keys)} keys with shapes:")
    for i, k in enumerate(keys):
        print(f"Key {i}: {k.shape}")
    print(f"{len(queries)} queries with shapes:")
    for i, q in enumerate(queries):
        print(f"Query {i}: {q.shape}")

    # Forward pass
    out = module(keys, queries)
    print(f"Found {len(out)} outputs, with shapes:")
    for i, o in enumerate(out):
        print(f"Output {i}: {o.shape}")
