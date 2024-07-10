import torch
from multi_sources.models.encoder_decoder_transformer import EncoderDecoderTransformer
from multi_sources.models.multisource_conv_attention import MultisourceConvAttention


if __name__ == "__main__":
    n_input_sources = 3
    n_masked_sources = 1
    n_input_channels = [4, 5, 6]
    n_output_channels = [7]
    n_inner_channels = 32

    n_blocks = 3
    block_class = MultisourceConvAttention
    patch_size = 32
    attention_dim = 256
    num_heads = 4
    kernel_size = 3

    transformer = EncoderDecoderTransformer(
        n_input_sources,
        n_masked_sources,
        n_input_channels,
        n_output_channels,
        n_inner_channels,
        block_class,
        n_blocks,
        patch_size=patch_size,
        attention_dim=attention_dim,
        num_heads=num_heads,
        kernel_size=kernel_size,
    )

    H, W = 64, 64
    # Create a dummy input: one image of shape (1, c, H, W) for each source
    input_sources = [torch.randn(1, c, H, W) for c in n_input_channels]
    # Create a dummy target: one image of shape (1, 3, H, W) for each masked source
    masked_sources = [torch.rand(1, 3, H, W) for _ in range(n_masked_sources)]

    print(f"Found {len(input_sources)} input sources")
    for i, s in enumerate(input_sources):
        print(f"Input source {i}: {s.shape}")
    print(f"Found {len(masked_sources)} masked sources")
    for i, s in enumerate(masked_sources):
        print(f"Masked source {i}: {s.shape}")

    # Forward pass
    output = transformer(input_sources, masked_sources)

    print(f"Found {len(output)} output sources")
    for i, o in enumerate(output):
        print(f"Output source {i}: {o.shape}")
