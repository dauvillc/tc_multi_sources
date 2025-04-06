"""Implements a general backbone for the mask-autoencoding task,
which can be used with custom blocks."""

import torch
import torch.nn as nn


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for the multisource mask-autoencoding task."""

    def __init__(
        self,
        n_blocks,
        coords_dim,
        values_dim,
        layers={},
    ):
        """
        Args:
            n_blocks (int): number of blocks in the backbone.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            values_dim (int): Embedding dimension for the values of each source.
            layers (dict): Dict defining the successive layers that compose the backbone,
                as {layer_name: layer_kwargs}.
                For each layer, the kwargs must include the key 'layer_class',
                which should be a nn.Module class. The other keys are the arguments
                to this class's constructor.
                The class's constructor must be of the form
                `layer_class(values_dim, coords_dim, **kwargs)`.
                It can accept an argument 'block_idx' (int), which is the index of the block
                in which the layer is applied.
                The class's forward method must be of the form
                `forward(values_seq, coords_seq) -> values_seq`.
                Each block in the backbone will be composed of these layers, in the order
                they appear in the dict.
        """
        super().__init__()
        self.values_dim, self.coords_dim = values_dim, coords_dim
        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            block = nn.ModuleList()
            for layer_name, layer_kwargs in layers.items():
                layer_class = layer_kwargs["layer_class"]
                kwargs = {k: v for k, v in layer_kwargs.items() if k != "layer_class"}
                kwargs["block_idx"] = block_idx
                # Each layer is wrapped in an adaptive conditional normalization
                # that applies the conditioning to the values embeddings based on the
                # coordinates embeddings.
                block.append(
                    AdapativeConditionalNormalization(
                        layer_class(self.values_dim, self.coords_dim, **kwargs),
                        self.values_dim,
                    )
                )
            self.blocks.append(block)

    def forward(self, x):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_coordinates",
                "embedded_values" and "tokens_shape".
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        for block in self.blocks:
            # Update the values via the successive layers
            for layer in block:
                # Apply the layer and update the values
                layer_otp = layer(x)
                # Check that all sources are present in the output
                assert set(x.keys()) == set(layer_otp.keys())
                # Update the values
                for source_name, source_output in layer_otp.items():
                    x[source_name]["embedded_values"] = source_output

        # Return only the predicted values
        return {source_name: x[source_name]["embedded_values"] for source_name in x.keys()}


class AdapativeConditionalNormalization(nn.Module):
    """Wraps a torch module to apply adaptive conditional normalization, as in DiT.
    The module expects the data to include a key "conditioning", of same shape
    as the values. If that key is absent, no conditioning is applied and the input
    is just passed through a LayerNorm, and a residual connection is applied to
    the wrapped module's output.
    """

    def __init__(self, module, dim):
        super().__init__()
        self.module = module
        self.input_norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)

    def forward(self, data, *args, **kwargs):
        """Args:
            data (dict of str: tensor): Dictionary of inputs, such that
                data[source_name] contains the keys
                "embedded_values" and "conditioning".
            args, kwargs: Additional arguments for the wrapped module.
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens,
                of shape (B, ..., values_dim).
        """
        skips, gates = {}, {}
        modulated_data = {}
        for src, source_data in data.items():
            # Create a new dict to avoid modifying the input one in-place
            modulated_data[src] = {k: v for k, v in source_data.items()}

            skip, cond = source_data["embedded_values"], source_data["conditioning"]
            x = self.input_norm(skip)
            skips[src] = skip

            if cond is not None:
                shift, scale, gate = self.cond_proj(cond).chunk(3, dim=-1)
                x = x * (scale + 1) + shift
                # Save the skip connection and gate for after the module
                gates[src] = gate

            # Save the module's input for that source
            modulated_data[src]["embedded_values"] = x

        # Apply the wrapped module with the updated inputs
        module_output = self.module(modulated_data, *args, **kwargs)
        # Multiply by the gate and the skip connections
        output = {}
        for src, source_output in module_output.items():
            if src in gates:
                output[src] = source_output * gates[src] + skips[src]
            else:
                output[src] = source_output + skips[src]
        return output
