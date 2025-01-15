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
                The class's forward method must be of the form
                `forward(values_seq, coords_seq) -> values_seq`.
                Each block in the backbone will be composed of these layers, in the order
                they appear in the dict.
        """
        super().__init__()
        self.values_dim, self.coords_dim = values_dim, coords_dim
        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleList()
            for layer_name, layer_kwargs in layers.items():
                layer_class = layer_kwargs["layer_class"]
                kwargs = {k: v for k, v in layer_kwargs.items() if k != "layer_class"}
                # Each layer is wrapped in an adaptive conditional normalization
                # that applies the conditioning to the values embeddings based on the
                # coordinates embeddings.
                block.append(
                    AdapativeConditionalNormalization(
                        layer_class(self.values_dim, self.coords_dim, **kwargs),
                        self.values_dim,
                        self.coords_dim,
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
                for source_name, source_output in layer_otp.items():
                    x[source_name]["embedded_values"] = source_output

        # Return only the predicted values
        return {source_name: x[source_name]["embedded_values"] for source_name in x.keys()}


class AdapativeConditionalNormalization(nn.Module):
    """Wraps a torch module to apply adaptive conditional normalization."""

    def __init__(self, module, input_dim, cond_dim):
        super().__init__()
        self.module = module
        self.input_norm = nn.LayerNorm(input_dim)
        self.cond_norm = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, input_dim * 3))

    def forward(self, data, *args, **kwargs):
        """Args:
            data (dict of str: tensor): Dictionary of inputs, such that
                data[source_name] contains the keys "embedded_coords" and
                "embedded_values".
            args, kwargs: Additional arguments for the wrapped module.
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens,
                of shape (B, L, D).
        """
        skips, gates = {}, {}
        for source_name, source_data in data.items():
            skip, cond = source_data["embedded_values"], source_data["embedded_coords"]
            x = self.input_norm(skip)
            shift, scale, gate = self.cond_norm(cond).chunk(3, dim=-1)
            x = x * (scale + 1) + shift
            # Save the module's input for that source
            source_data["embedded_values"] = x
            # Save the skip connection and gate for after the module
            skips[source_name] = skip
            gates[source_name] = gate
        # Apply the wrapped module with the updated inputs
        module_output = self.module(data, *args, **kwargs)
        # Multiply by the gate and the skip connections
        output = {}
        for source_name, source_output in module_output.items():
            output[source_name] = source_output * gates[source_name] + skips[source_name]
        return output
