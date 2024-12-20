"""Implements a general backbone for the mask-autoencoding task,
which can be used with custom blocks."""

import torch
import torch.nn as nn
from multi_sources.models.utils import pair


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for the multisource mask-autoencoding task."""

    def __init__(
        self,
        n_blocks,
        coords_dim,
        values_dim,
        layers={},
        sum_coords_to_values=False,
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
            sum_coords_to_values (bool): Whether to sum the coordinates embeddings to the
                values embeddings at the beginning of the backbone.
        """
        super().__init__()
        self.values_dim, self.coords_dim = values_dim, coords_dim
        self.sum_coords_to_values = sum_coords_to_values
        # Build the successive blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleList()
            for layer_name, layer_kwargs in layers.items():
                layer_class = layer_kwargs["layer_class"]
                kwargs = {k: v for k, v in layer_kwargs.items() if k != "layer_class"}
                # Before each block, add a layer normalization for the values
                # (the coordinates don't change between blocks, so no need to re-normalize them).
                block.append(
                    nn.ModuleList(
                        [
                            nn.LayerNorm(values_dim),
                            layer_class(self.values_dim, self.coords_dim, **kwargs),
                        ]
                    )
                )
            self.blocks.append(block)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                inputs[source_name] contains the keys "embedded_coordinates",
                "embedded_values" and "tokens_shape".
            attention_mask (list): List of attention masks for each source,
                of shape (b, n).
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Apply the blocks with skip connections
        for block in self.blocks:
            # Add masks embeddings at the start of each block if they exist
            block_input = {}
            for source_name, data in x.items():
                block_input[source_name] = {k: v for k, v in data.items()}
                if data.get("embedded_masks") is not None:
                    block_input[source_name]["embedded_values"] = (
                        data["embedded_values"] + data["embedded_masks"]
                    )
                else:
                    block_input[source_name]["embedded_values"] = data["embedded_values"]

            for norm, layer in block:
                # Create new dict for normalized values
                new_x = {}
                for source_name, data in block_input.items():
                    # Copy coordinates and shape
                    new_x[source_name] = {
                        "embedded_coords": data["embedded_coords"],
                        "tokens_shape": data["tokens_shape"],
                        "embedded_masks": data.get("embedded_masks"),  # Preserve masks
                    }
                    # Normalize the values
                    new_x[source_name]["embedded_values"] = norm(data["embedded_values"])
                
                # Apply layer and add skip connection
                new_values = layer(new_x, attention_mask=attention_mask)
                for source_name, data in new_x.items():
                    new_x[source_name]["embedded_values"] = (
                        new_values[source_name] + block_input[source_name]["embedded_values"]
                    )
                block_input = new_x

            x = block_input

        # Return only the predicted values
        return {source_name: x[source_name]["embedded_values"] for source_name in x.keys()}
