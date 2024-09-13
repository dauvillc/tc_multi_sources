"""Implements a general backbone for the mask-autoencoding task,
which can be used with custom blocks."""

import torch
import torch.nn as nn
from multi_sources.models.utils import pair


class MultisourceGeneralBackbone(nn.Module):
    """General backbone for the multisource mask-autoencoding task.
    The model receives a list of tuples (c, v) as input, where:
    - c (b, n, dim_coords) is the coordinates of the tokens.
    - v (b, n, dim_values) is the values of the tokens.
        Some tokens may be masked.
    The model returns a list of tensors of shape (b, n, dim_values), which are the
    predicted values of the tokens.
    """

    def __init__(
        self,
        patch_size,
        n_blocks,
        coords_dim=None,
        values_dim=None,
        layers={},
        output_block=None,
        sum_coords_to_values=False,
    ):
        """
        Args:
            patch_size (int or tuple): size of the patches.
            n_blocks (int): number of blocks in the backbone.
            coords_dim (int): Embedding dimension for the geographical coordinates.
                Defaults to patch_height * patch_width * 2.
            values_dim (int): Embedding dimension for the values.
                Defaults to patch_height * patch_width.
            layers (dict) Dict defining the successive layers that compose the backbone,
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
        self.patch_size = pair(patch_size)
        ph, pw = self.patch_size
        if coords_dim is None:
            coords_dim = ph * pw * 2  # 2 for the latitude and longitude
        if values_dim is None:
            values_dim = ph * pw
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
                # (the coords don't change between blocks, so no need to re-normalize them).
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
                inputs[source_name] contains the keys "dt", "embedded_dt", "embedded_coords",
                and "embedded_values".
            attention_mask (list): List of attention masks for each source,
                of shape (b, n).
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs[source_name] contains the predicted values of the tokens.
        """
        # Apply the blocks with skip connections
        for block in self.blocks:
            for norm, layer in block:
                # The layer receives its input in the same format as x. However, the values
                # need to be normalized before being passed to the layer.
                # In order not to modify the original x, we create a new dictionary, which
                # we pass to the layer.
                new_x = {}
                for source_name, data in x.items():
                    new_x[source_name] = {}
                    new_x[source_name]["embedded_coords"] = data["embedded_coords"]
                    new_x[source_name]["dt"] = data["dt"]
                    new_x[source_name]["embedded_dt"] = data["embedded_dt"]
                    # Normalize the values in each source
                    new_x[source_name]["embedded_values"] = norm(data["embedded_values"])
                # Apply the layer and add the skip connection
                new_values = layer(new_x, attention_mask=attention_mask)  # dict {source_name: tensor}
                for source_name, data in new_x.items():
                    new_x[source_name]["embedded_values"] = new_values[source_name] + data["embedded_values"]
                x = new_x
        return {source_name: x[source_name]['embedded_values'] for source_name in x.keys()}
