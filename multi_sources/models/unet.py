"""Implements a simple UNet model in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """A simple UNet model to perform field reconstruction or autoencoding.

    The model receives as input a dict {source: (S, DT, C, V) or None} where:
    - S is a tensor of shape (batch_size,) containing the source index.
    - DT is a tensor of shape (batch_size,) containing the time delta between the element's time
        and the reference time.
    - C is a tensor of shape (batch_size, 3, H, W) containing the coordinates
        (lat, lon) of each pixel, and the land-sea mask.
    - V is a tensor of shape (batch_size, channels, H, W) containing the values of each pixel.
    The model outputs a tensor of shape (batch_size, channels, H, W) containing
        the predicted values.
    The model expects H and W to be the same for all sources, and to be a multiple of 2^n_blocks.

    The model first projects S and DT into a tensor ST of shape (batch_size, 2, H, W).
    For each source, the concatenation of SDT, C and V is concatenated along the channel dimension.
    The result of that operation for all sources is again concatenated along the channel dimension.
    The resulting tensor is then fed to the UNet.
    """

    def __init__(self, sizes, n_variables,  n_blocks=4, base_filters=32, kernel_size=3):
        """
        Args:
            sizes (dict of str: tuple of int): dict {source: (H, W)} containing the sizes of the
                inputs for each source.
            n_variables (dict): dict {source: n_variables} containing the number of variables
                for each source.
            n_blocks (int): Number of blocks in the UNet. Each block downsamples the input by
                a factor of 2.
            base_filters (int): Number of filters in the output of the first block.
            kernel_size (int): Kernel size for the convolutional layers.
        """
        super().__init__()
        self.n_variables = n_variables
        self.size = sizes
        self.n_sources = len(n_variables)
        self.n_blocks = n_blocks
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        # Compute the number of input channels for the first block:
        # for each source:
        # 2 for the source and time delta, 3 for the coordinates and land mask,
        # and the number of variables
        self.input_channels = sum(2 + 3 + n for n in n_variables.values())
        # The number of output channels is the total number of variables
        self.output_channels = sum(n for n in n_variables.values())
        # Create the UNet
        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = self.input_channels
        for i in range(n_blocks):
            out_channels = base_filters * 2**i
            encoder_block = nn.ModuleList(
                [
                    # Apply two conv layers with GELU activation and batch normalization
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels, kernel_size, padding=kernel_size // 2
                        ),
                        nn.SiLU(),
                        nn.BatchNorm2d(out_channels),
                        nn.Conv2d(
                            out_channels, out_channels, kernel_size, padding=kernel_size // 2
                        ),
                        nn.SiLU(),
                        nn.BatchNorm2d(out_channels),
                    ),
                    # Apply a conv layer with stride 2 to downsample the input
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels * 2, 2, stride=2, padding=0),
                        nn.SiLU(),
                        nn.BatchNorm2d(out_channels * 2),
                    ),
                ]
            )
            in_channels = out_channels * 2
            self.encoder.append(encoder_block)
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(n_blocks):
            out_channels = in_channels // 2 if i < n_blocks - 1 else self.output_channels
            decoder_block = nn.ModuleList(
                [
                    # Upsample the input
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels,
                            in_channels,
                            2,
                            stride=2,
                        ),
                        nn.SiLU(),
                        nn.BatchNorm2d(in_channels),
                    ),
                    # Apply two conv layers with SiLU activation and batch normalization
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels + in_channels // 2,
                            out_channels,
                            kernel_size,
                            padding=kernel_size // 2,
                        ),
                        nn.SiLU(),
                        nn.BatchNorm2d(out_channels),
                        nn.Conv2d(
                            out_channels, out_channels, kernel_size, padding=kernel_size // 2
                        ),
                        nn.SiLU(),
                        nn.BatchNorm2d(out_channels),
                    ),
                ]
            )
            in_channels = in_channels // 2
            self.decoder.append(decoder_block)
        # Output conv layer
        self.output_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    self.output_channels,
                    self.output_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.SiLU(),
                nn.BatchNorm2d(self.output_channels),
                nn.Conv2d(
                    self.output_channels,
                    self.output_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                ),
            ]
        )

    def forward(self, batch):
        """Forward pass of the model.
        Args:
            x (dict): dict {source: S, DT, C, V} containing the input tensors.
        Returns:
            y (dict): dict {source: V'} containing the output tensors.
        """
        # Check that the number of sources is correct
        if len(batch) != self.n_sources:
            raise ValueError(f"Expected {self.n_sources} sources, got {len(batch)}.")
        # Transform the input dict into a list of tensors, ordered by source index
        x = {int(s[0].item()): (s, dt, c, v) for s, dt, c, v in batch.values()}
        x = [x[i] for i in range(self.n_sources)]
        # For each source, pad C and V to the next multiple of 2^n_blocks
        # Then project S and DT into a tensor of shape (batch_size, 2, H, W)
        # - First, browse all sources to find their sizes
        widths, heights = [], []
        for _, _, _, v in x:
            h, w = v.shape[-2:]
            heights.append(h)
            widths.append(w)
        # - Compute the next multiple of 2^n_blocks for the height and width
        power = 2**self.n_blocks
        h = (max(heights) + power - 1) // power * power
        w = (max(widths) + power - 1) // power * power
        # - Browse all sources to pad C and V to the computed size
        st, original_sizes = [], []
        for k, (s, dt, c, v) in enumerate(x):
            # Pad C and V
            original_sizes.append(v.shape[-2:])
            c = F.pad(c, (0, w - c.shape[-1], 0, h - c.shape[-2]))
            v = F.pad(v, (0, w - v.shape[-1], 0, h - v.shape[-2]))
            x[k] = (s, dt, c, v)
            # Project S and DT
            sdt = torch.stack((s, dt), dim=1)
            sdt = sdt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
            st.append(sdt)
        # For each source, concat SDT, C and V along the channel dimension
        inputs = []
        for k, (s, dt, c, v) in enumerate(x):
            inputs.append(torch.cat((st[k], c, v), dim=1))
        # Concatenate the inputs along the channel dimension
        x = torch.cat(inputs, dim=1)
        # Apply the encoder
        skips = []
        for conv, downsample in self.encoder:
            x = conv(x)
            skips.append(x)
            x = downsample(x)
        # Apply the decoder
        for i, (upsample, conv) in enumerate(self.decoder):
            x = upsample(x)
            x = torch.cat((x, skips[-i - 1]), dim=1)
            x = conv(x)
        # Apply the output layers
        for layer in self.output_layers:
            x = layer(x)
        # Split the output by source, and remove the padding
        y, n = {}, 0
        for k, source_name in enumerate(batch.keys()):
            h, w = original_sizes[k]
            n_vars = self.n_variables[source_name]
            y[source_name] = x[:, n: n + n_vars, :h, :w]
            n += n_vars
        return y
