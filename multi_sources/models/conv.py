import torch.nn as nn


class ResNet(nn.Module):
    """Implements a ResNet FCN model to place at the end of the VIT model,
    mostly to correct inconsistencies at the border of the patches.
    """

    def __init__(self, channels, inner_channels, n_blocks, kernel_size=3):
        """
        Args:
            channels (int): Number of input AND output channels.
            inner_channels (int): Number of channels in the intermediate layers.
            n_blocks (int): Number of residual blocks.
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        self.input_channels = channels
        # Create an input convolutional layer to match the number of channels
        self.input_layer = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(inner_channels),
        )
        self.layers = nn.ModuleList([])
        for i in range(n_blocks):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(inner_channels),
                    nn.Conv2d(
                        inner_channels, inner_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(inner_channels),
                    nn.Conv2d(
                        inner_channels, inner_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.GELU(),
                )
            )
            channels = inner_channels
        # Create an output convolutional layer for the regression task
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(inner_channels), nn.Conv2d(inner_channels, self.input_channels, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x) + x
        return self.output_layer(x)
