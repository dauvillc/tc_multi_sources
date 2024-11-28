import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm


class FeedForward(nn.Module):
    def __init__(self, values_dim, metadata_dim, inner_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(values_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, values_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        """
        Args:
            x (dict of str: dict of str: tensor): Dictionary of inputs, such that
                x['source_name'] contains at least the key "embedded_values". 
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs['source_name'] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, data in x.items():
            values = data["embedded_values"]
            outputs[source_name] = self.net(values)
        return outputs


class SPADE(nn.Module):
    """Spatially Adaptive Normalization (SPADE) layer - Park et al. (2019)."""
    def __init__(self, feature_size, style_size):
        """Args:
            feature_size (int): Number of features in the input tensor.
            style_size (int): Number of features in the style tensor.
        """
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(feature_size, affine=False)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(style_size, 128, 3, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv_gamma = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
    
    def forward(self, x, s):
        """Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            s (torch.Tensor): Style tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Normalized tensor of shape (B, C, H, W).
        """
        s = F.interpolate(s, size=(x.size(2), x.size(3)), mode='nearest')
        s = self.conv(s)
        return self.norm(x) * self.conv_gamma(s) + self.conv_beta(s)
