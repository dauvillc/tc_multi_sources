import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from .utils import pair


class FeedForward(nn.Module):
    def __init__(
        self, values_dim, coords_dim, inner_dim, dropout=0.0, act_layer=nn.GELU, **kwargs
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(values_dim, inner_dim),
            act_layer(),
            nn.Dropout(dropout),
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


class SwiGLU(nn.Module):
    """SwiGLU mlp, adapted from the timm package."""

    def __init__(
        self,
        values_dim,
        coords_dim,
        inner_dim=None,
        act_layer=nn.SiLU,
        bias=True,
        dropout=0.0,
    ):
        super().__init__()
        bias = pair(bias)
        drop_probs = pair(dropout)

        self.fc1_g = nn.Linear(values_dim, inner_dim, bias=bias[0])
        self.fc1_v = nn.Linear(values_dim, inner_dim, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.LayerNorm(inner_dim)
        self.fc2 = nn.Linear(inner_dim, values_dim, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

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
            v_gate = self.fc1_g(values)
            v = self.fc1_v(values)
            v = self.act(v_gate) * v
            v = self.drop1(v)
            v = self.norm(v)
            v = self.fc2(v)
            v = self.drop2(v)
            outputs[source_name] = v
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
            spectral_norm(nn.Conv2d(style_size, 128, 3, 1, 1)), nn.ReLU(inplace=True)
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
        s = F.interpolate(s, size=(x.size(2), x.size(3)), mode="nearest")
        s = self.conv(s)
        return self.norm(x) * self.conv_gamma(s) + self.conv_beta(s)
