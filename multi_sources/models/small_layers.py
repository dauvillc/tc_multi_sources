import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, values_dim, coords_dim, inner_dim, dropout=0.0):
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
                x['source_name'] contains the keys "dt", "embedded_dt", "embedded_coords",
                and "embedded_values".
        Returns:
            dict of str: tensor: Dictionary of outputs, such that
                outputs['source_name'] contains the predicted values of the tokens.
        """
        outputs = {}
        for source_name, data in x.items():
            values = data["embedded_values"]
            outputs[source_name] = self.net(values)
        return outputs
