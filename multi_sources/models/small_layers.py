import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, pixels_dim, coords_dim, inner_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(pixels_dim),
            nn.Linear(pixels_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, pixels_dim),
            nn.Dropout(dropout),
        )

    def forward(self, pixels_seq, coords_seq):
        return self.net(pixels_seq)
