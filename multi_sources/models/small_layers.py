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

    def forward(self, pixels_seq, coords_seq, **kwargs):
        """
        Args:
            pixels (list of torch.Tensor): Embedded sequence of tokens from the pixels,
                for each source. Each tensor should have shape (bs, n_tokens, pixel_dim).
            coords (list of torch.Tensor): Embedded sequence of tokens from the coordinates,
                for each source.
                Each tensor should have shape (bs, n_tokens, coords_dim).
        Returns:
            list of torch.Tensor: The output of the FFN for each source.
        """
        return [self.net(pixels) for pixels in pixels_seq]
