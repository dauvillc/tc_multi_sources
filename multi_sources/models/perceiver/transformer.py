import torch.nn as nn

from multi_sources.models.perceiver.self_att import ValuesCoordinatesSelfAttention
from multi_sources.models.perceiver.small_layers import MLP


class ValuesCoordinatesTransformerBlock(nn.Module):
    """Implements a block of the transformer part of the Perceiver model with
    two latent arrays instead of one: Lv for the values and Lc for the coordinates.
    A block is made up of a self-attention layer followed by an MLP.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        att_inner_ratio,
        num_heads,
        mlp_inner_ratio,
        mlp_hidden_layers,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            att_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the self-attention layer.
            num_heads (int): Number of heads in the attention block.
            mlp_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the MLP.
            mlp_hidden_layers (int): Number of hidden layers in the MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention = ValuesCoordinatesSelfAttention(
            values_dim, coords_dim, att_inner_ratio, num_heads, dropout
        )
        self.values_mlp = nn.Sequential(
            nn.LayerNorm(values_dim), MLP(values_dim, mlp_hidden_layers, mlp_inner_ratio, dropout)
        )
        self.coords_mlp = nn.Sequential(
            nn.LayerNorm(coords_dim), MLP(coords_dim, mlp_hidden_layers, mlp_inner_ratio, dropout)
        )

    def forward(self, V, C):
        """
        Args:
            V (tensor): Values tensor of shape (batch_size, num_values, values_dim).
            C (tensor): Coordinates tensor of shape (batch_size, num_values, coords_dim).

        Returns:
            tensor: Updated values tensor of shape (batch_size, num_values, values_dim).
            tensor: Updated coordinates tensor of shape (batch_size, num_values, coords_dim).
        """
        # Self-attention with residual connection.
        V_sa, C_sa = self.self_attention(V, C)
        V = V + V_sa
        C = C + C_sa

        # MLPs with residual connection too.
        V = self.values_mlp(V) + V
        C = self.coords_mlp(C) + V

        return V, C


class ValuesCoordinatesTransformer(nn.Module):
    """Implements the transformer part of the Perceiver model with two latent arrays
    instead of one: Lv for the values and Lc for the coordinates. The transformer is
    made up of multiple transformer blocks.
    """

    def __init__(
        self,
        values_dim,
        coords_dim,
        num_blocks,
        att_inner_ratio,
        num_heads,
        mlp_inner_ratio,
        mlp_layers,
        dropout=0.0,
    ):
        """
        Args:
            values_dim (int): Embedding dimension of the values.
            coords_dim (int): Embedding dimension of the coordinates.
            num_blocks (int): Number of transformer blocks.
            att_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the self-attention layer.
            num_heads (int): Number of heads in the attention block.
            mlp_inner_ratio (float): Ratio of the inner dimension to the input dimension in
                the MLP.
            mlp_layers (int): Number of layers in the MLP.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ValuesCoordinatesTransformerBlock(
                    values_dim,
                    coords_dim,
                    att_inner_ratio,
                    num_heads,
                    mlp_inner_ratio,
                    mlp_layers,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, V, C):
        """
        Args:
            V (tensor): Values tensor of shape (batch_size, num_values, values_dim).
            C (tensor): Coordinates tensor of shape (batch_size, num_values, coords_dim).

        Returns:
            tensor: Updated values tensor of shape (batch_size, num_values, values_dim).
            tensor: Updated coordinates tensor of shape (batch_size, num_values, coords_dim).
        """
        for block in self.blocks:
            V, C = block(V, C)

        return V, C
