import torch
from torch import nn

from src.dl_framework.core.core_modules.positional_encoding_2d import PositionalEncoding2D
from src.dl_framework.core.core_modules.transformer_layer import TransformerLayer2D


class Transformer2D(nn.Module):
    """
    Complete 2D transformer with multiple layers
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            ff_dim: int,
            dropout: float = 0.1,
            max_len: int = 5000,
            use_pos_encoding: bool = True
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding2D(embed_dim, max_len) if use_pos_encoding else None
        self.layers = nn.ModuleList([
            TransformerLayer2D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        Forward pass for the Transformer2D module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, N, E).
            padding_mask (torch.Tensor): Boolean tensor of shape (B, L, N). True values will be ignored in attention.

        Returns:
            torch.Tensor: Transformed tensor of shape (B, L, N, E).
        """
        # Optional positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        # Pass through all transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask)

        # Final layer norm
        x = self.final_norm(x)

        return x