import math

import torch
from torch import nn


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding that handles both time (L) and table (N) dimensions
    """

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim

        # Create a standard 1D positional encoding for the time dimension
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's not a learnable parameter
        self.register_buffer('pe', pe.unsqueeze(0).unsqueeze(2))
        print('Created 2D positional encoding with shape:', self.pe.shape)

    def forward(self, x):
        """
        Forward pass for the PositionalEncoding2D module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, N, E).

        Returns:
            torch.Tensor: The same tensor with positional encoding added to the time dimension.
        """
        B, L, N, E = x.shape
        
        # Slice the positional embeddings to match L, then broadcast across B and N
        pos_encoding = self.pe[:, :L, :, :]

        # Add positional encoding to x
        return x + pos_encoding