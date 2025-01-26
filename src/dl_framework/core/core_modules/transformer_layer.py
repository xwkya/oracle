import torch.nn as nn

from src.dl_framework.core.core_modules.double_attention import DoubleAttention


class FeedForward(nn.Module):
    """
    Standard transformer feed-forward network with GELU activation
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer2D(nn.Module):
    """
    A single transformer layer using DoubleAttention
    Uses pre-norm architecture (norm before attention/FFN)
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.attention = DoubleAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

        # Layer norms before attention and FF
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout for attention output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # Pre-norm architecture
        # Attention block
        normed_x = self.norm1(x)
        attn_out = self.attention(normed_x, padding_mask)
        x = x + self.dropout(attn_out)

        # FFN block
        normed_x = self.norm2(x)
        ff_out = self.ff(normed_x)
        x = x + ff_out

        return x