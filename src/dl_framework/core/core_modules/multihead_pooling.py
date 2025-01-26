import torch
import torch.nn as nn


class MultiHeadPooling(nn.Module):
    """
    Pools a set of embeddings (B, S, E) into (B, pool_heads * E)
    using a separate learnable query for each 'head'.
    """

    def __init__(self, embed_dim, pool_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_heads = pool_heads

        self.queries = nn.Parameter(torch.randn(pool_heads, embed_dim))

        # Multi-head attention module (num_heads=1 to avoid splitting the embedding dimension)
        # We manually create heads by using separate queries
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,  # do not split the embedding dimension
            batch_first=True
        )

    def forward(self, x):
        """
        x : (B, S, E)
        Returns: (B, pool_heads * E)
        """
        B, S, E = x.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, pool_heads, E)

        out, _ = self.attn(q, x, x)  # out: (B, pool_heads, E)

        # Flatten the pool_heads dimension
        out = out.view(B, self.pool_heads * self.embed_dim)  # (B, pool_heads*E)

        return out
