from torch import nn


class TableDecoder(nn.Module):
    def __init__(self, embed_dim, k_out):
        super().__init__()
        self.l1 = nn.Linear(embed_dim, embed_dim)
        self.l2 = nn.Linear(embed_dim, k_out)

    def forward(self, x):
        """
        x: (B, L, embed_dim)
        -> (B, L, k_out)
        """
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.l2(x)
        return x