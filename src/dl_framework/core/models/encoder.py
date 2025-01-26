from torch import nn
import torch.nn.functional as F


class TableEncoder(nn.Module):
    def __init__(self, k_in, embed_dim):
        super().__init__()
        self.l1 = nn.Linear(k_in, embed_dim)
        self.l2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, L, k_in)
        returns: (B, L, embed_dim)
        """
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
