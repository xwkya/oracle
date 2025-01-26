import torch
import torch.nn as nn
from src.dl_framework.core.core_dl_utils import CoreDlUtils

class DoubleAttention(nn.Module):
    """
    Perform two-step attention on data of shape (B, L, N, E):
    1) Attention over N dimension (tables)
    2) Attention over L dimension (time)
    """

    def __init__(self, embed_dim, num_heads):
        """
        Initialize the DoubleAttention module.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
        """
        super(DoubleAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attnN = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # across N
        self.attnL = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # across L

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        Forward pass for the DoubleAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, N, E).
            padding_mask (torch.Tensor): Boolean tensor of shape (B, L, N). True values will be ignored in attention.

        Returns:
            torch.Tensor: Output tensor of shape (B, L, N, E) after double attention.
        """
        B, L, N, E = x.shape

        # Clone padding mask to prevent any modification from affecting its reference
        padding_mask = padding_mask.clone()

        # -- Attention across N --
        # Flatten into (B*L, N, E)
        x_reshape = x.view(B * L, N, E).contiguous()
        padding_mask = padding_mask.view(B * L, N)
        # -----------------
        # This is where some issues can be introduced. We padded sequences to have same length.
        # We do attention across all clues for all pairs of month and elements in the batch.
        # /!\ We transform the padding mask into an attention mask and let padded elements attend to themselves.
        # /!\ We zero-out the attention scores for sequences that are entirely padded.
        # -----------------
        attn_mask = CoreDlUtils.key_padding_mask_to_attention_mask(padding_mask)
        attn_mask = CoreDlUtils.fix_fully_masked_rows(attn_mask, padding_mask)  # (B*L, N, N)

        # Expand the mask for the number of heads
        attn_mask = attn_mask.unsqueeze(1).expand(B * L, self.num_heads, N, N).reshape(B * L * self.num_heads, N,
                                                                                       N).contiguous()

        # Multi-head attention across N
        attn_outN, _ = self.attnN(x_reshape, x_reshape, x_reshape, attn_mask=attn_mask)  # (B*L, N, E)
        attn_outN = attn_outN.masked_fill(padding_mask.all(dim=1).view(-1, 1, 1), 0.0)

        # Reshape back to (B, L, N, E)
        xN = attn_outN.view(B, L, N, E)
        padding_mask = padding_mask.view(B, L, N)

        # -- Attention across L --
        # Permute to (B, N, L, E) and (B, N, L)
        xN = xN.permute(0, 2, 1, 3).contiguous()
        padding_mask = padding_mask.permute(0, 2, 1).contiguous()

        # Flatten into (B*N, L, E)
        xN_reshape = xN.view(B * N, L, E)
        padding_mask = padding_mask.view(B * N, L)

        # Multi-head attention across L
        attn_outL, _ = self.attnL(xN_reshape, xN_reshape, xN_reshape, key_padding_mask=padding_mask)

        # Reshape back to (B, N, L, E), we discard padding_mask
        xL = attn_outL.view(B, N, L, E)

        # Permute back to (B, L, N, E)
        out = xL.permute(0, 2, 1, 3).contiguous()

        return out