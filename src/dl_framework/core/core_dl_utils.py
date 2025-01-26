import torch


class CoreDlUtils:
    @staticmethod
    def key_padding_mask_to_attention_mask(key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a key_padding_mask of shape (B, S) to an attention_mask of shape (B, S, S).

        Args:
            key_padding_mask: Boolean tensor of shape (B, S) where True indicates padding tokens
                             and False indicates actual tokens.

        Returns:
            attention_mask: Boolean tensor of shape (B, S, S) where False indicates allowed
                           attention and True indicates masked (blocked) attention.
        """
        batch_size, seq_len = key_padding_mask.size()

        # First, we need to convert the key_padding_mask to the right shape
        # We want each position to not attend to padding tokens
        # So we expand the key_padding_mask to (B, 1, S) and broadcast it to (B, S, S)
        # Make it contiguous to ensure the memory layout allows setting elements
        expanded_mask = key_padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len).contiguous()

        return expanded_mask

    @staticmethod
    def fix_fully_masked_rows(attn_mask_3d: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        For any row `b` where `key_padding_mask[b]` is all `True` (i.e., fully masked),
        replace that entire `(L, L)` block in `attn_mask_3d` with `~torch.eye(L)`.
        This makes each token attend only to itself, preventing NaNs.

        Args:
            attn_mask_3d (torch.Tensor): A 3D attention mask tensor of shape `(B, L, L)`.
            key_padding_mask (torch.Tensor): A boolean tensor of shape `(B, L)` where `True` indicates padding tokens.

        Returns:
            torch.Tensor: The modified attention mask tensor with fully masked rows fixed.
        """
        B, L, _ = attn_mask_3d.shape
        fully_masked_rows = key_padding_mask.all(dim=1)  # shape (B,)
        attn_mask_3d = attn_mask_3d.clone()
        attn_mask_3d[fully_masked_rows] = ~torch.eye(L, L, dtype=torch.bool, device=attn_mask_3d.device,
                                                     requires_grad=False)
        return attn_mask_3d