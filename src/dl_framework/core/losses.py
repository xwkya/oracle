from typing import Optional

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.BoolTensor):
    """
    pred: (B, L, k_i)
    target: (B, L, k_i)
    mask: (B, L, k_i)  # 1 where ground truth is valid, 0 where no ground truth
    Returns average MSE over valid entries.
    """
    diff = (pred - target) ** 2
    diff = (diff * mask)

    valid_count = mask.sum()

    if valid_count > 0:
        return diff.sum() / valid_count
    else:
        return torch.tensor(0.0, device=pred.device)

def weighted_masked_mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.BoolTensor,
        k_weights: Optional[torch.Tensor] = None,
        l_weights: Optional[torch.Tensor] = None):
    """
    Args:
        pred (torch.Tensor): Predicted values of shape (B, L, k_i).
        target (torch.Tensor): Target values of shape (B, L, k_i).
        mask (torch.BoolTensor): Mask indicating valid ground truth entries of shape (B, L, k_i).
        k_weights (torch.Tensor): Weights for each entry along the k dimension. If None, no weights are applied. Defaults to None.
        l_weights (torch.Tensor): Weights for each entry along the l dimension. If None, no weights are applied. Defaults to None.

    Returns:
        torch.Tensor: Average MSE over valid entries.
    """
    diff = (pred - target) ** 2
    diff = diff * mask

    # Apply weights
    if k_weights is not None:
        normalization_ratio_k = k_weights.sum() / k_weights.shape[0]
        diff = (diff * k_weights.unsqueeze(0).unsqueeze(1)) / normalization_ratio_k
    if l_weights is not None:
        normalization_ratio_l = l_weights.sum() / l_weights.shape[0]
        diff = (diff * l_weights.unsqueeze(0).unsqueeze(-1)) / normalization_ratio_l

    valid_count = mask.sum()

    if valid_count > 0:
        return diff.sum() / valid_count
    else:
        return torch.tensor(0.0, device=pred.device)