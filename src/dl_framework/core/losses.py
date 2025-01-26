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