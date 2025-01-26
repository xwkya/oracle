import torch
import numpy as np
import time


def econ_collate_fn(batch):
    """
    Collate function for **EconDataset**. Takes a list of samples from the dataset and collates them into a batch.

    :param batch: List of size B, each element is a dict:
        {
          "full_data": { table_name -> np.ndarray (L, 3*k_i) },
          "mask": np.ndarray (L, num_tables)
        }

    :returns: A dict
        {
          "full_data": { table_name -> FloatTensor of shape (B, L_max, 3*k_i) },
          "mask": BoolTensor of shape (B, L_max, N),
          "padding_mask": BoolTensor of shape (B, L_max, N),
        }
    """

    # Get info about the batch
    batch_size = len(batch)
    lengths = [item["mask"].shape[0] for item in batch]  # each item["mask"] is (L, num_tables)
    L_max = max(lengths)  # maximum L among the batch
    num_tables = batch[0]["mask"].shape[1]

    # Build boolean masks
    mask_tensor = torch.zeros((batch_size, L_max, num_tables), dtype=torch.bool)
    padding_mask = torch.ones((batch_size, L_max, num_tables), dtype=torch.bool)

    # Fill them
    for b, item in enumerate(batch):
        L = item['mask'].shape[0]
        mask_tensor[b, :L, :] = torch.from_numpy(item['mask']).bool()
        padding_mask[b, :L, :] = False

    # Build full_data dict
    table_names = list(batch[0]["full_data"].keys())
    full_data_dict = {}

    # For each table, we figure out its feature dimension k_i by looking at the first item
    for tn in table_names:
        _, k_i = batch[0]["full_data"][tn].shape

        # Allocate a PyTorch tensor (B, L_max, k_i) for this table, filled with 0.0 (pad value)
        data_np = np.empty((batch_size, L_max, k_i), dtype=np.float32)
        data_np.fill(0.0)

        # Fill it with each item’s data
        for b, item in enumerate(batch):
            arr_np = item["full_data"][tn]  # shape = (L, 3*k_i)
            L = arr_np.shape[0]
            data_np[b, :L, :] = arr_np

        data_tensor = torch.from_numpy(data_np)

        # Store this in our dictionary
        full_data_dict[tn] = data_tensor

    # Construct the final collated batch output
    batch_output = {
        "full_data": full_data_dict,  # {table_name -> FloatTensor (B, L_max, k_i)}
        "mask": mask_tensor,  # BoolTensor (B, L_max, num_tables)
        "padding_mask": padding_mask,  # BoolTensor (B, L_max, num_tables)
    }

    return batch_output
