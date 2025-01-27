import os

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src.dl_framework.core.losses import masked_mse_loss, weighted_masked_mse_loss
from src.dl_framework.core.models.insee_econ_model import InseeEconModel
from src.dl_framework.data_pipeline.datasets.insee_collator import econ_collate_fn
from src.dl_framework.data_pipeline.datasets.insee_dataset import InseeDataset
from src.dl_framework.schedulers.schedulers_config import SchedulersConfig
from src.dl_framework.schedulers.table_importance_schedulers import TableImportanceScheduler
from src.dl_framework.schedulers.time_importance_scheduler import TimeImportanceScheduler


def train_econ_model(train_dataset: InseeDataset,
                     test_dataset: InseeDataset,
                     model: InseeEconModel,
                     scheduler_config: SchedulersConfig,
                     epochs=5,
                     batch_size=16,
                     lr=6e-4,
                     seed=42):
    tables_of_interest = ["CHOMAGE-TRIM-NATIONAL", "CNA-2020-PIB", "TAUX-CHOMAGE"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_tables = len(train_dataset.table_names)

    best_test_loss = float("inf")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=econ_collate_fn,
        pin_memory=True,
        num_workers=2 if batch_size >= 8 else 0,
        prefetch_factor=2 if batch_size >= 8 else 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=econ_collate_fn,
        pin_memory=True,
        num_workers=2 if batch_size >= 8 else 0,
        prefetch_factor=2 if batch_size >= 8 else 0
    )

    # Get p3 schedule
    p3_schedule = np.geomspace(scheduler_config.p3_min_value, scheduler_config.p3_max_value, epochs)

    # Get table schedules
    table_importance_scheduler = TableImportanceScheduler(epochs)
    for tn in tables_of_interest:
        table_importance_scheduler.add_linear_schedule(tn, scheduler_config.table_min_importance, scheduler_config.table_max_importance)

    # Get time schedules
    time_importance_scheduler = TimeImportanceScheduler(
        num_steps=epochs,
        decay_frequency="yearly",
        initial_decay=scheduler_config.time_min_decay,
        final_decay=scheduler_config.time_max_decay
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * 3

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    global_step = len(train_loader)
    # model = torch.compile(model, dynamic=True)

    # Training loop
    for ep in range(epochs):
        # Update p3
        train_loader.dataset.p_3_last1yr = p3_schedule[ep]
        train_loader.dataset.normalize_probabilities()
        table_importance = table_importance_scheduler.get_tables_importance()
        total_table_importance = sum(table_importance.values()) + num_tables - len(tables_of_interest)
        normalization_ratio = num_tables / total_table_importance

        time_importance = torch.tensor(
            time_importance_scheduler.get_time_importance(window_length=train_dataset.max_window_length_months),
            requires_grad=False).to(device)

        model.train()
        total_train_loss = 0.0
        for b_idx, batch_data in tqdm(enumerate(train_loader), desc=f"Train epoch {ep}", total=len(train_loader)):
            for tn in batch_data["full_data"]:
                batch_data["full_data"][tn] = batch_data["full_data"][tn].to(device, non_blocking=True)
            batch_data["mask"] = batch_data["mask"].to(device, non_blocking=True)
            batch_data["padding_mask"] = batch_data["padding_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_data)  # dict {tn -> (B, L, k_i)}

            # Compute loss
            losses = []
            padding_mask = batch_data["padding_mask"][:, :, 0].unsqueeze(-1)  # (B, L, N) => (B, L, 1)
            for tn in train_dataset.table_names:
                pred = outputs[tn]
                tgt = batch_data["full_data"][tn][:, :, 0::3]  # (B, L, 3*k_i) => (B, L, k_i)
                expected_missing_mask = batch_data["full_data"][tn][:, :, 1::3] == 1.0  # (B, L, k_i)
                true_missing_mask = batch_data["full_data"][tn][:, :, 2::3] == 1.0  # (B, L, k_i)
                valid_mask = ~(expected_missing_mask | true_missing_mask | padding_mask)  # (B, L, k_i)

                # Apply time importance
                table_loss = weighted_masked_mse_loss(
                    pred,
                    tgt,
                    valid_mask,
                    l_weights=time_importance[-tgt.shape[1]:])

                # Apply table importance
                if tn in table_importance:
                    table_loss = table_loss * table_importance[tn]

                losses.append(table_loss)

            loss_val = torch.stack(losses).mean() * normalization_ratio
            global_step += 1

            loss_val.backward()
            optimizer.step()
            scheduler.step()

            # Update running and total loss
            total_train_loss += loss_val.item()

        # Update schedulers
        table_importance_scheduler.step()
        time_importance_scheduler.step()

        avg_train_loss = total_train_loss / (b_idx + 1)

        # Evaluate
        model.eval()
        total_test_loss = 0.0
        loss_table_of_interest = [0.0 for _ in tables_of_interest]
        end_loss_table_of_interest = [0.0 for _ in tables_of_interest]
        with torch.no_grad():
            for b_idx, batch_data in enumerate(test_loader):
                for tn in batch_data["full_data"]:
                    batch_data["full_data"][tn] = batch_data["full_data"][tn].to(device, non_blocking=True)
                batch_data["mask"] = batch_data["mask"].to(device, non_blocking=True)
                batch_data["padding_mask"] = batch_data["padding_mask"].to(device, non_blocking=True)

                outputs = model(batch_data)
                losses = []
                padding_mask = batch_data["padding_mask"][:, :, 0].unsqueeze(-1)
                for tn in test_dataset.table_names:
                    pred = outputs[tn]
                    tgt = batch_data["full_data"][tn][:, :, 0::3]
                    expected_missing_mask = batch_data["full_data"][tn][:, :, 1::3] == 1.0
                    true_missing_mask = batch_data["full_data"][tn][:, :, 2::3] == 1.0
                    valid_mask = ~(expected_missing_mask | true_missing_mask | padding_mask)

                    # Compute the loss on the last year and the whole set
                    valid_mask_end = valid_mask.clone()
                    valid_mask_end[:, :-12, :] = False

                    table_loss = weighted_masked_mse_loss(
                        pred,
                        tgt,
                        valid_mask)

                    table_loss_end = weighted_masked_mse_loss(
                        pred,
                        tgt,
                        valid_mask_end)

                    # Apply table importance
                    if tn in table_importance:
                        table_loss = table_loss * table_importance[tn]
                        table_loss_end = table_loss_end * table_importance[tn]

                    losses.append(table_loss)
                    if tn in tables_of_interest:
                        loss_table_of_interest[tables_of_interest.index(tn)] += table_loss.item() / table_importance[tn]
                        end_loss_table_of_interest[tables_of_interest.index(tn)] += table_loss_end.item() / table_importance[tn]

                loss_val = torch.stack(losses).mean() * normalization_ratio
                total_test_loss += loss_val.item()

        avg_test_loss = total_test_loss / len(test_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # Update the best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), "best_model.pth")

        data_to_log = {
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "epoch": ep,
            "lr": current_lr
        }

        # Log the loss of tables of interest
        loss_table_of_interest = [l / len(test_loader) for l in loss_table_of_interest]
        for i, tn in enumerate(tables_of_interest):
            data_to_log[f"{tn}_test_loss"] = loss_table_of_interest[i]
            data_to_log[f"{tn}_inference_test_loss"] = end_loss_table_of_interest[i]

        # Log metrics to wandb
        wandb.log(data_to_log, step=global_step)

    # Log best model and best test loss to wandb
    model_name = "best_model"

    artifact = wandb.Artifact(model_name, type="model", metadata={"best_test_loss": best_test_loss})
    artifact.add_file("best_model.pth")
    wandb.log_artifact(artifact)
    wandb.log({"best_test_loss": best_test_loss})

    # Delete the best model file
    os.remove("best_model.pth")

    return model
