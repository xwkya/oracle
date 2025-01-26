import os

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from src.dl_framework.core.losses import masked_mse_loss
from src.dl_framework.core.models.insee_econ_model import InseeEconModel
from src.dl_framework.data_pipeline.datasets.insee_collator import econ_collate_fn
from src.dl_framework.data_pipeline.datasets.insee_dataset import InseeDataset


def train_econ_model(train_dataset: InseeDataset,
                     test_dataset: InseeDataset,
                     epochs=5,
                     batch_size=8,
                     num_layers=4,
                     n_heads=8,
                     embed_dim=32,
                     lr=6e-4,
                     dropout=0.1,
                     seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

    model = InseeEconModel(
        table_names=train_dataset.table_names,
        table_shapes=train_dataset.table_shapes,
        embed_dim=embed_dim,
        n_heads=n_heads,
        ff_dim=256,
        num_layers=num_layers,
        dropout=dropout,
        use_pos_encoding=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * 3

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    global_step = len(train_loader)

    # Training loop
    for ep in range(epochs):
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
                losses.append(masked_mse_loss(pred, tgt, valid_mask))

            loss_val = torch.stack(losses).mean()
            global_step += 1

            loss_val.backward()
            optimizer.step()
            scheduler.step()

            # Update running and total loss
            total_train_loss += loss_val.item()

        avg_train_loss = total_train_loss / (b_idx + 1)

        # Evaluate
        model.eval()
        total_test_loss = 0.0
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
                    losses.append(masked_mse_loss(pred, tgt, valid_mask))

                loss_val = torch.stack(losses).mean()
                total_test_loss += loss_val.item()

        avg_test_loss = total_test_loss / len(test_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # Update the best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), "best_model.pth")


        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "epoch": ep,
            "lr": current_lr
        }, step=global_step)

    # Log best model to wandb
    artifact = wandb.Artifact("best_model", type="model", metadata={"best_test_loss": best_test_loss})
    artifact.add_file("best_model.pth")
    wandb.log_artifact(artifact)

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Delete the best model file
    os.remove("best_model.pth")

    return model

