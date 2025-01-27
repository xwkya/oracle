import os
import argparse
import sys
import yaml
import dotenv
import wandb
import torch

from datetime import datetime

from src.data_sources.data_source import DataSource
from src.dl_framework.core.models.insee_econ_model import InseeEconModel
from src.dl_framework.data_pipeline.data_processor import DataProcessor
from src.dl_framework.data_pipeline.datasets.insee_dataset import InseeDataset
from src.dl_framework.data_pipeline.scalers.trend_scaler import TrendRemovalConfig
from src.dl_framework.schedulers.schedulers_config import SchedulersConfig
from src.dl_framework.trainers.insee_trainer import train_econ_model
from src.logging_config import setup_logging

dotenv.load_dotenv()
torch.set_float32_matmul_precision('high')

def main():
    sys.path.append(os.getcwd())
    setup_logging()

    # --------------------------------------------------
    # Parse arguments with argparse
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="Train script with argparse + wandb config")

    # Add arguments for all the hyperparams
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument("--fine_tune_artifact", type=str, default=None, help="Fine-tune model with this artifact")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument("--pool_heads", type=int, default=4, help="Number of heads in the pooling layer. Decoder will have input size = embed_dim * pool_heads")
    parser.add_argument('--p1', type=float, default=0.3, help='Probability p1')
    parser.add_argument('--p2', type=float, default=0.3, help='Probability p2')
    parser.add_argument('--p3', type=float, default=0.3, help='Probability p3')
    parser.add_argument('--p4', type=float, default=0.3, help='Probability p4')
    parser.add_argument('--min_window', type=int, default=6, help='Minimum window length in years')
    parser.add_argument('--max_window', type=int, default=12, help='Maximum window length in years')
    parser.add_argument("--schedulers_config", type=str, default="configs/scheduler_parameters_dynamic.yaml", help="Path to the scheduler parameters yaml file.")

    args = parser.parse_args()

    # --------------------------------------------------
    # WANDB login (if key is available)
    # --------------------------------------------------
    wandb_key = os.getenv('WANDB_KEY')
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        print("Warning: WANDB_KEY not found in environment! Make sure you're logged in via CLI or environment vars.")

    # --------------------------------------------------
    # Initialize wandb run
    # --------------------------------------------------
    run = wandb.init(
        project="oracle-v1",
        reinit=True
    )

    config = wandb.config
    wandb.config.update(vars(args), allow_val_change=True) # Values will be locked during sweeps

    state_dict = None
    if args.fine_tune_artifact:
        # Download artifact and get the run config
        api = wandb.Api()
        artifact = api.artifact(args.fine_tune_artifact, type='model')
        artifact_dir = artifact.download()
        state_dict = torch.load(f"{artifact_dir}/best_model.pth", weights_only=False)
        config_dict = artifact.logged_by().config
        config = wandb.config

        # Ingest previous run config
        config.update(config_dict, allow_val_change=True)

        # Update with training hyperparameters
        config.update({
            "fine_tune_artifact": args.fine_tune_artifact,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "p1": args.p1,
            "p2": args.p2,
            "p3_max": args.p3_max,
            "p3": args.p3,
            "p4": args.p4,
            "lr": args.lr
        }, allow_val_change=True)


    # --------------------------------------------------
    # Extract final hyperparameters from the config
    # --------------------------------------------------
    epochs = config.epochs
    batch_size = config.batch_size
    num_layers = config.num_layers
    n_heads = config.n_heads
    embed_dim = config.embed_dim
    dropout = config.dropout
    p1 = config.p1
    p2 = config.p2
    p3 = config.p3
    p4 = config.p4
    min_window = config.min_window
    max_window = config.max_window
    pool_heads = config.pool_heads
    fine_tune_artifact = config.fine_tune_artifact
    lr = config.lr

    interpolate = False
    with open(args.schedulers_config, 'r') as file:
        config_dict = yaml.safe_load(file)
        schedulers_config = SchedulersConfig.from_dict(config_dict)

    # --------------------------------------------------
    # Create the dataset
    # --------------------------------------------------
    data_processor = DataProcessor(
        data_source=DataSource.INSEE,
        min_date=datetime(1970, 1, 1),
        max_date=datetime(2017, 1, 1),
        train_cutoff=datetime(2016, 1, 1)
    )

    config = TrendRemovalConfig(exponential_mse_scale=0.20, max_iter=None, inverse_exponential_mse_scale=0.64)

    if os.path.exists("data_processor.pkl"):
        data_processor = DataProcessor.load("data_processor.pkl")

    else:
        data_processor.add_range_scaler().add_trend_removal(config, processor_id='trend').add_scaler()
        data_processor.fit_from_provider()
        data_processor.save("data_processor.pkl")

    train_dataset = InseeDataset(
        data_processor,
        min_window_length_year=min_window,
        max_window_length_year=max_window,
        interpolate=interpolate,
        p_1_none=p1,
        p_2_uniform=p2,
        p_3_last1yr=p3,
        p_4_table=p4,
        number_of_samples=30_000,
        seed=42)

    test_dataset = InseeDataset(
        data_processor,
        min_window_length_year=min_window,
        max_window_length_year=max_window,
        interpolate=interpolate,
        p_1_none=p1,
        p_2_uniform=p2,
        p_3_last1yr=p3,
        p_4_table=p4,
        number_of_samples=6,
        seed=42,
        inference_mode=True)

    # Create the model
    model = InseeEconModel(
        table_names=train_dataset.table_names,
        table_shapes=train_dataset.table_shapes,
        embed_dim=embed_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_pos_encoding=True,
        pool_heads=pool_heads
    )

    if args.fine_tune_artifact:
        model.load_state_dict(state_dict)

    # --------------------------------------------------
    # Call the train function
    # --------------------------------------------------
    trained_model = train_econ_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        scheduler_config=schedulers_config,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    os.makedirs(f"models/{run.name}", exist_ok=True)
    torch.save(trained_model.state_dict(), f"models/{run.name}/best_model.pth")

    run.finish()


if __name__ == "__main__":
    main()
