# src/main_experiment.py
import os
import logging
import torch
import pandas as pd
import numpy as np
import warnings
import math  # Add this import

from sklearn.preprocessing import MinMaxScaler

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# src/main_experiment.py
from .data_processing import process_all_datasets
from .datasets import CFDDataset
from .models import (
    PressurePINN, UVelocityPINN, VVelocityPINN, WVelocityPINN,
    TauXPINN, TauYPINN, TauZPINN
)
from .train import train_pinn, initialize_optimizer_scheduler
from .evaluate import evaluate_pinn
from .visualize import plot_loss_curves, plot_pressure_and_wss_magnitude_distribution, plot_wall_shear_3d_distribution
from .utils import Config, setup_logging, EarlyStopping, get_device, fit_scalers

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        # ...existing code...
        self.scaler_columns = {
            "features": ["x", "y", "z", "t"],
            "time": ["t"],
            "pressure": ["pressure"],
            "velocity_u": ["velocity_u"],
            "velocity_v": ["velocity_v"],
            "velocity_w": ["velocity_w"],
            "wall_shear_x": ["wall_shear_x"],
            "wall_shear_y": ["wall_shear_y"],
            "wall_shear_z": ["wall_shear_z"],
        }
        # ...existing code...

def main():
    # Initialize global configuration
    config = Config()
    config.device = get_device()

    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize logger
    logger = setup_logging("main", config)

    # Print device information
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {config.device}")
    else:
        logger.info("Using CPU.")

    # Create necessary directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    os.makedirs(config.metrics_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.processed_data_dir, exist_ok=True)

    # Define file paths for all datasets
    file_paths = {
        "0021_diastolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0021 Diastolic aneurysm.csv"),
        "0021_systolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0021 Systolic aneurysm.csv"),
        "0021_diastolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0021 Diastolic global.csv"),
        "0021_systolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0021 systolic global.csv"),
        "0022_systolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0022 systolic aneurysm.csv"),
        "0022_diastolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0022 diastolic aneurysm.csv"),
        "0022_systolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0022 systolic global.csv"),
        "0022_diastolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0022 diastolic global.csv"),
        "0023_diastolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0023 diastolic global.csv"),
        "0023_systolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0023 systolic aneurysm.csv"),
        "0023_diastolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0023 diastolic aneurysm.csv"),
        "0023_systolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "systolic global 0023.csv"),
        "0024_systolic": os.path.join(config.data_dir, "raw", "WSS_data", "0024 systolic.csv"),
        "0024_diastolic": os.path.join(config.data_dir, "raw", "WSS_data", "0024 diastolic.csv"),
        "0025_diastolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0025 diastolic aneurysm.csv"),
        "0025_diastolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0025 diastolic global.csv"),
        "0025_systolic_aneurysm": os.path.join(config.data_dir, "raw", "WSS_data", "0025 systolic aneurysm.csv"),
        "0025_systolic_global": os.path.join(config.data_dir, "raw", "WSS_data", "0025 systolic global.csv"),
        "0142_systolic": os.path.join(config.data_dir, "raw", "WSS_data", "0142 systolic.csv"),
        "0142_diastolic": os.path.join(config.data_dir, "raw", "WSS_data", "0142 diastolic.csv"),
        # Add any additional datasets here
    }

    # Process all datasets
    processed_data = process_all_datasets(file_paths, processed_dir=config.processed_data_dir)

    # Iterate over each dataset and run the experiment
    for dataset_name, df in processed_data.items():
        run_id = dataset_name
        config.run_id = run_id

        # Initialize logging for this run
        logger = setup_logging(run_id, config)

        # Fit scalers on the data using helper function
        logger.info("Fitting scalers on the data.")
        scalers = fit_scalers(df, config.scaler_columns)
        config.scalers = scalers

        # Initialize dataset and dataloader
        dataset = CFDDataset(df, config.scalers, config.scaler_columns)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        logger.info(f"Dataset and DataLoader created with {len(dataset)} samples.")

        # Initialize models using helper function
        models = initialize_models(config, logger)

        # Move models to the selected device
        for model in models.values():
            model.to(config.device)

        # Setup optimizer and scheduler
        optimizer, scheduler = initialize_optimizer_scheduler(models, config, logger)

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            logger=logger,
        )

        # Train the models
        loss_history = train_pinn(
            models, dataloader, config, optimizer, scheduler, early_stopping, logger, run_id
        )

        # Evaluate the models
        r2_scores, nrmse_scores, mae_scores, total_mae = evaluate_pinn(
            models, dataloader, config, logger, run_id
        )

        # Generate Visualizations
        plot_pressure_and_wss_magnitude_distribution(
            dataset, models, config, logger, run_id
        )
        plot_loss_curves(loss_history, config, logger, run_id)
        

        logger.info(f"Completed experiment for dataset '{run_id}'.")

    # Final logging for all experiments
    final_logger = setup_logging("final", config)
    final_logger.info("All experiments completed successfully.")

if __name__ == "__main__":
    main()
