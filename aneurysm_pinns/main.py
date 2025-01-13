# aneurysm_pinns/main.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from aneurysm_pinns.config import Config
from aneurysm_pinns.dataset import load_data, CFDDataset, process_and_save_all_datasets
from aneurysm_pinns.modeling.model import initialize_models
from aneurysm_pinns.modeling.train import train_pinn, EarlyStopping, setup_optimizer_scheduler
from aneurysm_pinns.modeling.predict import evaluate_pinn
from aneurysm_pinns.plots import (
    plot_loss_curves,
    plot_pressure_and_wss_magnitude_distribution,
    plot_wss_histogram,
    plot_time_varying_slices,
    plot_3d_representation
)
from aneurysm_pinns.utils import ensure_dir

def main():
    config = Config()
    

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.random_seed)
    print("Current device:", config.device)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    os.makedirs(config.metrics_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.processed_data_dir, exist_ok=True)

    # File paths for raw data
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

    # 1) Process & Save
    process_and_save_all_datasets(file_paths, config.processed_data_dir)

    # 2) Load processed data
    datasets = load_data(config)

    # 3) For each dataset, run an experiment
    for dataset_name, df in datasets.items():
        run_id = dataset_name
        config.run_id = run_id

        print(f"\n===== Starting experiment for '{run_id}' =====")

        # Fit scalers
        scalers = {}
        feature_cols = config.scaler_columns["features"]
        scalers["features"] = MinMaxScaler()
        scalers["features"].fit(df[feature_cols])

        scalers["time"] = MinMaxScaler()
        scalers["time"].fit(df[[config.scaler_columns["time"]]])

        other_vars = [
            "pressure",
            "velocity_u",
            "velocity_v",
            "velocity_w",
            "wall_shear_x",
            "wall_shear_y",
            "wall_shear_z"
        ]
        for var in other_vars:
            col = config.scaler_columns[var]
            sc = MinMaxScaler()
            sc.fit(df[[col]])
            scalers[var] = sc

        dataset = CFDDataset(df, scalers, config.scaler_columns)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        print(f"Dataset '{run_id}' with {len(dataset)} samples loaded.")

        # Initialize models
        models = initialize_models(config)
        # Setup optimizer
        optimizer, scheduler = setup_optimizer_scheduler(models, config)
        # Early stopping
        early_stopper = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )

        # Train
        loss_history = train_pinn(
            models, dataloader, config,
            optimizer, scheduler,
            early_stopper, run_id
        )

        # Evaluate
        metrics = evaluate_pinn(models, dataloader, dataset, config, run_id)

        # Visualization
        plot_loss_curves(loss_history, config, run_id, dataset_name)
        plot_pressure_and_wss_magnitude_distribution(dataset, models, config, run_id)
        plot_wss_histogram(dataset, models, config, run_id)

        # **New**: Example usage
        plot_time_varying_slices(dataset, models, config, run_id, variable="pressure", num_times=5)
        plot_3d_representation(dataset, models, config, run_id, variable="pressure", time_value=0.01)

        # Save final model
        final_model_path = os.path.join(config.model_dir, run_id, f"final_model_{run_id}.pt")
        ensure_dir(os.path.dirname(final_model_path))
        ckpt = {}
        for key, model in models.items():
            ckpt[f"{key}_state_dict"] = model.state_dict()
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
        ckpt["loss_history"] = loss_history

        torch.save(ckpt, final_model_path)
        print(f"Saved final model to '{final_model_path}'.")

        print(f"===== Finished experiment for '{run_id}' =====")

    print("All experiments done.")

if __name__ == "__main__":
    main()
