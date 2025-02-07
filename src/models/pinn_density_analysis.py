import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from aneurysm_pinns.dataset import load_data_with_metadata
from src.models.full_pinn_experiment import (
    Config, initialize_models, setup_optimizer_scheduler,
    compute_physics_loss, compute_boundary_loss,
    compute_data_loss, compute_inlet_loss
)

class DenseDataset:
    """Dataset class that ensures tensors require gradients"""
    def __init__(self, df: pd.DataFrame, scalers: dict, scaler_columns: dict):
        self.scalers = scalers
        self.scaler_columns = scaler_columns
        self.data = df.copy()
        self.normalize_data()
        self.prepare_tensors()

    def normalize_data(self):
        """Normalize the data using the provided scalers"""
        feature_cols = self.scaler_columns["features"]
        self.data[feature_cols] = self.scalers["features"].transform(self.data[feature_cols])
        
        self.data[[self.scaler_columns["time"]]] = self.scalers["time"].transform(
            self.data[[self.scaler_columns["time"]]]
        )
        
        other_vars = [
            "pressure",
            "velocity_u",
            "velocity_v",
            "velocity_w",
            "wall_shear_x",
            "wall_shear_y",
            "wall_shear_z",
        ]
        for var in other_vars:
            col = self.scaler_columns[var]
            self.data[[col]] = self.scalers[var].transform(self.data[[col]])

    def prepare_tensors(self):
        """Convert data to tensors with gradients enabled"""
        # Spatial coordinates and time with gradients
        self.x = torch.tensor(self.data[self.scaler_columns["features"][0]].values,
                            dtype=torch.float32, requires_grad=True).unsqueeze(1)
        self.y = torch.tensor(self.data[self.scaler_columns["features"][1]].values,
                            dtype=torch.float32, requires_grad=True).unsqueeze(1)
        self.z = torch.tensor(self.data[self.scaler_columns["features"][2]].values,
                            dtype=torch.float32, requires_grad=True).unsqueeze(1)
        self.t = torch.tensor(self.data[self.scaler_columns["time"]].values,
                            dtype=torch.float32, requires_grad=True).unsqueeze(1)
        
        # Target variables
        self.p = torch.tensor(self.data[self.scaler_columns["pressure"]].values,
                            dtype=torch.float32).unsqueeze(1)
        self.u = torch.tensor(self.data[self.scaler_columns["velocity_u"]].values,
                            dtype=torch.float32).unsqueeze(1)
        self.v = torch.tensor(self.data[self.scaler_columns["velocity_v"]].values,
                            dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(self.data[self.scaler_columns["velocity_w"]].values,
                            dtype=torch.float32).unsqueeze(1)
        self.tau_x = torch.tensor(self.data[self.scaler_columns["wall_shear_x"]].values,
                                dtype=torch.float32).unsqueeze(1)
        self.tau_y = torch.tensor(self.data[self.scaler_columns["wall_shear_y"]].values,
                                dtype=torch.float32).unsqueeze(1)
        self.tau_z = torch.tensor(self.data[self.scaler_columns["wall_shear_z"]].values,
                                dtype=torch.float32).unsqueeze(1)
        
        # Boundary condition mask
        epsilon = 1e-5
        self.is_boundary = (
            (torch.abs(self.u) < epsilon) &
            (torch.abs(self.v) < epsilon) &
            (torch.abs(self.w) < epsilon)
        ).squeeze()

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger for the analysis"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Also log to file
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/pinn_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def measure_memory_usage() -> float:
    """Measure current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert bytes to GB

def initialize_scalers(df: pd.DataFrame, config: Config) -> dict:
    """Initialize and fit scalers for the dataset"""
    scalers = {}
    
    # Fit scaler for features
    feature_cols = config.scaler_columns["features"]
    scalers["features"] = MinMaxScaler()
    scalers["features"].fit(df[feature_cols])
    
    # Fit scaler for time
    scalers["time"] = MinMaxScaler()
    scalers["time"].fit(df[[config.scaler_columns["time"]]])
    
    # Fit scalers for other variables
    other_vars = [
        "pressure",
        "velocity_u",
        "velocity_v",
        "velocity_w",
        "wall_shear_x",
        "wall_shear_y",
        "wall_shear_z",
    ]
    
    for var in other_vars:
        col = config.scaler_columns[var]
        scalers[var] = MinMaxScaler()
        scalers[var].fit(df[[col]])
    
    return scalers

def train_and_time_pinn(config: Config, dataset: DenseDataset, logger: logging.Logger):
    """Train PINN and measure training time and memory usage"""
    start_time = time.time()
    start_memory = measure_memory_usage()
    
    # Move data to device
    dataset.x = dataset.x.to(config.device)
    dataset.y = dataset.y.to(config.device)
    dataset.z = dataset.z.to(config.device)
    dataset.t = dataset.t.to(config.device)
    dataset.p = dataset.p.to(config.device)
    dataset.u = dataset.u.to(config.device)
    dataset.v = dataset.v.to(config.device)
    dataset.w = dataset.w.to(config.device)
    dataset.tau_x = dataset.tau_x.to(config.device)
    dataset.tau_y = dataset.tau_y.to(config.device)
    dataset.tau_z = dataset.tau_z.to(config.device)
    dataset.is_boundary = dataset.is_boundary.to(config.device)
    
    # Initialize models and optimizer
    models = initialize_models(config, logger)
    optimizer, scheduler = setup_optimizer_scheduler(models, config, logger)
    
    # Training loop with timing for each epoch
    epoch_times = []
    loss_history = {"total": [], "physics": [], "boundary": [], "data": [], "inlet": []}
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # Set models to training mode
        for m in models.values():
            m.train()
            
        # Single training step
        optimizer.zero_grad()
        
        # Forward pass
        p_pred = models["p"](dataset.x, dataset.y, dataset.z, dataset.t)
        u_pred = models["u"](dataset.x, dataset.y, dataset.z, dataset.t)
        v_pred = models["v"](dataset.x, dataset.y, dataset.z, dataset.t)
        w_pred = models["w"](dataset.x, dataset.y, dataset.z, dataset.t)
        tau_x_pred = models["tau_x"](dataset.x, dataset.y, dataset.z, dataset.t)
        tau_y_pred = models["tau_y"](dataset.x, dataset.y, dataset.z, dataset.t)
        tau_z_pred = models["tau_z"](dataset.x, dataset.y, dataset.z, dataset.t)
        
        # Compute losses
        physics_loss = compute_physics_loss(
            p_pred, u_pred, v_pred, w_pred,
            dataset.x, dataset.y, dataset.z, dataset.t,
            config.rho, config.mu
        )
        
        boundary_loss = compute_boundary_loss(
            u_pred[dataset.is_boundary],
            v_pred[dataset.is_boundary],
            w_pred[dataset.is_boundary]
        )
        
        data_loss = compute_data_loss(
            p_pred, dataset.p,
            u_pred, dataset.u,
            v_pred, dataset.v,
            w_pred, dataset.w,
            tau_x_pred, dataset.tau_x,
            tau_y_pred, dataset.tau_y,
            tau_z_pred, dataset.tau_z
        )
        
        inlet_loss = compute_inlet_loss(u_pred, v_pred, w_pred, dataset.t)
        
        # Total loss with adaptive weights
        total_loss = (
            torch.exp(models["p"].log_lambda_physics) * physics_loss +
            torch.exp(models["p"].log_lambda_boundary) * boundary_loss +
            torch.exp(models["p"].log_lambda_data) * data_loss +
            torch.exp(models["p"].log_lambda_inlet) * inlet_loss
        )
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record timing and losses
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        
        loss_history["total"].append(total_loss.item())
        loss_history["physics"].append(physics_loss.item())
        loss_history["boundary"].append(boundary_loss.item())
        loss_history["data"].append(data_loss.item())
        loss_history["inlet"].append(inlet_loss.item())
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{config.epochs} - "
                       f"Loss: {total_loss.item():.6f}, "
                       f"Time: {epoch_times[-1]:.3f}s")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    training_stats = {
        "total_time": end_time - start_time,
        "avg_epoch_time": np.mean(epoch_times),
        "memory_used": end_memory - start_memory,
        "final_loss": total_loss.item(),
        "loss_history": loss_history
    }
    
    return models, training_stats

def generate_dense_predictions(models: dict, original_data: pd.DataFrame, density_factor: int,
                             config: Config, logger: logging.Logger):
    """Generate predictions on a denser grid and measure performance"""
    # Create denser grid
    x_min, x_max = original_data['X [ m ]'].min(), original_data['X [ m ]'].max()
    y_min, y_max = original_data['Y [ m ]'].min(), original_data['Y [ m ]'].max()
    z_min, z_max = original_data['Z [ m ]'].min(), original_data['Z [ m ]'].max()
    t_min, t_max = original_data['Time [ s ]'].min(), original_data['Time [ s ]'].max()
    
    n_points = len(original_data)
    n_points_per_dim = int(np.ceil(np.power(n_points * density_factor, 1/3)))
    
    x = np.linspace(x_min, x_max, n_points_per_dim)
    y = np.linspace(y_min, y_max, n_points_per_dim)
    z = np.linspace(z_min, z_max, n_points_per_dim)
    t = np.linspace(t_min, t_max, 10)  # Fixed number of time points
    
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
    
    # Convert to tensors
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
    z_tensor = torch.tensor(Z.flatten(), dtype=torch.float32).unsqueeze(1)
    t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)
    
    # Measure inference time
    start_time = time.time()
    start_memory = measure_memory_usage()
    
    predictions = {}
    batch_size = 10000
    n_batches = len(x_tensor) // batch_size + (1 if len(x_tensor) % batch_size != 0 else 0)
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x_tensor))
            
            batch_x = x_tensor[start_idx:end_idx].to(config.device)
            batch_y = y_tensor[start_idx:end_idx].to(config.device)
            batch_z = z_tensor[start_idx:end_idx].to(config.device)
            batch_t = t_tensor[start_idx:end_idx].to(config.device)
            
            predictions.setdefault('p', []).append(models['p'](batch_x, batch_y, batch_z, batch_t).cpu())
            predictions.setdefault('u', []).append(models['u'](batch_x, batch_y, batch_z, batch_t).cpu())
            predictions.setdefault('v', []).append(models['v'](batch_x, batch_y, batch_z, batch_t).cpu())
            predictions.setdefault('w', []).append(models['w'](batch_x, batch_y, batch_z, batch_t).cpu())
            predictions.setdefault('tau_x', []).append(models['tau_x'](batch_x, batch_y, batch_z, batch_t).cpu())
            predictions.setdefault('tau_y', []).append(models['tau_y'](batch_x, batch_y, batch_z, batch_t).cpu())
            predictions.setdefault('tau_z', []).append(models['tau_z'](batch_x, batch_y, batch_z, batch_t).cpu())
    
    # Concatenate predictions
    for key in predictions:
        predictions[key] = torch.cat(predictions[key], dim=0).numpy()
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    inference_stats = {
        "total_time": end_time - start_time,
        "points_per_second": len(x_tensor) / (end_time - start_time),
        "memory_used": end_memory - start_memory,
        "total_points": len(x_tensor),
        "grid_shape": (n_points_per_dim, n_points_per_dim, n_points_per_dim, len(t))
    }
    
    return predictions, (X, Y, Z, T), inference_stats

def analyze_spatial_distribution(data: pd.DataFrame, save_dir: str, logger: logging.Logger):
    """
    Analyze and visualize the spatial distribution of data points.
    
    Args:
        data (pd.DataFrame): Input data containing spatial coordinates
        save_dir (str): Directory to save the plots
        logger (logging.Logger): Logger instance
    """
    logger.info("Analyzing spatial distribution of points...")
    
    # Create directory for mesh analysis plots
    mesh_dir = os.path.join(save_dir, 'mesh_analysis')
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Extract coordinates
    x = data['X [ m ]'].values
    y = data['Y [ m ]'].values
    z = data['Z [ m ]'].values
    
    # Create 2D projections plot
    fig = plt.figure(figsize=(15, 5))
    
    # XY Projection
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(x, y, c=z, s=1, cmap='viridis', alpha=0.6)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('XY Projection')
    plt.colorbar(scatter1, ax=ax1, label='Z [m]')
    
    # XZ Projection
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(x, z, c=y, s=1, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Z [m]')
    ax2.set_title('XZ Projection')
    plt.colorbar(scatter2, ax=ax2, label='Y [m]')
    
    # YZ Projection
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(y, z, c=x, s=1, cmap='viridis', alpha=0.6)
    ax3.set_xlabel('Y [m]')
    ax3.set_ylabel('Z [m]')
    ax3.set_title('YZ Projection')
    plt.colorbar(scatter3, ax=ax3, label='X [m]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(mesh_dir, 'mesh_2d_projections.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze point distribution
    fig = plt.figure(figsize=(15, 5))
    
    # X distribution
    ax1 = fig.add_subplot(131)
    ax1.hist(x, bins=50, density=True, alpha=0.7)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Density')
    ax1.set_title('X Distribution')
    
    # Y distribution
    ax2 = fig.add_subplot(132)
    ax2.hist(y, bins=50, density=True, alpha=0.7)
    ax2.set_xlabel('Y [m]')
    ax2.set_ylabel('Density')
    ax2.set_title('Y Distribution')
    
    # Z distribution
    ax3 = fig.add_subplot(133)
    ax3.hist(z, bins=50, density=True, alpha=0.7)
    ax3.set_xlabel('Z [m]')
    ax3.set_ylabel('Density')
    ax3.set_title('Z Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(mesh_dir, 'mesh_points_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log statistics
    logger.info(f"Total number of points: {len(data)}")
    logger.info(f"Point density statistics:")
    logger.info(f"X range: [{x.min():.6f}, {x.max():.6f}], Mean spacing: {(x.max()-x.min())/len(x):.6f}")
    logger.info(f"Y range: [{y.min():.6f}, {y.max():.6f}], Mean spacing: {(y.max()-y.min())/len(y):.6f}")
    logger.info(f"Z range: [{z.min():.6f}, {z.max():.6f}], Mean spacing: {(z.max()-z.min())/len(z):.6f}")

def perform_convergence_study(data: pd.DataFrame, config: Config, logger: logging.Logger):
    """
    Perform convergence study with different point densities.
    
    Args:
        data (pd.DataFrame): Input data
        config (Config): Configuration object
        logger (logging.Logger): Logger instance
        
    Returns:
        dict: Convergence study results
    """
    logger.info("Starting convergence study...")
    
    # Define density levels
    density_levels = [0.25, 0.5, 0.75, 1.0]
    results = {
        'density_levels': density_levels,
        'n_points': [],
        'training_time': [],
        'final_loss': [],
        'physics_loss': [],
        'data_loss': []
    }
    
    # Initialize scalers
    scalers = initialize_scalers(data, config)
    
    for density in density_levels:
        logger.info(f"\nTesting density level: {density}")
        
        # Sample data points
        n_samples = int(len(data) * density)
        sampled_data = data.sample(n=n_samples, random_state=42)
        
        # Create dataset
        dataset = DenseDataset(sampled_data, scalers, config.scaler_columns)
        
        # Train PINN
        models, training_stats = train_and_time_pinn(config, dataset, logger)
        
        # Record results
        results['n_points'].append(n_samples)
        results['training_time'].append(training_stats['total_time'])
        results['final_loss'].append(training_stats['final_loss'])
        results['physics_loss'].append(training_stats['loss_history']['physics'][-1])
        results['data_loss'].append(training_stats['loss_history']['data'][-1])
        
        logger.info(f"Completed density level {density}")
        logger.info(f"Number of points: {n_samples}")
        logger.info(f"Training time: {training_stats['total_time']:.2f} seconds")
        logger.info(f"Final loss: {training_stats['final_loss']:.6f}")
    
    return results

def analyze_accuracy_vs_density(results: dict, save_dir: str, logger: logging.Logger):
    """
    Analyze and visualize the relationship between accuracy and point density.
    
    Args:
        results (dict): Results from convergence study
        save_dir (str): Directory to save plots
        logger (logging.Logger): Logger instance
    """
    logger.info("Analyzing accuracy vs density relationship...")
    
    # Create directory for solution density analysis
    density_dir = os.path.join(save_dir, 'density_analysis')
    os.makedirs(density_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot total loss vs number of points
    ax1 = fig.add_subplot(131)
    ax1.plot(results['n_points'], results['final_loss'], 'o-')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Convergence with Point Density')
    ax1.set_yscale('log')
    
    # Plot physics loss vs number of points
    ax2 = fig.add_subplot(132)
    ax2.plot(results['n_points'], results['physics_loss'], 'o-')
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('Physics Loss')
    ax2.set_title('Physics Constraint Satisfaction')
    ax2.set_yscale('log')
    
    # Plot computational cost
    ax3 = fig.add_subplot(133)
    ax3.plot(results['n_points'], results['training_time'], 'o-')
    ax3.set_xlabel('Number of Points')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Computational Cost')
    
    plt.tight_layout()
    plt.savefig(os.path.join(density_dir, 'solution_density_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log analysis results
    logger.info("\nAccuracy vs Density Analysis Results:")
    for i, n_points in enumerate(results['n_points']):
        logger.info(f"\nPoint Count: {n_points}")
        logger.info(f"Total Loss: {results['final_loss'][i]:.6f}")
        logger.info(f"Physics Loss: {results['physics_loss'][i]:.6f}")
        logger.info(f"Training Time: {results['training_time'][i]:.2f} seconds")

def visualize_results(original_data: pd.DataFrame, dense_predictions: dict,
                     grid: tuple, training_stats: dict, inference_stats: dict,
                     logger: logging.Logger):
    """Create visualizations comparing original CFD data with dense PINN predictions"""
    os.makedirs('figures/density_analysis', exist_ok=True)
    
    # 1. Point Density Comparison
    plt.figure(figsize=(15, 5))
    
    # Original CFD points
    plt.subplot(131)
    plt.scatter(original_data['X [ m ]'], original_data['Y [ m ]'],
                c=original_data['Pressure [ Pa ]'], s=1, alpha=0.5)
    plt.title(f'CFD Data\n({len(original_data)} points)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar(label='Pressure [Pa]')
    
    # Dense PINN predictions (sample slice)
    X, Y, Z, T = grid
    slice_idx = X.shape[2] // 2  # Middle Z slice
    time_idx = 0  # First time step
    
    plt.subplot(132)
    plt.scatter(X[:, :, slice_idx, time_idx].flatten(),
                Y[:, :, slice_idx, time_idx].flatten(),
                c=dense_predictions['p'][:len(X[:, :, slice_idx, time_idx].flatten())],
                s=1, alpha=0.5)
    plt.title(f'PINN Predictions\n({inference_stats["total_points"]} points)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar(label='Pressure [Pa]')
    
    # Computational comparison
    plt.subplot(133)
    metrics = {
        'CFD Setup': 48,  # Estimated hours
        'CFD Simulation': 72,  # Estimated hours
        'PINN Training': training_stats['total_time'] / 3600,  # Convert to hours
        'PINN Inference': inference_stats['total_time'] / 3600  # Convert to hours
    }
    
    plt.bar(metrics.keys(), metrics.values())
    plt.yscale('log')
    plt.ylabel('Time (hours)')
    plt.xticks(rotation=45)
    plt.title('Computational Time Comparison')
    
    plt.tight_layout()
    plt.savefig('figures/density_analysis/density_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Metrics
    logger.info("\nPerformance Comparison:")
    logger.info(f"CFD Data Points: {len(original_data)}")
    logger.info(f"PINN Prediction Points: {inference_stats['total_points']}")
    logger.info(f"Density Increase: {inference_stats['total_points']/len(original_data):.1f}x")
    logger.info(f"\nComputational Performance:")
    logger.info(f"Training Time: {training_stats['total_time']/3600:.2f} hours")
    logger.info(f"Inference Time: {inference_stats['total_time']:.2f} seconds")
    logger.info(f"Points per Second: {inference_stats['points_per_second']:.0f}")
    logger.info(f"Memory Usage - Training: {training_stats['memory_used']:.2f} GB")
    logger.info(f"Memory Usage - Inference: {inference_stats['memory_used']:.2f} GB")

def main():
    """Main function to run the density analysis"""
    logger = setup_logger('pinn_density_analysis')
    logger.info("Starting PINN Density Analysis")
    
    # Load configuration and data
    config = Config()
    data_path = "data/processed/aneurysm/diastolic/0021_diastolic_aneurysm.csv"
    df = load_data_with_metadata(data_path)
    
    # Initialize scalers
    logger.info("Initializing scalers...")
    scalers = initialize_scalers(df, config)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = DenseDataset(df, scalers, config.scaler_columns)
    
    # Train PINN and measure performance
    logger.info("Training PINN...")
    models, training_stats = train_and_time_pinn(config, dataset, logger)
    
    # Generate dense predictions
    logger.info("Generating dense predictions...")
    density_factor = 10  # 10x more points than original
    predictions, grid, inference_stats = generate_dense_predictions(
        models, df, density_factor, config, logger
    )
    
    # Visualize results
    logger.info("Creating visualizations...")
    visualize_results(df, predictions, grid, training_stats, inference_stats, logger)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
