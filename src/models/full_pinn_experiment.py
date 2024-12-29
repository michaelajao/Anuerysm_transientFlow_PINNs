# =========================================
# full_pinn_experiment.py
# =========================================

"""
Full PINN Experiment Script
===========================

This script conducts experiments using Physics-Informed Neural Networks (PINNs) to predict various
flow-related quantities such as Pressure, Velocity components, and Wall Shear Stress (WSS)
from Computational Fluid Dynamics (CFD) datasets. The script encompasses the entire workflow
from data loading and preprocessing to model training, evaluation, and visualization.

Key Features:
- Custom PINN architectures for multiple flow variables.
- Self-adaptive loss weighting to balance different loss components.
- Early stopping mechanism to prevent overfitting.
- Comprehensive evaluation metrics including R², NRMSE, and MAE.
- Visualization of loss curves, distribution comparisons, and histogram variations of WSS.

Author: Your Name
Date: YYYY-MM-DD
"""

# =========================================
# 1. Imports and Plotting Styles
# =========================================

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# For mixed precision training
from torch.cuda.amp import autocast, GradScaler  # Corrected import to torch.cuda.amp

# For evaluation metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Additional imports for interpolation and 3D plotting
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# =========================================
# 2. Set the Default Plotting Style
# =========================================

# Use Seaborn's "paper" style for clean and professional aesthetics
plt.style.use("seaborn-v0_8-paper")

# =========================================
# 3. Update rcParams for Publication-Quality Plots
# =========================================

plt.rcParams.update(
    {
        # -------------------------------------
        # General Figure Settings
        # -------------------------------------
        "font.size": 12,                       # Base font size for all text elements
        "figure.figsize": [7, 4],              # Figure size suitable for double-column layouts
        "text.usetex": False,                  # Disable LaTeX rendering; set to True if needed
        "figure.facecolor": "white",           # White background for compatibility
        "figure.autolayout": True,             # Automatically adjust subplot params
        "figure.dpi": 300,                     # High resolution for print quality
        "savefig.dpi": 300,                    # High resolution for saved figures
        "savefig.format": "pdf",               # Vector format for scalability; use 'png' if raster is needed
        "savefig.bbox": "tight",               # Minimize whitespace around the figure

        # -------------------------------------
        # Axes and Titles
        # -------------------------------------
        "axes.labelweight": "bold",            # Bold axis labels for emphasis
        "axes.titleweight": "bold",            # Bold titles for emphasis
        "axes.labelsize": 12,                  # Font size for axis labels
        "axes.titlesize": 16,                  # Font size for plot titles
        "axes.facecolor": "white",             # White background for axes
        "axes.grid": False,                    # Disable gridlines for clarity
        "axes.spines.top": False,              # Remove top spine for a cleaner look
        "axes.spines.right": False,            # Remove right spine for a cleaner look
        "axes.formatter.limits": (0, 5),       # Limit exponent formatting
        "axes.formatter.use_mathtext": True,   # Use LaTeX-style formatting for tick labels
        "axes.formatter.useoffset": False,     # Disable offset in tick labels
        "axes.xmargin": 0,                      # Remove horizontal margin
        "axes.ymargin": 0,                      # Remove vertical margin

        # -------------------------------------
        # Legend Settings
        # -------------------------------------
        "legend.fontsize": 12,                  # Font size for legend text
        "legend.frameon": False,                # Remove legend frame for a cleaner look
        "legend.loc": "best",                   # Automatically place legend in the best location

        # -------------------------------------
        # Line and Marker Settings
        # -------------------------------------
        "lines.linewidth": 2,                   # Thickness of plot lines
        "lines.markersize": 6,                  # Size of plot markers

        # -------------------------------------
        # Tick Settings
        # -------------------------------------
        "xtick.labelsize": 12,                  # Font size for x-axis tick labels
        "xtick.direction": "in",                # Ticks point inward for better aesthetics
        "xtick.top": False,                     # Disable ticks on the top edge
        "ytick.labelsize": 12,                  # Font size for y-axis tick labels
        "ytick.direction": "in",                # Ticks point inward for better aesthetics
        "ytick.right": False,                   # Disable ticks on the right edge

        # -------------------------------------
        # Grid and Error Bar Settings
        # -------------------------------------
        "grid.color": "grey",                   # Color of gridlines
        "grid.linestyle": "--",                 # Style of gridlines
        "grid.linewidth": 0.5,                  # Thickness of gridlines
        "errorbar.capsize": 3,                  # Length of error bar caps

        # -------------------------------------
        # Subplot Spacing
        # -------------------------------------
        "figure.subplot.wspace": 0.3,           # Width space between subplots
        "figure.subplot.hspace": 0.3,           # Height space between subplots

        # -------------------------------------
        # Image Settings
        # -------------------------------------
        "image.cmap": "viridis",                 # Default colormap for images
    }
)

# =========================================
# 4. Model Definitions
# =========================================

class Swish(nn.Module):
    """
    Swish Activation Function Module.

    Swish is defined as:
        swish(x) = x * sigmoid(beta * x)

    The parameter 'beta' is learnable, allowing the activation function to adapt during training.
    This provides greater flexibility compared to fixed activation functions like ReLU.
    """
    def __init__(self, beta: float = 1.0):
        """
        Initializes the Swish activation function.

        Args:
            beta (float): Initial value for the beta parameter.
        """
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Swish activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return x * torch.sigmoid(self.beta * x)


def init_weights(m: nn.Module):
    """
    Initializes the weights of linear layers using Kaiming Normal initialization.

    Args:
        m (nn.Module): A module from the neural network.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BasePINN(nn.Module):
    """
    Base Class for Physics-Informed Neural Networks (PINNs).

    This class defines a fully connected neural network architecture with optional batch normalization
    and Swish activations. It also includes learnable parameters for self-adaptive loss weighting.
    """
    def __init__(self, config: 'Config', out_dim: int = 1):
        """
        Initializes the BasePINN.

        Args:
            config (Config): Configuration object containing hyperparameters and architecture details.
            out_dim (int): Output dimension of the network.
        """
        super().__init__()
        layers = []
        inp_dim = 4  # Input features: (x, y, z, t)
        for i in range(config.num_layers):
            in_features = inp_dim if i == 0 else config.units_per_layer
            layers.append(nn.Linear(in_features, config.units_per_layer))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.units_per_layer))
            layers.append(Swish())
        layers.append(nn.Linear(config.units_per_layer, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

        # Self-adaptive weights for different loss components
        self.log_lambda_physics = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_boundary = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_data = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_inlet = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): X-coordinate.
            y (torch.Tensor): Y-coordinate.
            z (torch.Tensor): Z-coordinate.
            t (torch.Tensor): Time.

        Returns:
            torch.Tensor: Network output.
        """
        inp = torch.cat([x, y, z, t], dim=1)
        return self.net(inp)


# Specialized PINN classes for different flow variables
class PressurePINN(BasePINN):
    """PINN for Predicting Pressure (p). Output Dimension: 1"""
    pass


class UVelocityPINN(BasePINN):
    """PINN for Predicting X-component of Velocity (u). Output Dimension: 1"""
    pass


class VVelocityPINN(BasePINN):
    """PINN for Predicting Y-component of Velocity (v). Output Dimension: 1"""
    pass


class WVelocityPINN(BasePINN):
    """PINN for Predicting Z-component of Velocity (w). Output Dimension: 1"""
    pass


class TauXPINN(BasePINN):
    """PINN for Predicting X-component of Wall Shear Stress (tau_x). Output Dimension: 1"""
    pass


class TauYPINN(BasePINN):
    """PINN for Predicting Y-component of Wall Shear Stress (tau_y). Output Dimension: 1"""
    pass


class TauZPINN(BasePINN):
    """PINN for Predicting Z-component of Wall Shear Stress (tau_z). Output Dimension: 1"""
    pass


def initialize_models(config: 'Config', logger: logging.Logger) -> Dict[str, nn.Module]:
    """
    Initializes all PINN models for different flow variables.

    Args:
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.

    Returns:
        Dict[str, nn.Module]: Dictionary containing initialized models.
    """
    model_p = PressurePINN(config=config, out_dim=1).to(config.device)
    model_u = UVelocityPINN(config=config, out_dim=1).to(config.device)
    model_v = VVelocityPINN(config=config, out_dim=1).to(config.device)
    model_w = WVelocityPINN(config=config, out_dim=1).to(config.device)
    model_tau_x = TauXPINN(config=config, out_dim=1).to(config.device)
    model_tau_y = TauYPINN(config=config, out_dim=1).to(config.device)
    model_tau_z = TauZPINN(config=config, out_dim=1).to(config.device)

    logger.info("Initialized 7 separate PINN models for p, u, v, w, tau_x, tau_y, tau_z.")
    return {
        "p": model_p,
        "u": model_u,
        "v": model_v,
        "w": model_w,
        "tau_x": model_tau_x,
        "tau_y": model_tau_y,
        "tau_z": model_tau_z,
    }

# =========================================
# 5. Configuration and Parameters
# =========================================

@dataclass
class Config:
    """
    Configuration Class for PINN Experiments.

    Holds all hyperparameters, file paths, and architectural details required for the experiment.
    """
    # Directory Paths
    model_dir: str = "../../models/pinn_experiment"             # Directory to save trained models
    plot_dir: str = "../../reports/figures/pinn_experiment"     # Directory to save generated plots
    metrics_dir: str = "../../metrics/pinn_experiment"          # Directory to save logs and metrics
    data_dir: str = "../../data"                                 # Directory containing raw data
    processed_data_dir: str = "../../data/processed"            # Directory containing processed data

    # Experiment Parameters
    categories: list = field(default_factory=lambda: ["aneurysm", "healthy"])  # Data categories
    phases: list = field(default_factory=lambda: ["systolic", "diastolic"])     # Phases of data

    run_id: str = field(init=False)  # Unique identifier for each run, set per dataset

    # Reproducibility
    random_seed: int = 142  # Seed for random number generators

    # Device Configuration
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"  # Computation device

    # Data Normalization Settings
    scaler_columns: Dict[str, Any] = field(default_factory=lambda: {
        "features": ["X [ m ]", "Y [ m ]", "Z [ m ]"],
        "time": "Time [ s ]",
        "pressure": "Pressure [ Pa ]",
        "velocity_u": "Velocity u [ m s^-1 ]",
        "velocity_v": "Velocity v [ m s^-1 ]",
        "velocity_w": "Velocity w [ m s^-1 ]",
        "wall_shear_x": "Wall Shear X [ Pa ]",
        "wall_shear_y": "Wall Shear Y [ Pa ]",
        "wall_shear_z": "Wall Shear Z [ Pa ]",
    })

    # Training Hyperparameters
    epochs: int = 500                 # Number of training epochs
    batch_size: int = 1024             # Batch size for training
    learning_rate: float = 1e-4        # Initial learning rate
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    })
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "step_size": 200,
        "gamma": 0.9,
    })

    # Physical Parameters for Navier-Stokes Equations
    rho: float = 1060.0  # Fluid density (kg/m^3)
    mu: float = 0.0035    # Dynamic viscosity (Pa.s)

    # Early Stopping Configuration
    early_stopping_patience: int = 5        # Patience for early stopping
    early_stopping_min_delta: float = 1e-6  # Minimum delta for improvement

    # Neural Network Architecture
    use_batch_norm: bool = True    # Flag to use Batch Normalization
    num_layers: int = 10            # Number of hidden layers
    units_per_layer: int = 64       # Number of neurons per hidden layer

    # Plotting Settings
    plot_resolution: int = 300       # DPI for saved plots

    # DataLoader Optimization
    num_workers: int = 0             # Number of worker threads for DataLoader
    pin_memory: bool = field(init=False)  # Pin memory flag, set based on device

    def __post_init__(self):
        """
        Post-initialization to set derived attributes.
        """
        self.pin_memory = self.device.startswith("cuda")


# =========================================
# 6. Logging Setup
# =========================================

def setup_logging(run_id: str, metrics_dir: str, config: Config) -> logging.Logger:
    """
    Configures logging with both console and file handlers.

    Args:
        run_id (str): Unique identifier for the experiment run.
        metrics_dir (str): Directory to save log files.
        config (Config): Configuration object.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(f"experiment_logger_{run_id}")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        # Define log file path
        log_file_path = os.path.join(metrics_dir, run_id, f"experiment_{run_id}.log")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # File handler for logging to a file
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler for logging to the terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.info(f"Experiment run initialized with Run ID: {run_id}")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Random seed set to {config.random_seed}")

    return logger


# =========================================
# 7. Data Loading and Preprocessing
# =========================================

class CFDDataset(Dataset):
    """
    Custom PyTorch Dataset for CFD Data.

    This dataset handles normalization of input features and target variables,
    and prepares tensors for model training. It also identifies boundary points
    where no-slip conditions are applied.
    """
    def __init__(self, data: pd.DataFrame, scalers: Dict[str, Any], scaler_columns: Dict[str, Any], tolerance: float = 1e-5):
        """
        Initializes the CFDDataset.

        Args:
            data (pd.DataFrame): Raw CFD data.
            scalers (Dict[str, Any]): Dictionary of fitted MinMaxScaler instances.
            scaler_columns (Dict[str, Any]): Mapping of variable names to column names.
            tolerance (float): Tolerance for identifying boundary points.
        """
        self.scalers = scalers
        self.scaler_columns = scaler_columns
        self.data = data.copy()
        self.normalize_data()
        self.prepare_tensors(tolerance)

    def normalize_data(self):
        """
        Applies normalization to the data using the provided scalers.
        """
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

    def prepare_tensors(self, tolerance: float):
        """
        Converts normalized data into PyTorch tensors for model input and targets.

        Args:
            tolerance (float): Tolerance for identifying boundary points.
        """
        feature_cols = self.scaler_columns["features"]
        self.x = torch.tensor(self.data[feature_cols[0]].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.data[feature_cols[1]].values, dtype=torch.float32).unsqueeze(1)
        self.z = torch.tensor(self.data[feature_cols[2]].values, dtype=torch.float32).unsqueeze(1)
        self.t = torch.tensor(self.data[self.scaler_columns["time"]].values, dtype=torch.float32).unsqueeze(1)

        # Targets
        self.p = torch.tensor(self.data[self.scaler_columns["pressure"]].values, dtype=torch.float32).unsqueeze(1)
        self.u = torch.tensor(self.data[self.scaler_columns["velocity_u"]].values, dtype=torch.float32).unsqueeze(1)
        self.v = torch.tensor(self.data[self.scaler_columns["velocity_v"]].values, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(self.data[self.scaler_columns["velocity_w"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_x = torch.tensor(self.data[self.scaler_columns["wall_shear_x"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_y = torch.tensor(self.data[self.scaler_columns["wall_shear_y"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_z = torch.tensor(self.data[self.scaler_columns["wall_shear_z"]].values, dtype=torch.float32).unsqueeze(1)

        # Boundary conditions: no-slip (u=v=w≈0) with tolerance
        epsilon = tolerance
        self.is_boundary = (
            (torch.abs(self.u) < epsilon) &
            (torch.abs(self.v) < epsilon) &
            (torch.abs(self.w) < epsilon)
        ).squeeze()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing input features and target variables.
        """
        return (
            self.x[idx],
            self.y[idx],
            self.z[idx],
            self.t[idx],
            self.p[idx],
            self.u[idx],
            self.v[idx],
            self.w[idx],
            self.tau_x[idx],
            self.tau_y[idx],
            self.tau_z[idx],
            self.is_boundary[idx],
        )


def load_data(config: Config, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """
    Loads and preprocesses all CFD datasets from the specified directories.

    Args:
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset names to DataFrames.
    """
    processed_files = []
    for category in config.categories:
        for phase in config.phases:
            phase_dir = os.path.join(config.processed_data_dir, category, phase)
            if os.path.isdir(phase_dir):
                for file in os.listdir(phase_dir):
                    if file.endswith(".csv"):
                        processed_files.append(os.path.join(phase_dir, file))

    datasets = {}
    required_columns = [
        "X [ m ]", "Y [ m ]", "Z [ m ]", "Time [ s ]", "Pressure [ Pa ]",
        "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]",
        "Wall Shear X [ Pa ]", "Wall Shear Y [ Pa ]", "Wall Shear Z [ Pa ]"
    ]

    for file in processed_files:
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Dataset '{dataset_name}' missing columns: {missing_cols}. Dropping missing rows.")
            df = df.dropna(subset=required_columns)
        else:
            df = df.dropna(subset=required_columns)
        datasets[dataset_name] = df

    logger.info(f"Loaded {len(datasets)} datasets.")
    for name, df in datasets.items():
        logger.info(f"Dataset '{name}' shape: {df.shape}")
    return datasets

# =========================================
# 8. Optimizer and Scheduler Setup
# =========================================

def setup_optimizer_scheduler(models: Dict[str, nn.Module], config: Config, logger: logging.Logger) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Configures the optimizer and learning rate scheduler for all PINN models.

    Args:
        models (Dict[str, nn.Module]): Dictionary of PINN models.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.

    Returns:
        Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]: Optimizer and scheduler instances.
    """
    all_params = []
    for model in models.values():
        all_params += list(model.parameters())

    optimizer = optim.AdamW(
        params=all_params,
        lr=config.learning_rate,
        betas=config.optimizer_params["betas"],
        eps=config.optimizer_params["eps"],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_params["step_size"],
        gamma=config.scheduler_params["gamma"],
    )

    logger.info("Optimizer and scheduler initialized for all separate PINNs.")
    return optimizer, scheduler

# =========================================
# 9. Early Stopping Mechanism
# =========================================

class EarlyStopping:
    """
    Early Stopping Utility to Halt Training When No Improvement is Observed.

    Monitors the total loss and stops training if it doesn't improve after a specified number of epochs.
    Also saves the best model checkpoint observed during training.
    """
    def __init__(self, patience: int = 100, min_delta: float = 1e-6, logger: Optional[logging.Logger] = None):
        """
        Initializes the EarlyStopping mechanism.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in loss to qualify as an improvement.
            logger (Optional[logging.Logger]): Logger for logging information.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.logger = logger

    def __call__(self, loss: float, models: Dict[str, nn.Module],
               optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
               run_id: str, model_dir: str):
        """
        Checks for improvement in loss and updates the early stopping counter.

        Args:
            loss (float): Current epoch's total loss.
            models (Dict[str, nn.Module]): Dictionary of PINN models.
            optimizer (optim.Optimizer): Optimizer instance.
            scheduler (optim.lr_scheduler._LRScheduler): Scheduler instance.
            run_id (str): Unique identifier for the experiment run.
            model_dir (str): Directory to save model checkpoints.
        """
        if self.best_loss is None:
            self.best_loss = loss
            if self.logger:
                self.logger.info(f"Initial loss set to {self.best_loss:.6f}.")
            self.save_checkpoint(models, optimizer, scheduler, run_id, model_dir)
        elif (self.best_loss - loss) > self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if self.logger:
                self.logger.info(f"Loss improved to {self.best_loss:.6f}. Resetting early stopping counter.")
            self.save_checkpoint(models, optimizer, scheduler, run_id, model_dir)
        else:
            self.counter += 1
            if self.logger:
                self.logger.info(f"No improvement in loss. Early stopping counter: {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.logger:
                    self.logger.info("Early stopping triggered.")

    def save_checkpoint(self, models: Dict[str, nn.Module],
                        optimizer: optim.Optimizer,
                        scheduler: optim.lr_scheduler._LRScheduler,
                        run_id: str, model_dir: str):
        """
        Saves the current best model state to a checkpoint file.

        Args:
            models (Dict[str, nn.Module]): Dictionary of PINN models.
            optimizer (optim.Optimizer): Optimizer instance.
            scheduler (optim.lr_scheduler._LRScheduler): Scheduler instance.
            run_id (str): Unique identifier for the experiment run.
            model_dir (str): Directory to save model checkpoints.
        """
        best_model_path = os.path.join(model_dir, run_id, f"best_model_{run_id}.pt")
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        checkpoint = {}
        for key, model in models.items():
            checkpoint[f"{key}_state_dict"] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint['best_loss'] = self.best_loss

        torch.save(checkpoint, best_model_path)
        if self.logger:
            self.logger.info(f"Saved best model checkpoint to '{best_model_path}'.")

# =========================================
# 10. Loss Functions
# =========================================

def compute_physics_loss(
    p_pred: torch.Tensor, u_pred: torch.Tensor, v_pred: torch.Tensor, w_pred: torch.Tensor,
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
    rho: float, mu: float
) -> torch.Tensor:
    """
    Computes the Physics-based Loss incorporating Navier-Stokes Equations and Continuity.

    This function calculates the residuals of the momentum equations and the continuity equation,
    and computes their Mean Squared Errors (MSE).

    Args:
        p_pred (torch.Tensor): Predicted pressure.
        u_pred (torch.Tensor): Predicted u-velocity.
        v_pred (torch.Tensor): Predicted v-velocity.
        w_pred (torch.Tensor): Predicted w-velocity.
        x (torch.Tensor): X-coordinate tensor with gradients enabled.
        y (torch.Tensor): Y-coordinate tensor with gradients enabled.
        z (torch.Tensor): Z-coordinate tensor with gradients enabled.
        t (torch.Tensor): Time tensor with gradients enabled.
        rho (float): Fluid density.
        mu (float): Dynamic viscosity.

    Returns:
        torch.Tensor: Combined physics-based loss.
    """
    # Calculate first-order derivatives
    u_x = grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_y = grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_z = grad(u_pred, z, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]

    v_x = grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_y = grad(v_pred, y, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_z = grad(v_pred, z, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_t = grad(v_pred, t, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]

    w_x = grad(w_pred, x, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_y = grad(w_pred, y, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_z = grad(w_pred, z, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_t = grad(w_pred, t, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]

    p_x = grad(p_pred, x, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]
    p_y = grad(p_pred, y, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]
    p_z = grad(p_pred, z, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]

    # Calculate second-order derivatives
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    # Compute residuals for Navier-Stokes Equations
    residual_u = (
        u_t + u_pred * u_x + v_pred * u_y + w_pred * u_z
        + (1 / rho) * p_x - (mu / rho) * (u_xx + u_yy + u_zz)
    )
    residual_v = (
        v_t + u_pred * v_x + v_pred * v_y + w_pred * v_z
        + (1 / rho) * p_y - (mu / rho) * (v_xx + v_yy + v_zz)
    )
    residual_w = (
        w_t + u_pred * w_x + v_pred * w_y + w_pred * w_z
        + (1 / rho) * p_z - (mu / rho) * (w_xx + w_yy + w_zz)
    )

    # Compute residual for Continuity Equation
    continuity = u_x + v_y + w_z

    mse = nn.MSELoss()
    loss_nse = (
        mse(residual_u, torch.zeros_like(residual_u)) +
        mse(residual_v, torch.zeros_like(residual_v)) +
        mse(residual_w, torch.zeros_like(residual_w))
    )
    loss_continuity = mse(continuity, torch.zeros_like(continuity))

    return loss_nse + loss_continuity


def compute_boundary_loss(u_bc_pred: torch.Tensor, v_bc_pred: torch.Tensor, w_bc_pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Boundary Condition Loss enforcing no-slip conditions.

    No-slip condition implies that the velocity components at the boundary are zero.

    Args:
        u_bc_pred (torch.Tensor): Predicted u-velocity at boundary points.
        v_bc_pred (torch.Tensor): Predicted v-velocity at boundary points.
        w_bc_pred (torch.Tensor): Predicted w-velocity at boundary points.

    Returns:
        torch.Tensor: Boundary condition loss.
    """
    mse = nn.MSELoss()
    loss_bc = (
        mse(u_bc_pred, torch.zeros_like(u_bc_pred)) +
        mse(v_bc_pred, torch.zeros_like(v_bc_pred)) +
        mse(w_bc_pred, torch.zeros_like(w_bc_pred))
    )
    return loss_bc


def compute_data_loss(
    p_pred: torch.Tensor, p_true: torch.Tensor,
    u_pred: torch.Tensor, u_true: torch.Tensor,
    v_pred: torch.Tensor, v_true: torch.Tensor,
    w_pred: torch.Tensor, w_true: torch.Tensor,
    tau_x_pred: torch.Tensor, tau_x_true: torch.Tensor,
    tau_y_pred: torch.Tensor, tau_y_true: torch.Tensor,
    tau_z_pred: torch.Tensor, tau_z_true: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Supervised Data Loss using Mean Squared Error (MSE).

    This loss measures the discrepancy between the PINN predictions and the actual CFD data.

    Args:
        p_pred (torch.Tensor): Predicted pressure.
        p_true (torch.Tensor): True pressure.
        u_pred (torch.Tensor): Predicted u-velocity.
        u_true (torch.Tensor): True u-velocity.
        v_pred (torch.Tensor): Predicted v-velocity.
        v_true (torch.Tensor): True v-velocity.
        w_pred (torch.Tensor): Predicted w-velocity.
        w_true (torch.Tensor): True w-velocity.
        tau_x_pred (torch.Tensor): Predicted tau_x.
        tau_x_true (torch.Tensor): True tau_x.
        tau_y_pred (torch.Tensor): Predicted tau_y.
        tau_y_true (torch.Tensor): True tau_y.
        tau_z_pred (torch.Tensor): Predicted tau_z.
        tau_z_true (torch.Tensor): True tau_z.

    Returns:
        torch.Tensor: Combined data loss for all variables.
    """
    mse = nn.MSELoss()
    loss_p = mse(p_pred, p_true)
    loss_u = mse(u_pred, u_true)
    loss_v = mse(v_pred, v_true)
    loss_w = mse(w_pred, w_true)
    loss_tau_x = mse(tau_x_pred, tau_x_true)
    loss_tau_y = mse(tau_y_pred, tau_y_true)
    loss_tau_z = mse(tau_z_pred, tau_z_true)
    return loss_p + loss_u + loss_v + loss_w + loss_tau_x + loss_tau_y + loss_tau_z


def compute_inlet_loss(u_pred: torch.Tensor, v_pred: torch.Tensor, w_pred: torch.Tensor, t: torch.Tensor, heart_rate: int = 120) -> torch.Tensor:
    """
    Computes the Inlet Velocity Profile Loss based on a sinusoidal wave.

    This loss enforces an inflow boundary condition with a specified velocity profile.

    Args:
        u_pred (torch.Tensor): Predicted u-velocity.
        v_pred (torch.Tensor): Predicted v-velocity.
        w_pred (torch.Tensor): Predicted w-velocity.
        t (torch.Tensor): Time tensor.
        heart_rate (int): Heart rate in beats per minute to define the sinusoidal period.

    Returns:
        torch.Tensor: Inlet condition loss.
    """
    period = 60 / heart_rate  # Period in seconds
    t_mod = torch.fmod(t, period)
    u_inlet_true = torch.where(
        t_mod <= 0.218,
        0.5 * torch.sin(4 * np.pi * (t_mod + 0.0160236)),
        torch.full_like(t_mod, 0.1)
    )
    v_inlet_true = torch.zeros_like(u_inlet_true)
    w_inlet_true = torch.zeros_like(u_inlet_true)
    mse = nn.MSELoss()
    loss_inlet = (
        mse(u_pred, u_inlet_true) +
        mse(v_pred, v_inlet_true) +
        mse(w_pred, w_inlet_true)
    )
    return loss_inlet

# =========================================
# 11. Evaluation Metrics
# =========================================

def evaluate_pinn(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    dataset: CFDDataset,
    config: Config,
    logger: logging.Logger,
    run_id: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
    """
    Evaluates the trained PINN models on the entire dataset and computes evaluation metrics.

    Metrics Computed:
    - R² Score
    - Normalized Root Mean Squared Error (NRMSE)
    - Mean Absolute Error (MAE)

    Args:
        models (Dict[str, nn.Module]): Dictionary of trained PINN models.
        dataloader (DataLoader): DataLoader for evaluation data.
        dataset (CFDDataset): Dataset instance for inverse transformations.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
            Dictionaries containing R² scores, NRMSE scores, MAE scores, and total MAE.
    """
    for m in models.values():
        m.eval()

    r2_scores = {}
    nrmse_scores = {}
    mae_scores = {}
    variables = [
        "pressure",
        "velocity_u",
        "velocity_v",
        "velocity_w",
        "wall_shear_x",
        "wall_shear_y",
        "wall_shear_z",
    ]

    predictions = {var: [] for var in variables}
    truths = {var: [] for var in variables}

    # Convenient references to models
    model_p = models["p"]
    model_u = models["u"]
    model_v = models["v"]
    model_w = models["w"]
    model_tau_x = models["tau_x"]
    model_tau_y = models["tau_y"]
    model_tau_z = models["tau_z"]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            (
                x_batch,
                y_batch,
                z_batch,
                t_batch,
                p_true,
                u_true,
                v_true,
                w_true,
                tau_x_true,
                tau_y_true,
                tau_z_true,
                _
            ) = batch

            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            z_batch = z_batch.to(config.device)
            t_batch = t_batch.to(config.device)

            # Forward passes
            p_pred = model_p(x_batch, y_batch, z_batch, t_batch)
            u_pred = model_u(x_batch, y_batch, z_batch, t_batch)
            v_pred = model_v(x_batch, y_batch, z_batch, t_batch)
            w_pred = model_w(x_batch, y_batch, z_batch, t_batch)
            tau_x_pred = model_tau_x(x_batch, y_batch, z_batch, t_batch)
            tau_y_pred = model_tau_y(x_batch, y_batch, z_batch, t_batch)
            tau_z_pred = model_tau_z(x_batch, y_batch, z_batch, t_batch)

            # Collect predictions & ground truth
            predictions["pressure"].append(p_pred.cpu().numpy())
            truths["pressure"].append(p_true.numpy())

            predictions["velocity_u"].append(u_pred.cpu().numpy())
            truths["velocity_u"].append(u_true.numpy())

            predictions["velocity_v"].append(v_pred.cpu().numpy())
            truths["velocity_v"].append(v_true.numpy())

            predictions["velocity_w"].append(w_pred.cpu().numpy())
            truths["velocity_w"].append(w_true.numpy())

            predictions["wall_shear_x"].append(tau_x_pred.cpu().numpy())
            truths["wall_shear_x"].append(tau_x_true.numpy())

            predictions["wall_shear_y"].append(tau_y_pred.cpu().numpy())
            truths["wall_shear_y"].append(tau_y_true.numpy())

            predictions["wall_shear_z"].append(tau_z_pred.cpu().numpy())
            truths["wall_shear_z"].append(tau_z_true.numpy())

            # Cleanup to free GPU memory
            del (x_batch, y_batch, z_batch, t_batch,
                 p_true, u_true, v_true, w_true,
                 tau_x_true, tau_y_true, tau_z_true,
                 p_pred, u_pred, v_pred, w_pred,
                 tau_x_pred, tau_y_pred, tau_z_pred)
            torch.cuda.empty_cache()

    # Convert lists to arrays and inverse transform to original scale
    for var in variables:
        predictions[var] = np.concatenate(predictions[var], axis=0)
        truths[var] = np.concatenate(truths[var], axis=0)
        predictions[var] = dataset.scalers[var].inverse_transform(predictions[var].reshape(-1, 1)).flatten()
        truths[var] = dataset.scalers[var].inverse_transform(truths[var].reshape(-1, 1)).flatten()

    # Compute evaluation metrics with epsilon to prevent divide-by-zero
    epsilon = 1e-8
    for var in variables:
        r2 = r2_score(truths[var], predictions[var])
        denominator = (truths[var].max() - truths[var].min()) + epsilon
        nrmse = np.sqrt(mean_squared_error(truths[var], predictions[var])) / denominator
        mae = mean_absolute_error(truths[var], predictions[var])
        r2_scores[var] = r2
        nrmse_scores[var] = nrmse
        mae_scores[var] = mae

    # Calculate total MAE across all variables
    total_mae = np.mean(list(mae_scores.values()))
    metrics_summary = {
        "Run_ID": run_id,
        "Total_MAE": total_mae,
    }
    for var in variables:
        metrics_summary[f"{var}_R2"] = r2_scores[var]
        metrics_summary[f"{var}_NRMSE"] = nrmse_scores[var]
        metrics_summary[f"{var}_MAE"] = mae_scores[var]

    # Save metrics summary to CSV
    df_metrics = pd.DataFrame([metrics_summary])
    metrics_summary_path = os.path.join(config.metrics_dir, run_id, f"metrics_summary_{run_id}.csv")
    os.makedirs(os.path.dirname(metrics_summary_path), exist_ok=True)
    df_metrics.to_csv(metrics_summary_path, index=False)
    logger.info(f"Saved metrics summary to '{metrics_summary_path}'.")

    return r2_scores, nrmse_scores, mae_scores, total_mae

# =========================================
# 10. Visualization Functions
# =========================================

def plot_loss_curves(loss_history: Dict[str, list], config: Config, logger: logging.Logger, run_id: str, dataset_name: str):
    """
    Plots the training loss curves for different loss components.

    Args:
        loss_history (Dict[str, list]): Dictionary containing loss history.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.
        dataset_name (str): Name of the dataset.
    """
    epochs = range(1, len(loss_history["total"]) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, loss_history["total"], label="Total Loss")
    plt.plot(epochs, loss_history["physics"], label="Physics Loss")
    plt.plot(epochs, loss_history["boundary"], label="Boundary Loss")
    plt.plot(epochs, loss_history["data"], label="Data Loss")
    plt.plot(epochs, loss_history["inlet"], label="Inlet Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.yscale("log")
    plt.title(f"Training Loss Curves - {run_id}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    run_plot_dir = os.path.join(config.plot_dir, run_id)
    os.makedirs(run_plot_dir, exist_ok=True)
    plot_filename = f"loss_curves_{run_id}.png"
    plot_path = os.path.join(run_plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss curves plot to '{plot_path}'.")


def plot_pressure_and_wss_magnitude_distribution(
    dataset: CFDDataset,
    models: Dict[str, nn.Module],
    config: Config,
    logger: logging.Logger,
    run_id: str,
):
    """
    Generates and saves plots comparing the distribution of Pressure and WSS Magnitude
    between CFD data and PINN predictions in both XY and XZ planes.

    Args:
        dataset (CFDDataset): Dataset instance containing CFD data.
        models (Dict[str, nn.Module]): Dictionary of trained PINN models.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.
    """
    for m in models.values():
        m.eval()

    # Convenient references to models
    model_p = models["p"]
    model_u = models["u"]
    model_v = models["v"]
    model_w = models["w"]
    model_tau_x = models["tau_x"]
    model_tau_y = models["tau_y"]
    model_tau_z = models["tau_z"]

    variables = [
        "pressure",
        "velocity_u",
        "velocity_v",
        "velocity_w",
        "wall_shear_x",
        "wall_shear_y",
        "wall_shear_z",
    ]

    predictions = {var: [] for var in variables}
    truths = {var: [] for var in variables}

    x_sample = dataset.x.numpy()
    y_sample = dataset.y.numpy()
    z_sample = dataset.z.numpy()
    t_sample = dataset.t.numpy()

    batch_size = 1024
    num_samples = len(x_sample)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = i + batch_size
            batch_x = torch.tensor(x_sample[i:end], dtype=torch.float32, device=config.device)
            batch_y = torch.tensor(y_sample[i:end], dtype=torch.float32, device=config.device)
            batch_z = torch.tensor(z_sample[i:end], dtype=torch.float32, device=config.device)
            batch_t = torch.tensor(t_sample[i:end], dtype=torch.float32, device=config.device)

            p_pred = model_p(batch_x, batch_y, batch_z, batch_t)
            u_pred = model_u(batch_x, batch_y, batch_z, batch_t)
            v_pred = model_v(batch_x, batch_y, batch_z, batch_t)
            w_pred = model_w(batch_x, batch_y, batch_z, batch_t)
            tau_x_pred = model_tau_x(batch_x, batch_y, batch_z, batch_t)
            tau_y_pred = model_tau_y(batch_x, batch_y, batch_z, batch_t)
            tau_z_pred = model_tau_z(batch_x, batch_y, batch_z, batch_t)

            predictions["pressure"].append(p_pred.cpu().numpy())
            truths["pressure"].append(dataset.p[i:end].numpy())

            predictions["velocity_u"].append(u_pred.cpu().numpy())
            truths["velocity_u"].append(dataset.u[i:end].numpy())

            predictions["velocity_v"].append(v_pred.cpu().numpy())
            truths["velocity_v"].append(dataset.v[i:end].numpy())

            predictions["velocity_w"].append(w_pred.cpu().numpy())
            truths["velocity_w"].append(dataset.w[i:end].numpy())

            predictions["wall_shear_x"].append(tau_x_pred.cpu().numpy())
            truths["wall_shear_x"].append(dataset.tau_x[i:end].numpy())

            predictions["wall_shear_y"].append(tau_y_pred.cpu().numpy())
            truths["wall_shear_y"].append(dataset.tau_y[i:end].numpy())

            predictions["wall_shear_z"].append(tau_z_pred.cpu().numpy())
            truths["wall_shear_z"].append(dataset.tau_z[i:end].numpy())

            # Cleanup to free GPU memory
            del batch_x, batch_y, batch_z, batch_t
            del p_pred, u_pred, v_pred, w_pred, tau_x_pred, tau_y_pred, tau_z_pred
            torch.cuda.empty_cache()

    # Convert lists to arrays and inverse transform to original scale
    for var in variables:
        predictions[var] = np.concatenate(predictions[var], axis=0)
        truths[var] = np.concatenate(truths[var], axis=0)
        predictions[var] = dataset.scalers[var].inverse_transform(predictions[var].reshape(-1, 1)).flatten()
        truths[var] = dataset.scalers[var].inverse_transform(truths[var].reshape(-1, 1)).flatten()

    # Compute WSS Magnitude for Predictions and True Data
    wss_magnitude_pred = np.sqrt(
        predictions["wall_shear_x"]**2 +
        predictions["wall_shear_y"]**2 +
        predictions["wall_shear_z"]**2
    )
    wss_magnitude_true = np.sqrt(
        truths["wall_shear_x"]**2 +
        truths["wall_shear_y"]**2 +
        truths["wall_shear_z"]**2
    )

    run_plot_dir = os.path.join(config.plot_dir, run_id)
    os.makedirs(run_plot_dir, exist_ok=True)

    # =========================================
    # Pressure Distribution Plots with Shared Colorbars
    # =========================================

    fig_p, axs_p = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=False)

    # Define overall min and max for pressure to ensure consistent color mapping
    pressure_min = min(truths["pressure"].min(), predictions["pressure"].min())
    pressure_max = max(truths["pressure"].max(), predictions["pressure"].max())

    # Plot CFD Pressure (XY Plane)
    sc1 = axs_p[0, 0].scatter(x_sample, y_sample, c=truths["pressure"], cmap="viridis", vmin=pressure_min, vmax=pressure_max, s=5, alpha=0.8)
    axs_p[0, 0].set_title("CFD Pressure (XY Plane)", fontsize=16)
    axs_p[0, 0].set_xlabel("X [m]", fontsize=14)
    axs_p[0, 0].set_ylabel("Y [m]", fontsize=14)

    # Plot PINN Pressure (XY Plane)
    sc2 = axs_p[0, 1].scatter(x_sample, y_sample, c=predictions["pressure"], cmap="viridis", vmin=pressure_min, vmax=pressure_max, s=5, alpha=0.8)
    axs_p[0, 1].set_title("PINN Pressure (XY Plane)", fontsize=16)
    axs_p[0, 1].set_xlabel("X [m]", fontsize=14)
    axs_p[0, 1].set_ylabel("Y [m]", fontsize=14)

    # Plot CFD Pressure (XZ Plane)
    sc3 = axs_p[1, 0].scatter(x_sample, z_sample, c=truths["pressure"], cmap="viridis", vmin=pressure_min, vmax=pressure_max, s=5, alpha=0.8)
    axs_p[1, 0].set_title("CFD Pressure (XZ Plane)", fontsize=16)
    axs_p[1, 0].set_xlabel("X [m]", fontsize=14)
    axs_p[1, 0].set_ylabel("Z [m]", fontsize=14)

    # Plot PINN Pressure (XZ Plane)
    sc4 = axs_p[1, 1].scatter(x_sample, z_sample, c=predictions["pressure"], cmap="viridis", vmin=pressure_min, vmax=pressure_max, s=5, alpha=0.8)
    axs_p[1, 1].set_title("PINN Pressure (XZ Plane)", fontsize=16)
    axs_p[1, 1].set_xlabel("X [m]", fontsize=14)
    axs_p[1, 1].set_ylabel("Z [m]", fontsize=14)

    # Add a single colorbar for all pressure plots
    cbar_p = fig_p.colorbar(sc1, ax=axs_p, orientation='vertical', fraction=0.02, pad=0.04)
    cbar_p.set_label("Pressure [Pa]", fontsize=14)

    plt.suptitle(f"Pressure Distribution - {run_id}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pressure_plot_path = os.path.join(run_plot_dir, f"pressure_distribution_{run_id}.png")
    plt.savefig(pressure_plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close(fig_p)
    logger.info(f"Saved Pressure distribution plots to '{pressure_plot_path}'.")

    # =========================================
    # WSS Magnitude Distribution Plots with Shared Colorbars
    # =========================================

    fig_wss, axs_wss = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=False)

    # Define overall min and max for WSS magnitude to ensure consistent color mapping
    magnitude_min = min(wss_magnitude_true.min(), wss_magnitude_pred.min())
    magnitude_max = max(wss_magnitude_true.max(), wss_magnitude_pred.max())

    # Plot CFD WSS Magnitude (XY Plane)
    sc1_wss = axs_wss[0, 0].scatter(x_sample, y_sample, c=wss_magnitude_true, cmap="inferno", vmin=magnitude_min, vmax=magnitude_max, s=5, alpha=0.8)
    axs_wss[0, 0].set_title("CFD WSS Magnitude (XY Plane)", fontsize=16)
    axs_wss[0, 0].set_xlabel("X [m]", fontsize=14)
    axs_wss[0, 0].set_ylabel("Y [m]", fontsize=14)

    # Plot PINN WSS Magnitude (XY Plane)
    sc2_wss = axs_wss[0, 1].scatter(x_sample, y_sample, c=wss_magnitude_pred, cmap="inferno", vmin=magnitude_min, vmax=magnitude_max, s=5, alpha=0.8)
    axs_wss[0, 1].set_title("PINN WSS Magnitude (XY Plane)", fontsize=16)
    axs_wss[0, 1].set_xlabel("X [m]", fontsize=14)
    axs_wss[0, 1].set_ylabel("Y [m]", fontsize=14)

    # Plot CFD WSS Magnitude (XZ Plane)
    sc3_wss = axs_wss[1, 0].scatter(x_sample, z_sample, c=wss_magnitude_true, cmap="inferno", vmin=magnitude_min, vmax=magnitude_max, s=5, alpha=0.8)
    axs_wss[1, 0].set_title("CFD WSS Magnitude (XZ Plane)", fontsize=16)
    axs_wss[1, 0].set_xlabel("X [m]", fontsize=14)
    axs_wss[1, 0].set_ylabel("Z [m]", fontsize=14)

    # Plot PINN WSS Magnitude (XZ Plane)
    sc4_wss = axs_wss[1, 1].scatter(x_sample, z_sample, c=wss_magnitude_pred, cmap="inferno", vmin=magnitude_min, vmax=magnitude_max, s=5, alpha=0.8)
    axs_wss[1, 1].set_title("PINN WSS Magnitude (XZ Plane)", fontsize=16)
    axs_wss[1, 1].set_xlabel("X [m]", fontsize=14)
    axs_wss[1, 1].set_ylabel("Z [m]", fontsize=14)

    # Add a single colorbar for all WSS magnitude plots
    cbar_wss = fig_wss.colorbar(sc1_wss, ax=axs_wss, orientation='vertical', fraction=0.02, pad=0.04)
    cbar_wss.set_label("WSS Magnitude [Pa]", fontsize=14)

    plt.suptitle(f"WSS Magnitude Distribution - {run_id}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    wss_magnitude_plot_path = os.path.join(run_plot_dir, f"wss_magnitude_distribution_{run_id}.png")
    plt.savefig(wss_magnitude_plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close(fig_wss)
    logger.info(f"Saved WSS Magnitude distribution plots to '{wss_magnitude_plot_path}'.")

    # =========================================
    # Histogram Variation of WSS from PINNs and CFD
    # =========================================

    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    ax_hist.hist(wss_magnitude_true, bins=50, alpha=0.5, label='CFD', color='blue', density=True)
    ax_hist.hist(wss_magnitude_pred, bins=50, alpha=0.5, label='PINN', color='red', density=True)
    ax_hist.set_xlabel("WSS Magnitude [Pa]", fontsize=14)
    ax_hist.set_ylabel("Density", fontsize=14)
    ax_hist.set_title(f"WSS Magnitude Distribution Histogram - {run_id}", fontsize=16)
    ax_hist.legend(fontsize=12)
    ax_hist.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    histogram_plot_path = os.path.join(run_plot_dir, f"wss_magnitude_histogram_{run_id}.png")
    plt.savefig(histogram_plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close(fig_hist)
    logger.info(f"Saved WSS Magnitude histogram plot to '{histogram_plot_path}'.")


def plot_wss_line_profiles(
    dataset: CFDDataset,
    models: Dict[str, nn.Module],
    config: Config,
    logger: logging.Logger,
    run_id: str,
    fixed_y: float = 0.5,
    fixed_z: float = 0.5
):
    """
    Plots 1D profiles of WSS components and their magnitude along a specified line.

    This provides a quantitative comparison between CFD data and PINN predictions along a slice.

    Args:
        dataset (CFDDataset): Dataset instance containing CFD data.
        models (Dict[str, nn.Module]): Dictionary of trained PINN models.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.
        fixed_y (float, optional): Y-coordinate to fix for the line profile. Defaults to 0.5.
        fixed_z (float, optional): Z-coordinate to fix for the line profile. Defaults to 0.5.
    """
    for m in models.values():
        m.eval()

    # Convenient references to WSS models
    model_tau_x = models["tau_x"]
    model_tau_y = models["tau_y"]
    model_tau_z = models["tau_z"]

    # Extract a subset of data along y=fixed_y and z=fixed_z with increased tolerance
    df = dataset.data
    tol = 1e-3  # Increased tolerance for floating point comparison
    df_line = df[
        (np.abs(df[dataset.scaler_columns["features"][1]] - fixed_y) < tol) &
        (np.abs(df[dataset.scaler_columns["features"][2]] - fixed_z) < tol)
    ].copy()

    if df_line.empty:
        logger.warning(f"No data found for y={fixed_y}, z={fixed_z}. WSS line profile won't be plotted.")
        return

    # Normalize coordinates for network input
    features_scaled = dataset.scalers["features"].transform(df_line[dataset.scaler_columns["features"]])
    time_scaled = dataset.scalers["time"].transform(df_line[[dataset.scaler_columns["time"]]]).flatten()

    # Prepare tensors for prediction
    x_tensor = torch.tensor(features_scaled[:, 0], dtype=torch.float32, device=config.device).unsqueeze(1)
    y_tensor = torch.tensor(features_scaled[:, 1], dtype=torch.float32, device=config.device).unsqueeze(1)
    z_tensor = torch.tensor(features_scaled[:, 2], dtype=torch.float32, device=config.device).unsqueeze(1)
    t_tensor = torch.tensor(time_scaled, dtype=torch.float32, device=config.device).unsqueeze(1)

    with torch.no_grad():
        tau_x_pred = model_tau_x(x_tensor, y_tensor, z_tensor, t_tensor).cpu().numpy().flatten()
        tau_y_pred = model_tau_y(x_tensor, y_tensor, z_tensor, t_tensor).cpu().numpy().flatten()
        tau_z_pred = model_tau_z(x_tensor, y_tensor, z_tensor, t_tensor).cpu().numpy().flatten()

    # Inverse transform predictions to original scale
    tau_x_pred = dataset.scalers["wall_shear_x"].inverse_transform(tau_x_pred.reshape(-1, 1)).flatten()
    tau_y_pred = dataset.scalers["wall_shear_y"].inverse_transform(tau_y_pred.reshape(-1, 1)).flatten()
    tau_z_pred = dataset.scalers["wall_shear_z"].inverse_transform(tau_z_pred.reshape(-1, 1)).flatten()

    # True values from CFD data
    tau_x_true = df_line[dataset.scaler_columns["wall_shear_x"]].values
    tau_y_true = df_line[dataset.scaler_columns["wall_shear_y"]].values
    tau_z_true = df_line[dataset.scaler_columns["wall_shear_z"]].values

    # Original X coordinates for plotting
    X_original = df_line[dataset.scaler_columns["features"][0]].values

    # Compute WSS magnitudes
    wss_pred_magnitude = np.sqrt(tau_x_pred**2 + tau_y_pred**2 + tau_z_pred**2)
    wss_true_magnitude = np.sqrt(tau_x_true**2 + tau_y_true**2 + tau_z_true**2)

    # Sort data by X for smooth plotting
    sorted_indices = np.argsort(X_original)
    X_original_sorted = X_original[sorted_indices]
    tau_x_pred_sorted = tau_x_pred[sorted_indices]
    tau_y_pred_sorted = tau_y_pred[sorted_indices]
    tau_z_pred_sorted = tau_z_pred[sorted_indices]
    tau_x_true_sorted = tau_x_true[sorted_indices]
    tau_y_true_sorted = tau_y_true[sorted_indices]
    tau_z_true_sorted = tau_z_true[sorted_indices]
    wss_pred_magnitude_sorted = wss_pred_magnitude[sorted_indices]
    wss_true_magnitude_sorted = wss_true_magnitude[sorted_indices]

    # Plot WSS Components and Magnitude
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(X_original_sorted, tau_x_true_sorted, 'k-', label='CFD')
    axs[0, 0].plot(X_original_sorted, tau_x_pred_sorted, 'r--', label='PINN')
    axs[0, 0].set_xlabel("X [m]")
    axs[0, 0].set_ylabel("Tau_x [Pa]")
    axs[0, 0].set_title(f"Tau_x along line (y={fixed_y}, z={fixed_z})")
    axs[0, 0].legend()

    axs[0, 1].plot(X_original_sorted, tau_y_true_sorted, 'k-', label='CFD')
    axs[0, 1].plot(X_original_sorted, tau_y_pred_sorted, 'r--', label='PINN')
    axs[0, 1].set_xlabel("X [m]")
    axs[0, 1].set_ylabel("Tau_y [Pa]")
    axs[0, 1].set_title(f"Tau_y along line (y={fixed_y}, z={fixed_z})")
    axs[0, 1].legend()

    axs[1, 0].plot(X_original_sorted, tau_z_true_sorted, 'k-', label='CFD')
    axs[1, 0].plot(X_original_sorted, tau_z_pred_sorted, 'r--', label='PINN')
    axs[1, 0].set_xlabel("X [m]")
    axs[1, 0].set_ylabel("Tau_z [Pa]")
    axs[1, 0].set_title(f"Tau_z along line (y={fixed_y}, z={fixed_z})")
    axs[1, 0].legend()

    axs[1, 1].plot(X_original_sorted, wss_true_magnitude_sorted, 'k-', label='CFD')
    axs[1, 1].plot(X_original_sorted, wss_pred_magnitude_sorted, 'r--', label='PINN')
    axs[1, 1].set_xlabel("X [m]")
    axs[1, 1].set_ylabel("WSS Magnitude [Pa]")
    axs[1, 1].set_title(f"WSS Magnitude along line (y={fixed_y}, z={fixed_z})")
    axs[1, 1].legend()

    plt.tight_layout()
    run_plot_dir = os.path.join(config.plot_dir, run_id)
    os.makedirs(run_plot_dir, exist_ok=True)
    line_profile_plot_path = os.path.join(run_plot_dir, f"wss_line_profiles_{run_id}.png")
    plt.savefig(line_profile_plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved WSS line profiles to '{line_profile_plot_path}'.")


def plot_wss_histogram(
    dataset: CFDDataset,
    models: Dict[str, nn.Module],
    config: Config,
    logger: logging.Logger,
    run_id: str,
):
    """
    Plots a histogram comparing the distribution of WSS magnitudes between CFD data and PINN predictions.

    Args:
        dataset (CFDDataset): Dataset instance containing CFD data.
        models (Dict[str, nn.Module]): Dictionary of trained PINN models.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.
    """
    for m in models.values():
        m.eval()

    # Convenient references to WSS models
    model_tau_x = models["tau_x"]
    model_tau_y = models["tau_y"]
    model_tau_z = models["tau_z"]

    variables = [
        "wall_shear_x",
        "wall_shear_y",
        "wall_shear_z",
    ]

    predictions = {var: [] for var in variables}
    truths = {var: [] for var in variables}

    x_sample = dataset.x.numpy()
    y_sample = dataset.y.numpy()
    z_sample = dataset.z.numpy()
    t_sample = dataset.t.numpy()

    batch_size = 1024
    num_samples = len(x_sample)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = i + batch_size
            batch_x = torch.tensor(x_sample[i:end], dtype=torch.float32, device=config.device)
            batch_y = torch.tensor(y_sample[i:end], dtype=torch.float32, device=config.device)
            batch_z = torch.tensor(z_sample[i:end], dtype=torch.float32, device=config.device)
            batch_t = torch.tensor(t_sample[i:end], dtype=torch.float32, device=config.device)

            tau_x_pred = model_tau_x(batch_x, batch_y, batch_z, batch_t)
            tau_y_pred = model_tau_y(batch_x, batch_y, batch_z, batch_t)
            tau_z_pred = model_tau_z(batch_x, batch_y, batch_z, batch_t)

            predictions["wall_shear_x"].append(tau_x_pred.cpu().numpy())
            predictions["wall_shear_y"].append(tau_y_pred.cpu().numpy())
            predictions["wall_shear_z"].append(tau_z_pred.cpu().numpy())

            truths["wall_shear_x"].append(dataset.tau_x[i:end].numpy())
            truths["wall_shear_y"].append(dataset.tau_y[i:end].numpy())
            truths["wall_shear_z"].append(dataset.tau_z[i:end].numpy())

            # Cleanup to free GPU memory
            del batch_x, batch_y, batch_z, batch_t
            del tau_x_pred, tau_y_pred, tau_z_pred
            torch.cuda.empty_cache()

    # Convert lists to arrays and inverse transform to original scale
    for var in variables:
        predictions[var] = np.concatenate(predictions[var], axis=0)
        truths[var] = np.concatenate(truths[var], axis=0)
        predictions[var] = dataset.scalers[var].inverse_transform(predictions[var].reshape(-1, 1)).flatten()
        truths[var] = dataset.scalers[var].inverse_transform(truths[var].reshape(-1, 1)).flatten()

    # Compute WSS Magnitudes
    wss_pred_magnitude = np.sqrt(
        predictions["wall_shear_x"]**2 +
        predictions["wall_shear_y"]**2 +
        predictions["wall_shear_z"]**2
    )
    wss_true_magnitude = np.sqrt(
        truths["wall_shear_x"]**2 +
        truths["wall_shear_y"]**2 +
        truths["wall_shear_z"]**2
    )

    # Plot histogram
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    ax_hist.hist(wss_true_magnitude, bins=50, alpha=0.5, label='CFD', color='blue', density=True)
    ax_hist.hist(wss_pred_magnitude, bins=50, alpha=0.5, label='PINN', color='red', density=True)
    ax_hist.set_xlabel("WSS Magnitude [Pa]", fontsize=14)
    ax_hist.set_ylabel("Density", fontsize=14)
    ax_hist.set_title(f"WSS Magnitude Distribution Histogram - {run_id}", fontsize=16)
    ax_hist.legend(fontsize=12)
    ax_hist.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    run_plot_dir = os.path.join(config.plot_dir, run_id)
    histogram_plot_path = os.path.join(run_plot_dir, f"wss_magnitude_histogram_{run_id}.png")
    plt.savefig(histogram_plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close(fig_hist)
    logger.info(f"Saved WSS Magnitude histogram plot to '{histogram_plot_path}'.")


# =========================================
# 10. Training Loop
# =========================================

def train_pinn(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    config: Config,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    early_stopping: EarlyStopping,
    logger: logging.Logger,
    run_id: str,
):
    """
    Trains the PINN models using the provided DataLoader and configuration.

    The training process incorporates physics-based loss, boundary condition loss,
    data loss, and inlet condition loss, each weighted by self-adaptive parameters.

    Args:
        models (Dict[str, nn.Module]): Dictionary of PINN models.
        dataloader (DataLoader): DataLoader for training data.
        config (Config): Configuration object.
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler instance.
        early_stopping (EarlyStopping): EarlyStopping instance.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.

    Returns:
        Dict[str, list]: History of losses recorded during training.
    """
    scaler = GradScaler()
    loss_history = {"total": [], "physics": [], "boundary": [], "data": [], "inlet": []}

    epochs = config.epochs
    rho = config.rho
    mu = config.mu

    logger.info("Starting training loop for multi-model PINNs.")

    # Convenient references to each model
    model_p = models["p"]
    model_u = models["u"]
    model_v = models["v"]
    model_w = models["w"]
    model_tau_x = models["tau_x"]
    model_tau_y = models["tau_y"]
    model_tau_z = models["tau_z"]

    for epoch in tqdm(range(1, epochs + 1), desc="Training", leave=False):
        # Set models to training mode
        for m in models.values():
            m.train()

        epoch_loss_total = 0.0
        epoch_loss_physics = 0.0
        epoch_loss_boundary = 0.0
        epoch_loss_data = 0.0
        epoch_loss_inlet = 0.0

        for batch in dataloader:
            (
                x_batch,
                y_batch,
                z_batch,
                t_batch,
                p_true,
                u_true,
                v_true,
                w_true,
                tau_x_true,
                tau_y_true,
                tau_z_true,
                is_boundary
            ) = batch

            # Move data to the configured device and enable gradients for inputs
            x_batch = x_batch.to(config.device).requires_grad_(True)
            y_batch = y_batch.to(config.device).requires_grad_(True)
            z_batch = z_batch.to(config.device).requires_grad_(True)
            t_batch = t_batch.to(config.device).requires_grad_(True)

            p_true = p_true.to(config.device)
            u_true = u_true.to(config.device)
            v_true = v_true.to(config.device)
            w_true = w_true.to(config.device)
            tau_x_true = tau_x_true.to(config.device)
            tau_y_true = tau_y_true.to(config.device)
            tau_z_true = tau_z_true.to(config.device)
            is_boundary = is_boundary.to(config.device)

            optimizer.zero_grad()

            with autocast():
                # Forward pass for each variable
                p_pred = model_p(x_batch, y_batch, z_batch, t_batch)
                u_pred = model_u(x_batch, y_batch, z_batch, t_batch)
                v_pred = model_v(x_batch, y_batch, z_batch, t_batch)
                w_pred = model_w(x_batch, y_batch, z_batch, t_batch)
                tau_x_pred = model_tau_x(x_batch, y_batch, z_batch, t_batch)
                tau_y_pred = model_tau_y(x_batch, y_batch, z_batch, t_batch)
                tau_z_pred = model_tau_z(x_batch, y_batch, z_batch, t_batch)

                # Compute individual loss components
                loss_physics = compute_physics_loss(
                    p_pred, u_pred, v_pred, w_pred,
                    x_batch, y_batch, z_batch, t_batch,
                    rho, mu
                )

                # Boundary condition loss (no-slip) for velocity components
                if is_boundary.sum() > 0:
                    u_bc_pred = u_pred[is_boundary]
                    v_bc_pred = v_pred[is_boundary]
                    w_bc_pred = w_pred[is_boundary]
                    loss_boundary = compute_boundary_loss(u_bc_pred, v_bc_pred, w_bc_pred)
                else:
                    loss_boundary = torch.tensor(0.0, device=config.device)

                # Data loss for all flow variables
                loss_data = compute_data_loss(
                    p_pred, p_true,
                    u_pred, u_true,
                    v_pred, v_true,
                    w_pred, w_true,
                    tau_x_pred, tau_x_true,
                    tau_y_pred, tau_y_true,
                    tau_z_pred, tau_z_true
                )

                # Inlet condition loss based on a sinusoidal profile
                loss_inlet = compute_inlet_loss(u_pred, v_pred, w_pred, t_batch)

                # Retrieve self-adaptive weights from the PressurePINN
                lambda_physics = torch.exp(model_p.log_lambda_physics)
                lambda_boundary = torch.exp(model_p.log_lambda_boundary)
                lambda_data = torch.exp(model_p.log_lambda_data)
                lambda_inlet = torch.exp(model_p.log_lambda_inlet)

                # Total loss: sum of weighted individual losses
                total_loss = (
                    lambda_physics * loss_physics +
                    lambda_boundary * loss_boundary +
                    lambda_data * loss_data +
                    lambda_inlet * loss_inlet
                )

            # Backpropagation with mixed precision
            scaler.scale(total_loss).backward()

            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model_p.parameters()) +
                list(model_u.parameters()) +
                list(model_v.parameters()) +
                list(model_w.parameters()) +
                list(model_tau_x.parameters()) +
                list(model_tau_y.parameters()) +
                list(model_tau_z.parameters()),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            # Accumulate losses for the epoch
            epoch_loss_total += total_loss.item()
            epoch_loss_physics += loss_physics.item()
            epoch_loss_boundary += loss_boundary.item()
            epoch_loss_data += loss_data.item()
            epoch_loss_inlet += loss_inlet.item()

            # Cleanup to free GPU memory
            del (x_batch, y_batch, z_batch, t_batch, p_true, u_true, v_true, w_true,
                 tau_x_true, tau_y_true, tau_z_true, is_boundary,
                 p_pred, u_pred, v_pred, w_pred, tau_x_pred, tau_y_pred, tau_z_pred,
                 loss_physics, loss_boundary, loss_data, loss_inlet, total_loss)
            torch.cuda.empty_cache()

        # Update the learning rate scheduler
        scheduler.step()

        # Compute average losses for the epoch
        avg_loss_total = epoch_loss_total / len(dataloader)
        avg_loss_physics = epoch_loss_physics / len(dataloader)
        avg_loss_boundary = epoch_loss_boundary / len(dataloader)
        avg_loss_data = epoch_loss_data / len(dataloader)
        avg_loss_inlet = epoch_loss_inlet / len(dataloader)

        # Record loss history
        loss_history["total"].append(avg_loss_total)
        loss_history["physics"].append(avg_loss_physics)
        loss_history["boundary"].append(avg_loss_boundary)
        loss_history["data"].append(avg_loss_data)
        loss_history["inlet"].append(avg_loss_inlet)

        # Early stopping check
        early_stopping(avg_loss_total, models, optimizer, scheduler, run_id, config.model_dir)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Periodic logging for monitoring
        if epoch % 50 == 0 or epoch == 1:
            log_message = (
                f"Epoch {epoch}/{epochs} - Total: {avg_loss_total:.6f}, "
                f"Physics: {avg_loss_physics:.6f}, Boundary: {avg_loss_boundary:.6f}, "
                f"Data: {avg_loss_data:.6f}, Inlet: {avg_loss_inlet:.6f}"
            )
            logger.info(log_message)

            # Log current self-adaptive weights
            log_weights = (
                f"Lambda Physics: {lambda_physics.item():.4f} "
                f"Lambda Boundary: {lambda_boundary.item():.4f} "
                f"Lambda Data: {lambda_data.item():.4f} "
                f"Lambda Inlet: {lambda_inlet.item():.4f}"
            )
            logger.info(log_weights)

    logger.info("Training completed.")
    return loss_history

# =========================================
# 12. Main Experiment Loop
# =========================================

def main():
    """
    Main function to execute the PINN experiment workflow.

    This includes setting up configurations, loading data, initializing models,
    training, evaluating, and generating visualizations for each dataset.
    """
    # Global config
    config = Config()

    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.random_seed)

    # Create necessary directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    os.makedirs(config.metrics_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.processed_data_dir, exist_ok=True)

    # Temporary logger for initial data loading
    temp_run_id = "temp"
    temp_logger = setup_logging(temp_run_id, config.metrics_dir, config)
    datasets = load_data(config, temp_logger)
    del temp_logger  # Remove temporary logger

    # Iterate over each dataset for separate experiments
    for dataset_name, df in datasets.items():
        run_id = dataset_name
        config.run_id = run_id

        # Initialize logger for the current run
        logger = setup_logging(run_id, config.metrics_dir, config)

        # Initialize scalers for the current dataset
        scalers = {}
        logger.info("Fitting scalers on the data.")
        feature_cols = config.scaler_columns["features"]
        scalers["features"] = MinMaxScaler()
        scalers["features"].fit(df[feature_cols])
        logger.info(f"Fitted scaler for 'features'.")

        scalers["time"] = MinMaxScaler()
        scalers["time"].fit(df[[config.scaler_columns["time"]]])
        logger.info(f"Fitted scaler for 'time'.")

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
            logger.info(f"Fitted scaler for '{var}'.")

        # Prepare dataset and dataloader
        dataset = CFDDataset(df, scalers, config.scaler_columns)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        logger.info(f"Dataset and DataLoader created with {len(dataset)} samples.")

        # Initialize the 7 PINN models
        models = initialize_models(config, logger)

        # Setup optimizer and scheduler
        optimizer, scheduler = setup_optimizer_scheduler(models, config, logger)

        # Initialize Early Stopping
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            logger=logger,
        )

        # Train the models
        loss_history = train_pinn(
            models,
            dataloader,
            config,
            optimizer,
            scheduler,
            early_stopping,
            logger,
            run_id
        )

        # Evaluate the trained models
        r2_scores, nrmse_scores, mae_scores, total_mae = evaluate_pinn(
            models,
            dataloader,
            dataset,
            config,
            logger,
            run_id
        )

        # Generate visualizations
        plot_pressure_and_wss_magnitude_distribution(
            dataset,
            models,
            config,
            logger,
            run_id
        )
        plot_wss_line_profiles(
            dataset,
            models,
            config,
            logger,
            run_id,
            fixed_y=0.5,
            fixed_z=0.5
        )
        plot_wss_histogram(
            dataset,
            models,
            config,
            logger,
            run_id
        )
        plot_loss_curves(
            loss_history,
            config,
            logger,
            run_id,
            dataset_name
        )

        # Save the final model checkpoint
        final_model_path = os.path.join(config.model_dir, run_id, f"final_model_{run_id}.pt")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        checkpoint = {}
        for key, model in models.items():
            checkpoint[f"{key}_state_dict"] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint['loss_history'] = loss_history
        torch.save(checkpoint, final_model_path)
        logger.info(f"Saved final model checkpoint to '{final_model_path}'.")

        logger.info(f"Completed experiment for dataset '{run_id}'.")

    # Final logging after all experiments
    final_run_id = "final"
    final_logger = setup_logging(final_run_id, config.metrics_dir, config)
    final_logger.info("All experiments completed successfully.")

if __name__ == "__main__":
    main()
