# src/utils.py
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # Add this import

# Define the Swish activation function
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.sigmoid(x)

# Define the EarlyStopping class
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=logging.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: logging.info
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, models, optimizer, scheduler, run_id, model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, scheduler, run_id, model_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, scheduler, run_id, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, optimizer, scheduler, run_id, model_dir):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        for name, model in models.items():
            torch.save(model.state_dict(), os.path.join(model_dir, f'{name}_{run_id}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, f'optimizer_{run_id}.pt'))
        torch.save(scheduler.state_dict(), os.path.join(model_dir, f'scheduler_{run_id}.pt'))
        self.val_loss_min = val_loss

@dataclass
class Config:
    """Configuration parameters for the PINN training."""
    learning_rate: float = 1e-3
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "betas": (0.9, 0.999),
        "eps": 1e-8
    })
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "step_size": 100,
        "gamma": 0.9
    })
    epochs: int = 1000
    rho: float = 1.0  # Example value
    mu: float = 0.01  # Example value
    device: str = "cpu"
    log_dir: str = "logs"
    model_dir: str = "models"
    plot_dir: str = "plots"
    metrics_dir: str = "metrics"
    data_dir: str = "data"
    processed_data_dir: str = "data/processed"
    scaler_columns: Dict[str, Any] = field(default_factory=lambda: {
        "features": ["x", "y", "z"],
        "time": "t",
        "pressure": "p",
        "velocity_u": "u",
        "velocity_v": "v",
        "velocity_w": "w",
        "wall_shear_x": "tau_x",
        "wall_shear_y": "tau_y",
        "wall_shear_z": "tau_z"
    })
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    scaler: MinMaxScaler = field(default_factory=MinMaxScaler)
    run_id: str = ""

def setup_logging(name: str, config: Config) -> logging.Logger:
    """
    Sets up logging for the application.

    Args:
        name (str): Name of the logger.
        config (Config): Configuration object.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(os.path.join(config.log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_device() -> str:
    """Returns the available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def fit_scalers(df: pd.DataFrame, scaler_columns: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fits MinMaxScalers for specified columns.

    Args:
        df (pd.DataFrame): Dataframe to fit scalers on.
        scaler_columns (dict): Columns to apply scalers.

    Returns:
        dict: Fitted scalers.
    """
    from sklearn.preprocessing import MinMaxScaler
    scalers = {}
    for key, cols in scaler_columns.items():
        scaler = MinMaxScaler()
        scalers[key] = scaler.fit(df[cols])
    return scalers

def initialize_models(config: Config, logger: logging.Logger) -> dict:
    """
    Initializes all PINN models.

    Args:
        config (Config): Configuration object.
        logger (logging.Logger): Logger object.

    Returns:
        dict: Dictionary of initialized models.
    """
    from .models import (
        PressurePINN, UVelocityPINN, VVelocityPINN, WVelocityPINN,
        TauXPINN, TauYPINN, TauZPINN
    )
    models = {
        "pressure": PressurePINN(),
        "u_velocity": UVelocityPINN(),
        "v_velocity": VVelocityPINN(),
        "w_velocity": WVelocityPINN(),
        "tau_x": TauXPINN(),
        "tau_y": TauYPINN(),
        "tau_z": TauZPINN(),
    }
    logger.info("Initialized all PINN models.")
    return models

def setup_plotting_style():
    """
    Sets up the default matplotlib plotting style for consistency across the project.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {   
            "font.family": "serif",
            "font.size": 14,
            "figure.figsize": [10, 6],
            "text.usetex": False,
            "figure.facecolor": "white",
            "figure.autolayout": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.titlesize": 18,
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.limits": (0, 5),
            "axes.formatter.use_mathtext": True,
            "axes.formatter.useoffset": False,
            "axes.xmargin": 0,
            "axes.ymargin": 0,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "legend.loc": "best",
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "xtick.labelsize": 12,
            "xtick.direction": "in",
            "xtick.top": False,
            "ytick.labelsize": 12,
            "ytick.direction": "in",
            "ytick.right": False,
            "grid.color": "grey",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "errorbar.capsize": 4,
            "figure.subplot.wspace": 0.4,
            "figure.subplot.hspace": 0.4,
            "image.cmap": "viridis",
        }
    )
