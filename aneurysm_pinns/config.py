# aneurysm_pinns/config.py

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Config:
    """
    Configuration Class for PINN Experiments.
    """
    # Directory Paths
    model_dir: str = os.path.join('models')                         
    plot_dir: str = os.path.join('figures')                          
    metrics_dir: str = os.path.join('reports', 'metrics')               
    data_dir: str = os.path.join('data')                             
    processed_data_dir: str = os.path.join(data_dir, 'processed')         

    # Experiment Parameters
    categories: list = field(default_factory=lambda: ["aneurysm", "healthy"])  
    phases: list = field(default_factory=lambda: ["systolic", "diastolic"])     

    run_id: str = field(init=False)  # Unique identifier for each run

    # Reproducibility
    random_seed: int = 142  

    # Device Configuration
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"  

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
    epochs: int = 500                 
    batch_size: int = 1024             
    learning_rate: float = 1e-4        
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    })
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "step_size": 200,
        "gamma": 0.9,
    })

    # Physical Parameters for Navier-Stokes
    rho: float = 1060.0  
    mu: float = 0.0035    

    # Early Stopping
    early_stopping_patience: int = 5        
    early_stopping_min_delta: float = 1e-6  

    # Neural Network Architecture
    use_batch_norm: bool = True    
    num_layers: int = 10            
    units_per_layer: int = 64       

    # Plotting Settings
    plot_resolution: int = 300

    # DataLoader Optimization
    num_workers: int = 0             
    pin_memory: bool = field(init=False)

    def __post_init__(self):
        self.pin_memory = self.device.startswith("cuda")
