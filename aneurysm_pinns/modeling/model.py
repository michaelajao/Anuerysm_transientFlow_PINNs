# aneurysm_pinns/modeling/model.py

import torch
import torch.nn as nn
from typing import Dict
from aneurysm_pinns.config import Config

class Swish(nn.Module):
    """
    Swish activation: x * sigmoid(beta*x), with learnable beta.
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class BasePINN(nn.Module):
    """
    Base class for PINNs: fully-connected layers + optional batch norm + Swish.
    """
    def __init__(self, config: Config, out_dim: int = 1):
        super().__init__()
        layers = []
        inp_dim = 4  # (x, y, z, t)
        for i in range(config.num_layers):
            in_features = inp_dim if i == 0 else config.units_per_layer
            layers.append(nn.Linear(in_features, config.units_per_layer))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.units_per_layer))
            layers.append(Swish())
        layers.append(nn.Linear(config.units_per_layer, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

        # Self-adaptive log-lambda for weighting
        self.log_lambda_physics = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_boundary = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_data = nn.Parameter(torch.tensor(0.0))
        self.log_lambda_inlet = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        inp = torch.cat([x, y, z, t], dim=1)
        return self.net(inp)

# Specialized PINNs
class PressurePINN(BasePINN):
    pass

class UVelocityPINN(BasePINN):
    pass

class VVelocityPINN(BasePINN):
    pass

class WVelocityPINN(BasePINN):
    pass

class TauXPINN(BasePINN):
    pass

class TauYPINN(BasePINN):
    pass

class TauZPINN(BasePINN):
    pass

def initialize_models(config: Config) -> Dict[str, nn.Module]:
    """
    Initialize PINN models for pressure, velocities, and wall shear stresses.
    """
    
    model_p = PressurePINN(config, out_dim=1).to(config.device)
    model_u = UVelocityPINN(config, out_dim=1).to(config.device)
    model_v = VVelocityPINN(config, out_dim=1).to(config.device)
    model_w = WVelocityPINN(config, out_dim=1).to(config.device)
    model_tau_x = TauXPINN(config, out_dim=1).to(config.device)
    model_tau_y = TauYPINN(config, out_dim=1).to(config.device)
    model_tau_z = TauZPINN(config, out_dim=1).to(config.device)

    print("Initialized PINN models: p, u, v, w, tau_x, tau_y, tau_z")
    return {
        "p": model_p,
        "u": model_u,
        "v": model_v,
        "w": model_w,
        "tau_x": model_tau_x,
        "tau_y": model_tau_y,
        "tau_z": model_tau_z,
    }
