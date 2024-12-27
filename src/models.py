# src/models.py
import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, inplace: bool = True, beta: float = 1.0):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)

    def forward(self, x):
        if self.inplace:
            return x.mul_(torch.sigmoid(self.beta * x))
        else:
            return x * torch.sigmoid(self.beta * x)

def init_weights(m):
    """
    Initializes weights for linear layers using Kaiming Normal initialization.

    Args:
        m (nn.Module): Neural network module.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

class BasePINN(nn.Module):
    """
    A generic PINN that takes (x, y, z, t) as input and returns <output_size> outputs.
    """
    def __init__(self, output_size=1, num_layers=8, units_per_layer=64, use_batch_norm=False):
        super(BasePINN, self).__init__()
        layers = []
        input_size = 4  # x, y, z, t
        for i in range(num_layers):
            in_dim = input_size if i == 0 else units_per_layer
            layers.append(nn.Linear(in_dim, units_per_layer))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units_per_layer))
            layers.append(Swish(inplace=False))
        layers.append(nn.Linear(units_per_layer, output_size))
        self.network = nn.Sequential(*layers)
        
        self.apply(init_weights)

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        return self.network(inputs)

class PressurePINN(BasePINN):
    """Predicts pressure from (x, y, z, t)."""
    pass

class UVelocityPINN(BasePINN):
    """Predicts velocity u from (x, y, z, t)."""
    pass

class VVelocityPINN(BasePINN):
    """Predicts velocity v from (x, y, z, t)."""
    pass

class WVelocityPINN(BasePINN):
    """Predicts velocity w from (x, y, z, t)."""
    pass

class TauXPINN(BasePINN):
    """Predicts wall shear stress x-component."""
    pass

class TauYPINN(BasePINN):
    """Predicts wall shear stress y-component."""
    pass

class TauZPINN(BasePINN):
    """Predicts wall shear stress z-component."""
    pass
