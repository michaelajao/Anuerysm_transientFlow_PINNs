# Aneurysm Transient Flow PINNs

![GitHub](https://img.shields.io/github/license/michaelajao/Anuerysm_transientFlow_PINNs)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)

## Overview

Aneurysm Transient Flow PINNs is a research project that applies Physics-Informed Neural Networks (PINNs) to model and predict transient blood flow dynamics in cerebral aneurysms. Cerebral aneurysms are pathological dilations of blood vessels in the brain that can rupture and cause severe medical complications. By combining deep learning with fundamental physics principles, this project aims to create accurate and efficient models for hemodynamic simulation, potentially contributing to improved risk assessment and treatment planning.

The models in this project incorporate the Navier-Stokes equations and learn from real computational fluid dynamics (CFD) data to predict pressure distributions, velocity fields, and wall shear stress in both healthy and aneurysmal blood vessels under various cardiac cycle conditions.

## Project Features

- 🧠 Modeling of both healthy and aneurysmal blood flow dynamics
- 🔄 Capturing transient flow patterns across cardiac cycles
- ⚖️ Physics-informed constraint enforcement using Navier-Stokes equations
- 📊 Comprehensive data visualization tools for flow analysis
- 📈 Performance metrics for model evaluation
- 🧪 Comparative analysis between healthy and pathological conditions

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/michaelajao/Anuerysm_transientFlow_PINNs.git
   cd Anuerysm_transientFlow_PINNs
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Description

The project uses a collection of CFD simulation data from both healthy and aneurysmal blood vessels. The datasets include:

- Spatial coordinates (X, Y, Z)
- Pressure values (P)
- Velocity components (u, v, w)
- Wall shear stress components (X, Y, Z)
- Time-varying data for systolic and diastolic cardiac phases (t)

All raw data files should be placed in the `data/WSS_data/` directory before processing. The system supports multiple patient cases, with both healthy and aneurysmal conditions.

## Project Structure

```
Anuerysm_transientFlow_PINNs/
├── aneurysm_pinns/              # Main Python package
│   ├── __init__.py
│   ├── config.py                # Configuration settings
│   ├── dataset.py               # Data loading and processing
│   ├── main.py                  # Main execution script
│   ├── plots.py                 # Plotting utilities for model outputs
│   ├── utils.py                 # Utility functions
│   ├── visualization.py         # Data visualization tools
│   ├── modeling/                # Neural network models
│       ├── __init__.py
│       ├── model.py             # PINNs architecture 
│       ├── train.py             # Training procedures
│       ├── predict.py           # Model prediction functions
├── data/                        # Data directory
│   ├── processed/               # Processed data
│   │   ├── aneurysm/            # Aneurysmal cases
│   │   │   ├── systolic/
│   │   │   └── diastolic/ 
│   │   └── healthy/             # Healthy cases
│   │       ├── systolic/
│   │       └── diastolic/
│   └── WSS_data/                # Raw data files
├── figures/                     # Output visualizations
│   ├── data_plots/              # Data analysis visualizations
│   └── [patient_id]/            # Patient-specific results
├── models/                      # Saved model checkpoints
├── reports/                     # Performance reports
│   └── metrics/                 # Quantitative evaluation metrics
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Usage Guide

### 1. Data Processing

The first step is to process the raw CFD data:

```bash
python -m aneurysm_pinns.dataset
```

This script will:
- Load raw data from CSV files in `data/WSS_data/`
- Clean and normalize the data
- Add cardiac cycle time information
- Save processed datasets to `data/processed/`

### 2. Training PINNs Models
The `main.py` file contains the full training pipeline for the PINNs models, including data loading, model initialization, training, and evaluation. You can skip the dataset preprocessing step and run the `main.py` script directly.


```bash
python -m aneurysm_pinns.main
```

The training process will:

- Initialize neural networks for pressure, velocity components, and wall shear stress
- Apply physics-informed constraints using Navier-Stokes equations
- Enforce boundary conditions
- Save model checkpoints to the `models/` directory

### 3. Data Visualization

To generate visualizations of the raw data:

```bash
python -m aneurysm_pinns.visualization
```

This will produce various plots including:
- Wall shear stress distributions
- Velocity vector fields
- Pressure contours
- Interactive 3D representations

### 4. Configuration

The system behavior can be customized by modifying parameters in `config.py`:

- Data paths and directory structure
- Training hyperparameters (learning rate, batch size, epochs)
- Physical parameters (fluid density, viscosity)
- Neural network architecture (layer count, units per layer)
- Early stopping criteria

## Scientific Background

### Physics-Informed Neural Networks

PINNs combine neural networks with physical constraints derived from governing equations. For fluid dynamics, the models are trained to satisfy:

1. **Data-driven loss**: Minimizing error between predictions and CFD data
2. **Physics-informed loss**: Enforcing adherence to Navier-Stokes equations
3. **Boundary condition loss**: Ensuring proper behavior at domain boundaries

### Navier-Stokes Equations

The incompressible Navier-Stokes equations governing blood flow are:

- **Continuity**: ∇·u = 0
- **Momentum**: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u

Where:
- u is the velocity vector field
- p is pressure
- ρ is fluid density (1060 kg/m³ for blood)
- μ is dynamic viscosity (0.0035 Pa·s for blood)

## Results

The trained models demonstrate the ability to:

1. Accurately predict pressure distributions in vessels
2. Capture complex velocity flow patterns
3. Estimate wall shear stress distributions
4. Identify regions of high hemodynamic stress in aneurysms

Comparative analysis between healthy and aneurysmal vessels reveals significant differences in flow patterns and wall shear stress distributions.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- ## Citation

If you use this project in your research, please cite it as:

```
@software{aneurysm_transient_pinns,
  author = {Ajao, Michael},
  title = {Aneurysm Transient Flow PINNs},
  year = {2023},
  url = {https://github.com/michaelajao/Anuerysm_transientFlow_PINNs}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the collaborators who provided CFD datasets
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [SciPy](https://scipy.org/) for scientific computing tools -->
