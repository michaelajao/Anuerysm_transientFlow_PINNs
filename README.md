# Fluid–Structure Interaction Analysis of Arterial Aneurysms with Physics-Informed Neural Networks

<!-- ![GitHub](https://img.shields.io/github/license/michaelajao/Anuerysm_transientFlow_PINNs) -->
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
[![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0259296-blue)](https://doi.org/10.1063/5.0259296)
[![Physics of Fluids](https://img.shields.io/badge/Published%20in-Physics%20of%20Fluids-red)](https://doi.org/10.1063/5.0259296)

## Overview

This repository contains the implementation and code for the research published in **Physics of Fluids** (2025): "*Fluid–structure interaction analysis of pulsatile flow in arterial aneurysms with physics-informed neural networks and computational fluid dynamics*".

**Abstract:** Marfan syndrome (MS) is a genetic disorder often associated with the development of aortic aneurysms, leading to severe vascular complications. The progression of this condition is intricately linked to hemodynamic factors such as wall shear stress (WSS) and von Mises stress, as abnormal distributions can contribute to thrombus formation, endothelial damage, and the worsening of aneurysmal conditions. In this study, six vascular models were analyzed: four representing diseased aortas with Marfan syndrome aneurysms and two healthy aortic models for comparison. The models were sourced from Vascular Model Repository, and computational fluid dynamics (CFD) simulations were conducted using a Newtonian fluid model and the shear stress transport (SST) k-ω turbulent transitional model to evaluate WSS and von Mises stress. Fluid–structure interaction was employed to incorporate vessel wall interaction, and pulsatile inlet velocity profiles were used to simulate physiological blood flow, capturing time-dependent hemodynamic variations.

The results revealed significant differences between healthy and diseased aortic models. In healthy models, WSS was uniformly distributed, with values consistently below 40 Pa, reflecting stable vascular conditions. Conversely, the diseased models exhibited highly non-uniform WSS distributions, with notably lower values in aneurysmal regions, contributing to thrombus formation, with elevated WSS in areas like the carotid and subclavian arteries due to geometric and hemodynamic complexities. The von Mises stress analysis identified regions of heightened rupture risk, particularly on the superior side of case MS1, where both von Mises stress and WSS reached their highest values among all cases. **Physics-informed neural networks demonstrated strong agreement with CFD results while significantly reducing computational cost**, highlighting their potential for real-time clinical applications.

This project applies Physics-Informed Neural Networks (PINNs) to model and predict transient blood flow dynamics in aortic aneurysms. By combining deep learning with fundamental physics principles, the models incorporate the Navier-Stokes equations and learn from real computational fluid dynamics (CFD) data to predict pressure distributions, velocity fields, and wall shear stress in both healthy and aneurysmal blood vessels under various cardiac cycle conditions.


## Project Features

- 🧠 Modeling of both healthy and aneurysmal blood flow dynamics
- 🔄 Capturing transient flow patterns across cardiac cycles
- ⚖️ Physics-informed constraint enforcement using Navier-Stokes equations
- 📊 Comprehensive data visualization tools for flow analysis
- 📈 Performance metrics for model evaluation
- 🧪 Comparative analysis between healthy and pathological conditions

## Installation

### Prerequisites

- Python 3.10+
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

This study analyzes **six vascular models** sourced from the Vascular Model Repository:
- **Four diseased aortas** with Marfan syndrome aneurysms (Cases MS1-MS4)
- **Two healthy aortic models** for comparison (Cases 0024, 0142)

### Data Structure

The CFD simulation datasets include:

**Spatial and Flow Variables:**
- Spatial coordinates (X, Y, Z)
- Pressure values (P) [Pa]
- Velocity components (u, v, w) [m/s]
- Wall shear stress components (X, Y, Z) [Pa]

**Temporal Information:**
- Time-varying data for systolic and diastolic cardiac phases (t)
- Pulsatile inlet velocity profiles simulating physiological blood flow

**CFD Simulation Parameters:**
- Newtonian fluid model for blood
- Shear stress transport (SST) k-ω turbulent transitional model
- Fluid–structure interaction for vessel wall dynamics
- Blood density: ρ = 1060 kg/m³
- Blood dynamic viscosity: μ = 0.0035 Pa·s

All raw data files should be placed in the `data/WSS_data/` directory before processing. The system supports multiple patient cases with both healthy and aneurysmal conditions, enabling comparative hemodynamic analysis.

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

To generate visualizations of the raw data for analysis, run:

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

For each experiment run (identified by the dataset name, like "0021_diastolic_aneurysm"), the system creates:

1. **A detailed log file** named `experiment_[dataset_name].log` that records:
   - Training progress with timestamps
   - Loss values for different components (physics, boundary, data, inlet)
   - Early stopping triggers and model checkpoints
   - Dataset processing information
   - Visualization generation confirmations

2. **A metrics summary CSV file** named `metrics_summary_[dataset_name].csv` containing evaluation metrics like:
   - R² (coefficient of determination)
   - NRMSE (normalized root mean square error)
   - MAE (mean absolute error) for each output variable

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/*name of choice*`)
3. Commit your changes (`git commit -m 'Add some *name of choice*'`)
4. Push to the branch (`git push origin feature/*name of choice*`)
5. Open a Pull Request


## Citation

If you use this project in your research, please cite the original paper:

```bibtex
@article{ur2025fluid,
  title={Fluid--structure interaction analysis of pulsatile flow in arterial aneurysms with physics-informed neural networks and computational fluid dynamics},
  author = {Ur Rehman, M. Abaid and Ekici, Ozgur (Özgür Ekici) and Farooq, M. Asif and Butt, Khayam and Ajao-Olarinoye, Michael and Wang, Zhen and Liu, Haipeng},
  journal = {Physics of Fluids},
  volume = {37},
  number = {3},
  pages = {031913},
  year = {2025},
  month = {03},
  issn = {1070-6631},
  publisher = {AIP Publishing},
  doi = {10.1063/5.0259296},
  url = {https://doi.org/10.1063/5.0259296}
}
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.