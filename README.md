# Anuerysm Transient Flow PINNs

## Overview

Anuerysm Transient Flow PINNs is a Physics-Informed Neural Networks (PINNs) project designed to model transient flow in aneurysms. The project leverages PyTorch for model development and training, providing a robust framework for simulating and analyzing blood flow dynamics. Cerebral aneurysms are pathological dilations of blood vessels in the brain, posing significant health risks due to their potential to rupture. Understanding the transient blood flow dynamics within aneurysms is crucial for risk assessment and treatment planning. Traditional computational fluid dynamics (CFD) methods, while accurate, are computationally intensive and time-consuming.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/michaelajao/Anuerysm_transientFlow_PINNs.git
   cd Anuerysm_transientFlow_PINNs
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Data**

   Ensure that your raw data is placed in the `data/raw/WSS_data/` directory. The data should contain spatial coordinates `'X [m]'`, `'Y [m]'`, `'Z [m]'`, and other relevant flow variables.

2. **Process the Data**

   ```bash
   python src/data_processing.py
   ```

   This script will clean the data, rename columns to `'x'`, `'y'`, `'z'`, and save the processed data in `data/processed/`.

3. **Train the Models**

   ```bash
   python src/full_pinn_experiment.py
   ```

   This will initialize the models, train them using the processed data, and save the trained models along with evaluation metrics.


## Methodology

The project employs Physics-Informed Neural Networks (PINNs) to integrate CFD data with physical laws governing fluid dynamics. The methodology encompasses data preprocessing, model architecture design, training with self-adaptive loss weighting, and comprehensive evaluation using metrics such as R², NRMSE, and MAE. Boundary conditions and inlet velocity profiles are enforced to ensure realistic simulations. Visualization tools are utilized to analyze loss curves and model performance.

## Project Structure

```
Anuerysm_transientFlow_PINNs/
├── data/
│   ├── raw/
│   │   └── WSS_data/
│   │       ├── 0021 Diastolic aneurysm.csv
│   │       ├── 0021 Diastolic global.csv
│   │       └── ... (other CSV files)
│   └── processed/
│       └── ... (processed CSV files organized by condition and phase)
├── aneurysm_pinns/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── utils.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── plots.py
│   └── main.py
├── src/
|   ├──models/
│   │   └── full_pinn_experiment.py
│   ├── data
│   │   └── data_processing.py
├── logs/
├── models/
├── reports/
│   ├── metrics/
│   │   └── ... (CSV files with performance metrics)
│   └── paper/
│       ├── main.tex
│       ├── references.bib
│       └── figures/
├── figures/
├── requirements.txt
|── .gitignore
└── README.md
```

## Data Processing

The `data_processing.py` script handles loading, cleaning, and transforming the raw data. It ensures that the spatial coordinates are correctly named `'x'`, `'y'`, and `'z'` for consistency across the project.

## Training

The training pipeline is managed by `full_pinn_experiment.py`, which initializes models, optimizers, and schedulers. It also handles logging and early stopping based on validation loss.

## Results

The trained PINN models achieved high accuracy in predicting flow-related variables, as evidenced by evaluation metrics. Loss curves indicate stable convergence, and visualizations of pressure distributions and wall shear stress align closely with CFD data. Histograms and distribution plots further validate the models' performance, demonstrating their potential applicability in biomedical research.

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@misc{anuerysm2024transient,
  author = {Your Name},
  title = {Anuerysm Transient Flow PINNs},
  year = {2024},
  howpublished = {\url{https://github.com/yourusername/Anuerysm_transientFlow_PINNs}},
  note = {Accessed: YYYY-MM-DD}
}
```

## License

This project is licensed under the MIT License.
