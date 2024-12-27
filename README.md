
# Anuerysm Transient Flow PINNs

## Overview

Anuerysm Transient Flow PINNs is a Physics-Informed Neural Networks (PINNs) project designed to model transient flow in aneurysms. The project leverages PyTorch for model development and training.

## Table of Contents

- [Anuerysm Transient Flow PINNs](#anuerysm-transient-flow-pinns)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Data Processing](#data-processing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [License](#license)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Anuerysm_transientFlow_PINNs.git
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
   python src/main_experiment.py
   ```

   This will initialize the models, train them using the processed data, and save the trained models along with evaluation metrics.

4. **Evaluate the Models**

   Evaluation metrics are automatically generated during training and saved in the `metrics/` directory.

## Project Structure

```
Anuerysm_transientFlow_PINNs/
├── data/
│   ├── raw/
│   │   └── WSS_data/
│   └── processed/
├── src/
│   ├── data_processing.py
│   ├── datasets.py
│   ├── evaluate.py
│   ├── main_experiment.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── logs/
├── models/
├── metrics/
├── plots/
├── requirements.txt
└── README.md
```

## Data Processing

The `data_processing.py` script handles loading, cleaning, and transforming the raw data. It ensures that the spatial coordinates are correctly named `'x'`, `'y'`, and `'z'` for consistency across the project.

## Training

The training pipeline is managed by `main_experiment.py`, which initializes models, optimizers, and schedulers. It also handles logging and early stopping based on validation loss.

## Evaluation

After training, models are evaluated using `evaluate.py`, which computes metrics such as R², NRMSE, and MAE. Results are saved for further analysis.

## License

This project is licensed under the MIT License.