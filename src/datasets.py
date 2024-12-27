# src/datasets.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Any
from .utils import fit_scalers
import logging

class CFDDataset(Dataset):
    def __init__(self, data: pd.DataFrame, scalers: Dict[str, Any], scaler_columns: Dict[str, Any]):
        """
        Initializes the CFD Dataset.

        Args:
            data (pd.DataFrame): The processed CFD data.
            scalers (dict): Dictionary of fitted MinMaxScalers for each variable.
            scaler_columns (dict): Mapping of scaler keys to DataFrame column names.
        """
        self.scalers = scalers
        self.scaler_columns = scaler_columns
        self.data = data.copy()
        self.handle_missing_values()
        self.normalize_data()
        self.prepare_tensors()

    def handle_missing_values(self):
        """
        Handles missing values by filling NaNs with the mean of each column.
        """
        logger = logging.getLogger(__name__)
        if self.data.isnull().values.any():
            logger.warning("NaN values found in the dataset. Filling NaNs with column means.")
            self.data.fillna(self.data.mean(), inplace=True)
        # Optionally, handle infinite values
        if np.isinf(self.data.values).any():
            logger.warning("Infinite values found in the dataset. Replacing infinities with finite numbers.")
            self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.data.fillna(self.data.mean(), inplace=True)

    def normalize_data(self):
        """
        Normalizes the data using the provided scalers.
        """
        # Use scaler utilities from utils.py
        feature_cols = self.scaler_columns["features"]
        self.data[feature_cols] = self.scalers["features"].transform(self.data[feature_cols])

        # Normalize time
        self.data[[self.scaler_columns["time"]]] = self.scalers["time"].transform(
            self.data[[self.scaler_columns["time"]]]
        )

        # Normalize other variables
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
        """
        Prepares tensors for features and targets.
        """
        # Features
        feature_cols = self.scaler_columns["features"]
        self.x = torch.tensor(self.data[feature_cols[0]].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.data[feature_cols[1]].values, dtype=torch.float32).unsqueeze(1)
        self.z = torch.tensor(self.data[feature_cols[2]].values, dtype=torch.float32).unsqueeze(1)

        # Time
        self.t = torch.tensor(self.data[self.scaler_columns["time"]].values, dtype=torch.float32).unsqueeze(1)

        # Targets
        self.p = torch.tensor(self.data[self.scaler_columns["pressure"]].values, dtype=torch.float32).unsqueeze(1)
        self.u = torch.tensor(self.data[self.scaler_columns["velocity_u"]].values, dtype=torch.float32).unsqueeze(1)
        self.v = torch.tensor(self.data[self.scaler_columns["velocity_v"]].values, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(self.data[self.scaler_columns["velocity_w"]].values, dtype=torch.float32).unsqueeze(1)

        self.tau_x = torch.tensor(self.data[self.scaler_columns["wall_shear_x"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_y = torch.tensor(self.data[self.scaler_columns["wall_shear_y"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_z = torch.tensor(self.data[self.scaler_columns["wall_shear_z"]].values, dtype=torch.float32).unsqueeze(1)

        # Boundary conditions: no-slip (u, v, w = 0)
        self.is_boundary = ((self.u == 0) & (self.v == 0) & (self.w == 0)).squeeze()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
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
