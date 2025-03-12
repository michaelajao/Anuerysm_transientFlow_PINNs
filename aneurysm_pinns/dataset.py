# aneurysm_pinns/dataset.py

import os
import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, Any

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler

from aneurysm_pinns.config import Config
from aneurysm_pinns.utils import ensure_dir, load_csv, save_csv


def load_data_with_metadata(file_path: str, data_start_keyword: str = "[Data]", delimiter: str = ",") -> pd.DataFrame:
    """
    Load data from a CSV with metadata sections. Skips lines until the data section starts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    try:
        start_index = next(i for i, line in enumerate(lines) if data_start_keyword in line) + 1
        data_section = "".join(lines[start_index:])
        df = pd.read_csv(StringIO(data_section), delimiter=delimiter)
        print(f"Loaded data from '{data_start_keyword}' in '{os.path.basename(file_path)}'.")
    except StopIteration:
        df = pd.read_csv(file_path, delimiter=delimiter)
        print(f"Loaded entire file '{os.path.basename(file_path)}' (no '{data_start_keyword}' used).")

    return df


def clean_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a dataframe: strip whitespace in column names & convert columns to numeric.
    """
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip()
    if original_cols != list(df.columns):
        print("Stripped whitespace from column names.")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def process_all_datasets(file_paths: Dict[str, str], processed_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Process multiple datasets: load, clean, and assign time values
    into a structured directory based on condition/phase.
    """
    processed_data = {}

    # Cardiac cycle parameters
    cardiac_cycle_duration = 0.5
    systolic_duration = 0.218
    diastolic_duration = cardiac_cycle_duration - systolic_duration
    num_cycles = 4
    total_time = num_cycles * cardiac_cycle_duration

    systolic_times = []
    diastolic_times = []
    for n in range(num_cycles):
        start_sys = n * cardiac_cycle_duration
        end_sys = start_sys + systolic_duration
        start_dia = end_sys
        end_dia = (n + 1) * cardiac_cycle_duration
        systolic_times.append((start_sys, end_sys))
        diastolic_times.append((start_dia, end_dia))

    # Patient data
    patient_data = {
        "0021": {"age": 18, "sex": "M", "inlet_diameter_cm": 3.00, "outlet_diameter_cm": 1.42},
        "0022": {"age": 17, "sex": "M", "inlet_diameter_cm": 3.00, "outlet_diameter_cm": 1.58},
        "0023": {"age": 15, "sex": "M", "inlet_diameter_cm": 2.25, "outlet_diameter_cm": 1.22},
        "0024": {"age": 16.7, "sex": "F", "inlet_diameter_cm": 2.75, "outlet_diameter_cm": 1.95},
        "0025": {"age": 18, "sex": "M", "inlet_diameter_cm": 2.25, "outlet_diameter_cm": 1.65},
        "0142": {"age": 17, "sex": "M", "inlet_diameter_cm": 2.25, "outlet_diameter_cm": 1.12},
    }

    for name, path in file_paths.items():
        print(f"\nProcessing '{name}' from '{path}'...")
        try:
            df = load_data_with_metadata(path)
            df_cleaned = clean_and_convert(df)

            # Determine condition & phase
            parts = name.split('_')
            if len(parts) >= 3:
                phase = parts[1].lower()  # e.g. systolic or diastolic
                condition = '_'.join(parts[2:]).lower()
            elif len(parts) == 2:
                phase = parts[1].lower()
                condition = 'healthy'
            else:
                phase = 'unknown_phase'
                condition = 'healthy'

            model_id = parts[0]
            num_samples = len(df_cleaned)

            # Time assignment
            if phase == 'systolic':
                total_sys = systolic_duration * num_cycles
                t_values = np.linspace(0, total_sys, num_samples, endpoint=False)
                time_values = []
                cycle_index = 0
                for t in t_values:
                    while t >= (cycle_index + 1) * systolic_duration and cycle_index < num_cycles - 1:
                        cycle_index += 1
                    actual_t = systolic_times[cycle_index][0] + (t - cycle_index * systolic_duration)
                    time_values.append(actual_t)
                df_cleaned['Time [ s ]'] = time_values

            elif phase == 'diastolic':
                total_dia = diastolic_duration * num_cycles
                t_values = np.linspace(0, total_dia, num_samples, endpoint=False)
                time_values = []
                cycle_index = 0
                for t in t_values:
                    while t >= (cycle_index + 1) * diastolic_duration and cycle_index < num_cycles - 1:
                        cycle_index += 1
                    actual_t = diastolic_times[cycle_index][0] + (t - cycle_index * diastolic_duration)
                    time_values.append(actual_t)
                df_cleaned['Time [ s ]'] = time_values
            else:
                df_cleaned['Time [ s ]'] = total_time / 2

            # Patient-specific info
            if model_id in patient_data:
                p_data = patient_data[model_id]
                df_cleaned['Model ID'] = model_id
                df_cleaned['Age'] = p_data['age']
                df_cleaned['Sex'] = p_data['sex']
                df_cleaned['Inlet Diameter [cm]'] = p_data['inlet_diameter_cm']
                df_cleaned['Outlet Diameter [cm]'] = p_data['outlet_diameter_cm']
            else:
                print(f"Warning: Model ID '{model_id}' not in patient data dictionary.")

            condition_dir = os.path.join(processed_dir, condition)
            phase_dir = os.path.join(condition_dir, phase)
            ensure_dir(phase_dir)

            out_name = f"{name}.csv"
            out_path = os.path.join(phase_dir, out_name)
            save_csv(df_cleaned, out_path)

            processed_data[name] = df_cleaned
            print(f"Processed {len(df_cleaned)} records for '{name}'.")

        except Exception as e:
            print(f"Error processing '{name}': {e}")

    return processed_data


class CFDDataset(Dataset):
    """
    Custom PyTorch Dataset for CFD Data.
    """
    def __init__(self, data: pd.DataFrame, scalers: Dict[str, Any], scaler_columns: Dict[str, Any], tolerance: float = 1e-5):
        super().__init__()
        self.scalers = scalers
        self.scaler_columns = scaler_columns
        self.data = data.copy()
        self.normalize_data()
        self.prepare_tensors(tolerance)

    def normalize_data(self):
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
        feature_cols = self.scaler_columns["features"]
        self.x = torch.tensor(self.data[feature_cols[0]].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.data[feature_cols[1]].values, dtype=torch.float32).unsqueeze(1)
        self.z = torch.tensor(self.data[feature_cols[2]].values, dtype=torch.float32).unsqueeze(1)
        self.t = torch.tensor(self.data[self.scaler_columns["time"]].values, dtype=torch.float32).unsqueeze(1)

        self.p = torch.tensor(self.data[self.scaler_columns["pressure"]].values, dtype=torch.float32).unsqueeze(1)
        self.u = torch.tensor(self.data[self.scaler_columns["velocity_u"]].values, dtype=torch.float32).unsqueeze(1)
        self.v = torch.tensor(self.data[self.scaler_columns["velocity_v"]].values, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(self.data[self.scaler_columns["velocity_w"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_x = torch.tensor(self.data[self.scaler_columns["wall_shear_x"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_y = torch.tensor(self.data[self.scaler_columns["wall_shear_y"]].values, dtype=torch.float32).unsqueeze(1)
        self.tau_z = torch.tensor(self.data[self.scaler_columns["wall_shear_z"]].values, dtype=torch.float32).unsqueeze(1)

        epsilon = tolerance
        self.is_boundary = (
            (torch.abs(self.u) < epsilon) &
            (torch.abs(self.v) < epsilon) &
            (torch.abs(self.w) < epsilon)
        ).squeeze()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
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


def load_data(config: Config) -> Dict[str, pd.DataFrame]:
    """
    Loads and preprocesses all CFD datasets from processed directories.
    """
    processed_files = []
    for category in config.categories:
        for phase in config.phases:
            phase_dir = os.path.join(config.processed_data_dir, category, phase)
            if os.path.isdir(phase_dir):
                for f in os.listdir(phase_dir):
                    if f.endswith(".csv"):
                        processed_files.append(os.path.join(phase_dir, f))

    datasets = {}
    required_columns = [
        "X [ m ]", "Y [ m ]", "Z [ m ]", "Time [ s ]", "Pressure [ Pa ]",
        "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]",
        "Wall Shear X [ Pa ]", "Wall Shear Y [ Pa ]", "Wall Shear Z [ Pa ]"
    ]

    for file in processed_files:
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        df = load_csv(file)
        df.columns = df.columns.str.strip()
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Dataset '{dataset_name}' missing columns: {missing_cols}. Dropping those rows.")
            df = df.dropna(subset=required_columns)
        else:
            df = df.dropna(subset=required_columns)
        datasets[dataset_name] = df

    print(f"Loaded {len(datasets)} datasets from processed folder.")
    for name, df in datasets.items():
        print(f"  -> '{name}' shape: {df.shape}")
    return datasets


def process_and_save_all_datasets(file_paths: Dict[str, str], processed_dir: str):
    """
    Wrapper to process & save all datasets.
    """
    processed_data = process_all_datasets(file_paths, processed_dir)
    
    return processed_data
    print("All datasets processed and saved.")
