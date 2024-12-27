# src/data_processing.py
import pandas as pd
import os
import numpy as np
from io import StringIO

def load_data_with_metadata(file_path, data_start_keyword="[Data]", delimiter=","):
    """
    Load data from a CSV file with metadata sections. Skips lines until the data section starts.

    Args:
        file_path (str): Path to the file.
        data_start_keyword (str): The keyword marking the start of the data section.
        delimiter (str): The delimiter for the data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the starting line for the data section
    try:
        start_index = next(
            i for i, line in enumerate(lines) if data_start_keyword in line
        ) + 1  # Data starts the line after the keyword
        data_section = "".join(lines[start_index:])
        df = pd.read_csv(StringIO(data_section), delimiter=delimiter)
        print(f"Loaded data from '{data_start_keyword}' in '{os.path.basename(file_path)}'.")
    except StopIteration:
        # If the data_start_keyword is not found, load the entire file
        df = pd.read_csv(file_path, delimiter=delimiter)
        print(f"Loaded entire file '{os.path.basename(file_path)}' without using '{data_start_keyword}'.")

    return df

def clean_and_convert(df):
    """
    Cleans a dataframe by stripping whitespace from column names and converting numeric columns to floats.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Strip leading and trailing whitespace from column names
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip()
    cleaned_columns = df.columns.tolist()
    if original_columns != cleaned_columns:
        print("Stripped whitespace from column names.")

    # Convert all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def process_all_datasets(file_paths, processed_dir="../../data/processed"):
    """
    Process multiple datasets: loads, cleans, and assigns time values
    into a structured directory based on condition and phase.

    Args:
        file_paths (dict): Dictionary mapping dataset names to file paths.
        processed_dir (str): Directory to save processed data.

    Returns:
        dict: Dictionary mapping dataset names to processed DataFrames.
    """
    processed_data = {}

    # Define cardiac cycle parameters
    cardiac_cycle_duration = 0.5  # seconds
    systolic_duration = 0.218     # seconds
    diastolic_duration = cardiac_cycle_duration - systolic_duration  # 0.282 s
    num_cycles = 4
    total_time = cardiac_cycle_duration * num_cycles  # 2 seconds

    # Time intervals for systolic and diastolic phases
    systolic_times = []
    diastolic_times = []
    for n in range(num_cycles):
        systolic_start = n * cardiac_cycle_duration
        systolic_end = systolic_start + systolic_duration
        diastolic_start = systolic_end
        diastolic_end = (n + 1) * cardiac_cycle_duration
        systolic_times.append((systolic_start, systolic_end))
        diastolic_times.append((diastolic_start, diastolic_end))

    for name, path in file_paths.items():
        print(f"\nProcessing '{name}' from '{path}'...")
        try:
            # Load data
            df = load_data_with_metadata(path)
            df_cleaned = clean_and_convert(df)

            # Add this section to rename spatial coordinate columns
            # Ensure that the DataFrame has 'x', 'y', 'z' columns
            # Replace 'X [m]', 'Y [m]', 'Z [m]' with actual column names if they differ
            rename_mapping = {
                'X [m]': 'x',
                'Y [m]': 'y',
                'Z [m]': 'z'
            }
            df_cleaned.rename(columns=rename_mapping, inplace=True)

            # Determine condition and phase from the dataset name
            parts = name.split('_')
            if len(parts) >= 3:
                phase = parts[1].lower()  # 'systolic' or 'diastolic'
                condition = '_'.join(parts[2:]).lower()
            elif len(parts) == 2:
                phase = parts[1].lower()
                condition = 'healthy'
            else:
                phase = 'unknown_phase'
                condition = 'healthy'

            model_id = parts[0]

            # Assign time data
            num_samples = len(df_cleaned)
            if phase == 'systolic':
                total_phase_duration = systolic_duration * num_cycles  # 0.872 s
                t_values = np.linspace(0, total_phase_duration, num_samples, endpoint=False)
                time_values = []
                cycle_index = 0
                for t in t_values:
                    while t >= (cycle_index + 1) * systolic_duration and cycle_index < num_cycles -1:
                        cycle_index += 1
                    actual_time = systolic_times[cycle_index][0] + (t - cycle_index * systolic_duration)
                    time_values.append(actual_time)
                df_cleaned['Time [ s ]'] = time_values
            elif phase == 'diastolic':
                total_phase_duration = diastolic_duration * num_cycles  # 1.128 s
                t_values = np.linspace(0, total_phase_duration, num_samples, endpoint=False)
                time_values = []
                cycle_index = 0
                for t in t_values:
                    while t >= (cycle_index + 1) * diastolic_duration and cycle_index < num_cycles -1:
                        cycle_index += 1
                    actual_time = diastolic_times[cycle_index][0] + (t - cycle_index * diastolic_duration)
                    time_values.append(actual_time)
                df_cleaned['Time [ s ]'] = time_values
            else:
                # Assign default or average time value
                df_cleaned['Time [ s ]'] = total_time / 2  # Midpoint of total simulation time

            # Patient data
            patient_data = {
                "0021": {"age": 18, "sex": "M", "inlet_diameter_cm": 3.00, "outlet_diameter_cm": 1.42},
                "0022": {"age": 17, "sex": "M", "inlet_diameter_cm": 3.00, "outlet_diameter_cm": 1.58},
                "0023": {"age": 15, "sex": "M", "inlet_diameter_cm": 2.25, "outlet_diameter_cm": 1.22},
                "0024": {"age": 16.7, "sex": "F", "inlet_diameter_cm": 2.75, "outlet_diameter_cm": 1.95},
                "0025": {"age": 18, "sex": "M", "inlet_diameter_cm": 2.25, "outlet_diameter_cm": 1.65},
                "0142": {"age": 17, "sex": "M", "inlet_diameter_cm": 2.25, "outlet_diameter_cm": 1.12},
            }
            if model_id in patient_data:
                df_cleaned['Model ID'] = model_id
                df_cleaned['Age'] = patient_data[model_id]['age']
                df_cleaned['Sex'] = patient_data[model_id]['sex']
                df_cleaned['Inlet Diameter [cm]'] = patient_data[model_id]['inlet_diameter_cm']
                df_cleaned['Outlet Diameter [cm]'] = patient_data[model_id]['outlet_diameter_cm']
            else:
                print(f"Warning: Model ID '{model_id}' not found in patient data.")

            # Save processed DataFrame
            condition_dir = os.path.join(processed_dir, condition)
            phase_dir = os.path.join(condition_dir, phase)
            os.makedirs(phase_dir, exist_ok=True)
            output_filename = f"{name}.csv"
            output_path = os.path.join(phase_dir, output_filename)
            df_cleaned.to_csv(output_path, index=False)
            print(f"Saved processed data to '{output_path}'.")

            processed_data[name] = df_cleaned
            print(f"Processed {len(df_cleaned)} records for dataset '{name}'.")

        except Exception as e:
            print(f"Error processing '{name}': {e}")

    return processed_data
