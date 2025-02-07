import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from aneurysm_pinns.dataset import load_data_with_metadata
from src.models.full_pinn_experiment import Config, initialize_models

def setup_logger():
    """Set up a basic logger for the analysis"""
    logger = logging.getLogger('pinn_analysis')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger

def analyze_solution_density():
    """Analyze and compare the solution point density between CFD and PINN"""
    # Load CFD data
    data_path = "data/WSS_data/0021 Diastolic aneurysm.csv"
    cfd_df = load_data_with_metadata(data_path)
    
    # Calculate CFD point density
    x_range = cfd_df['X [ m ]'].max() - cfd_df['X [ m ]'].min()
    y_range = cfd_df[' Y [ m ]'].max() - cfd_df[' Y [ m ]'].min()
    z_range = cfd_df[' Z [ m ]'].max() - cfd_df[' Z [ m ]'].min()
    volume = x_range * y_range * z_range
    cfd_density = len(cfd_df) / volume
    
    # Create a denser grid for PINN evaluation
    x = np.linspace(cfd_df['X [ m ]'].min(), cfd_df['X [ m ]'].max(), 100)
    y = np.linspace(cfd_df[' Y [ m ]'].min(), cfd_df[' Y [ m ]'].max(), 100)
    z = np.linspace(cfd_df[' Z [ m ]'].min(), cfd_df[' Z [ m ]'].max(), 100)
    pinn_points = len(x) * len(y) * len(z)
    pinn_density = pinn_points / volume
    
    print("\nSolution Density Analysis:")
    print(f"Domain Dimensions:")
    print(f"  X: {x_range*1000:.2f} mm")
    print(f"  Y: {y_range*1000:.2f} mm")
    print(f"  Z: {z_range*1000:.2f} mm")
    print(f"  Volume: {volume*1e9:.2f} mm3")
    print("\nPoint Distribution:")
    print(f"CFD Data Points: {len(cfd_df):,}")
    print(f"CFD Point Density: {cfd_density/1e6:.2f} million points/m3")
    print(f"PINN Potential Points: {pinn_points:,}")
    print(f"PINN Potential Density: {pinn_density/1e6:.2f} million points/m3")
    print(f"Density Ratio (PINN/CFD): {pinn_density/cfd_density:.2f}x")
    
    # Plot point distribution
    plt.figure(figsize=(15, 5))
    
    # CFD points
    plt.subplot(131)
    plt.scatter(cfd_df['X [ m ]'], cfd_df[' Y [ m ]'], s=1, alpha=0.5, label='CFD Points')
    plt.title('CFD Solution Points\n(Original Data)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    
    # PINN points (sample for visualization)
    plt.subplot(132)
    X, Y = np.meshgrid(x[::5], y[::5])  # Sample every 5th point for clarity
    plt.scatter(X.flatten(), Y.flatten(), s=1, alpha=0.5, label='PINN Points')
    plt.title('PINN Solution Points\n(Potential Resolution)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    
    # Overlay comparison
    plt.subplot(133)
    plt.scatter(cfd_df['X [ m ]'], cfd_df[' Y [ m ]'], s=1, alpha=0.3, label='CFD', color='blue')
    plt.scatter(X.flatten(), Y.flatten(), s=1, alpha=0.3, label='PINN', color='red')
    plt.title('Overlay Comparison')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('figures/pinn_analysis', exist_ok=True)
    plt.savefig('figures/pinn_analysis/solution_density_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_computational_cost():
    """Analyze computational cost between CFD and PINN inference"""
    # Load a trained PINN model
    config = Config()
    logger = setup_logger()
    models = initialize_models(config, logger)
    
    # Load sample data point
    data_path = "data/WSS_data/0021 Diastolic aneurysm.csv"
    df = load_data_with_metadata(data_path)
    
    # Prepare sample points for timing
    sample_sizes = [100, 1000, 10000]
    times = []
    
    print("\nComputational Cost Analysis:")
    print("PINN Inference Time Analysis:")
    
    for size in sample_sizes:
        x = torch.tensor(df['X [ m ]'].values[:size], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(df[' Y [ m ]'].values[:size], dtype=torch.float32).unsqueeze(1)
        z = torch.tensor(df[' Z [ m ]'].values[:size], dtype=torch.float32).unsqueeze(1)
        t = torch.tensor(np.zeros(size), dtype=torch.float32).unsqueeze(1)
        
        # Warm-up run
        with torch.no_grad():
            _ = models['p'](x, y, z, t)
        
        # Timing runs
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Average over 10 runs
                _ = models['p'](x, y, z, t)
                _ = models['u'](x, y, z, t)
                _ = models['v'](x, y, z, t)
                _ = models['w'](x, y, z, t)
                _ = models['tau_x'](x, y, z, t)
                _ = models['tau_y'](x, y, z, t)
                _ = models['tau_z'](x, y, z, t)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        times.append(avg_time)
        points_per_second = size / avg_time
        
        print(f"\nSample size: {size:,} points")
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Points processed per second: {points_per_second:,.2f}")
    
    print("\nComparison with CFD:")
    print("Traditional CFD Simulation:")
    print("- Setup time: Hours to days")
    print("- Mesh generation: Hours")
    print("- Simulation runtime: Hours to days")
    print("- Total time: Days to weeks")
    
    print("\nPINN Approach:")
    print("- Training time: Hours")
    print("- Inference time: Milliseconds to seconds")
    print("- No mesh generation required")
    print("- Real-time capable for new queries")

def analyze_generalization():
    """Analyze PINN's generalization capabilities"""
    # Load data from different geometries/conditions
    data_files = [f for f in os.listdir('data/WSS_data') if f.endswith('.csv')]
    
    print("\nGeneralization Analysis:")
    print("\nPINN Advantages in Generalization:")
    print("1. Continuous Solution Space:")
    print("   - PINN provides solutions at any point in space and time")
    print("   - Not limited to discrete mesh points like CFD")
    print("   - Can interpolate between training conditions")
    print("   - Smooth solutions across the domain")
    
    print("\n2. Parameter Space Exploration:")
    print("   - Can vary input parameters continuously")
    print("   - Faster than running new CFD simulations")
    print("   - Suitable for real-time applications")
    print("   - Enables rapid what-if analysis")
    
    print("\n3. Available Training Data:")
    print("Dataset Coverage Analysis:")
    conditions = {}
    geometries = set()
    phases = set()
    
    for file in data_files:
        parts = file.split()
        patient_id = parts[0]
        geometries.add(patient_id)
        
        if len(parts) > 1:
            condition = parts[1].lower()
            phases.add(condition)
            if patient_id not in conditions:
                conditions[patient_id] = set()
            conditions[patient_id].add(condition)
    
    print(f"\nTotal Unique Geometries: {len(geometries)}")
    print(f"Total Phases: {len(phases)}")
    print(f"Phases: {', '.join(phases)}")
    print("\nPer-Patient Conditions:")
    for patient_id, patient_conditions in conditions.items():
        print(f"Patient {patient_id}: {', '.join(sorted(patient_conditions))}")

if __name__ == "__main__":
    print("Analyzing PINN vs CFD Characteristics...")
    
    # Analyze solution density
    analyze_solution_density()
    
    # Analyze computational cost
    analyze_computational_cost()
    
    # Analyze generalization capabilities
    analyze_generalization()
