import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from aneurysm_pinns.dataset import load_data_with_metadata

def analyze_pinn_architecture():
    """Analyze the PINN architecture and training approach"""
    print("\nPINN Architecture Analysis:")
    print("1. Network Structure:")
    print("   - 7 separate networks (p, u, v, w, tau_x, tau_y, tau_z)")
    print("   - Each network: 4D input (x, y, z, t) → 1D output")
    print("   - Hidden layers use Swish activation (learnable beta)")
    print("   - Kaiming normal initialization")
    
    print("\n2. Physics-Informed Components:")
    print("   - Navier-Stokes equations enforced via physics loss")
    print("   - Continuity equation for incompressible flow")
    print("   - No-slip boundary conditions")
    print("   - Self-adaptive loss weighting")

def analyze_training_data():
    """Analyze the CFD training data characteristics"""
    # Load a sample dataset
    data_path = "data/WSS_data/0021 Diastolic aneurysm.csv"
    df = load_data_with_metadata(data_path)
    
    print("\nCFD Dataset Analysis:")
    print(f"1. Data Points: {len(df):,}")
    
    # Analyze spatial distribution
    x_range = df['X [ m ]'].max() - df['X [ m ]'].min()
    y_range = df[' Y [ m ]'].max() - df[' Y [ m ]'].min()
    z_range = df[' Z [ m ]'].max() - df[' Z [ m ]'].min()
    volume = x_range * y_range * z_range
    
    print("\n2. Spatial Coverage:")
    print(f"   X range: {x_range*1000:.2f} mm")
    print(f"   Y range: {y_range*1000:.2f} mm")
    print(f"   Z range: {z_range*1000:.2f} mm")
    print(f"   Volume: {volume*1e9:.2f} mm3")
    
    # Analyze flow variables
    print("\n3. Flow Variable Ranges:")
    print(f"   Pressure: {df[' Pressure [ Pa ]'].min():.2f} to {df[' Pressure [ Pa ]'].max():.2f} Pa")
    print(f"   Velocity: {df[' Velocity [ m s^-1 ]'].min():.2f} to {df[' Velocity [ m s^-1 ]'].max():.2f} m/s")
    print(f"   Wall Shear: {df[' Wall Shear [ Pa ]'].min():.2f} to {df[' Wall Shear [ Pa ]'].max():.2f} Pa")

def analyze_loss_components():
    """Analyze the different loss components in PINN training"""
    print("\nPINN Loss Components:")
    print("1. Physics Loss:")
    print("   - Navier-Stokes residuals")
    print("   - Continuity equation residual")
    print("   - Adaptive weighting via log_lambda_physics")
    
    print("\n2. Boundary Loss:")
    print("   - No-slip condition enforcement")
    print("   - Adaptive weighting via log_lambda_boundary")
    
    print("\n3. Data Loss:")
    print("   - MSE between predictions and CFD data")
    print("   - Covers all flow variables (p, u, v, w, τx, τy, τz)")
    print("   - Adaptive weighting via log_lambda_data")
    
    print("\n4. Inlet Loss:")
    print("   - Enforces inlet velocity profile")
    print("   - Sinusoidal variation for systolic/diastolic phases")
    print("   - Adaptive weighting via log_lambda_inlet")

def create_architecture_diagram():
    """Create a visual representation of the PINN architecture"""
    plt.figure(figsize=(12, 8))
    
    # Network structure
    plt.subplot(121)
    plt.title('PINN Architecture', pad=20)
    
    # Input layer
    plt.plot([-0.2, 0.2], [0.9, 0.9], 'k-', linewidth=2)
    plt.text(0, 0.95, 'Input Layer\n(x, y, z, t)', ha='center')
    
    # Hidden layers
    for i in range(3):
        y = 0.7 - i*0.2
        plt.plot([-0.2, 0.2], [y, y], 'b-', linewidth=2)
        plt.text(0, y+0.05, f'Hidden Layer {i+1}\nSwish Activation', ha='center')
    
    # Output layer
    plt.plot([-0.2, 0.2], [0.1, 0.1], 'g-', linewidth=2)
    plt.text(0, 0.15, 'Output Layer\n(p, u, v, w, τx, τy, τz)', ha='center')
    
    # Connections
    for y1 in [0.9, 0.7, 0.5, 0.3]:
        for y2 in [y1-0.2]:
            plt.plot([-0.1, 0.1], [y1, y2], 'k-', alpha=0.3)
    
    plt.axis('off')
    
    # Loss components
    plt.subplot(122)
    plt.title('Loss Components', pad=20)
    
    components = ['Physics Loss', 'Boundary Loss', 'Data Loss', 'Inlet Loss']
    weights = [0.3, 0.2, 0.3, 0.2]  # Example weights
    colors = ['#AED6F1', '#A2D9A2', '#F5B7B1', '#D7BDE2']
    
    plt.pie(weights, labels=components, colors=colors, autopct='%1.1f%%',
            startangle=90)
    
    plt.axis('equal')
    
    # Save the visualization
    os.makedirs('figures/pinn_analysis', exist_ok=True)
    plt.savefig('figures/pinn_analysis/pinn_architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Analyzing PINN Implementation and CFD Data...")
    
    # Analyze PINN architecture
    analyze_pinn_architecture()
    
    # Analyze training data
    analyze_training_data()
    
    # Analyze loss components
    analyze_loss_components()
    
    # Create architecture visualization
    create_architecture_diagram()
    print("\nArchitecture diagram saved to figures/pinn_analysis/pinn_architecture.png")
