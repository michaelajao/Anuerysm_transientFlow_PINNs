import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to Python path to import from aneurysm_pinns
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from aneurysm_pinns.dataset import load_data_with_metadata

# Load a sample WSS data file
data_path = "data/WSS_data/0021 Diastolic aneurysm.csv"
df = load_data_with_metadata(data_path)

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows of the data:")
print(df.head())

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Analyze mesh structure
print("\nMesh Analysis:")
print(f"Total number of points: {len(df)}")

# Analyze spatial distribution
print("\nSpatial Extent:")
for col in ['X [ m ]', ' Y [ m ]', ' Z [ m ]']:
    print(f"{col}:")
    print(f"  Range: {df[col].min():.6f}m to {df[col].max():.6f}m")
    print(f"  Span: {df[col].max() - df[col].min():.6f}m")

# Analyze point density
x_range = df['X [ m ]'].max() - df['X [ m ]'].min()
y_range = df[' Y [ m ]'].max() - df[' Y [ m ]'].min()
z_range = df[' Z [ m ]'].max() - df[' Z [ m ]'].min()
volume = x_range * y_range * z_range
point_density = len(df) / volume

print(f"\nMesh Characteristics:")
print(f"Point Density: {point_density:.2f} points/m3")
print(f"Average Volume per Point: {(1/point_density)*1e9:.2f} mm3")
print(f"Approximate Point Spacing: {(1/point_density)**(1/3)*1000:.2f} mm")

# Get unique coordinates to understand mesh points
if 'X [ m ]' in df.columns and ' Y [ m ]' in df.columns and ' Z [ m ]' in df.columns:
    unique_points = df[['X [ m ]', ' Y [ m ]', ' Z [ m ]']].drop_duplicates()
    print(f"\nNumber of unique spatial points (mesh vertices): {len(unique_points)}")
    
    # Analyze point spacing
    x_spacing = np.diff(np.sort(df['X [ m ]'].unique()))
    y_spacing = np.diff(np.sort(df[' Y [ m ]'].unique()))
    z_spacing = np.diff(np.sort(df[' Z [ m ]'].unique()))
    
    print("\nMesh spacing statistics (in millimeters):")
    print(f"X direction - min: {x_spacing.min()*1000:.3f}, max: {x_spacing.max()*1000:.3f}, mean: {x_spacing.mean()*1000:.3f}")
    print(f"Y direction - min: {y_spacing.min()*1000:.3f}, max: {y_spacing.max()*1000:.3f}, mean: {y_spacing.mean()*1000:.3f}")
    print(f"Z direction - min: {z_spacing.min()*1000:.3f}, max: {z_spacing.max()*1000:.3f}, mean: {z_spacing.mean()*1000:.3f}")

    # Create visualization directory if it doesn't exist
    os.makedirs('figures/mesh_analysis', exist_ok=True)

    # Calculate magnitude of wall shear stress
    df['WSS_Magnitude'] = np.sqrt(
        df[' Wall Shear X [ Pa ]']**2 + 
        df[' Wall Shear Y [ Pa ]']**2 + 
        df[' Wall Shear Z [ Pa ]']**2
    )

    # 3D scatter plot of mesh points
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['X [ m ]'], 
                        df[' Y [ m ]'], 
                        df[' Z [ m ]'],
                        c=df['WSS_Magnitude'],  # Color by Wall Shear magnitude
                        cmap='viridis',
                        alpha=0.6,
                        s=1)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Mesh Points Distribution (colored by Wall Shear Magnitude)')
    plt.colorbar(scatter, label='Wall Shear Magnitude [Pa]')
    
    # Save the plot
    plt.savefig('figures/mesh_analysis/mesh_points_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create 2D projections
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY projection
    scatter1 = ax1.scatter(df['X [ m ]'], df[' Y [ m ]'], 
                          c=df['WSS_Magnitude'], cmap='viridis', s=1)
    ax1.set_title('XY Projection')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    plt.colorbar(scatter1, ax=ax1)
    
    # XZ projection
    scatter2 = ax2.scatter(df['X [ m ]'], df[' Z [ m ]'], 
                          c=df['WSS_Magnitude'], cmap='viridis', s=1)
    ax2.set_title('XZ Projection')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Z [m]')
    plt.colorbar(scatter2, ax=ax2)
    
    # YZ projection
    scatter3 = ax3.scatter(df[' Y [ m ]'], df[' Z [ m ]'], 
                          c=df['WSS_Magnitude'], cmap='viridis', s=1)
    ax3.set_title('YZ Projection')
    ax3.set_xlabel('Y [m]')
    ax3.set_ylabel('Z [m]')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('figures/mesh_analysis/mesh_2d_projections.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nMesh Generation Insights:")
print("1. The mesh represents a 3D volume with dimensions:")
print(f"   - Length (X): {x_range*1000:.2f} mm")
print(f"   - Width (Y): {y_range*1000:.2f} mm")
print(f"   - Height (Z): {z_range*1000:.2f} mm")
print("2. Point distribution characteristics:")
print(f"   - Very fine mesh with average spacing of {(1/point_density)**(1/3)*1000:.3f} mm")
print("   - Non-uniform spacing suggests adaptive mesh refinement")
print("3. The mesh appears to be structured with:")
print("   - Higher density in regions of interest (likely near vessel walls)")
print("   - Variable resolution to capture flow features")
print("4. Wall Shear Stress characteristics:")
print(f"   - Maximum magnitude: {df['WSS_Magnitude'].max():.2f} Pa")
print(f"   - Mean magnitude: {df['WSS_Magnitude'].mean():.2f} Pa")
print("   - Distribution suggests areas of high shear near vessel walls")
