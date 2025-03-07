# aneurysm_pinns/visualization.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import griddata
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple

# Import project-specific modules
from aneurysm_pinns.config import Config
from aneurysm_pinns.utils import ensure_dir

# For better visualization aesthetics
plt.style.use("seaborn-v0_8-paper")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = False
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.titlesize': 16})

def load_processed_datasets(config: Config) -> Dict[str, pd.DataFrame]:
    """
    Loads all processed datasets from the processed folder in the data directory.
    
    Args:
        config: Configuration object with paths and settings
        
    Returns:
        Dictionary of datasets with their names as keys
    """
    processed_files = []
    for category in config.categories:
        for phase in config.phases:
            phase_dir = os.path.join(config.processed_data_dir, category, phase)
            if os.path.isdir(phase_dir):
                for file in os.listdir(phase_dir):
                    if file.endswith('.csv'):
                        file_path = os.path.join(phase_dir, file)
                        processed_files.append(file_path)
            else:
                print(f"Directory does not exist: {phase_dir}")

    # Load all datasets into a dictionary for easy access
    datasets = {}
    for file in processed_files:
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        datasets[dataset_name] = df

    return clean_datasets(datasets)

def clean_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Cleans the datasets by handling missing columns and values.
    
    Args:
        datasets: Dictionary of loaded datasets
        
    Returns:
        Dictionary of cleaned datasets
    """
    required_columns = [
        "X [ m ]", "Y [ m ]", "Z [ m ]",
        "Pressure [ Pa ]", 
        "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]",
        "Wall Shear X [ Pa ]", "Wall Shear Y [ Pa ]", "Wall Shear Z [ Pa ]",
    ]
    
    # Check for "Velocity [ m s^-1 ]" and "Wall Shear [ Pa ]", these might not be in all datasets
    velocity_column = "Velocity [ m s^-1 ]"
    wall_shear_column = "Wall Shear [ Pa ]"
    
    for name, df in datasets.items():
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Dataset '{name}' is missing columns: {missing_cols}")
        
        # Handle missing values by dropping rows with NaN in required columns
        available_cols = [col for col in required_columns if col in df.columns]
        if available_cols:
            df.dropna(subset=available_cols, inplace=True)
        
        # Calculate velocity magnitude if it doesn't exist
        if velocity_column not in df.columns:
            u = df["Velocity u [ m s^-1 ]"]
            v = df["Velocity v [ m s^-1 ]"]
            w = df["Velocity w [ m s^-1 ]"]
            df[velocity_column] = np.sqrt(u**2 + v**2 + w**2)
            print(f"Created '{velocity_column}' for dataset '{name}'")
            
        # Calculate wall shear stress magnitude if it doesn't exist
        if wall_shear_column not in df.columns:
            tx = df["Wall Shear X [ Pa ]"]
            ty = df["Wall Shear Y [ Pa ]"]
            tz = df["Wall Shear Z [ Pa ]"]
            df[wall_shear_column] = np.sqrt(tx**2 + ty**2 + tz**2)
            print(f"Created '{wall_shear_column}' for dataset '{name}'")
        
        # Update the dataset in the dictionary
        datasets[name] = df
        
    return datasets

def save_plot(filename: str, config: Config):
    """
    Save the current plot to the specified directory.
    
    Args:
        filename: Name of the file to save
        config: Configuration object with plot directory
    """
    plot_dir = os.path.join(config.plot_dir, "data_plots")
    ensure_dir(plot_dir)
    plt.savefig(os.path.join(plot_dir, filename), dpi=config.plot_resolution, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to '{os.path.join(plot_dir, filename)}'")


def visualize_2d_velocity_field(df: pd.DataFrame, downsample: int = 200, scale: int = 50, color: str = 'blue', config: Config = None):
    """
    Visualizes the velocity field as a 2D quiver plot.
    
    Args:
        df: DataFrame containing velocity data
        downsample: Factor to downsample the data by
        scale: Scale factor for the quiver plot
        color: Color for the quiver arrows
        config: Configuration object with plot settings
    """
    x = df["X [ m ]"]
    y = df["Y [ m ]"]
    velocity_u = df["Velocity u [ m s^-1 ]"]
    velocity_v = df["Velocity v [ m s^-1 ]"]

    # Downsample to avoid overcrowding the plot
    quiver_x = x[::downsample]
    quiver_y = y[::downsample]
    quiver_u = velocity_u[::downsample]
    quiver_v = velocity_v[::downsample]

    plt.figure(figsize=(12, 8))
    plt.quiver(quiver_x, quiver_y, quiver_u, quiver_v, angles='xy', scale_units='xy', scale=scale, color=color, alpha=0.8)
    plt.title("2D Velocity Vector Field", fontsize=14, fontweight="bold")
    plt.xlabel("X [m]", fontsize=12)
    plt.ylabel("Y [m]", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if config:
        save_plot("2D_Velocity_Vector_Field.png", config)
    plt.show()

def create_grouped_visualizations(data: List[pd.DataFrame], param: str, planes: List[Tuple[str, str]], 
                                 titles: List[str], config: Config = None):
    """
    Create grouped subplots for a specific parameter across different datasets and planes.
    
    Args:
        data: List of DataFrames to visualize
        param: Parameter to visualize (column name)
        planes: List of tuples (x_col, y_col) defining the planes to visualize
        titles: List of titles for each dataset
        config: Configuration object with plot settings
    """
    fig, axes = plt.subplots(nrows=len(data), ncols=len(planes), figsize=(15, len(data) * 5))
    axes = axes.flatten() if len(data) > 1 else [axes]

    for i, df in enumerate(data):
        for j, (x_col, y_col) in enumerate(planes):
            idx = i * len(planes) + j
            ax = axes[idx]
            scatter = ax.scatter(df[x_col], df[y_col], c=df[param], cmap="viridis", alpha=0.7)
            ax.set_title(f"{titles[i]} ({x_col}-{y_col} Plane)")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            fig.colorbar(scatter, ax=ax, label=param)
            ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    if config:
        save_plot(f"Grouped_Visualizations_{param.replace(' ', '_')}.png", config)
    plt.show()

def visualize_wss(df: pd.DataFrame, title: str = "Wall Shear Stress Distribution", config: Config = None):
    """
    Visualizes Wall Shear Stress (WSS) distribution using a scatter plot.
    
    Args:
        df: DataFrame containing WSS data
        title: Title for the plot
        config: Configuration object with plot settings
    """
    required_columns = ['Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear [ Pa ]']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(df['X [ m ]'], df['Y [ m ]'], c=df['Wall Shear [ Pa ]'], 
                          cmap="viridis", rasterized=True, s=5, alpha=0.7)
    plt.colorbar(scatter, label="Wall Shear Stress (Pa)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(title)

    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_wss_contours(df: pd.DataFrame, title_prefix: str = "Wall Shear Stress Distribution", config: Config = None):
    """
    Visualizes Wall Shear Stress (WSS) distribution using contour plots from multiple views.
    
    Args:
        df: DataFrame containing WSS data
        title_prefix: Prefix for the plot title
        config: Configuration object with plot settings
    """
    required_columns = ['X [ m ]', 'Y [ m ]', 'Z [ m ]', 
                        'Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear Z [ Pa ]']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

    fig = plt.figure(figsize=(25, 20))
    norm = plt.Normalize(df['Wall Shear [ Pa ]'].min(), df['Wall Shear [ Pa ]'].max())
    cmap = plt.cm.viridis

    views = {
        'Anterior View': ('X [ m ]', 'Z [ m ]'),
        'Posterior View': ('X [ m ]', 'Z [ m ]'),
        'Lateral View': ('Y [ m ]', 'Z [ m ]'),
        'Superior View': ('X [ m ]', 'Y [ m ]'),
        'Overall Vessel': ('X [ m ]', 'Y [ m ]')
    }

    for i, (title, (x_col, y_col)) in enumerate(views.items()):
        ax = fig.add_subplot(2, 3, i + 1)
        scatter = ax.scatter(df[x_col], df[y_col], c=df['Wall Shear [ Pa ]'], 
                            cmap=cmap, norm=norm, s=5, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{title_prefix} - {title}")
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label="WSS Magnitude (Pa)")

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter = ax5.scatter(df['X [ m ]'], df['Y [ m ]'], df['Z [ m ]'], 
                         c=df['Wall Shear [ Pa ]'], cmap=cmap, norm=norm, s=2, alpha=0.7)
    ax5.set_xlabel("X [m]")
    ax5.set_ylabel("Y [m]")
    ax5.set_zlabel("Z [m ]")
    ax5.set_title(f"{title_prefix} - Overall Vessel View")
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(df['Wall Shear [ Pa ]'])
    plt.colorbar(m, ax=ax5, label="WSS Magnitude (Pa)")

    plt.tight_layout()
    if config:
        save_plot(f"{title_prefix.replace(' ', '_')}_Contours.png", config)
    plt.show()


def plot_wss_contour(ax, df: pd.DataFrame, projection: Tuple[str, str], title: str):
    """
    Plots the WSS contour on the given Axes object based on the projection.
    
    Args:
        ax: Matplotlib Axes object to plot on
        df: DataFrame containing WSS data
        projection: Tuple (x_col, y_col) defining the projection plane
        title: Title for the plot
    """
    x_col, y_col = projection
    x = df[x_col].values
    y = df[y_col].values
    wss = df['Wall Shear [ Pa ]'].values

    xi = np.linspace(x.min(), x.max(), 300)
    yi = np.linspace(y.min(), y.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), wss, (xi, yi), method='linear')

    contour = ax.contourf(xi, yi, zi, levels=50, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(projection[0])
    ax.set_ylabel(projection[1])
    return contour

def visualize_distribution(df: pd.DataFrame, column: str, title: str, bins: int = 50, kde: bool = True, 
                          config: Config = None):
    """
    Visualizes the distribution of a specific column.
    
    Args:
        df: DataFrame containing the data
        column: Column to visualize
        title: Title for the plot
        bins: Number of bins for the histogram
        kde: Whether to include a KDE plot
        config: Configuration object with plot settings
    """
    sns.histplot(df[column], kde=kde, bins=bins)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()


def visualize_time_series(df: pd.DataFrame, time_col: str, value_col: str, title: str, 
                         rolling_window: int = 100, config: Config = None):
    """
    Visualizes a time series plot with an optional rolling average.
    
    Args:
        df: DataFrame containing the time series data
        time_col: Column name for the time values
        value_col: Column name for the values to plot
        title: Title for the plot
        rolling_window: Window size for the rolling average
        config: Configuration object with plot settings
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[time_col], df[value_col], label=value_col)
    df_copy = df.copy()
    df_copy[f'{value_col}_Rolling'] = df_copy[value_col].rolling(window=rolling_window).mean()
    plt.plot(df_copy[time_col], df_copy[f'{value_col}_Rolling'], 
            label=f'Rolling Average ({rolling_window} points)', linestyle='--')
    plt.title(title)
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.legend()
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_spatial_distribution(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, 
                                  cmap: str = 'viridis', s: int = 1, alpha: float = 0.7, config: Config = None):
    """
    Visualizes the spatial distribution of a specific column.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for the x-coordinate
        y_col: Column name for the y-coordinate
        value_col: Column name for the values to visualize
        title: Title for the plot
        cmap: Colormap to use
        s: Size of the points
        alpha: Transparency of the points
        config: Configuration object with plot settings
    """
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df[x_col], df[y_col], c=df[value_col], cmap=cmap, s=s, alpha=alpha)
    plt.colorbar(scatter, label=value_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()


def visualize_velocity_vector_field(df: pd.DataFrame, x_col: str, y_col: str, u_col: str, v_col: str, title: str, 
                                   scale: int = 50, config: Config = None):
    """
    Visualizes the velocity vector field.
    
    Args:
        df: DataFrame containing the velocity data
        x_col: Column name for the x-coordinate
        y_col: Column name for the y-coordinate
        u_col: Column name for the x-component of velocity
        v_col: Column name for the y-component of velocity
        title: Title for the plot
        scale: Scale factor for the quiver plot
        config: Configuration object with plot settings
    """
    plt.figure(figsize=(10, 7))
    plt.quiver(df[x_col], df[y_col], df[u_col], df[v_col], scale=scale)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_interactive_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, value_col: str, title: str, 
                                    config: Config = None):
    """
    Visualizes an interactive 3D scatter plot using Plotly and saves it.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for the x-coordinate
        y_col: Column name for the y-coordinate
        z_col: Column name for the z-coordinate
        value_col: Column name for the values to visualize
        title: Title for the plot
        config: Configuration object with plot settings
    """
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col,
        color=value_col, size_max=4,
        title=title,
        labels={x_col: x_col, y_col: y_col, z_col: z_col, value_col: value_col},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    
    if config:
        plot_dir = os.path.join(config.plot_dir, "data_plots")
        ensure_dir(plot_dir)
        html_path = os.path.join(plot_dir, f"{title.replace(' ', '_')}.html")
        fig.write_html(html_path)
        print(f"Saved interactive 3D plot to '{html_path}'")
    
    fig.show()

def visualize_jointplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, config: Config = None):
    """
    Visualizes a joint plot.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for the x-coordinate
        y_col: Column name for the y-coordinate
        title: Title for the plot
        config: Configuration object with plot settings
    """
    sns.jointplot(x=x_col, y=y_col, data=df, kind='scatter', alpha=0.3)
    plt.suptitle(title, y=1.02)
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_pca(df: pd.DataFrame, features: List[str], hue_col: str, title: str, config: Config = None):
    """
    Visualizes PCA of selected features.
    
    Args:
        df: DataFrame containing the data
        features: List of column names to include in PCA
        hue_col: Column name to use for coloring points
        title: Title for the plot
        config: Configuration object with plot settings
    """
    x = df[features].dropna()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', data=principalDf, hue=df[hue_col], palette='viridis', s=10)
    plt.title(title)
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_tsne(df: pd.DataFrame, features: List[str], hue_col: str, title: str, config: Config = None):
    """
    Visualizes t-SNE of selected features.
    
    Args:
        df: DataFrame containing the data
        features: List of column names to include in t-SNE
        hue_col: Column name to use for coloring points
        title: Title for the plot
        config: Configuration object with plot settings
    """
    x = df[features].dropna()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data=tsne_results, columns=['Dim1', 'Dim2'])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Dim1', y='Dim2', data=tsne_df, hue=df[hue_col], palette='coolwarm', s=10)
    plt.title(title)
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_pairplot(df: pd.DataFrame, features: List[str], title: str, config: Config = None):
    """
    Visualizes a pair plot.
    
    Args:
        df: DataFrame containing the data
        features: List of column names to include in the pair plot
        title: Title for the plot
        config: Configuration object with plot settings
    """
    sns.pairplot(df[features], diag_kind='kde')
    plt.suptitle(title, y=1.02)
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def visualize_spatial_distribution_with_pressure(df: pd.DataFrame, x_col: str, y_col: str, 
                                               wall_shear_col: str, pressure_col: str, title: str, 
                                               config: Config = None):
    """
    Visualizes spatial distribution with pressure represented by marker size.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for the x-coordinate
        y_col: Column name for the y-coordinate
        wall_shear_col: Column name for wall shear stress values
        pressure_col: Column name for pressure values
        title: Title for the plot
        config: Configuration object with plot settings
    """
    x = df[x_col]
    y = df[y_col]
    wall_shear = df[wall_shear_col]
    pressure = df[pressure_col]

    size_min = 20
    size_max = 200
    pressure_normalized = (pressure - pressure.min()) / (pressure.max() - pressure.min())
    sizes = size_min + pressure_normalized * (size_max - size_min)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=df[wall_shear_col], cmap='viridis', s=sizes, alpha=0.6, edgecolors='w', linewidth=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label(wall_shear_col)

    pressure_levels = np.linspace(pressure.min(), pressure.max(), num=5)
    size_levels = size_min + (pressure_levels - pressure.min()) / (pressure.max() - pressure.min()) * (size_max - size_min)

    handles = [plt.scatter([], [], s=size, color='gray', alpha=0.6, edgecolors='w', linewidth=0.5) for size in size_levels]
    labels = [f"{int(p)} Pa" for p in pressure_levels]

    plt.legend(handles, labels, title='Pressure [Pa]', scatterpoints=1, loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    if config:
        save_plot(f"{title.replace(' ', '_')}.png", config)
    plt.show()

def main():
    """
    Main function to run the visualization pipeline.
    """
    # Initialize configuration
    config = Config()
    
    # Make sure directories exist
    ensure_dir(os.path.join(config.plot_dir, "data_plots"))
    
    # Load and clean datasets
    datasets = load_processed_datasets(config)
    
    if not datasets:
        print("No datasets were loaded. Please check the paths and data files.")
        return
    
    print(f"Loaded {len(datasets)} datasets")
    
    # Select specific datasets for visualization
    datasets_to_visualize = []
    titles = []
    
    if "0024_diastolic" in datasets:
        datasets_to_visualize.append(datasets["0024_diastolic"])
        titles.append("0024 Diastolic (Healthy)")
    
    if "0024_systolic" in datasets:
        datasets_to_visualize.append(datasets["0024_systolic"])
        titles.append("0024 Systolic (Healthy)")
        
    if "0021_diastolic_aneurysm" in datasets:
        datasets_to_visualize.append(datasets["0021_diastolic_aneurysm"])
        titles.append("0021 Diastolic (Aneurysmal)")
        
    if "0021_systolic_aneurysm" in datasets:
        datasets_to_visualize.append(datasets["0021_systolic_aneurysm"])
        titles.append("0021 Systolic (Aneurysmal)")
    
    # Define planes for visualization
    planes = [("X [ m ]", "Y [ m ]"), ("X [ m ]", "Z [ m ]")]
    
    # Generate visualizations if datasets are available
    if datasets_to_visualize:
        # Generate visualizations for Wall Shear Stress
        create_grouped_visualizations(
            data=datasets_to_visualize,
            param="Wall Shear [ Pa ]",
            planes=planes,
            titles=titles,
            config=config
        )
        
        # Generate visualizations for Velocity Magnitude
        create_grouped_visualizations(
            data=datasets_to_visualize,
            param="Velocity [ m s^-1 ]",
            planes=planes,
            titles=titles,
            config=config
        )
        
        # Generate visualizations for Pressure Distribution
        create_grouped_visualizations(
            data=datasets_to_visualize,
            param="Pressure [ Pa ]",
            planes=planes,
            titles=titles,
            config=config
        )
        
        # Example of using other visualization functions with the first dataset
        if datasets_to_visualize:
            first_dataset = datasets_to_visualize[0]
            first_title = titles[0]
            
            # Visualize Wall Shear Stress distribution
            visualize_wss(first_dataset, title=f"Wall Shear Stress Distribution ({first_title})", config=config)
            
            # Visualize distribution of Pressure
            visualize_distribution(first_dataset, "Pressure [ Pa ]", f"Distribution of Pressure ({first_title})", config=config)
            
            # Visualize interactive 3D plot
            visualize_interactive_3d_scatter(
                first_dataset, 
                x_col='X [ m ]', 
                y_col='Y [ m ]', 
                z_col='Z [ m ]', 
                value_col='Wall Shear [ Pa ]', 
                title=f'3D Scatter Plot of Wall Shear Stress ({first_title})',
                config=config
            )
    else:
        print("Warning: No datasets were selected for visualization.")

if __name__ == "__main__":
    main()