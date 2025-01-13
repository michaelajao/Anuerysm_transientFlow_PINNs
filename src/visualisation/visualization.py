# Importing essential libraries
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

# For better visualization aesthetics
plt.style.use("seaborn-v0_8-paper")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = False
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.titlesize': 16})

# Define the directory containing processed datasets and the plot directory
processed_dir = "../../data/processed/"
plot_dir = "../../figures/data_plots/"
os.makedirs(plot_dir, exist_ok=True)

# List all CSV files in the processed directory
processed_files = []
categories = ['aneurysm', 'global', 'healthy']
phases = ['systolic', 'diastolic']

for category in categories:
    for phase in phases:
        phase_dir = os.path.join(processed_dir, category, phase)
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

# Ensure that all required columns are present and clean the data
required_columns = [
    "X [ m ]", "Y [ m ]", "Z [ m ]",
    "Pressure [ Pa ]", "Velocity [ m s^-1 ]",
    "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "Velocity w [ m s^-1 ]",
    "Wall Shear [ Pa ]", "Wall Shear X [ Pa ]", "Wall Shear Y [ Pa ]", "Wall Shear Z [ Pa ]",
]

for name, df in datasets.items():
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Dataset '{name}' is missing columns: {missing_cols}")
    # Handle missing values by dropping rows with NaN in required columns
    df.dropna(subset=required_columns, inplace=True)
    # Update the dataset in the dictionary
    datasets[name] = df

# Define visualization functions
def save_plot(filename, save_path=plot_dir):
    """Save the current plot to the specified directory."""
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')


def visualize_2d_velocity_field(df, downsample=200, scale=50, color='blue'):
    """Visualizes the velocity field as a 2D quiver plot."""
    x = df["X [ m ]"]
    y = df["Y [ m ]"]
    velocity_u = df["Velocity u [ m s^-1 ]"]
    velocity_v = df["Velocity v [ m s^-1 ]"]

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

    save_plot("2D_Velocity_Vector_Field.png")
    plt.show()

def create_grouped_visualizations(data, param, planes, titles):
    """Create grouped subplots for a specific parameter across different datasets and planes."""
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
    save_plot(f"Grouped_Visualizations_{param.replace(' ', '_')}.png")
    plt.show()

def visualize_wss(df, title="Wall Shear Stress Distribution", save_path=None):
    """Visualizes Wall Shear Stress (WSS) distribution using a scatter plot."""
    required_columns = ['Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear [ Pa ]']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(df['X [ m ]'], df['Y [ m ]'], c=df['Wall Shear [ Pa ]'], cmap="viridis", rasterized=True, s=5, alpha=0.7)
    plt.colorbar(scatter, label="Wall Shear Stress (Pa)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(title)

    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_wss_contours(df, title_prefix="Wall Shear Stress Distribution", save_path=None):
    """Visualizes Wall Shear Stress (WSS) distribution using contour plots from multiple views."""
    required_columns = ['X [ m ]', 'Y [ m ]', 'Z [ m ]', 'Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear Z [ Pa ]']
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
        scatter = ax.scatter(df[x_col], df[y_col], c=df['Wall Shear [ Pa ]'], cmap=cmap, norm=norm, s=5, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{title_prefix} - {title}")
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label="WSS Magnitude (Pa)")

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter = ax5.scatter(df['X [ m ]'], df['Y [ m ]'], df['Z [ m ]'], c=df['Wall Shear [ Pa ]'], cmap=cmap, norm=norm, s=2, alpha=0.7)
    ax5.set_xlabel("X [m]")
    ax5.set_ylabel("Y [m]")
    ax5.set_zlabel("Z [m ]")
    ax5.set_title(f"{title_prefix} - Overall Vessel View")
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(df['Wall Shear [ Pa ]'])
    plt.colorbar(m, ax=ax5, label="WSS Magnitude (Pa)")

    plt.tight_layout()
    save_plot(f"{title_prefix.replace(' ', '_')}_Contours.png")
    plt.show()


def plot_wss_contour(ax, df, projection, title):
    """Plots the WSS contour on the given Axes object based on the projection."""
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

def visualize_distribution(df, column, title, bins=50, kde=True):
    """Visualizes the distribution of a specific column."""
    sns.histplot(df[column], kde=kde, bins=bins)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()


def visualize_time_series(df, time_col, value_col, title, rolling_window=100):
    """Visualizes a time series plot with an optional rolling average."""
    plt.figure(figsize=(10, 6))
    plt.plot(df[time_col], df[value_col], label=value_col)
    df[f'{value_col}_Rolling'] = df[value_col].rolling(window=rolling_window).mean()
    plt.plot(df[time_col], df[f'{value_col}_Rolling'], label=f'Rolling Average ({rolling_window} points)', linestyle='--')
    plt.title(title)
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.legend()
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_spatial_distribution(df, x_col, y_col, value_col, title, cmap='viridis', s=1, alpha=0.7):
    """Visualizes the spatial distribution of a specific column."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df[x_col], df[y_col], c=df[value_col], cmap=cmap, s=s, alpha=alpha)
    plt.colorbar(scatter, label=value_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()


def visualize_velocity_vector_field(df, x_col, y_col, u_col, v_col, title, scale=50):
    """Visualizes the velocity vector field."""
    plt.figure(figsize=(10, 7))
    plt.quiver(df[x_col], df[y_col], df[u_col], df[v_col], scale=scale)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_interactive_3d_scatter(df, x_col, y_col, z_col, value_col, title):
    """Visualizes an interactive 3D scatter plot using Plotly and saves it."""
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
    fig.write_html(os.path.join(plot_dir, f"{title.replace(' ', '_')}.html"))
    fig.show()

def visualize_jointplot(df, x_col, y_col, title):
    """Visualizes a joint plot."""
    sns.jointplot(x=x_col, y=y_col, data=df, kind='scatter', alpha=0.3)
    plt.suptitle(title, y=1.02)
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_pca(df, features, hue_col, title):
    """Visualizes PCA of selected features."""
    x = df[features].dropna()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', data=principalDf, hue=df[hue_col], palette='viridis', s=10)
    plt.title(title)
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_tsne(df, features, hue_col, title):
    """Visualizes t-SNE of selected features."""
    x = df[features].dropna()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(x)
    tsne_df = pd.DataFrame(data=tsne_results, columns=['Dim1', 'Dim2'])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Dim1', y='Dim2', data=tsne_df, hue=df[hue_col], palette='coolwarm', s=10)
    plt.title(title)
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_pairplot(df, features, title):
    """Visualizes a pair plot."""
    sns.pairplot(df[features], diag_kind='kde')
    plt.suptitle(title, y=1.02)
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_spatial_distribution_with_pressure(df, x_col, y_col, wall_shear_col, pressure_col, title):
    """Visualizes spatial distribution with pressure represented by marker size."""
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
    save_plot(f"{title.replace(' ', '_')}.png")
    plt.show()

# Data and titles for the visualizations
datasets_to_visualize = [
    datasets["0024_diastolic"],
    datasets["0024_systolic"],
    datasets["0021_diastolic_aneurysm"],
    datasets["0021_systolic_aneurysm"]
]
titles = ["0024 Diastolic (Healthy)", "0024 Systolic (Healthy)", "0021 Diastolic (Aneurysmal)", "0021 Systolic (Aneurysmal)"]
planes = [("X [ m ]", "Y [ m ]"), ("X [ m ]", "Z [ m ]")]

# Generate visualizations for Wall Shear Stress
create_grouped_visualizations(
    data=datasets_to_visualize,
    param="Wall Shear [ Pa ]",
    planes=planes,
    titles=titles
)

# Generate visualizations for Velocity Magnitude
create_grouped_visualizations(
    data=datasets_to_visualize,
    param="Velocity [ m s^-1 ]",
    planes=planes,
    titles=titles
)

# Generate visualizations for Pressure Distribution
create_grouped_visualizations(
    data=datasets_to_visualize,
    param="Pressure [ Pa ]",
    planes=planes,
    titles=titles
)

# Visualize Wall Shear Stress distribution for a specific dataset
visualize_wss(datasets["0024_diastolic"], title="Wall Shear Stress Distribution (0024 Diastolic)")


# Visualize distribution of Pressure
visualize_distribution(datasets["0024_diastolic"], "Pressure [ Pa ]", "Distribution of Pressure")


# Visualize Pressure Over Time
visualize_time_series(datasets["0024_diastolic"], time_col='Time [ s ]', value_col='Pressure [ Pa ]', title='Pressure Over Time')

# Visualize Spatial Distribution of Pressure
visualize_spatial_distribution(datasets["0024_diastolic"], x_col='X [ m ]', y_col='Y [ m ]', value_col='Pressure [ Pa ]', title='Spatial Distribution of Pressure')

# Visualize Velocity Vector Field
visualize_velocity_vector_field(datasets["0024_diastolic"], x_col='X [ m ]', y_col='Y [ m ]', u_col='Velocity u [ m s^-1 ]', v_col='Velocity v [ m s^-1 ]', title='Velocity Vector Field')

# Visualize Distribution of WSS Magnitude
visualize_distribution(datasets["0024_diastolic"], "Wall Shear [ Pa ]", "Distribution of WSS Magnitude", bins=50, kde=True)

# Visualize Joint Plot of Pressure vs Velocity
visualize_jointplot(datasets["0024_diastolic"], x_col='Pressure [ Pa ]', y_col='Velocity [ m s^-1 ]', title='Pressure vs Velocity')

# Visualize PCA of Selected Features
visualize_pca(datasets["0024_diastolic"], features=['Pressure [ Pa ]', 'Velocity [ m s^-1 ]', 'Wall Shear [ Pa ]', 'Inlet Diameter [cm]', 'Outlet Diameter [cm]'], hue_col='Sex', title='PCA of Selected Features')

# Visualize t-SNE of Selected Features
visualize_tsne(datasets["0024_diastolic"], features=['Pressure [ Pa ]', 'Velocity [ m s^-1 ]', 'Wall Shear [ Pa ]', 'Inlet Diameter [cm]', 'Outlet Diameter [cm]'], hue_col='Sex', title='t-SNE Visualization')

# Visualize Pair Plot of Selected Features
visualize_pairplot(datasets["0024_diastolic"], features=['Pressure [ Pa ]', 'Velocity [ m s^-1 ]', 'Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear Z [ Pa ]', 'Inlet Diameter [cm]', 'Outlet Diameter [cm]'], title='Pair Plot of Selected Features')

# Visualize Spatial Distribution with Pressure represented by marker size
visualize_spatial_distribution_with_pressure(datasets["0021_systolic_aneurysm"], x_col='X [ m ]', y_col='Y [ m ]', wall_shear_col='Wall Shear X [ Pa ]', pressure_col='Pressure [ Pa ]', title='Spatial Distribution of Wall Shear Stress and Pressure')

# Visualize 3D Scatter Plot
visualize_interactive_3d_scatter(datasets["0024_diastolic"], x_col='X [ m ]', y_col='Y [ m ]', z_col='Z [ m ]', value_col='Wall Shear [ Pa ]', title='3D Scatter Plot of Wall Shear Stress')