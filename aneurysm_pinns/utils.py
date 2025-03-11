# aneurysm_pinns/utils.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(directory: str):
    """
    Ensures that a directory exists; if not, creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_csv(df: pd.DataFrame, path: str):
    """
    Saves a DataFrame to a CSV file.
    """
    df.to_csv(path, index=False)
    print(f"Saved DataFrame to '{path}'.")

def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")
    df = pd.read_csv(path)
    print(f"Loaded DataFrame from '{path}'.")
    return df

def plot_scatter(ax, x, y, c, title, xlabel, ylabel, cmap, label, vmin=None, vmax=None, num_ticks=5):
    """
    Helper function to create scatter plots with color mapping and enhanced colorbars.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        c (array-like): Color-mapped data.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        cmap (str): Colormap name.
        label (str): Colorbar label.
        vmin (float, optional): Minimum color value.
        vmax (float, optional): Maximum color value.
        num_ticks (int, optional): Number of ticks on the colorbar. Defaults to 5.
    """
    sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, rasterized=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    cbar = plt.colorbar(sc, ax=ax, label=label)
    if vmin is not None and vmax is not None:
        ticks = np.linspace(vmin, vmax, num_ticks)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
