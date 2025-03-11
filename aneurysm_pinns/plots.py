# aneurysm_pinns/plots.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List

from aneurysm_pinns.config import Config
from aneurysm_pinns.dataset import CFDDataset
from aneurysm_pinns.utils import ensure_dir, plot_scatter


# Use Seaborn's "paper" style for clean and professional aesthetics
plt.style.use("seaborn-v0_8-paper")

# =========================================
# 2. Update rcParams for Publication-Quality Plots
# =========================================

plt.rcParams.update(
    {
        # -------------------------------------
        # General Figure Settings
        # -------------------------------------
        "font.size": 12,                       # Base font size for all text elements
        "figure.figsize": [7, 4],              # Figure size suitable for double-column layouts
        "text.usetex": False,                  # Disable LaTeX rendering; set to True if needed
        "figure.facecolor": "white",           # White background for compatibility
        "figure.autolayout": True,             # Automatically adjust subplot params
        "figure.dpi": 300,                     # High resolution for print quality
        "savefig.dpi": 300,                    # High resolution for saved figures
        "savefig.format": "pdf",               # Vector format for scalability; use 'png' if raster is needed
        "savefig.bbox": "tight",               # Minimize whitespace around the figure

        # -------------------------------------
        # Axes and Titles
        # -------------------------------------
        # "axes.labelweight": "bold",            # Bold axis labels for emphasis
        # "axes.titleweight": "bold",            # Bold titles for emphasis
        "axes.labelsize": 12,                  # Font size for axis labels
        "axes.titlesize": 16,                  # Font size for plot titles
        "axes.facecolor": "white",             # White background for axes
        "axes.grid": False,                    # Disable gridlines for clarity
        "axes.spines.top": False,              # Remove top spine for a cleaner look
        "axes.spines.right": False,            # Remove right spine for a cleaner look
        "axes.formatter.use_mathtext": True,   # Use LaTeX-style formatting for tick labels
        "axes.formatter.useoffset": False,     # Disable offset in tick labels
        # "axes.xmargin": 0,                      # Remove horizontal margin
        # "axes.ymargin": 0,                      # Remove vertical margin

        # -------------------------------------
        # Legend Settings
        # -------------------------------------
        "legend.fontsize": 12,                  # Font size for legend text
        # "legend.frameon": False,                # Remove legend frame for a cleaner look
        "legend.loc": "best",                   # Automatically place legend in the best location


        # -------------------------------------
        # Image Settings
        # -------------------------------------
        "image.cmap": "viridis",                 # Default colormap for images
    }
)


def plot_loss_curves(loss_history: Dict[str, List[float]], config: Config, run_id: str, dataset_name: str):
    """
    Plots the training loss curves for different loss components.

    Args:
        loss_history (Dict[str, list]): Dictionary containing loss history.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.
        dataset_name (str): Name of the dataset.
    """
    epochs = range(1, len(loss_history["total"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history["total"], label="Total")
    plt.plot(epochs, loss_history["physics"], label="Physics")
    plt.plot(epochs, loss_history["boundary"], label="Boundary")
    plt.plot(epochs, loss_history["data"], label="Data")
    plt.plot(epochs, loss_history["inlet"], label="Inlet")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"Training Loss - {run_id}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    run_plot_dir = os.path.join(config.plot_dir, dataset_name, run_id)
    ensure_dir(run_plot_dir)
    plot_filename = f"loss_curves_{run_id}.png"
    plot_path = os.path.join(run_plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to '{plot_path}'.")


def plot_pressure_and_wss_magnitude_distribution(
    dataset: CFDDataset,
    models: Dict[str, torch.nn.Module],
    config: Config,
    run_id: str
):
    """
    Generates and saves plots comparing the distribution of Pressure and WSS Magnitude
    between CFD data and PINN predictions in both XY and XZ planes.

    Args:
        dataset (CFDDataset): Dataset instance containing CFD data.
        models (Dict[str, nn.Module]): Dictionary of trained PINN models.
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.
    """
    for m in models.values():
        m.eval()

    variables = ["pressure", "velocity_u", "velocity_v", "velocity_w", "wall_shear_x", "wall_shear_y", "wall_shear_z"]
    predictions = {var: [] for var in variables}
    truths = {var: [] for var in variables}

    x_sample = dataset.x.numpy()
    y_sample = dataset.y.numpy()
    z_sample = dataset.z.numpy()
    t_sample = dataset.t.numpy()

    batch_size = 1024
    num_samples = len(x_sample)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = i + batch_size
            bx = torch.tensor(x_sample[i:end], dtype=torch.float32, device=config.device)
            by = torch.tensor(y_sample[i:end], dtype=torch.float32, device=config.device)
            bz = torch.tensor(z_sample[i:end], dtype=torch.float32, device=config.device)
            bt = torch.tensor(t_sample[i:end], dtype=torch.float32, device=config.device)

            ppred = models["p"](bx, by, bz, bt)
            upred = models["u"](bx, by, bz, bt)
            vpred = models["v"](bx, by, bz, bt)
            wpred = models["w"](bx, by, bz, bt)
            txpred = models["tau_x"](bx, by, bz, bt)
            typred = models["tau_y"](bx, by, bz, bt)
            tzpred = models["tau_z"](bx, by, bz, bt)

            predictions["pressure"].append(ppred.cpu().numpy())
            truths["pressure"].append(dataset.p[i:end].numpy())

            predictions["velocity_u"].append(upred.cpu().numpy())
            truths["velocity_u"].append(dataset.u[i:end].numpy())

            predictions["velocity_v"].append(vpred.cpu().numpy())
            truths["velocity_v"].append(dataset.v[i:end].numpy())

            predictions["velocity_w"].append(wpred.cpu().numpy())
            truths["velocity_w"].append(dataset.w[i:end].numpy())

            predictions["wall_shear_x"].append(txpred.cpu().numpy())
            truths["wall_shear_x"].append(dataset.tau_x[i:end].numpy())

            predictions["wall_shear_y"].append(typred.cpu().numpy())
            truths["wall_shear_y"].append(dataset.tau_y[i:end].numpy())

            predictions["wall_shear_z"].append(tzpred.cpu().numpy())
            truths["wall_shear_z"].append(dataset.tau_z[i:end].numpy())

            del (bx, by, bz, bt,
                 ppred, upred, vpred, wpred,
                 txpred, typred, tzpred)
            torch.cuda.empty_cache()

    for var in variables:
        predictions[var] = np.concatenate(predictions[var], axis=0)
        truths[var] = np.concatenate(truths[var], axis=0)

        predictions[var] = dataset.scalers[var].inverse_transform(predictions[var].reshape(-1,1)).flatten()
        truths[var] = dataset.scalers[var].inverse_transform(truths[var].reshape(-1,1)).flatten()

    wss_pred = np.sqrt(
        predictions["wall_shear_x"]**2 + predictions["wall_shear_y"]**2 + predictions["wall_shear_z"]**2
    )
    wss_true = np.sqrt(
        truths["wall_shear_x"]**2 + truths["wall_shear_y"]**2 + truths["wall_shear_z"]**2
    )

    run_plot_dir = os.path.join(config.plot_dir, run_id)
    ensure_dir(run_plot_dir)

    # Pressure
    fig_p, axs_p = plt.subplots(2, 2, figsize=(12, 8))
    pmin = min(truths["pressure"].min(), predictions["pressure"].min())
    pmax = max(truths["pressure"].max(), predictions["pressure"].max())

    # CFD Pressure (XY)
    plot_scatter(axs_p[0,0], x_sample, y_sample, truths["pressure"],
                 "CFD Pressure (XY)", "X [m]", "Y [m]", "viridis", "Pressure [Pa]",
                 vmin=pmin, vmax=pmax)
    # PINN Pressure (XY)
    plot_scatter(axs_p[0,1], x_sample, y_sample, predictions["pressure"],
                 "PINN Pressure (XY)", "X [m]", "Y [m]", "viridis", "Pressure [Pa]",
                 vmin=pmin, vmax=pmax)
    # CFD Pressure (XZ)
    plot_scatter(axs_p[1,0], x_sample, z_sample, truths["pressure"],
                 "CFD Pressure (XZ)", "X [m]", "Z [m]", "viridis", "Pressure [Pa]",
                 vmin=pmin, vmax=pmax)
    # PINN Pressure (XZ)
    plot_scatter(axs_p[1,1], x_sample, z_sample, predictions["pressure"],
                 "PINN Pressure (XZ)", "X [m]", "Z [m]", "viridis", "Pressure [Pa]",
                 vmin=pmin, vmax=pmax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_p.suptitle(f"Pressure Distribution - {run_id}", fontsize=16)
    press_path = os.path.join(run_plot_dir, f"pressure_distribution_{run_id}.png")
    plt.savefig(press_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close(fig_p)
    print(f"Saved Pressure distribution to '{press_path}'.")

    # WSS Magnitude
    fig_w, axs_w = plt.subplots(2, 2, figsize=(12, 8))
    wmin = min(wss_true.min(), wss_pred.min())
    wmax = max(wss_true.max(), wss_pred.max())

    # CFD WSS (XY)
    plot_scatter(axs_w[0,0], x_sample, y_sample, wss_true,
                 "CFD WSS (XY)", "X [m]", "Y [m]", "inferno", "WSS [Pa]",
                 vmin=wmin, vmax=wmax)
    # PINN WSS (XY)
    plot_scatter(axs_w[0,1], x_sample, y_sample, wss_pred,
                 "PINN WSS (XY)", "X [m]", "Y [m]", "inferno", "WSS [Pa]",
                 vmin=wmin, vmax=wmax)
    # CFD WSS (XZ)
    plot_scatter(axs_w[1,0], x_sample, z_sample, wss_true,
                 "CFD WSS (XZ)", "X [m]", "Z [m]", "inferno", "WSS [Pa]",
                 vmin=wmin, vmax=wmax)
    # PINN WSS (XZ)
    plot_scatter(axs_w[1,1], x_sample, z_sample, wss_pred,
                 "PINN WSS (XZ)", "X [m]", "Z [m]", "inferno", "WSS [Pa]",
                 vmin=wmin, vmax=wmax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_w.suptitle(f"WSS Magnitude - {run_id}", fontsize=16)
    wss_path = os.path.join(run_plot_dir, f"wss_magnitude_distribution_{run_id}.png")
    plt.savefig(wss_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close(fig_w)
    print(f"Saved WSS magnitude distribution to '{wss_path}'.")


def plot_wss_histogram(
    dataset: CFDDataset,
    models: Dict[str, torch.nn.Module],
    config: Config,
    run_id: str
):
    """
    Histogram of WSS magnitude (CFD vs PINN).
    """
    for m in models.values():
        m.eval()

    varz = ["wall_shear_x", "wall_shear_y", "wall_shear_z"]
    preds = {v: [] for v in varz}
    truth = {v: [] for v in varz}

    x_s = dataset.x.numpy()
    y_s = dataset.y.numpy()
    z_s = dataset.z.numpy()
    t_s = dataset.t.numpy()

    bs = 1024
    nsamples = len(x_s)

    with torch.no_grad():
        for i in range(0, nsamples, bs):
            end = i + bs
            bx = torch.tensor(x_s[i:end], dtype=torch.float32, device=config.device)
            by = torch.tensor(y_s[i:end], dtype=torch.float32, device=config.device)
            bz = torch.tensor(z_s[i:end], dtype=torch.float32, device=config.device)
            bt = torch.tensor(t_s[i:end], dtype=torch.float32, device=config.device)

            tx = models["tau_x"](bx, by, bz, bt)
            ty = models["tau_y"](bx, by, bz, bt)
            tz = models["tau_z"](bx, by, bz, bt)

            preds["wall_shear_x"].append(tx.cpu().numpy())
            preds["wall_shear_y"].append(ty.cpu().numpy())
            preds["wall_shear_z"].append(tz.cpu().numpy())

            truth["wall_shear_x"].append(dataset.tau_x[i:end].numpy())
            truth["wall_shear_y"].append(dataset.tau_y[i:end].numpy())
            truth["wall_shear_z"].append(dataset.tau_z[i:end].numpy())

            del (bx, by, bz, bt, tx, ty, tz)
            torch.cuda.empty_cache()

    for v in varz:
        preds[v] = np.concatenate(preds[v], axis=0)
        truth[v] = np.concatenate(truth[v], axis=0)
        preds[v] = dataset.scalers[v].inverse_transform(preds[v].reshape(-1, 1)).flatten()
        truth[v] = dataset.scalers[v].inverse_transform(truth[v].reshape(-1, 1)).flatten()

    wss_pred = np.sqrt(
        preds["wall_shear_x"]**2 + preds["wall_shear_y"]**2 + preds["wall_shear_z"]**2
    )
    wss_true = np.sqrt(
        truth["wall_shear_x"]**2 + truth["wall_shear_y"]**2 + truth["wall_shear_z"]**2
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(wss_true, bins=50, alpha=0.5, label='CFD', color='blue', density=True)
    ax.hist(wss_pred, bins=50, alpha=0.5, label='PINN', color='red', density=True)
    ax.set_xlabel("WSS Magnitude [Pa]")
    ax.set_ylabel("Density")
    ax.set_title(f"WSS Magnitude Histogram - {run_id}")
    ax.legend()

    run_plot_dir = os.path.join(config.plot_dir, run_id)
    ensure_dir(run_plot_dir)
    out_path = os.path.join(run_plot_dir, f"wss_histogram_{run_id}.png")
    plt.savefig(out_path, dpi=config.plot_resolution, bbox_inches='tight')
    plt.close()
    print(f"Saved WSS histogram to '{out_path}'.")


# def plot_time_varying_slices(
#     dataset: CFDDataset,
#     models: Dict[str, torch.nn.Module],
#     config: Config,
#     run_id: str,
#     variable: str = "pressure",
#     num_times: int = 5
# ):
#     """
#     Example: Show how 'variable' changes at different times (slices).
#     We'll pick N time slices and do 2D scatter for each slice.
#     """
#     for m in models.values():
#         m.eval()

#     # Determine time range
#     t_vals = dataset.t.numpy().flatten()
#     t_min, t_max = t_vals.min(), t_vals.max()
#     time_slices = np.linspace(t_min, t_max, num_times)

#     x_s = dataset.x.numpy().flatten()
#     y_s = dataset.y.numpy().flatten()
#     z_s = dataset.z.numpy().flatten()

#     # We'll do a row of subplots
#     fig, axes = plt.subplots(1, num_times, figsize=(5*num_times, 5), sharey=True)

#     # Precompute predictions for entire dataset
#     batch_size = 1024
#     N = len(x_s)

#     predictions = []
#     with torch.no_grad():
#         for i in range(0, N, batch_size):
#             end = i + batch_size
#             bx = torch.tensor(x_s[i:end], dtype=torch.float32, device=config.device).unsqueeze(1)
#             by = torch.tensor(y_s[i:end], dtype=torch.float32, device=config.device).unsqueeze(1)
#             bz = torch.tensor(z_s[i:end], dtype=torch.float32, device=config.device).unsqueeze(1)
#             bt = torch.tensor(t_vals[i:end], dtype=torch.float32, device=config.device).unsqueeze(1)

#             if variable in ["pressure"]:
#                 pred_var = models["p"](bx, by, bz, bt)
#             elif variable == "velocity_u":
#                 pred_var = models["u"](bx, by, bz, bt)
#             elif variable == "velocity_v":
#                 pred_var = models["v"](bx, by, bz, bt)
#             elif variable == "velocity_w":
#                 pred_var = models["w"](bx, by, bz, bt)
#             else:
#                 # fallback: assume user wants 'pressure'
#                 pred_var = models["p"](bx, by, bz, bt)

#             predictions.append(pred_var.cpu().numpy())

#             del (bx, by, bz, bt, pred_var)
#             torch.cuda.empty_cache()

#     predictions = np.concatenate(predictions, axis=0).flatten()
#     # Inverse transform
#     predictions = dataset.scalers[variable].inverse_transform(predictions.reshape(-1, 1)).flatten()

#     # Plot each time slice
#     for i, ax in enumerate(axes):
#         t_slice = time_slices[i]
#         # pick a small tolerance
#         tol = 1e-3
#         mask = np.abs(t_vals - t_slice) < tol
#         if not np.any(mask):
#             # fallback: pick nearest time
#             idx = np.argmin(np.abs(t_vals - t_slice))
#             mask = np.array([False]*len(t_vals))
#             mask[idx] = True

#         x_sel = x_s[mask]
#         y_sel = y_s[mask]
#         var_sel = predictions[mask]

#         sc = ax.scatter(x_sel, y_sel, c=var_sel, cmap="plasma", alpha=0.8)
#         ax.set_title(f"Time ~ {t_slice:.3f} s")
#         ax.set_xlabel("X [m]")
#         if i == 0:
#             ax.set_ylabel("Y [m]")

#         cbar = plt.colorbar(sc, ax=ax)
#         cbar.set_label(f"{variable.title()}")

#     fig.suptitle(f"Time-Varying {variable.title()} - {run_id}", fontsize=16)
#     plt.tight_layout()
#     out_dir = os.path.join(config.plot_dir, run_id)
#     ensure_dir(out_dir)
#     out_file = os.path.join(out_dir, f"{variable}_time_varying_{run_id}.png")
#     plt.savefig(out_file, dpi=config.plot_resolution, bbox_inches='tight')
#     plt.close()
#     print(f"Saved time-varying {variable} slices to '{out_file}'.")


# def plot_3d_representation(
#     dataset: CFDDataset,
#     models: Dict[str, torch.nn.Module],
#     config: Config,
#     run_id: str,
#     variable: str = "pressure",
#     time_value: float = 0.1,
#     tolerance: float = 1e-3
# ):
#     """
#     Create a 3D scatter plot (x,y,z) of a chosen variable at a specific time_value ± tolerance.
#     """
#     for m in models.values():
#         m.eval()

#     x_s = dataset.x.numpy().flatten()
#     y_s = dataset.y.numpy().flatten()
#     z_s = dataset.z.numpy().flatten()
#     t_s = dataset.t.numpy().flatten()

#     # mask the data
#     mask = np.abs(t_s - time_value) < tolerance
#     if not np.any(mask):
#         print(f"No points found near time={time_value}s within tolerance={tolerance}.")
#         return

#     x_sel = x_s[mask]
#     y_sel = y_s[mask]
#     z_sel = z_s[mask]
#     t_sel = t_s[mask]

#     # Predict
#     with torch.no_grad():
#         bx = torch.tensor(x_sel, dtype=torch.float32, device=config.device).unsqueeze(1)
#         by = torch.tensor(y_sel, dtype=torch.float32, device=config.device).unsqueeze(1)
#         bz = torch.tensor(z_sel, dtype=torch.float32, device=config.device).unsqueeze(1)
#         bt = torch.tensor(t_sel, dtype=torch.float32, device=config.device).unsqueeze(1)

#         if variable == "pressure":
#             v_pred = models["p"](bx, by, bz, bt)
#         elif variable == "velocity_u":
#             v_pred = models["u"](bx, by, bz, bt)
#         elif variable == "velocity_v":
#             v_pred = models["v"](bx, by, bz, bt)
#         elif variable == "velocity_w":
#             v_pred = models["w"](bx, by, bz, bt)
#         else:
#             print(f"Unsupported variable: {variable}, defaulting to 'pressure'.")
#             v_pred = models["p"](bx, by, bz, bt)

#     v_pred = v_pred.cpu().numpy().flatten()
#     v_pred = dataset.scalers[variable].inverse_transform(v_pred.reshape(-1,1)).flatten()

#     # 3D scatter
#     from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
#     fig = plt.figure(figsize=(10,7))
#     ax = fig.add_subplot(111, projection='3d')
#     p = ax.scatter(x_sel, y_sel, z_sel, c=v_pred, cmap="viridis", alpha=0.8)
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_zlabel("Z [m]")
#     ax.set_title(f"{variable.title()} at t={time_value:.3f}s ± {tolerance}")

#     cbar = plt.colorbar(p, ax=ax, shrink=0.6)
#     cbar.set_label(variable)

#     out_dir = os.path.join(config.plot_dir, run_id)
#     ensure_dir(out_dir)
#     out_file = os.path.join(out_dir, f"3D_{variable}_{run_id}_t{time_value:.3f}.png")
#     plt.savefig(out_file, dpi=config.plot_resolution, bbox_inches='tight')
#     plt.close()
#     print(f"Saved 3D representation to '{out_file}'.")
