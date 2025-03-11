# aneurysm_pinns/modeling/predict.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from aneurysm_pinns.config import Config
from aneurysm_pinns.modeling.model import initialize_models
from aneurysm_pinns.dataset import CFDDataset
from aneurysm_pinns.utils import ensure_dir


def compute_metrics(predictions: Dict[str, np.ndarray], truths: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Computes evaluation metrics for the model predictions against the true values.

    Args:
        predictions (Dict[str, np.ndarray]): Model predictions for different variables.
        truths (Dict[str, np.ndarray]): True values for the corresponding variables.

    Returns:
        Dict[str, float]: A dictionary containing computed metrics including R², NRMSE, and MAE for each variable.
    """
    variables = [
        "pressure",
        "velocity_u",
        "velocity_v",
        "velocity_w",
        "wall_shear_x",
        "wall_shear_y",
        "wall_shear_z",
    ]
    out = {}
    for var in variables:
        r2 = r2_score(truths[var], predictions[var])
        denom = (truths[var].max() - truths[var].min()) + 1e-8
        nrmse = np.sqrt(mean_squared_error(truths[var], predictions[var])) / denom
        mae = mean_absolute_error(truths[var], predictions[var])
        out[f"{var}_R2"] = r2
        out[f"{var}_NRMSE"] = nrmse
        out[f"{var}_MAE"] = mae
    out["Total_MAE"] = np.mean([out[f"{v}_MAE"] for v in variables])
    return out


def evaluate_pinn(models: Dict[str, torch.nn.Module], dataloader: DataLoader, dataset: CFDDataset, config: Config, run_id: str) -> Dict[str, float]:
    """
    Evaluates the trained PINN models on the given dataset.
    
    Args:
        models (Dict[str, torch.nn.Module]): Dictionary of trained PINN models.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        dataset (CFDDataset): Dataset object for the evaluation dataset.
        config (Config): Configuration object.
        run_id (str): Run ID for the current experiment.
        
    """
    for m in models.values():
        m.eval()

    variables = [
        "pressure",
        "velocity_u",
        "velocity_v",
        "velocity_w",
        "wall_shear_x",
        "wall_shear_y",
        "wall_shear_z",
    ]

    preds = {var: [] for var in variables}
    truth = {var: [] for var in variables}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            (x_batch, y_batch, z_batch, t_batch,
             p_true, u_true, v_true, w_true,
             tau_x_true, tau_y_true, tau_z_true,
             _) = batch

            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            z_batch = z_batch.to(config.device)
            t_batch = t_batch.to(config.device)

            p_pred = models["p"](x_batch, y_batch, z_batch, t_batch)
            u_pred = models["u"](x_batch, y_batch, z_batch, t_batch)
            v_pred = models["v"](x_batch, y_batch, z_batch, t_batch)
            w_pred = models["w"](x_batch, y_batch, z_batch, t_batch)
            tx_pred = models["tau_x"](x_batch, y_batch, z_batch, t_batch)
            ty_pred = models["tau_y"](x_batch, y_batch, z_batch, t_batch)
            tz_pred = models["tau_z"](x_batch, y_batch, z_batch, t_batch)

            preds["pressure"].append(p_pred.cpu().numpy())
            truth["pressure"].append(p_true.numpy())

            preds["velocity_u"].append(u_pred.cpu().numpy())
            truth["velocity_u"].append(u_true.numpy())

            preds["velocity_v"].append(v_pred.cpu().numpy())
            truth["velocity_v"].append(v_true.numpy())

            preds["velocity_w"].append(w_pred.cpu().numpy())
            truth["velocity_w"].append(w_true.numpy())

            preds["wall_shear_x"].append(tx_pred.cpu().numpy())
            truth["wall_shear_x"].append(tau_x_true.numpy())

            preds["wall_shear_y"].append(ty_pred.cpu().numpy())
            truth["wall_shear_y"].append(tau_y_true.numpy())

            preds["wall_shear_z"].append(tz_pred.cpu().numpy())
            truth["wall_shear_z"].append(tau_z_true.numpy())

            del (x_batch, y_batch, z_batch, t_batch, p_true, u_true, v_true, w_true,
                 tau_x_true, tau_y_true, tau_z_true, p_pred, u_pred, v_pred, w_pred,
                 tx_pred, ty_pred, tz_pred)
            torch.cuda.empty_cache()

    # Merge & inverse transform
    for var in variables:
        preds[var] = np.concatenate(preds[var], axis=0)
        truth[var] = np.concatenate(truth[var], axis=0)

        preds[var] = dataset.scalers[var].inverse_transform(preds[var].reshape(-1, 1)).flatten()
        truth[var] = dataset.scalers[var].inverse_transform(truth[var].reshape(-1, 1)).flatten()

    metrics = compute_metrics(preds, truth)

    # Save metrics
    out_dict = {"Run_ID": run_id, "Total_MAE": metrics["Total_MAE"]}
    for var in variables:
        out_dict[f"{var}_R2"] = metrics[f"{var}_R2"]
        out_dict[f"{var}_NRMSE"] = metrics[f"{var}_NRMSE"]
        out_dict[f"{var}_MAE"] = metrics[f"{var}_MAE"]

    dfm = pd.DataFrame([out_dict])
    out_path = os.path.join(config.metrics_dir, run_id, f"metrics_summary_{run_id}.csv")
    ensure_dir(os.path.dirname(out_path))
    dfm.to_csv(out_path, index=False)
    print(f"Saved metrics to '{out_path}'.")

    return metrics
