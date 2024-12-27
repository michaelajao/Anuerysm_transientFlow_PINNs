# src/train.py
import numpy as np
import math  # Add this import

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # Add this import
from torch.autograd import grad
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .utils import EarlyStopping, get_device, initialize_models

def compute_physics_loss(p_pred, u_pred, v_pred, w_pred, x, y, z, t, rho, mu):
    """
    Computes the physics-based loss incorporating Navier-Stokes equations and continuity.

    Args:
        p_pred, u_pred, v_pred, w_pred: Predicted flow variables.
        x, y, z, t: Input variables with gradients enabled.
        rho: Density of the fluid.
        mu: Dynamic viscosity.

    Returns:
        torch.Tensor: Computed physics-based loss.
    """
    # Compute first derivatives
    u_x = grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_y = grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_z = grad(u_pred, z, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]

    v_x = grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_y = grad(v_pred, y, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_z = grad(v_pred, z, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_t = grad(v_pred, t, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]

    w_x = grad(w_pred, x, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_y = grad(w_pred, y, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_z = grad(w_pred, z, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_t = grad(w_pred, t, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]

    p_x = grad(p_pred, x, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]
    p_y = grad(p_pred, y, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]
    p_z = grad(p_pred, z, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]

    # Compute second derivatives
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    # Navier-Stokes Equations Residuals
    residual_u = (
        u_t + u_pred * u_x + v_pred * u_y + w_pred * u_z
        + (1 / rho) * p_x - (mu / rho) * (u_xx + u_yy + u_zz)
    )
    residual_v = (
        v_t + u_pred * v_x + v_pred * v_y + w_pred * v_z
        + (1 / rho) * p_y - (mu / rho) * (v_xx + v_yy + v_zz)
    )
    residual_w = (
        w_t + u_pred * w_x + v_pred * w_y + w_pred * w_z
        + (1 / rho) * p_z - (mu / rho) * (w_xx + w_yy + w_zz)
    )

    # Continuity Equation
    continuity = u_x + v_y + w_z

    mse = nn.MSELoss()
    loss_nse = mse(residual_u, torch.zeros_like(residual_u)) + \
               mse(residual_v, torch.zeros_like(residual_v)) + \
               mse(residual_w, torch.zeros_like(residual_w))
    loss_continuity = mse(continuity, torch.zeros_like(continuity))

    return loss_nse + loss_continuity

def compute_boundary_loss(u_bc_pred, v_bc_pred, w_bc_pred):
    """
    Computes the boundary condition loss enforcing no-slip conditions.

    Args:
        u_bc_pred, v_bc_pred, w_bc_pred: Predicted velocities at boundary points.

    Returns:
        torch.Tensor: Computed boundary loss.
    """
    mse = nn.MSELoss()
    loss_bc = mse(u_bc_pred, torch.zeros_like(u_bc_pred)) + \
              mse(v_bc_pred, torch.zeros_like(v_bc_pred)) + \
              mse(w_bc_pred, torch.zeros_like(w_bc_pred))
    return loss_bc

def compute_data_loss(
    p_pred, p_true,
    u_pred, u_true,
    v_pred, v_true,
    w_pred, w_true,
    tau_x_pred, tau_x_true,
    tau_y_pred, tau_y_true,
    tau_z_pred, tau_z_true,
):
    mse = nn.MSELoss()
    loss_p = mse(p_pred, p_true)
    loss_u = mse(u_pred, u_true)
    loss_v = mse(v_pred, v_true)
    loss_w = mse(w_pred, w_true)
    loss_tau_x = mse(tau_x_pred, tau_x_true)
    loss_tau_y = mse(tau_y_pred, tau_y_true)
    loss_tau_z = mse(tau_z_pred, tau_z_true)
    return loss_p + loss_u + loss_v + loss_w + loss_tau_x + loss_tau_y + loss_tau_z

def compute_inlet_loss(u_pred, v_pred, w_pred, t, heart_rate=120):
    """
    Computes the inlet velocity profile loss based on a sinusoidal wave.

    Args:
        u_pred, v_pred, w_pred: Predicted velocities at inlet.
        t: Time tensor.
        heart_rate (int): Heart rate in beats per minute.

    Returns:
        torch.Tensor: Computed inlet loss.
    """
    period = 60 / heart_rate  # seconds
    t_mod = torch.fmod(t, period)
    u_inlet_true = torch.where(
        t_mod <= 0.218,
        0.5 * torch.sin(4 * np.pi * (t_mod + 0.0160236)),
        torch.tensor(0.1, device=u_pred.device),
    )
    v_inlet_true = torch.zeros_like(u_inlet_true)
    w_inlet_true = torch.zeros_like(u_inlet_true)
    mse = nn.MSELoss()
    loss_inlet = mse(u_pred, u_inlet_true) + mse(v_pred, v_inlet_true) + mse(w_pred, w_inlet_true)
    return loss_inlet

def initialize_optimizer_scheduler(models, config, logger):
    """
    Sets up the optimizer and learning rate scheduler.

    Args:
        models (dict): Dictionary of models.
        config (Config): Configuration object.
        logger (logging.Logger): Logger object.

    Returns:
        tuple: Optimizer and scheduler objects.
    """
    params = []
    for model in models.values():
        params += list(model.parameters())
    
    optimizer = optim.AdamW(
        params=params,
        lr=config.learning_rate,
        betas=config.optimizer_params["betas"],
        eps=config.optimizer_params["eps"],
        weight_decay=1e-4,
    )
    scheduler = StepLR(
        optimizer,
        step_size=config.scheduler_params["step_size"],
        gamma=config.scheduler_params["gamma"],
    )
    logger.info("Optimizer and scheduler initialized.")

    return optimizer, scheduler

def train_pinn(
    models: dict,
    dataloader,
    config,
    optimizer,
    scheduler,
    early_stopping: EarlyStopping,
    logger,
    run_id: str,
):
    """
    Trains the PINN models.

    Args:
        models (dict): Dictionary of PINN models.
        dataloader: DataLoader for training data.
        config: Configuration object.
        optimizer: Optimizer object.
        scheduler: Scheduler object.
        early_stopping (EarlyStopping): Early stopping object.
        logger: Logger object.
        run_id (str): Identifier for the current run.

    Returns:
        dict: History of losses.
    """
    scaler = GradScaler()
    loss_history = {"total": [], "physics": [], "boundary": [], "data": [], "inlet": []}

    epochs = config.epochs
    rho = config.rho
    mu = config.mu

    logger.info("Starting training loop.")
    for epoch in tqdm(range(1, epochs + 1), desc="Training", leave=False):
        for model in models.values():
            model.train()

        epoch_loss_total = 0.0
        epoch_loss_physics = 0.0
        epoch_loss_boundary = 0.0
        epoch_loss_data = 0.0
        epoch_loss_inlet = 0.0

        for batch in dataloader:
            (
                x_batch,
                y_batch,
                z_batch,
                t_batch,
                p_true,
                u_true,
                v_true,
                w_true,
                tau_x_true,
                tau_y_true,
                tau_z_true,
                is_boundary,
            ) = batch

            # Move batch to device
            x_batch = x_batch.to(config.device).requires_grad_(True)
            y_batch = y_batch.to(config.device).requires_grad_(True)
            z_batch = z_batch.to(config.device).requires_grad_(True)
            t_batch = t_batch.to(config.device).requires_grad_(True)

            p_true = p_true.to(config.device)
            u_true = u_true.to(config.device)
            v_true = v_true.to(config.device)
            w_true = w_true.to(config.device)
            tau_x_true = tau_x_true.to(config.device)
            tau_y_true = tau_y_true.to(config.device)
            tau_z_true = tau_z_true.to(config.device)
            is_boundary = is_boundary.to(config.device)

            optimizer.zero_grad()

            with autocast():
                # Forward pass
                p_pred = models["pressure"](x_batch, y_batch, z_batch, t_batch)
                u_pred = models["u_velocity"](x_batch, y_batch, z_batch, t_batch)
                v_pred = models["v_velocity"](x_batch, y_batch, z_batch, t_batch)
                w_pred = models["w_velocity"](x_batch, y_batch, z_batch, t_batch)
                tau_x_pred = models["tau_x"](x_batch, y_batch, z_batch, t_batch)
                tau_y_pred = models["tau_y"](x_batch, y_batch, z_batch, t_batch)
                tau_z_pred = models["tau_z"](x_batch, y_batch, z_batch, t_batch)

                # Compute losses
                loss_physics = compute_physics_loss(p_pred, u_pred, v_pred, w_pred, x_batch, y_batch, z_batch, t_batch, rho, mu)
                loss_boundary = compute_boundary_loss(u_pred, v_pred, w_pred)
                loss_data = compute_data_loss(
                    p_pred, p_true,
                    u_pred, u_true,
                    v_pred, v_true,
                    w_pred, w_true,
                    tau_x_pred, tau_x_true,
                    tau_y_pred, tau_y_true,
                    tau_z_pred, tau_z_true,
                )
                loss_inlet = compute_inlet_loss(u_pred, v_pred, w_pred, t_batch)
                total_loss = loss_physics + loss_boundary + loss_data + loss_inlet

            # Backward pass and optimization with mixed precision
            scaler.scale(total_loss).backward()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(list(models["pressure"].parameters()) +
                            list(models["u_velocity"].parameters()) +
                            list(models["v_velocity"].parameters()) +
                            list(models["w_velocity"].parameters()) +
                            list(models["tau_x"].parameters()) +
                            list(models["tau_y"].parameters()) +
                            list(models["tau_z"].parameters()), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # Check for NaN in loss
            if torch.isnan(total_loss):
                logger.error("NaN detected in total_loss. Stopping training.")
                early_stopping(1e9, list(models.values()), optimizer, scheduler, run_id, config.model_dir)
                break

            # Accumulate losses
            epoch_loss_total += total_loss.item()
            epoch_loss_physics += loss_physics.item()
            epoch_loss_boundary += loss_boundary.item()
            epoch_loss_data += loss_data.item()
            epoch_loss_inlet += loss_inlet.item()

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

        # Step the scheduler
        scheduler.step()

        # Calculate average losses
        avg_loss_total = epoch_loss_total / len(dataloader)
        avg_loss_physics = epoch_loss_physics / len(dataloader)
        avg_loss_boundary = epoch_loss_boundary / len(dataloader)
        avg_loss_data = epoch_loss_data / len(dataloader)
        avg_loss_inlet = epoch_loss_inlet / len(dataloader)

        # Record loss history
        loss_history["total"].append(avg_loss_total)
        loss_history["physics"].append(avg_loss_physics)
        loss_history["boundary"].append(avg_loss_boundary)
        loss_history["data"].append(avg_loss_data)
        loss_history["inlet"].append(avg_loss_inlet)

        # Early stopping check and best model saving
        early_stopping(avg_loss_total, list(models.values()), optimizer, scheduler, run_id, config.model_dir)
        if early_stopping.early_stop:
            logger.info("Early stopping condition met.")
            break

        # Logging at intervals
        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"Epoch [{epoch}/{epochs}] - Total Loss: {avg_loss_total:.6f}, "
                        f"Physics Loss: {avg_loss_physics:.6f}, Boundary Loss: {avg_loss_boundary:.6f}, "
                        f"Data Loss: {avg_loss_data:.6f}, Inlet Loss: {avg_loss_inlet:.6f}")

    logger.info("Training completed.")
    return loss_history
