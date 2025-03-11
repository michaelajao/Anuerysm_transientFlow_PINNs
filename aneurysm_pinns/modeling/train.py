# aneurysm_pinns/modeling/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict

import numpy as np
from tqdm.auto import tqdm

from aneurysm_pinns.config import Config
from aneurysm_pinns.modeling.model import initialize_models
from aneurysm_pinns.utils import ensure_dir

# ---------------------------------------
# Loss Functions
# ---------------------------------------

def compute_physics_loss(
    p_pred, u_pred, v_pred, w_pred,
    x, y, z, t,
    rho, mu
):
    """
    Computes the Physics-based Loss incorporating Navier-Stokes Equations and Continuity.

    This function calculates the residuals of the momentum equations and the continuity equation,
    and computes their Mean Squared Errors (MSE).

    Args:
        p_pred (torch.Tensor): Predicted pressure.
        u_pred (torch.Tensor): Predicted u-velocity.
        v_pred (torch.Tensor): Predicted v-velocity.
        w_pred (torch.Tensor): Predicted w-velocity.
        x (torch.Tensor): X-coordinate tensor with gradients enabled.
        y (torch.Tensor): Y-coordinate tensor with gradients enabled.
        z (torch.Tensor): Z-coordinate tensor with gradients enabled.
        t (torch.Tensor): Time tensor with gradients enabled.
        rho (float): Fluid density.
        mu (float): Dynamic viscosity.

    Returns:
        torch.Tensor: Combined physics-based loss.
    """
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u_pred, z, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]

    v_x = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v_pred, y, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v_pred, z, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]
    v_t = torch.autograd.grad(v_pred, t, grad_outputs=torch.ones_like(v_pred), retain_graph=True, create_graph=True)[0]

    w_x = torch.autograd.grad(w_pred, x, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w_pred, y, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w_pred, z, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]
    w_t = torch.autograd.grad(w_pred, t, grad_outputs=torch.ones_like(w_pred), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p_pred, x, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p_pred, y, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p_pred, z, grad_outputs=torch.ones_like(p_pred), retain_graph=True, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    # Momentum eqns:
    residual_u = (u_t + u_pred*u_x + v_pred*u_y + w_pred*u_z) + (1.0 / rho)*p_x - (mu/rho)*(u_xx + u_yy + u_zz)
    residual_v = (v_t + u_pred*v_x + v_pred*v_y + w_pred*v_z) + (1.0 / rho)*p_y - (mu/rho)*(v_xx + v_yy + v_zz)
    residual_w = (w_t + u_pred*w_x + v_pred*w_y + w_pred*w_z) + (1.0 / rho)*p_z - (mu/rho)*(w_xx + w_yy + w_zz)

    # Continuity
    continuity = u_x + v_y + w_z

    mse = nn.MSELoss()
    loss_nse = mse(residual_u, torch.zeros_like(residual_u)) \
             + mse(residual_v, torch.zeros_like(residual_v)) \
             + mse(residual_w, torch.zeros_like(residual_w))
    loss_cont = mse(continuity, torch.zeros_like(continuity))

    return loss_nse + loss_cont


def compute_boundary_loss(u_bc_pred, v_bc_pred, w_bc_pred) -> torch.Tensor:
    """
    Computes the Boundary Condition Loss enforcing no-slip conditions.

    No-slip condition implies that the velocity components at the boundary are zero.

    Args:
        u_bc_pred (torch.Tensor): Predicted u-velocity at boundary points.
        v_bc_pred (torch.Tensor): Predicted v-velocity at boundary points.
        w_bc_pred (torch.Tensor): Predicted w-velocity at boundary points.

    Returns:
        torch.Tensor: Boundary condition loss.
    """
    mse = nn.MSELoss()
    return (mse(u_bc_pred, torch.zeros_like(u_bc_pred))
          + mse(v_bc_pred, torch.zeros_like(v_bc_pred))
          + mse(w_bc_pred, torch.zeros_like(w_bc_pred)))


def compute_data_loss(
    p_pred, p_true,
    u_pred, u_true,
    v_pred, v_true,
    w_pred, w_true,
    tau_x_pred, tau_x_true,
    tau_y_pred, tau_y_true,
    tau_z_pred, tau_z_true
) -> torch.Tensor:
        """
    Computes the Supervised Data Loss using Mean Squared Error (MSE).

    This loss measures the discrepancy between the PINN predictions and the actual CFD data.

    Args:
        p_pred (torch.Tensor): Predicted pressure.
        p_true (torch.Tensor): True pressure.
        u_pred (torch.Tensor): Predicted u-velocity.
        u_true (torch.Tensor): True u-velocity.
        v_pred (torch.Tensor): Predicted v-velocity.
        v_true (torch.Tensor): True v-velocity.
        w_pred (torch.Tensor): Predicted w-velocity.
        w_true (torch.Tensor): True w-velocity.
        tau_x_pred (torch.Tensor): Predicted tau_x.
        tau_x_true (torch.Tensor): True tau_x.
        tau_y_pred (torch.Tensor): Predicted tau_y.
        tau_y_true (torch.Tensor): True tau_y.
        tau_z_pred (torch.Tensor): Predicted tau_z.
        tau_z_true (torch.Tensor): True tau_z.

    Returns:
        torch.Tensor: Combined data loss for all variables.
    """
    mse = nn.MSELoss()
    loss_p = mse(p_pred, p_true)
    loss_u = mse(u_pred, u_true)
    loss_v = mse(v_pred, v_true)
    loss_w = mse(w_pred, w_true)
    loss_tx = mse(tau_x_pred, tau_x_true)
    loss_ty = mse(tau_y_pred, tau_y_true)
    loss_tz = mse(tau_z_pred, tau_z_true)
    return loss_p + loss_u + loss_v + loss_w + loss_tx + loss_ty + loss_tz


def compute_inlet_loss(u_pred, v_pred, w_pred, t, heart_rate=120) -> torch.Tensor:
    """
    Computes the Inlet Velocity Profile Loss based on a sinusoidal wave.

    This loss enforces an inflow boundary condition with a specified velocity profile.

    Args:
        u_pred (torch.Tensor): Predicted u-velocity.
        v_pred (torch.Tensor): Predicted v-velocity.
        w_pred (torch.Tensor): Predicted w-velocity.
        t (torch.Tensor): Time tensor.
        heart_rate (int): Heart rate in beats per minute to define the sinusoidal period.

    Returns:
        torch.Tensor: Inlet condition loss.
    """
    period = 60.0 / heart_rate
    t_mod = torch.fmod(t, period)

    # up to 0.218s is peak systole, else 0.1
    u_inlet_true = torch.where(
        t_mod <= 0.218,
        0.5 * torch.sin(4.0 * np.pi * (t_mod + 0.0160236)),
        torch.full_like(t_mod, 0.1)
    )
    v_inlet_true = torch.zeros_like(u_inlet_true)
    w_inlet_true = torch.zeros_like(u_inlet_true)

    mse = nn.MSELoss()
    return (mse(u_pred, u_inlet_true)
          + mse(v_pred, v_inlet_true)
          + mse(w_pred, w_inlet_true))


# ---------------------------------------
# Early Stopping
# ---------------------------------------

class EarlyStopping:
    """
    Stops training if no improvement in `patience` epochs.
    """
    def __init__(self, patience: int = 100, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss: float, models: Dict[str, nn.Module], optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, run_id: str, model_dir: str):
        if self.best_loss is None:
            self.best_loss = loss
            print(f"Initial loss: {self.best_loss:.6f}")
            self.save_checkpoint(models, optimizer, scheduler, run_id, model_dir)
        elif (self.best_loss - loss) > self.min_delta:
            self.best_loss = loss
            self.counter = 0
            print(f"Loss improved to {self.best_loss:.6f}; reset counter.")
            self.save_checkpoint(models, optimizer, scheduler, run_id, model_dir)
        else:
            self.counter += 1
            print(f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")

    def save_checkpoint(self, models, optimizer, scheduler, run_id, model_dir):
        best_model_path = os.path.join(model_dir, run_id, f"best_model_{run_id}.pt")
        ensure_dir(os.path.dirname(best_model_path))
        checkpoint = {}
        for key, model in models.items():
            checkpoint[f"{key}_state_dict"] = model.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint["best_loss"] = self.best_loss
        torch.save(checkpoint, best_model_path)
        print(f"Saved best model to '{best_model_path}'.")


# ---------------------------------------
# Optimizer & Training
# ---------------------------------------

def setup_optimizer_scheduler(models: Dict[str, nn.Module], config: Config):
    """
    AdamW + StepLR
    """
    all_params = []
    for m in models.values():
        all_params += list(m.parameters())

    optimizer = optim.AdamW(
        all_params,
        lr=config.learning_rate,
        betas=config.optimizer_params["betas"],
        eps=config.optimizer_params["eps"],
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_params["step_size"],
        gamma=config.scheduler_params["gamma"]
    )

    print("Optimizer & scheduler initialized.")
    return optimizer, scheduler


def train_pinn(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    config: Config,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    early_stopping: EarlyStopping,
    run_id: str,
):
    """
    Trains the PINN models using the provided DataLoader and configuration.

    The training process incorporates physics-based loss, boundary condition loss,
    data loss, and inlet condition loss, each weighted by self-adaptive parameters.

    Args:
        models (Dict[str, nn.Module]): Dictionary of PINN models.
        dataloader (DataLoader): DataLoader for training data.
        config (Config): Configuration object.
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler instance.
        early_stopping (EarlyStopping): EarlyStopping instance.
        logger (logging.Logger): Logger for logging information.
        run_id (str): Unique identifier for the experiment run.

    Returns:
        Dict[str, list]: History of losses recorded during training.
    """
    scaler = torch.cuda.amp.GradScaler() if config.device.startswith("cuda") else None
    loss_history = {"total": [], "physics": [], "boundary": [], "data": [], "inlet": []}

    epochs = config.epochs
    rho = config.rho
    mu = config.mu

    print("Starting training...")

    model_p = models["p"]
    model_u = models["u"]
    model_v = models["v"]
    model_w = models["w"]
    model_tau_x = models["tau_x"]
    model_tau_y = models["tau_y"]
    model_tau_z = models["tau_z"]

    for epoch in tqdm(range(1, epochs+1), desc="Training"):
        for m in models.values():
            m.train()

        epoch_loss_total = 0.0
        epoch_loss_physics = 0.0
        epoch_loss_boundary = 0.0
        epoch_loss_data = 0.0
        epoch_loss_inlet = 0.0

        for batch in dataloader:
            (x_batch, y_batch, z_batch, t_batch,
             p_true, u_true, v_true, w_true,
             tau_x_true, tau_y_true, tau_z_true,
             is_boundary) = batch

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

            if scaler:
                with torch.cuda.amp.autocast():
                    p_pred = model_p(x_batch, y_batch, z_batch, t_batch)
                    u_pred = model_u(x_batch, y_batch, z_batch, t_batch)
                    v_pred = model_v(x_batch, y_batch, z_batch, t_batch)
                    w_pred = model_w(x_batch, y_batch, z_batch, t_batch)
                    tau_x_pred = model_tau_x(x_batch, y_batch, z_batch, t_batch)
                    tau_y_pred = model_tau_y(x_batch, y_batch, z_batch, t_batch)
                    tau_z_pred = model_tau_z(x_batch, y_batch, z_batch, t_batch)

                    loss_physics = compute_physics_loss(
                        p_pred, u_pred, v_pred, w_pred,
                        x_batch, y_batch, z_batch, t_batch,
                        rho, mu
                    )

                    if is_boundary.sum() > 0:
                        u_bc_pred = u_pred[is_boundary]
                        v_bc_pred = v_pred[is_boundary]
                        w_bc_pred = w_pred[is_boundary]
                        loss_boundary = compute_boundary_loss(u_bc_pred, v_bc_pred, w_bc_pred)
                    else:
                        loss_boundary = torch.tensor(0.0, device=config.device)

                    loss_data_ = compute_data_loss(
                        p_pred, p_true,
                        u_pred, u_true,
                        v_pred, v_true,
                        w_pred, w_true,
                        tau_x_pred, tau_x_true,
                        tau_y_pred, tau_y_true,
                        tau_z_pred, tau_z_true
                    )

                    loss_inlet_ = compute_inlet_loss(u_pred, v_pred, w_pred, t_batch)

                    # Self-adaptive weights from pressure PINN
                    lambda_physics = torch.exp(model_p.log_lambda_physics)
                    lambda_boundary = torch.exp(model_p.log_lambda_boundary)
                    lambda_data = torch.exp(model_p.log_lambda_data)
                    lambda_inlet = torch.exp(model_p.log_lambda_inlet)

                    total_loss = (lambda_physics*loss_physics
                                + lambda_boundary*loss_boundary
                                + lambda_data*loss_data_
                                + lambda_inlet*loss_inlet_)

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(models["p"].parameters()) +
                                               list(models["u"].parameters()) +
                                               list(models["v"].parameters()) +
                                               list(models["w"].parameters()) +
                                               list(models["tau_x"].parameters()) +
                                               list(models["tau_y"].parameters()) +
                                               list(models["tau_z"].parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                p_pred = model_p(x_batch, y_batch, z_batch, t_batch)
                u_pred = model_u(x_batch, y_batch, z_batch, t_batch)
                v_pred = model_v(x_batch, y_batch, z_batch, t_batch)
                w_pred = model_w(x_batch, y_batch, z_batch, t_batch)
                tau_x_pred = model_tau_x(x_batch, y_batch, z_batch, t_batch)
                tau_y_pred = model_tau_y(x_batch, y_batch, z_batch, t_batch)
                tau_z_pred = model_tau_z(x_batch, y_batch, z_batch, t_batch)

                loss_physics = compute_physics_loss(
                    p_pred, u_pred, v_pred, w_pred,
                    x_batch, y_batch, z_batch, t_batch,
                    rho, mu
                )
                if is_boundary.sum() > 0:
                    u_bc_pred = u_pred[is_boundary]
                    v_bc_pred = v_pred[is_boundary]
                    w_bc_pred = w_pred[is_boundary]
                    loss_boundary = compute_boundary_loss(u_bc_pred, v_bc_pred, w_bc_pred)
                else:
                    loss_boundary = torch.tensor(0.0, device=config.device)

                loss_data_ = compute_data_loss(
                    p_pred, p_true,
                    u_pred, u_true,
                    v_pred, v_true,
                    w_pred, w_true,
                    tau_x_pred, tau_x_true,
                    tau_y_pred, tau_y_true,
                    tau_z_pred, tau_z_true
                )
                loss_inlet_ = compute_inlet_loss(u_pred, v_pred, w_pred, t_batch)

                lambda_physics = torch.exp(model_p.log_lambda_physics)
                lambda_boundary = torch.exp(model_p.log_lambda_boundary)
                lambda_data = torch.exp(model_p.log_lambda_data)
                lambda_inlet = torch.exp(model_p.log_lambda_inlet)

                total_loss = (lambda_physics*loss_physics
                            + lambda_boundary*loss_boundary
                            + lambda_data*loss_data_
                            + lambda_inlet*loss_inlet_)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(models["p"].parameters()) +
                                               list(models["u"].parameters()) +
                                               list(models["v"].parameters()) +
                                               list(models["w"].parameters()) +
                                               list(models["tau_x"].parameters()) +
                                               list(models["tau_y"].parameters()) +
                                               list(models["tau_z"].parameters()), 1.0)
                optimizer.step()

            epoch_loss_total += total_loss.item()
            epoch_loss_physics += loss_physics.item()
            epoch_loss_boundary += loss_boundary.item()
            epoch_loss_data += loss_data_.item()
            epoch_loss_inlet += loss_inlet_.item()

            del (x_batch, y_batch, z_batch, t_batch,
                 p_true, u_true, v_true, w_true,
                 tau_x_true, tau_y_true, tau_z_true, is_boundary,
                 p_pred, u_pred, v_pred, w_pred,
                 tau_x_pred, tau_y_pred, tau_z_pred,
                 loss_physics, loss_boundary, loss_data_, loss_inlet_, total_loss)
            torch.cuda.empty_cache()

        scheduler.step()

        n_batches = len(dataloader)
        avg_t = epoch_loss_total / n_batches
        avg_phy = epoch_loss_physics / n_batches
        avg_bdy = epoch_loss_boundary / n_batches
        avg_dat = epoch_loss_data / n_batches
        avg_inl = epoch_loss_inlet / n_batches

        loss_history["total"].append(avg_t)
        loss_history["physics"].append(avg_phy)
        loss_history["boundary"].append(avg_bdy)
        loss_history["data"].append(avg_dat)
        loss_history["inlet"].append(avg_inl)

        early_stopping(avg_t, models, optimizer, scheduler, run_id, config.model_dir)
        if early_stopping.early_stop:
            print(f"Early stop at epoch {epoch}")
            break

        if epoch % 50 == 1 or epoch == epochs:
            print(f"[Epoch {epoch}/{epochs}] Tot: {avg_t:.6f}, Phy: {avg_phy:.6f}, Bdy: {avg_bdy:.6f}, "
                  f"Data: {avg_dat:.6f}, Inl: {avg_inl:.6f}")
            print(f"Lambdas -> Phy: {lambda_physics.item():.4f}, Bdy: {lambda_boundary.item():.4f}, "
                  f"Data: {lambda_data.item():.4f}, Inl: {lambda_inlet.item():.4f}")

    print("Training finished.")
    return loss_history
