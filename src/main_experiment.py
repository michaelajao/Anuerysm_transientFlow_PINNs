# ============================================
# 1. Imports and Parameter Definitions
# ============================================

import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 2. Define Physical and Numerical Parameters
# ============================================

# Physical Parameters
params = {
    'nu': 1.0,          # Kinematic viscosity
    'g': 9.81,          # Acceleration due to gravity
    'beta': 0.1,        # Thermal expansion coefficient
    'beta_prime': 0.05, # Solutal expansion coefficient
    'sigma': 0.5,       # Electrical conductivity
    'rho': 1.0,         # Density
    'B': 1.0,           # Magnetic field strength
    'E': 0.1,           # External electric field
    'k': 0.6,           # Thermal conductivity
    'cp': 1.0,          # Specific heat at constant pressure
    'tau': 0.2,         # Brownian motion parameter
    'D_B': 0.1,         # Brownian diffusion coefficient
    'D_T': 0.1,         # Thermophoresis diffusion coefficient
    'sigma_star': 0.5,  # Radiation constant
    'k_star': 0.6,      # Radiation coefficient
    'k2': 0.1,          # Reaction rate constant
    'Ea': 1.0,          # Activation energy
    'k_val': 1.0,       # Boltzmann constant
    'Dm': 0.1,          # Microorganism diffusion coefficient
    'b_star': 0.05,     # Bioconvection parameter
    'We': 0.1,          # Weissenberg number
    'Cw': 1.0,          # Surface concentration
    'Tw': 1.0,          # Surface temperature
    'N_inf': 0.0,       # Microorganism density at infinity
    'T_inf': 0.0,       # Temperature at infinity
    'C_inf': 0.0,       # Concentration at infinity
    'bx': 1.0,          # Stretching parameter
    'hw': 1.0,          # Heat transfer coefficient at surface
    'h_c': 1.0,         # Convective heat transfer coefficient
    'p': 1.0,           # Reaction order
    'r': 1.0            # Parameter in momentum equation (for sin^2(pi y / r))
}

# Domain Parameters
Lx = 2.0  # Length in x-direction
Ly = 4.0  # Length in y-direction (approximates y -> infinity)

# ============================================
# 3. Geometry and Domain Setup
# ============================================

# Define the 2D rectangular geometry [x, y] in [0, Lx] x [0, Ly]
geom = dde.geometry.Rectangle([0.0, 0.0], [Lx, Ly])

# ============================================
# 4. Boundary Conditions
# ============================================

# Define boundary conditions at y = 0 and y = Ly

# Helper functions to identify boundaries
def boundary_y0(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

def boundary_yL(x, on_boundary):
    return on_boundary and np.isclose(x[1], Ly)

# Parameters for boundary conditions
U0 = lambda x: params['bx'] * x[:, 0:1]  # u = bx at y=0
Tw = params['Tw']                       # T = Tw at y=0
Cw = params['Cw']                       # C = Cw at y=0
N_inf = params['N_inf']                 # N = N_inf at y=Ly
T_inf = params['T_inf']                 # T = T_inf at y=Ly
C_inf = params['C_inf']                 # C = C_inf at y=Ly

# Boundary Conditions at y = 0
bc_u_y0 = dde.DirichletBC(
    geom, U0, boundary_y0, component=0
)

# For temperature, implement -k (dT/dy) = hw (Tw - T) at y=0
def boundary_T_y0(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

bc_T_y0 = dde.NeumannBC(
    geom, lambda x: params['hw'] * (params['Tw'] - x[:, 2:3]), boundary_T_y0, component=2
)

# For concentration, D_B (dC/dy) + (D_T / Tc) (dT/dy) = 0 at y=0
# Assuming Tc is some characteristic temperature, set Tc = 1 for simplicity
Tc = 1.0
def boundary_C_y0(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

bc_C_y0 = dde.NeumannBC(
    geom, lambda x: -params['D_T']/Tc * x[:, 2:3], boundary_C_y0, component=3
)

# For microorganism density at y=0, N = N_inf
bc_N_y0 = dde.DirichletBC(
    geom, lambda x: params['N_inf'], boundary_y0, component=4
)

# Boundary Conditions at y = Ly (far field)
# u -> 0, T -> T_inf, C -> C_inf, N -> 0
bc_u_yL = dde.DirichletBC(
    geom, lambda x: 0.0, boundary_yL, component=0
)
bc_T_yL = dde.DirichletBC(
    geom, lambda x: T_inf, boundary_yL, component=2
)
bc_C_yL = dde.DirichletBC(
    geom, lambda x: C_inf, boundary_yL, component=3
)
bc_N_yL = dde.DirichletBC(
    geom, lambda x: 0.0, boundary_yL, component=4
)

# Collect all boundary conditions
bcs = [bc_u_y0, bc_T_y0, bc_C_y0, bc_N_y0, bc_u_yL, bc_T_yL, bc_C_yL, bc_N_yL]

# ============================================
# 5. Define PDE Residuals
# ============================================

def pde(inputs, outputs):
    """
    Define the residuals of the PDEs:
    1. Continuity Equation
    2. Momentum Equation
    3. Energy Equation
    4. Concentration Equation
    5. Microorganism Equation
    """
    x = inputs[:, 0:1]
    y = inputs[:, 1:2]

    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    T = outputs[:, 2:3]
    C = outputs[:, 3:4]
    N = outputs[:, 4:5]

    # Compute necessary derivatives
    # Continuity: du/dx + dv/dy = 0
    du_dx = dde.grad.jacobian(u, inputs, i=0, j=0)
    dv_dy = dde.grad.jacobian(v, inputs, i=0, j=1)
    continuity = du_dx + dv_dy

    # Momentum Equation:
    # v * du/dy + u * du/dx = nu * d²u/dy² + g * beta * (T - T_inf) + g * beta_prime * (C - C_inf)
    # - (sigma / rho) * sin²(pi * y / r) * B² * u - E * B0
    du_dy = dde.grad.jacobian(u, inputs, i=0, j=1)
    d2u_dy2 = dde.grad.hessian(u, inputs, i=0, j=1, k=1)

    momentum = (v * du_dy) + (u * du_dx) \
               - (params['nu'] * d2u_dy2) \
               - (params['g'] * params['beta'] * (T - T_inf)) \
               - (params['g'] * params['beta_prime'] * (C - C_inf)) \
               + (params['sigma'] / params['rho']) * (tf.sin(np.pi * y / params['r']))**2 * params['B']**2 * u \
               + params['E'] * params['B0']

    # Energy Equation:
    # v * dT/dy + u * dT/dx = (k / (rho * cp)) * d²T/dy² + tau * D_B * dC/dy * dT/dy
    # + (16 * sigma_star / (3 * k_star)) * T³ * d²T/dy²
    dT_dx = dde.grad.jacobian(T, inputs, i=0, j=0)
    dT_dy = dde.grad.jacobian(T, inputs, i=0, j=1)
    d2T_dy2 = dde.grad.hessian(T, inputs, i=0, j=1, k=1)

    dC_dy = dde.grad.jacobian(C, inputs, i=0, j=1)

    energy = (v * dT_dy) + (u * dT_dx) \
             - (params['k'] / (params['rho'] * params['cp'])) * d2T_dy2 \
             - (params['tau'] * params['D_B'] * dC_dy * dT_dy) \
             - (16 * params['sigma_star'] / (3 * params['k_star'])) * T**3 * d2T_dy2

    # Concentration Equation:
    # v * dC/dy + u * dC/dx = (D_T / T_inf) * d²T/dy² + D_B * d²C/dy²
    # - k2 * (C - C_inf)**p * exp(-Ea / (k * T))
    dC_dx = dde.grad.jacobian(C, inputs, i=0, j=0)
    d2C_dy2 = dde.grad.hessian(C, inputs, i=0, j=1, k=1)
    concentration = (v * dC_dy) + (u * dC_dx) \
                    - (params['D_T'] / params['T_inf']) * d2T_dy2 \
                    - (params['D_B'] * d2C_dy2) \
                    + (params['k2'] * (C - C_inf)**params['p'] * tf.exp(-params['Ea'] / (params['k_val'] * T)))

    # Microorganism Equation:
    # Dm * (d²N/dx² + d²N/dy²) - b_star * We * (Cw - C_inf) * (N * dC/dx + N * dC/dy) = 0
    # Note: Assuming ∇(N ∇C) = N * d²C/dx² + ... which requires clarification
    # Here, simplifying to Dm * Laplacian(N) - b_star * We * (Cw - C_inf) * tf.gradients(N * dC_dx, x)

    dN_dx = dde.grad.jacobian(N, inputs, i=0, j=0)
    dN_dy = dde.grad.jacobian(N, inputs, i=0, j=1)
    d2N_dx2 = dde.grad.hessian(N, inputs, i=0, j=0, k=0)
    d2N_dy2 = dde.grad.hessian(N, inputs, i=0, j=1, k=1)
    laplacian_N = d2N_dx2 + d2N_dy2

    # Assuming ∇(N ∇C) = N * d²C/dx² + ∇N ⋅ ∇C
    # For simplicity, using N * d2C/dx2 + dN_dx * dC_dx + dN_dy * dC_dy
    d2C_dx2 = dde.grad.hessian(C, inputs, i=0, j=0, k=0)
    grad_N_dot_grad_C = (dN_dx * dde.grad.jacobian(C, inputs, i=0, j=0)) + \
                         (dN_dy * dde.grad.jacobian(C, inputs, i=0, j=1))
    microorganism = params['Dm'] * laplacian_N - params['b_star'] * params['We'] * (params['Cw'] - params['C_inf']) * (N * d2C_dx2 + grad_N_dot_grad_C)

    return [continuity, momentum, energy, concentration, microorganism]

# ============================================
# 6. Define the Neural Network Architecture
# ============================================

# Define the number of inputs and outputs
input_dim = 2   # (x, y)
output_dim = 5  # (u, v, T, C, N)

# Define the neural network
layer_size = [input_dim] + [50] * 4 + [output_dim]  # 4 hidden layers with 50 neurons each
activation = "tanh"
initializer = "Glorot normal"

net = dde.maps.FNN(layer_size, activation, initializer)

# ============================================
# 7. Create the Data Object
# ============================================

# Define the data with the PDE, geometry, and boundary conditions
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=10000,     # Number of collocation points inside the domain
    num_boundary=2000,    # Number of collocation points on the boundary
    solution=None,        # No analytical solution provided
    anchors=None,
    train_distribution="uniform"  # Uniform distribution of collocation points
)

# ============================================
# 8. Define and Compile the Model
# ============================================

# Instantiate the model
model = dde.Model(data, net)

# Compile the model with the Adam optimizer
model.compile("adam", lr=1e-3)

# ============================================
# 9. Train the Model
# ============================================

# Train the model using Adam optimizer
print("Training with Adam optimizer...")
losshistory, train_state = model.train(epochs=10000)

# Optionally, switch to L-BFGS optimizer for fine-tuning
print("Training with L-BFGS optimizer...")
model.compile("L-BFGS")
losshistory, train_state = model.train()

# ============================================
# 10. Evaluate and Visualize the Results
# ============================================

# Define a grid for prediction
nx, ny = 100, 100
x_space = np.linspace(0, Lx, nx)
y_space = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x_space, y_space)
XY = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

# Predict the solution
print("Predicting the solution...")
predictions = model.predict(XY)
u_pred = predictions[:, 0].reshape((ny, nx))
v_pred = predictions[:, 1].reshape((ny, nx))
T_pred = predictions[:, 2].reshape((ny, nx))
C_pred = predictions[:, 3].reshape((ny, nx))
N_pred = predictions[:, 4].reshape((ny, nx))

# Plotting Function
def plot_contour(X, Y, Z, title, xlabel='x', ylabel='y', cmap='jet'):
    plt.figure(figsize=(8,6))
    cp = plt.contourf(X, Y, Z, 50, cmap=cmap)
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Plot Velocity (u)
plot_contour(X, Y, u_pred, "Velocity Field (u)")

# Plot Velocity (v)
plot_contour(X, Y, v_pred, "Velocity Field (v)")

# Plot Temperature (T)
plot_contour(X, Y, T_pred, "Temperature Field (T)")

# Plot Concentration (C)
plot_contour(X, Y, C_pred, "Concentration Field (C)")

# Plot Microorganism Density (N)
plot_contour(X, Y, N_pred, "Microorganism Density (N)")

# Optionally, plot profiles at specific x or y
def plot_profile(x_val, predictions, variable, xlabel='y', ylabel='Value', title='Profile'):
    index = np.argmin(np.abs(x_space - x_val))
    plt.figure(figsize=(8,6))
    plt.plot(y_space, predictions[:, index])
    plt.title(f"{variable} Profile at x = {x_val}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Example: Plot u vs y at x = Lx/2
plot_profile(Lx/2, u_pred, "Velocity u", ylabel='u', title=f"Velocity u vs y at x = {Lx/2}")

# ============================================
# 11. Save and Load the Model (Optional)
# ============================================

# Save the trained model
# model.save("pinn_nanofluid_flow")

# To load the model later:
# model = dde.Model(data, net)
# model.compile("adam", lr=1e-3)
# model.restore("pinn_nanofluid_flow")

# ============================================
# 12. Summary and Key Points
# ============================================

print("Training and prediction completed successfully.")
