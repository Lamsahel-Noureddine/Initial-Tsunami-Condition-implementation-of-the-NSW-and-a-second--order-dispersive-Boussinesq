# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:41:39 2026

@author: Lenovo/Lamsahel
"""

import matplotlib.pyplot as plt
import numpy as np

from solver_general_nondimensional_system import *
from Bottom_profiles import *
from plots_fun import *
from Euler_analytic_solutions import *
from Relative_errors import *
from NSW_MacComarck import *







# ==========================================
# 1. Model Parameters & Scaling Factors
# ==========================================
A_typical = 1.0              # Typical amplitude scale (meters)
H_typical = 3000.0           # Typical water depth scale (meters)
Eps_nonL =1 # A_typical / H_typical  # Nonlinearity parameter (amplitude-to-depth ratio)
Lambda_0 = 1.0               # Characteristic wavelength scale (meters)
sigma_dispersive = 1.0       # Dispersive parameter (H_typical / Lambda_0)^2, set to 1 for now
print(f"\u03B5 = {Eps_nonL}")

# Reference wave speed  using g = 9.812 m/s²
C_0 = np.sqrt(9.812 * H_typical)

# Dimensional conversion factor: non-dimensional time -> seconds
time_scale = H_typical / C_0


# ==========================================
# 2. Computational Grid Setup
# ==========================================
L_x = 150.0                   # Non-dimensional domain length in x
N_x = 401                     # Number of grid points in x 
Dx = L_x / (N_x - 1)          # Grid spacing in x
x = np.linspace(0, L_x, N_x)

L_y = 150.0                   # Non-dimensional domain length in y
N_y = 401                     # Number of grid points in y
Dy = L_y / (N_y - 1)          # Grid spacing in y
y = np.linspace(0, L_y, N_y)


# ==========================================
# 3. Numerical Scheme Parameters
# ==========================================
# CFL_NSW = 0.1               # CFL number for adaptive time-stepping
nu = 1e-6                     # Artificial viscosity/regularization parameter
T_f = 25.0                    # Final non-dimensional simulation time

Dt=T_f/1000
print(f'dt={Dt}')
# Convert final time to physical seconds for reference
t_physical = time_scale * T_f
print(f"Physical time: {t_physical / 60.0:.5f} min")


# ==========================================
# 4. Initial Conditions & Bathymetry
# ==========================================
# Generate the seafloor deformation B(x,y) using an Okada fault model.
# The deformation is scaled to non-dimensional units.
deep_scale = H_typical / 1000.0   # Convert from meters to kilometers for the fault model
B = (1.0 / H_typical) * Okada_vecrtical_dip_slip_U_3_func(deep_scale * (x - L_x / 2) + 20.0,   deep_scale * (y - L_y / 2),   A_typical)
# B = (1.0 / H_typical) *   Okada_vecrtical_dip_slip_U_2_func(deep_scale * (x - L_x / 2) + 20.0,   deep_scale * (y - L_y / 2),   A_typical)

#Plot the bottom profile
plot_3d_B(L_x, L_y,  A_typical, H_typical)
# Initialize velocity fields to zero (initially at rest)
u_1_initial = np.zeros_like(B)
u_2_initial = np.zeros_like(B)


# ==========================================
# 5. Model Selection
# ==========================================
model = "Boussinesq"          # Select the Boussinesq model 


# ==========================================
# 5a. "Correct" Static Bottom Simulation
# ==========================================
# Static bottom case (type_motion = 0). The solver automatically computes the 
# hydrostatic initial condition eta_0 (by solving the elliptic balance) inside 
# general_solver_return, ensuring a consistent start.
t_eps_physical = 1.0                          # Physical rise time (unused for static)
t_eps = t_eps_physical / time_scale           # Non-dimensional rise time
type_motion = 0.0

eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat = general_solver_return(
    type_motion, model, t_eps, B, sigma_dispersive, u_1_initial, u_2_initial,
    L_x, Dx, Dy, L_y, H_typical, Eps_nonL, nu, T_f, Dt
)


# ==========================================
# 5b. "Incorrect" Static Bottom Simulation (Zero Initial Surface)
# ==========================================
# This simulation deliberately uses zeta_initial = B, 
eta_initial_inc = np.zeros_like(B)            # passive:  initial surface=B, no filter
sigma_general = 1.0                           

eta_tensor_type_inco, u_1_tensor_type_inco, u_2_tensor_type1_inco, t_array_inco = run_update_general(
    eta_initial_inc, u_1_initial, u_2_initial, B, L_x, L_y,
    sigma_general, Eps_nonL, nu, T_f, Dt, type_motion, t_eps
)


# ==========================================
# 6. Post-Processing: Total Free Surface Elevation
# ==========================================
# The total free surface is zeta = eta + B,
zeta_tensor_stat = B + eta_tensor_stat
zeta_tensor_inc = B + eta_tensor_type_inco


# ==========================================
# 7. 3D Plot of the Final State
# ==========================================
# Select the final time step for visualization.
indx_t = -1
t_physical_chosen = time_scale * t_array_stat[indx_t]
zeta_chosen = B + eta_tensor_stat[indx_t, :, :]

# Define observation points for time-series extraction
points = [
    (75, 75),   # P_1: Center of the domain
    (75, 90),   # P_2: Off-center in y
    (95, 85),   # P_3: Off-center in x and y
    (75, 60)    # P_4: Off-center in the negative y direction
]
point_labels = ['$p_1$', '$p_2$', '$p_3$', '$p_4$']

plot_3d(x, y, zeta_chosen, t_physical_chosen, H_typical, points, point_labels)


# ==========================================
# 8. Analytical Solutions for Validation
# ==========================================
# Compute the linear Euler  and linear Boussinesq  
# analytical solutions at the same time instances for error analysis.
t_array_phi = time_scale * t_array_stat       # Convert time array to physical seconds
zeta_analytic_list = []
zeta_bous_list = []

for i, t_ph in enumerate(t_array_phi):
    # Linear Euler solution (using Fourier/Green's function approach)
    # Note: gravity is scaled by 1/H_typical to maintain non-dimensional consistency
    zeta_analytic_t = compute_zeta_i(B, Dx, Dy, t_ph, 1, g=9.812 / H_typical)
    
    # Linear Boussinesq solution (dispersive correction)
    zeta_bous_t = compute_analytic_bous(B, Dx, Dy, t_array_stat[i])
    
    zeta_analytic_list.append(zeta_analytic_t)
    zeta_bous_list.append(zeta_bous_t)

# Convert lists to tensors for easy slicing
zeta_analytic_tensor = np.array(zeta_analytic_list)
zeta_bous_tensor = np.array(zeta_bous_list)


# ==========================================
# 9. Generate Validation Plots
# ==========================================
# Compare the "correct" static simulation, the "incorrect"  simulation,
# and the analytical solutions at specific observation points.

# 9a. Time-series of the passive vs  filter 
loc = points[0]
plot_passive_Filter_T1(x, y, t_array_stat, zeta_tensor_stat, zeta_tensor_inc, loc, H_typical)

# 9b. Full comparison: Numerical (filter & passive) vs. Analytical (Euler & Boussinesq)
loc = points[0]
plot_tensors_at_locations_T1(
    x, y, t_array_stat, zeta_tensor_stat, zeta_tensor_inc,
    zeta_analytic_tensor, zeta_bous_tensor, loc, H_typical
)

# # 9c. Relative error analysis (excluding the transient rise period t < t_eps)
# mask = (t_array_stat > t_eps) & (t_array_stat < T_f)
# plot_relative_erro_T1( x, y, t_array_stat[mask], zeta_tensor_stat[mask], zeta_tensor_inc[mask], loc)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
