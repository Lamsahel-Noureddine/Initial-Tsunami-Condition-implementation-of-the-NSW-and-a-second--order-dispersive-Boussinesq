# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:41:39 2026

@author: Lenovo/Lamsahel
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# from solver_general_nondimensional_system import *     #Mixed FE-FD
from General_solver_fully_FEM import *     #Fully FE
from Bottom_profiles import *
from plots_fun import *
from Euler_analytic_solutions import *
from Relative_errors import *
from NSW_MacComarck import *
from pathlib import Path
import gc







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
N_x = 801                   # Number of grid points in x 
Dx = L_x / (N_x - 1)          # Grid spacing in x
print(f'dx={Dx}')

x = np.linspace(0, L_x, N_x)

L_y = 150.0                   # Non-dimensional domain length in y
N_y = 801                  # Number of grid points in y
Dy = L_y / (N_y - 1)          # Grid spacing in y
y = np.linspace(0, L_y, N_y)


# ==========================================
# 3. Numerical Scheme Parameters
# ==========================================
# CFL_NSW = 0.1               # CFL number for adaptive time-stepping
# nu = 0.                     # Artificial viscosity/regularization parameter
T_f =20                  # Final non-dimensional simulation time

Dt=T_f/3000
print(f'dt={Dt}')
# Convert final time to physical seconds for reference
t_physical = time_scale * T_f
print(f"Physical time: {t_physical / 60.0:.5f} min")

start = time.perf_counter()

# ==========================================
# 4. Initial Conditions & Bathymetry
# ==========================================
# Generate the seafloor deformation B(x,y) using an Okada fault model.
# The deformation is scaled to non-dimensional units.
deep_scale = H_typical / 1000.0   # Convert from meters to kilometers for the fault model
B = (1.0 / H_typical) * Okada_vecrtical_dip_slip_U_3_func(deep_scale * (x - L_x / 2) + 10.0,   deep_scale * (y - L_y / 2),   A_typical)
# B = (1.0 / H_typical) *   Okada_vecrtical_dip_slip_U_2_func(deep_scale * (x - L_x / 2) + 20.0,   deep_scale * (y - L_y / 2),   A_typical)
# B=(1.0 / H_typical) *supe_Gaussian_bottom(x,y)

# compute the wave number of the bottom 
R = np.sqrt(6)  # YOUR CHOICE of spectral cutoff
r_B, F_R = bottom_wavenumber(B, Dx, Dy, R)
print("Dominant nonzero wavenumber:", r_B)
print(f"Energy in r <= R: {100 * F_R:.4f}%")
print(f"Energy in r > R:  {100 * (1 - F_R):.4f}%")

#Plot the bottom profile
plot_3d_B(L_x, L_y,  A_typical, H_typical)
# Initialize velocity fields to zero (initially at rest)
u_1_initial = np.zeros_like(B)
u_2_initial = np.zeros_like(B)


# ==========================================
# 5. Model Selection
# ==========================================
model = "Boussinesq"          # Select the Boussinesq model 

def cleanup(folder):
    gc.collect()
    for name in ("eta.npy", "u1.npy", "u2.npy", "t.npy",
                 "zeta_analytic.npy", "zeta_bous.npy"):
        p = folder / name
        if p.exists():
            p.unlink()
# ==========================================
# 5a. "Correct" Static Bottom Simulation
# ==========================================
# Static bottom case (type_motion = 0). The solver automatically computes the 
# hydrostatic initial condition eta_0 (by solving the elliptic balance) inside 
# general_solver_return, ensuring a consistent start.
t_eps_physical = 1.0                          # Physical rise time (unused for static)
t_eps = t_eps_physical / time_scale           # Non-dimensional rise time
type_motion = 0.0

# output_dir = Path.home() / "tsunami_results" / "static_1051_3000"
output_dir   = Path("E:/tsunami_results/static_1051_3000")

print("Results will be saved in:", output_dir)

output_dir.mkdir(parents=True, exist_ok=True)
for n in ("eta.npy","u1.npy","u2.npy","t.npy"):   # wipe old files first
    p = output_dir / n
    if p.exists(): p.unlink()


eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat = (
    general_solver_return(
        type_motion, model, t_eps, B, sigma_dispersive,
        u_1_initial, u_2_initial,
        L_x, Dx, Dy, L_y, H_typical,
        Eps_nonL, T_f, Dt,
        output_dir=output_dir
    )
)

# ==========================================
# 5b. "Incorrect" Static Bottom Simulation (Zero Initial Surface)
# ==========================================
# This simulation deliberately uses zeta_initial = B, 
eta_initial_inc = np.zeros_like(B)            # passive:  initial surface=B, no filter
sigma_general = 1.0                           

# output_dir_c = Path.home() / "tsunami_results" / "static_inco_1051_3000"
output_dir_c = Path("E:/tsunami_results/static_inco_1051_3000")
output_dir_c.mkdir(parents=True, exist_ok=True)
for n in ("eta.npy","u1.npy","u2.npy","t.npy"):
    p = output_dir_c / n
    if p.exists(): p.unlink()
            


print("Results will be saved in:", output_dir_c)

eta_tensor_type_inco, u_1_tensor_type_inco, u_2_tensor_type1_inco, t_array_inco = run_update_general(
    eta_initial_inc, u_1_initial, u_2_initial, B, L_x, L_y,
    sigma_general, Eps_nonL, T_f, Dt, type_motion, t_eps, output_dir=output_dir_c
)


# ==========================================
# 6. Post-Processing: Total Free Surface Elevation
# ==========================================
# The total free surface is zeta = eta + B,
# zeta_tensor_stat = B + eta_tensor_stat
# zeta_tensor_inc = B + eta_tensor_type_inco


# ==========================================
# 7. 3D Plot of the Final State
# ==========================================
# Select the final time step for visualization.
indx_t = -1
t_physical_chosen = time_scale * t_array_stat[indx_t]
zeta_chosen = B + eta_tensor_stat[indx_t, :, :]

# Define observation points for time-series extraction
points = [
    (L_x/2, L_x/2),   # P_1: Center of the domain
    (L_x/2,20+(L_x/2)),   # P_2: Off-center in y
    (15+(L_x/2),5+(L_x/2)),   # P_3: Off-center in x and y
    ((L_x/2), -15+(L_x/2))  # P_4: Off-center in the negative y direction
]
point_labels = ['$p_1$', '$p_2$', '$p_3$', '$p_4$']

plot_3d(x, y, zeta_chosen, t_physical_chosen, H_typical, points, point_labels)

# ==========================================
# 8. Analytical Solutions for Validation
# ==========================================
# Compute the linear Euler  and linear Boussinesq  
# analytical solutions at the same time instances for error analysis.
# ==========================================
# 8. Analytical Solutions — written to disk
# ==========================================
t_array_phi = time_scale * t_array_stat

# analysis_dir = Path.home() / "tsunami_results" / "analysis_1051_3000"
analysis_dir = Path("E:/tsunami_results/analysis_1051_3000")

analysis_dir.mkdir(parents=True, exist_ok=True)

# Release any memmap left open by a previous run in this kernel
for _n in ("zeta_analytic_tensor", "zeta_bous_tensor"):
    if _n in globals():
        try: globals()[_n].flush()
        except Exception: pass
        del globals()[_n]
gc.collect()

n_t = len(t_array_phi)
history_shape = (n_t, N_x, N_y)

zeta_analytic_tensor = np.lib.format.open_memmap(
    analysis_dir / "zeta_analytic.npy", mode="w+", dtype=float, shape=history_shape)
zeta_bous_tensor = np.lib.format.open_memmap(
    analysis_dir / "zeta_bous.npy", mode="w+", dtype=float, shape=history_shape)

for i, t_ph in enumerate(t_array_phi):
    zeta_analytic_tensor[i] = compute_zeta_i(B, Dx, Dy, t_ph, 1, g=9.812 / H_typical)
    zeta_bous_tensor[i]     = compute_analytic_bous(B, Dx, Dy, t_array_stat[i])

zeta_analytic_tensor.flush()
zeta_bous_tensor.flush()

# ==========================================
# 9. Generate Validation Plots
# ==========================================
# Compare the "correct" static simulation, the "incorrect"  simulation,
# and the analytical solutions at specific observation points.

# 9a. Time-series of the passive vs  filter 
loc = points[0]
idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()

# 2. Extract time-series data at the location
v1 =  B[idx_x, idx_y] + eta_tensor_stat[:, idx_x, idx_y]
v2 =  B[idx_x, idx_y] + eta_tensor_type_inco[:, idx_x, idx_y]
plot_passive_Filter_T1(x, y, t_array_stat, v1, v2, loc, H_typical)

# 9b. Full comparison: Numerical (filter & passive) vs. Analytical (Euler & Boussinesq)
loc = points[0]
idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()
# 2. Extract time-series data at the location
v1 = B[idx_x, idx_y] + eta_tensor_stat[:, idx_x, idx_y]
v2 =  B[idx_x, idx_y] + eta_tensor_type_inco[:, idx_x, idx_y]
v3=zeta_analytic_tensor[:, idx_x, idx_y].copy()
v4=zeta_bous_tensor[:, idx_x, idx_y].copy()
plot_tensors_at_locations_T1(
    x, y, t_array_stat, v1, v2,
    v3, v4, loc, H_typical
)

# 9c. Relative error analysis (excluding the transient rise period t < t_eps)
loc = points[0]
# mask = (t_array_stat > t_eps) & (t_array_stat < T_f)
idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()

# 2. Extract time-series data at the location
v1 =  B[idx_x, idx_y] + eta_tensor_stat[:, idx_x, idx_y]
v2 =  B[idx_x, idx_y] + eta_tensor_type_inco[:, idx_x, idx_y]
plot_relative_erro_T1( x, y, t_array_stat, v1, v2, loc)
    

    
 # Plot slices

x_mid_idx, y_mid_idx = N_x // 2, N_y // 2
V1= B[:, y_mid_idx] + eta_tensor_stat[-1,:, y_mid_idx]
V2=zeta_bous_tensor[-1,:, y_mid_idx]
V3=zeta_analytic_tensor[-1,:, y_mid_idx]

V4=B[x_mid_idx, :]+eta_tensor_stat[-1,x_mid_idx, :]
V5=zeta_bous_tensor[-1,x_mid_idx, :]
V6=zeta_analytic_tensor[-1,x_mid_idx, :]
plot_static_Linear_Euler(x,y,x_mid_idx, y_mid_idx,V1,V2,V3,V4,V5,V6,H_typical, model)



# Plot impact of nonlinearity term
max_error = plot_relative_error_linear_nonlinear_T1(
    t_array_stat,
    eta_tensor_stat,
    zeta_bous_tensor,
    B
)

print(f"Maximum relative error over time: {max_error:.6e}")







end = time.perf_counter()
print(f"Simulation took {end - start:.4f} seconds")
print(f"Simulation took {(end - start)/60:.4f} min")



# del eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat
# del eta_tensor_type_inco, u_1_tensor_type_inco, u_2_tensor_type1_inco, t_array_inco
# del  zeta_chosen
# del zeta_analytic_tensor, zeta_bous_tensor
# del v3, v4

# gc.collect()

# cleanup(output_dir)
# cleanup(output_dir_c)
# cleanup(analysis_dir)

    
    
    
    
    
