
"""
Created on Tue Sep  8 16:24:11 2026

@author: Lenovo/Lamsahel
"""


import matplotlib.pyplot as plt
import numpy as np
 
# from solver_general_nondimensional_system import *     #Mixed FE-FD
from General_solver_fully_FEM import *     #Fully FE

from Bottom_profiles import *
from plots_fun import *
from Euler_analytic_solutions import *
from Relative_errors import *
from NSW_MacComarck import *

import time

from NSW_MacComarck import *
from pathlib import Path
import gc







# ==========================================
# 1. Model Parameters & Scaling Factors
# ==========================================
A_typical = 1.0              # Typical amplitude scale (meters)
H_typical = 3000.0           # Typical water depth scale (meters)
Eps_nonL = 1   #A_typical / H_typical  # Nonlinearity parameter (amplitude-to-depth ratio)
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
N_x =801                 # Number of grid points in x 
Dx = L_x / (N_x - 1)          # Grid spacing in x
x = np.linspace(0, L_x, N_x)

L_y = 150.0                   # Non-dimensional domain length in y
N_y = 801         # Number of grid points in y
Dy = L_y / (N_y - 1)          # Grid spacing in y
y = np.linspace(0, L_y, N_y)
print(f'dx={Dx}')


# ==========================================
# 3. Numerical Scheme Parameters
# ==========================================
# CFL_NSW = 0.05               # CFL number for adaptive time-stepping
# nu = 1e-6                     # Artificial viscosity/regularization parameter
T_f = 20.0                    # Final non-dimensional simulation time

Dt=T_f/3000
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
B = (1.0 / H_typical) * Okada_vecrtical_dip_slip_U_3_func(deep_scale * (x - L_x / 2) + 10.0,   deep_scale * (y - L_y / 2),   A_typical)
# B = (1.0 / H_typical) *   Okada_vecrtical_dip_slip_U_2_func(deep_scale * (x - L_x / 2) + 20.0,   deep_scale * (y - L_y / 2),   A_typical)
plot_3d_B(L_x, L_y,  A_typical, H_typical)

Bmax_boundary = max(np.abs(B[0, :]).max(),   # first row
                    np.abs(B[-1, :]).max(),  # last row
                    np.abs(B[:, 0]).max(),   # first column
                    np.abs(B[:, -1]).max())  # last column

print(f"max |B| on boundary = {Bmax_boundary:.6e}")

# Initialize velocity fields to zero (initially at rest)
u_1_initial = np.zeros_like(B)
u_2_initial = np.zeros_like(B)

start = time.perf_counter()
# ==========================================
# 5. Model Selection
# ==========================================
model = "Shallow water"         # Select the Boussinesq model 


def cleanup(folder):
    gc.collect()
    for name in ("eta.npy", "u1.npy", "u2.npy", "t.npy",
                 "zeta_analytic.npy", "zeta_nsw.npy"):
        p = folder / name
        if p.exists():
            p.unlink()

# --- Kinematic (Moving Bottom) Simulation ---
type_motion = 2.                # 0. static  2.Trigonometric,   3. Linear,       
t_eps_physical =60            # Physical rise time in seconds
t_eps = t_eps_physical / time_scale  # Non-dimensional rise time
print(f" rising time: {t_eps:.6f} ")



output_dir_k   = Path("E:/tsunami_results/Kinematic_1051_3000")

print("Results will be saved in:", output_dir_k)

output_dir_k.mkdir(parents=True, exist_ok=True)
for n in ("eta.npy","u1.npy","u2.npy","t.npy"):   # wipe old files first
    p = output_dir_k / n
    if p.exists(): p.unlink()

eta_tensor_eps, u_1_tensor_eps, u_2_tensor_eps, t_array_eps = (
    general_solver_return(
        type_motion, model, t_eps, B, sigma_dispersive,
        u_1_initial, u_2_initial,
        L_x, Dx, Dy, L_y, H_typical,
        Eps_nonL, T_f, Dt,
        output_dir=output_dir_k
    )
)

# --- Static Bottom Simulation ---
output_dir   = Path("E:/tsunami_results/static_1051_3000")

print("Results will be saved in:", output_dir)

output_dir.mkdir(parents=True, exist_ok=True)
for n in ("eta.npy","u1.npy","u2.npy","t.npy"):   # wipe old files first
    p = output_dir / n
    if p.exists(): p.unlink()

# Correct bottom 
type_motion = 0.0
eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat = (
    general_solver_return(
        type_motion, model, t_eps, B, sigma_dispersive,
        u_1_initial, u_2_initial,
        L_x, Dx, Dy, L_y, H_typical,
        Eps_nonL, T_f, Dt,
        output_dir=output_dir
    )
)


  
# 8. Analytical Solutions — written to disk
# Release any memmap left open by a previous run in this kernel
t_array_phi = time_scale * t_array_stat
analysis_dir = Path("E:/tsunami_results/analysis_1051_3000")

analysis_dir.mkdir(parents=True, exist_ok=True)
for _n in ("zeta_analytic_tensor", "zeta_NSW_tensor"):
    if _n in globals():
        try: globals()[_n].flush()
        except Exception: pass
        del globals()[_n]
gc.collect()

n_t = len(t_array_phi)
history_shape = (n_t, N_x, N_y)

zeta_analytic_tensor = np.lib.format.open_memmap(
    analysis_dir / "zeta_analytic.npy", mode="w+", dtype=float, shape=history_shape)
zeta_NSW_tensor = np.lib.format.open_memmap(
    analysis_dir / "zeta_nsw.npy", mode="w+", dtype=float, shape=history_shape)

for i, t_ph in enumerate(t_array_phi):
    zeta_analytic_tensor[i] = compute_zeta_i(B, Dx, Dy, t_ph, 1, g=9.812 / H_typical)
    zeta_NSW_tensor[i]     = compute_analytic_NSW(B, Dx, Dy, t_array_stat[i])

zeta_analytic_tensor.flush()
zeta_NSW_tensor.flush()



points = [
    (L_x/2, L_x/2),   # P_1: Center of the domain
    (L_x/2,20+(L_x/2)),   # P_2: Off-center in y
    (15+(L_x/2),5+(L_x/2)),   # P_3: Off-center in x and y
    ((L_x/2), -15+(L_x/2))  # P_4: Off-center in the negative y direction
]




# 9b. Full comparison:

loc = points[3]
idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()
# 2. Extract time-series data at the location
v1 = B[idx_x, idx_y].copy() + eta_tensor_stat[:, idx_x, idx_y].copy()
v2=zeta_analytic_tensor[:, idx_x, idx_y].copy()
v3=zeta_NSW_tensor[:, idx_x, idx_y].copy()

plot_tensors_at_locations_T3_compa(x, y,t_array_stat ,v1 ,v2,v3, loc, H_typical)



# convergence like test 2
loc=points[3]
idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()

# 2. Extract time-series data at the location
v1 = B[idx_x, idx_y].copy() + eta_tensor_stat[:, idx_x, idx_y].copy()

v2= np.zeros_like(t_array_eps)
for n, t in enumerate(t_array_eps):
    f_t = rising_function(t, t_eps, 2)
    v2[n] = f_t * B[idx_x, idx_y] + eta_tensor_eps[n,idx_x, idx_y].copy()

plot_tensors_at_locations_T3(x, y, t_array_stat, v1,v2, loc,H_typical,t_eps)







# Local relative eroor computation

loc = points[1]
mask = (t_array_stat >= t_eps) & (t_array_stat <= T_f)

idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()

# Both simulations use the same time grid.
v1 = B[idx_x, idx_y] + eta_tensor_stat[mask, idx_x, idx_y].copy()
v2 = B[idx_x, idx_y] + eta_tensor_eps[mask, idx_x, idx_y].copy()

R = plot_relative_erro_T2(
    x, y, t_array_stat, v1, v2, mask, loc
)
print(f'Local error={np.max(R)}')



t_eps_v =(1./ time_scale)*np.array([60, 30, 15, 7, 4, 2, 1])



max_errors_v = {
    'p_1': np.array([ 0.9780281237264294, 0.5296686218700848, 0.2751721753899989,0.13089264920440802, 0.07506524349655966,0.03758261939785688, 0.018815484486892037]),
    'p_2': np.array([ 0.8927636113182024,0.5484695324357043, 0.29222403922146223, 0.13873210558188492, 0.0795309345579348 ,0.039812570394151287,0.019931442063922118]),
    'p_3': np.array([  0.5116865807233901, 0.288555150453614,0.14991277075903514, 0.07073485420435945, 0.04050704264684271 , 0.020270082726720484, 0.01014690192812242]),
    'p_4': np.array([0.6166863202836697, 0.3406462873755714, 0.17511466988068133, 0.083967190991592373, 0.05145738346593632 ,0.026861636416866494, 0.013697545367195807]),
}



plot_relative_error_4locations_SW(t_eps_v, max_errors_v)



# Prevent post-processing from modifying these arrays.
for a in (
    B,
    eta_tensor_eps, u_1_tensor_eps, u_2_tensor_eps,
    eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat,
    t_array_eps, t_array_stat,
):
    a.flags.writeable = False

del a



s = 1
mask = (t_array_stat >= t_eps) & (t_array_stat <= T_f)

# Use 2 for your kinematic trigonometric motion.
f_values = np.array([
    rising_function(t, t_eps, 2) for t in t_array_eps
], dtype=float)

max_error_sw, E_sw, norm_B = compute_Es_sw(
    eta_tensor_eps, eta_tensor_stat,
    u_1_tensor_eps, u_2_tensor_eps,
    u_1_tensor_stat, u_2_tensor_stat,
    B, Dx, Dy, s, mask, f_values
)


# print(f"||B||_{{H^{s}}} = {norm_B:.6e}")
print(f"Max error in ({t_eps}, {T_f}): {max_error_sw:.6e}")

t_eps_v= (1./ time_scale)*np.array([60, 30, 15, 7, 4, 2, 1])

max_E_bous_values=np.array([    1.088723e+00,  5.927339e-01,3.034861e-01,  1.425393e-01,8.155132e-02,  4.079463e-02, 2.040322e-02])
plot_max_Es_vs_teps_log_sw(t_eps_v,max_E_bous_values , 1, T_f=None)









# indx_t = -1
# t_physical_chosen = time_scale * t_array_stat[indx_t]
# zeta_chosen = B + eta_tensor_stat[indx_t, :, :]

# # Define observation points for time-series extraction

# points = [
#     (L_x/2, L_x/2),   # P_1: Center of the domain
#     (L_x/2,20+(L_x/2)),   # P_2: Off-center in y
#     (15+(L_x/2),5+(L_x/2)),   # P_3: Off-center in x and y
#     ((L_x/2), -15+(L_x/2))  # P_4: Off-center in the negative y direction
# ]



# point_labels = ['$p_1$', '$p_2$', '$p_3$', '$p_4$']

# plot_3d(x, y, zeta_chosen, t_physical_chosen, H_typical, points, point_labels)












# del eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat
# del eta_tensor_eps, u_1_tensor_eps, u_2_tensor_eps, t_array_eps
# del zeta_analytic_tensor, zeta_NSW_tensor


# del v1, v2,v3
# del max_error_sw, E_sw, norm_B,R

# gc.collect()

# cleanup(output_dir)
# cleanup(output_dir_k)
# cleanup(analysis_dir)

    
    


end = time.perf_counter()
print(f"Simulation took {end - start:.4f} seconds")
print(f"Simulation took {(end - start)/60:.4f} min")

    












