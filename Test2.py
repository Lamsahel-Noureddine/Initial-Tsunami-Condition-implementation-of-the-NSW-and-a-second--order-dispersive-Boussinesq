
"""
Created on Tue Sep  8 16:24:11 2026

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
Eps_nonL = 1# A_typical / H_typical  # Nonlinearity parameter (amplitude-to-depth ratio)
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
N_x = 801                # Number of grid points in x 
Dx = L_x / (N_x - 1)          # Grid spacing in x
x = np.linspace(0, L_x, N_x)

L_y = 150.0                   # Non-dimensional domain length in y
N_y = 801             # Number of grid points in y
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
#Plot the bottom profile
plot_3d_B(L_x, L_y,  A_typical, H_typical)
# Initialize velocity fields to zero (initially at rest)
u_1_initial = np.zeros_like(B)
u_2_initial = np.zeros_like(B)

start = time.perf_counter()

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

# --- Kinematic (Moving Bottom) Simulation ---
type_motion = 2.                # 0. static  2.Trigonometric,   3. Linear,       
t_eps_physical =1          # Physical rise time in seconds
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
# Correct bottom 
type_motion = 0.0
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


# Build full zeta(t) = f(t)*B + eta(t) for the kinematic case
# zeta_tensor_eps = np.zeros_like(eta_tensor_eps)
# for n, t in enumerate(t_array_eps):
#     f_t = rising_function(t, t_eps, 2)
#     zeta_tensor_eps[n, :, :] = f_t * B + eta_tensor_eps[n, :, :]

# For static case, f(t) = 1 for all t
# zeta_tensor_stat = B + eta_tensor_stat  # Broadcasting adds B to every time slice

points = [
    (L_x/2, L_x/2),   # P_1: Center of the domain
    (L_x/2,20+(L_x/2)),   # P_2: Off-center in y
    (15+(L_x/2),5+(L_x/2)),   # P_3: Off-center in x and y
    ((L_x/2), -15+(L_x/2))  # P_4: Off-center in the negative y direction
]


loc=points[0]
idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()

# 2. Extract time-series data at the location
v1 = B[idx_x, idx_y] + eta_tensor_stat[:, idx_x, idx_y]

v2= np.zeros_like(t_array_eps)
for n, t in enumerate(t_array_eps):
    f_t = rising_function(t, t_eps, 2)
    v2[n] = f_t * B[idx_x, idx_y] + eta_tensor_eps[n,idx_x, idx_y]

plot_tensors_at_locations_T2(x, y, t_array_stat, v1,v2, loc,H_typical,t_eps)



# Local relative eroor computation
loc = points[0]
mask = (t_array_stat >= t_eps) & (t_array_stat <= T_f)

idx_x = np.abs(x - loc[0]).argmin()
idx_y = np.abs(y - loc[1]).argmin()

# Both simulations use the same time grid.
v1 = B[idx_x, idx_y] + eta_tensor_stat[mask, idx_x, idx_y]
v2 = B[idx_x, idx_y] + eta_tensor_eps[mask, idx_x, idx_y]

R = plot_relative_erro_T2(
    x, y, t_array_stat, v1, v2, mask, loc
)
print(f'Local error={np.max(R)}')


t_eps_v =(1./ time_scale)*np.array([60, 30, 15, 7, 4, 2, 1])

max_errors_v = {
    'p_1': np.array([ 0.9784776090226989, 0.5291225363791202,0.2697552595247899,0.12652367442936227, 0.07236843011570099,0.03619787620403323, 0.018118009447542802]),
    'p_2': np.array([ 0.8027757831299258,0.42503367166323314,0.21568912724895709, 0.10105033072043834,0.05778652825619007,0.028902067856202234,0.014466030475539652]),
    'p_3': np.array([ 0.7469593570644112,0.3936790731644088, 0.1994994520544128, 0.09342802344239222,0.05342344351678702, 0.02671909621581597, 0.013373305374840529]),
    'p_4': np.array([ 0.5144221879265181,0.26927067561458917,0.1362509061830927,0.06378518314515269, 0.03647087405853421,0.018240030544275712, 0.00912933800604513]),
}


plot_relative_error_4locations(t_eps_v, max_errors_v)





# Relative error computation to validate the theorem 
s = 1
# Use 2 for your kinematic trigonometric motion.
f_values = np.array([
    rising_function(t, t_eps, 2) for t in t_array_eps
], dtype=float)

max_error, E_bous, norm_B = compute_Es_Bous(
    eta_tensor_eps, eta_tensor_stat,
    u_1_tensor_eps, u_2_tensor_eps,
    u_1_tensor_stat, u_2_tensor_stat,
    B, Dx, Dy, s, mask, f_values
)


# print(f"||B||_{{H^{s}}} = {norm_B:.6e}")
print(f"Max error in ({t_eps}, {T_f}): {max_error:.6e}")


t_eps_v= (1./ time_scale)*np.array([60, 30, 15, 7, 4, 2, 1])

max_E_bous_values=np.array([  8.642396e-01,  4.623817e-01,2.347278e-01,  1.099785e-01,6.289287e-02,   3.145602e-02, 1.573025e-02])
plot_max_Es_Bous_vs_teps_log(t_eps_v,max_E_bous_values , s, T_f=None)








end = time.perf_counter()
print(f"Simulation took {end - start:.4f} seconds")
print(f"Simulation took {(end - start)/60:.4f} min")



# del eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat
# del eta_tensor_eps, u_1_tensor_eps, u_2_tensor_eps, t_array_eps

# del v1, v2

# gc.collect()

# cleanup(output_dir)
# cleanup(output_dir_k)

    
    
    










