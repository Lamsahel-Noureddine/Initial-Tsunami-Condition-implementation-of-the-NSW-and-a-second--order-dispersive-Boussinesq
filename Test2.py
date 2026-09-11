
"""
Created on Tue Sep  8 16:24:11 2026

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
# CFL_NSW = 0.05               # CFL number for adaptive time-stepping
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

# Initialize velocity fields to zero (initially at rest)
u_1_initial = np.zeros_like(B)
u_2_initial = np.zeros_like(B)


# ==========================================
# 5. Model Selection
# ==========================================
model = "Boussinesq"          # Select the Boussinesq model 



# --- Kinematic (Moving Bottom) Simulation ---
type_motion = 2.                # 0. static  2.Trigonometric,   3. Linear,       
t_eps_physical =1             # Physical rise time in seconds
t_eps = t_eps_physical / time_scale  # Non-dimensional rise time
print(f" nond rising time: {t_eps:.6f} ")



eta_tensor_eps, u_1_tensor_eps, u_2_tensor_eps, t_array_eps = general_solver_return(
    type_motion, model, t_eps, B, sigma_dispersive, u_1_initial, u_2_initial,
    L_x, Dx, Dy, L_y, H_typical, Eps_nonL, nu, T_f, Dt
)


# --- Static Bottom Simulation ---

# Correct bottom 
type_motion = 0.0
eta_tensor_stat, u_1_tensor_stat, u_2_tensor_stat, t_array_stat = general_solver_return(
    type_motion, model, t_eps, B, sigma_dispersive, u_1_initial, u_2_initial,
    L_x, Dx, Dy, L_y, H_typical, Eps_nonL, nu, T_f, Dt)



# Build full zeta(t) = f(t)*B + eta(t) for the kinematic case
zeta_tensor_eps = np.zeros_like(eta_tensor_eps)
for n, t in enumerate(t_array_eps):
    f_t = rising_function(t, t_eps, 2)
    zeta_tensor_eps[n, :, :] = f_t * B + eta_tensor_eps[n, :, :]

# For static case, f(t) = 1 for all t
zeta_tensor_stat = B + eta_tensor_stat  # Broadcasting adds B to every time slice




points = [
    (75, 75),   # P_1
    (75, 90),   # P_2
    (95, 85),   # P_3
    (75, 60)   # P_4
    ]

# loc=points[0]
# plot_tensors_at_locations_T2(x, y, t_array_stat, zeta_tensor_stat,zeta_tensor_eps, loc,H_typical,t_eps)






loc=points[0]
mask = ( t_array_stat > t_eps) & ( t_array_stat < T_f)
R=plot_relative_erro_T2(x, y, t_array_stat, zeta_tensor_stat,zeta_tensor_eps,mask, loc)
print(max(R))



t_eps_v =(1./ time_scale)*np.array([60, 30, 15, 7, 4, 2, 1])



max_errors_v = {
    'p_1': np.array([ 0.766351951297854, 0.4014142128229978,0.20328511547190153,0.09546420168936803, 0.05484648070121692,0.027907144049850847, 0.014779393232901686]),
    'p_2': np.array([ 0.7784212886438131,0.41284575229147463, 0.20939431975925996, 0.09789509035033617,0.05583159243500143,0.027931404191235944,0.014350853909174257]),
    'p_3': np.array([ 0.5804275569005843, 0.3061181645239312, 0.15507123749601218, 0.07252814380504435, 0.04140418741391679, 0.020758033673275125, 0.010701767643942702]),
    'p_4': np.array([ 0.6141052836571881, 0.32198634509828633,0.162924502721546, 0.07626868084375538, 0.043626813936707115,0.021971424318577917, 0.011425321219378103]),
}


plot_relative_error_4locations(t_eps_v, max_errors_v)


s=1
max_error,E_bous, norm_B = compute_Es_Bous(
zeta_tensor_eps, zeta_tensor_stat,
u_1_tensor_eps, u_2_tensor_eps,
u_1_tensor_stat, u_2_tensor_stat,
B, Dx, Dy, s,mask 
)

print(f"||B||_{{H^{s}}} = {norm_B:.6e}")
print(f"Max error in ({t_eps}, {T_f}): {max_error:.6e}")

t_eps_v= (1./ time_scale)*np.array([60, 30, 15, 7, 4, 2, 1])

max_E_bous_values=np.array([  7.844205e-01,  4.153862e-01,2.103664e-01,9.853952e-02,5.638453e-02,  2.828730e-02, 1.422433e-02])
plot_max_Es_Bous_vs_teps_log(t_eps_v,max_E_bous_values , s, T_f=None)



















