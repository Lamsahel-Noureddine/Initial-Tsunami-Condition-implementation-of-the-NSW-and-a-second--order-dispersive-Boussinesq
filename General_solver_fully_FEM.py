#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 13:07:08 2026

@author: lamsahel


In this solver we use a fully finite element (FE) discretization. 
Typically, the right-hand sides are approximated using the Gauss quadrature rule on each rectangular element,
and the spatial derivatives are approximated by differentiating the basis functions. Note that for the linear terms one may also use the advection matrix;
 however, for nonlinear terms one still needs to compute these approximations, or one may use a precomputed tensor of three-index basis-function products.
"""

import numpy as np
from pathlib import Path
from scipy.sparse import diags, kron
from scipy.fft import dstn, idstn
from Tsunami_initial_conditions import *


def static_model(t):
    return 1


def Instantaneous_motion(t):
    if t <= 0:
        F_inst = 0
    else:
        F_inst = 1
    return F_inst


def Trigonometric_motion(t, t_eps):
    if t <= 0:
        F_Tri = 0.0
    elif 0 < t <= t_eps:
        F_Tri = 0.5 * (1 - np.cos(np.pi * t / t_eps))
    else:
        F_Tri = 1
    return F_Tri


def Linear_motion(t, t_eps):
    if t <= 0:
        F_L = 0.0
    elif 0 < t <= t_eps:
        F_L = t / t_eps
    else:
        F_L = 1
    return F_L


def rising_function(t, t_eps, type_motion):
    # 0: static; 1: instantaneous; 2: trigonometric; 3: linear.
    if type_motion == 0.0:
        return static_model(t)
    elif type_motion == 1.0:
        return Instantaneous_motion(t)
    elif type_motion == 2.0:
        return Trigonometric_motion(t, t_eps)
    elif type_motion == 3.0:
        return Linear_motion(t, t_eps)
    else:
        raise ValueError("type_motion must be 0, 1, 2, or 3.")


def apply_boundary_conditions(eta, u_1, u_2):
    # Homogeneous Dirichlet conditions for all three fields on every wall.
    eta[0, :] = 0.0
    eta[-1, :] = 0.0
    eta[:, 0] = 0.0
    eta[:, -1] = 0.0
    u_1[0, :] = 0.0
    u_1[-1, :] = 0.0
    u_1[:, 0] = 0.0
    u_1[:, -1] = 0.0
    u_2[0, :] = 0.0
    u_2[-1, :] = 0.0
    u_2[:, 0] = 0.0
    u_2[:, -1] = 0.0
    return eta, u_1, u_2


class _DirichletTensorSolver:
    """DST-I solver for (M + alpha*K) X = RHS 
    """

    def __init__(self, N_x, N_y, Dx, Dy, alpha):
        self.shape = (N_x, N_y)
        theta_x = np.pi * np.arange(1, N_x - 1) / (N_x - 1)
        theta_y = np.pi * np.arange(1, N_y - 1) / (N_y - 1)
        mass_x = (Dx / 3.0) * (2.0 + np.cos(theta_x))
        mass_y = (Dy / 3.0) * (2.0 + np.cos(theta_y))
        stiff_x = (4.0 / Dx) * np.sin(theta_x / 2.0)**2
        stiff_y = (4.0 / Dy) * np.sin(theta_y / 2.0)**2
        
        # Here we form the Sylvester matrix M+ alpha K.
        self._eigenvalues = (
            mass_x[:, None] * mass_y[None, :]
            + alpha * (stiff_x[:, None] * mass_y[None, :]
                       + mass_x[:, None] * stiff_y[None, :])
        )

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != np.prod(self.shape):
            raise ValueError("right-hand side must be N_x x N_y")
        extra_shape = rhs.shape[1:]
        result = rhs.copy().reshape(self.shape + extra_shape)
        interior = result[1:-1, 1:-1, ...]
        axes = tuple(axis for axis in (0, 1) if interior.shape[axis] > 1)
        if axes:
            coefficients = dstn(interior, type=1, axes=axes, norm="ortho")
        else:
            coefficients = interior.copy()
        denominator = self._eigenvalues.reshape(
            self._eigenvalues.shape + (1,) * len(extra_shape)
        )
        coefficients /= denominator
        if axes:
            result[1:-1, 1:-1, ...] = idstn(
                coefficients, type=1, axes=axes, norm="ortho"
            )
        else:
            result[1:-1, 1:-1, ...] = coefficients
        return result.reshape(rhs.shape)


def assemble_fem_matrices(N_x, N_y, L_x, L_y, sigma):
    if min(N_x, N_y) < 3:
        raise ValueError("At least 3 points are required in each direction.")
    Dx = L_x / (N_x - 1)
    Dy = L_y / (N_y - 1)

    # P1 mass and stiffness matrices, including the one-element endpoint weights.
    M_x = diags([1, 4, 1], [-1, 0, 1], shape=(N_x, N_x), format='csr', dtype=float)
    M_x[0, 0] = 2
    M_x[-1, -1] = 2
    M_x = (Dx / 6.0) * M_x

    M_y = diags([1, 4, 1], [-1, 0, 1], shape=(N_y, N_y), format='csr', dtype=float)
    M_y[0, 0] = 2
    M_y[-1, -1] = 2
    M_y = (Dy / 6.0) * M_y

    K_x = diags([-1, 2, -1], [-1, 0, 1], shape=(N_x, N_x), format='csr', dtype=float)
    K_x[0, 0] = 1
    K_x[-1, -1] = 1
    K_x = (1.0 / Dx) * K_x

    K_y = diags([-1, 2, -1], [-1, 0, 1], shape=(N_y, N_y), format='csr', dtype=float)
    K_y[0, 0] = 1
    K_y[-1, -1] = 1
    K_y = (1.0 / Dy) * K_y

    M_tensor = kron(M_x, M_y, format='csr')
    K_tensor = kron(M_x, K_y, format='csr') + kron(K_x, M_y, format='csr')
  
    mass_solver = _DirichletTensorSolver(N_x, N_y, Dx, Dy, 0.0)
    lu_u = (mass_solver, mass_solver)
    lu_eta = (mass_solver if sigma == 0.0 else
              _DirichletTensorSolver(N_x, N_y, Dx, Dy, sigma / 3.0))
    return M_tensor, K_tensor, lu_eta, lu_u


def compute_right_hand_sides(eta, u_1, u_2, B, Dx, Dy, eps, t, t_eps, type_motion):
    """
   # Local corner order: (00, 10, 01, 11).
    """
    zeta = eta + rising_function(t, t_eps, type_motion) * B
    # Preserve depth
    H = np.maximum(1.0 + eps * eta, 1e-6)

    R_1 = np.zeros_like(eta, dtype=float)
    R_2 = np.zeros_like(eta, dtype=float)
    R_3 = np.zeros_like(eta, dtype=float)

    corners = (
        (slice(None, -1), slice(None, -1)),  # 00
        (slice(1, None),  slice(None, -1)),  # 10
        (slice(None, -1), slice(1, None)),   # 01
        (slice(1, None),  slice(1, None)),   # 11
    )

    zeta_nodes = tuple(zeta[c] for c in corners)
    H_nodes = tuple(H[c] for c in corners)
    u1_nodes = tuple(u_1[c] for c in corners)
    u2_nodes = tuple(u_2[c] for c in corners)

    def evaluate(nodes, weights):
        return (weights[0] * nodes[0] + weights[1] * nodes[1]
                + weights[2] * nodes[2] + weights[3] * nodes[3])

    gauss_points = (0.5 - 0.5 / np.sqrt(3.0),
                    0.5 + 0.5 / np.sqrt(3.0))
    weight = Dx * Dy / 4.0

    for xi in gauss_points:
        for theta in gauss_points:
            N = ((1.0 - xi) * (1.0 - theta), xi * (1.0 - theta),
                 (1.0 - xi) * theta, xi * theta)
            dNdx = (-(1.0 - theta) / Dx, (1.0 - theta) / Dx,
                    -theta / Dx, theta / Dx)
            dNdy = (-(1.0 - xi) / Dy, -xi / Dy,
                    (1.0 - xi) / Dy, xi / Dy)

            zeta_q = evaluate(zeta_nodes, N)
            H_q = evaluate(H_nodes, N)
            u1_q = evaluate(u1_nodes, N)
            u2_q = evaluate(u2_nodes, N)

            # Differentiate the FE polynomials
            if eps != 0.0:
                adv1 = eps * (u1_q * evaluate(u1_nodes, dNdx)
                              + u2_q * evaluate(u1_nodes, dNdy))
                adv2 = eps * (u1_q * evaluate(u2_nodes, dNdx)
                              + u2_q * evaluate(u2_nodes, dNdy))
            else:
                adv1 = adv2 = 0.0

            flux_x = H_q * u1_q
            flux_y = H_q * u2_q
            for a, corner in enumerate(corners):
                R_1[corner] += weight * (dNdx[a] * flux_x + dNdy[a] * flux_y)
                R_2[corner] += weight * (N[a] * adv1 - dNdx[a] * zeta_q)
                R_3[corner] += weight * (N[a] * adv2 - dNdy[a] * zeta_q)

    R_1, R_2, R_3 = apply_boundary_conditions(R_1, R_2, R_3)
    return R_1, R_2, R_3


# Original CFL helper remains inactive: dt is supplied to the driver.
# def compute_Dt(u_1, u_2, eta, Dx, Dy, CFL, Eps_nonL):
#     H = np.maximum(1 + Eps_nonL * eta, 1e-8)
#     C_wave = np.sqrt(H)
#     Speed_x = np.max(C_wave + Eps_nonL * np.abs(u_1))
#     Speed_y = np.max(C_wave + Eps_nonL * np.abs(u_2))
#     # Dt = CFL * np.minimum(Dx / Speed_x, Dy / Speed_y)
#     Dt = CFL * Dx
#     return Dt


def step_euler(eta, u_1, u_2, B, Dx, Dy, dt, t, eps,
               M_matrix, K_matrix, lu_eta, lu_u, sigma, type_motion, t_eps):
    Nx, Ny = eta.shape
    R_1, R_2, R_3 = compute_right_hand_sides(
        eta, u_1, u_2, B, Dx, Dy, eps, t, t_eps, type_motion
    )

    f_diff = rising_function(t + dt, t_eps, type_motion) - rising_function(t, t_eps, type_motion)
    source_eta = -(sigma / 2.0) * f_diff * K_matrix.dot(B.ravel())
    rhs_Q = dt * R_1.ravel() + source_eta
    
    rhs_Q.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_Q.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    
    P0 = lu_eta.solve(rhs_Q).reshape(Nx, Ny)
    eta_next = eta + P0

    rhs_P1 = -dt * R_2.ravel()
    rhs_P2 = -dt * R_3.ravel()
    
    rhs_P1.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_P1.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    rhs_P2.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_P2.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    
    Q1_0 = lu_u[0].solve(rhs_P1).reshape(Nx, Ny)
    Q2_0 = lu_u[1].solve(rhs_P2).reshape(Nx, Ny)
    u1_next = u_1 + Q1_0
    u2_next = u_2 + Q2_0

    eta_next, u1_next, u2_next = apply_boundary_conditions(eta_next, u1_next, u2_next)
    return eta_next, u1_next, u2_next


def boussinesq_step_leapfrog(eta_prev, u1_prev, u2_prev,
                             eta_curr, u1_curr, u2_curr,
                             B, Dx, Dy, dt, t_curr, t_prev, eps,
                             M_matrix, K_matrix, lu_eta, lu_u,
                             sigma, type_motion, t_eps):
    
    time_weight = 2.0 * dt

    Nx, Ny = eta_curr.shape
    R_1, R_2, R_3 = compute_right_hand_sides(
        eta_curr, u1_curr, u2_curr, B, Dx, Dy, eps, t_curr, t_eps, type_motion
    )

    f_diff = (
        rising_function(t_curr + dt, t_eps, type_motion)
        - rising_function(t_prev, t_eps, type_motion)
    )
    source_eta = -(sigma / 2.0) * f_diff * K_matrix.dot(B.ravel())
    rhs_Q = time_weight * R_1.ravel() + source_eta
    
    rhs_Q.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_Q.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    
    Pn = lu_eta.solve(rhs_Q).reshape(Nx, Ny)
    eta_next = eta_prev + Pn

    rhs_Q1 = -time_weight * R_2.ravel()
    rhs_Q2 = -time_weight * R_3.ravel()
    
    rhs_Q1.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_Q1.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    rhs_Q2.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_Q2.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    
    Q1_n = lu_u[0].solve(rhs_Q1).reshape(Nx, Ny)
    Q2_n = lu_u[1].solve(rhs_Q2).reshape(Nx, Ny)
    
    u1_next = u1_prev + Q1_n
    u2_next = u2_prev + Q2_n

    eta_next, u1_next, u2_next = apply_boundary_conditions(eta_next, u1_next, u2_next)
    return eta_next, u1_next, u2_next


def run_update_general(eta_init, u1_init, u2_init, B, Lx, Ly,
                       sigma, eps, T_f, dt, type_motion, t_eps,
                       output_dir=None):
    
    """Run with prescribed constant dt; all fields have shape (Nx, Ny).

    Set output_dir to save every time level in disk-backed .npy arrays.
  
    """
    

    n_steps = int(round(T_f / dt))
    if not np.isclose(n_steps * dt, T_f, rtol=1e-12, atol=0.0):
        raise ValueError("T_f/dt must be an integer; for example, choose Dt = T_f/200.")

    eta = eta_init.copy()
    u_1 = u1_init.copy()
    u_2 = u2_init.copy()
    eta, u_1, u_2 = apply_boundary_conditions(eta, u_1, u_2)
    Nx, Ny = eta.shape
    if min(Nx, Ny) < 3:
        raise ValueError("At least three grid points are required in each direction.")
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    M_matrix, K_matrix, lu_eta, lu_u = assemble_fem_matrices(Nx, Ny, Lx, Ly, sigma)

    # Allocate the complete histories once
    history_shape = (n_steps + 1, Nx, Ny)
    if output_dir is None:
        eta_tensor = np.empty(history_shape, dtype=float)
        u1_tensor = np.empty(history_shape, dtype=float)
        u2_tensor = np.empty(history_shape, dtype=float)
        t_array = np.empty(n_steps + 1, dtype=float)
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filenames = ("eta.npy", "u1.npy", "u2.npy", "t.npy")
        if any((output_path / name).exists() for name in filenames):
            raise FileExistsError("Output files already exist; choose a new output_dir.")
        eta_tensor = np.lib.format.open_memmap(
            output_path / "eta.npy", mode="w+", dtype=float, shape=history_shape)
        u1_tensor = np.lib.format.open_memmap(
            output_path / "u1.npy", mode="w+", dtype=float, shape=history_shape)
        u2_tensor = np.lib.format.open_memmap(
            output_path / "u2.npy", mode="w+", dtype=float, shape=history_shape)
        t_array = np.lib.format.open_memmap(
            output_path / "t.npy", mode="w+", dtype=float, shape=(n_steps + 1,))
        
    eta_tensor[0] = eta
    u1_tensor[0] = u_1
    u2_tensor[0] = u_2
    t_array[0] = 0.0
    t = 0.0
    print('start')

    if n_steps >= 1:
        eta_next, u1_next, u2_next = step_euler(
            eta, u_1, u_2, B, dx, dy, dt, t, eps,
            M_matrix, K_matrix, lu_eta, lu_u, sigma, type_motion, t_eps
        )
        t = dt
        if not (np.isfinite(eta_next).all() and np.isfinite(u1_next).all() and np.isfinite(u2_next).all()):
            raise FloatingPointError(f"Divergence detected at t = {t:.4f}!")
        eta_tensor[1] = eta_next
        u1_tensor[1] = u1_next
        u2_tensor[1] = u2_next
        t_array[1] = t

        eta_prev, u1_prev, u2_prev = eta, u_1, u_2
        eta_curr, u1_curr, u2_curr = eta_next, u1_next, u2_next
        t_prev = 0.0
        t_curr = t

        for n in range(1, n_steps):
            eta_next, u1_next, u2_next = boussinesq_step_leapfrog(
                eta_prev, u1_prev, u2_prev,
                eta_curr, u1_curr, u2_curr,
                B, dx, dy, dt, t_curr, t_prev, eps,
                M_matrix, K_matrix, lu_eta, lu_u, sigma, type_motion, t_eps
            )
            t_next = (n + 1) * dt
            eta_prev, u1_prev, u2_prev = eta_curr, u1_curr, u2_curr
            eta_curr, u1_curr, u2_curr = eta_next, u1_next, u2_next
            t_prev = t_curr
            t_curr = t_next
            if not (np.isfinite(eta_curr).all() and np.isfinite(u1_curr).all() and np.isfinite(u2_curr).all()):
                raise FloatingPointError(f"Divergence detected at t = {t_curr:.4f}!")
            eta_tensor[n + 1] = eta_curr
            u1_tensor[n + 1] = u1_curr
            u2_tensor[n + 1] = u2_curr
            t_array[n + 1] = t_curr

    if output_dir is not None:
        for history in (eta_tensor, u1_tensor, u2_tensor, t_array):
            history.flush()
    print('end')
    return eta_tensor, u1_tensor, u2_tensor, t_array


def general_solver_return(type_motion, model, t_eps, B, sigma_dispersive,
                           u_1_initial, u_2_initial, L_x, Dx, Dy, L_y, H_typical,
                           Eps_nonL, T_f, Dt, output_dir=None):
    
    if model == "Shallow water":
        sigma_general = 0.0
        eta_initial = np.zeros_like(B)
    else:
        sigma_general = sigma_dispersive
        if type_motion != 0.0:
            eta_initial = np.zeros_like(B)
        else:
            # Static initialization with eta_initial=0 on every wall.
            # (I - sigma_general/3 * Delta) eta_initial = sigma_general/2 * Delta B.
            Nx, Ny = B.shape
            M_matrix, K_matrix, lu_eta, lu_u = assemble_fem_matrices(
                Nx, Ny, L_x, L_y, sigma_general
            )
            rhs_Q = -(sigma_general / 2.0) * K_matrix.dot(B.ravel())
            rhs_Q.reshape(Nx, Ny)[[0, -1], :] = 0.0
            rhs_Q.reshape(Nx, Ny)[:, [0, -1]] = 0.0
            eta_initial = lu_eta.solve(rhs_Q).reshape(Nx, Ny)
            # Release initialization operators before assembling the time solver.
            del M_matrix, K_matrix, lu_eta, lu_u, rhs_Q

    print(f"sigma={sigma_general}")
    eta_tensor_type1, u_1_tensor_type1, u_2_tensor_type1, t_array = run_update_general(
        eta_initial, u_1_initial, u_2_initial, B, L_x, L_y,
        sigma_general, Eps_nonL, T_f, Dt, type_motion, t_eps,
        output_dir=output_dir
    )
    return eta_tensor_type1, u_1_tensor_type1, u_2_tensor_type1, t_array
