#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 13:07:08 2026
@author: lamsahel

Constant-step Boussinesq/shallow-water solver.
P1 in each coordinate gives Q1 elements on rectangles. The elliptic
operators use these finite elements; the right-hand sides use differences.
"""

import numpy as np
from scipy.sparse import diags, kron
from scipy.sparse.linalg import splu
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
    # Neumann conditions enter the weak form; enforce only normal velocities here.
    u_1[0, :] = 0.0
    u_1[-1, :] = 0.0
    u_2[:, 0] = 0.0
    u_2[:, -1] = 0.0
    return eta, u_1, u_2


def assemble_fem_matrices(N_x, N_y, L_x, L_y, sigma, nu):
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

    # C-order flattening: x is the first axis and y is the second.
    M_tensor = kron(M_x, M_y, format='csr')
    K_tensor = kron(M_x, K_y, format='csr') + kron(K_x, M_y, format='csr')
    A_op = M_tensor + (sigma / 3.0) * K_tensor
    B_op = M_tensor + nu * K_tensor

    # Impose u_1=0 on x walls and u_2=0 on y walls inside their linear systems.
    wall_x = np.zeros((N_x, N_y))
    wall_x[[0, -1], :] = 1.0
    wall_y = np.zeros((N_x, N_y))
    wall_y[:, [0, -1]] = 1.0
    free_x = diags(1.0 - wall_x.ravel(), format='csr')
    free_y = diags(1.0 - wall_y.ravel(), format='csr')
    B_op_x = free_x @ B_op @ free_x + diags(wall_x.ravel())
    B_op_y = free_y @ B_op @ free_y + diags(wall_y.ravel())
    lu_u = (splu(B_op_x.tocsc()), splu(B_op_y.tocsc()))
    return M_tensor, K_tensor, splu(A_op.tocsc()), lu_u


def compute_right_hand_sides(eta, u_1, u_2, B, Dx, Dy, eps, t, t_eps, type_motion):
    Nx, Ny = eta.shape
    zeta = eta + rising_function(t, t_eps, type_motion) * B
    H = np.maximum(1.0 + eps * (zeta - rising_function(t, t_eps, type_motion) * B), 1e-6)

    # Centered differences in the interior; second-order one-sided differences at walls.
    Hu1 = H * u_1
    Hu2 = H * u_2
    div_Hu = (
        np.gradient(Hu1, Dx, axis=0, edge_order=2)
        + np.gradient(Hu2, Dy, axis=1, edge_order=2)
    )
    R_1 = -div_Hu

    dzetadx, dzetady = np.gradient(zeta, Dx, Dy, edge_order=2)
    du1dx, du1dy = np.gradient(u_1, Dx, Dy, edge_order=2)
    du2dx, du2dy = np.gradient(u_2, Dx, Dy, edge_order=2)

    R_2_1 = eps * (u_1 * du1dx + u_2 * du1dy) + dzetadx
    R_3_1 = eps * (u_1 * du2dx + u_2 * du2dy) + dzetady
    R_2 = R_2_1
    R_3 = R_3_1
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
               M_matrix, K_matrix, lu_eta, lu_u, sigma, nu, type_motion, t_eps):
    """Euler starter: increments are measured from the initial state."""
    Nx, Ny = eta.shape
    R_1, R_2, R_3 = compute_right_hand_sides(
        eta, u_1, u_2, B, Dx, Dy, eps, t, t_eps, type_motion
    )

    # Weak bottom source, retaining the original zero-boundary-term convention.
    f_diff = rising_function(t + dt, t_eps, type_motion) - rising_function(t, t_eps, type_motion)
    source_eta = -(sigma / 2.0) * f_diff * K_matrix.dot(B.ravel())
    rhs_Q = dt * M_matrix.dot(R_1.ravel()) + source_eta
    P0 = lu_eta.solve(rhs_Q).reshape(Nx, Ny)
    eta_next = eta + P0

    rhs_P1 = -dt * M_matrix.dot(R_2.ravel())
    rhs_P2 = -dt * M_matrix.dot(R_3.ravel())
    rhs_P1.reshape(Nx, Ny)[[0, -1], :] = 0.0
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
                             sigma, nu, type_motion, t_eps):
    
    time_weight = 2.0 * dt

    Nx, Ny = eta_curr.shape
    R_1, R_2, R_3 = compute_right_hand_sides(
        eta_curr, u1_curr, u2_curr, B, Dx, Dy, eps, t_curr, t_eps, type_motion
    )

    # The bottom primitive uses the same two-step interval as the state update.
    f_diff = (
        rising_function(t_curr + dt, t_eps, type_motion)
        - rising_function(t_prev, t_eps, type_motion)
    )
    source_eta = -(sigma / 2.0) * f_diff * K_matrix.dot(B.ravel())
    rhs_Q = time_weight * M_matrix.dot(R_1.ravel()) + source_eta
    Pn = lu_eta.solve(rhs_Q).reshape(Nx, Ny)
    eta_next = eta_prev + Pn

    rhs_Q1 = -time_weight * M_matrix.dot(R_2.ravel())
    rhs_Q2 = -time_weight * M_matrix.dot(R_3.ravel())
    rhs_Q1.reshape(Nx, Ny)[[0, -1], :] = 0.0
    rhs_Q2.reshape(Nx, Ny)[:, [0, -1]] = 0.0
    Q1_n = lu_u[0].solve(rhs_Q1).reshape(Nx, Ny)
    Q2_n = lu_u[1].solve(rhs_Q2).reshape(Nx, Ny)
    u1_next = u1_prev + Q1_n
    u2_next = u2_prev + Q2_n

    eta_next, u1_next, u2_next = apply_boundary_conditions(eta_next, u1_next, u2_next)
    '''
    # Original artificial-viscosity block, still inactive.
    nu=1e-4
    eta_next[1:-1, 1:-1] += nu * (
        eta_next[2:, 1:-1] - 2*eta_next[1:-1, 1:-1] + eta_next[:-2, 1:-1]
        + eta_next[1:-1, 2:] - 2*eta_next[1:-1, 1:-1] + eta_next[1:-1, :-2]
    )
    u1_next[1:-1, 1:-1] += nu * (
        u1_next[2:, 1:-1] - 2*u1_next[1:-1, 1:-1] + u1_next[:-2, 1:-1]
        + u1_next[1:-1, 2:] - 2*u1_next[1:-1, 1:-1] + u1_next[1:-1, :-2]
    )
    u2_next[1:-1, 1:-1] += nu * (
        u2_next[2:, 1:-1] - 2*u2_next[1:-1, 1:-1] + u2_next[:-2, 1:-1]
        + u2_next[1:-1, 2:] - 2*u2_next[1:-1, 1:-1] + u2_next[1:-1, :-2]
    )
    '''
    return eta_next, u1_next, u2_next


def run_update_general(eta_init, u1_init, u2_init, B, Lx, Ly,
                       sigma, eps, nu, T_f, dt, type_motion, t_eps):
    
    """Run with prescribed constant dt; all fields have shape (Nx, Ny)."""
    

    # Reach T_f with whole steps; never alter the prescribed dt.
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
    M_matrix, K_matrix, lu_eta, lu_u = assemble_fem_matrices(Nx, Ny, Lx, Ly, sigma, nu)

    eta_list = [eta.copy()]
    u1_list = [u_1.copy()]
    u2_list = [u_2.copy()]
    t_list = [0.0]
    t = 0.0
    print('start')

    if n_steps >= 1:
        eta_next, u1_next, u2_next = step_euler(
            eta, u_1, u_2, B, dx, dy, dt, t, eps,
            M_matrix, K_matrix, lu_eta, lu_u, sigma, nu, type_motion, t_eps
        )
        t = dt
        if not (np.isfinite(eta_next).all() and np.isfinite(u1_next).all() and np.isfinite(u2_next).all()):
            raise FloatingPointError(f"Divergence detected at t = {t:.4f}!")
        eta_list.append(eta_next.copy())
        u1_list.append(u1_next.copy())
        u2_list.append(u2_next.copy())
        t_list.append(t)

        eta_prev, u1_prev, u2_prev = eta, u_1, u_2
        eta_curr, u1_curr, u2_curr = eta_next, u1_next, u2_next
        t_prev = 0.0
        t_curr = t

        for n in range(1, n_steps):
            eta_next, u1_next, u2_next = boussinesq_step_leapfrog(
                eta_prev, u1_prev, u2_prev,
                eta_curr, u1_curr, u2_curr,
                B, dx, dy, dt, t_curr, t_prev, eps,
                M_matrix, K_matrix, lu_eta, lu_u, sigma, nu, type_motion, t_eps
            )
            t_next = (n + 1) * dt
            eta_prev, u1_prev, u2_prev = eta_curr, u1_curr, u2_curr
            eta_curr, u1_curr, u2_curr = eta_next, u1_next, u2_next
            t_prev = t_curr
            t_curr = t_next
            if not (np.isfinite(eta_curr).all() and np.isfinite(u1_curr).all() and np.isfinite(u2_curr).all()):
                raise FloatingPointError(f"Divergence detected at t = {t_curr:.4f}!")
            eta_list.append(eta_curr.copy())
            u1_list.append(u1_curr.copy())
            u2_list.append(u2_curr.copy())
            t_list.append(t_curr)

    eta_tensor = np.array(eta_list)
    u1_tensor = np.array(u1_list)
    u2_tensor = np.array(u2_list)
    t_array = np.array(t_list)
    print('end')
    return eta_tensor, u1_tensor, u2_tensor, t_array


def general_solver_return(type_motion, model, t_eps, B, sigma_dispersive,
                           u_1_initial, u_2_initial, L_x, Dx, Dy, L_y, H_typical,
                           Eps_nonL, nu, T_f, Dt):
    if model == "Shallow water":
        sigma_general = 0.0
        eta_initial = np.zeros_like(B)
    else:
        sigma_general = sigma_dispersive
        if type_motion != 0.0:
            eta_initial = np.zeros_like(B)
        else:
            # Keep the supplied initializer call; its external definition was not provided.
            eta_initial = compute_zeta_0(B, Dx, Dy, 1) - B

    print(f"sigma={sigma_general}")
    eta_tensor_type1, u_1_tensor_type1, u_2_tensor_type1, t_array = run_update_general(
        eta_initial, u_1_initial, u_2_initial, B, L_x, L_y,
        sigma_general, Eps_nonL, nu, T_f, Dt, type_motion, t_eps
    )
    return eta_tensor_type1, u_1_tensor_type1, u_2_tensor_type1, t_array
