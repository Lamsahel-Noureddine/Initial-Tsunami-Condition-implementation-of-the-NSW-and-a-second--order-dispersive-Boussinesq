#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:41:20 2026
@author: lamsahel

Inviscid MacCormack solver for eta and the discharges Q_1, Q_2.
"""

import numpy as np
from solver_general_nondimensional_system import rising_function


def compute_fluxes(eta, Q_1, Q_2, H_0, Eps_nonL):
    H = np.maximum(H_0 + Eps_nonL * eta, 1e-6)
    u_1 = Q_1 / H
    u_2 = Q_2 / H
    E_1 = H * u_1**2
    E_2 = u_1 * u_2 * H
    E_3 = H * u_2**2
    return H, E_1, E_2, E_3


def apply_boundary_conditions(eta, Q_1, Q_2):
    # Original finite-difference wall conditions; H_0 is constant at the walls.
    eta[0, :] = eta[1, :]
    eta[-1, :] = eta[-2, :]
    eta[:, 0] = eta[:, 1]
    eta[:, -1] = eta[:, -2]

    Q_1[0, :] = 0.0
    Q_1[-1, :] = 0.0
    Q_2[:, 0] = 0.0
    Q_2[:, -1] = 0.0

    Q_2[0, :] = Q_2[1, :]
    Q_2[-1, :] = Q_2[-2, :]
    Q_1[:, 0] = Q_1[:, 1]
    Q_1[:, -1] = Q_1[:, -2]
    return eta, Q_1, Q_2


def compute_source_forward(eta, H_0, B, g, Eps_nonL):
    H = np.maximum(H_0 + Eps_nonL * eta, 1e-6)
    zeta = eta + B
    dzetax_f = zeta[2:, 1:-1] - zeta[1:-1, 1:-1]
    dzetay_f = zeta[1:-1, 2:] - zeta[1:-1, 1:-1]
    Sx_f = -g * H[1:-1, 1:-1] * dzetax_f
    Sy_f = -g * H[1:-1, 1:-1] * dzetay_f
    return Sx_f, Sy_f


def compute_source_backward(eta, H_0, B, g, Eps_nonL):
    H = np.maximum(H_0 + Eps_nonL * eta, 1e-6)
    zeta = eta + B
    dzetax_b = zeta[1:-1, 1:-1] - zeta[:-2, 1:-1]
    dzetay_b = zeta[1:-1, 1:-1] - zeta[1:-1, :-2]
    Sx_b = -g * H[1:-1, 1:-1] * dzetax_b
    Sy_b = -g * H[1:-1, 1:-1] * dzetay_b
    return Sx_b, Sy_b


def MacCormack_diff_solver(eta, Q_1, Q_2, H_0, B, g, Dx, Dy, Dt,
                           nu, Eps_nonL, B_next=None):
    """Use B at the start and B_next at the end of the step.

    B_next=None assumes a static bottom during the step.
    nu is retained for compatibility; this scheme has no viscosity term.
    """
    if B_next is None:
        B_next = B

    rx = Dt / Dx
    ry = Dt / Dy
    H, E_1, E_2, E_3 = compute_fluxes(eta, Q_1, Q_2, H_0, Eps_nonL)

    # Forward predictor with the bottom at t_n.
    Sx_f, Sy_f = compute_source_forward(eta, H_0, B, g, Eps_nonL)
    eta_p = np.copy(eta)
    Q_1_p = np.copy(Q_1)
    Q_2_p = np.copy(Q_2)

    eta_p[1:-1, 1:-1] -= (
        rx * (Q_1[2:, 1:-1] - Q_1[1:-1, 1:-1])
        + ry * (Q_2[1:-1, 2:] - Q_2[1:-1, 1:-1])
    )
    Q_1_p[1:-1, 1:-1] -= Eps_nonL * (
        rx * (E_1[2:, 1:-1] - E_1[1:-1, 1:-1])
        + ry * (E_2[1:-1, 2:] - E_2[1:-1, 1:-1])
    ) - rx * Sx_f
    Q_2_p[1:-1, 1:-1] -= Eps_nonL * (
        rx * (E_2[2:, 1:-1] - E_2[1:-1, 1:-1])
        + ry * (E_3[1:-1, 2:] - E_3[1:-1, 1:-1])
    ) - ry * Sy_f

    eta_p, Q_1_p, Q_2_p = apply_boundary_conditions(eta_p, Q_1_p, Q_2_p)

    # Backward corrector with the predicted fields and the bottom at t_(n+1).
    H_p, E_1_p, E_2_p, E_3_p = compute_fluxes(
        eta_p, Q_1_p, Q_2_p, H_0, Eps_nonL
    )
    Sx_b, Sy_b = compute_source_backward(eta_p, H_0, B_next, g, Eps_nonL)

    eta_c = np.copy(eta_p)
    Q_1_c = np.copy(Q_1_p)
    Q_2_c = np.copy(Q_2_p)

    eta_c[1:-1, 1:-1] -= (
        rx * (Q_1_p[1:-1, 1:-1] - Q_1_p[:-2, 1:-1])
        + ry * (Q_2_p[1:-1, 1:-1] - Q_2_p[1:-1, :-2])
    )
    Q_1_c[1:-1, 1:-1] -= Eps_nonL * (
        rx * (E_1_p[1:-1, 1:-1] - E_1_p[:-2, 1:-1])
        + ry * (E_2_p[1:-1, 1:-1] - E_2_p[1:-1, :-2])
    ) - rx * Sx_b
    Q_2_c[1:-1, 1:-1] -= Eps_nonL * (
        rx * (E_2_p[1:-1, 1:-1] - E_2_p[:-2, 1:-1])
        + ry * (E_3_p[1:-1, 1:-1] - E_3_p[1:-1, :-2])
    ) - ry * Sy_b

    eta_c, Q_1_c, Q_2_c = apply_boundary_conditions(eta_c, Q_1_c, Q_2_c)

    eta_n_plus1 = 0.5 * (eta + eta_c)
    Q1_n_plus1 = 0.5 * (Q_1 + Q_1_c)
    Q2_n_plus1 = 0.5 * (Q_2 + Q_2_c)

    # Correct eta only if the depth floor is active; allow Eps_nonL=0.
    h_n_plus1 = np.maximum(H_0 + Eps_nonL * eta_n_plus1, 1e-6)
    if Eps_nonL != 0.0:
        eta_n_plus1 = eta_n_plus1 + (
            h_n_plus1 - (H_0 + Eps_nonL * eta_n_plus1)
        ) / Eps_nonL

    eta_n_plus1, Q1_n_plus1, Q2_n_plus1 = apply_boundary_conditions(
        eta_n_plus1, Q1_n_plus1, Q2_n_plus1
    )
    return eta_n_plus1, Q1_n_plus1, Q2_n_plus1


def run_time_update(eta_init, Q1_init, Q2_init, B, H_0, g, Dx, Dy,
                    T_f, nu, Eps_nonL, t_eps, type_motion, Dt):
    """Use prescribed Dt, with a shorter last step if necessary.

    CFL and nu are retained for compatibility; neither changes Dt or the PDE.
    """
   
    eta = eta_init.copy()
    Q_1 = Q1_init.copy()
    Q_2 = Q2_init.copy()
    eta, Q_1, Q_2 = apply_boundary_conditions(eta, Q_1, Q_2)
    B_n = rising_function(0, t_eps, type_motion) * B
    if type_motion == 1.0:
        B_n = B.copy()  # Evolve from t=0+ after the instantaneous bottom jump.

    eta_list = [eta.copy()]
    Q_1_list = [Q_1.copy()]
    Q_2_list = [Q_2.copy()]
    t_list = [0.0]
    t = 0.0

    # Indexed times avoid accumulated drift and unnecessary tiny final steps.
    n_steps = int(round(T_f / Dt))
    if not np.isclose(n_steps * Dt, T_f, rtol=1e-12, atol=0.0):
        n_steps = int(np.ceil(T_f / Dt))

    for n in range(n_steps):
        t_next = T_f if n == n_steps - 1 else (n + 1) * Dt
        B_next = rising_function(t_next, t_eps, type_motion) * B

        eta_n_plus1, Q1_n_plus1, Q2_n_plus1 = MacCormack_diff_solver(
            eta, Q_1, Q_2, H_0, B_n, g, Dx, Dy,
            t_next - t, nu, Eps_nonL, B_next=B_next
        )

        if not (
            np.isfinite(eta_n_plus1).all()
            and np.isfinite(Q1_n_plus1).all()
            and np.isfinite(Q2_n_plus1).all()
        ):
            raise FloatingPointError(f"Divergence detected at t = {t_next:.6g}!")

        eta = eta_n_plus1
        Q_1 = Q1_n_plus1
        Q_2 = Q2_n_plus1
        t = t_next
        B_n = B_next

        eta_list.append(eta.copy())
        Q_1_list.append(Q_1.copy())
        Q_2_list.append(Q_2.copy())
        t_list.append(t)

    eta_tensor = np.array(eta_list)
    Q_1_tensor = np.array(Q_1_list)
    Q_2_tensor = np.array(Q_2_list)
    t_array = np.array(t_list)
    print(t)
    return eta_tensor, Q_1_tensor, Q_2_tensor, t_array
