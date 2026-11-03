#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:07:33 2026

@author: lamsahel
"""

import numpy as np
from solver_general_nondimensional_system import  rising_function


def max_relative_Hs_norm_bous(V1, V2, reference, t_array, t_eps, T_f, dx, dy, s):
    Nt, Nx, Ny = V1.shape

    norm_reference = hs_norm_R2(reference, dx, dy, s)
    
   
    if norm_reference < 1e-14:
        raise ValueError("Reference H^s norm is effectively zero.")

    diff = V1 - V2
    Error_norms = np.empty(Nt)
    
    for n in range(Nt):
        norm_diff = hs_norm_R2(diff[n], dx, dy, s)
        Error_norms[n] = norm_diff 
    relative_norms=Error_norms/norm_reference
   
    mask = (t_array > t_eps) & (t_array < T_f)
    if not np.any(mask):
        return np.nan, relative_norms, norm_reference
    
    max_relative_error = np.max(relative_norms[mask])
    return max_relative_error, relative_norms, norm_reference



def compute_max_relative_error(zeta_eps, zeta, x, y, t_array, t_eps, T_f):
    mask = (t_array > t_eps) & (t_array < T_f)
    z = zeta[mask, :, :]                    # static solution in time window
    z_eps = zeta_eps[mask, :, :]            # kinematic solution in time window
    
    denom = np.max(np.abs(z), axis=0)      
    denom[denom < 1e-14] = 1.0
    
    R = np.abs(z_eps - z) / denom          
    R_max = np.max(R)
    return R_max



def hs_norm_R2(u, dx, dy, s, pad_factor=1):
    Nx, Ny = u.shape
    Mx, My = pad_factor * Nx, pad_factor * Ny
    
    
    u_padded = np.zeros((Mx, My))
    u_padded[:Nx, :Ny] = u
    
  
    freq_x = np.fft.fftfreq(Mx, d=dx)
    freq_y = np.fft.fftfreq(My, d=dy)
    kx = 2 * np.pi * freq_x
    ky = 2 * np.pi * freq_y
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    weight = (1.0 + KX**2 + KY**2) ** s
    u_hat = np.fft.fft2(u_padded, norm='ortho')
    

    norm_sq = dx * dy * np.sum(weight * np.abs(u_hat)**2)
    return np.sqrt(norm_sq)



def hs_norm_R2_vector(U1, U2, dx, dy, s, pad_factor=1):
    """
    Compute the H^s(R^2; R^2) norm of a vector field U = (U1, U2)
    using the existing scalar hs_norm_R2.
    """
    n1 = hs_norm_R2(U1, dx, dy, s, pad_factor)
    n2 = hs_norm_R2(U2, dx, dy, s, pad_factor)
    return np.sqrt(n1**2 + n2**2)



def compute_Es_Bous(eta_eps, eta, u1_eps, u2_eps, u1_stat, u2_stat,
                    B, dx, dy, s, mask, f_values, pad_factor=1):
    """
    Compute E_bous at every stored time.

    """
    Nt = eta_eps.shape[0]
    f_values = np.asarray(f_values, dtype=float)

    if f_values.shape != (Nt,):
        raise ValueError("f_values must contain one value per stored time.")

    if not np.any(mask):
        raise ValueError("The mask does not select any time steps.")

    norm_B = hs_norm_R2(B, dx, dy, s, pad_factor)

    if norm_B < 1e-14:
        raise ValueError("Reference norm ||B||_{H^s} is effectively zero.")

    E_bous = np.empty(Nt, dtype=float)

    for n in range(Nt):
       
        diff_zeta = (
            eta_eps[n] - eta[n]
            + (f_values[n] - 1.0) * B
        )
        norm_zeta = hs_norm_R2(
            diff_zeta, dx, dy, s, pad_factor
        )

        diff_u1 = u1_eps[n] - u1_stat[n]
        diff_u2 = u2_eps[n] - u2_stat[n]
        norm_U = hs_norm_R2_vector(
            diff_u1, diff_u2, dx, dy, s - 1, pad_factor
        )

        E_bous[n] = (norm_zeta + norm_U) / norm_B

    max_error = np.max(E_bous[mask])

    return max_error, E_bous, norm_B




def compute_Es_sw(eta_eps, eta, u1_eps, u2_eps, u1_stat, u2_stat,
                    B, dx, dy, s, mask, f_values, pad_factor=1):
    """
    Compute E_bous at every stored time.
 
    """
    Nt = eta_eps.shape[0]
    f_values = np.asarray(f_values, dtype=float)

    if f_values.shape != (Nt,):
        raise ValueError("f_values must contain one value per stored time.")

    if not np.any(mask):
        raise ValueError("The mask does not select any time steps.")

    norm_B = hs_norm_R2(B, dx, dy, s-1, pad_factor)

    if norm_B < 1e-14:
        raise ValueError("Reference norm ||B||_{H^s-1} is effectively zero.")

    E_sw = np.empty(Nt, dtype=float)

    for n in range(Nt):
       
        a_eps=eta_eps[n]
        a=eta[n]
        diff_zeta = (
            a_eps - a
            + (f_values[n] - 1.0) * B
        )
        norm_zeta = hs_norm_R2(
            diff_zeta, dx, dy, s-1, pad_factor
        )
        b_1_eps= u1_eps[n]
        b_2_eps= u2_eps[n]
        b_1= u1_stat[n]
        b_2= u2_stat[n]



        diff_u1 = b_1_eps -  b_1
        diff_u2 = b_2_eps -  b_2
        norm_U = hs_norm_R2_vector(
            diff_u1, diff_u2, dx, dy, s - 1, pad_factor
        )

        E_sw[n] = (norm_zeta + norm_U) / norm_B

    max_error = np.max(E_sw[mask])

    return max_error, E_sw, norm_B



def bottom_wavenumber(B, Dx, Dy, R):
    B = np.asarray(B)
    nx, ny = B.shape

    if Dx <= 0 or Dy <= 0 or R < 0:
        raise ValueError("Require Dx > 0, Dy > 0, and R >= 0.")

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Dy)
    r = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    energy = np.abs(np.fft.fft2(B))**2
    if energy.sum() == 0:
        raise ValueError("B is zero: its normalized spectrum is undefined.")

    # Fraction of energy in wavenumbers r <= R
    F_R = float(energy[r <= R].sum() / energy.sum())

 
    energy[0, 0] = 0.0
    if energy.max() == 0:
        r_dominant = None
    else:
        i, j = np.unravel_index(np.argmax(energy), energy.shape)
        r_dominant = float(r[i, j])

    return r_dominant, F_R
