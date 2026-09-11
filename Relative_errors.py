#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:07:33 2026

@author: lamsahel
"""

import numpy as np



def max_relative_Hs_norm_bous(V1, V2, reference, t_array, t_eps, T_f, dx, dy, s):
    Nt, Nx, Ny = V1.shape

    # ---- denominator -------------------------------------------------------
    norm_reference = hs_norm_R2(reference, dx, dy, s)
    
    # Use a tolerance instead of exact zero
    if norm_reference < 1e-14:
        raise ValueError("Reference H^s norm is effectively zero.")

    # ---- numerator ---------------------------------------------------------
    diff = V1 - V2
    Error_norms = np.empty(Nt)
    
    for n in range(Nt):
        norm_diff = hs_norm_R2(diff[n], dx, dy, s)
        Error_norms[n] = norm_diff 
    relative_norms=Error_norms/norm_reference
    # ---- max over (t_eps, T_f) --------------------------------------------
    # Use <= / >= if you want inclusive bounds
    mask = (t_array > t_eps) & (t_array < T_f)
    if not np.any(mask):
        return np.nan, relative_norms, norm_reference
    
    max_relative_error = np.max(relative_norms[mask])
    return max_relative_error, relative_norms, norm_reference



def compute_max_relative_error(zeta_eps, zeta, x, y, t_array, t_eps, T_f):
    mask = (t_array > t_eps) & (t_array < T_f)
    z = zeta[mask, :, :]                    # static solution in time window
    z_eps = zeta_eps[mask, :, :]            # kinematic solution in time window
    
    # For each point P: what is max_t |zeta(P,t)|?
    denom = np.max(np.abs(z), axis=0)      # shape: (Nx, Ny)
    denom[denom < 1e-14] = 1.0
    
    # R(P,t) everywhere at once
    R = np.abs(z_eps - z) / denom           # broadcasting: (Nt,Nx,Ny) / (Nx,Ny)
    
    # Global max
    R_max = np.max(R)
    return R_max



def hs_norm_R2(u, dx, dy, s, pad_factor=1):
    Nx, Ny = u.shape
    Mx, My = pad_factor * Nx, pad_factor * Ny
    
    # Zero-pad
    u_padded = np.zeros((Mx, My))
    u_padded[:Nx, :Ny] = u
    
    # Wavenumbers on the padded grid
    freq_x = np.fft.fftfreq(Mx, d=dx)
    freq_y = np.fft.fftfreq(My, d=dy)
    kx = 2 * np.pi * freq_x
    ky = 2 * np.pi * freq_y
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    weight = (1.0 + KX**2 + KY**2) ** s
    u_hat = np.fft.fft2(u_padded, norm='ortho')
    
    # The domain area factor is still dx*dy, but now summed over the padded grid
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




def compute_Es_Bous(zeta_eps, zeta, u1_eps, u2_eps, u1_stat, u2_stat,
                    B, dx, dy, s,mask , pad_factor=1):
    """
    Compute the time-dependent Boussinesq error:
    
        E_s^{Bous}(t) = [ ||zeta^eps(t) - zeta(t)||_{H^s} 
                        + ||U^eps(t) - U(t)||_{H^{s-1}} ] / ||B||_{H^s}
    
    Uses the already-defined hs_norm_R2 for scalar fields.
    
    """
    Nt = zeta_eps.shape[0]
    
    # --- denominator ---------------------------------------------------------
    norm_B = hs_norm_R2(B, dx, dy, s, pad_factor)
    if norm_B < 1e-14:
        raise ValueError("Reference norm ||B||_{H^s} is effectively zero.")
    
    # --- compute error at each time step -------------------------------------
    E_bous = np.empty(Nt)
    
    for n in range(Nt):
        # ||zeta^eps - zeta||_{H^s}
        diff_zeta = zeta_eps[n] - zeta[n]
        norm_zeta = hs_norm_R2(diff_zeta, dx, dy, s, pad_factor)
        
        # ||U^eps - U||_{H^{s-1}}
        diff_u1 = u1_eps[n] - u1_stat[n]
        diff_u2 = u2_eps[n] - u2_stat[n]
        norm_U = hs_norm_R2_vector(diff_u1, diff_u2, dx, dy, s-1, pad_factor)
        
        E_bous[n] = (norm_zeta + norm_U) / norm_B
    
    
    max_error = np.max(E_bous[mask])
    return  max_error,E_bous, norm_B





def compute_Es_Sw(zeta_eps, zeta, u1_eps, u2_eps, u1_stat, u2_stat,
                    B, dx, dy, s,mask , pad_factor=1):
    """
    Compute the time-dependent Boussinesq error:
    
        E_s^{Bous}(t) = [ ||zeta^eps(t) - zeta(t)||_{H^s-1} 
                        + ||U^eps(t) - U(t)||_{H^{s-1}} ] / ||B||_{H^s-1}
    
    Uses the already-defined hs_norm_R2 for scalar fields.
    
    """
    Nt = zeta_eps.shape[0]
    
    # --- denominator ---------------------------------------------------------
    norm_B = hs_norm_R2(B, dx, dy, s-1, pad_factor)
    if norm_B < 1e-14:
        raise ValueError("Reference norm ||B||_{H^s-1} is effectively zero.")
    
    # --- compute error at each time step -------------------------------------
    E_bous = np.empty(Nt)
    
    for n in range(Nt):
        # ||zeta^eps - zeta||_{H^s}
        diff_zeta = zeta_eps[n] - zeta[n]
        norm_zeta = hs_norm_R2(diff_zeta, dx, dy, s-1, pad_factor)
        
        # ||U^eps - U||_{H^{s-1}}
        diff_u1 = u1_eps[n] - u1_stat[n]
        diff_u2 = u2_eps[n] - u2_stat[n]
        norm_U = hs_norm_R2_vector(diff_u1, diff_u2, dx, dy, s-1, pad_factor)
        
        E_bous[n] = (norm_zeta + norm_U) / norm_B
    
    
    max_error = np.max(E_bous[mask])
    return  max_error,E_bous, norm_B
