#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:21:41 2026

@author: lamsahel
"""

import numpy as np

"""
2-D Spectral Filter for Linear Water-Wave Theory.

Computes:
    η_i(x, y, t) = F⁻¹{ ζ̂(k,ℓ) · cos(ω(m)·t) / cosh(m·h) }

where F⁻¹ is the inverse 2-D Fourier transform, ζ̂ = F{ζ},
m = √(k² + ℓ²) is the radial wavenumber, and
ω(m) = √(g·m·tanh(m·h)) is the linear dispersion relation.
"""


def compute_zeta_i(zeta, dx, dy, t, h, g=9.81):
    """
    Compute η_i(x, y, t) via 2-D FFT spectral filtering.

    Parameters
    ----------
    zeta : (Ny, Nx) ndarray
        Input field in physical space.
    dx, dy : float
        Grid spacing in x and y directions.
    t : float
        Time.
    h : float
        Water depth (must be > 0).
    g : float, optional
        Gravitational acceleration (default 9.81 m/s²).

    Returns
    -------
    eta_i : (Ny, Nx) ndarray
        Filtered field in physical space (real-valued).
    """
    if h <= 0:
        raise ValueError("Water depth h must be positive.")

    Ny, Nx = zeta.shape

    # 1. Forward 2-D FFT
    zeta_hat = np.fft.fft2(zeta)

    # 2. Angular wavenumber grids (cyclic order: x corresponds to axis 1)
    k = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 1 (x)
    l = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 0 (y)
    K, L = np.meshgrid(k, l)

    # 3. Radial wavenumber with safe masking at the origin
    m = np.hypot(K, L)
    m_safe = np.where(m == 0, 1.0, m)   # 1.0 avoids division-by-zero warnings

    # 4. Dispersion relation and spectral filter
    omega = np.sqrt(g * m_safe * np.tanh(m_safe * h))
    filter_kernel = np.cos(omega * t) / np.cosh(m_safe * h)
    #no filter
    #filter_kernel = np.cos(omega * t) 

    # At m=0, enforce the correct physical limit:
    #   lim_{m→0} cosh(m·h) = 1  and  lim_{m→0} ω(m) = 0
    #   → filter(0) = cos(0)/1 = 1
    filter_kernel[m == 0] = 1.0

    # 5. Apply filter in Fourier space and invert
    integrand_hat = zeta_hat * filter_kernel
    eta_i = np.fft.ifft2(integrand_hat).real

    return eta_i

def compute_analytic_bous(B, dx, dy, t):
    
    Ny, Nx = B.shape

    # 1. Forward 2-D FFT
    B_hat = np.fft.fft2(B)

    # 2. Angular wavenumber grids (cyclic order: x corresponds to axis 1)
    k = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 1 (x)
    l = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 0 (y)
    K, L = np.meshgrid(k, l)

    
    # 4. Dispersion relation and spectral filter
    kl=(K**2+L**2)
    omega =np.sqrt( kl/(  1+(1/3)*kl )  )
    filter_kernel = np.cos(omega * t)* (1-(1/6)*kl)/(1+(1/3)*kl)
    #no filter
    #filter_kernel = np.cos(omega * t) 

    # At m=0, enforce the correct physical limit:
    #   lim_{m→0} cosh(m·h) = 1  and  lim_{m→0} ω(m) = 0
    #   → filter(0) = cos(0)/1 = 1

    # 5. Apply filter in Fourier space and invert
    integrand_hat = B_hat * filter_kernel
    zeta_i = np.fft.ifft2(integrand_hat).real

    return zeta_i


def compute_analytic_NSW(B, dx, dy, t):
    
    Ny, Nx = B.shape

    # 1. Forward 2-D FFT
    B_hat = np.fft.fft2(B)

    # 2. Angular wavenumber grids (cyclic order: x corresponds to axis 1)
    k = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 1 (x)
    l = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 0 (y)
    K, L = np.meshgrid(k, l)

    
    # 4. Dispersion relation and spectral filter
    kl=(K**2+L**2)
    omega =np.sqrt( kl   )
    filter_kernel = np.cos(omega * t)
    #no filter
    #filter_kernel = np.cos(omega * t) 

    # At m=0, enforce the correct physical limit:
    #   lim_{m→0} cosh(m·h) = 1  and  lim_{m→0} ω(m) = 0
    #   → filter(0) = cos(0)/1 = 1

    # 5. Apply filter in Fourier space and invert
    integrand_hat = B_hat * filter_kernel
    zeta_i = np.fft.ifft2(integrand_hat).real

    return zeta_i