#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:22:24 2026

@author: lamsahel
"""

import numpy as np





def compute_zeta_0(B, dx, dy, H0):
    """
    Compute the initial free-surface displacement ζ_0(x,y) from the
    bathymetry B(x,y) via the spectral filter

        ζ̂_0(k) = (1 - α/2 |k|²) / (1 + α |k|²) · B̂(k),
        α = H0² / 3.

    Parameters
    ----------
    B : ndarray, shape (Nx, Ny)
        Bathymetry field.  First axis = x, second axis = y.
    dx, dy : float
        Grid spacing in x and y.
    H0 : float
        Reference water depth (> 0).

    Returns
    -------
    zeta_0 : ndarray, shape (Nx, Ny)
        Initial free-surface displacement (real-valued).
    """
    if H0 <= 0:
        raise ValueError("H0 must be positive.")

    Nx, Ny = B.shape

    # 1. Forward 2-D FFT of the bathymetry
    B_hat = np.fft.fft2(B)

    # 2. Angular wavenumber grids
    #    fftfreq gives the frequencies in the order fft2 uses them.
    kx = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 0 (x)
    ky = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 1 (y)

    # 3. Build 2-D wavenumber arrays with shape (Nx, Ny)
    #    indexing='ij'  →  KX[i,j] = kx[i],  KY[i,j] = ky[j]
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # 4. Radial wavenumber squared |k|² = kx² + ky²
    k2 = KX**2 + KY**2

    # 5. Filter coefficient α = H0² / 3
    alpha = H0**2 / 3.0

    # 6. Spectral filter  (well-behaved at k=0: limit = 1)
    filter_kernel = (1.0 - 0.5 * alpha * k2) / (1.0 + alpha * k2)

    # 7. Apply filter in Fourier space and invert
    zeta_0_hat = B_hat * filter_kernel
    zeta_0 = np.fft.ifft2(zeta_0_hat).real

    return zeta_0
