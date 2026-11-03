#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:22:24 2026

@author: lamsahel
"""

import numpy as np





def compute_zeta_0(B, dx, dy, H0):
    """
    Compute ζ_0 from  B
    """
    if H0 <= 0:
        raise ValueError("H0 must be positive.")

    Nx, Ny = B.shape


    B_hat = np.fft.fft2(B)

   
    kx = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 0 (x)
    ky = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 1 (y)


    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    k2 = KX**2 + KY**2

    alpha = H0**2 / 3.0

    filter_kernel = (1.0 - 0.5 * alpha * k2) / (1.0 + alpha * k2)

    zeta_0_hat = B_hat * filter_kernel
    zeta_0 = np.fft.ifft2(zeta_0_hat).real

    return zeta_0
