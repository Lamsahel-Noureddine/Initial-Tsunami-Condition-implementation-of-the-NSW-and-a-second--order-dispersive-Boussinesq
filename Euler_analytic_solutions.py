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
   
    if h <= 0:
        raise ValueError("Water depth h must be positive.")

    Ny, Nx = zeta.shape

   
    zeta_hat = np.fft.fft2(zeta)

 
    k = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 1 (x)
    l = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 0 (y)
    K, L = np.meshgrid(k, l)

    m = np.hypot(K, L)
    m_safe = np.where(m == 0, 1.0, m)   

    
    omega = np.sqrt(g * m_safe * np.tanh(m_safe * h))
    filter_kernel = np.cos(omega * t) / np.cosh(m_safe * h)
  

 
    filter_kernel[m == 0] = 1.0

    integrand_hat = zeta_hat * filter_kernel
    eta_i = np.fft.ifft2(integrand_hat).real

    return eta_i

def compute_analytic_bous(B, dx, dy, t):
    
    Ny, Nx = B.shape

    B_hat = np.fft.fft2(B)

    k = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 1 (x)
    l = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 0 (y)
    K, L = np.meshgrid(k, l)

    
    kl=(K**2+L**2)
    omega =np.sqrt( kl/(  1+(1/3)*kl )  )
    filter_kernel = np.cos(omega * t)* (1-(1/6)*kl)/(1+(1/3)*kl)
    
    integrand_hat = B_hat * filter_kernel
    zeta_i = np.fft.ifft2(integrand_hat).real

    return zeta_i


def compute_analytic_NSW(B, dx, dy, t):
    
    Ny, Nx = B.shape

    # 1. Forward 2-D FFT
    B_hat = np.fft.fft2(B)

    k = np.fft.fftfreq(Nx, d=dx) * 2.0 * np.pi   # axis 1 (x)
    l = np.fft.fftfreq(Ny, d=dy) * 2.0 * np.pi   # axis 0 (y)
    K, L = np.meshgrid(k, l)

    
    kl=(K**2+L**2)
    omega =np.sqrt( kl   )
    filter_kernel = np.cos(omega * t)
  
    integrand_hat = B_hat * filter_kernel
    zeta_i = np.fft.ifft2(integrand_hat).real

    return zeta_i
