#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:27:15 2026

@author: lamsahel
"""

import numpy as np






#########Paper Prof Min chen 

def supe_Gaussian_bottom(x,y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    rho=10
    alpha=0.1
    m=8
    x_0=120.
    y_0=120.
    
    T_exp=- ( alpha**(2*m) ) *(  (rho**(m))* ( (X-x_0)**(2*m)) +      (rho**(-m))* ( (Y-y_0)**(2*m))      )
    
    return 5*( alpha**2)*np.exp(T_exp)





############ 
def B_Hammack_2d(x, y):
    """
    2D spatial bottom profile (elliptical extension of the 1D tanh bump).
    
    Parameters
    ----------
    x, y : 1-D ndarray
        Coordinate vectors (will be meshed with indexing='ij').
    z0 : float
        Maximum displacement amplitude.
    b : float
        Characteristic width / radius parameter.
    a : float, optional
        Stretching factor in x-direction. Default is 1.0.
    c : float, optional
        Stretching factor in y-direction. Default is 1.0.
        
    Returns
    -------
    H : 2-D ndarray
        Spatial profile H(x,y) of shape (len(x), len(y)).
        
        
    """
    a=1.0
    c=1.0
    b=10
    z0=0.3
    X, Y = np.meshgrid(x, y, indexing="ij")
    '''
    # Effective squared radial distance (ellipse when a != c)
    r_eff_sq = (X / a) ** 2 + (Y / c) ** 2
    
    denom = 1.0 + np.tanh(0.4 * b ** 2)
    numer = 1.0 + np.tanh(0.4 * (b ** 2 - r_eff_sq))
    
    return z0 * numer / denom
    '''
     #Rectangular (separable) extension: level sets are rounded rectangles.
    bx=b
    by=b
    denom_x = 1.0 + np.tanh(0.4 * bx ** 2)
    denom_y = 1.0 + np.tanh(0.4 * by ** 2)
    numer_x = 1.0 + np.tanh(0.4 * (bx ** 2 - X ** 2))
    numer_y = 1.0 + np.tanh(0.4 * (by ** 2 - Y ** 2))
    return z0 * (numer_x / denom_x) * (numer_y / denom_y)





def Okada_vecrtical_dip_slip_U_3_func(x, y,A_typical):
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    # Fault parameters
    delta_o = 13 * np.pi / 180     # Dip angle
    d_o = 10 #(Missotakis)   #20  (Dyth waves generation)        #3  #(km) (Paper Linear Thoery)
    
    W_o = 20    #  40                         #4
    L_o =  40    # 60                           #6  #(Km)
    D_amount = 6     #(m)?
    
    nu_o=0.27
    E_young=9.5
    mu_o = E_young/(2*(1+nu_o))
    lamda_o = (E_young*nu_o)/((1+nu_o)*(1-2*nu_o))
    
    # Check for division by zero before computing terms
    if np.cos(delta_o) == 0:
        print("check the formula: np.cos(delta_o) is zero")
        return None

    def compute_term(zeta_offset, eta_offset):
        zeta_o = X - zeta_offset
        p_o = Y * np.cos(delta_o) + d_o * np.sin(delta_o)
        eta_o = p_o - eta_offset
        
        q_o = Y * np.sin(delta_o) - d_o * np.cos(delta_o)
        d_tilde_o = eta_o * np.sin(delta_o) - q_o * np.cos(delta_o)
        R_o = np.sqrt(zeta_o**2 + eta_o**2 + q_o**2)
        X_o = np.sqrt(zeta_o**2 + q_o**2)
        
        # Compute I_5_o
        numerator = eta_o * (X_o + q_o * np.cos(delta_o)) + X_o * (R_o + X_o) * np.sin(delta_o)
        denominator = zeta_o * (R_o + X_o) * np.cos(delta_o)
        T_I5 = numerator / (denominator+1e-12)
        
        lambda_mu_factor = mu_o / (lamda_o + mu_o)
        I_5_o = lambda_mu_factor * (2.0 / np.cos(delta_o)) * np.arctan(T_I5)
        
        # Compute final term component
        term1 = (d_tilde_o * q_o) / (R_o * (R_o + zeta_o))
        term2 = np.sin(delta_o) * np.arctan((zeta_o * eta_o) / (q_o * R_o))
        term3 = I_5_o * np.sin(delta_o) * np.cos(delta_o)
        
        return term1 + term2 - term3

    # Evaluate the 4 corners of the rectangular fault
    T_1_O = compute_term(zeta_offset=0.0, eta_offset=0.0)
    T_2_O = compute_term(zeta_offset=0.0, eta_offset=W_o)
    T_3_O = compute_term(zeta_offset=L_o, eta_offset=0.0)
    T_4_O = compute_term(zeta_offset=L_o, eta_offset=W_o)

    # Combine results
    result = -(D_amount / (2 * np.pi)) * (T_1_O - T_2_O - T_3_O + T_4_O)
    
    return np.array(result)

def Okada_vecrtical_dip_slip_U_2_func(x, y,A_typical):
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    # Fault parameters
    delta_o = 13 * np.pi / 180     # Dip angle
    d_o = 10  #10 #(Missotakis)   #20  (Dyth waves generation)        #3  #(km) (Paper Linear Thoery)
    
    W_o = 20  #20      #  40                         #4
    L_o = 40  #40    # 60                           #6  #(Km)
    D_amount = 15# A_typical     #(m)?
    
    nu_o=0.27
    E_young=9.5
    mu_o = E_young/(2*(1+nu_o))
    lamda_o = (E_young*nu_o)/((1+nu_o)*(1-2*nu_o))
    
    # Check for division by zero before computing terms
    if np.cos(delta_o) == 0:
        print("check the formula: np.cos(delta_o) is zero")
        return None

    def compute_term(zeta_offset, eta_offset):
        zeta_o = X - zeta_offset
        p_o = Y * np.cos(delta_o) + d_o * np.sin(delta_o)
        eta_o = p_o - eta_offset
        
        q_o = Y * np.sin(delta_o) - d_o * np.cos(delta_o)
        R_o = np.sqrt(zeta_o**2 + eta_o**2 + q_o**2)
        X_o = np.sqrt(zeta_o**2 + q_o**2)
        
        d_tilde_o = eta_o * np.sin(delta_o) - q_o * np.cos(delta_o)
        y_tilde_o=eta_o * np.cos(delta_o) + q_o * np.sin(delta_o)
        
        # Compute I_5_o
        
        numerator = eta_o * (X_o + q_o * np.cos(delta_o)) + X_o * (R_o + X_o) * np.sin(delta_o)
        denominator = zeta_o * (R_o + X_o) * np.cos(delta_o)
        T_I5 = numerator / (denominator +1e-12)
        
        lambda_mu_factor = mu_o / (lamda_o + mu_o)
        I_5_o = lambda_mu_factor * (2.0 / np.cos(delta_o)) * np.arctan(T_I5)
        
        # Compute I_1_o
        term_inter= zeta_o/((R_o+d_tilde_o)*np.cos(delta_o))
        I_1_o=-lambda_mu_factor*term_inter-np.tan(delta_o)*I_5_o
        
        # Compute final term component
        term1 = (y_tilde_o * q_o) / (R_o * (R_o + zeta_o))
        term2 = np.cos(delta_o) * np.arctan((zeta_o * eta_o) / (q_o * R_o))
        term3 = I_1_o * np.sin(delta_o) * np.cos(delta_o)
        
        return term1 + term2 - term3

    # Evaluate the 4 corners of the rectangular fault
    T_1_O = compute_term(zeta_offset=0.0, eta_offset=0.0)
    T_2_O = compute_term(zeta_offset=0.0, eta_offset=W_o)
    T_3_O = compute_term(zeta_offset=L_o, eta_offset=0.0)
    T_4_O = compute_term(zeta_offset=L_o, eta_offset=W_o)

    # Combine results
    result = -(D_amount / (2 * np.pi)) * (T_1_O - T_2_O - T_3_O + T_4_O)
    
    return result