#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:06:47 2026

@author: lamsahel
"""





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from Bottom_profiles import Okada_vecrtical_dip_slip_U_3_func, Okada_vecrtical_dip_slip_U_2_func





# # # # # # # # # # # # # # # # # # # # # # # # # #  Test 1# # # # # # # # # # # # # # # # # # # # # # 



def plot_3d_B(L_x, L_y,  A_typical, H_typical):

    N_x = 601                     # Number of grid points in x 
    Dx = L_x / (N_x - 1)          # Grid spacing in x
    x = np.linspace(0, L_x, N_x)

    N_y = 601                     # Number of grid points in y
    Dy = L_y / (N_y - 1)          # Grid spacing in y
    y = np.linspace(0, L_y, N_y)

    X, Y = np.meshgrid(x, y, indexing="ij")
    deep_scale = H_typical / 1000.0   # Convert from meters to kilometers for the fault model
    B = (1.0 / H_typical) * Okada_vecrtical_dip_slip_U_3_func(deep_scale * (x - L_x / 2) + 20.0,   deep_scale * (y - L_y / 2),   A_typical)
    # B = (1.0 / H_typical) *   Okada_vecrtical_dip_slip_U_2_func(deep_scale * (x - L_x / 2) + 20.0,   deep_scale * (y - L_y / 2),   A_typical)


    fig = plt.figure(figsize=(10, 6))
    
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, H_typical * B, cmap='jet', rstride=4, cstride=4, edgecolor='none')
    
    #ax.set_title("Bathymetry Profile")
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_xlim(0, L_x)   # Force x-axis to exactly [0, L_x]
    ax.set_ylim(0, L_y)   # Force y-axis to exactly [0, L_y]
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$B$ (m)", rotation=0, labelpad=12, fontsize=12)    
    ax.view_init(20, 2)
    
    plt.tight_layout()
    plt.savefig("Dip_slip_fault.pdf", 
                format="pdf", 
                bbox_inches="tight",  
                pad_inches=0.1         
               )
    plt.show()
    
  
  
def plot_3d(x, y, zeta_choosen, t_phyiscal, H_0, 
                         points, point_labels):
    """
    Plot 3D surface + contour with marked locations.
    
    Parameters
    ----------
    points : list of tuples, optional
        List of (x_i, y_i) coordinates, e.g. [(30,40), (60,70), (90,50), (120,80)]
    point_labels : list of str, optional
        Labels for each point, e.g. ['P_1', 'P_2', 'P_3', 'P_4']
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig = plt.figure(figsize=(14, 7))
    z_data = H_0 * zeta_choosen

    # --- 3D Surface ---
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(
        X, Y, z_data, cmap='jet', rstride=4, cstride=4, edgecolor='none'
    )
    ax1.set_title(f'Surface Profile at t={t_phyiscal/60:.3f} min')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel(r'$\zeta$  (m)', rotation=0, labelpad=12, fontsize=12)
    ax1.view_init(30, 0)
    ax1.set_zticks(np.linspace(np.min(z_data), np.max(z_data), 5))

    # --- Contour with Points ---
    ax2 = fig.add_subplot(122)

    levels = np.linspace(np.min(z_data), np.max(z_data), 20)
    levels = levels[np.abs(levels) > 1e-4]
    cf = ax2.contour(X, Y, z_data, levels=levels, cmap='jet')
    ax2.set_title(f'Contour at t={t_phyiscal/60:.3f} min')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')

    # Mark locations with small squares and labels
    if points is not None:
        if point_labels is None:
            point_labels = [f'P_{i+1}' for i in range(len(points))]
        
        for (px, py), label in zip(points, point_labels):
            # Small black square marker with white edge
            ax2.plot(px, py, 's', color='black', markersize=6, 
                     markeredgecolor='white', markeredgewidth=0.5, zorder=5)
            # Label with offset and white background box
            ax2.annotate(label, xy=(px, py), xytext=(6, 6),
                        textcoords='offset points', fontsize=10,
                        color='black', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', 
                                  facecolor='white', alpha=0.8, edgecolor='none'))

    cbar = plt.colorbar(cf, ax=ax2, shrink=0.8, ticks=cf.levels)
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig('3d_tsunami_points.pdf', format='pdf', 
                bbox_inches='tight', pad_inches=0.1)
    plt.show()




def plot_passive_Filter_T1(x, y, t, V1, V2, loc, H_typical):
  # 1. Find the nearest grid indices for the location
  idx_x = np.abs(x - loc[0]).argmin()
  idx_y = np.abs(y - loc[1]).argmin()

  # 2. Extract time-series data at the location
  v1 = V1[:, idx_x, idx_y]
  v2 = V2[:, idx_x, idx_y]
 

  # 3. Set up a single plot layout  passive approach
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.plot(t, H_typical * v1, 'b-', linewidth=2.0, label='Boussinesq filter')

  ax.plot(t, H_typical * v2, 'r--', linewidth=2.0, label=r'Passive approach' )
  
  ax.set_title(f'Tide Gauge at $p_4=$({loc[0]}, {loc[1]})')
  ax.set_xlabel('$t$ ')
  ax.set_ylabel('Surface elevations (m)')
  ax.set_xlim(t[0], t[-1])

  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()

  # Save as PDF
  plt.savefig(
      'T1_Gauge4_comapres_Filter_No.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
  )

  plt.show()
  
def plot_tensors_at_locations_T1(x, y, t, V1, V2,V3,V4, loc, H_typical):
  # 1. Find the nearest grid indices for the location
  idx_x = np.abs(x - loc[0]).argmin()
  idx_y = np.abs(y - loc[1]).argmin()

  # 2. Extract time-series data at the location
  v1 = V1[:, idx_x, idx_y]
  v2 = V2[:, idx_x, idx_y]
  v3=V3[:, idx_x, idx_y]
  v4=V4[:, idx_x, idx_y]

  # 3. Set up 1 single plot layout
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.plot(t, H_typical * v1, 'b-', linewidth=2.0, label='Boussinesq filter')
  ax.plot(t, H_typical * v3, color='#000000', linestyle='--', linewidth=2.0, label='Linear Euler')
  ax.plot(t, H_typical * v4, color='#2ca02c', linestyle=':',  linewidth=2.0, label='Linear Boussinesq')
  ax.set_title(f'Tide Gauge at $p_4=$({loc[0]}, {loc[1]})')
  ax.set_xlabel('$t$ ')
  ax.set_ylabel('Surface elevations (m)')
  ax.set_xlim(t[0], t[-1])

  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()

  # Save as PDF
  plt.savefig(
      'T1_selected2_comapres_Filter.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
  )

  plt.show()
  
  
  # 3. Set up 2 single plot layout
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.plot(t, H_typical * v2, 'r--', linewidth=2.0, label=r'Passive approach' )
  ax.plot(t, H_typical * v3, color='#000000', linestyle='--', linewidth=2.0, label='Linear Euler')
  ax.plot(t, H_typical * v4, color='#2ca02c', linestyle=':',  linewidth=2.0, label='Linear Boussinesq')
  ax.set_title(f'Tide Gauge at $p_4=$({loc[0]}, {loc[1]})')
  ax.set_xlabel('$t$ ')
  ax.set_ylabel('Surface elevations (m)')
  ax.set_xlim(t[0], t[-1])

  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()

  # Save as PDF
  plt.savefig(
      'T1_selected2_comapres_NoFilter.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
  )

  plt.show()
  



def plot_relative_erro_T1(x, y, t, V1, V2, loc):
  # 1. Find the nearest grid indices for the location
  idx_x = np.abs(x - loc[0]).argmin()
  idx_y = np.abs(y - loc[1]).argmin()

  # 2. Extract time-series data at the location
  v1 = V1[:, idx_x, idx_y]
  v2 = V2[:, idx_x, idx_y]
 
  fig, ax = plt.subplots(figsize=(8, 5))

  # --- Plot: Single Location ---
  ax.plot(
      t,
      np.abs(v1 - v2) / np.max(np.abs(v1)),
      'k--',
      linewidth=1.2,
      label='Relative Error',
  )

  ax.set_title(f'Tide gauge at ({loc[0]}, {loc[1]}).')
  ax.set_xlabel('$t$')
  ax.set_ylabel('Relative Error')
  ax.set_xlim(t[0], t[-1])
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()
  plt.savefig(
      'relative_error_single.pdf',
      format='pdf',
      bbox_inches='tight',
      pad_inches=0.1,
  )
  plt.show()







# # # # # # # # # # # # # # # # # # # # # # # # # #  Test 2# # # # # # # # # # # # # # # # # # # # # # 



def plot_tensors_at_locations_T2(x, y, t, V1, V2, loc, H_typical,t_eps1):
  # 1. Find the nearest grid indices for the location
  idx_x = np.abs(x - loc[0]).argmin()
  idx_y = np.abs(y - loc[1]).argmin()

  # 2. Extract time-series data at the location
  v1 = V1[:, idx_x, idx_y]
  v2 = V2[:, idx_x, idx_y]

  # 3. Set up a single plot layout
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.plot(t, H_typical * v1, 'b-', linewidth=2.0, label='Static Boussinesq')
  ax.plot(t, H_typical * v2, 'r--', linewidth=2.0, label=rf"Kinematic  Boussinesq: $t_\epsilon = {t_eps1:.2f}$" )
 
  ax.set_title(f'Tide Gauge at $p_1=$({loc[0]}, {loc[1]})')
  ax.set_xlabel('$t$ ')
  ax.set_ylabel('Surface elevations (m)')
  ax.set_xlim(t[0], t[-1])
# Vertical line — no label here
  ax.axvline(x=t_eps1, color='k', linestyle='--', linewidth=1.5)

# Label placed below the x-axis
  ax.annotate(
    r'$t_\epsilon$',
    xy=(t_eps1+0.4, 0.045),
    xycoords=('data', 'axes fraction'),
    xytext=(0, -15),
    textcoords='offset points',
    ha='center',
    va='top',
    fontsize=12,
    color='k'
   )  
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()

  # Save as PDF
  plt.savefig(
      '1s_Bous_static_Kinematic.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
  )

  plt.show()





def plot_relative_erro_T2(x, y, t, V1, V2,mask,loc):
  # 1. Find the nearest grid indices for the location
  idx_x = np.abs(x - loc[0]).argmin()
  idx_y = np.abs(y - loc[1]).argmin()

  # 2. Extract time-series data at the location
  v1 = V1[mask, idx_x, idx_y]
  v2 = V2[mask, idx_x, idx_y]

  fig, ax = plt.subplots(figsize=(8, 5))

  # --- Plot: Single Location ---
  denom=np.max(np.abs(v1))
  R =np.abs(v1 - v2)/ denom

  ax.plot(
      t[mask],
     R,
      'k--',
      linewidth=1.2,
      label='Relative Error',
  )

  ax.set_title(f'Tide gauge at ({loc[0]}, {loc[1]}).')
  ax.set_xlabel('$t$')
  ax.set_ylabel('Relative Error')
  ax.set_xlim(t[0], t[-1])
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()
  plt.savefig(
      'err_location.pdf',
      format='pdf',
      bbox_inches='tight',
      pad_inches=0.1,
  )
  return R
  plt.show()



def plot_relative_error_4locations(t_eps, max_errors_dict):
    """
    Plot maximum relative error vs t_epsilon on a normal linear scale.
    x-axis goes from large to small with clear t_eps values.
    """
    t_eps = np.asarray(t_eps)
    
    # Sort the data so x-axis goes from LARGE to SMALL
    idx = np.argsort(t_eps)[::-1]
    t_eps_sorted = t_eps[idx]
    
    colors = ['b', 'r', 'g', 'm']
    markers = ['^', 's', 'o', 'D']
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    for i, (loc_key, max_err) in enumerate(max_errors_dict.items()):
        max_err = np.asarray(max_err)
        max_err = max_err[idx]
        loc_num = loc_key.split('_')[-1]
        
        ax.plot(
            t_eps_sorted,
            100 * max_err,
            marker=markers[i],
            markersize=10,
            markerfacecolor='none',
            markeredgecolor=colors[i],
            markeredgewidth=1.5,
            linewidth=1.5,
            color=colors[i],
            label=rf'$p_{{{loc_num}}}$'
        )
    
    # ----- X-AXIS WITH CLEAR t_eps VALUES -----
    ax.set_xlim(3.5, -0.1)
    
    # Show ALL the actual t_eps values on x-axis
    ax.set_xticks(t_eps_sorted)
    ax.set_xticklabels([f'{val:.3f}' for val in t_eps_sorted], rotation=90, ha='right')
    
    ax.set_xlabel(r'$t_\epsilon$', fontsize=14)
    ax.set_ylabel(r'$\max_{t_\epsilon \leq \tau \leq T} \mathcal{R}(\tau, X)$ (%)', fontsize=13)
    # ax.set_title(
    #     r'Maximum relative error for $t \in (t_\epsilon, T)$',
    #     fontsize=13
    # )
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize=12, ncol=2)
    
    plt.tight_layout()
    
    plt.savefig('Bous_local_Error.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

    plt.show()



def plot_max_Es_Bous_vs_teps_log(t_eps, max_E_bous_values, s, T_f=None):
   t_eps = np.asarray(t_eps)
   
   # Sort the data so x-axis goes from LARGE to SMALL
   idx = np.argsort(t_eps)[::-1]
   t_eps_sorted = t_eps[idx]
   
   max_E_bous_values = np.asarray(max_E_bous_values)
   max_E_bous_values = max_E_bous_values[idx]
   
   # ----- TAKE log10 (base 10) of both quantities -----
   log_t_eps = np.log10(t_eps_sorted)              # x-axis data:  log10(t_eps)
   log_E     = np.log10(100*max_E_bous_values)   # y-axis data:  log10(max E in %)
   
   fig, ax = plt.subplots(figsize=(10, 5.5))
   
   ax.plot(
       log_t_eps,                                  # <- was t_eps_sorted
       log_E,                                      # <- was 100 * max_E_bous_values
       marker='^',
       markersize=10,
       markerfacecolor='none',
       markeredgecolor='b',
       markeredgewidth=1.5,
       linewidth=1.5,
       color='b',
       label=rf'$E_{{{s}}}^{{\mathrm{{Bous}}}}$'
   )
   
   # ----- X-AXIS WITH CLEAR log10(t_eps) VALUES (large -> small) -----
   ax.set_xlim(log_t_eps.max() + 0.05, log_t_eps.min() - 0.05)   # reversed, with margin
   # Show ALL the actual log10(t_eps) values on x-axis
   ax.set_xticks(log_t_eps)
   ax.set_xticklabels([f'{v:.3f}' for v in log_t_eps], rotation=90, ha='right')
   
   ax.set_xlabel(r'$\log_{10}(t_\epsilon)$', fontsize=14)
   
   T_label = f'{T_f}' if T_f is not None else 'T'
   ax.set_ylabel(
       rf'$\log_{{10}}\left(\max_{{t_\epsilon \leq \tau \leq {T_label}}} E_{{{s}}}^{{\mathrm{{Bous}}}}(\tau) \, (\%)\right)  $',
       fontsize=13
   )
   
   ax.grid(True, linestyle=':', alpha=0.6)
   ax.legend(loc='best', fontsize=12)
   
   plt.tight_layout()
 
    
    
   p, c = np.polyfit(log_t_eps, log_E, 1)
     
   print(f"Approximate slope p = {p:.4f}")
   print(f"Approximate constant C = {10**(c)/100:.4f}") 
     # ----- Plot the fitted slope line over the data -----
   fit_x = np.linspace(log_t_eps.min(), log_t_eps.max(), 100)
   fit_y = p * fit_x + c
    
   ax.plot(fit_x, fit_y,
   'k--',
    linewidth=1.8,
    label=rf'approximate slope $p = {p:.3f}$')
       
    # ----- Annotate the slope value on the plot -----
   mid = len(fit_x) // 2
   ax.annotate(rf'$slope \approx {p:.3f}$',
   xy=(fit_x[mid], fit_y[mid]),
     xytext=(fit_x[mid] - 0.5, fit_y[mid] + 0.3),
    fontsize=12, color='black',
    arrowprops=dict(arrowstyle='->', color='black'))
   plt.savefig(f'Es_bous_s{s}_vs_teps_loglog.pdf', format='pdf',
                bbox_inches='tight', pad_inches=0.1)
    
   plt.show()







# # # # # # # # # # # # # # # # # # # # # # # # # #  Test 3# # # # # # # # # # # # # # # # # # # # # # 


def plot_tensors_at_locations_T3(x, y, t, V1, V2,V3, loc, H_typical):
  # 1. Find the nearest grid indices for the location
  idx_x = np.abs(x - loc[0]).argmin()
  idx_y = np.abs(y - loc[1]).argmin()

  # 2. Extract time-series data at the location
  v1 = V1[:, idx_x, idx_y]
  v2 = V2[:, idx_x, idx_y]
  v3=V3[:, idx_x, idx_y]
  # v4=V4[:, idx_x, idx_y]

 
  # 3. Set up a single plot layout
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.plot(t, H_typical * v1, 'b-', linewidth=2.0, label='Shallow-water')
  ax.plot(t, H_typical * v2, color='#d62728', linestyle='-.',  marker='o', markersize=4, markevery=20, linewidth=2.0, label='Shallow-water: MacCormack')
  ax.plot(t, H_typical * v3, color='#000000', linestyle='--', linewidth=2.0, label='Linear Euler')
  # ax.plot(t, H_typical * v4, color='#2ca02c', linestyle=':',  linewidth=2.0, label='Linear shallow-Water')
  ax.set_title(f'Tide Gauge at $p_4=$({loc[0]}, {loc[1]})')
  ax.set_xlabel('$t$ ')
  ax.set_ylabel('Surface elevations (m)')
  ax.set_xlim(t[0], t[-1])
# Vertical line — no label here
 
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend()

  plt.tight_layout()

  # Save as PDF
  plt.savefig(
      'SW_comap_Euler_Gauge4.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1
  )

  plt.show()
  




def plot_relative_error_4locations_SW(t_eps, max_errors_dict):
    """
    Plot maximum relative error vs t_epsilon on a normal linear scale.
    x-axis goes from large to small with clear t_eps values.
    """
    t_eps = np.asarray(t_eps)
    
    # Sort the data so x-axis goes from LARGE to SMALL
    idx = np.argsort(t_eps)[::-1]
    t_eps_sorted = t_eps[idx]
    
    colors = ['b', 'r', 'g', 'm']
    markers = ['^', 's', 'o', 'D']
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    for i, (loc_key, max_err) in enumerate(max_errors_dict.items()):
        max_err = np.asarray(max_err)
        max_err = max_err[idx]
        loc_num = loc_key.split('_')[-1]
        
        ax.plot(
            t_eps_sorted,
            100 * max_err,
            marker=markers[i],
            markersize=10,
            markerfacecolor='none',
            markeredgecolor=colors[i],
            markeredgewidth=1.5,
            linewidth=1.5,
            color=colors[i],
            label=rf'$p_{{{loc_num}}}$'
        )
    
    # ----- X-AXIS WITH CLEAR t_eps VALUES -----
    ax.set_xlim(3.5, -0.1)
    
    # Show ALL the actual t_eps values on x-axis
    ax.set_xticks(t_eps_sorted)
    ax.set_xticklabels([f'{val:.3f}' for val in t_eps_sorted], rotation=90, ha='right')
    
    ax.set_xlabel(r'$t_\epsilon$', fontsize=14)
    ax.set_ylabel(r'$\max_{t_\epsilon \leq \tau \leq T} \mathcal{R}(\tau, X)$ (%)', fontsize=13)
    # ax.set_title(
    #     r'Maximum relative error for $t \in (t_\epsilon, T)$',
    #     fontsize=13
    # )
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize=12, ncol=2)
    
    plt.tight_layout()
    
    plt.savefig('Sw_local_Error.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

    plt.show()




def plot_max_Es_Bous_vs_teps_log_sw(t_eps, max_E_bous_values, s, T_f=None):
   t_eps = np.asarray(t_eps)
   
   # Sort the data so x-axis goes from LARGE to SMALL
   idx = np.argsort(t_eps)[::-1]
   t_eps_sorted = t_eps[idx]
   
   max_E_bous_values = np.asarray(max_E_bous_values)
   max_E_bous_values = max_E_bous_values[idx]
   
   # ----- TAKE log10 (base 10) of both quantities -----
   log_t_eps = np.log10(t_eps_sorted)              # x-axis data:  log10(t_eps)
   log_E     = np.log10(100*max_E_bous_values)   # y-axis data:  log10(max E in %)
   
   fig, ax = plt.subplots(figsize=(10, 5.5))
   
   ax.plot(
       log_t_eps,                                  # <- was t_eps_sorted
       log_E,                                      # <- was 100 * max_E_bous_values
       marker='^',
       markersize=10,
       markerfacecolor='none',
       markeredgecolor='b',
       markeredgewidth=1.5,
       linewidth=1.5,
       color='b',
       label=rf'$E_{{{s}}}$'
   )
   
   # ----- X-AXIS WITH CLEAR log10(t_eps) VALUES (large -> small) -----
   ax.set_xlim(log_t_eps.max() + 0.05, log_t_eps.min() - 0.05)   # reversed, with margin
   # Show ALL the actual log10(t_eps) values on x-axis
   ax.set_xticks(log_t_eps)
   ax.set_xticklabels([f'{v:.3f}' for v in log_t_eps], rotation=90, ha='right')
   
   ax.set_xlabel(r'$\log_{10}(t_\epsilon)$', fontsize=14)
   
   T_label = f'{T_f}' if T_f is not None else 'T'
   ax.set_ylabel(
       rf'$\log_{{10}}\left(\max_{{t_\epsilon \leq \tau \leq {T_label}}} E_{{{s}}}(\tau) \, (\%)\right)  $',
       fontsize=13
   )
   
   ax.grid(True, linestyle=':', alpha=0.6)
   ax.legend(loc='best', fontsize=12)
   
   plt.tight_layout()
 
    
    
   p, c = np.polyfit(log_t_eps, log_E, 1)
     
   print(f"Approximate slope p = {p:.4f}")
   print(f"Approximate constant C = {10**(c)/100:.4f}") 
     # ----- Plot the fitted slope line over the data -----
   fit_x = np.linspace(log_t_eps.min(), log_t_eps.max(), 100)
   fit_y = p * fit_x + c
    
   ax.plot(fit_x, fit_y,
   'k--',
    linewidth=1.8,
    label=rf'approximate slope $p = {p:.3f}$')
       
    # ----- Annotate the slope value on the plot -----
   mid = len(fit_x) // 2
   ax.annotate(rf'$slope \approx {p:.3f}$',
   xy=(fit_x[mid], fit_y[mid]),
     xytext=(fit_x[mid] - 0.5, fit_y[mid] + 0.3),
    fontsize=12, color='black',
    arrowprops=dict(arrowstyle='->', color='black'))
   plt.savefig(f'Es_SW_s{s}_vs_teps_loglog.pdf', format='pdf',
                bbox_inches='tight', pad_inches=0.1)
    
   plt.show()
























# def plot_static_eps(x,y,N_x,N_y,zeta_choosen_eps1,zeta_choosen_static,A_typical, t_eps1, model):
#     X, Y = np.meshgrid(x, y, indexing="ij")

#     x_mid_idx, y_mid_idx = N_x // 2, N_y // 2

#     fig4, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 4.5))
    
#     # (a) Slice along X direction
#     ax5.plot(x, A_typical*zeta_choosen_static[:, y_mid_idx],"b-",label=rf"Static {model}" )
#     ax5.plot(x, A_typical*zeta_choosen_eps1[:, y_mid_idx],"r--",label=rf" $t_\epsilon = {t_eps1:.2f}$ " )
#     ax5.set_title(f"Slice at y={int(y[y_mid_idx])} ")
#     ax5.set_xlabel("x")
#     ax5.set_ylabel(r" Surface elevations $\;$(m)")
#     ax5.grid(True, linestyle=":")
#     ax5.legend()

#     # (b) Slice along Y direction
#     ax6.plot(y, A_typical*zeta_choosen_static[x_mid_idx, :], "b-",label=rf"Static {model}" )
#     ax6.plot(y, A_typical*zeta_choosen_eps1[x_mid_idx, :], "r--",label=rf" $t_\epsilon = {t_eps1:.2f}$ " )
#     ax6.set_title(f"Slice at x={int(x[x_mid_idx])} " )
#     ax6.grid(True, linestyle=":")
#     ax6.set_xlabel("y")
#     ax5.set_ylabel(r" Surface elevations $\;$(m)")
#     ax6.legend()

#     plt.tight_layout()
#     plt.savefig("my_slices_statuc_time_2s.pdf", 
#             format="pdf", 
#             bbox_inches="tight",  # Ensures no labels get cut off
#             pad_inches=0.1        # Adds a tiny bit of breathing room around the plot
#            )
#     plt.show()


# def plot_Euler_and_azeta_app(x,y,N_x,N_y,zeta_choosen,zeta_analytic,B,A_typical, model):
#     X, Y = np.meshgrid(x, y, indexing="ij")

#     x_mid_idx, y_mid_idx = N_x // 2, N_y // 2

#     fig4, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 4.5))
    
#     # (a) Slice along X direction
#     ax5.plot(x, A_typical*zeta_analytic[:, y_mid_idx],"b-",label=r"Linear Euler" )
#     ax5.plot(x, A_typical*zeta_choosen[:, y_mid_idx],"r--",label=rf"{model} model" )
#     #ax5.plot(x, -1 + A_typical * B[:, y_mid_idx], color='black', linestyle='-', label=r"Bottom")
#     ax5.set_title(f"Slice at y={int(y[y_mid_idx])} km")
#     ax5.set_xlabel("x(km)")
#     ax5.set_ylabel(r"$\zeta \;$(m)")
#     ax5.grid(True, linestyle=":")
#     ax5.legend()

#     # (b) Slice along Y direction
#     ax6.plot(y, A_typical*zeta_analytic[x_mid_idx, :], "b-",label=r" Linear Euler" )
#     ax6.plot(y, A_typical*zeta_choosen[x_mid_idx, :], "r--",label=rf"{model} model" )
#     #ax6.plot(y,-1+ A_typical*B[x_mid_idx, :],  color='black', linestyle='-', label=r"Bottom")     

#     ax6.set_title(f"Slice at x={int(x[x_mid_idx])} km" )
#     ax6.grid(True, linestyle=":")
#     ax6.set_xlabel("y(km)")
#     ax5.set_ylabel(r"$\zeta \;$(m)")
#     ax6.legend()

#     plt.tight_layout()
#     plt.savefig("my_slices_Euler_and_model.pdf", 
#             format="pdf", 
#             bbox_inches="tight",  # Ensures no labels get cut off
#             pad_inches=0.1        # Adds a tiny bit of breathing room around the plot
#            )
#     plt.show()


# def plot_3d_single(x, y, B, H_typical):
#     X, Y = np.meshgrid(x, y, indexing="ij")
#     fig = plt.figure(figsize=(10, 6))
    
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, H_typical * B, cmap='jet', rstride=4, cstride=4, edgecolor='none')
    
#     #ax.set_title("Bathymetry Profile")
#     ax.set_xlabel("x (km)")
#     ax.set_ylabel("y (km)")
#     ax.zaxis.set_rotate_label(False)
#     ax.set_zlabel(r"$B$ (m)", rotation=0, labelpad=12, fontsize=12)    
#     ax.view_init(20, 2)
    
#     plt.tight_layout()
#     plt.savefig("Bottom_3d.pdf", 
#                 format="pdf", 
#                 bbox_inches="tight",  
#                 pad_inches=0.1         
#                )
#     plt.show()
    
  






# def plot_relative_erro(x, y, t, V1, V2,B, loc):
#   # 1. Find the nearest grid indices for the location
#   idx_x = np.abs(x - loc[0]).argmin()
#   idx_y = np.abs(y - loc[1]).argmin()

#   # 2. Extract time-series data at the location
#   v1 = V1[:, idx_x, idx_y]
#   v2 = V2[:, idx_x, idx_y]
#   B_loc=B[idx_x, idx_y]
#   #print(B_loc)
#   #print(np.max(np.abs(v1 - v2)))
#   # 3. Set up a single plot layout
#   fig, ax = plt.subplots(figsize=(8, 5))

#   # --- Plot: Single Location ---
#   ax.plot(
#       t,
#       np.abs(v1 - v2) / np.max(np.abs(v1)),
#       'k--',
#       linewidth=1.2,
#       label='Relative Error',
#   )

#   ax.set_title(f'Tide gauge at ({loc[0]}, {loc[1]}).')
#   ax.set_xlabel('$t$')
#   ax.set_ylabel('Relative Error')
#   ax.set_xlim(t[0], t[-1])
#   ax.grid(True, linestyle=':', alpha=0.6)
#   ax.legend()

#   plt.tight_layout()
#   plt.savefig(
#       'relative_error_single.pdf',
#       format='pdf',
#       bbox_inches='tight',
#       pad_inches=0.1,
#   )
#   plt.show()
  

# def plot_relative_error_4locations(t_eps, max_errors_dict):
#     """
#     Plot maximum relative error vs t_epsilon on a normal linear scale.
#     x-axis goes from large to small with clear t_eps values.
#     """
#     t_eps = np.asarray(t_eps)
    
#     # Sort the data so x-axis goes from LARGE to SMALL
#     idx = np.argsort(t_eps)[::-1]
#     t_eps_sorted = t_eps[idx]
    
#     colors = ['b', 'r', 'g', 'm']
#     markers = ['^', 's', 'o', 'D']
    
#     fig, ax = plt.subplots(figsize=(10, 5.5))
    
#     for i, (loc_key, max_err) in enumerate(max_errors_dict.items()):
#         max_err = np.asarray(max_err)
#         max_err = max_err[idx]
#         loc_num = loc_key.split('_')[-1]
        
#         ax.plot(
#             t_eps_sorted,
#             100 * max_err,
#             marker=markers[i],
#             markersize=10,
#             markerfacecolor='none',
#             markeredgecolor=colors[i],
#             markeredgewidth=1.5,
#             linewidth=1.5,
#             color=colors[i],
#             label=rf'$p_{{{loc_num}}}$'
#         )
    
#     # ----- X-AXIS WITH CLEAR t_eps VALUES -----
#     ax.set_xlim(3.5, -0.1)
    
#     # Show ALL the actual t_eps values on x-axis
#     ax.set_xticks(t_eps_sorted)
#     ax.set_xticklabels([f'{val:.3f}' for val in t_eps_sorted], rotation=90, ha='right')
    
#     ax.set_xlabel(r'$t_\epsilon$', fontsize=14)
#     ax.set_ylabel(r'$\max_{t_\epsilon \leq \tau \leq T} \mathcal{R}(\tau, X)$ (%)', fontsize=13)
#     # ax.set_title(
#     #     r'Maximum relative error for $t \in (t_\epsilon, T)$',
#     #     fontsize=13
#     # )
    
#     ax.grid(True, linestyle=':', alpha=0.6)
#     ax.legend(loc='best', fontsize=12, ncol=2)
    
#     plt.tight_layout()
    
#     plt.savefig('relative_error_sqrt_NSW.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

#     plt.show()



# def plot_max_Es_Bous_vs_teps(t_eps, max_E_bous_values, s=3, T_f=None):
#     """
#     Plot maximum Boussinesq error vs t_epsilon on a normal linear scale.
#     x-axis goes from large to small with clear t_eps values.
#     """
#     t_eps = np.asarray(t_eps)
    
#     # Sort the data so x-axis goes from LARGE to SMALL
#     idx = np.argsort(t_eps)[::-1]
#     t_eps_sorted = t_eps[idx]
    
#     max_E_bous_values = np.asarray(max_E_bous_values)
#     max_E_bous_values = max_E_bous_values[idx]
    
#     fig, ax = plt.subplots(figsize=(10, 5.5))
    
#     ax.plot(
#         t_eps_sorted,
#         100 * max_E_bous_values,
#         marker='^',
#         markersize=10,
#         markerfacecolor='none',
#         markeredgecolor='b',
#         markeredgewidth=1.5,
#         linewidth=1.5,
#         color='b',
#         label=rf'$E_{{{s}}}^{{\mathrm{{Bous}}}}$'
#     )
    
#     # ----- X-AXIS WITH CLEAR t_eps VALUES -----
#     ax.set_xlim(3.5, -0.1)
#     # Show ALL the actual t_eps values on x-axis
#     ax.set_xticks(t_eps_sorted)
#     ax.set_xticklabels([f'{val:.3f}' for val in t_eps_sorted], rotation=90, ha='right')
    
#     ax.set_xlabel(r'$t_\epsilon$', fontsize=14)
    
#     T_label = f'{T_f}' if T_f is not None else 'T'
#     # ax.set_ylabel(
#     #     rf'$\max_{{t_\epsilon \leq \tau \leq {T_label}}} E_{{{s}}}^{{\mathrm{{Bous}}}}(\tau)$ (%)',
#     #     fontsize=13
#     # )
#     ax.set_ylabel(
#         rf'$\max_{{t_\epsilon \leq \tau \leq {T_label}}} E_{{{s}}}(\tau)$ (%)',
#         fontsize=13
#     )
    
    
#     ax.grid(True, linestyle=':', alpha=0.6)
#     ax.legend(loc='best', fontsize=12)
    
#     plt.tight_layout()
#     plt.savefig(f'Es_NSW_s{s}_vs_teps.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
#     plt.show()
    
