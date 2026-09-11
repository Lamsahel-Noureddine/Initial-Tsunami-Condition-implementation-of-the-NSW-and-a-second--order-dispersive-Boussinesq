#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:56:38 2026

@author: lamsahel
"""

import numpy as np
import matplotlib.pyplot as plt

xi = np.linspace(0.01, 6.0, 1000)   # xi = |k| H_0

# Nondimensional phase speed squared: C^2 = C_dim^2 / (g H_0)
C2_euler = np.tanh(xi) / xi
C2_nsw   = np.ones_like(xi)
C2_bouss = 1.0 / (1.0 + xi**2 / 3.0)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(xi, C2_euler, 'k-',  linewidth=2.2, label='Euler')
ax.plot(xi, C2_nsw,   'r--', linewidth=1.8, label='Shallow Water')
ax.plot(xi, C2_bouss, 'b-.', linewidth=1.8, label='Boussinesq')

ax.set_xlabel(r'$\sqrt{\sigma}|k|$', fontsize=13)
ax.set_ylabel(r'$C^2$', fontsize=12)
ax.set_title('Nondimensional Phase Velocity\n', fontsize=12, fontweight='bold')
ax.set_xlim(0, 6)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)

  
# # Optional: annotate the formulas
# ax.text(3.5, 0.75,
#         r'$C^2_{\rm Euler}=\dfrac{\tanh\xi}{\xi}$' + '\n'
#         r'$C^2_{\rm NSW}=1$' + '\n'
#         r'$C^2_{\rm Bouss}=\dfrac{1}{1+\xi^2/3}$',
#         fontsize=11, verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

  
plt.tight_layout()
plt.savefig('nondimensional_dispersion.pdf', format='pdf', bbox_inches='tight')
plt.show()