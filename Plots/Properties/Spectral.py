#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:55:16 2022

@author: iliana
"""

# =============================================================================
# Packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import simps
import scipy.constants as ct
from astropy.table import QTable, Table, Column
from astropy.io import ascii
from scipy import stats






# =============================================================================
# Datos
# =============================================================================



CIV_non = ascii.read('DR14Q_nonDetected_CIV_triple_gaussian_fit.dat')
CIV_ps = ascii.read('DR14Q_psDetected_CIV_triple_gaussian_fit.dat')
MgII_non = ascii.read('DR14Q_nonDetected_MgII_triple_gaussian_fit.dat')
MgII_ps = ascii.read('DR14Q_psDetected_MgII_triple_gaussian_fit.dat')


# Peak velocity

v_peak_CIV_non = []
v_broad_CIV_non = []
R2_CIV_non = []
EW_CIV_non = []
L1350_non = []
dL1350_non = []
LEdd_CIV_non = []
dLEdd_CIV_non = []

for i in range(len(CIV_non)):
    if CIV_non['g_no'][i]!=0:
        v_peak_CIV_non.append(CIV_non['vel_peak'][i])
        v_broad_CIV_non.append(CIV_non['vel_broad'][i])
        R2_CIV_non.append(CIV_non['R2'][i])
        EW_CIV_non.append(CIV_non['EW_T'][i])
        L1350_non.append(CIV_non['LOG_L1350'][i])
        dL1350_non.append(CIV_non['LOG_L1350_ERR'][i])
        LEdd_CIV_non.append(CIV_non['LOG_MBH'][i] + np.log10(3e4))
        dLEdd_CIV_non.append(CIV_non['LOG_MBH_ERR'][i])
    


v_peak_CIV_ps = []
v_broad_CIV_ps = []
R2_CIV_ps = []
EW_CIV_ps = []
L1350_ps = []
dL1350_ps = []
LEdd_CIV_ps = []
dLEdd_CIV_ps = []

for i in range(len(CIV_ps)):
    if CIV_ps['g_no'][i]!=0:
        v_peak_CIV_ps.append(CIV_ps['vel_peak'][i])
        v_broad_CIV_ps.append(CIV_ps['vel_broad'][i])
        R2_CIV_ps.append(CIV_ps['R2'][i])
        EW_CIV_ps.append(CIV_ps['EW_T'][i])
        L1350_ps.append(CIV_ps['LOG_L1350'][i])
        dL1350_ps.append(CIV_ps['LOG_L1350_ERR'][i])
        LEdd_CIV_ps.append(CIV_ps['LOG_MBH'][i] + np.log10(3e4))
        dLEdd_CIV_ps.append(CIV_ps['LOG_MBH_ERR'][i])
        
        
v_peak_MgII_non = []
v_broad_MgII_non = []
R2_MgII_non = []
EW_MgII_non = []
L3000_non = []
dL3000_non = []
LEdd_MgII_non = []
dLEdd_MgII_non = []

for i in range(len(MgII_non)):
    if MgII_non['g_no'][i]!=0:
        v_peak_MgII_non.append(MgII_non['vel_peak'][i])
        v_broad_MgII_non.append(MgII_non['vel_broad'][i])
        R2_MgII_non.append(MgII_non['R2'][i])
        EW_MgII_non.append(MgII_non['EW_T'][i])
        L3000_non.append(MgII_non['LOG_L1350'][i])
        dL3000_non.append(MgII_non['LOG_L1350_ERR'][i])
        LEdd_MgII_non.append(MgII_non['LOG_MBH'][i] + np.log10(3e4))
        dLEdd_MgII_non.append(MgII_non['LOG_MBH_ERR'][i])



v_peak_MgII_ps = []
v_broad_MgII_ps = []
R2_MgII_ps = []
EW_MgII_ps = []
L3000_ps = []
dL3000_ps = []
LEdd_MgII_ps = []
dLEdd_MgII_ps = []

for i in range(len(MgII_ps)):
    if MgII_ps['g_no'][i]!=0:
        v_peak_MgII_ps.append(MgII_ps['vel_peak'][i])
        v_broad_MgII_ps.append(MgII_ps['vel_broad'][i])
        R2_MgII_ps.append(MgII_ps['R2'][i])
        EW_MgII_ps.append(MgII_ps['EW_T'][i])
        L3000_ps.append(MgII_ps['LOG_L1350'][i])
        dL3000_ps.append(MgII_ps['LOG_L1350_ERR'][i])
        LEdd_MgII_ps.append(MgII_ps['LOG_MBH'][i] + np.log10(3e4))
        dLEdd_MgII_ps.append(MgII_ps['LOG_MBH_ERR'][i])
        
        


bins_CIV_peak=np.histogram(np.hstack((v_peak_CIV_non, v_peak_CIV_ps)),\
                           bins=10, range=(-500, 4000))[1]

bins_MgII_peak=np.histogram(np.hstack((v_peak_MgII_non, v_peak_MgII_ps)),\
                            bins=10, range=(-500, 2000))[1]

# Broad velocity

bins_CIV_broad=np.histogram(np.hstack((v_broad_CIV_non, v_broad_CIV_ps)),\
                            bins=10, range=(-4000,4000))[1]

bins_MgII_broad=np.histogram(np.hstack((v_broad_MgII_non, v_broad_MgII_ps)),\
                             bins=10, range=(-4000,4000))[1]


bins_R2=np.histogram(np.hstack((R2_CIV_non, R2_CIV_ps, R2_MgII_non, R2_MgII_ps)),\
                     bins=20, range=(0,1))[1]



    


# =============================================================================
# Histogram
# =============================================================================





# # Colors
Color1 = "#490727"
Color2 = "#b34a1d"
Color3 = "#0e253e"
Color4 = "#119a54"
Color5 = "#3b6d0c"


# plt.style.use("classic")




plt.style.use("classic")






    
    
    
hist_type = 'step'
alpha_val = 1
range_R2 = (0,1)
density_R2 = True
cumulative_R2 = True
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(R2_CIV_non, bins_R2,  color=Color1, alpha=alpha_val,\
          histtype=hist_type, density=density_R2, range=range_R2, cumulative=cumulative_R2,\
              edgecolor=Color1, linewidth=3)
ax1.hist(R2_CIV_ps, bins_R2,  color=Color3, alpha=alpha_val,\
          histtype=hist_type, density=density_R2, range=range_R2, cumulative=cumulative_R2,\
              edgecolor=Color3, linewidth=3)
ax1.hist(R2_MgII_non, bins_R2,  color=Color2, alpha=alpha_val,\
          histtype=hist_type, density=density_R2, range=range_R2, cumulative=cumulative_R2,\
              edgecolor=Color2, linewidth=3)
ax1.hist(R2_MgII_ps, bins_R2,  color=Color4, alpha=alpha_val,\
          histtype=hist_type, density=density_R2, range=range_R2, cumulative=cumulative_R2,\
              edgecolor=Color4, linewidth=3)
ax1.axvline(x=np.median(R2_CIV_non), color=Color1, linestyle='--', linewidth=1.5)#, label='C IV (nonDetected)')
ax1.axvline(x=np.median(R2_CIV_ps), color=Color3, linestyle='--', linewidth=1.5)#, label='C IV (psDetected)')
ax1.axvline(x=np.median(R2_MgII_non), color=Color2, linestyle='--', linewidth=1.5)#, label='Mg II (nonDetected)')
ax1.axvline(x=np.median(R2_MgII_ps), color=Color4, linestyle='--', linewidth=1.5)#, label='Mg II (psDetected)')
# ax1.set_xscale('log')
ax1.axvline(x=10, color=Color1, linestyle='-', linewidth=3, label='C IV (nonDetected)')
ax1.axvline(x=10, color='white', linestyle='-', linewidth=0, label='Median: 0.91')
ax1.axvline(x=10, color=Color3, linestyle='-', linewidth=3, label='C IV (psDetected)')
ax1.axvline(x=10, color='white', linestyle='-', linewidth=0, label='Median: 0.89')
ax1.axvline(x=10, color=Color2, linestyle='-', linewidth=3, label='Mg II (nonDetected)')
ax1.axvline(x=10, color='white', linestyle='-', linewidth=0, label='Median: 0.52')
ax1.axvline(x=10, color=Color4, linestyle='-', linewidth=3, label='Mg II (psDetected)')
ax1.axvline(x=10, color='white', linestyle='-', linewidth=0, label='Median: 0.85')
ax1.set_xlabel(r'$\mathrm{R^2}$',fontsize=18)
ax1.set_ylabel(r'$\mathrm{Fraction\ of\ objects}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='upper left', fontsize = 12, numpoints=1)#, prop={'size': 12})
# ax1.set_title('nonDetected', fontsize=18)
ax1.set_xlim(0,1)
ax1.set_ylim(0, 1)
plt.tight_layout()
plt.show()
fig.savefig('R2_hist.jpg',format="jpg",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )

    
    
    
hist_type = 'bar'
alpha_val = 0.7
range_v_peak = None#(0,1)
density_v_peak = False
cumulative_v_peak = False
w_peak_non = np.ones(len(v_peak_CIV_non))/len(v_peak_CIV_non)
w_peak_ps = np.ones(len(v_peak_CIV_ps))/len(v_peak_CIV_ps)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(v_peak_CIV_non, bins_CIV_peak,  color=Color3, alpha=alpha_val, label='nonDetected',\
          histtype=hist_type, density=density_v_peak, range=range_v_peak,\
              cumulative=cumulative_v_peak, weights=w_peak_non)
ax1.hist(v_peak_CIV_ps, bins_CIV_peak,  color=Color4, alpha=alpha_val, label='psDetected',\
          histtype=hist_type, density=density_v_peak, range=range_v_peak,\
              cumulative=cumulative_v_peak, weights=w_peak_ps)
ax1.set_xlabel(r'$\mathrm{Velocity\ km^{}s^{-1}}$',fontsize=18)
ax1.set_ylabel(r'$\mathrm{Fraction\ of\ objects}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
ax1.set_title('Peak (C IV)', fontsize=18)
# ax.set_xlim(-1000, 3000)
# ax1.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()
fig.savefig('v_peak_CIV.jpg',format="jpg",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )


hist_type = 'bar'
alpha_val = 0.7
range_v_broad = None#(0,1)
density_v_broad = False
cumulative_v_broad = False
w_broad_non = np.ones(len(v_broad_CIV_non))/len(v_broad_CIV_non)
w_broad_ps = np.ones(len(v_broad_CIV_ps))/len(v_broad_CIV_ps)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(v_broad_CIV_non, bins_CIV_broad,  color=Color3, alpha=alpha_val, label='nonDetected',\
          histtype=hist_type, density=density_v_broad, range=range_v_broad,\
              cumulative=cumulative_v_broad, weights=w_broad_non)
ax1.hist(v_broad_CIV_ps, bins_CIV_broad,  color=Color4, alpha=alpha_val, label='psDetected',\
          histtype=hist_type, density=density_v_broad, range=range_v_broad,\
              cumulative=cumulative_v_broad, weights=w_broad_ps)
ax1.set_xlabel(r'$\mathrm{Velocity\ km^{}s^{-1}}$',fontsize=18)
ax1.set_ylabel(r'$\mathrm{Fraction\ of\ objects}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
ax1.set_title('Broad (C IV)', fontsize=18)
# ax.set_xlim(-1000, 3000)
# ax1.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()
fig.savefig('v_broad_CIV.jpg',format="jpg",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )




hist_type = 'bar'
alpha_val = 0.7
range_v_peak = None#(0,1)
density_v_peak = False
cumulative_v_peak = False
w_peak_non = np.ones(len(v_peak_MgII_non))/len(v_peak_MgII_non)
w_peak_ps = np.ones(len(v_peak_MgII_ps))/len(v_peak_MgII_ps)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(v_peak_MgII_non, bins_MgII_peak,  color=Color3, alpha=alpha_val, label='nonDetected',\
          histtype=hist_type, density=density_v_peak, range=range_v_peak,\
              cumulative=cumulative_v_peak, weights=w_peak_non)
ax1.hist(v_peak_MgII_ps, bins_MgII_peak,  color=Color4, alpha=alpha_val, label='psDetected',\
          histtype=hist_type, density=density_v_peak, range=range_v_peak,\
              cumulative=cumulative_v_peak, weights=w_peak_ps)
ax1.set_xlabel(r'$\mathrm{Velocity\ km^{}s^{-1}}$',fontsize=18)
ax1.set_ylabel(r'$\mathrm{Fraction\ of\ objects}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
ax1.set_title('Peak (Mg II)', fontsize=18)
# ax.set_xlim(-1000, 3000)
# ax1.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()
fig.savefig('v_peak_MgII.jpg',format="jpg",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )


hist_type = 'bar'
alpha_val = 0.7
range_v_broad = None#(0,1)
density_v_broad = False
cumulative_v_broad = False
w_broad_non = np.ones(len(v_broad_MgII_non))/len(v_broad_MgII_non)
w_broad_ps = np.ones(len(v_broad_MgII_ps))/len(v_broad_MgII_ps)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(v_broad_MgII_non, bins_MgII_broad,  color=Color3, alpha=alpha_val, label='nonDetected',\
          histtype=hist_type, density=density_v_broad, range=range_v_broad,\
              cumulative=cumulative_v_broad, weights=w_broad_non)
ax1.hist(v_broad_MgII_ps, bins_MgII_broad,  color=Color4, alpha=alpha_val, label='psDetected',\
          histtype=hist_type, density=density_v_broad, range=range_v_broad,\
              cumulative=cumulative_v_broad, weights=w_broad_ps)
ax1.set_xlabel(r'$\mathrm{Velocity\ km^{}s^{-1}}$',fontsize=18)
ax1.set_ylabel(r'$\mathrm{Fraction\ of\ objects}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
ax1.set_title('Broad (Mg II)', fontsize=18)
# ax.set_xlim(-1000, 3000)
# ax1.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()
fig.savefig('v_broad_MgII.jpg',format="jpg",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )




# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.errorbar(L1350_non, EW_CIV_non, fmt='o', color=Color3,\
#              label='nonDetected', mec=Color3)
# ax1.errorbar(L1350_ps, EW_CIV_ps, fmt='o', color=Color4,\
#              label='psDetected', mec=Color4)
# ax1.set_xlabel(r'$\mathrm{\log\ L1350\ [L_\odot]}$',fontsize=18)
# ax1.set_ylabel(r'$\mathrm{EW(C\ IV)}$',fontsize=18)
# ax1.set_facecolor('#F7F7F7')
# ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
# # ax1.set_title('nonDetected', fontsize=18)
# ax1.set_xlim(35,55)
# # ax1.set_ylim(-300, 300)
# plt.tight_layout()
# plt.show()
# fig.savefig('CIV_L1350.jpg',format="jpg",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.errorbar(LEdd_CIV_non, EW_CIV_non, fmt='o', color=Color3,\
#              label='nonDetected', mec=Color3)
# ax1.errorbar(LEdd_CIV_ps, EW_CIV_ps, fmt='o', color=Color4,\
#              label='psDetected', mec=Color4)
# ax1.set_xlabel(r'$\mathrm{\log\ L1350\ [L_\odot]}$',fontsize=18)
# ax1.set_ylabel(r'$\mathrm{EW(C\ IV)}$',fontsize=18)
# ax1.set_facecolor('#F7F7F7')
# ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
# # ax1.set_title('nonDetected', fontsize=18)
# ax1.set_xlim(0, 20)
# # ax1.set_ylim(-300, 300)
# plt.tight_layout()
# plt.show()
# fig.savefig('CIV_LEdd.jpg',format="jpg",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )



# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.errorbar(L3000_non, EW_MgII_non, fmt='o', color=Color3,\
#              label='nonDetected', mec=Color3)
# ax1.errorbar(L3000_ps, EW_MgII_ps, fmt='o', color=Color4,\
#              label='psDetected', mec=Color4)
# ax1.set_xlabel(r'$\mathrm{\log\ L3000\ [L_\odot]}$',fontsize=18)
# ax1.set_ylabel(r'$\mathrm{EW(C\ IV)}$',fontsize=18)
# ax1.set_facecolor('#F7F7F7')
# ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
# # ax1.set_title('nonDetected', fontsize=18)
# ax1.set_xlim(35,55)
# # ax1.set_ylim(-300, 300)
# plt.tight_layout()
# plt.show()
# fig.savefig('MgII_L3000.jpg',format="jpg",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.errorbar(LEdd_MgII_non, EW_MgII_non, fmt='o', color=Color3,\
#              label='nonDetected', mec=Color3)
# ax1.errorbar(LEdd_MgII_ps, EW_MgII_ps, fmt='o', color=Color4,\
#              label='psDetected', mec=Color4)
# ax1.set_xlabel(r'$\mathrm{\log\ L3000\ [L_\odot]}$',fontsize=18)
# ax1.set_ylabel(r'$\mathrm{EW(C\ IV)}$',fontsize=18)
# ax1.set_facecolor('#F7F7F7')
# ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.legend(loc='best', fontsize = 12, numpoints=1)#, prop={'size': 12})
# # ax1.set_title('nonDetected', fontsize=18)
# ax1.set_xlim(0,20)
# # ax1.set_ylim(-300, 300)
# plt.tight_layout()
# plt.show()
# fig.savefig('MgII_LEdd.jpg',format="jpg",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )





# =============================================================================
# MADDOX
# =============================================================================
hist_type = 'step'
alpha_val = 1
range_v_peak = None#(0,1)
density_v_peak = False
cumulative_v_peak = False
w_peak_non = np.ones(len(v_peak_CIV_non))/len(v_peak_CIV_non)
w_peak_ps = np.ones(len(v_peak_CIV_ps))/len(v_peak_CIV_ps)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.hist(v_peak_CIV_non, bins_CIV_peak,  color=Color3, alpha=alpha_val, label='nonDetected',\
#           histtype=hist_type, density=density_v_peak, range=range_v_peak,\
#               cumulative=cumulative_v_peak, weights=w_peak_non)

fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(8)
ax1 = plt.subplot2grid(shape=(4, 1), loc=(0, 0), colspan=1, rowspan=3)
ax1.hist(v_peak_CIV_non, bins_CIV_peak,  color=Color3, alpha=alpha_val, label='nonDetected',\
          histtype=hist_type, density=True, range=range_v_peak,\
              cumulative=True, edgecolor=Color3, linewidth=2)#, weights=w_peak_non)
ax1.hist(v_peak_CIV_ps, bins_CIV_peak,  color=Color4, alpha=alpha_val, label='psDetected',\
          histtype=hist_type, density=True, range=range_v_peak,\
              cumulative=True, edgecolor=Color4, linewidth=2)#, weights=w_peak_ps)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='upper left', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax1.set_ylabel('$\mathrm{Cumulative\ fraction}$', fontsize=18)
# ax1.set_title(r'$\mathrm{C\ IV}$', fontsize=18)
# ax1.set_xlim(0, 6)
# ax1.set_ylim(0,1.1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot2grid(shape=(4, 1), loc=(3, 0), colspan=1, rowspan=1, sharex=ax1)
ax2.hist(v_peak_CIV_non, bins_CIV_peak,  color=Color3, alpha=alpha_val, label='nonDetected',\
          histtype=hist_type, density=False, range=range_v_peak,\
              cumulative=False, edgecolor=Color3, linewidth=2, weights=w_peak_non)
ax2.hist(v_peak_CIV_ps, bins_CIV_peak,  color=Color4, alpha=alpha_val, label='psDetected',\
          histtype=hist_type, density=False, range=range_v_peak,\
              cumulative=False, edgecolor=Color4, linewidth=2, weights=w_peak_ps)
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
# ax2.set_title(r'$\mathrm{C\ IV}$', fontsize=18)
# ax2.set_xlim(0, 6)
ax2.set_ylim(0,0.18)
ax2.set_yticklabels([])
ax2.set_xlabel(r'$\mathrm{Velocity\ km^{}s^{-1}}$',fontsize=18)
ax2.set_facecolor('#F7F7F7')
# ax2.set_title(r'$\mathrm{Mg\ II}$', fontsize=18)
plt.tight_layout()
plt.show()
fig.savefig('redshift_CIV_MgII.jpg',format="jpg",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )
