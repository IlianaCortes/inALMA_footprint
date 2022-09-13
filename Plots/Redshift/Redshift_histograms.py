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

non_Full = ascii.read('non_Full.txt')
non_Clean = ascii.read('non_Clean.txt')
non_Final = ascii.read('non_Final.txt')


ps_Full = ascii.read('ps_Full.txt')
ps_Clean = ascii.read('ps_Clean.txt')
ps_Final = ascii.read('ps_Final.txt')


z_non_Full = non_Full['REDSHIFT']
z_non_Clean = non_Clean['REDSHIFT']
z_non_Final = non_Final['REDSHIFT']

z_ps_Full = ps_Full['REDSHIFT']
z_ps_Clean = ps_Clean['REDSHIFT']
z_ps_Final = ps_Final['REDSHIFT']


L_non_Full = non_Full['LOG_LBOL']
L_non_Clean = non_Clean['LOG_LBOL']
L_non_Final = non_Final['LOG_LBOL']

L_ps_Full = ps_Full['LOG_LBOL']
L_ps_Clean = ps_Clean['LOG_LBOL']
L_ps_Final = ps_Final['LOG_LBOL']



# ------------------------------ CIV - MGII --------------------------------- #

non_CIV = ascii.read('non_CIV.txt')['REDSHIFT']
non_MgII = ascii.read('non_MgII.txt')['REDSHIFT']

ps_CIV = ascii.read('ps_CIV.txt')['REDSHIFT']
ps_MgII = ascii.read('ps_MgII.txt')['REDSHIFT']
# non_CIV = non_CIV['REDSHIFT']

bins_CIV=np.histogram(np.hstack((non_CIV, ps_CIV)), bins=10)[1]



CIV_z = ascii.read('CIV_z.txt')
zi_CIV = CIV_z['LOW']
w_CIV = CIV_z['HIGH']-CIV_z['LOW']
z_non_CIV = CIV_z['t8_CIV_final_COUNT'] #nonDetected
z_ps_CIV = CIV_z['t10_CIV_final_COUNT']


MgII_z = ascii.read('MgII_z.txt')
zi_MgII = MgII_z['LOW']
w_MgII = MgII_z['HIGH']-MgII_z['LOW']
z_non_MgII = MgII_z['t8_MgII_final_COUNT']
z_ps_MgII = MgII_z['t10_MgII_final_COUNT']

# =============================================================================
# KS test
# =============================================================================

z_Full_ks = stats.ks_2samp(z_non_Full, z_ps_Full)[0]
z_Clean_ks = stats.ks_2samp(z_non_Clean, z_ps_Clean)[0]
z_Final_ks = stats.ks_2samp(z_non_Final, z_ps_Final)[0]


z_Full_p = stats.ks_2samp(z_non_Full, z_ps_Full)[1]
z_Clean_p = stats.ks_2samp(z_non_Clean, z_ps_Clean)[1]
z_Final_p = stats.ks_2samp(z_non_Final, z_ps_Final)[1]



L_Full_ks = stats.ks_2samp(L_non_Full, L_ps_Full)[0]
L_Clean_ks = stats.ks_2samp(L_non_Clean, L_ps_Clean)[0]
L_Final_ks = stats.ks_2samp(L_non_Final, L_ps_Final)[0]


L_Full_p = stats.ks_2samp(L_non_Full, L_ps_Full)[1]
L_Clean_p = stats.ks_2samp(L_non_Clean, L_ps_Clean)[1]
L_Final_p = stats.ks_2samp(L_non_Final, L_ps_Final)[1]

print('Full z')
print(z_Full_ks)
print(z_Full_p)
print(' ')
print('Clean z')
print(z_Clean_ks)
print(z_Clean_p)
print(' ')
print('Final z')
print(z_Final_ks)
print(z_Final_p)
print(' ')
print(' ')
print('Full L')
print(L_Full_ks)
print(L_Full_p)
print(' ')
print('Clean L')
print(L_Clean_ks)
print(L_Clean_p)
print(' ')
print('Final L')
print(L_Final_ks)
print(L_Final_p)



# =============================================================================
# Histogram
# =============================================================================


bins_non=np.histogram(np.hstack((z_non_Full, z_non_Clean, z_non_Final)), bins=20)[1]
bins_ps=np.histogram(np.hstack((z_ps_Full, z_ps_Clean, z_ps_Final)), bins=20)[1]
# bins_broad=np.histogram(np.hstack((vel_broad_nonDetected_CVI_clean, vel_broad_psDetected_CVI_clean)), bins=10)[1]

bins_L_non = np.histogram(np.hstack((L_non_Full, L_non_Clean, L_non_Final)), \
                          bins=20, range=(43,49))[1]
bins_L_ps = np.histogram(np.hstack((L_ps_Full, L_ps_Clean, L_ps_Final)), \
                         bins=20, range=(43,49))[1]


# # Colors
Color1 = "#490727"
Color2 = "#b34a1d"
Color3 = "#0e253e"
Color4 = "#119a54"
Color5 = "#3b6d0c"


# plt.style.use("classic")




plt.style.use("classic")


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.hist(z_non_Full, bins_non,  color=Color1, alpha=1, label='Full sample',\
#           histtype='bar', density=False)#, cumulative=True)
# ax1.hist(z_non_Clean, bins_non,  color=Color3, alpha=1, label='Clean sample',\
#           histtype='bar', density=False)#, cumulative=True)
# ax1.hist(z_non_Final, bins_non,  color=Color5, alpha=1, label='Final sample',\
#           histtype='bar', density=False)#, cumulative=True)
# ax1.set_xlabel(r'$\mathrm{Redshift}$',fontsize=18)
# ax1.set_facecolor('#F7F7F7')
# ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
# ax1.set_title('nonDetected', fontsize=18)
# # ax.set_xlim(-1000, 3000)
# # ax.set_ylim(0, 1.1)
# plt.tight_layout()
# plt.show()
# fig.savefig('peak_hist.pdf',format="pdf",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )
density_v = False

type_hist= 'bar'
alpha_val = 0.7
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(8)



# weights_v = np.ones(len(z_non_Full))/len(z_non_Full)
# weights_v = np.ones(len(z_non_Full))/len(z_non_Full)
# weights_v = np.ones(len(z_non_Full))/len(z_non_Full)

# nonDetected
ax1 = plt.subplot2grid(shape=(4, 1), loc=(0, 0), colspan=1, rowspan=2)
weights_v = np.ones(len(z_non_Full))/len(z_non_Full)
# weights_v = None
ax1.hist(z_non_Full, bins_non,  color=Color1, alpha=alpha_val, label=r'$\mathrm{Full\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(z_non_Clean))/len(z_non_Full)
# weights_v = None
ax1.hist(z_non_Clean, bins_non,  color=Color3, alpha=alpha_val, label=r'$\mathrm{Clean\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(z_non_Final))/len(z_non_Full)
# weights_v = None
ax1.hist(z_non_Final, bins_non,  color=Color4, alpha=alpha_val, label=r'$\mathrm{Final\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
# ax1.set_xlabel(r'$\mathrm{Redshift}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax1.set_title(r'$\mathrm{nonDetected}$', fontsize=18)
plt.setp(ax1.get_xticklabels(), visible=False)
# ax.set_xlim(-1000, 3000)
# ax.set_ylim(0, 1.1)
# psDetected
ax2 = plt.subplot2grid(shape=(4, 1), loc=(2, 0), colspan=1, rowspan=2, sharex=ax1)
weights_v = np.ones(len(z_ps_Full))/len(z_ps_Full)
ax2.hist(z_ps_Full, bins_ps,  color=Color1, alpha=alpha_val, label=r'$\mathrm{Full\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(z_ps_Clean))/len(z_ps_Full)
ax2.hist(z_ps_Clean, bins_ps,  color=Color3, alpha=alpha_val, label=r'$\mathrm{Clean\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(z_ps_Final))/len(z_ps_Full)
ax2.hist(z_ps_Final, bins_ps,  color=Color4, alpha=alpha_val, label=r'$\mathrm{Final\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
ax2.set_xlabel(r'$\mathrm{Redshift}$',fontsize=18)
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax2.set_title(r'$\mathrm{psDetected}$', fontsize=18)
# ax.set_xlim(-1000, 3000)
# ax.set_ylim(0, 1.1)
fig.supylabel(r'$\mathrm{Fraction\ of\ objects}$', fontsize=18)
plt.tight_layout()
plt.show()
fig.savefig('redshift_hist.pdf',format="pdf",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )




type_hist= 'bar'
alpha_val = 0.7
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(8)
# nonDetected
weights_v = np.ones(len(L_non_Full))/len(L_non_Full)
ax1 = plt.subplot2grid(shape=(4, 1), loc=(0, 0), colspan=1, rowspan=2)
ax1.hist(L_non_Full, bins_L_non,  color=Color1, alpha=alpha_val, label=r'$\mathrm{Full\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(L_non_Clean))/len(L_non_Full)
ax1.hist(L_non_Clean, bins_L_non,  color=Color3, alpha=alpha_val, label=r'$\mathrm{Clean\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(L_non_Final))/len(L_non_Full)
ax1.hist(L_non_Final, bins_L_non,  color=Color4, alpha=alpha_val, label=r'$\mathrm{Final\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
# ax1.set_xlabel(r'$\mathrm{Redshift}$',fontsize=18)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax1.set_title(r'$\mathrm{nonDetected}$', fontsize=18)
plt.setp(ax1.get_xticklabels(), visible=False)
# ax.set_xlim(-1000, 3000)
# ax.set_ylim(0, 1.1)
# psDetected
ax2 = plt.subplot2grid(shape=(4, 1), loc=(2, 0), colspan=1, rowspan=2, sharex=ax1)
weights_v = np.ones(len(L_ps_Full))/len(L_ps_Full)
ax2.hist(L_ps_Full, bins_L_ps,  color=Color1, alpha=alpha_val, label=r'$\mathrm{Full\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(L_ps_Clean))/len(L_ps_Full)
ax2.hist(L_ps_Clean, bins_L_ps,  color=Color3, alpha=alpha_val, label=r'$\mathrm{Clean\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
weights_v = np.ones(len(L_ps_Final))/len(L_ps_Full)
ax2.hist(L_ps_Final, bins_L_ps,  color=Color4, alpha=alpha_val, label=r'$\mathrm{Final\ sample}$',\
          histtype=type_hist, density = density_v, weights = weights_v)#, cumulative=True)
ax2.set_xlabel(r'$\mathrm{\log L_{BOL}\ [erg^{}s^{-1}]}$',fontsize=18)
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax2.set_title(r'$\mathrm{psDetected}$', fontsize=18)
# ax.set_xlim(-1000, 3000)
# ax.set_ylim(0, 1.1)
fig.supylabel(r'$\mathrm{Fraction\ of\ objects}$', fontsize=18)
plt.tight_layout()
plt.show()
fig.savefig('luminosity_hist.pdf',format="pdf",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )




fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(8)
ax1 = plt.subplot2grid(shape=(4, 1), loc=(0, 0), colspan=1, rowspan=2)
ax1.bar(zi_CIV, z_non_CIV, w_CIV, label=r'$\mathrm{nonDetected}$', color=Color3, alpha=0.7)
ax1.bar(zi_CIV, z_ps_CIV, w_CIV, label=r'$\mathrm{psDetected}$', color=Color4, alpha=0.7)
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax1.set_title(r'$\mathrm{C\ IV}$', fontsize=18)
ax1.set_xlim(0, 6)
ax1.set_ylim(0,1.1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot2grid(shape=(4, 1), loc=(2, 0), colspan=1, rowspan=2, sharex=ax1)#, sharey=ax1)
ax2.bar(zi_MgII, z_non_MgII, w_MgII, label=r'$\mathrm{nonDetected}$', color=Color3, alpha=0.7)
ax2.bar(zi_MgII, z_ps_MgII, w_MgII, label=r'$\mathrm{psDetected}$', color=Color4, alpha=0.7)
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax2.set_title(r'$\mathrm{C\ IV}$', fontsize=18)
ax2.set_xlim(0, 6)
ax2.set_ylim(0,1.1)
ax2.set_xlabel(r'$\mathrm{Redshift}$',fontsize=18)
ax2.set_facecolor('#F7F7F7')
ax2.set_title(r'$\mathrm{Mg\ II}$', fontsize=18)
fig.supylabel(r'$\mathrm{Normalized\ counts}$', fontsize=18)
plt.tight_layout()
plt.show()
fig.savefig('redshift_CIV_MgII.pdf',format="pdf",dpi=300,pad_inches = 0,\
            bbox_inches='tight' )





