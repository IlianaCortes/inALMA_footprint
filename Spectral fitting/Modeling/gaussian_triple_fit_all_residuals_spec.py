# This program fits three gaussian functions to the line line,

"""
Created on Mon Aug  8 15:33:53 2022

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
from astropy import modeling
from scipy import integrate
from scipy.integrate import simps
from astropy.modeling import models, fitting
import warnings
import scipy.constants as ct
from astropy.table import QTable, Table, Column
from astropy.io import ascii
import warnings
from astropy.stats import sigma_clip
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline
from uncertainties import ufloat

warnings.filterwarnings("ignore")

# =============================================================================
# Useful constants
# =============================================================================


spectral_lines = ascii.read('spectral_lines.dat')
spectral_lines.add_index('Species')
c = ct.c / 1000 # Speed of light [km/s]



wl0_OVI = spectral_lines.loc['OVI'][1]
wl0_Lya = spectral_lines.loc['Lya'][1]
wl0_NV = spectral_lines.loc['NV'][1]
wl0_OIa = spectral_lines.loc['OI'][0][1]
wl0_CIIa = spectral_lines.loc['CII'][0][1]
wl0_SiIV = spectral_lines.loc['SiIV'][1]
wl0_SiIV_OIV = spectral_lines.loc['SiIV+OIV'][1]
wl0_CIV = spectral_lines.loc['CIV'][1]
wl0_HeII = spectral_lines.loc['HeII'][1]
wl0_OIIIa = spectral_lines.loc['OIII'][0][1]
wl0_AlIII = spectral_lines.loc['AlIII'][1]
wl0_CIII = spectral_lines.loc['CIII'][1]
wl0_CIIb = spectral_lines.loc['CII'][1][1]
wl0_NeIV = spectral_lines.loc['NeIV'][1]
wl0_MgII = spectral_lines.loc['MgII'][1]
wl0_NeV = spectral_lines.loc['NeV'][1]
wl0_NeVI = spectral_lines.loc['NeVI'][1]
wl0_OIIa = spectral_lines.loc['OII'][0][1]
wl0_OIIb = spectral_lines.loc['OII'][1][1]
wl0_HeI = spectral_lines.loc['HeI'][1]
wl0_SIIa = spectral_lines.loc['SII'][0][1]
wl0_Hd = spectral_lines.loc['Hd'][1]
wl0_Hg = spectral_lines.loc['Hg'][1]
wl0_OIIIb = spectral_lines.loc['OIII'][1][1]
wl0_Hb = spectral_lines.loc['Hb'][1]
wl0_OIIIc = spectral_lines.loc['OIII'][2][1]
wl0_OIIId = spectral_lines.loc['OIII'][3][1]
wl0_OIIIe = spectral_lines.loc['OIII'][4][1]
wl0_OIb = spectral_lines.loc['OI'][1][1]
wl0_OIc = spectral_lines.loc['OI'][2][1]
wl0_NI = spectral_lines.loc['NI'][1]
wl0_NIIa = spectral_lines.loc['NII'][0][1]
wl0_Ha = spectral_lines.loc['Ha'][1]
wl0_NIIb = spectral_lines.loc['NII'][1][1]
wl0_SIIb = spectral_lines.loc['SII'][1][1]
wl0_SIIc = spectral_lines.loc['SII'][2][1]


# Estimated redshift
z = 0.0

# wl0_line = wl0_MgII*(1+z)
wl0_line = wl0_CIV*(1+z)

# line rest frame wavelength - Same as Rakshit (2020)
range_low = 1500 # CIV
range_high = 1600 # CIV

# range_low = 2700 # Mg II
# range_high = 2900 # Mg II


li_line = range_low*(1+z)
lf_line = range_high*(1+z)


# =============================================================================
# Models from Rakshit
# =============================================================================

# Sample


filename = "DR14Q_nonDetected_CIV"
# filename = "DR14Q_psDetected_CIV"
# filename = "DR14Q_nonDetected_MgII"
# filename = "DR14Q_psDetected_MgII"
    
    
data = ascii.read(filename+'.txt')

plate = data['PLATE']
mjd = data['MJD']
fiber = data['FIBERID']

# plate2, mjd2, fiber2 = np.loadtxt(filename+'.txt', unpack=True, skiprows=1)

plate = plate.astype(int).astype(str)
mjd = mjd.astype(int).astype(str)
fiber = fiber.astype(int).astype(str)


# Spectral properties from Rakshit20
RA = data['RA_1_1']
DEC = data['DEC_1_1']
REDSHIFT = data['REDSHIFT']
LOG_L1350 = data['LOG_L1350']
LOG_L1350_ERR = data['LOG_L1350_ERR']
LOG_L3000 = data['LOG_L3000']
LOG_L3000_ERR = data['LOG_L3000_ERR']
LOG_MBH = data['LOG_MBH']
LOG_MBH_ERR = data['LOG_MBH_ERR']
Band = data['Band']
Cont_sens = data['Cont._sens.']


plate_f = np.char.zfill(plate, 4)
mjd_f = np.char.zfill(mjd, 5)
fiber_f = np.char.zfill(fiber, 4)

sdss_id_dr14 = []

for i in range(len(plate_f)):
    sdss_id_dr14.append(plate_f[i]+'-'+mjd_f[i]+'-'+fiber_f[i]+'.fits')
    
    

# Read files with astropy
model1 = []
model = []
# Model components
model_cont = []
flux = []
wl = []
flux_c = []
flux_err = []


for i in range(len(sdss_id_dr14)):
    # Model data
    model1.append(fits.open(sdss_id_dr14[i]))
    model.append(model1[i][1].data)
    # Continuum
    model_cont.append(model[i]['model_pl_bc']+model[i]['model_feii'])
    # Check if host galaxy is decomposed: 
    if np.max(model[i]['model_host']>0):
        flux.append(model[i]['flux']-model[i]['model_host'])
    else:
        flux.append(model[i]['flux'])
    # Flux with substracted continuum
    flux_c.append(flux[i]-model_cont[i])
    # Wavelength
    wl.append(model[i]['wave'])
    flux_err.append(model[i]['err'])
    



for i in range(len(sdss_id_dr14)):
    print(sdss_id_dr14[i], i)


# =============================================================================
# Spectral line isolation
# =============================================================================






ii_low = []
ii_high = []

wl_line = []
flux_line = []
err_line = []

for i in range(len(sdss_id_dr14)):
    # Find indexes 
    ii_low.append((np.abs(wl[i]-li_line)).argmin())
    ii_high.append((np.abs(wl[i]-lf_line)).argmin())
    wl_line.append(wl[i][ii_low[i]:ii_high[i]])
    flux_line.append(flux_c[i][ii_low[i]:ii_high[i]])
    err_line.append(flux_err[i][ii_low[i]:ii_high[i]])
    




# =============================================================================
# Functions
# =============================================================================

def gauss(x, A, x0, sigma):
    '''
    Gaussian function
    Input:
        x: wavelength array
        A: amplitude of the distribution
        x0: central wavelength
        sigma: standard deviation
    Output:
        y0 + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    '''
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))




param = 9

def gauss3_init (x,y):
    av = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - av) ** 2) / sum(y))
    amp = max(y)-min(y)
    base = min(y)
    g1 = models.Gaussian1D(amplitude=amp/3, mean=av, stddev=sigma)#, bounds={'mean':(li_line, lf_line)})
    g1.amplitude.bounds = (0, amp)
    # g1.amplitude.min = 0
    g1.mean.bounds = (li_line, lf_line)
    g1.stddev.min = 0
    g2 = models.Gaussian1D(amplitude=amp/3, mean=av, stddev=sigma)#, bounds={'mean':(li_line, lf_line)})
    g2.amplitude.bounds = (0, amp)
    # g2.amplitude.min = 0
    g2.mean.bounds = (li_line, lf_line)
    g2.stddev.min = 0
    g3 = models.Gaussian1D(amplitude=amp/5, mean=av, stddev=sigma)#, bounds={'mean':(li_line, lf_line)})
    g3.amplitude.bounds = (0, amp)
    # g3.amplitude.min = 0
    g3.mean.bounds = (li_line, lf_line)
    g3.stddev.min = 0
    # c = modeling.models.Const1D(base)
    # c.amplitude.min = 0
    guess = g1 + g2 + g3 #+ c
    return guess

def gauss1_init (x,y):
    av = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - av) ** 2) / sum(y))
    amp = max(y)-min(y)
    base = min(y)
    g1 = models.Gaussian1D(amplitude=amp/3, mean=av, stddev=sigma)
    g1.amplitude.bounds = (0, amp)
    g1.mean.bounds = (li_line, lf_line)
    g1.stddev.min = 0
    return g1


def gauss3_fit(x,y):
    comp_mod = gauss3_init(x,y)
    # fitter = fitting.LevMarLSQFitter()
    fitter = fitting.LevMarLSQFitter()
    # with warnings.catch_warnings():
    #     # Ignore a warning on clipping to bounds from the fitter
    #     warnings.filterwarnings('ignore', message='Values in x were outside bounds',
    #                             category=RuntimeWarning)
    return fitter(comp_mod, x, y)




def gauss1_fit(x,y):
    gaussian = gauss1_init(x,y)
    # fitter = fitting.LevMarLSQFitter()
    fitter = fitting.LevMarLSQFitter()
    # with warnings.catch_warnings():
    #     # Ignore a warning on clipping to bounds from the fitter
    #     warnings.filterwarnings('ignore', message='Values in x were outside bounds',
    #                             category=RuntimeWarning)
    return fitter(gaussian, x, y)



def integrate(x, y):
    '''
    Area under a curve
    Input:
        x: wavelength array
        y: curve
    Output:
        area: integration in the interval of x
    '''
    area = simps(y=y, x=x)
    return area

def FWHM_f(sigma):
    '''
    Full Width Half Maximum
    Input:
        sigma: standard deviation
    Output:
        FWHM_f: FWHM
    '''
    return 2*np.sqrt(2*np.log(2)) * sigma


def FWHM_spline(x, y):
    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots()
    return np.abs(r2-r1)

def EW_f(x, y):
    '''
    Equivalent Width
    Input:
        x: wavelength array
        y: gaussian distribution of the spectral line
        cont: continuum level
    Output:
        EW_f: EW
    '''
    area_curve = integrate(x,y)
    height = np.max(y)
    return area_curve / height

def velocity(x0, x):
    '''
    Velocity
    Input:
        x0: wavelength in the rest frame
        x: central wavelength
    Output:
        velocity
    '''
    return c*(x0 - x) / x0


def chi2_f(O, E, sigma, p):
    '''
    Chi square
    Input:
        O = Observed value
        E = Expected value (fit)
        p = numer of fitted parameters
    Output:
        x_n = Reduced chi-squared
        
    '''
    ddof = len(E) - p # Degrees of freedom
    sum_x = np.sum(((O-E)/sigma)**2)
    x_n = sum_x / ddof
    return x_n

def R2_f(O, E):
    '''
    R-squared
    Input:
        O = Observed value
        E = Expected value (fit)
    Output:
        1 - RSS/TSS = Coefficient of determination
    '''
    num = (O-E)**2
    den = (O-np.median(O))**2
    RSS = np.sum(num) # Sum of squares of residuals
    TSS = np.sum(den) # Total sum of squares
    out = 1 - RSS/TSS
    return out

def error_f(value, R2, data):
    y = value
    dy = y*np.sqrt(np.abs(1/R2-1)/(len(data)-2))
    return dy



# =============================================================================
# Results 
# =============================================================================


fit_arr = []
A1 = []
wl1 = []
sigma1 = []
FWHM1 = []
EW1 = []
Gauss1 = []
A2 = []
wl2 = []
sigma2 = []
FWHM2 = []
EW2 = []
Gauss2 = []
A3 = []
wl3 = []
sigma3 = []
FWHM3 = []
EW3 = []
Gauss3 = []
Gauss_T = []
EW_T = []
FWHM_T = []




    
for i in range(len(sdss_id_dr14)):
    fit_arr.append(gauss3_fit(wl_line[i], flux_line[i]))
    # Constant
    # f0.append(fit_arr[i][3].amplitude)
    # Gauss 1
    A1.append(fit_arr[i][0].amplitude.value)
    wl1.append(fit_arr[i][0].mean.value)
    sigma1.append(fit_arr[i][0].stddev.value)
    Gauss1.append(gauss(wl_line[i], A1[i], wl1[i], sigma1[i]))
    FWHM1.append(FWHM_f(sigma1[i]))
    # FWHM1.append(FWHM_f(wl_line[i], Gauss1[i]))
    EW1.append(EW_f(wl_line[i], Gauss1[i]))
    # Gauss 2
    A2.append(fit_arr[i][1].amplitude.value)
    wl2.append(fit_arr[i][1].mean.value)
    sigma2.append(fit_arr[i][1].stddev.value)
    Gauss2.append(gauss(wl_line[i], A2[i], wl2[i], sigma2[i]))
    FWHM2.append(FWHM_f(sigma2[i]))
    # FWHM2.append(FWHM_f(wl_line[i], Gauss2[i]))
    EW2.append(EW_f(wl_line[i], Gauss2[i]))
    # Gauss 3
    A3.append(fit_arr[i][2].amplitude.value)
    wl3.append(fit_arr[i][2].mean.value)
    sigma3.append(fit_arr[i][2].stddev.value)
    Gauss3.append(gauss(wl_line[i], A3[i], wl3[i], sigma3[i]))
    FWHM3.append(FWHM_f(sigma1[i]))
    # FWHM3.append(FWHM_f(wl_line[i], Gauss3[i]))
    EW3.append(EW_f(wl_line[i], Gauss3[i]))
    # Total gauss
    Gauss_T.append(Gauss1[i] + Gauss2[i] + Gauss3[i])
    EW_T.append(EW_f(wl_line[i], Gauss_T[i]))
    # FWHM_T.append(FWHM_f(wl_line[i], Gauss_T[i]))
    try:
        FWHM_T.append(FWHM_spline(wl_line[i], Gauss_T[i]))
    except :
        FWHM_T.append(0)



chi2 = []
R2 = []



for i in range(len(sdss_id_dr14)):
    chi2.append(chi2_f(flux_line[i], Gauss_T[i], err_line[i], param))
    R2.append(R2_f(flux_line[i], Gauss_T[i]))
    

dA1 = []
dA2 = []
dA3 = []
dwl1 = []
dwl2 = []
dwl3 = []
dsigma1 = []
dsigma2 = []
dsigma3 = []
# dFWHM1 = []
# dFWHM2 = []
# dFWHM3 = []
# dFWHMT = []
# dEW1 = []
# dEW2 = []
# dEW3 = []
# dEWT = []



for i in range(len(sdss_id_dr14)):
    # Amplitude
    dA1.append(error_f(A1[i], R2[i], flux_line[i]))
    dA2.append(error_f(A2[i], R2[i], flux_line[i]))
    dA3.append(error_f(A3[i], R2[i], flux_line[i]))
    # Standard deviation
    dwl1.append(error_f(wl1[i], R2[i], flux_line[i]))
    dwl2.append(error_f(wl2[i], R2[i], flux_line[i]))
    dwl3.append(error_f(wl3[i], R2[i], flux_line[i]))
    # Standard deviation
    dsigma1.append(error_f(sigma1[i], R2[i], flux_line[i]))
    dsigma2.append(error_f(sigma2[i], R2[i], flux_line[i]))
    dsigma3.append(error_f(sigma3[i], R2[i], flux_line[i]))
    
    


# for i in range(len(sdss_id_dr14)):
#     A1[i] = ufloat(A1[i], dA1[i])
#     wl1[i] = ufloat(wl1[i], dwl1[i])
#     sigma1[i] = ufloat(sigma1[i], dsigma1[i])
#     A2[i] = ufloat(A2[i], dA2[i])
#     wl2[i] = ufloat(wl2[i], dwl2[i])
#     sigma2[i] = ufloat(sigma2[i], dsigma2[i])
#     A3[i] = ufloat(A3[i], dA3[i])
#     wl3[i] = ufloat(wl3[i], dwl3[i])
#     sigma3[i] = ufloat(sigma3[i], dsigma3[i])




# =============================================================================
# Number of gaussians
# =============================================================================

g_no = []
idx_g_no = []

for i in range(len(sdss_id_dr14)):
    # Three gaussians
    if (A1[i]>0 and A2[i]>0 and A3[i]>0\
        and sigma1[i]>0 and sigma2[i]>0 and sigma3[i]>0):
        g_no.append(3)
        idx_g_no.append(i)
    # Two gaussians
    elif ((A1[i]==0 or sigma1[i]==0) and\
        (A2[i]>0 and A3[i]>0 and sigma2[i]>0 and sigma3[i]>0)):
        g_no.append(2)
        idx_g_no.append(i)
    elif ((A2[i]==0 or sigma2[i]==0) and\
        (A1[i]>0 and A3[i]>0 and sigma1[i]>0 and sigma3[i]>0)):
        g_no.append(2)
        idx_g_no.append(i)
    elif ((A3[i]==0 or sigma3[i]==0) and\
        (A2[i]>0 and A1[i]>0 and sigma2[i]>0 and sigma1[i]>0)):
        g_no.append(2)
        idx_g_no.append(i)
    # One gaussian
    elif ((A1[i]>0 and sigma1[i]>0) and\
        ((A2[i]==0 or sigma2[i]==0) and (A3[i]==0 or sigma3[i]==0))):
        g_no.append(1)
        idx_g_no.append(i)    
    elif ((A2[i]>0 and sigma2[i]>0) and\
        ((A1[i]==0 or sigma1[i]==0) and (A3[i]==0 or sigma3[i]==0))):
        g_no.append(1)
        idx_g_no.append(i)  
    elif ((A3[i]>0 and sigma3[i]>0) and\
        ((A2[i]==0 or sigma2[i]==0) and (A1[i]==0 or sigma1[i]==0))):
        g_no.append(1)
        idx_g_no.append(i)
    else :
        g_no.append(0)
        idx_g_no.append(i)
        


idx_inc = []
id_inc = []
idx_exc = []
id_exc = []

for i in range(len(sdss_id_dr14)):
    if (g_no[i]==2 or g_no[i]==3):
        idx_inc.append(i)
        id_inc.append(sdss_id_dr14[i])
    else :
        idx_exc.append(i)
        id_exc.append(sdss_id_dr14[i])     
        



# =============================================================================
# Sorted data - First fit
# =============================================================================


# # Elements sorted by broadness

# A1_s = []
# wl1_s = []
# sigma1_s = []
# FWHM1_s = []
# EW1_s = []
# Gauss1_s = []
# A2_s = []
# wl2_s = []
# sigma2_s = []
# FWHM2_s = []
# EW2_s = []
# Gauss2_s = []
# A3_s = []
# wl3_s = []
# sigma3_s = []
# FWHM3_s = []
# EW3_s = []
# Gauss3_s = []



# for i in range(len(sdss_id_dr14)):
#     if (g_no[i]==3):
#         if ((sigma2[i]<sigma3[i]<sigma1[i]) or ((sigma3[i]<sigma2[i]<sigma1[i]))) :
#             wl_broad.append(wl1[i])
#             vel_broad.append(velocity(wl_T[i], wl1[i]))
#         elif ((sigma1[i]<sigma3[i]<sigma2[i]) or ((sigma3[i]<sigma1[i]<sigma2[i]))) :
#             vel_broad.append(velocity(wl_T[i], wl2[i]))
#             wl_broad.append(wl2[i])
#         elif ((sigma1[i]<sigma2[i]<sigma3[i]) or ((sigma2[i]<sigma1[i]<sigma3[i]))) :
#             vel_broad.append(velocity(wl_T[i], wl3[i]))
#             wl_broad.append(wl3[i])
#     elif (g_no[i]==2):
#         if (A1[i]==0 or sigma1[i]==0):



# =============================================================================
# Velocity
# =============================================================================
        

idx_wl = []
wl_T = []
vel_peak = []





for i in range(len(sdss_id_dr14)):
    # Wavelength of the peak
    idx_wl.append((np.abs(Gauss_T[i]-np.max(Gauss_T[i]))).argmin())
    wl_T.append(np.max(wl_line[i][idx_wl[i]]))
    vel_peak.append(velocity(wl0_line, wl_T[i]))        
        
        
vel_broad = []
wl_broad = []


for i in range(len(sdss_id_dr14)):
    if (g_no[i]==3):
        if ((sigma2[i]<sigma3[i]<sigma1[i]) or ((sigma3[i]<sigma2[i]<sigma1[i]))) :
            wl_broad.append(wl1[i])
            vel_broad.append(velocity(wl_T[i], wl1[i]))
        elif ((sigma1[i]<sigma3[i]<sigma2[i]) or ((sigma3[i]<sigma1[i]<sigma2[i]))) :
            vel_broad.append(velocity(wl_T[i], wl2[i]))
            wl_broad.append(wl2[i])
        elif ((sigma1[i]<sigma2[i]<sigma3[i]) or ((sigma2[i]<sigma1[i]<sigma3[i]))) :
            vel_broad.append(velocity(wl_T[i], wl3[i]))
            wl_broad.append(wl3[i])
    elif (g_no[i]==2):
        if (A1[i]==0 or sigma1[i]==0):
            if (sigma2[i]>sigma3[i]):
                vel_broad.append(velocity(wl_T[i], wl2[i]))
                wl_broad.append(wl2[i])
            elif (sigma3[i]>sigma2[i]):
                vel_broad.append(velocity(wl_T[i], wl3[i]))
                wl_broad.append(wl3[i])
        elif (A2[i]==0 or sigma2[i]==0):
            if (sigma1[i]>sigma3[i]):
                vel_broad.append(velocity(wl_T[i], wl1[i]))
                wl_broad.append(wl1[i])
            elif (sigma3[i]>sigma1[i]):
                vel_broad.append(velocity(wl_T[i], wl3[i]))
                wl_broad.append(wl3[i])
        elif (A3[i]==0 or sigma3[i]==0):
            if (sigma2[i]>sigma1[i]):
                vel_broad.append(velocity(wl_T[i], wl2[i]))
                wl_broad.append(wl2[i])
            elif (sigma1[i]>sigma2[i]):
                vel_broad.append(velocity(wl_T[i], wl1[i]))
                wl_broad.append(wl1[i])
    else :
        vel_broad.append(-9999)
        wl_broad.append(0)

    # print(i, sdss_id_dr14[i])
    # if Gauss_T[i].all() == 0:
    #     FWHM_T.append(0)
    # else :
    #     FWHM_T.append(FWHM_spline(wl_line[i], Gauss_T[i]))
    
# for i in range(len(sdss_id_dr14)):
#     print(i)



# =============================================================================
# Residuals
# =============================================================================


Gauss_T_res = list(Gauss_T)


fit_1g = []
A_1g = []
wl_1g = []
sigma_1g = []
Gauss_1g = []




    
for i in range(len(sdss_id_dr14)):
    fit_1g.append(gauss1_fit(wl_line[i], flux_line[i]))
    # Constant
    # f0.append(fit_arr[i][3].amplitude)
    # Gauss 1
    A_1g.append(fit_1g[i].amplitude.value)
    wl_1g.append(fit_1g[i].mean.value)
    sigma_1g.append(fit_1g[i].stddev.value)
    Gauss_1g.append(gauss(wl_line[i], A_1g[i], wl_1g[i], sigma_1g[i]))



for i in range(len(sdss_id_dr14)):
    if g_no[i]==0 :
        # Gauss_T[i][:] = 0#np.median(flux_line[i])
        Gauss_T_res[i] = Gauss_1g[i]#np.median(flux_line[i])






    # A1.append(fit_arr[i][0].amplitude.value)
    # wl1.append(fit_arr[i][0].mean.value)
    # sigma1.append(fit_arr[i][0].stddev.value)
    # FWHM1.append(FWHM_f(sigma1[i]))
    # Gauss1.append(gauss(wl_line[i], A1[i], wl1[i], sigma1[i]))
    # EW1.append(EW_f(wl_line[i], Gauss1[i]))
        
    
# fit1g = []
# for i in range(len(sdss_id_dr14)):
#     if g_no[i]==0 :
#         fit1g.append(gauss1_fit(wl_line[i], flux_line[i]))
#         print(i, fit1g[i])

res = []

for i in range(len(sdss_id_dr14)):
    res.append(flux_line[i]-Gauss_T_res[i])




def line_fit(x, y):
    fit = fitting.LinearLSQFitter(calc_uncertainties=True)
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma_lower=2, sigma_upper=5)
    line_init = models.Const1D()
    return (or_fit(line_init, x, y))
    




fitted_ct = []
mask = []
wl_mask = []
fl_mask = []
err_mask = []


for i in range(len(sdss_id_dr14)):
    fitted_ct.append(line_fit(wl_line[i], res[i])[0].amplitude.value)
    mask.append(line_fit(wl_line[i], res[i])[1])
    wl_mask.append(np.ma.masked_array(wl_line[i], mask=mask[i]))
    # wl_mask.append(np.asarray(list(filter(None, np.ma.masked_array(wl_line[i], mask=mask[i])))))
    fl_mask.append(np.ma.masked_array(flux_line[i], mask=mask[i]))
    # fl_mask.append(np.asarray(list(filter(None, np.ma.masked_array(flux_line[i], mask=mask[i])))))
    err_mask.append(np.ma.masked_array(err_line[i], mask=mask[i]))


# x = list()



for i in range(len(sdss_id_dr14)):
    wl_mask[i] = np.asarray(list(filter(None, wl_mask[i])))
    fl_mask[i] = np.asarray(list(filter(None, fl_mask[i])))
    err_mask[i] = np.asarray(list(filter(None, err_mask[i])))


# fl_new = []

# for i in range(len(sdss_id_dr14)):
#     fl_new.append(interp1d(wl_mask[i],fl_mask[i],fill_value="extrapolate")(wl_line[i]))
    


# for i in range(len(idx_exc)):
#     wl_low.append(wl_line[low_idx[i]])
#     fl_low.append(flux_line[idx_exc[i]][low_idx[i]])
#     flux_low.append(interp1d(wl_low[i],fl_low[i],fill_value="extrapolate")(wl_line))
#     wl_high.append(wl_line[high_idx[i]])
#     fl_high.append(flux_line[idx_exc[i]][high_idx[i]])
#     flux_high.append(interp1d(wl_high[i],fl_high[i],fill_value="extrapolate")(wl_line))
#     flux_mean.append((flux_low[i]+flux_low[i])/2.)




fit_arr_r = []
A1_r = []
wl1_r = []
sigma1_r = []
FWHM1_r = []
EW1_r = []
Gauss1_r = []
A2_r = []
wl2_r = []
sigma2_r = []
FWHM2_r = []
EW2_r = []
Gauss2_r = []
A3_r = []
wl3_r = []
sigma3_r = []
FWHM3_r = []
EW3_r = []
Gauss3_r = []
Gauss_T_r = []
EW_T_r = []
FWHM_T_r = []

Gauss1_r_plot = []
Gauss2_r_plot = []
Gauss3_r_plot = []
Gauss_T_r_plot = []

res_r = []

    
for i in range(len(sdss_id_dr14)):
    # fit_arr_r.append(gauss3_fit(wl_line[i], fl_new[i]))
    fit_arr_r.append(gauss3_fit(wl_mask[i], fl_mask[i]))
    # Constant
    # f0_r.append(fit_arr_r[i][3].amplitude)
    # Gauss 1
    A1_r.append(fit_arr_r[i][0].amplitude.value)
    wl1_r.append(fit_arr_r[i][0].mean.value)
    sigma1_r.append(fit_arr_r[i][0].stddev.value)
    Gauss1_r.append(gauss(wl_mask[i], A1_r[i], wl1_r[i], sigma1_r[i]))
    Gauss1_r_plot.append(gauss(wl_line[i], A1_r[i], wl1_r[i], sigma1_r[i]))
    EW1_r.append(EW_f(wl_mask[i], Gauss1_r[i]))
    FWHM1_r.append(FWHM_f(sigma1_r[i]))
    # FWHM1_r.append(FWHM_f(wl_mask[i], Gauss1_r[i]))
    # Gauss 2
    A2_r.append(fit_arr_r[i][1].amplitude.value)
    wl2_r.append(fit_arr_r[i][1].mean.value)
    sigma2_r.append(fit_arr_r[i][1].stddev.value)
    Gauss2_r.append(gauss(wl_mask[i], A2_r[i], wl2_r[i], sigma2_r[i]))
    Gauss2_r_plot.append(gauss(wl_line[i], A2_r[i], wl2_r[i], sigma2_r[i]))
    EW2_r.append(EW_f(wl_mask[i], Gauss2_r[i]))
    # FWHM2_r.append(FWHM_f(wl_mask[i], Gauss2_r[i]))
    FWHM2_r.append(FWHM_f(sigma2_r[i]))
    # Gauss 3
    A3_r.append(fit_arr_r[i][2].amplitude.value)
    wl3_r.append(fit_arr_r[i][2].mean.value)
    sigma3_r.append(fit_arr_r[i][2].stddev.value)
    Gauss3_r.append(gauss(wl_mask[i], A3_r[i], wl3_r[i], sigma3_r[i]))
    Gauss3_r_plot.append(gauss(wl_line[i], A3_r[i], wl3_r[i], sigma3_r[i]))
    EW3_r.append(EW_f(wl_mask[i], Gauss3_r[i]))
    FWHM3_r.append(FWHM_f(sigma3_r[i]))
    # FWHM3_r.append(FWHM_f(wl_mask[i], Gauss3_r[i]))
    # Total gauss
    Gauss_T_r.append(Gauss1_r[i] + Gauss2_r[i] + Gauss3_r[i])
    Gauss_T_r_plot.append(Gauss1_r_plot[i] + Gauss2_r_plot[i] + Gauss3_r_plot[i])
    res_r.append(flux_line[i]-Gauss_T_r_plot[i])
    EW_T_r.append(EW_f(wl_mask[i], Gauss_T_r[i]))
    # FWHM_T_r.append(FWHM_f(wl_mask[i], Gauss_T_r[i]))
    try :
        FWHM_T_r.append(FWHM_spline(wl_line[i], Gauss_T[i]))
    except :
        FWHM_T_r.append(0)



chi2_r = []
R2_r = []

for i in range(len(sdss_id_dr14)):
    chi2_r.append(chi2_f(fl_mask[i], Gauss_T_r[i], err_mask[i], param))
    R2_r.append(R2_f(fl_mask[i], Gauss_T_r[i]))
    


# Numer of gaussians

g_no_r = []
idx_g_no_r = []

for i in range(len(sdss_id_dr14)):
    # Three gaussians
    if (A1_r[i]>0 and A2_r[i]>0 and A3_r[i]>0\
        and sigma1_r[i]>0 and sigma2_r[i]>0 and sigma3_r[i]>0):
        g_no_r.append(3)
        idx_g_no_r.append(i)
    # Two gaussians
    elif ((A1_r[i]==0 or sigma1_r[i]==0) and\
        (A2_r[i]>0 and A3_r[i]>0 and sigma2_r[i]>0 and sigma3_r[i]>0)):
        g_no_r.append(2)
        idx_g_no_r.append(i)
    elif ((A2_r[i]==0 or sigma2_r[i]==0) and\
        (A1_r[i]>0 and A3_r[i]>0 and sigma1_r[i]>0 and sigma3_r[i]>0)):
        g_no_r.append(2)
        idx_g_no_r.append(i)
    elif ((A3_r[i]==0 or sigma3_r[i]==0) and\
        (A2_r[i]>0 and A1_r[i]>0 and sigma2_r[i]>0 and sigma1_r[i]>0)):
        g_no_r.append(2)
        idx_g_no_r.append(i)
    # One gaussian
    elif ((A1_r[i]>0 and sigma1_r[i]>0) and\
        ((A2_r[i]==0 or sigma2_r[i]==0) and (A3_r[i]==0 or sigma3_r[i]==0))):
        g_no_r.append(1)
        idx_g_no_r.append(i)    
    elif ((A2_r[i]>0 and sigma2_r[i]>0) and\
        ((A1_r[i]==0 or sigma1_r[i]==0) and (A3_r[i]==0 or sigma3_r[i]==0))):
        g_no_r.append(1)
        idx_g_no_r.append(i)  
    elif ((A3_r[i]>0 and sigma3_r[i]>0) and\
        ((A2_r[i]==0 or sigma2_r[i]==0) and (A1_r[i]==0 or sigma1_r[i]==0))):
        g_no_r.append(1)
        idx_g_no_r.append(i)
    else :
        g_no_r.append(0)
        idx_g_no_r.append(i)
        



idx_wl_r = []
wl_T_r = []
vel_peak_r = []





for i in range(len(sdss_id_dr14)):
    # Wavelength of the peak
    idx_wl_r.append((np.abs(Gauss_T_r[i]-np.max(Gauss_T_r[i]))).argmin())
    wl_T_r.append(np.max(wl_line[i][idx_wl_r[i]]))
    vel_peak_r.append(velocity(wl0_line, wl_T_r[i]))        
        
        
vel_broad_r = []
wl_broad_r = []


for i in range(len(sdss_id_dr14)):
    if (g_no_r[i]==3):
        if ((sigma2_r[i]<sigma3_r[i]<sigma1_r[i]) or ((sigma3_r[i]<sigma2_r[i]<sigma1_r[i]))) :
            wl_broad_r.append(wl1_r[i])
            vel_broad_r.append(velocity(wl_T_r[i], wl1_r[i]))
        elif ((sigma1_r[i]<sigma3_r[i]<sigma2_r[i]) or ((sigma3_r[i]<sigma1_r[i]<sigma2_r[i]))) :
            vel_broad_r.append(velocity(wl_T_r[i], wl2_r[i]))
            wl_broad_r.append(wl2_r[i])
        elif ((sigma1_r[i]<sigma2_r[i]<sigma3_r[i]) or ((sigma2_r[i]<sigma1_r[i]<sigma3_r[i]))) :
            vel_broad_r.append(velocity(wl_T_r[i], wl3_r[i]))
            wl_broad_r.append(wl3_r[i])
    elif (g_no_r[i]==2):
        if (A1_r[i]==0 or sigma1_r[i]==0):
            if (sigma2_r[i]>sigma3_r[i]):
                vel_broad_r.append(velocity(wl_T_r[i], wl2_r[i]))
                wl_broad_r.append(wl2_r[i])
            elif (sigma3_r[i]>sigma2_r[i]):
                vel_broad_r.append(velocity(wl_T_r[i], wl3_r[i]))
                wl_broad_r.append(wl3_r[i])
        elif (A2_r[i]==0 or sigma2_r[i]==0):
            if (sigma1_r[i]>sigma3_r[i]):
                vel_broad_r.append(velocity(wl_T_r[i], wl1_r[i]))
                wl_broad_r.append(wl1_r[i])
            elif (sigma3_r[i]>sigma1_r[i]):
                vel_broad_r.append(velocity(wl_T_r[i], wl3_r[i]))
                wl_broad_r.append(wl3_r[i])
        elif (A3_r[i]==0 or sigma3_r[i]==0):
            if (sigma2_r[i]>sigma1_r[i]):
                vel_broad_r.append(velocity(wl_T_r[i], wl2_r[i]))
                wl_broad_r.append(wl2_r[i])
            elif (sigma1_r[i]>sigma2_r[i]):
                vel_broad_r.append(velocity(wl_T_r[i], wl1_r[i]))
                wl_broad_r.append(wl1_r[i])
    else :
        vel_broad_r.append(-9999)
        wl_broad_r.append(0)





# =============================================================================
# Join results
# =============================================================================


wl_f = list(wl_line)
flux_f = list(flux_line)
A1_f = list(A1)
wl1_f = list(wl1)
sigma1_f = list(sigma1)
FWHM1_f = list(FWHM1)
EW1_f = list(EW1)
A2_f = list(A2)
wl2_f = list(wl2)
sigma2_f = list(sigma2)
FWHM2_f = list(FWHM2)
EW2_f = list(EW2)
A3_f = list(A3)
wl3_f = list(wl3)
sigma3_f = list(sigma3)
FWHM3_f = list(FWHM3)
EW3_f = list(EW3)
EW_T_f = list(EW_T)
g_no_f = list(g_no)
wl_T_f = list(wl_T)
vel_peak_f = list(vel_peak)
wl_broad_f = list(wl_broad)
vel_broad_f = list(vel_broad)
Gauss1_f = list(Gauss1)
Gauss2_f = list(Gauss2)
Gauss3_f = list(Gauss3)
Gauss_T_f = list(Gauss_T)
FWHM_T_f = list(FWHM_T)
chi2_ff = list(chi2)
R2_ff = list(R2)
res_f = list(res)

for i in range(len(sdss_id_dr14)):
    if(R2_r[i] > R2[i]):
        wl_f[i] = wl_mask[i]
        flux_f[i] = fl_mask[i]
        A1_f[i] = A1_r[i]
        wl1_f[i] = wl1_r[i]
        sigma1_f[i] = sigma1_r[i]
        FWHM1_f[i] = FWHM1_r[i]
        EW1_f[i] = EW1_r[i]
        A2_f[i] = A2_r[i]
        wl2_f[i] = wl2_r[i]
        sigma2_f[i] = sigma2_r[i]
        FWHM2_f[i] = FWHM2_r[i]
        EW2_f[i] = EW2_r[i]
        A3_f[i] = A3_r[i]
        wl3_f[i] = wl3_r[i]
        sigma3_f[i] = sigma3_r[i]
        FWHM3_f[i] = FWHM3_r[i]
        EW3_f[i] = EW3_r[i]
        EW_T_f[i] = EW_T_r[i]
        FWHM_T_f[i] = FWHM_T_r[i]
        g_no_f[i] = g_no_r[i]
        wl_T_f[i] = wl_T_r[i]
        vel_peak_f[i] = vel_peak_r[i]
        wl_broad_f[i] = wl_broad_r[i]
        vel_broad_f[i] = vel_broad_r[i]
        Gauss1_f[i] = Gauss1_r_plot[i]
        Gauss2_f[i] = Gauss2_r_plot[i]
        Gauss3_f[i] = Gauss3_r_plot[i]
        Gauss_T_f[i] = Gauss_T_r_plot[i]
        res_f[i] = res_r[i]
        chi2_ff[i] = chi2_r[i]
        R2_ff[i] = R2_r[i]




g0_n = []
g1_n = []
g2_n = []
g3_n = []

for i in range(len(sdss_id_dr14)):
    if (g_no_f[i]==0):
        g0_n.append(i)
    elif (g_no_f[i]==1):
        g1_n.append(i)
    elif (g_no_f[i]==2):
        g2_n.append(i)
    elif (g_no_f[i]==3):
        g3_n.append(i)


# =============================================================================
# Outputs
# =============================================================================

# Table with the results






t = QTable([plate, mjd, fiber, RA, DEC, REDSHIFT, A1_f, wl1_f, sigma1_f, FWHM1_f,\
            EW1_f, A2_f, wl2_f, sigma2_f, FWHM2_f, EW2_f, A3_f, wl3_f,\
                sigma3_f, FWHM3_f, EW3_f, EW_T_f, FWHM_T_f, g_no_f, R2_ff, wl_T_f, vel_peak_f,\
                    wl_broad_f, vel_broad_f, LOG_L1350, LOG_L1350_ERR, LOG_L3000, LOG_L3000_ERR,\
                        LOG_MBH, LOG_MBH_ERR, Band, Cont_sens],\
            names=['plate', 'mjd', 'fiber', 'RA', 'DEC', 'REDSHIFT','A1', 'wl1',\
                  'sigma1', 'FWHM1', 'EW1', 'A2', 'wl2', 'sigma2', 'FWHM2',
                  'EW2', 'A3', 'wl3', 'sigma3', 'FWHM3', 'EW3', 'EW_T', 'FWHM_T', 'g_no',\
                      'R2', 'wl_T', 'vel_peak', 'wl_broad', 'vel_broad', 'LOG_L1350',\
                          'LOG_L1350_ERR', 'LOG_L3000', 'LOG_L3000_ERR', 'LOG_MBH', 'LOG_MBH_ERR',\
                              'Band', 'Cont_sens'])


t['A1'].format = '%.3f'
t['wl1'].format = '%.3f'
t['sigma1'].format = '%.3f'
t['FWHM1'].format = '%.3f'
t['EW1'].format = '%.3f'
t['A2'].format = '%.3f'
t['wl2'].format = '%.3f'
t['sigma2'].format = '%.3f'
t['FWHM2'].format = '%.3f'
t['EW2'].format = '%.3f'
t['A3'].format = '%.3f'
t['wl3'].format = '%.3f'
t['sigma3'].format = '%.3f'
t['FWHM3'].format = '%.3f'
t['EW3'].format = '%.3f'
t['EW_T'].format = '%.3f'
t['wl_T'].format = '%.3f'
t['vel_peak'].format = '%.3f'
t['wl_broad'].format = '%.3f'
t['vel_broad'].format = '%.3f'
    


t.write(filename+'_triple_gaussian_fit.dat', format='ascii.tab', overwrite=True)






# t_exc = QTable([id_exc])
# t_exc.write(filename+'_excluded', format='ascii.no_header', overwrite=True)






# =============================================================================
# Plots
# =============================================================================

# Colors
Color1 = "#490727"
Color2 = "#b34a1d"
Color3 = "#0e253e"
Color4 = "#119a54"
Color5 = "#3b6d0c"


plt.style.use("classic")
# fig = plt.figure()
# ax = fig.add_subplot(111)







for i in range(len(sdss_id_dr14)):
    if (g_no_f[i]!=0):
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(10)
        yloc = np.min(flux[i]) + 8*(np.max(flux[i])-np.min(flux[i]))/10
        #  LINE + TRIPLE GAUSSIAN FIT
        ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2, rowspan=2)#, sharex=ax2) 
        ax1.errorbar(wl_line[i], flux_line[i], fmt='-', color='black', label='$\mathrm{Data}$')
        ax1.plot(wl_line[i], Gauss1_f[i], linestyle='-', linewidth=1, color=Color3, label='$\mathrm{Gaussian\ 1 \ fit}$')
        ax1.plot(wl_line[i], Gauss2_f[i], linestyle='-', linewidth=1, color=Color3, label='$\mathrm{Gaussian\ 2 \ fit}$')
        ax1.plot(wl_line[i], Gauss3_f[i], linestyle='-', linewidth=1, color=Color3, label='$\mathrm{Gaussian\ 3 \ fit}$')
        ax1.errorbar(wl_line[i], Gauss_T_f[i], fmt='-', color=Color4, label='$\mathrm{Fit}$')
        ax1.set_facecolor('#F7F7F7')
        ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
        ax1.set_title(sdss_id_dr14[i][:-5]+'('+str(i)+') '+'(z='+str(REDSHIFT[i])+')',fontsize=24)
        ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
        ax1.set_xlim(li_line, lf_line)
        # RESIDUALS
        ax2 = plt.subplot2grid(shape=(4, 2), loc=(2, 0), colspan=2, rowspan=1, sharex=ax1)
        ax2.errorbar(wl_line[i], res[i], fmt='.', color='grey', label='$\mathrm{Residuals}$')
        ax2.set_facecolor('#F7F7F7')
        ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
        ax2.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$',fontsize=24)
        ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlim(li_line, lf_line)
        # FULL SPECTRUM
        ax3 = plt.subplot2grid(shape=(4, 2), loc=(3, 0), colspan=2, rowspan=1)
        ax3.errorbar(wl[i], flux[i], fmt='-', color='grey', label='$\mathrm{Data}$')
        ax3.axvline(x=wl0_Lya	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_CIV	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_CIII	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_MgII	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_Hg	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_Hb	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_OIIId	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_OIIIe	, color='grey', linestyle='--', clip_on=True)
        ax3.axvline(x=wl0_Ha	, color='grey', linestyle='--', clip_on=True)
        ax3.text(wl0_Lya+25, yloc, '$\mathrm{Lya}$', fontsize=14, clip_on=True)
        ax3.text(wl0_CIV+25, yloc, '$\mathrm{CIV}$', fontsize=14, clip_on=True)
        ax3.text(wl0_CIII+25, yloc, '$\mathrm{CIII]}$', fontsize=14, clip_on=True)
        ax3.text(wl0_MgII+25, yloc, '$\mathrm{MgII}$', fontsize=14, clip_on=True)
        ax3.text(wl0_Hg+25, yloc, r'$\mathrm{H}\gamma$', fontsize=14, clip_on=True)
        ax3.text(wl0_Hb-125, yloc, r'$\mathrm{H}\beta$', fontsize=14, clip_on=True)
        ax3.text(wl0_OIIIe+25, yloc, '$\mathrm{[OIII]}$', fontsize=14, clip_on=True)
        ax3.text(wl0_Ha+25, yloc, r'$\mathrm{H}a$', fontsize=14, clip_on=True)
        ax3.set_facecolor('#F7F7F7')
        ax3.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
        ax3.set_xlabel(r'$\mathrm{Rest\ wavelength\ [\AA]}$',fontsize=24)
        ax3.set_xlim(np.min(wl[i]), np.max(wl[i]))
        plt.tight_layout()
        plt.show()



#   PEAK VELOCITY DISTRIBUTION

fig = plt.figure()
ax = fig.add_subplot(111)
# plt.hist(vel_peak,  color=Color3, alpha=0.7, label='Velocity Peak',\
#           histtype='bar', range=(-1000, 3000))#, density=True)
# plt.hist(vel_peak_r,  color=Color1, alpha=0.7, label='Velocity Peak_r',\
#           histtype='bar', range=(-1000, 3000))#, density=True)
plt.hist(vel_peak_f,  color=Color3, alpha=1, label='Velocity Peak',\
          histtype='bar', range=(-1000, 3000))#, density=True)
ax.set_xlabel(r'$\mathrm{Velocity\ [km/s]}$',fontsize=18)
ax.set_facecolor('#F7F7F7')
ax.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
# ax.set_title(filename, fontsize=18)
ax.set_xlim(-1000, 3000)
# ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()




#   WINGS VELOCITY DISTRIBUTION

fig = plt.figure()
ax = fig.add_subplot(111)
# plt.hist(vel_broad,  color=Color3, alpha=0.7, label='Velocity wings',\
#           histtype='bar', range = (-5000, 5000))#, density=True)
# plt.hist(vel_broad_r,  color=Color1, alpha=.7, label='Velocity wings_r',\
#           histtype='bar', range = (-5000, 5000))#, density=True)
plt.hist(vel_broad_f,  color=Color3, alpha=1, label='Velocity wings',\
          histtype='bar', range = (-5000, 5000))#, density=True)
ax.set_xlabel(r'$\mathrm{Velocity\ [km/s]}$',fontsize=18)
ax.set_facecolor('#F7F7F7')
ax.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
# ax.set_title(filename, fontsize=18)
# ax.set_xlim(-1000, 6500)
# ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()




#   LINE - ALL OBJECTS

# # Gaussian fits
# for i in range(len(sdss_id_dr14)):
#     if (g_no_f[i]!=0):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.errorbar(wl_line[i], flux_line[i], fmt='-', color='grey', label='$\mathrm{Data}$')
#         ax.errorbar(wl_line[i], Gauss1[i], fmt='-', color=Color3)
#         ax.errorbar(wl_line[i], Gauss2[i], fmt='-', color=Color3)
#         ax.errorbar(wl_line[i], Gauss3[i], fmt='-', color=Color3)
#         ax.errorbar(wl_line[i], Gauss_T[i], fmt='-', color=Color4, label='$\mathrm{Fit}$')
#         ax.xaxis.set_major_locator(plt.MultipleLocator(20))
#         ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
#         ax.set_facecolor('#F7F7F7')
#         ax.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
#         ax.set_xlabel(r'$\mathrm{Rest\ wavelength\ [\AA]}$',fontsize=18)
#         ax.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$',fontsize=18)
#         ax.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
#         ax.set_title(sdss_id_dr14[i][:-5]+'('+str(i)+')', fontsize=18)
#         ax.set_xlim(li_line, lf_line)
#         plt.tight_layout()
#         plt.show()
#         # fig.savefig(sdss_id_dr14[i][:-5]+'_line.pdf',format="pdf",dpi=300,pad_inches = 0,\
#         #             bbox_inches='tight' )







#   LINE + TRIPLE FIT 
j = 9


fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(wl_line[j], flux_line[j], fmt = '-', color = 'grey', label = '$\mathrm{Data}$')
ax.errorbar(wl_line[j], Gauss1_f[j], fmt = '-', color = Color5, label = '$\mathrm{Gaussian\ 1}$')
ax.errorbar(wl_line[j], Gauss2_f[j], fmt = '-', color = Color3, label = '$\mathrm{Gaussian\ 2}$')
ax.errorbar(wl_line[j], Gauss3_f[j], fmt = '-', color = Color2, label = '$\mathrm{Gaussian\ 3}$')#Subgrid
ax.errorbar(wl_line[j], Gauss_T_f[j], fmt = '-', color = Color1, label = '$\mathrm{G1+G2+G3}$')
ax.xaxis.set_major_locator(plt.MultipleLocator(20))
ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
ax.set_facecolor('#F7F7F7')
ax.grid(True, color = '#9999993c', linestyle = ':', linewidth = 0.5)
ax.set_xlabel(r'$\mathrm{Rest\ wavelength\ [\AA]}$', fontsize = 18)
ax.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$', fontsize = 18)
ax.legend(loc = 'upper right', fontsize = 14, numpoints = 1)#, prop = {'size':12})
ax.set_title(sdss_id_dr14[j][:-5]+'('+str(j)+')', fontsize = 18)
ax.set_xlim(li_line, lf_line)
plt.tight_layout()
plt.show()
#fig.savefig(sdss_id_dr14[j][:-5]+'_line.pdf', format = "pdf", dpi = 300, pad_inches = 0, \
#bbox_inches = 'tight')


#  LINE + RESIDUALS

fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
# line LINE + TRIPLE GAUSSIAN FIT
ax1 = plt.subplot2grid(shape=(4, 1), loc=(0, 0), colspan=1, rowspan=3)
ax1.errorbar(wl_line[j], flux_line[j], fmt='-', color='black', label='$\mathrm{Data}$')
ax1.plot(wl_line[j], Gauss1_f[j], linestyle='-', linewidth=1, color=Color2, label='$\mathrm{Gaussian\ 1 \ fit}$')
ax1.plot(wl_line[j], Gauss2_f[j], linestyle='-', linewidth=1, color=Color3, label='$\mathrm{Gaussian\ 2 \ fit}$')
ax1.plot(wl_line[j], Gauss3_f[j], linestyle='-', linewidth=1, color=Color1, label='$\mathrm{Gaussian\ 3 \ fit}$')
ax1.errorbar(wl_line[j], Gauss_T_f[j], fmt='-', color=Color4, label='$\mathrm{G1+G2+G3}$')
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.set_title(sdss_id_dr14[j][:-5],fontsize=24)
ax1.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$',fontsize=24)
ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax1.set_xlim(li_line, lf_line)
# RESIDUALS
ax2 = plt.subplot2grid(shape=(4, 1), loc=(3, 0), colspan=1, rowspan=1, sharex=ax1)
ax2.errorbar(wl_line[j], res[j], fmt='-', color='black', label='$\mathrm{Residuals}$')
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax2.set_xlabel(r'$\mathrm{Rest\ wavelength\ [\AA]}$',fontsize=24)
ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set_xlim(li_line, lf_line)
plt.tight_layout()
plt.show()





# LINE + RESIDUALS + FULL SPECTRUM




fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
yloc = np.min(flux[j]) + 8*(np.max(flux[j])-np.min(flux[j]))/10
# line LINE + TRIPLE GAUSSIAN FIT
ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2, rowspan=2)#, sharex=ax2) 
ax1.errorbar(wl_line[j], flux_line[j], fmt='-', color='grey', label='$\mathrm{Data}$')
# ax1.errorbar(wl_mask[j], fl_mask[j], color='black', label='$\mathrm{Data-Filt}$')
ax1.plot(wl_line[j], Gauss1_f[j], linestyle='-', linewidth=1, color=Color3)#, label='$\mathrm{Gaussian\ 1 \ fit}$')
ax1.plot(wl_line[j], Gauss2_f[j], linestyle='-', linewidth=1, color=Color3)#, label='$\mathrm{Gaussian\ 2 \ fit}$')
ax1.plot(wl_line[j], Gauss3_f[j], linestyle='-', linewidth=1, color=Color3)#, label='$\mathrm{Gaussian\ 3 \ fit}$')
ax1.errorbar(wl_line[j], Gauss_T_f[j], fmt='-', color=Color4, label='$\mathrm{Fit}$')
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax1.set_title(sdss_id_dr14[j][:-5],fontsize=24)
ax1.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
ax1.set_xlim(li_line, lf_line)
# RESIDUALS
ax2 = plt.subplot2grid(shape=(4, 2), loc=(2, 0), colspan=2, rowspan=1, sharex=ax1)
ax2.errorbar(wl_line[j], res[j], fmt='.', color='grey', mew=0 , label='$\mathrm{Residuals}$')
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax2.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$',fontsize=24)
ax2.legend(loc='upper right', fontsize = 14, numpoints=1)#, prop={'size': 12})
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set_xlim(li_line, lf_line)
# FULL SPECTRUM
ax3 = plt.subplot2grid(shape=(4, 2), loc=(3, 0), colspan=2, rowspan=1)
ax3.errorbar(wl[j], flux[j], fmt='-', color='grey', label='$\mathrm{Data}$')
# ax3.axvline(x=wl0_Lya, color='grey', linestyle='--', clip_on=True)
# ax3.axvline(x=wl0_CIII, color='grey', linestyle='--', clip_on=True)
# ax3.axvline(x=wl0_MgII, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_Lya	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_CIV	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_CIII	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_MgII	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_Hg	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_Hb	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_OIIId	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_OIIIe	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=wl0_Ha	, color='grey', linestyle='--', clip_on=True)
ax3.text(wl0_Lya+25, yloc, '$\mathrm{Lya}$', fontsize=14, clip_on=True)
ax3.text(wl0_CIV+25, yloc, '$\mathrm{CIV}$', fontsize=14, clip_on=True)
ax3.text(wl0_CIII+25, yloc, '$\mathrm{CIII]}$', fontsize=14, clip_on=True)
ax3.text(wl0_MgII+25, yloc, '$\mathrm{MgII}$', fontsize=14, clip_on=True)
ax3.text(wl0_Hg+25, yloc, r'$\mathrm{H}\gamma$', fontsize=14, clip_on=True)
ax3.text(wl0_Hb-125, yloc, r'$\mathrm{H}\beta$', fontsize=14, clip_on=True)
ax3.text(wl0_OIIIe+25, yloc, '$\mathrm{[OIII]}$', fontsize=14, clip_on=True)
ax3.text(wl0_Ha+25, yloc, r'$\mathrm{H}a$', fontsize=14, clip_on=True)
ax3.set_facecolor('#F7F7F7')
ax3.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax3.set_xlabel(r'$\mathrm{Rest\ wavelength\ [\AA]}$',fontsize=24)
ax3.set_xlim(np.min(wl[j]), np.max(wl[j]))
plt.tight_layout()
plt.show()
fig.savefig(sdss_id_dr14[j][:-5]+'_model.jpg',format="jpg",dpi=600,pad_inches = 0,\
            bbox_inches='tight' )




#       FIRST TRY / RESIDUALS / SECOND TRY



fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
# ----------------- FIRST TRY ------------------------------ #
ax1 = plt.subplot2grid(shape=(6, 2), loc=(0, 0), colspan=1, rowspan=2)
ax1.errorbar(wl_line[j], flux_line[j], fmt='-', color='grey', label='$\mathrm{Data}$')
# ax1.plot(wl_line[j], Gauss1[j], linestyle='-', linewidth=1, color=Color1)#, label='$\mathrm{Gaussian\ 1 \ fit}$')
# ax1.plot(wl_line[j], Gauss2[j], linestyle='-', linewidth=1, color=Color1)#, label='$\mathrm{Gaussian\ 2 \ fit}$')
# ax1.plot(wl_line[j], Gauss3[j], linestyle='-', linewidth=1, color=Color1)#, label='$\mathrm{Gaussian\ 3 \ fit}$')
ax1.errorbar(wl_line[j], Gauss_T[j], fmt='-', color=Color4, label='$\mathrm{Fit}$')
ax1.set_facecolor('#F7F7F7')
ax1.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.set_title(sdss_id_dr14[j][:-5],fontsize=24)
# ax1.legend(loc='upper right', numpoints=1)#, prop={'size': 12})
ax1.set_xlim(li_line, lf_line)
# ------------- FIRST TRY RESIDUALS ------------------------ #
ax2 = plt.subplot2grid(shape=(6, 2), loc=(2, 0), colspan=1, rowspan=1, sharex=ax1)
ax2.errorbar(wl_line[j], flux_line[j]-Gauss_T[j], fmt='o', markersize=3,\
             color='grey', mew=0, label='$\mathrm{Residuals}$')
ax2.set_facecolor('#F7F7F7')
ax2.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax2.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$',fontsize=16)
# ax2.legend(loc='upper right', numpoints=1)#, prop={'size': 12})
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set_xlim(li_line, lf_line)
# ----------------- SINGLE GAUSSIAN ------------------------------ #
ax3 = plt.subplot2grid(shape=(6, 2), loc=(0, 1), colspan=1, rowspan=2)
ax3.errorbar(wl_line[j], flux_line[j], fmt='-', color='grey', label='$\mathrm{Data}$')
ax3.errorbar(wl_line[j], Gauss_1g[j], fmt='-', color=Color4, label='$\mathrm{Fit}$')
ax3.set_facecolor('#F7F7F7')
ax3.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax3.legend(loc='upper right', numpoints=1)#, prop={'size': 12})
ax3.set_xlim(li_line, lf_line)
# ------------- GAUSSIAN RESIDUALS ------------------------ #
res_1g = flux_line[j]-Gauss_1g[j]
res_filt = np.ma.masked_array(res_1g, mask=mask[j])
wl_filt = np.ma.masked_array(wl_line[j], mask=mask[j])
const = np.full(wl_line[j].shape, fitted_ct[j])
ax4 = plt.subplot2grid(shape=(6, 2), loc=(2, 1), colspan=1, rowspan=1, sharex=ax1)
ax4.errorbar(wl_line[j], res_1g, fmt='o', markersize=3,\
             color='grey', mew=0, label='$\mathrm{Residuals}$')
ax4.errorbar(wl_filt, res_filt, fmt='o', markersize=3,\
             color=Color3, mew=0, label='$\mathrm{Residuals}$')
ax4.errorbar(wl_line[j], const, fmt='-', color=Color3, linewidth=1)
ax4.set_facecolor('#F7F7F7')
ax4.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax4.legend(loc='upper right', numpoints=1)#, prop={'size': 12})
plt.setp(ax3.get_xticklabels(), visible=False)
ax4.set_xlim(li_line, lf_line)
# ----------------- SECOND TRY ------------------------------ #
fl_filt = np.ma.masked_array(flux_line[j], mask=mask[j])
ax5 = plt.subplot2grid(shape=(6, 2), loc=(3, 0), colspan=2, rowspan=2)
ax5.errorbar(wl_line[j], flux_line[j], fmt='-', color='grey', label='$\mathrm{Data}$')
ax5.errorbar(wl_filt, fl_filt, fmt='-', color='black', label='$\mathrm{Filtered data}$')
ax5.plot(wl_line[j], Gauss1_f[j], linestyle='-', linewidth=1, color=Color3)#, label='$\mathrm{Gaussian\ 1 \ fit}$')
ax5.plot(wl_line[j], Gauss2_f[j], linestyle='-', linewidth=1, color=Color3)#, label='$\mathrm{Gaussian\ 2 \ fit}$')
ax5.plot(wl_line[j], Gauss3_f[j], linestyle='-', linewidth=1, color=Color3)#, label='$\mathrm{Gaussian\ 3 \ fit}$')
ax5.errorbar(wl_line[j], Gauss_T_f[j], fmt='-', color=Color4, label='$\mathrm{Fit}$')
ax5.set_facecolor('#F7F7F7')
ax5.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax1.set_title(sdss_id_dr14[j][:-5],fontsize=24)
# ax5.legend(loc='upper right', numpoints=1)#, prop={'size': 12})
ax5.set_xlim(li_line, lf_line)
# ------------- SECOND TRY RESIDUALS ------------------------ #
ax6 = plt.subplot2grid(shape=(6, 2), loc=(5, 0), colspan=2, rowspan=1, sharex=ax5)
ax6.errorbar(wl_line[j], res_f[j], fmt='o', markersize=3,\
             color='grey', mew=0, label='$\mathrm{Residuals}$')
ax6.set_facecolor('#F7F7F7')
ax6.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
# ax6.legend(loc='upper right', numpoints=1)#, prop={'size': 12})
ax6.set_xlim(li_line, lf_line)
plt.setp(ax5.get_xticklabels(), visible=False)
ax6.set_xlabel(r'$\mathrm{Rest\ wavelength\ [\AA]}$',fontsize=16)
# fig.savefig(sdss_id_dr14[j]+'residuals_line.pdf',format="pdf",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )


# FULL SPECTRUM
j=27
fig = plt.figure()
ax3 = fig.add_subplot(111)
yloc = np.min(flux[j]) + 8*(np.max(flux[j])-np.min(flux[j]))/10
# ax3.xaxis.set_major_locator(plt.MultipleLocator(20))
# ax3.xaxis.set_minor_locator(plt.MultipleLocator(5))
ax3.set_facecolor('#F7F7F7')
ax3.grid(True, color = '#9999993c', linestyle = ':', linewidth = 0.5)
ax3.set_xlabel(r'$\mathrm{Observed\ Wavelength\ [\AA]}$', fontsize = 18)
ax3.set_ylabel(r'$\mathrm{f_\lambda\ [10^{-17}erg^{}s^{-1}cm^{-2}\AA^{-1}]}$', fontsize = 18)
# ax3.legend(loc = 'upper right', fontsize = 14, numpoints = 1)#, prop = {'size':12})
ax3.set_title(sdss_id_dr14[i][:-5]+' (z='+str(REDSHIFT[j])+')',fontsize=18)
ax3.set_xlim(li_line, lf_line)
ax3.errorbar((1+REDSHIFT[j])*wl[j], flux[j], fmt='-', color='grey', label='$\mathrm{Data}$')
ax3.axvline(x=(1+REDSHIFT[j])*wl0_Lya	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_CIV	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_CIII	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_MgII	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_Hg	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_Hb	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_OIIId	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_OIIIe	, color='grey', linestyle='--', clip_on=True)
ax3.axvline(x=(1+REDSHIFT[j])*wl0_Ha	, color='grey', linestyle='--', clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_Lya+25, yloc, '$\mathrm{Lya}$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_CIV+25, yloc, '$\mathrm{CIV}$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_CIII+25, yloc, '$\mathrm{CIII]}$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_MgII+25, yloc, '$\mathrm{MgII}$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_Hg+25, yloc, r'$\mathrm{H}\gamma$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_Hb-125, yloc, r'$\mathrm{H}\beta$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_OIIIe+25, yloc, '$\mathrm{[OIII]}$', fontsize=14, clip_on=True)
ax3.text((1+REDSHIFT[j])*wl0_Ha+25, yloc, r'$\mathrm{H}a$', fontsize=14, clip_on=True)
ax3.set_facecolor('#F7F7F7')
ax3.grid(True, color='#9999993c', linestyle=':', linewidth=0.5)
ax3.set_xlabel(r'$\mathrm{Observed\ wavelength\ [\AA]}$',fontsize=24)
ax3.set_xlim((1+REDSHIFT[j])*np.min(wl[j]), (1+REDSHIFT[j])*np.max(wl[j]))
plt.tight_layout()
plt.show()
# fig.savefig(sdss_id_dr14[j][:-5]+'_full_spectrum.jpg',format="jpg",dpi=300,pad_inches = 0,\
#             bbox_inches='tight' )

