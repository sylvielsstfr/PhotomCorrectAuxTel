###########################################################################################################
#   Fit atmosphere parameters Airmass, VAOD, PWV, O3, including clouds OD
# - author Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : September 17th 2020
#
# launch python MLfit_atmlsst_cloud_varz.py --config config/default.ini
#
#############################################################################################################

####################
#  Import
####################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import os,sys,re

from astropy.io import fits
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.dates as mdates
from matplotlib import gridspec

# to enlarge the sizes
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# for sed tables
from astropy.table import Table
from scipy import interpolate

# pysynphot
import pysynphot as S
pysynphot_root_path=os.environ['PYSYN_CDBS']
#pysynphot_root_path="/Users/dagoret/MacOSX/External/PySynPhotData/grp/hst/cdbs"
path_sed_calspec=os.path.join(pysynphot_root_path,'calspec')

# for atmospheric parameters
from scipy.stats import rayleigh,beta,gamma,uniform

# analytic for removing rayleigh
sys.path.append("../../tools/atmanalyticsim") # go to parent dir
import libatmscattering as atm

# for correlation
import seaborn as sns

# for order 1-2 spectrum
import scipy.special as sp

# for LSST transmission
PATH_LSSTFiltersKG='../../data/lsst/LSSTFiltersKG'
sys.path.append(PATH_LSSTFiltersKG)
PATH_LSSTFiltersKGDATA='../../data/lsst/LSSTFiltersKG/fdata'
sys.path.append(PATH_LSSTFiltersKGDATA)

import libLSSTFiltersKG as lsst

# for time handling
import time
from datetime import datetime,date
import dateutil.parser
import pytz


# for logging
import logging
import coloredlogs

# for arguments
import argparse

# for configuration
import configparser


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score,explained_variance_score


################################################################
# Constants
#################################################################

# Range of wavelengths
WLMINSEL=340.
WLMAXSEL=1100.

################################################################
# Functions
#################################################################

# tool:
#-----------------------------------------------------------
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],int(idx)
#-----------------------------------------------------------
def GetDistribFromName(name):
    """

    GetDistribFromName(name): Get distribution from name in config file

    input arg:

    - name : string name of the distribution

    """

    rayleigh, beta, gamma, uniform

    if name == "rayleigh":
        return rayleigh
    elif name == "beta":
        return beta
    elif name == "gamma":
        return gamma
    else:
        return uniform
#-----------------------------------------------------------

# to work with calspec SED
#------------------------------------------------------------------------------------
def GetListOfCalspec(file_sedsummary="sed/table_summary_allcalspec_torenorm.fits"):
    """
    GetListOfCalspec() : return a list of SED

    the file include SED fro which color term has been caculated
    in /examples_sed/calspec/ViewCalspecColors.ipynb

    - input:
      file_sedsummary : filename of SED summary

    - output:
      t : astropy table of summary
    """

    t = Table.read(file_sedsummary)
    return t
#---------------------------------------------------------------------------------------
def SelectFewSED(t):
    """
    SelectFewSED(t) : Select a few SED of different corlos

    input :
     - t astropy table of SED

    output :
    - t_selected : table of presselected SED

    """

    all_colors = t["VmI"]

    # list of target colors
    target_VmI = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    NBVmItarget = len(target_VmI)

    filesselected_index = np.zeros(NBVmItarget)
    filesselected_VmI = np.zeros(NBVmItarget)

    idx = 0
    for vmi in target_VmI:
        thevmi, theidx = find_nearest(all_colors, vmi)
        # print(thevmi,theidx)
        filesselected_index[idx] = int(theidx)
        filesselected_VmI[idx] = thevmi
        idx += 1

    t_selected = Table(t[0])
    for idx in filesselected_index[1:]:
        t_selected.add_row(t[int(idx)])

    return t_selected
#-----------------------------------------------------------------------------------------

def plot_sed(t, ax):
    """

    plot_sed(t, ax) : Plot SED from the astropy table t

    input:
        - t : tables of SED
        - ax : matplotlib axis


    """

    NBFILES = len(t)

    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBFILES)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBFILES), alpha=1)

    idx = 0

    flatsp = S.FlatSpectrum(10, fluxunits='photlam')

    for filename in t["FILES"]:

        fullfilename = os.path.join(path_sed_calspec, filename)

        if filename == "flat":
            spec = flatsp
            spec_norm = spec
        else:
            spec = S.FileSpectrum(fullfilename)

        spec_norm = spec.renorm(10, 'vegamag', S.ObsBandpass('johnson,v'))

        spec_norm.convert('photlam')
        # spec_norm.convert('nm')

        # WLMINSEL=340.
        # WLMAXSEL=1100.

        magV = t["magV"][idx]
        label = t["OBJNAME"][idx] + " ( " + t["SP_TYPE"][idx] + " ) " + " magV = {:.2g}".format(
            magV) + " v-i ={:.2f}".format(t["VmI"][idx])

        # X= spec_norm.wave[wavelengths_indexes]
        # Y= spec_norm.flux[wavelengths_indexes]

        X = spec_norm.wave
        Y = spec_norm.flux

        wavelengths_indexes = np.where(np.logical_and(spec_norm.wave > WLMINSEL * 10, spec_norm.wave < WLMAXSEL * 10))[
            0]

        # if filename!="flat":
        cm = ax.plot(X[wavelengths_indexes], Y[wavelengths_indexes], color=all_colors[idx], label=label)

        idx += 1

    # ax.set_xlim(3500.,11000.)
    # ax.set_ylim(0.,3.)
    ax.legend()
    ax.grid()

    xlabel = ' $\\lambda$ (Angstrom)'
    ylabel = ' Flux (photlam) normalised to Mag-10'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title("Preselected CALSPEC relative SED")

    return cm

#------------------------------------------------------------------------------------------------------------------
# Function to build the observed spectra
#------------------------------------------------------------------------------------------------------------------
def GetSpectra(sed, wl_atm, atm_transmission, order2=False,cut=False):
    """

    GetSpectra(sed, wl_atm, atm_transmission, order2=False) : compute a series of observed spectra from
    an input SED and many atmospheric transmission

    * input :
     - sed : Pysynphot SED
     -  wl  : wavelength of the atmospheric transmission (nm)
     - transmission : atmospheric transmission array
     - order2 : flag to compute the atmospheric transmission

    * output :
     - array of spectra (corresponding to wl in nm)

    """

    wl0 = sed.wave  # in angstrom
    spectra = np.zeros_like(atm_transmission)
    spectra2 = np.zeros_like(atm_transmission)  # order 2

    wl_atm_ang = 10 * wl_atm

    # quantum efficiency
    wl_ccd, throughput, ccdqe, trans_opt_elec = lsst.GetThroughputAndCCDQE("../../data/lsst")
    bp_qe = S.ArrayBandpass(wl_ccd * 10., ccdqe, name='QE')

    # passband for atmosphere
    Natm = atm_transmission.shape[0]
    all_bp_atm = []
    for i_atm in np.arange(Natm):
        label_atm = "atm{:d}".format(i_atm)
        bp_atm = S.ArrayBandpass(wl_atm_ang, atm_transmission[i_atm, :], name=label_atm)
        all_bp_atm.append(bp_atm)

    # Hologram transmission (analytic)
    X0 = 10000  # in antstrom
    X = wl0
    X2 = np.sort(np.unique(np.concatenate((wl0 / 2, wl0))))
    Eff_holo = sp.jv(1, X0 / X) ** 2
    Eff_holo2 = sp.jv(2, X0 / X2) ** 2

    bp_holo_order1 = S.ArrayBandpass(X, Eff_holo, name='Holo')
    bp_holo_order2 = S.ArrayBandpass(X2, Eff_holo2, name='Holo2')

    all_obs1 = []
    all_obs2 = []

    # compute spectra for order 1 and order 2
    for i_atm in np.arange(Natm):
        bp_all_order1 = bp_qe * all_bp_atm[i_atm] * bp_holo_order1
        bp_all_order2 = bp_qe * all_bp_atm[i_atm] * bp_holo_order2

        obs1 = S.Observation(sed, bp_all_order1, force='taper')  # order 1
        obs2 = S.Observation(sed, bp_all_order2, force='taper')  # order 2

        all_obs1.append(obs1)
        all_obs2.append(obs2)

    for i_atm in np.arange(Natm):
        obs1 = all_obs1[i_atm]
        func_order1 = interpolate.interp1d(obs1.binwave, obs1.binflux, bounds_error=False, fill_value=(0, 0))

        spectra[i_atm, :] = func_order1(wl_atm_ang)

        obs2 = all_obs2[i_atm]
        func_order2 = interpolate.interp1d(2 * obs2.binwave, obs2.binflux / 2, bounds_error=False, fill_value=(0, 0))
        spectra2[i_atm, :] = func_order2(wl_atm_ang)

    if cut:
        wlcut_indexes = np.where(wl_atm_ang < 7600)[0]
        spectra2[:, wlcut_indexes] = 0

    # add order 1 + order 2
    if order2:
        spectra = spectra + spectra2

    spectra = spectra * 10  # to get per unit of nm
    spectra2 = spectra2 * 10

    return spectra, spectra2

#------------------------------------------------------------------------------------------------------------------
# Function to plot
#------------------------------------------------------------------------------------------------------------------
def plot_param(iparam, ax, all_Yin, all_Yout, mode, nsig=10):
    """
    plot estimated parameters

    iparam = 0,1,2,3 : VAOD,PWV,O3,CLD
    ax     = axis

    all_Yin : inverse transform of the input in the simulation
    all_Yout : inverse transform predicted parameters

    mode :
    - 0 : Yout vs Yin
    - 1 : Yout - Yin vs Yin
    - 2 : (Yout - Yin)/Yin vs Yin

    """
    N = 4  # color mode corres ponding to parameter
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

    xlabel = ["VAOD in ", "PWV in (mm)", "Ozone in (dBU)", "Cloud OD in "]
    ylabel = ["value out", "value rec - value in", "$\delta V/V$"]
    ttitle = ["Estimation of VAOD",
              "Estimation of PWV",
              "Estimation of Ozone",
              "Estimation of Cloud OD"]

    DY = all_Yout[:, iparam] - all_Yin[:, iparam]
    Y0 = all_Yin[:, iparam]
    Y1 = all_Yout[:, iparam]
    RY = DY / Y0

    if mode == 0:
        cm = ax.plot(Y0, Y1, "o", color=all_colors[iparam])
    elif mode == 1:

        mu = DY.mean()
        median = np.median(DY)
        sigma = DY.std()
        textstr = '\n'.join((
            r'$\mu$={:3.2g}'.format(mu),
            r'$median$={:3.2g}'.format(median),
            r'$\sigma$={:3.2g}'.format(sigma)))

        cm = ax.plot(Y0, DY, "o", color=all_colors[iparam])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        ax.set_ylim(mu - nsig * sigma, mu + nsig * sigma)

    else:

        mu = RY.mean()
        median = np.median(RY)

        sigma = RY.std()

        textstr = '\n'.join((
            '$\mu$={:.3g}'.format(mu),
            '$median$={:.3g}'.format(median),
            '$\sigma$={:.3g}'.format(sigma)))

        cm = ax.plot(Y0, RY, "o", color=all_colors[iparam])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        ax.set_ylim(mu - nsig * sigma, mu + nsig * sigma)

    ax.grid()
    ax.set_title(ttitle[iparam])
    ax.set_xlabel(xlabel[iparam])
    ax.set_ylabel(ylabel[mode])

    ax.ticklabel_format(axis='y', style='sci',
                        scilimits=None,
                        useOffset=None,
                        useLocale=None,
                        useMathText=True)

    return cm
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
def plot_param_histo(iparam, ax, all_Yin, all_Yout, nsig=10):
    """
    plot histo of parameters

    iparam = 0,1,2,3 : VAOD,PWV,O3,CLD
    ax     = axis

    all_Yin : inverse transform of the input in the simulation
    all_Yout : inverse transform predicted parameters

    mode :
    - 0 : Yout vs Yin
    - 1 : Yout - Yin vs Yin
    - 2 : (Yout - Yin)/Yin vs Yin

    """
    N = 4  # color mode corres ponding to parameter
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

    DY = all_Yout[:, iparam] - all_Yin[:, iparam]
    Y0 = all_Yin[:, iparam]
    Y1 = all_Yout[:, iparam]
    RY = DY / Y0

    xlabel = ["$\Delta$ VAOD", "$\Delta$ PWV (mm)", "$\Delta$ Ozone (dBU)", "$\Delta$ OD"]

    ttitle = ["Estimation of VAOD",
              "Estimation of PWV",
              "Estimation of Ozone",
              "Estimation of Cloud OD"]

    mu = DY.mean()
    median = np.median(DY)
    sigma = DY.std()
    textstr = '\n'.join((
        '$\mu$={:.3g}'.format(mu),
        '$median$={:.3g}'.format(median),
        '$\sigma$={:.3g}'.format(sigma)))

    cm = ax.hist(DY, bins=50, color=all_colors[iparam])
    ax.grid()
    ax.set_title(ttitle[iparam])
    ax.set_xlabel(xlabel[iparam])
    ax.set_xlim(mu - nsig * sigma, mu + nsig * sigma)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    ax.ticklabel_format(axis='x',
                        style='sci',
                        scilimits=(-3, -3),
                        useOffset=None,
                        useLocale=None,
                        useMathText=True)

    return cm

#-----------------------------------------------------------------------------------------------------------------------
def plotcorrelation(ax, all_Y):
    """
    Plot correlation between all estimated parameters
    """

    df = pd.DataFrame(data=all_Y, columns=['VAOD', 'PWV', 'OZONE', "CLD-OD"])
    Var_Corr = df.corr()
    mask_ut = np.triu(np.ones(Var_Corr.shape)).astype(np.bool)
    mask_lt = np.logical_not(np.tril(np.ones(Var_Corr.shape)).astype(np.bool))

    sns.heatmap(Var_Corr, mask=mask_lt, vmin=0, vmax=1, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns,
                annot=True, ax=ax, cmap="jet")
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
def plotatmparameters(config,airmass,pwv,ozone,vaod,cld):
    """

    :param config:
    :param airmass:
    :param pwv:
    :param ozone:
    :param vaod:
    :param cld:
    :return:
    """

    fig = plt.figure(figsize=(12, 7.7))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)


    ax1.hist(airmass, bins=50, facecolor='blue', alpha=0.2, density=True);
    ax1.set_title("airmass")
    ax1.set_xlabel("z")

    if 'AIRMASS' in config_section:
        distrib_name = config['AIRMASS']['distrib']
        distrib = GetDistribFromName(distrib_name)
        # fit dist to data
        params = distrib.fit(airmass)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        x = np.linspace(airmass.min(), airmass.max(), 100)
        pdf = distrib.pdf(x, loc=loc, scale=scale, *arg)
        label = distrib_name + " : a={:2.2f} b={:2.2f} s={:2.2f}".format(arg[0], arg[1], scale)
        # label=distrib_name + " : s={:2.2f}".format(scale)
        ax1.plot(x, pdf, "r-", label=label)
        ax1.legend()
        ax1.grid()

    ax2.hist(pwv, bins=50, facecolor='blue', alpha=0.2, density=True);
    ax2.set_title("precipitable water vapor")
    ax2.set_xlabel("pwv (mm)")

    if 'PWV' in config_section:
        distrib_name = config['PWV']['distrib']
        distrib = GetDistribFromName(distrib_name)
        # fit dist to data
        params = distrib.fit(pwv)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        x = np.linspace(pwv.min(), pwv.max(), 100)
        pdf = distrib.pdf(x, loc=loc, scale=scale, *arg)
        # label=distrib_name + " : s={:2.2f}".format(scale)
        label = distrib_name + " : a={:2.2f} b={:2.2f} s={:2.2f}".format(arg[0], arg[1], scale)
        ax2.plot(x, pdf, "r-", label=label)
        ax2.legend()
        ax2.grid()

    ax3.hist(ozone, bins=50, facecolor='blue', alpha=0.2, density=True);
    ax3.set_title("ozone")
    ax3.set_xlabel("Ozone (DbU)")

    if 'OZONE' in config_section:
        distrib_name = config['OZONE']['distrib']
        distrib = GetDistribFromName(distrib_name)
        # fit dist to data
        params = distrib.fit(ozone)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        x = np.linspace(ozone.min(), ozone.max(), 100)
        pdf = distrib.pdf(x, loc=loc, scale=scale, *arg)
        # label=distrib_name + " : s={:2.2f}".format(scale)
        label = distrib_name + " : a={:2.2f} b={:2.2f} s={:2.2f}".format(arg[0], arg[1], scale)
        ax3.plot(x, pdf, "r-", label=label)
        ax3.legend()
        ax3.grid()

    ax4.hist(vaod, bins=50, facecolor='blue', alpha=0.2, density=True);
    ax4.set_title("Aerosols")
    ax4.set_xlabel("vaod")

    if 'AEROSOL' in config_section:
        distrib_name = config['AEROSOL']['distrib']
        distrib = GetDistribFromName(distrib_name)
        # fit dist to data
        params = distrib.fit(vaod)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        x = np.linspace(vaod.min(), vaod.max(), 100)
        pdf = distrib.pdf(x, loc=loc, scale=scale, *arg)
        # label=distrib_name + " : s={:2.2f}".format(scale)
        label = distrib_name + " : s={:2.2f}".format(scale)
        ax4.plot(x, pdf, "r-", label=label)
        ax4.legend()
        ax4.grid()

    ax5.hist(cld, bins=50, facecolor='blue', alpha=0.2, density=True);
    ax5.set_title("Cloud")
    ax5.set_xlabel("od")

    if 'CLOUD' in config_section:
        distrib_name = config['CLOUD']['distrib']
        distrib = GetDistribFromName(distrib_name)
        # fit dist to data
        params = distrib.fit(cld)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        x = np.linspace(cld.min(), cld.max(), 100)
        pdf = distrib.pdf(x, loc=loc, scale=scale, *arg)
        # label=distrib_name + " : s={:2.2f}".format(scale)
        label = distrib_name + " : s={:2.2f}".format(scale)
        # ax5.plot(x,pdf,"r-",label=label)
        ax5.legend()
        ax5.grid()


    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#    Read simulation of atmosphere
#----------------------------------------------------------------------------------------------------------------------
def GetAtmTransparency(inputdir, filelist):
    """
    GetAtmTransparency(inputdir, filelist): retrieve atmospheric transparency from simulation

    input arguments:
     - inputdir : input directory
     - filelist : list of file to read

    """

    filename = filelist[0]
    full_inputfilename = os.path.join(inputdir, filename)
    hduin = fits.open(full_inputfilename)

    headerin = hduin[0].header
    datain = hduin[0].data

    hdr = headerin
    NSIMH = hdr['NBATMSIM']
    idx_num = hdr['ID_NUM']
    idx_am = hdr['ID_AM']
    idx_vaod = hdr['ID_VAOD']
    idx_pwv = hdr['ID_PWV']
    idx_o3 = hdr['ID_O3']
    idx_cld = hdr['ID_CLD']
    idx_res = hdr['ID_RES']

    wl = datain[0, idx_res:]

    hduin.close()

    all_data = []
    # loop on files to extract data
    for file in filelist:
        full_inputfilename = os.path.join(inputdir, file)
        hdu = fits.open(full_inputfilename)

        datain = hdu[0].data
        all_data.append(datain[1:, :])
        hdu.close

    dataout = np.concatenate(all_data, axis=0)

    return wl, dataout, idx_num, idx_am, idx_vaod, idx_pwv, idx_o3, idx_cld, idx_res, headerin
#-----------------------------------------------------------------------------------------------------------------------
def plot_ml_result(Yin,Yout,mode,title):
    """
    plot_ml_result(Yin,Yout,mode,title)

    :param Yin:
    :param Yout:
    :param mode:
    :param title:
    :return:
    """

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(221)
    plot_param(0, ax, Yin, Yout, mode)

    ax = fig.add_subplot(222)
    plot_param(1, ax,Yin, Yout, mode)

    ax = fig.add_subplot(223)
    plot_param(2, ax, Yin, Yout, mode)

    ax = fig.add_subplot(224)
    plot_param(3, ax, Yin, Yout, mode)

    plt.tight_layout()
    plt.suptitle(title, Y=1.02, fontsize=18)
    plt.show()
#--------------------------------------------

def histo_ml_result(Yin, Yout, title):
    """

    :param Yin:
    :param Yout:
    :param mode:
    :param title:
    :return:
    """

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(221)
    plot_param_histo(0, ax, Yin, Yout)

    ax = fig.add_subplot(222)
    plot_param_histo(1, ax, Yin, Yout)

    ax = fig.add_subplot(223)
    plot_param_histo(2, ax, Yin, Yout)

    ax = fig.add_subplot(224)
    plot_param_histo(3, ax, Yin, Yout)

    plt.tight_layout()
    plt.suptitle(title, Y=1.02, fontsize=18)
    plt.show()
#---------------------------------------------

def plot_regularisation_coeff(alphas,alpha0,allcoefs,title):
    """

    plot_regularisation_coeff(alpha,all_coeff,title) : plot coefficient vrt regularisation parameter

    :param alphas: regularisation parameters
    : param alpha0 : vertical line
    :param allcoefs: coefficient
    :param title: title
    :return:
    """

    N = allcoefs.shape[2] # number of coefficients
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    for idx in np.arange(allcoefs.shape[2]):
        ax1.plot(alphas, allcoefs[:, 0, idx], color=all_colors[idx])
    ax1.set_xscale('log')
    ax1.axvline(x=alpha0, color='red')
    ax1.set_ylabel('weights - vaod')
    ax1.set_xlim(ax1.get_xlim()[::-1])  # reverse axis
    ax1.grid(True)
    ax1.set_title(title)

    ax2 = fig.add_subplot(222, sharex=ax1)
    for idx in np.arange(allcoefs.shape[2]):
        ax2.plot(alphas, allcoefs[:, 1, idx], color=all_colors[idx])
    ax2.axvline(x=alpha0, color='red')
    ax2.set_xscale('log')
    ax2.set_ylabel('weights - H2O')
    ax2.set_xlim(ax2.get_xlim()[::-1])  # reverse axis
    ax2.grid(True)

    ax3 = fig.add_subplot(223, sharex=ax1)
    for idx in np.arange(allcoefs.shape[2]):
        ax3.plot(alphas, allcoefs[:, 2, idx], color=all_colors[idx])
    ax3.axvline(x=alpha0, color='red')
    ax3.set_ylabel('weights - O3')
    ax3.set_xscale('log')
    ax3.set_xlim(ax3.get_xlim()[::-1])  # reverse axis
    ax3.grid(True)

    ax4 = fig.add_subplot(224, sharex=ax1)
    for idx in np.arange(allcoefs.shape[2]):
        ax4.plot(alphas, allcoefs[:, 3, idx], color=all_colors[idx])
    ax4.axvline(x=alpha0, color='red')
    ax4.set_ylabel('weights - CLD')
    ax4.set_xscale('log')
    # ax4.set_xlim(ax4.get_xlim()[::-1])  # reverse axis
    ax4.set_xlim(ax4.get_xlim())  # reverse axis
    ax4.grid(True)

    plt.xlabel('alpha')
    plt.axis('tight')

    plt.tight_layout()
    plt.show()


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":


    # start with logs
    #-----------------
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    handle = __name__

    logger = logging.getLogger(handle)
    # logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    # If you don't want to see log messages from libraries, you can pass a
    # specific logger object to the install() function. In this case only log
    # messages originating from that logger will show up on the terminal.
    coloredlogs.install(level='DEBUG', logger=logger)
    coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s')



    # set time
    # date
    today = date.today()
    string_date = today.strftime("%Y-%m-%d")

    # time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    tim = time.localtime()
    current_time = time.strftime("%H:%M:%S", tim)

    # timezones
    tz_LA = pytz.timezone('America/Los_Angeles')
    datetime_LA = datetime.now(tz_LA)
    msg="LA time:"+  datetime_LA.strftime("%H:%M:%S")
    logger.info(msg)

    tz_NY = pytz.timezone('America/New_York')
    datetime_NY = datetime.now(tz_NY)
    msg="NY time:"+ datetime_NY.strftime("%H:%M:%S")
    logger.info(msg)

    tz_London = pytz.timezone('Europe/London')
    datetime_London = datetime.now(tz_London)
    msg="London time:"+ datetime_London.strftime("%H:%M:%S")
    logger.info(msg)

    tz_Paris = pytz.timezone('Europe/Paris')
    datetime_Paris = datetime.now(tz_Paris)
    msg="Paris time:"+ datetime_Paris.strftime("%H:%M:%S")
    logger.info(msg)

    msg="************************ START *********************"
    logger.info(msg)

    # input arguments
    # ----------

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", dest="configfile",
                        help=f" run generate -config configfilename, with by ex configfilename = default.ini")
    results_args = parser.parse_args()

    # config file
    # --------------
    config_filename = results_args.configfile

    msg = f"Configuration file : {config_filename}"
    logger.info(msg)

    #################################################################################################################
    # 1) CONFIGURATION
    #################################################################################################################
    logger.info('1) Configuration')

    config = configparser.ConfigParser()

    if os.path.exists(config_filename):
        config.read(config_filename)
    else:
        msg = f"config file {config_filename} does not exist !"
        logger.error(msg)

    config_section = config.sections()

    if len(config_section) == 0:
        msg = f"empty config file {config_filename} !"
        logger.error(msg)

    if 'GENERAL' in config_section:

        FLAG_DEBUG = bool(int(config['GENERAL']['FLAG_DEBUG']))
        FLAG_VERBOSE = bool(int(config['GENERAL']['FLAG_VERBOSE']))
        FLAG_PLOT = bool(int(config['GENERAL']['FLAG_PLOT']))
        FLAG_PRINT = bool(int(config['GENERAL']['FLAG_PRINT']))
    else:
        msg = f"Configuration file : empty section GENERAL in config file {config_filename} !"
        logger.error(msg)
        sys.exit()

    if 'MLFIT' in config_section:
        input_file = config['MLFIT']['inputfile']
        input_dir = config['MLFIT']['inputdir']

        packetsize = int(config['MLFIT']['packetsize'])
        maxnbpacket = int(config['MLFIT']['maxnbpacket'])

        input_file_split = input_file.split(".")
        basefilename = input_file_split[0]
        extendfilename = input_file_split[1]
    else:
        msg=f"Configuration file : Missing section MLFIT in config file {config_filename} !"
        logger.error(msg)
        sys.exit()


    # Atmospheric transparency file and selection
    #----------------------------------------------
    all_files = os.listdir(input_dir)
    sorted_files = sorted(all_files)

    # search string
    search_str = "^" + f"{basefilename}.*fits$"

    selected_files = []
    for file in sorted_files:
        if re.search(search_str, file):
            selected_files.append(file)

    if maxnbpacket > 0:
        selected_files = selected_files[:maxnbpacket]

    NFiles = len(selected_files)

    msg=f"Number of input files : {NFiles}"
    logger.info(msg)

    separator=' , '
    logger.info(separator.join(selected_files))

    #################################################################################################################
    # 2) Read atmospheric transparency files
    #################################################################################################################
    logger.info('2) Read atmospheric transparency files')

    # Read input atmospheric transparency files
    wl, datain, idx_num, idx_am, idx_vaod, idx_pwv, idx_o3, idx_cld, idx_res, header = GetAtmTransparency(input_dir,
                                                                                                          selected_files)
    logger.info(header)

    ## retrieve indexes
    num = datain[0:, idx_num]
    airmass = datain[0:, idx_am]
    vaod = datain[0:, idx_vaod]  # vertical aerosol depth
    pwv = datain[0:, idx_pwv]  # precipitable water vapor (mm)
    ozone = datain[0:, idx_o3]  # ozone
    cld = datain[0:, idx_cld]  # clouds (not ussed)

    ## transmission array
    transm = datain[:, idx_res:]

    if FLAG_PLOT:
        plotatmparameters(config, airmass, pwv, ozone, vaod, cld)


    # Select wavelength range
    indexes_selected = np.where(np.logical_and(wl >= WLMINSEL, wl <= WLMAXSEL))[0]

    # need even number of bins
    if len(indexes_selected) % 2:
        indexes_selected = indexes_selected[:-1]

    wl = wl[indexes_selected]
    transm_tot = transm[:, indexes_selected]

    transm = transm_tot

    NWL = wl.shape[0]

    # plot transmission
    if FLAG_PLOT:
        N = 20
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=N)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        for idx in np.arange(N):
            ax.plot(wl, transm[idx, :], c=all_colors[idx])
        ax1 = ax.twinx()
        ax1.set_ylim(ax.get_ylim())
        ax.grid()
        ax.set_xlabel("$\lambda$ (nm)")
        ax.set_ylabel("transparency")
        ax.set_title("atmospheric transmission (no cloud)")
        plt.tight_layout()
        plt.show()




    # remove Rayleigh
    # Note that airmass is not 1.2 but varies
    # thus Rayleigh should not be removed
    od_rayl = atm.RayOptDepth_adiabatic(wl, altitude=atm.altitude0, costh=1 / 1.2)
    att_rayleigh = np.exp(-od_rayl)
    #transm = transm_tot / att_rayleigh


    # Create target arrays
    #----------------------

    airmassarr = airmass[:, np.newaxis]
    vaodarr = vaod[:, np.newaxis]
    pwvarr = pwv[:, np.newaxis]
    o3arr = ozone[:, np.newaxis]
    cldarr = cld[:, np.newaxis]


    # Create cloud array
    #-------------------

    transm_cloud = np.exp(-cld * airmass)
    transm_cloud_arr = np.exp(-cldarr * airmassarr)

    if FLAG_PLOT:
        fig = plt.figure(figsize=(8, 4))

        ax = fig.add_subplot(111)
        ax.hist(cld, bins=50, facecolor="b", label="in data", alpha=0.5)
        ax.set_yscale('log')
        ax.set_xlabel("cloud optical depth ")
        ax.grid()
        plt.tight_layout()
        plt.show()


    ##################
    # Prepare Y
    #################
    Y = np.concatenate((vaodarr, pwvarr, o3arr, cldarr), axis=1)
    Ylabel = ["vaod", "pwv", "ozone", "cld"]


    # SED
    #-----
    t = GetListOfCalspec()
    t_sel = SelectFewSED(t)

    if FLAG_PLOT:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        plot_sed(t_sel, ax)
        plt.show()


    # Retreive the SED
    idx_sed_sel = 3
    sed_filename = t_sel[idx_sed_sel]["FILES"]
    sed_objname = t_sel[idx_sed_sel]["OBJNAME"]
    sed_fullfilename = os.path.join(path_sed_calspec, sed_filename)


    # case of flat
    if sed_filename == "flat":
        flatsp = S.FlatSpectrum(10, fluxunits='photlam')
        spec = flatsp
    else:
        spec = S.FileSpectrum(sed_fullfilename)


    # renormalisation and conversion to photlam
    spec_norm = spec.renorm(10, 'vegamag', S.ObsBandpass('johnson,v'))
    spec_norm.convert('photlam')


    #################
    order2 = True
    cut    = True
    ################
    if order2 and not cut:
        specarrayfile = "spec_" + sed_objname + "_ord12.npy"
        specarrayfile2 = "spec2_" + sed_objname + "_ord2.npy"
        title_spec1 = "spectra order 1 and 2"
        title_spec2 = "spectra order 2"
    elif order2 and cut:
        specarrayfile = "speccut_" + sed_objname + "_ord12.npy"
        specarrayfile2 = "spec2cut_" + sed_objname + "_ord2.npy"
        title_spec1 = "spectra order 1 and 2"
        title_spec2 = "spectra order 2"

    else:
        specarrayfile = "spec_" + sed_objname + "_ord1.npy"
        specarrayfile2 = "spec2_" + sed_objname + "_ord2.npy"
        title_spec1 = "spectra order 1"
        title_spec2 = "spectra order 2"




    # Get Spectra unless already save in files
    if not os.path.isfile(specarrayfile) or not os.path.isfile(specarrayfile2):
        spectra, spectra2 = GetSpectra(sed=spec_norm, wl_atm=wl, atm_transmission=transm, order2=order2,cut=cut)
        np.save(specarrayfile, spectra)
        np.save(specarrayfile2, spectra2)
    else:
        spectra = np.load(specarrayfile)
        spectra2 = np.load(specarrayfile2)

    if FLAG_PLOT:

        N = 20
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=N)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

        themax = 0
        themin = 0

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121)
        for idx in np.arange(N):
            if spectra[idx, :].max() > themax:
                themax = spectra[idx, :].max()
            ax.plot(wl, spectra[idx, :], color=all_colors[idx])
        ax.set_ylim(0, 1.1 * themax)
        ax.grid()
        ax.set_xlabel("$\lambda$ (nm)")
        ax.set_title(title_spec1)

        ax = fig.add_subplot(122)
        for idx in np.arange(N):
            ax.plot(wl, spectra2[idx, :], color=all_colors[idx])
        ax.set_xlabel("$\lambda$ (nm)")
        ax.set_ylim(0, 1.1 * themax)
        ax.set_title(title_spec2)
        ax.grid()
    plt.tight_layout()
    plt.show()

    ##############################
    ##   X
    ##############################
    X = -2.5 * np.log10(spectra * transm_cloud_arr) / airmassarr

    if FLAG_PLOT:
        N = 50
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=N)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        for idx in np.arange(N):
            ax.plot(wl, X[idx, :], color=all_colors[idx])
        ax.set_xlabel("$\lambda$ (nm)")
        ax.set_ylabel("mag")
        ax.set_title("Observed spectra with cloud")
        ax.grid()
        ax1 = ax.twinx()
        ax1.set_ylim(ax.get_ylim())
        plt.tight_layout()
        plt.show()


    ##################################################################################################################
    #  Correlation coefficient
    ############################################################################################################
    if FLAG_PLOT:

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        Ny = Y.shape[1]
        Nx = X.shape[1]

        N = Ny
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=N)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(N), alpha=1)

        corr = np.zeros((Ny, Nx))

        for iy in np.arange(Ny):
            y = Y[:, iy]

            for ix in np.arange(Nx):
                x = X[:, ix]
                R = np.corrcoef(x=x, y=y, rowvar=False)
                corr[iy, ix] = R[0, 1]

            ax.plot(wl, corr[iy, :], color=all_colors[iy], label=Ylabel[iy])

        ax.legend()
        ax.set_xlabel("$\lambda$  (nm)")
        ax.set_ylabel("Correlation")
        ax.set_title("ML : Correlation coefficient Y - X")
        ax.grid()
        ax.set_ylim(0,1.)
        plt.show()



    #################################################################################################################
    # 3) Machine Learning
    #################################################################################################################
    logger.info('3) Machine Learning')


    # splitting in trainning and validation and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

    FLAG_SCALING = True

    # scaling
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # scale
    scaler_X.fit(X_train)
    scaler_Y.fit(Y_train)

    # transform
    X_train_scaled = scaler_X.transform(X_train)
    Y_train_scaled = scaler_Y.transform(Y_train)
    X_val_scaled = scaler_X.transform(X_val)
    Y_val_scaled = scaler_Y.transform(Y_val)
    X_test_scaled = scaler_X.transform(X_test)
    Y_test_scaled = scaler_Y.transform(Y_test)

    if FLAG_PLOT:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(np.concatenate((Y_train_scaled, Y_val_scaled, Y_test_scaled), axis=0), bins=50)
        ax.set_title("Renormalised target (VAOD, O3, PWV,CLD)")
        ax.grid()
        plt.tight_layout()
        plt.show()




    # LEARNING CURVE



    nb_tot_test = len(Y_test)
    nb_tot_train = len(Y_train)

    nsamples_test = np.arange(10, nb_tot_test, 100)
    nsamples_train = np.arange(10, nb_tot_train, 100)

    if 'LINEARREGRESSION' in config_section:
        FLAG_LINEARREGRESSION = bool(int(config['LINEARREGRESSION']['FLAG_LINEARREGRESSION']))
        FLAG_LINEARREGRESSION_RIDGE = bool(int(config['LINEARREGRESSION']['FLAG_LINEARREGRESSION_RIDGE']))
        FLAG_LINEARREGRESSION_LASSO = bool(int(config['LINEARREGRESSION']['FLAG_LINEARREGRESSION_LASSO']))
    else:
        msg = f"Configuration file : Missing section LINEARREGRESSION in config file {config_filename} !"
        logger.error(msg)
        sys.exit()








    ##################################################
    # Linear Regression
    ##################################################

    if FLAG_LINEARREGRESSION:


        logger.info('4) Linear Regression, no regularisation')




        all_MSE_train = np.zeros(len(nsamples_train))
        all_MSE_test = np.zeros(len(nsamples_test))
        all_MSE_test_full = np.zeros(len(nsamples_train))

        count = 0
        for n in nsamples_train:

            regr = linear_model.LinearRegression(fit_intercept=True)

            if FLAG_SCALING:
                X_train_cut = np.copy(X_train_scaled[:n, :])
                Y_train_cut = np.copy(Y_train_scaled[:n, :])
                if n in nsamples_test:
                    X_test_cut = np.copy(X_test_scaled[:n, :])
                    Y_test_cut = np.copy(Y_test_scaled[:n, :])
            else:
                X_train_cut = X_train[:n, :]
                Y_train_cut = Y_train[:n, :]
                if n in nsamples_test:
                    X_test_cut = X_test[:n, :]
                    Y_test_cut = Y_test[:n, :]

            # does the fit
            regr.fit(X_train_cut, Y_train_cut)

            # calculate metric
            # Make predictions using the testing set
            Y_pred_train = regr.predict(X_train_cut)
            if n in nsamples_test:
                Y_pred_test = regr.predict(X_test_cut)

            if FLAG_SCALING:
                Y_pred_test_full = regr.predict(np.copy(X_test_scaled))
            else:
                Y_pred_test_full = regr.predict(np.copy(X_test))

            MSE_train = mean_squared_error(Y_train_cut, Y_pred_train)

            if n in nsamples_test:
                MSE_test = mean_squared_error(Y_test_cut, Y_pred_test)

            if FLAG_SCALING:
                MSE_test_full = mean_squared_error(Y_test_scaled, Y_pred_test_full)
            else:
                MSE_test_full = mean_squared_error(Y_test, Y_pred_test_full)

            all_MSE_train[count] = MSE_train
            all_MSE_test_full[count] = MSE_test_full

            if n in nsamples_test:
                all_MSE_test[count] = MSE_test

            count += 1
            # end of loop


        if FLAG_PLOT:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(111)
            ax.plot(nsamples_train, all_MSE_train, 'b-o', label="train")
            # ax.plot(nsamples_test, all_MSE_test,'r-o',label="test")
            ax.plot(nsamples_train, all_MSE_test_full, 'r:o', label="test")
            ax.legend()
            ax.set_yscale("log")
            ax.set_xlabel("$N$")
            ax.set_ylabel("MSE")
            ax.set_title("Linear Regression - No reg : MSE with test vs N")
            ax.grid()
            # ax.set_ylim(1e-6,1e-2)
            ax1 = ax.twinx()
            ax1.set_ylim(ax.get_ylim())
            ax1.set_yscale("log")
            ax1.grid()
            plt.tight_layout()
            plt.show()


        ##############
        # Final fit
        #############


        regr = linear_model.LinearRegression(fit_intercept=True)

        if FLAG_SCALING:
            regr.fit(X_train_scaled, Y_train_scaled)
        else:
            regr.fit(X_train, Y_train)

        # calculate metric
        # Make predictions using the testing set

        if FLAG_SCALING:
            Y_pred_test = regr.predict(X_test_scaled)
            Y_pred_test_inv = scaler_Y.inverse_transform(Y_pred_test)
            DY = Y_pred_test - Y_test_scaled
        else:
            Y_pred_test = regr.predict(X_test)
            DY = Y_pred_test - Y_test

        if FLAG_SCALING:
            # The mean squared error
            msg='Mean squared error: %.5f' % mean_squared_error(Y_test_scaled, Y_pred_test)
            logger.info(msg)
            # The coefficient of determination: 1 is perfect prediction
            msg='Coefficient of determination: %.5f' % r2_score(Y_test_scaled, Y_pred_test)
            logger.info(msg)
            # Explained variance : 1 is perfect prediction
            msg='Explained variance: %.5f' % explained_variance_score(Y_test_scaled, Y_pred_test)
            logger.info(msg)

        else:
            # The mean squared error
            msg='Mean squared error: %.5f' % mean_squared_error(Y_test, Y_pred_test)
            logger.info(msg)
            # The coefficient of determination: 1 is perfect prediction
            msg='Coefficient of determination: %.5f' % r2_score(Y_test, Y_pred_test)
            logger.info(msg)
            # Explained variance : 1 is perfect prediction
            msg='Explained variance: %.5f' % explained_variance_score(Y_test, Y_pred_test)
            logger.info(msg)



        if FLAG_PLOT:
            plot_ml_result(Y_test, Y_pred_test_inv, mode=0, title="Linear Regression - No reg (with cloud)")
            plot_ml_result(Y_test, Y_pred_test_inv, mode=1, title="Linear Regression - No reg (with cloud)")
            plot_ml_result(Y_test, Y_pred_test_inv, mode=2, title="Linear Regression - No reg (with cloud)")
            #plot_ml_result(Y_test, Y_pred_test_inv, mode=3, title="Linear Regression - No reg (with cloud)")

            histo_ml_result(Y_test, Y_pred_test_inv, title="Linear Regression - No reg (with cloud)")


        if FLAG_PLOT:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(121)
            plotcorrelation(ax, Y_test)
            ax.set_title("atmospheric param at input")
            ax = fig.add_subplot(122)
            plotcorrelation(ax, Y_pred_test_inv)
            ax.set_title("atmospheric param at output")
            plt.tight_layout()
            plt.suptitle("Linear Regression - No reg (with cloud)", Y=1.02, fontsize=18)
            plt.show()


    if FLAG_LINEARREGRESSION_RIDGE:
        logger.info('5) Linear Regression, Ridge regularisation')


        ##############################################################################
        # Compute paths

        n_alphas = 200
        alphas = np.logspace(-10, -1, n_alphas)
        all_MSE = []

        coefs = []
        for a in alphas:
            ridge = linear_model.Ridge(alpha=a, fit_intercept=True)

            if FLAG_SCALING:

                ridge.fit(X_train_scaled, Y_train_scaled)
                coefs.append(ridge.coef_)

                # calculate metric
                # Make predictions using the testing set
                Y_pred = ridge.predict(X_val_scaled)
                MSE = mean_squared_error(Y_val_scaled, Y_pred)
            else:
                ridge.fit(X_train, Y_train)
                coefs.append(ridge.coef_)

                # calculate metric
                # Make predictions using the testing set
                Y_pred = ridge.predict(X_val)
                MSE = mean_squared_error(Y_val, Y_pred)

            # book MSE result
            all_MSE.append(MSE)

        allcoefs = np.array(coefs)
        alphas = np.array(alphas)
        all_MSE = np.array(all_MSE)

        idx_min = np.where(all_MSE == all_MSE.min())[0][0]
        alpha_ridge_min = alphas[idx_min]


        if FLAG_PLOT:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.plot(alphas, all_MSE, 'b-o')
            ax.set_xscale("log")
            alpha_ridge = alphas[idx_min]
            ax.axvline(x=alpha_ridge, color='red')
            ax.set_xlabel("$\\alpha$")
            ax.set_ylabel("MSE")
            ax.set_title("Ridge : MSE with validation test")
            ax.grid()
            ax1 = ax.twinx()
            ax1.set_ylim(ax.get_ylim())
            ax1.grid()
            plt.show()

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.plot(alphas, all_MSE, 'b-')
            ax.set_xscale('log')
            ax.axvline(x=alpha_ridge, color='red')
            ax.grid()
            # ax.set_yscale('log')
            ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
            ax.set_xlabel("$\\alpha$")
            ax.set_ylabel("MSE")
            ax.set_title("Ridge : MSE with validation test")
            ax1 = ax.twinx()
            ax1.set_ylim(ax.get_ylim())
            ax1.grid()
            plt.show()

        # plot coefficients wrt regularisation parameter alpha
        if FLAG_PLOT:
            plot_regularisation_coeff(alphas, alpha_ridge_min, allcoefs, title='Ridge coefficients as a function of the regularization')


        # learning curve for ridge

        all_MSE_train = np.zeros(len(nsamples_train))
        all_MSE_test = np.zeros(len(nsamples_test))
        all_MSE_test_full = np.zeros(len(nsamples_train))

        count = 0
        for n in nsamples_train:

            ridge = linear_model.Ridge(alpha=alpha_ridge_min, fit_intercept=True)

            if FLAG_SCALING:
                X_train_cut = np.copy(X_train_scaled[:n, :])
                Y_train_cut = np.copy(Y_train_scaled[:n, :])
                if n in nsamples_test:
                    X_test_cut = np.copy(X_test_scaled[:n, :])
                    Y_test_cut = np.copy(Y_test_scaled[:n, :])
            else:
                X_train_cut = X_train[:n, :]
                Y_train_cut = Y_train[:n, :]
                if n in nsamples_test:
                    X_test_cut = X_test[:n, :]
                    Y_test_cut = Y_test[:n, :]

            ridge.fit(X_train_cut, Y_train_cut)

            # calculate metric
            # Make predictions using the testing set
            Y_pred_train = ridge.predict(X_train_cut)

            if FLAG_SCALING:
                Y_pred_test_full = ridge.predict(X_test_scaled)
            else:
                Y_pred_test_full = ridge.predict(X_test)

            if n in nsamples_test:
                Y_pred_test = ridge.predict(X_test_cut)

            MSE_train = mean_squared_error(Y_train_cut, Y_pred_train)

            if FLAG_SCALING:
                MSE_test_full = mean_squared_error(Y_test_scaled, Y_pred_test_full)
            else:
                MSE_test_full = mean_squared_error(Y_test, Y_pred_test_full)

            if n in nsamples_test:
                MSE_test = mean_squared_error(Y_test_cut, Y_pred_test)

            all_MSE_train[count] = MSE_train
            all_MSE_test_full[count] = MSE_test_full

            if n in nsamples_test:
                all_MSE_test[count] = MSE_test

            count += 1
            # end of loop


        #Plot learning curve
        if FLAG_PLOT:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
            ax.plot(nsamples_train, all_MSE_train, 'b-o', label="train")
            # ax.plot(nsamples_test, all_MSE_test,'r-o',label="test")
            ax.plot(nsamples_train, all_MSE_test_full, 'r-o', label="test")
            ax.legend()
            ax.set_yscale("log")
            ax.set_xlabel("$N$")
            ax.set_ylabel("MSE")
            ax.set_title("RIDGE : MSE with test vs N")
            ax.grid()
            # ax.set_ylim(1e-6,1e-2)
            ax1 = ax.twinx()
            ax1.set_ylim(ax.get_ylim())
            ax1.set_yscale("log")
            ax1.grid()
            plt.show()

        ## Final fit for ridge


        ridge = linear_model.Ridge(alpha=alpha_ridge_min, fit_intercept=True)

        if FLAG_SCALING:
            ridge.fit(X_train_scaled, Y_train_scaled)
        else:
            ridge.fit(X_train, Y_train)

            # calculate metric
            # Make predictions using the testing set

        if FLAG_SCALING:
            Y_pred_test = ridge.predict(X_test_scaled)
            Y_pred_test_inv = scaler_Y.inverse_transform(Y_pred_test)
            DY = Y_pred_test - Y_test_scaled
        else:
            Y_pred_test = ridge.predict(X_test)
            DY = Y_pred_test - Y_test

        if FLAG_SCALING:
            # The mean squared error
            msg='Mean squared error: %.5f' % mean_squared_error(Y_test_scaled, Y_pred_test)
            logger.info(msg)
            # The coefficient of determination: 1 is perfect prediction
            msg='Coefficient of determination: %.5f' % r2_score(Y_test_scaled, Y_pred_test)
            logger.info(msg)
            # Explained variance : 1 is perfect prediction
            msg='Explained variance: %.5f' % explained_variance_score(Y_test_scaled, Y_pred_test)
            logger.info(msg)

        else:
            # The mean squared error
            msg='Mean squared error: %.5f' % mean_squared_error(Y_test, Y_pred_test)
            logger.info(msg)
            # The coefficient of determination: 1 is perfect prediction
            msg='Coefficient of determination: %.5f' % r2_score(Y_test, Y_pred_test)
            logger.info(msg)
            # Explained variance : 1 is perfect prediction
            msg='Explained variance: %.5f' % explained_variance_score(Y_test, Y_pred_test)
            logger.info(msg)


        if FLAG_PLOT:
            plot_ml_result(Y_test, Y_pred_test_inv, mode=0, title="Linear Regression - Ridge reg (with cloud)")
            plot_ml_result(Y_test, Y_pred_test_inv, mode=1, title="Linear Regression - Ridge reg (with cloud)")
            plot_ml_result(Y_test, Y_pred_test_inv, mode=2, title="Linear Regression - Ridge reg (with cloud)")
            #plot_ml_result(Y_test, Y_pred_test_inv, mode=3, title="Linear Regression - Ridge reg (with cloud)")

            histo_ml_result(Y_test, Y_pred_test_inv, title="Linear Regression - Ridge reg (with cloud)")


        if FLAG_PLOT:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(121)
            plotcorrelation(ax, Y_test)
            ax.set_title("atmospheric param at input")
            ax = fig.add_subplot(122)
            plotcorrelation(ax, Y_pred_test_inv)
            ax.set_title("atmospheric param at output")
            plt.tight_layout()
            plt.suptitle("Linear Regression - Ridge reg (with cloud)", Y=1.02, fontsize=18)
            plt.show()
