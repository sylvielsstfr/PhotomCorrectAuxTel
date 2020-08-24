#########################################################################################
# libPySynPhotSpectra :
# - python library to produce AuxTel Spectra
# - author : Sylvie Dagoret-Campagne
# - cration date : August 24th 2020
########################################################################################

# import section

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from astropy.table import Table

# pysynphot
import pysynphot as S

pysynphot_root_path=os.environ['PYSYN_CDBS']
path_sed_calspec=os.path.join(pysynphot_root_path,'calspec')

print("path_sed_calspec=",path_sed_calspec)

PATH_LSSTFiltersKG='../../data/lsst/LSSTFiltersKG'
sys.path.append(PATH_LSSTFiltersKG)
PATH_LSSTFiltersKGDATA='../../data/lsst/LSSTFiltersKG/fdata'
sys.path.append(PATH_LSSTFiltersKGDATA)

import libLSSTFiltersKG as lsst

import scipy.special as sp
from scipy import interpolate


WLMINSEL=340.
WLMAXSEL=1100.

#-----------------------------------------------------------------------------------------
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx],int(idx)

#--------------------------------------------------------------------------------------
def GetListOfCalspec(file_sedsummary="sed/table_summary_allcalspec_torenorm.fits"):
    """
    GetListOfCalspec()

    the file include SED fro which color term has been caculated
    in /examples_sed/calspec/ViewCalspecColors.ipynb

    - input:
      file_sedsummary : filename of SED summary

    - output:
      t : astropy table of summary
    """

    t = Table.read(file_sedsummary)
    return t

#--------------------------------------------------------------------------------------

def SelectFewSED(t):
    """
    SelectFewSED(t)

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
#------------------------------------------------------------------------------------------

def plot_sed(t, ax):
    """
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
        #cm = ax.plot(X[wavelengths_indexes], Y[wavelengths_indexes], color=all_colors[idx])
        
        idx += 1

    # ax.set_xlim(3500.,11000.)
    # ax.set_ylim(0.,3.)
    ax.legend(prop={'size': 12})
    ax.grid()

    xlabel = ' $\\lambda$ (Angstrom)'
    ylabel = ' Flux (photlam) normalised to Mag-10'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title("Preselected CALSPEC relative SED")

    return cm


#-----------------------------------------------------------------------------------------------------------------

def GetSpectra(sed, wl_atm, atm_transmission, order2=False):
    """

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

        # add order 1 + order 2
    if order2:
        spectra = spectra + spectra2

    spectra = spectra * 10  # to get per unit of nm
    spectra2 = spectra2 * 10

    return spectra, spectra2
#------------------------------------------------------------------------------------------------------------
