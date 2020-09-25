####################################################################################################################
# Simulate atmospheric transparency
#
#    author : Sylvie Dagoret-Campagne
#    creation date : September 25th 2020
#    update        : September 25th 2020
#
# run :
#       >python simulateatmtransparency_pdm.py --config config/picdumidi.ini -n 1
#     -n 1 process packet number n=1
#
###################################################################################################################


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import os,sys
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import itertools

from astropy import units as u
from astropy.coordinates import Angle

import time
from datetime import datetime,date
import dateutil.parser
import pytz

import argparse

import logging
import coloredlogs
import configparser


from astropy.io import fits

from scipy.stats import rayleigh,beta,gamma,uniform

# to enlarge the sizes
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

os.getenv('LIBRADTRANDIR')
sys.path.append('../libradtran')
import libsimulateVisible_pdm






#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":


    # date
    today = date.today()
    string_date = today.strftime("%Y-%m-%d")


    # time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    tim = time.localtime()
    current_time = time.strftime("%H:%M:%S", tim)



    #timezones
    tz_LA = pytz.timezone('America/Los_Angeles')
    datetime_LA = datetime.now(tz_LA)
    print("LA time:", datetime_LA.strftime("%H:%M:%S"))


    tz_NY = pytz.timezone('America/New_York')
    datetime_NY = datetime.now(tz_NY)
    print("NY time:", datetime_NY.strftime("%H:%M:%S"))

    tz_London = pytz.timezone('Europe/London')
    datetime_London = datetime.now(tz_London)
    print("London time:", datetime_London.strftime("%H:%M:%S"))

    tz_Paris = pytz.timezone('Europe/Paris')
    datetime_Paris = datetime.now(tz_Paris)
    print("Paris time:", datetime_Paris.strftime("%H:%M:%S"))


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


    # arguments
    #----------


    parser = argparse.ArgumentParser()
    parser.add_argument("--config",action="store", dest="configfile",help=f" run simulate -config configfilename, with by ex configfilename = default.ini")
    parser.add_argument("-n", action="store", dest="packetnum",
                        help=f" run simulate --config configfilename -n 10, with by ex configfilenam, n packet number")


    results_args = parser.parse_args()


    msg = f"Start {parser.prog} at date : {string_date} and time :{current_time} and with arguments:{results_args}"
    logger.info(msg)



    # config file
    # --------------
    configfile = ""

    config_filename = results_args.configfile
    msg = f"Configuration file : {config_filename}"
    logger.info(msg)


    packetnum = int(results_args.packetnum)
    packet_str = str(packetnum).zfill(4)

    msg = f"Packet number : {packetnum}  Packet string {packet_str}"
    logger.warning(msg)

    ###########################################################################################################
    # 1) CONFIGURATION
    # ###########################################################################################################
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

        FLAG_DEBUG = bool(config['GENERAL']['FLAG_DEBUG'])
        FLAG_VERBOSE = bool(config['GENERAL']['FLAG_VERBOSE'])
        FLAG_PLOT = bool(config['GENERAL']['FLAG_PLOT'])
        FLAG_PRINT = bool(config['GENERAL']['FLAG_PRINT'])
    else:
        msg = f"empty section GENERAL in config file {config_filename} !"
        logger.error(msg)

    if 'GENERAL' in config_section:

        FLAG_DEBUG = bool(int(config['GENERAL']['FLAG_DEBUG']))
        FLAG_VERBOSE = bool(int(config['GENERAL']['FLAG_VERBOSE']))
        FLAG_PLOT = bool(int(config['GENERAL']['FLAG_PLOT']))
        FLAG_PRINT = bool(int(config['GENERAL']['FLAG_PRINT']))
    else:
        msg = f"empty section GENERAL in config file {config_filename} !"
        logger.error(msg)

    if 'SIMTRANSPARENCY' in config_section:
        input_file = config['SIMTRANSPARENCY']['inputfile']
        input_dir = config['SIMTRANSPARENCY']['inputdir']

        output_file = config['SIMTRANSPARENCY']['outputfile']
        output_dir = config['SIMTRANSPARENCY']['outputdir']

        packetsize = int(config['SIMTRANSPARENCY']['packetsize'])

    output_file_split = output_file.split(".")
    output_file = output_file_split[0] + "_" + packet_str + "." + output_file_split[1]

    full_inputfilename = os.path.join(input_dir, input_file)
    full_outputfilename = os.path.join(output_dir, output_file)




    msg = f"FLAG_PLOT = {FLAG_PLOT}"
    logger.info(msg)

    # range inside the atmospheric parameter file
    NROWMIN = (packetnum - 1) * packetsize
    NROWMAX = (packetnum) * packetsize - 1

    msg = f"NROWMIN = {NROWMIN} , NROWMAX  = {NROWMAX}  "
    logger.info(msg)

    msg = f"FLAG_PLOT = {FLAG_PLOT}"
    logger.info(msg)

    msg = f"FLAG_VERBOSE = {FLAG_VERBOSE}"
    logger.info(msg)

    msg = f"input file name = {full_inputfilename}"
    logger.info(msg)

    msg = f"output file name = {full_outputfilename}"
    logger.info(msg)


    ##########################################################################################################
    # 2) Open input file atmospheric parameters
    ##########################################################################################################

    logger.info('2) Atmospheric Parameter')

    hduin = fits.open(full_inputfilename)


    if FLAG_VERBOSE:
        msg = "{}".format(hduin.info())
        logger.info(msg)

    headerin = hduin[0].header
    datain = hduin[0].data

    NSIM = len(datain)

    if NROWMIN > NSIM:
        msg = f" >>> NROWMIN = {NROWMIN} greater than  NSIM={NSIM} ==> stop simulation"
        logger.error(msg)

    if FLAG_VERBOSE:
        logger.info(headerin)
        logger.info(datain[0, :])



    # decode the header
    hdr = headerin
    NSIMH = hdr['NBATMSIM']
    idx_num = hdr['ID_NUM']
    idx_am = hdr['ID_AM']
    idx_vaod = hdr['ID_VAOD']
    idx_pwv = hdr['ID_PWV']
    idx_o3 = hdr['ID_O3']
    idx_cld = hdr['ID_CLD']
    idx_res = hdr['ID_RES']


    ##################################################################################################################
    #  3) Simulate atmosphere with libradtran
    ##################################################################################################################

    logger.info('3) Simulation')

    all_wl = []
    all_transm = []
    all_z = []

    # first simulation to determine which is the wavelength length
    path, thefile = libsimulateVisible_pdm.ProcessSimulation(1, 0, 0, 0, prof_str='us', proc_str='sa', cloudext=0)
    data = np.loadtxt(os.path.join(path, thefile))
    wl = data[:, 0]
    atm = data[:, 1]
    NWL = len(wl)
    dataout = np.zeros((packetsize + 1, idx_res + NWL))
    dataout[0, idx_res:] = wl

    IDXMIN=min(NROWMIN, NSIM)
    IDXMAX=min(NSIM, NROWMAX)

    msg=f" - simulation from row {IDXMIN} to {IDXMAX}"
    logger.info(msg)

    idx = 0 # counter
    for irow in np.arange(IDXMIN,IDXMAX + 1):

        am = datain[irow, idx_am]
        pwv = datain[irow, idx_pwv]
        ozone = datain[irow, idx_o3]
        aer = datain[irow, idx_vaod]
        pressure = 0
        cloudext = datain[irow, idx_cld]
        msg = f"run libradtran : idx= {idx:3d} irow={irow:3d}, am={am:2.2f}, pwv={pwv:2.2f}, ozone={ozone:3.2f}, aer={aer:2.2f}"
        if FLAG_VERBOSE:
            logger.info(msg)
        else:
            if idx % 10 == 0:
                logger.info(msg)

        # path,thefile=libsimulateVisible.ProcessSimulation(am,pwv,ozone,pressure,prof_str='us',proc_str='sa',cloudext=cloudext)
        path, thefile = libsimulateVisible_pdm.ProcessSimulationaer(am, pwv, ozone, aer, pressure, prof_str='us',
                                                                proc_str='as', cloudext=0)

        data = np.loadtxt(os.path.join(path, thefile))
        wl = data[:, 0]
        atm = data[:, 1]
        all_wl.append(wl)
        all_transm.append(atm)
        all_z.append(am)

        dataout[idx + 1, idx_num] = irow
        dataout[idx + 1, idx_am] = am
        dataout[idx + 1, idx_vaod] = aer
        dataout[idx + 1, idx_pwv] = pwv
        dataout[idx + 1, idx_o3] = ozone
        dataout[idx + 1, idx_cld] = cloudext
        dataout[idx + 1, idx_res:] = atm


        idx += 1  # increase row number


    ################################################################################################################
    # 4)  Plot
    ################################################################################################################

    if FLAG_PLOT:

        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=packetsize)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(packetsize), alpha=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for index in np.arange(packetsize):
            am = all_z[index]
            label = f'z={am:2.2f}'
            ax.plot(all_wl[index], -2.5 * np.log10(all_transm[index]) / am, color=all_colors[index], label=label);
        if packetsize <= 10:
            ax.legend()
        ax.set_ylim(0, 2)
        ax.grid()
        ax.set_xlabel("$\lambda$ (nm)")
        ax.set_ylabel("mag")
        plt.show()

    if FLAG_PLOT:
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(111)
        ax.imshow(dataout, origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap='jet',extent=[wl.min(), wl.max(), NROWMIN - 1, NROWMAX])
        ax.grid()
        ax.set_xlabel("$\lambda$ (nm)")
        ax.set_ylabel("row number")
        ax.set_title("dataout")
        plt.show()

    ################################################################################################################
    # 5)  Save output file
    ################################################################################################################
    msg = f"save in output file {full_outputfilename}"
    logger.info(msg)

    headerout = headerin

    hdu = fits.PrimaryHDU(dataout, header=headerout)
    hdu.writeto(full_outputfilename, overwrite=True)


