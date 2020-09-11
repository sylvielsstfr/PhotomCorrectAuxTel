####################################################################################################################
# Simulate atmospheric transparency
#
#    author : Sylvie Dagoret-Campagne
#    creation date : September 11th 2020
#    update        : September 11th 2020
#
# run :
#       >python generateatmparameters.py --config config/default.ini
#
###################################################################################################################


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import os
import matplotlib as mpl
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

    msg = f"Packet number : {packetnum}"
    logger.info(msg)

    # 1) CONFIGURATION
    #------------------
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


    if 'SIMTRANSPARENCY'in config_section:
        input_file = config['SIMTRANSPARENCY']['inputfile']
        input_dir = config['SIMTRANSPARENCY']['inputdir']

        output_file = config['SIMTRANSPARENCY']['outputfile']
        output_dir = config['SIMTRANSPARENCY']['outputdir']

        packetsize = int(config['SIMTRANSPARENCY']['packetsize'])



    msg = f"FLAG_PLOT = {FLAG_PLOT}"
    logger.info(msg)

    # range inside the atmospheric parameter file
    NROWMIN = (packetnum - 1) * packetsize
    NROWMAX = (packetnum) * packetsize - 1

    msg = f"NROWMIN = {NROWMIN} , NROWMAX  = {NROWMAX}  "
    logger.info(msg)