####################################################################################################################
# python looponsimuatmtransp.py -f 6 -l 14
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/CNRS/IN2P3
#- creation date: September 11th 2020
####################################################################################################################


import subprocess

import numpy as np
import sys

import contextlib
import os

import sys, getopt

import logging
import coloredlogs

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:l:h", ['f=', "l=", 'help'])
    except getopt.GetoptError:
        print(' Exception bad getopt with :: ' + sys.argv[0] + ' -f <first-number> -l <last-number>')
        sys.exit(2)

    str_fnum = ""
    fnum = 0

    str_lnum = ""
    lnum = 0

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(sys.argv[0] + ' -f <first-number> -l <last-number>')
            sys.exit(2)
        elif opt in ('-f', '--fnum'):
            str_fnum = arg
            fnum = int(str_fnum)
        elif opt in ('-l', '--lnum'):
            str_lnum = arg
            lnum = int(str_lnum)
        else:
            print(sys.argv[0] + ' -f <first-number> -l <last-number>')
            sys.exit(2)

        # start with logs
        # -----------------
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

    IDX_START = fnum
    IDX_STOP = lnum

    msg="IDX_START={} , IDX_STOP={} ".format(IDX_START, IDX_STOP)
    logger.info(msg)

    with contextlib.redirect_stdout(None):
        for idx in np.arange(IDX_START, IDX_STOP + 1):
            cmd = f"python simulateatmtransparency.py --config config/default.ini -n {idx}"
            logger.info(cmd)
            os.system(cmd)


