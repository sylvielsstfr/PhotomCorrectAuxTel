# README.md

Simulate atmospheric transparency at LSST with libradtran

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/CNRS/IN2P3 
- creation date: September 11th 2020
- Update : September 15th 2020
- Last Update : December 13th 2021



The control of the simulation is to be done in the files in config directory.
In particular the number of simulations is controled there.


## A. Generate atmospheric parameters

- **generateatmparameters.ipynb**	
- **generateatmparameters.py**

examples :
             python GenerateAtmParameters.py --config config/default.ini


- **checkatmparameters.ipynb** : check output on generated atmospheric parameters

## B. Simulate atmospheric transparency

- **simulateatmtransparency.ipynb**
- **simulateatmtransparency.py**

-**viewatmtransparency.ipynb** : check output on simulated atmospheric transparency

## C. Configuration
- **config/default.ini** : single configuration file for all applications in that directory

## D. data
- **atmsimdata**

## E. Loop to simulate atmospheric parameters over packets

- **looponsimuatmtransp.py**

example to simulate for packet 1 to 3: 
      >python looponsimuatmtransp.py -f 1 -l 3


Packet number start from 1 (not 0)




