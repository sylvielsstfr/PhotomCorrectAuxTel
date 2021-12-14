# README.md

- Machine learning and wavelet decomposition techniques and gaussian process on atmospheric transmission and spectra

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : September 25th 2020

- Last verification : December 14th 2021


There are tow kind of simulated transparencies used at input , the old and hte new :
A) the old : 10 years of LSST stored in ../data/atm
B) the new : Many simulations handled in   ../tools/MakeAtmosphericTransparencyProd/atmsimdata/
The new simulation must be handled with a **config file**.


## MLfit_atm : 
- Machine learning : linear regression on the full spectrum at fixed airmass 	
- use the old transparency datset

## WLSeries  : 
- Machine learning : linear regression on the coefficient of wavelet decomposition  at fixed airmass
- Use the old dataset

## MLfit_atm2 : 
- Machine learning : linear regression on the full spectrum at variable airmass
- Use the new transparency dataset 


## WLSeries2  : NOT IMPLEMENTED
- Machine learning : linear regression on the coefficient of wavelet decomposition  at variable airmass (Not implemented)
- Use the new transparency dataset

## MLfitspline:
- generate splines to simulate telescope transmission
- 

## MLfit_gp	:
- try to fit bouguer lines on toy model to recover instument transmission using
- 1) standard fit
- 2) linear regression
- 3) gaussian process

## MLfit_gp_data	:
-try to fit bouguer lines on Pic du Midi to recover instument transmission
- extract throught and order1/order2

## MLfit_atm2_data  : 
- Machine learning : linear regression on the full spectrum at variable airmass at Pic du Midi


