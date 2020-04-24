# README.md

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : April 20th 2020
- update : April 21th 2020


Tools show how to do something

## Read the atmospheric simulation

- **ViewAtmosphere.ipynb** : read one file (365 simulations)	

- **ViewAllAtmosphere.ipynb** : read all files (365x10) simulations

- **ViewAverageAtmosphere.ipynb** : View the average atmospheric profile and upper and lower limit


## Examples that Read the LSST throughput

Official throughput to use for the internship

- **PlotSimpleLSSTFilters.ipynb** : very simple system transmission profile

- **FiltersWithPySynPhot.py**  or **FiltersWithPySynPhot.ipynb** : Use pysynphot to combine system and atmospheric throughput and view the observation.

## Play with Magnitudes units
 - **MyVegaCalib.ipynb** use pysynphot to play with magnitudes, counts, using the Vega standard source.

 
## Generate atmospheric profiles with libradtran

- **libradtran/**
Simulate absorption profiles with libradtran

## Generate individual molecule absorption profile from librandtran output

- **MakeAbsorptionPatterns/** : Generate fits and csv files for atmospheric profiles of O2, O3, PWV
