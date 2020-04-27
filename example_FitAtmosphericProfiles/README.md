# README.md

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : April 27 2020

## topic
- Fit of atmospheric profile as the deviation from a reference profile.
- use absorption templates (or patterns) for molecule H2O, O2,O3
- Fit the continuous scattering supposed from aerosols, may be also a mixture of Rayleigh and aerosols
- Fit a common grey component.
- The model for the continuous scattering component is a piecewise second order polynomial in wavelength
- Each piece is related to the underlying filter in LSST : U,G,R I,Z, Y

## Notebooks

Progressive implementation of the fit step by step in notebooks:
		
- 1_ViewRationtoStandardAtmosphere.ipynb	
- 2_ViewRatioInWavebands.ipynb				
- 3_SplitAtmosphericAbsProfilesInFilters.ipynb
- 4_ViewOneRatioProfile.ipynb
- 5_FitOneRatioProfileOneBand.ipynb
- 6_FitOneRatioProfileTwoBands.ipynb
- 7_FitOneRatioProfileThreeBands.ipynb
- 8_FitOneRatioProfileFourBands.ipynb
- 9_FitOneRatioProfileMultiBands.ipynb
- 10_FitNonLinearOneRatioProfileMultiBands.ipynb
- 11_ReconstructAtmosphericProfilesFromNonLinearFit.ipynb




- 
-  