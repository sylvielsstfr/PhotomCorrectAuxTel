# Analysis of Time Series

- creation date : August 20th 2020

- Mostly wavelets decomposition on atmospheric transmission obtained from simulations with LibRadTran



## Time series Fourier transform

- **StFt_decomposition.ipynb**

## Wavelet scalogram

- **Waveletsfit_PSDw.ipynb**

- **Waveletsfit.ipynb**	


- **Wavelets_decomposition_scalogram.ipynb** : decomposition of coefficients

## Wavelet decomposition

### Decomposition one by one transmission

- **Wavelets_decomposition_6levels.ipynb**
- **Wavelets_decomposition_8levels.ipynb**

- **Wavelets_decomposition.ipynb** : the most advanced study

### Study most significant coefficients

work on all the statistics of the simulated transmission

- **Find_Wavelets_coefficients.ipynb**
- 
- **Optimum_number_Wavelets_coefficients.ipynb** : how mainy number of coeff sould be kept

## Production of datasets with wavelet decomposition coefficents

several levels of decompostion from 1 to 9		
	
- **ProduceWaveletCoeffDataset.ipynb**: produce dataset using atmospheric transparency excluding clouds.	

- **ProduceWaveletCoeffDataset_cloud.ipynb** : produce dataset using atmospheric transparency including clouds.	


- **ProduceWaveletCoeffDatasetSpectra.ipynb**: produce dataset unisng spectra including clouds.	



## Machine Learning

Linear Fit to estimate VAOD,PWV, O3, CLD.
Try regularisation with Ridge and Lasso, using scikit learn

- **MLfit_atmlsst_wavelets_nocld.ipynb**	
- **MLfit_atmlsst_wavelets_cloud.ipynb**


- **MLfitOptim_atmlsst_wavelets_cloud.ipynb** : loop on the coefficient dataset, use only Linear Fit (No reg)	



