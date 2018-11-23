import numpy as np
import sir3d
from astropy.io import fits

# Cheung
input_stokes = '/scratch1/3dcubes/cheung_stokes_sir.h5'
output_spectral_stokes = '/scratch1/3dcubes/cheung_stokes_est_degraded_sir.h5'


psf_spectral = np.loadtxt('EST_ins_prof.psf')

lambda_hinode = np.loadtxt('wavelength_Hinode.txt') - 6301.5080
lambda_hinode *= 1e3

tmp = sir3d.psf.PSF(use_mpi=True, batch=256)
tmp.run_all_pixels_spectral(input_stokes, output_spectral_stokes, spectral_psf=psf_spectral, final_wavelength_axis=lambda_hinode)