import numpy as np
import sir3d
from astropy.io import fits

# Rempel
input_stokes = '/scratch1/3dcubes/cheung_stokes.h5'
input_model = '/scratch1/3dcubes/cheung_model.h5'

output_spatial_stokes = '/scratch1/3dcubes/cheung_stokes_spat_degraded.h5'
output_spatial_model = '/scratch1/3dcubes/cheung_model_spat_degraded.h5'

output_spatial_spectral_stokes = '/scratch1/3dcubes/cheung_stokes_spat_spec_degraded.h5'

tmp = fits.open('hinode_psf_size_256_def_-0.32.fits')
psf_spatial = tmp[0].data[0:-1,0:-1]
tmp.close()

psf_spectral = np.loadtxt('HINODE_SP_ins_prof.psf')

lambda_hinode = np.loadtxt('wavelength_Hinode.txt') - 6301.5080
lambda_hinode *= 1e3

hinode_pixel = 0.16 * 725.
simulation_pixel = 48.0
zoom = simulation_pixel / hinode_pixel

tmp = sir3d.psf.PSF(input_stokes, input_model, output_spatial_stokes, output_spatial_model, output_spatial_spectral_stokes,
    spatial_psf=psf_spatial, spectral_psf=psf_spectral, final_wavelength_axis=lambda_hinode, zoom_factor=zoom, batch=256)

tmp.run_all_pixels()