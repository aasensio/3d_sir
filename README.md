# sir3d

## Introduction

SIR3D is a parallel synthesis code for the synthesis of LTE lines based on SIR.

## Installation

Download the code and type:


    python setup.py install

If you want to work on the code it is much comfortable to install a development version
by typing:


    python setup.py develop

You need a working GFortran compiler available.

## Features

- It uses MPI in a master-agent topology for the fast calculation of Stokes profiles in 3D cubes
- It can be easily adapted to read the output of different 3D MHD codes. Currently we allow for MuRaM cubes.
- It uses a modern equation of state to compute the electron pressure from the temperature and total pressure.
- It uses a modern continuum opacity package to compute the optical depth axis needed for the synthesis.
- It interpolates the model to a fixed optical depth scale to output isotau surfaces.
- It scales linearly with the number of available agents if not dominated by I/O
  processes.
- It can also be used to spectrally and spatially degrade synthesis.

## Example

### Non-parallel

    import sir3d

    iterator = sir3d.Iterator(use_mpi=False)

    mod = sir3d.Model('rempel.ini', rank=iterator.get_rank())
    iterator.use_model(model=mod)

    iterator.run_all_pixels()

You first instantiate an `Iterator` with `use_mpi=False` to work in serial mode. Then instantiate the `Model` passing the configuration file. Finally, we make the iterator use the model and finally synthesize all pixels.


### Parallel using MPI


    iterator = sir3d.Iterator(use_mpi=True, batch=256)

    mod = sir3d.Model('rempel.ini', rank=iterator.get_rank())
    iterator.use_model(model=mod)

    iterator.run_all_pixels()

You first instantiate an `Iterator` with `use_mpi=True` to work in parallel mode. You also need to pass the `batch`. Note that since synthesizing a single pixel is quite fast, you can be dominated by I/O processes if `batch` is very small so we recommend to use a sufficiently large value (which also depends on the number of available agents). Then again instantiate the `Model` passing the configuration file. Finally, we make the iterator use the model and finally synthesize all pixels.

### Subcubes

    iterator.run_all_pixels(rangex=[0,100], rangey=[0,100])

Subcubes can be synthesized by providing the `rangex` and `rangey` arguments to the `run_all_pixels` method. If any of those are absent or `None`, the full range along the specific direction is used.

## Configuration

Here is an example of a configuration file, which should be easy to understand.

    # SIR3D configuration File
    [General]
    Stokes output = '/scratch1/3dcubes/output_stokes.h5'
    Interpolated model output = '/scratch1/3dcubes/output_model.h5'
    Interpolate tau = 0.0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0
    EOS = 'SIR'

    [Spectral regions]
        [[Region 1]]
        Name = '6301-6302'
        Wavelength range = 6300.8521, 6303.3119
        N. wavelengths = 458
        Spectral lines = 200, 201

    [Atmosphere]
    # Remember that "y" is the vertical direction in MURAM simulations but it is the vertical in this code
    Type = 'MURAM'
    Dimensions = 1536, 128, 1536
    deltaz = 8e5
    Maximum tau = 2.0
    Tau delta = 0.1
    Temperature = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/temp.float'
    Pressure = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/pres.float'
    Density = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/dens.float'
    vz = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/vely.float'
    Bx = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/magz.float'
    By = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/magx.float'
    Bz = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/magy.float'


        [[Multipliers]]
        Bx = 1.0
        By = 1.0
        Bz = 1.0
        vz = 1.0

Here is a decription of each parameter:

- `Stokes output`: HDF5 file with the output
- `Interpolated model output`: HDF5 file with the output model interpolated to
  certain optical depth surfaces
- `Interpolate tau`: logarithm of the optical depth at 500 nm where to output
  the surfaces
- `EOS`: specific equation of state to use. It can be either `SIR` (preferred
  option for consistency reasons) or `MANCHA` (the one used in the MANCHA code)
- `Name`: name used for each spectral region
- `Wavelength range`: starting and ending wavelengths used for the synthesis of
  each spectral region
- `N. wavelengths`: number of wavelengths samples to consider in the range
- `Spectral lines`: indices of the spectral lines to consider. The lines are
  listed on the `LINEAS` file
- `Type`: type of input model atmosphere. For the moment the code only
  synthesizes Stokes profiles in `MURAM` snapshots but it is easy to adapt it to
  other formats.
- `Dimensions`: number of (x,y,z) samples of the cube
- `deltaz`: vertical size of the step in the simulation in km
- `Maximum tau`: logarithmic optical depths above this value are considered
  thermalized and removed from the synthesis
- `Tau delta`: maximum optical depth step during the integration of the
  radiative transfer equation. Vertical columns are reinterpolated to this step
- `Temperature`: path to the file with the temperature cube
- `Pressure`: path to the file with the pressure cube
- `Density`: path to the file with the density cube
- `vz`: path to the file with the vz cube (vertical velocity)
- `Bx`: path to the file with the Bx cube (horizontal x component)
- `By`: path to the file with the By cube (horizontal y component)
- `Bz`: path to the file with the Bz cube (vertical z component)
- `Multipliers`: these numbers will be multiplied by the corresponding `vz`,
  `Bx`, `By` or `Bz` cubes. This is useful because some cubes do not directly
  give the magnetic field

## Degradation

This package can also be used to degrade observations with a spectral and/or
spatial point spread function, also providing the output rebinned for the
desired pixel size.

    import sir3d

    input_stokes = 'cube_stokes.h5'
    input_model = 'cube_model.h5'

    output_spatial_stokes = 'cube_stokes_spat_degraded.h5'
    output_spatial_model = 'cube_model_spat_degraded.h5'

    output_spatial_spectral_stokes = 'cube_stokes_spat_spec_degraded.h5'

    # Read the spatial and spectral PSF from external files
    ...
    
    lambda = np.loadtxt('wavelength_Hinode.txt') - 6301.5080
    lambda_hinode *= 1e3

    hinode_pixel = 0.16 * 725.
    simulation_pixel = 48.0
    zoom = simulation_pixel / hinode_pixel

    tmp = sir3d.psf.PSF(use_mpi=True, batch=256)
    tmp.run_all_pixels_spatial(input_stokes, input_model, output_spatial_stokes, output_spatial_model, spatial_psf=psf_spatial, zoom_factor=zoom)
    tmp.run_all_pixels_spectral(input_stokes, output_spectral_stokes, spectral_psf=psf_spectral, final_wavelength_axis=lambda_hinode)
    
You first instantiate the `sir3d.psf.PSF` iterator like when you do synthesis.

The method `run_all_pixels_spatial` of this class carries out the spatial smearing
for the Stokes and model cubes. Note that, although this smearing is well
defined for the Stokes parameters, it is not for the model. What the code does
is convolve the monochromatic Stokes images with the spatial PSF and then rebin
to the desired pixel size. Additionally, it only rebins the model without using
any PSF.

  - `input_stokes`: file with the Stokes synthesis
  - `input_model`: file with the model generated during the synthesis
  - `output_spatial_stokes`: file with the output smeared Stokes profiles
  - `output_spatial_model`: file with the output rebinned model
  - `spatial_psf`: 2D numpy array with the PSF
  - `zoom_factor`: ratio simulation_pixelsize/output_pixelsize

The method `run_all_pixels_spectral` of this class carries out the spectral
smearing of the Stokes profiles.

  - `input_stokes`: file with the input Stokes profiles
  - `output_spectral_stokes`: file with the output spectrally smeared Stokes profiles
  - `spectral_psf`: 2D numpy array with the spectral PSF. The first column is
    the wavelength in mA. The second one is the PSF.
  - `final_wavelength_axis`: final wavelength axis in mA

## Dependencies

- numpy
- mpi4py
- h5py
- scipy
- tqdm
- scikit-image
- pyfftw
