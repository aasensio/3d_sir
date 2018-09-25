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
- It intepolates the model to a fixed optical depth scale to output isotau surfaces.
- It scales linearly with the number of available agents if not dominated by I/O processes.

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
    Delta tau min = 0.1
    Invert magnetic field = True
    Temperature = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/temp.float'
    Pressure = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/pres.float'
    Density = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/dens.float'
    vz = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/vely.float'
    Bx = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/magz.float'
    By = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/magx.float'
    Bz = '/scratch1/3dcubes/rempel/spot_32x16x32km_ng/magy.float'

## Dependencies

- numpy
- mpi4py
- h5py
- scipy
- tqdm