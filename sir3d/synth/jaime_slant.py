import numpy as np
import sys

"""
Model slant+projection tools
"""


# *****************************************************************************

def fftshift_image(im_in, dy=0.0, dx=0.0, isPeriodic=True, useLog=False):
    """
    FFTSHIFT_IMAGE, shifts an image by dy, dx pixels using 
    Fourier transforms.
    
    Input:
         im:         2D numpy array with the image (ny,nx)

         dy:         shift along the leftmost axis in pixels

         dx:         shift along the rightmost axis in pixels

         isPeriodic: if the image is not periodic, it places it
                     into a (2*Ny,2*Nx) container so it can make
                     a periodic version of the image. Much slower.


    AUTHOR: J. de la Cruz Rodriguez (ISP-SU 2020)
    """

    #
    # scale image to numbers and amplitudes around 1
    #
    if(useLog):
        im = np.log(np.ascontiguousarray(im_in, dtype='float64'))
    else:
        im = np.ascontiguousarray(im_in, dtype='float64')
        
    ny, nx = im.shape
    me = im.mean()
    st = np.std(im)
    
    im = (im-me)/st  
    
    #
    # FFT of the input image, check for periodicity
    #
    if(isPeriodic):
        ny1, nx1 = im.shape
        ft = np.fft.rfft2(im)
    else:
        ny, nx = im.shape
        ft = np.zeros((2*ny, 2*nx), dtype='float64', order='c')

        ft[0:ny,0:nx] = im
        ft[0:ny,nx::] = im[:,::-1]
        ft[ny::,0:nx] = im[::-1,:]
        ft[ny::,nx::] = im[::-1,::-1]

        ny1, nx1 = ft.shape        
        ft =  np.fft.rfft2(ft)

        
    #
    # get spatial frequency mesh, the x-axis has only the positive frequencies
    # because the input data were non-complex numbers, so the negative part is
    # redundant.
    #
    fx, fy = np.meshgrid(np.fft.rfftfreq(nx1), np.fft.fftfreq(ny1))

    
    #
    # Multiply by exponential phase factor and return to image space
    #

    if(useLog):
        return np.exp((np.real((np.fft.irfft2(ft * np.exp(-2j*np.pi*(fx*-dx + fy*-dy))))[0:ny,0:nx])*st)+me)
    else:
        return (np.real((np.fft.irfft2(ft * np.exp(-2j*np.pi*(fx*-dx + fy*-dy))))[0:ny,0:nx])*st)+me


# *****************************************************************************

class SlantModel:
    """
    SlantModel class
    it computes the shift that needs to by applied to each layer in order to simulate an
    off-center observations with MHD models. Assumes axis ordering (nz, ny, nx) in python 
    ordering.

    Source: Inspired in M. Carlsson's "mu_atmos3d.pro", but with extra functionality
            (can slant in both axes) and it keeps z=0 unshifted instead of the upper 
            boundary. Plane shifts are performed by adding a global phase in Fourier
            space.
    
    Coded by J. de la Cruz Rodriguez (ISP-SU 2020)
    """
    
    # ----------------------------------------------------------------------------------
    
    def __init__(self, z, y, x, mu_x = 1.0, mu_y=1.0):
        """

        This constructor precomputes the shift that needs to be applied to each layer and the
        new Z-scale.
        
        Input:
             z: 1D array with the z-scale of the model
             x: 1D array with the x-scale of the model
             y: 1D array with the y-scale of the model

             mu_x: heliocentric distance in the x axis (mu_x = cos(theta_x))
             mu_y: heliocentric distance in the y axis (mu_y = cos(theta_y))

        """
        

        #
        # If z=0 is in the photosphere, we can keep z=0 unshifted
        #
        self.idx = np.argmin(np.abs(z))


        
        #
        # Precompute shift for each plane of the model
        #
        zx2 = (z - z[self.idx]) / np.abs(mu_x) + z[self.idx]    
        self.shift_x = -np.sign(mu_x)*(zx2 - zx2[self.idx]) * np.sqrt(1.0 - mu_x**2) / x.max() * (x.size - 1.0)

        zy2 = (z - z[self.idx]) / np.abs(mu_y) + z[self.idx]
        self.shift_y = -np.sign(mu_y)*(zy2 - zy2[self.idx]) * np.sqrt(1.0 - mu_y**2) / y.max() * (y.size - 1.0)
        
        #
        # stretch z-axis and store it
        #
        xangle = np.arccos(mu_x)
        yangle = np.arccos(mu_y)

        tmu = np.cos(np.sqrt(xangle**2 + yangle**2))
        print("SlantModel::__init__: mu={0}".format(tmu))


        self.z_new = (z - z[self.idx]) / abs(tmu) + z[self.idx]
        
        self.mu_x = mu_x
        self.mu_y = mu_y
        
        
    # ----------------------------------------------------------------------------------

    def slantVariable(self, var, useLog=False):
        """
        Slants a variable of the model. 
        This routine should be applied to all variables. 
        The shifting is performed by applying a phase shift in Fourier domain.

        Input:
             var: 3D cubes with dimensions (nz, ny, nx)

        """
        var1 = np.empty(var.shape, dtype=var.dtype)
        
        nz, ny, nx = var.shape
        if(nz != self.z_new.size):
            print("slantVariable: ERROR, the object was initialized with nz={0}, but the provided cube has nz={1}".format(self.z_new.size, nz))

        per = 0; oper = -1; scl = 100.0 / (nz-1.0)
        for kk  in range(nz):
            if((np.abs(self.shift_y[kk])<1.e-3) and (np.abs(self.shift_x[kk])<1.e-3)):
                var1[kk] = var[kk]
            else:
                var1[kk] = fftshift_image(var[kk], dy=self.shift_y[kk], dx=self.shift_x[kk], isPeriodic = True, useLog=useLog)
                
            per = int(kk*scl)
            if(per != oper):
                oper = per
                sys.stdout.write("\rSlantModel::slantVariable: {0}{1}".format(per, "%"))
                sys.stdout.flush()
        sys.stdout.write("\rSlantModel::SlantVariable: {0}{1}\n".format(100, "%"))

        return var1

    # ----------------------------------------------------------------------------------

    def get_new_z(self):
        """
        Returns the new z-scale of the slanted model
        """
        
        return self.z_new*1
    
    # ----------------------------------------------------------------------------------

    def project_field(self, vy, vx, vz):
        """
        Projects vector variables into the new LOS.
        This routines should be applied to velocities and magnetic field after performing the slant.
        The projection is applied in-place, so it does not return anything, but it overwrites the input
        arrays.

        """
        ysign = np.sign(self.mu_y)
        xsign = np.sign(self.mu_x)

        xmu = self.mu_x
        ymu = self.mu_y
        
        ymu2 = np.sqrt(1.0 - ymu**2)
        xmu2 = np.sqrt(1.0 - xmu**2)
        
        vz1 = vz * xmu * ymu - ysign * vy * xmu * ymu2 - xsign * vx * xmu2
        vy1 = vy * ymu + vz * ymu2 * ysign
        vx1 = vx * xmu + (vz * ymu - ysign * vy * ymu2) * xmu2 * xsign

        vz[:] = vz1
        vx[:] = vx1
        vy[:] = vy1
        
    # ----------------------------------------------------------------------------------

    
    # *****************************************************************************
