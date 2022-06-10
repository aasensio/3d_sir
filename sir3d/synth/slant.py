import numpy as np
import sys

"""
Model slant+projection tools
"""


# *****************************************************************************

def fftshift_image(im_in, dy=0.0, dx=0.0, useLog=False):
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
    if (useLog):
        im = np.log(im_in)
    else:
        im = im_in
        
    ny, nx = im.shape
    me = im.mean()
    st = np.std(im)
    
    im = (im-me)/st  
    
    #
    # FFT of the input image, check for periodicity
    #
    
    ft = np.fft.rfft2(im)
        
    #
    # get spatial frequency mesh, the x-axis has only the positive frequencies
    # because the input data were non-complex numbers, so the negative part is
    # redundant.
    #
    fx, fy = np.meshgrid(np.fft.rfftfreq(nx), np.fft.fftfreq(ny))
    
    #
    # Multiply by exponential phase factor and return to image space
    #

    if (useLog):
        return np.exp((np.real((np.fft.irfft2(ft * np.exp(-2j*np.pi*(fx*-dx + fy*-dy))))[0:ny,0:nx])*st)+me)
    else:
        return (np.real((np.fft.irfft2(ft * np.exp(-2j*np.pi*(fx*-dx + fy*-dy))))[0:ny,0:nx])*st)+me



def project_field(vy, vx, vz, xmu, ymu):
    """
    Projects vector variables into the new LOS.
    This routines should be applied to velocities and magnetic field after performing the slant.
    The projection is applied in-place, so it does not return anything, but it overwrites the input
    arrays.

    """
    ysign = np.sign(xmu)
    xsign = np.sign(ymu)
    
    ymu2 = np.sqrt(1.0 - ymu**2)
    xmu2 = np.sqrt(1.0 - xmu**2)
    
    vz1 = vz * xmu * ymu - ysign * vy * xmu * ymu2 - xsign * vx * xmu2
    vy1 = vy * ymu + vz * ymu2 * ysign
    vx1 = vx * xmu + (vz * ymu - ysign * vy * ymu2) * xmu2 * xsign

    return vy1, vx1, vz1