import numpy as np
try:
    from mpi4py import MPI
    _mpi_available = True
except:
    _mpi_available = False

from enum import IntEnum
import h5py
from tqdm import tqdm, trange
import logging
import scipy.interpolate
import skimage.transform
import pyfftw.interfaces as fft
# from ipdb import set_trace as stop


class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

class PSF(object):
    def __init__(self, input_stokes, input_model, output_spatial_stokes, output_spatial_model, output_spatial_spectral_stokes,  
        spatial_psf=None, spectral_psf=None, final_wavelength_axis=None, zoom_factor=None, use_mpi=True, batch=256):
        
        # Initializations and preliminaries        
        self.use_mpi = use_mpi
        
        if (self.use_mpi):
            if (not _mpi_available):
                raise Exception("You need MPI and mpi4py installed in your system to use this option.")
            self.comm = MPI.COMM_WORLD   # get MPI communicator object
            self.size = self.comm.size        # total number of processes
            self.rank = self.comm.rank        # rank of this process
            self.status = MPI.Status()   # get MPI status object            

            if (self.size == 1):
                raise Exception("You do not have agents available or you need to start the code with mpiexec.")
        else:
            self.rank = 0            

        self.logger = logging.getLogger("iterator")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.batch = batch

        self.lambda_final = final_wavelength_axis
        self.n_lambda_new = len(self.lambda_final)

        self.zoom_factor = zoom_factor

        if (self.rank == 0):
            
            self.input_stokes = h5py.File(input_stokes, 'r')
            self.input_cube = h5py.File(input_model, 'r')
            
            self.output_spatial_stokes = h5py.File(output_spatial_stokes, 'w')
            self.output_spatial_model = h5py.File(output_spatial_model, 'w')

            self.output_full_stokes = h5py.File(output_spatial_spectral_stokes, 'w')

            self.nx, self.ny, self.n_stokes, self.n_lambda = self.input_stokes['stokes'].shape
            self.nx, self.ny, self.n_var, self.n_tau = self.input_cube['model'].shape

            self.nx_new = np.round(self.zoom_factor * self.nx)
            self.ny_new = np.round(self.zoom_factor * self.ny)

            self.stokes_spatial = self.output_spatial_stokes.create_dataset('stokes', (self.nx_new, self.ny_new, self.n_stokes, self.n_lambda))
            self.lambda_spatial = self.output_spatial_stokes.create_dataset('lambda', (self.n_lambda,))
            self.model_spatial = self.output_spatial_model.create_dataset('model', (self.n_var, self.n_tau, self.nx_new, self.ny_new))
            
            self.stokes_full = self.output_full_stokes.create_dataset('stokes', (self.nx_new, self.ny_new, self.n_stokes, self.n_lambda_new))
            self.lambda_full = self.output_full_stokes.create_dataset('lambda', (self.n_lambda_new,))
            
            self.spatial_psf = spatial_psf
            self.spectral_psf = spectral_psf
            self.zoom_factor = zoom_factor

            self.lambda_spatial[:] = self.input_stokes['lambda']
    
            if (not self.zoom_factor):
                self.zoom_factor = 1.0    


            # Spatial PSF
            if (self.spatial_psf is not None):
                print("Reading spatial PSF")
                psf_size = self.spatial_psf.shape[0]

                psf = np.zeros((self.nx, self.ny))
                psf[int(self.nx/2.-psf_size/2.+1):int(self.nx/2.+psf_size/2.+1),int(self.ny/2.-psf_size/2.+1):int(self.ny/2.+psf_size/2.+1)] = self.spatial_psf

                psf = np.fft.fftshift(psf)
                self.psf_spatial_fft = np.fft.fft2(psf)

            # Spectral PSF
            if (self.spectral_psf is not None):
                interpolator = scipy.interpolate.interp1d(spectral_psf[:,0], spectral_psf[:,1], bounds_error=False, fill_value=0.0)
                psf_spectral = interpolator(self.input_stokes['lambda'])

                self.psf_spectral_fft = np.fft.fft(psf_spectral)
                
                ind = np.searchsorted(self.input_stokes['lambda'], self.lambda_final)

                self.delta1 = (self.input_stokes['lambda'][:][ind+1] - self.lambda_final) / (self.input_stokes['lambda'][:][ind+1] - self.input_stokes['lambda'][:][ind])
                self.delta2 = (self.lambda_final - self.input_stokes['lambda'][:][ind]) / (self.input_stokes['lambda'][:][ind+1] - self.input_stokes['lambda'][:][ind])

                self.ind = ind

    def get_rank(self, n_agents=0):        
        if (self.use_mpi):
            if (n_agents >= self.size):
                raise Exception("Number of requested agents {0} is >= number number of available cores ({1})".format(n_agents, size))        
        return self.rank

    def broadcast(self):
        if (self.rank == 0):            
            self.comm.Barrier()
            self.comm.bcast(self.psf_spatial_fft, root=0)
            self.comm.bcast(self.psf_spectral_fft, root=0)
            self.comm.bcast(self.delta1, root=0)
            self.comm.bcast(self.delta2, root=0)
            self.comm.bcast(self.ind, root=0)
            self.comm.bcast(self.n_lambda_new, root=0)
            self.comm.Barrier()                
        else:
            self.comm.Barrier()
            self.psf_spatial_fft = self.comm.bcast(None, root=0)
            self.psf_spectral_fft = self.comm.bcast(None, root=0)
            self.delta1 = self.comm.bcast(None, root=0)
            self.delta2 = self.comm.bcast(None, root=0)
            self.ind = self.comm.bcast(None, root=0)
            self.n_lambda_new = self.comm.bcast(None, root=0)
            self.comm.Barrier()
                                                    
    def mpi_master_work(self, rangex, rangey):
        """
        MPI master work

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
    
        # The spatial smearing is done by the master because it is fast
        for i in trange(self.n_stokes, desc='stokes'):
            for j in trange(self.n_lambda, desc='lambda', leave=False):

                if (self.spectral_psf is not None):
                    print('fft1', flush=True)
                    im_fft = fft.numpy_fft.fft2(self.input_stokes['stokes'][:,:,i,j])
                    print('fft2', flush=True)
                    im_conv = np.real(fft.numpy_fft.ifft2(self.psf_spatial_fft * im_fft))
                    print('fft3', flush=True)
                else:
                    im_conv = np.copy(self.input_stokes['stokes'][:,:,i,j])

                if (self.zoom_factor != 1.0):
                    print('hey1', flush=True)
                    im_final = skimage.transform.rescale(im_conv, scale=[self.zoom_factor, self.zoom_factor], order=1)
                    print('hey2', flush=True)
                else:
                    im_final = np.copy(im_conv)

                # stop()

                self.stokes_spatial[:,:,i,j] = im_final

        for i in trange(self.n_var, desc='variable'):
            for j in trange(self.n_tau, desc='nz', leave=False):
                im = self.input_model['cube'][:,:,i,j]

                if (self.zoom_factor != 1.0):
                    # im_final = nd.zoom(im, [zoom_factor, zoom_factor])
                    im_final = skimage.transform.rescale(im, scale=[self.zoom_factor, self.zoom_factor])
                else:
                    im_final = np.copy(im)

                self.model_spatial[:,:,i,j] = im_final


        self.output_spatial_stokes.close()
        self.output_spatial_model.close()

        return

        x = np.arange(self.nx_new)
        y = np.arange(self.nz_new)

        self.n_pixels = len(x) * len(y)

        self.n_batches = self.n_pixels // self.batch
        
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        
        divX = np.array_split(X, self.n_batches)
        divY = np.array_split(Y, self.n_batches)

        task_index = 0
        num_workers = self.size - 1
        closed_workers = 0
        self.last_received = 0
        self.last_sent = 0            
        

        with tqdm(total=self.n_batches, ncols=140) as pbar:
            while (closed_workers < num_workers):
                data_received = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
                source = self.status.Get_source()
                tag = self.status.Get_tag()
            
                if tag == tags.READY:
                    # Worker is ready, send a task
                    if (task_index < self.n_batches):

                        ix = divX[task_index]
                        iy = divY[task_index]
                        data_to_send = {'index': task_index, 'indX': ix, 'indY': iy}

                        data_to_send['model'] = self.stokes_spatial[ix,iy,:,:]
                                                                    
                        self.comm.send(data_to_send, dest=source, tag=tags.START)
                    
                        task_index += 1
                        pbar.update(1)
                        self.last_sent = '{0}->{1}'.format(task_index, source)
                        pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                    else:
                        self.comm.send(None, dest=source, tag=tags.EXIT)
                
                elif tag == tags.DONE:
                    index = data_received['index']
                    stokes = data_received['stokes']
                    indX = data_received['indX']
                    indY = data_received['indY']
                    
                    for i in range(len(indX)):
                        self.stokes_full[indX[i],indY[i],:,:] = stokes[i,:,:]
                                                    
                    self.last_received = '{0}->{1}'.format(index, source)
                    pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                elif tag == tags.EXIT:                    
                    closed_workers += 1


        self.output_full_stokes.close()


    def mpi_agents_work(self):
        """
        MPI agents work

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        while True:
            self.comm.send(None, dest=0, tag=tags.READY)
            data_received = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)

            tag = self.status.Get_tag()
            
            if tag == tags.START:                                
                task_index = data_received['index']
                indX = data_received['indX']
                indY = data_received['indY']
                
                data_to_send = {'index': task_index, 'indX': indX, 'indY': indY}

                n = len(indX)

                stokes = np.zeros((4,self.n_lambda_new,n))

                for i in len(indX):
                    # Compute the convolution using FFT along the wavelength axis
                    f_im = np.fft.fft(f['stokes'][i,:,:], axis=1)
                    
                    tmp = np.real(np.fft.ifft(f_im * self.psf_spectral_fft[None,:], axis=1))

                    # Finally carry out the linear interpolation along the wavelength axis to rebin to the Hinode wavelength axis
                    stokes[i,:,:] = tmp[:,self.ind] * delta1[None,:] + tmp[:,self.ind+1] * delta2[None,:]
                
                data_to_send['stokes'] = stokes
                
                self.comm.send(data_to_send, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break

        self.comm.send(None, dest=0, tag=tags.EXIT)           

    def run_all_pixels(self, rangex=None, rangey=None):
        """
        Run synthesis/inversion for all pixels

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (self.use_mpi):
            if (self.rank == 0):
                self.mpi_master_work(rangex=rangex, rangey=rangey)
            else:
                self.mpi_agents_work()
        else:
            self.nonmpi_work(rangex=rangex, rangey=rangey)