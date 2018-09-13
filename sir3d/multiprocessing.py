import numpy as np
try:
    from mpi4py import MPI
    _mpi_available = True
except:
    _mpi_available = False

from enum import IntEnum
import h5py
from hazel.codes import hazel_code
from tqdm import tqdm, trange
import logging
from ipdb import set_trace as stop

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

class Iterator(object):
    def __init__(self, use_mpi=False, batch=1, n_batches=None):
        
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
        self.stop_after_n_batches = n_batches

    def get_rank(self, n_agents=0):        
        if (self.use_mpi):
            if (n_agents >= self.size):
                raise Exception("Number of requested agents {0} is >= number number of available cores ({1})".format(n_agents, size))        
        return self.rank
    
    def use_model(self, model=None):
        
        # Then broadcast        
        if (self.use_mpi):
            if (self.rank == 0):
                self.model = model

                self.logger.info('Broadcasting models to all agents')

                self.comm.Barrier()
                self.comm.bcast(self.model, root=0)                
                self.comm.Barrier()                
            else:
                model = None

                self.comm.Barrier()
                self.model = self.comm.bcast(model, root=0)
                self.comm.Barrier()

                self.model.init_sir_agents()
                            
        else:
            self.model = model

        if (self.rank == 0):
            self.logger.info('All agents ready')

    def nonmpi_work(self, rangex, rangey):
        """
        Do the synthesis/inversion for all pixels in the models

        Parameters
        ----------
        model : model
            Model to be synthesized

        Returns
        -------
        None
        """

        if (self.model.atmosphere_type == 'MURAM'):
            self.T = np.memmap(self.model.T_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.P = np.memmap(self.model.P_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.rho = np.memmap(self.model.rho_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.vz = np.memmap(self.model.vz_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.Bx = np.memmap(self.model.Bx_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.By = np.memmap(self.model.By_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.Bz = np.memmap(self.model.Bz_file, dtype='float32', mode='r', shape=self.model.model_shape)

            self.n_pixels = self.model.nx * self.model.nz

        self.f_stokes_out = h5py.File(self.model.output_file, 'w')
        self.stokes_db = self.f_stokes_out.create_dataset('stokes', (self.model.nx, self.model.nz, 4, self.model.n_lambda_sir))
        self.lambda_db = self.f_stokes_out.create_dataset('lambda', (self.model.n_lambda_sir,))

        # If we want to extract a model sampled at selected taus
        interpolate_model = False
        if (self.model.interpolated_model_filename is not None):
            self.f_model_out = h5py.File(self.model.interpolated_model_filename, 'w')
            self.model_db = self.f_model_out.create_dataset('model', (self.model.nx, self.model.nz, 7, self.model.n_tau))
            interpolate_model = True

        
        for ix in trange(self.model.nx, desc='x'):            
            for iz in trange(self.model.nz, desc='x'):

                if (interpolate_model):
                    stokes, model = self.model.synth(self.T[ix,:,iz].astype('float64'), self.P[ix,:,iz].astype('float64'), 
                        self.rho[ix,:,iz].astype('float64'), self.vz[ix,:,iz].astype('float64'), self.Bx[ix,:,iz].astype('float64'), 
                        self.By[ix,:,iz].astype('float64'), self.Bz[ix,:,iz].astype('float64'), interpolate_model=interpolate_model)

                    self.stokes_db[ix,iz,:,:] = stokes[1:,:]
                    self.model_db[ix,iz,:,:] = model

                else:
                    stokes = self.model.synth(self.T[ix,:,iz].astype('float64'), self.P[ix,:,iz].astype('float64'), 
                        self.rho[ix,:,iz].astype('float64'), self.vz[ix,:,iz].astype('float64'), self.Bx[ix,:,iz].astype('float64'), 
                        self.By[ix,:,iz].astype('float64'), self.Bz[ix,:,iz].astype('float64'), interpolate_model=interpolate_model)

                    self.stokes_db[ix,iz,:,:] = stokes[1:,:]

        self.lambda_db[:] = stokes[0,:]

        self.f_stokes_out.close()
        self.f_model_out.close()
                                            

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
        
        if (self.model.atmosphere_type == 'MURAM'):
            self.T = np.memmap(self.model.T_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.P = np.memmap(self.model.P_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.rho = np.memmap(self.model.rho_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.vz = np.memmap(self.model.vz_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.Bx = np.memmap(self.model.Bx_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.By = np.memmap(self.model.By_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.Bz = np.memmap(self.model.Bz_file, dtype='float32', mode='r', shape=self.model.model_shape)


            if (rangex is not None):
                x = np.arange(rangex[0], rangex[1])
            else:
                x = np.arange(self.model.nx)

            if (rangey is not None):
                y = np.arange(rangey[0], rangey[1])
            else:
                y = np.arange(self.model.nz)

            self.n_pixels = len(x) * len(y)

            self.n_batches = self.n_pixels // self.batch
            
            X, Y = np.meshgrid(x, y)
            X = X.flatten()
            Y = Y.flatten()
            
            divX = np.array_split(X, self.n_batches)
            divY = np.array_split(Y, self.n_batches)

        self.f_stokes_out = h5py.File(self.model.output_file, 'w')
        self.stokes_db = self.f_stokes_out.create_dataset('stokes', (self.model.nx, self.model.nz, 4, self.model.n_lambda_sir))
        self.lambda_db = self.f_stokes_out.create_dataset('lambda', (self.model.n_lambda_sir,))

        # If we want to extract a model sampled at selected taus
        interpolate_model = False
        if (self.model.interpolated_model_filename is not None):
            self.f_model_out = h5py.File(self.model.interpolated_model_filename, 'w')
            self.model_db = self.f_model_out.create_dataset('model', (self.model.nx, self.model.nz, 7, self.model.n_tau))
            interpolate_model = True
                
        
        # Loop over all pixels doing the synthesis/inversion and saving the results
        task_index = 0
        num_workers = self.size - 1
        closed_workers = 0
        self.last_received = 0
        self.last_sent = 0

        self.logger.info("Starting calculation with {0} workers and {1} batches".format(num_workers, self.n_batches))

        with tqdm(total=self.n_batches, ncols=140) as pbar:            
            while (closed_workers < num_workers):
                data_received = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
                source = self.status.Get_source()
                tag = self.status.Get_tag()
            
                if tag == tags.READY:
                    # Worker is ready, send a task
                    if (task_index < self.n_batches):

                        ix = divX[task_index]
                        iz = divY[task_index]
                        data_to_send = {'index': task_index, 'indX': ix, 'indY': iz, 'interpolate': interpolate_model}
                        
                        data_to_send['model'] = [self.T[ix,:,iz].astype('float64'), self.P[ix,:,iz].astype('float64'), 
                            self.rho[ix,:,iz].astype('float64'), self.vz[ix,:,iz].astype('float64'), self.Bx[ix,:,iz].astype('float64'), 
                            self.By[ix,:,iz].astype('float64'), self.Bz[ix,:,iz].astype('float64')]
                    
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

                    if (interpolate_model):
                        model = data_received['model']
                        for i in range(len(indX)):
                            self.model_db[indX[i],indY[i],:,:] = model[i,:,:]
                    
                    for i in range(len(indX)):
                        self.stokes_db[indX[i],indY[i],:,:] = stokes[i,1:,:]
                                                    
                    self.last_received = '{0}->{1}'.format(index, source)
                    pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                elif tag == tags.EXIT:                    
                    closed_workers += 1

        self.lambda_db[:] = stokes[0,0,:]
        self.f_stokes_out.close()
        self.f_model_out.close()

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
                interpolate_model = data_received['interpolate']

                data_to_send = {'index': task_index, 'indX': indX, 'indY': indY}

                T, P, rho, vz, Bx, By, Bz = data_received['model']

                if (interpolate_model):
                    stokes, model = self.model.synth2d(T, P, rho, vz, Bx, By, Bz, interpolate_model=interpolate_model)
                    data_to_send['model'] = model
                else:
                    stokes = self.model.synth2d(T, P, rho, vz, Bx, By, Bz, interpolate_model=None)

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