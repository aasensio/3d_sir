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
from . import slant

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

class Iterator(object):
    def __init__(self, use_mpi=False, batch=1, n_batches=None, workers_slant=None, withstokes=True):
        
        # Initializations and preliminaries        
        self.use_mpi = use_mpi
        
        if (self.use_mpi):
            if (not _mpi_available):
                raise Exception("You need MPI and mpi4py installed in your system to use this option.")
            self.comm = MPI.COMM_WORLD   # get MPI communicator object
            self.size = self.comm.size        # total number of processes
            self.rank = self.comm.rank        # rank of this process
            self.status = MPI.Status()   # get MPI status object            

            if (workers_slant is None):
                self.workers_slant = self.size
            else:
                if (workers_slant > self.size):
                    self.workers_slant = self.size
                else:
                    self.workers_slant = workers_slant

            if (self.size == 1):
                print("You have activated mpi but you do not have agents available or you need to start the code with mpiexec.")
                # The code can still run in single-core with a message
                self.use_mpi = False
        else:
            self.rank = 0            

        self.logger = logging.getLogger("Iterator")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.batch = batch
        self.stop_after_n_batches = n_batches
        self.withstokes = withstokes

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

                # self.model.init_sir_agents()
                self.model.init_sir(self.model.spectral_regions_dict)
                            
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

        if (rangex is not None):
            x = np.arange(rangex[0], rangex[1])
        else:
            x = np.arange(self.model.nx)

        if (rangey is not None):
            y = np.arange(rangey[0], rangey[1])
        else:
            y = np.arange(self.model.nz)

        if (self.model.atmosphere_type == 'MURAM'):
            self.T = np.memmap(self.model.T_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.P = np.memmap(self.model.P_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.rho = np.memmap(self.model.rho_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.vz = np.memmap(self.model.vz_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.Bx = np.memmap(self.model.Bx_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.By = np.memmap(self.model.By_file, dtype='float32', mode='r', shape=self.model.model_shape)
            self.Bz = np.memmap(self.model.Bz_file, dtype='float32', mode='r', shape=self.model.model_shape)

            self.n_pixels = self.model.nx * self.model.nz

        # Showing to the user that no stokes will by synthetized
        if not (self.withstokes):
            self.logger.info("Avoiding the synthesis module and only saving the model.")

        # Write stokes file only if withstokes
        if (self.withstokes):
            self.f_stokes_out = h5py.File(self.model.output_file, 'w')
            self.stokes_db = self.f_stokes_out.create_dataset('stokes', (len(x), len(y), 4, self.model.n_lambda_sir))
            self.lambda_db = self.f_stokes_out.create_dataset('lambda', (self.model.n_lambda_sir,))

        # If we want to extract a model sampled at selected taus
        interpolate_model = False
        if (self.model.interpolated_model_filename is not None):
            self.f_model_out = h5py.File(self.model.interpolated_model_filename, 'w')
            self.model_db = self.f_model_out.create_dataset('model', (len(x), len(y), 7, self.model.n_tau))
            interpolate_model = True

        # To save the result in a model with the same size of the range
        for cx, ix in enumerate(tqdm(x, desc='x')):
            for cz, iz in enumerate(tqdm(y, desc='y')):
        # for ix in tqdm(x, desc='x'):
            # for iz in tqdm(y, desc='x'):
                if (self.model.vz_type == 'vz'):
                    vz = self.vz[ix,:,iz]
                else:
                    vz = self.vz[ix,:,iz] / self.rho[ix,:,iz]

                # Generic implementation: always get two outputs (stokes=None when no synthetsis)
                stokes, model = self.model.synth(self.model.deltaz, self.T[ix,:,iz].astype('float64'), self.P[ix,:,iz].astype('float64'), 
                    self.rho[ix,:,iz].astype('float64'), vz.astype('float64'), self.Bx[ix,:,iz].astype('float64'), 
                    self.By[ix,:,iz].astype('float64'), self.Bz[ix,:,iz].astype('float64'), interpolate_model=interpolate_model,withstokes=self.withstokes)

                if (self.withstokes):
                    self.stokes_db[cx,cz,:,:] = stokes[1:,:]
                
                self.model_db[cx,cz,:,:] = model


        if (self.withstokes):
            self.lambda_db[:] = stokes[0,:]
            self.f_stokes_out.close()
        
        self.f_model_out.close()

        # To fix print problem of tqdm
        print()                                 

    def mpi_master_work_synth(self, rangex, rangey):
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

            if (self.model.need_slant):
                self.vx = np.memmap(self.model.vx_file, dtype='float32', mode='r', shape=self.model.model_shape)
                self.vy = np.memmap(self.model.vy_file, dtype='float32', mode='r', shape=self.model.model_shape)


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

        # Showing to the user that no stokes will by synthetized
        if not (self.withstokes):
            self.logger.info("Avoiding the synthesis module and only saving the model")

        # Write stokes file only if withstokes
        if (self.withstokes):
            self.f_stokes_out = h5py.File(self.model.output_file, 'w')
            self.stokes_db = self.f_stokes_out.create_dataset('stokes', (self.model.nx, self.model.nz, 4, self.model.n_lambda_sir), dtype='float32')
            self.lambda_db = self.f_stokes_out.create_dataset('lambda', (self.model.n_lambda_sir,), dtype='float32')

        # If we want to extract a model sampled at selected taus
        interpolate_model = False
        if (self.model.interpolated_model_filename is not None):
            self.f_model_out = h5py.File(self.model.interpolated_model_filename, 'w')
            self.model_db = self.f_model_out.create_dataset('model', (self.model.nx, self.model.nz,7, self.model.n_tau), dtype='float32')
            interpolate_model = True

        ##############################################    
        # Slant models if needed
        ##############################################
        if (self.model.need_slant):

            self.thetaX = np.arccos(self.model.mux)
            self.thetaY = np.arccos(self.model.muy)

            # Shift in both directions at each height in pixel units
            self.shiftx = self.model.deltaz * np.tan(self.thetaX) / self.model.deltaxy
            self.shifty = self.model.deltaz * np.tan(self.thetaY) / self.model.deltaxy

            task_index = 0
            num_workers = self.workers_slant - 1 #self.size - 1
            closed_workers = 0
            self.last_received = 0
            self.last_sent = 0

            self.logger.info("Starting slanting of models with {0} workers and {1} heights".format(num_workers, self.model.ny))

            self.T_new = np.empty(self.T.shape, dtype=self.T.dtype)
            self.P_new = np.empty(self.P.shape, dtype=self.P.dtype)
            self.rho_new = np.empty(self.rho.shape, dtype=self.rho.dtype)
            self.vx_new = np.empty(self.vx.shape, dtype=self.vx.dtype)
            self.vy_new = np.empty(self.vy.shape, dtype=self.vy.dtype)
            self.vz_new = np.empty(self.vz.shape, dtype=self.vz.dtype)
            self.Bx_new = np.empty(self.Bx.shape, dtype=self.Bx.dtype)
            self.By_new = np.empty(self.By.shape, dtype=self.By.dtype)
            self.Bz_new = np.empty(self.Bz.shape, dtype=self.Bz.dtype)

            with tqdm(total=self.model.ny, ncols=140) as pbar:
                while (closed_workers < num_workers):
                    data_received = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
                    source = self.status.Get_source()
                    tag = self.status.Get_tag()

                    if tag == tags.READY:
                        # Worker is ready, send a task
                        if (task_index < self.model.ny):
                            
                            data_to_send = {'index': task_index, 
                            'shX': self.shiftx[task_index],
                            'shY': self.shifty[task_index],
                            'mux': self.model.mux,
                            'muy': self.model.muy}
                            
                            data_to_send['model'] = [
                                self.T[:,task_index,:], self.P[:,task_index,:], 
                                self.rho[:,task_index,:], self.vx[:,task_index,:],
                                self.vy[:,task_index,:], self.vz[:,task_index,:], 
                                self.Bx[:,task_index,:], self.By[:,task_index,:], 
                                self.Bz[:,task_index,:]]
                        
                            self.comm.send(data_to_send, dest=source, tag=tags.START)
                        
                            task_index += 1
                            pbar.update(1)
                            self.last_sent = '{0} to {1}'.format(task_index, source)
                            pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                        else:
                            self.comm.send(None, dest=source, tag=tags.EXIT)
                    
                    elif tag == tags.DONE:
                        index = data_received['index']
                        model = data_received['model']

                        self.T_new[:, index, :] = model[0]
                        self.P_new[:, index, :] = model[1]
                        self.rho_new[:, index, :] = model[2]
                        self.vx_new[:, index, :] = model[3]
                        self.vy_new[:, index, :] = model[4]
                        self.vz_new[:, index, :] = model[5]
                        self.Bx_new[:, index, :] = model[6]
                        self.By_new[:, index, :] = model[7]
                        self.Bz_new[:, index, :] = model[8]

                        self.last_received = '{0} from {1}'.format(index, source)
                        pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                    elif tag == tags.EXIT:                    
                        closed_workers += 1

            del self.T
            del self.P
            del self.rho
            del self.vx
            del self.vy
            del self.vz
            del self.Bx
            del self.By
            del self.Bz
            
            self.deltaz = self.model.deltaz_new
            self.T = self.T_new
            self.P = self.P_new
            self.rho = self.rho_new
            self.vz = self.vz_new
            self.Bx = self.Bx_new
            self.By = self.By_new
            self.Bz = self.Bz_new            

        self.comm.Barrier()
        
        
        #########################################
        # Loop over all pixels doing the synthesis/inversion and saving the results
        #########################################
        task_index = 0
        num_workers = self.size - 1
        closed_workers = 0
        self.last_received = 0
        self.last_sent = 0

        self.logger.info("Starting calculations with {0} workers and {1} batches".format(num_workers, self.n_batches))

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

                        if (self.model.vz_type == 'vz'):
                            vz = self.vz[ix,:,iz]
                        else:
                            vz = self.vz[ix,:,iz] / self.rho[ix,:,iz]
                        
                        data_to_send['model'] = [self.model.deltaz, self.T[ix,:,iz].astype('float64'), self.P[ix,:,iz].astype('float64'), 
                            self.rho[ix,:,iz].astype('float64'), vz.astype('float64'), self.Bx[ix,:,iz].astype('float64'), 
                            self.By[ix,:,iz].astype('float64'), self.Bz[ix,:,iz].astype('float64')]
                    
                        self.comm.send(data_to_send, dest=source, tag=tags.START)
                    
                        task_index += 1
                        pbar.update(1)
                        self.last_sent = '{0} to {1}'.format(task_index, source)
                        pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                    else:
                        self.comm.send(None, dest=source, tag=tags.EXIT)
                
                elif tag == tags.DONE:
                    index = data_received['index']
                    indX = data_received['indX']
                    indY = data_received['indY']

                    if (interpolate_model):
                        model = data_received['model']
                        for i in range(len(indX)):
                            self.model_db[indX[i],indY[i],:,:] = model[i,:,:]
                    
                    if (self.withstokes):
                        stokes = data_received['stokes']
                        for i in range(len(indX)):
                            self.stokes_db[indX[i],indY[i],:,:] = stokes[i,1:,:]

                                                    
                    self.last_received = '{0} from {1}'.format(index, source)
                    pbar.set_postfix(sent=self.last_sent, received=self.last_received)

                elif tag == tags.EXIT:                    
                    closed_workers += 1

        if (self.withstokes):
            self.lambda_db[:] = stokes[0,0,:]
            self.f_stokes_out.close()
        
        self.f_model_out.close()




    def mpi_agents_work_synth(self):
        """
        MPI agents work

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        ###############################
        # Slant models if needed
        ###############################
        if (self.model.need_slant):

            if (self.rank <= self.workers_slant):
                while True:
                    self.comm.send(None, dest=0, tag=tags.READY)
                    data_received = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)

                    tag = self.status.Get_tag()
                    
                    if tag == tags.START:                                
                        task_index = data_received['index']
                        shX = data_received['shX']
                        shY = data_received['shY']
                        xmu = data_received['mux']
                        ymu = data_received['muy']

                        ysign = np.sign(xmu)
                        xsign = np.sign(ymu)
                        
                        ymu2 = np.sqrt(1.0 - ymu**2)
                        xmu2 = np.sqrt(1.0 - xmu**2)
                        
                        data_to_send = {'index': task_index}

                        T, P, rho, vx, vy, vz, Bx, By, Bz = data_received['model']

                        T = slant.fftshift_image(T, dx=shX, dy=shY, useLog=True)
                        P = slant.fftshift_image(P, dx=shX, dy=shY, useLog=True)
                        rho = slant.fftshift_image(rho, dx=shX, dy=shY, useLog=True)
                        vx = slant.fftshift_image(vx, dx=shX, dy=shY, useLog=False)
                        vy = slant.fftshift_image(vy, dx=shX, dy=shY, useLog=False)
                        vz = slant.fftshift_image(vz, dx=shX, dy=shY, useLog=False)
                        Bx = slant.fftshift_image(Bx, dx=shX, dy=shY, useLog=False)
                        By = slant.fftshift_image(By, dx=shX, dy=shY, useLog=False)
                        Bz = slant.fftshift_image(Bz, dx=shX, dy=shY, useLog=False)

                        # Project
                        vz1 = vz * xmu * ymu - ysign * vy * xmu * ymu2 - xsign * vx * xmu2
                        vy1 = vy * ymu + vz * ymu2 * ysign
                        vx1 = vx * xmu + (vz * ymu - ysign * vy * ymu2) * xmu2 * xsign

                        Bz1 = Bz * xmu * ymu - ysign * By * xmu * ymu2 - xsign * Bx * xmu2
                        By1 = By * ymu + Bz * ymu2 * ysign
                        Bx1 = Bx * xmu + (Bz * ymu - ysign * By * ymu2) * xmu2 * xsign

                        model = [T, P, rho, vx1, vy1, vz1, Bx1, By1, Bz1]

                        data_to_send['model'] = model
                        
                        self.comm.send(data_to_send, dest=0, tag=tags.DONE)
                    elif tag == tags.EXIT:
                        break

                self.comm.send(None, dest=0, tag=tags.EXIT)

        self.comm.Barrier()
        
        ###############################
        # Synthesis
        ###############################
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

                z, T, P, rho, vz, Bx, By, Bz = data_received['model']

                # if (interpolate_model):
                #     stokes, model = self.model.synth2d(z, T, P, rho, vz, Bx, By, Bz, interpolate_model=interpolate_model)
                #     data_to_send['model'] = model
                # else:
                #     stokes = self.model.synth2d(T, P, rho, vz, Bx, By, Bz, interpolate_model=None)
                # data_to_send['stokes'] = stokes

                # Generic implementation: always get two outputs (stokes=None when no synthetsis)
                stokes, model = self.model.synth2d(z, T, P, rho, vz, Bx, By, Bz, interpolate_model=interpolate_model,withstokes=self.withstokes)
                if (interpolate_model): data_to_send['model'] = model
                if (self.withstokes): data_to_send['stokes'] = stokes
                
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
                self.mpi_master_work_synth(rangex=rangex, rangey=rangey)
            else:
                self.mpi_agents_work_synth()
        else:
            self.nonmpi_work(rangex=rangex, rangey=rangey)