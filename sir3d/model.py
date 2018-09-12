from collections import OrderedDict
from sir3d import sir_code
from sir3d.configuration import Configuration
import numpy as np
import copy
import os
from pathlib import Path
import scipy.stats
import scipy.special
import warnings
import logging
import h5py
import scipy.integrate as integ

from ipdb import set_trace as stop

__all__ = ['Model']

class Model(object):
    def __init__(self, config=None, rank=0):

        if (rank != 0):
            return
        
        self.logger = logging.getLogger("model")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.rank = rank

        self.macroturbulence = 0.0

        ch = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        if (config is not None):
            self.configuration = Configuration(config)

            self.use_configuration(self.configuration.config_dict)

        if (self.rank == 0):

            self.logger.info('Reading EOS')

            filename = os.path.join(os.path.dirname(__file__), 'data/eos.h5')
            f = h5py.File(filename, 'r')
            self.T_eos = np.log10(f['T'][:])
            self.P_eos = np.log10(f['P'][:])
            self.Pe_eos = np.log10(f['Pel'][:])
            f.close()
            
            self.logger.info('Reading kappa5000')
            self.T_kappa5 = np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50, 
                3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70, 
                3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00, 
                4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50, 
                4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00, 
                5.05, 5.10, 5.15, 5.20, 5.25, 5.30 ])

            self.P_kappa5 = np.array([-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6. ,6.5, 7., 7.5, 8. ])

            self.kappa = np.zeros((56,21))

            filename = os.path.join(os.path.dirname(__file__), 'data/kappa.5000.dat')
            f = open(filename, 'r')
                    
            for it in range(56):
                for ip in range(21):
                    self.kappa[it,ip] = float(f.readline().split()[-1])

            f.close()

        
    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)
            
    def use_configuration(self, config_dict):
        """
        Use a configuration file

        Parameters
        ----------
        config_dict : dict
            Dictionary containing all the options from the configuration file previously read

        Returns
        -------
        None
        """

        # Deal with the spectral regions        
        tmp = config_dict['spectral regions']

        # Output file and atmosphere type
        self.output_file = config_dict['general']['output file']
        self.atmosphere_type = config_dict['general']['atmosphere type']

        # Add spectral regions
        self.init_sir(config_dict['spectral regions'])
        
        # Read atmosphere
        if (self.atmosphere_type == 'MURAM'):
            if (self.rank == 0):
                self.model_shape = tuple([int(k) for k in config_dict['atmosphere']['dimensions']])
                self.nx, self.ny, self.nz = self.model_shape
                self.deltaz = float(config_dict['atmosphere']['deltaz']) * np.arange(self.ny)
                self.T_file = config_dict['atmosphere']['temperature']
                self.P_file = config_dict['atmosphere']['pressure']
                self.rho_file = config_dict['atmosphere']['density']
                self.vz_file = config_dict['atmosphere']['vz']
                self.Bx_file = config_dict['atmosphere']['bx']
                self.By_file = config_dict['atmosphere']['by']
                self.Bz_file = config_dict['atmosphere']['bz']

                self.zeros = np.zeros(self.ny)

                self.logger.info('Using MURAM atmosphere')
                self.logger.info(' - T file : {0}'.format(self.T_file))
                self.logger.info(' - P file : {0}'.format(self.P_file))
                self.logger.info(' - rho file : {0}'.format(self.rho_file))
                self.logger.info(' - vz file : {0}'.format(self.vz_file))
                self.logger.info(' - Bx file : {0}'.format(self.Bx_file))
                self.logger.info(' - By file : {0}'.format(self.By_file))
                self.logger.info(' - Bz file : {0}'.format(self.Bz_file))
        
                            
    def init_sir(self, spectral):
        """
        Initialize SIR for this synthesis
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        """

        filename = os.path.join(os.path.dirname(__file__),'data/LINEAS')
        ff = open(filename, 'r')
        flines = ff.readlines()
        ff.close()
        
            
        f = open('malla.grid', 'w')
        f.write("IMPORTANT: a) All items must be separated by commas.                 \n")
        f.write("           b) The first six characters of the last line                \n")
        f.write("          in the header (if any) must contain the symbol ---       \n")
        f.write("\n")                                                                       
        f.write("Line and blends indices   :   Initial lambda     Step     Final lambda \n")
        f.write("(in this order)                    (mA)          (mA)         (mA)     \n")
        f.write("-----------------------------------------------------------------------\n")

        for k, v in spectral.items():            
            self.logger.info('Adding spectral regions {0}'.format(v['name']))

            left = float(v['wavelength range'][0])
            right = float(v['wavelength range'][1])
            n_lambda = int(v['n. wavelengths'])
            
            delta = (right - left) / n_lambda

            for i in range(len(v['spectral lines'])):
                for l in flines:
                    tmp = l.split()
                    index = int(tmp[0].split('=')[0])                    
                    if (index == int(v['spectral lines'][0])):
                        wvl = float(tmp[2])

            lines = ''
            n_lines = len(v['spectral lines'])
            for i in range(n_lines):
                lines += v['spectral lines'][i]
                if (i != n_lines - 1):
                    lines += ', '

            f.write("{0}            :  {1}, {2}, {3}\n".format(lines, 1e3*(left-wvl), 1e3*delta, 1e3*(right-wvl)))
            f.close()
        
        self.n_lambda_sir = sir_code.init(1, filename)

    def init_sir_agents(self):

        filename = os.path.join(os.path.dirname(__file__),'data/LINEAS')
        self.n_lambda_sir = sir_code.init(1, filename)



    def synth(self, T, P, rho, vz, Bx, By, Bz):

        # Get ltau500 axis
        log_T = np.log10(T)
        log_P = np.log10(P)

        it0 = np.searchsorted(self.T_kappa5, log_T) - 1
        it1 = it0 + 1
        ip0 = np.searchsorted(self.P_kappa5, log_P) - 1
        ip1 = ip0 + 1

        kappa = self.kappa[it0,ip0] * (self.T_kappa5[it1] - log_T) * (self.P_kappa5[ip1] - log_P) + \
                self.kappa[it1,ip0] * (log_T - self.T_kappa5[it0]) * (self.P_kappa5[ip1] - log_P) + \
                self.kappa[it0,ip1] * (self.T_kappa5[it1] - log_T) * (log_P - self.P_kappa5[ip0]) + \
                self.kappa[it1,ip1] * (log_T - self.T_kappa5[it0]) * (log_P - self.P_kappa5[ip0])

        kappa /= ((self.T_kappa5[it1] - self.T_kappa5[it0]) * (self.P_kappa5[ip1] - self.P_kappa5[ip0]))

        chi = (kappa * rho)[::-1]

        tau = integ.cumtrapz(chi,x=self.deltaz)
        ltau = np.log10(np.insert(tau, 0, 0.5*tau[0]))[::-1]

        ind = np.where(ltau < 2.0)[0]

        # Get electron pressure
        it0 = np.searchsorted(self.T_eos, log_T) - 1
        it1 = it0 + 1
        ip0 = np.searchsorted(self.P_eos, log_P) - 1
        ip1 = ip0 + 1

        log_Pe = self.Pe_eos[ip0,it0] * (self.T_eos[it1] - log_T) * (self.P_eos[ip1] - log_P) + \
                self.Pe_eos[ip1,it0] * (log_T - self.T_eos[it0]) * (self.P_eos[ip1] - log_P) + \
                self.Pe_eos[ip0,it1] * (self.T_eos[it1] - log_T) * (log_P - self.P_eos[ip0]) + \
                self.Pe_eos[ip1,it1] * (log_T - self.T_eos[it0]) * (log_P - self.P_eos[ip0])

        log_Pe /= ((self.T_eos[it1] - self.T_eos[it0]) * (self.P_eos[ip1] - self.P_eos[ip0]))

        stokes = sir_code.synth(1, self.n_lambda_sir, ltau[ind], T[ind], 10**log_Pe[ind], self.zeros[ind], vz[ind], Bx[ind], By[ind], Bz[ind], self.macroturbulence)

        return stokes

    def synth2d(self, T, P, rho, vz, Bx, By, Bz):

        n = T.shape[0]

        stokes_out = np.zeros((n,5,self.n_lambda_sir))

        for loop in range(n):

            # Get ltau500 axis
            log_T = np.log10(T[loop,:])
            log_P = np.log10(P[loop,:])

            it0 = np.searchsorted(self.T_kappa5, log_T) - 1
            it1 = it0 + 1
            ip0 = np.searchsorted(self.P_kappa5, log_P) - 1
            ip1 = ip0 + 1

            kappa = self.kappa[it0,ip0] * (self.T_kappa5[it1] - log_T) * (self.P_kappa5[ip1] - log_P) + \
                    self.kappa[it1,ip0] * (log_T - self.T_kappa5[it0]) * (self.P_kappa5[ip1] - log_P) + \
                    self.kappa[it0,ip1] * (self.T_kappa5[it1] - log_T) * (log_P - self.P_kappa5[ip0]) + \
                    self.kappa[it1,ip1] * (log_T - self.T_kappa5[it0]) * (log_P - self.P_kappa5[ip0])

            kappa /= ((self.T_kappa5[it1] - self.T_kappa5[it0]) * (self.P_kappa5[ip1] - self.P_kappa5[ip0]))

            chi = (kappa * rho[loop,:])[::-1]

            tau = integ.cumtrapz(chi, x=self.deltaz)
            ltau = np.log10(np.insert(tau, 0, 0.5*tau[0]))[::-1]

            ind = np.where(ltau < 2.0)[0]

            # Get electron pressure
            it0 = np.searchsorted(self.T_eos, log_T) - 1
            it1 = it0 + 1
            ip0 = np.searchsorted(self.P_eos, log_P) - 1
            ip1 = ip0 + 1

            log_Pe = self.Pe_eos[ip0,it0] * (self.T_eos[it1] - log_T) * (self.P_eos[ip1] - log_P) + \
                    self.Pe_eos[ip1,it0] * (log_T - self.T_eos[it0]) * (self.P_eos[ip1] - log_P) + \
                    self.Pe_eos[ip0,it1] * (self.T_eos[it1] - log_T) * (log_P - self.P_eos[ip0]) + \
                    self.Pe_eos[ip1,it1] * (log_T - self.T_eos[it0]) * (log_P - self.P_eos[ip0])

            log_Pe /= ((self.T_eos[it1] - self.T_eos[it0]) * (self.P_eos[ip1] - self.P_eos[ip0]))

            stokes_out[loop,:,:] = sir_code.synth(1, self.n_lambda_sir, ltau[ind], T[loop,ind], 10**log_Pe[ind], self.zeros[ind], vz[loop,ind], Bx[loop,ind], By[loop,ind], Bz[loop,ind], self.macroturbulence)

        return stokes_out