from collections import OrderedDict
from sir3d import sir_code
from sir3d.configuration import Configuration
import numpy as np
import os
import scipy.stats
import logging
import h5py
import scipy.integrate as integ
from scipy import interpolate

# from ipdb import set_trace as stop

__all__ = ['Model']

class Model(object):
    def __init__(self, config=None, rank=0):

        if (rank != 0):
            return
        
        self.logger = logging.getLogger("model")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.rank = rank

        filename = os.path.join(os.path.dirname(__file__),'data/LINEAS')
        ff = open(filename, 'r')
        self.LINES = ff.readlines()
        ff.close()

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

            if (self.eos_type == 'MANCHA'):            

                self.logger.info('Reading EOS - MANCHA')

                filename = os.path.join(os.path.dirname(__file__), 'data/eos_mancha.h5')
                f = h5py.File(filename, 'r')
                self.T_eos = np.log10(f['T'][:])
                self.P_eos = np.log10(f['P'][:])
                self.Pe_eos = np.log10(f['Pel'][:])
                f.close()
                
                self.logger.info('Reading kappa5000 - MANCHA')
                self.T_kappa5 = np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50, 
                    3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70, 
                    3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00, 
                    4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50, 
                    4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00, 
                    5.05, 5.10, 5.15, 5.20, 5.25, 5.30 ])

                self.P_kappa5 = np.array([-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6. ,6.5, 7., 7.5, 8. ])

                self.kappa = np.zeros((56,21))

                filename = os.path.join(os.path.dirname(__file__), 'data/kappa5000_mancha.dat')
                f = open(filename, 'r')
                        
                for it in range(56):
                    for ip in range(21):
                        self.kappa[it,ip] = float(f.readline().split()[-1])

                f.close()

            else:

                self.logger.info('Reading EOS and kappa5000 - SIR')

                filename = os.path.join(os.path.dirname(__file__), 'data/kappa5000_eos_sir.h5')
                f = h5py.File(filename, 'r')
                self.T_eos = np.log10(f['T'][:])
                self.P_eos = np.log10(f['P'][:])
                self.Pe_eos = np.log10(f['Pe'][:])
                                
                self.T_kappa5 = np.log10(f['T'][:])
                self.P_kappa5 = np.log10(f['P'][:])
                self.kappa = f['kappa5000'][:]
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
        self.output_file = config_dict['general']['stokes output']
        self.atmosphere_type = config_dict['atmosphere']['type']
        self.eos_type = config_dict['general']['eos']

        self.logger.info('Output Stokes file : {0}'.format(self.output_file))

        if (config_dict['general']['interpolated model output'] != 'None'):
            self.interpolated_model_filename = config_dict['general']['interpolated model output']
            self.interpolated_tau = np.array([float(i) for i in config_dict['general']['interpolate tau']])
            self.n_tau = len(self.interpolated_tau)
            self.logger.info('Output model file : {0}'.format(self.interpolated_model_filename))
        else:
            self.interpolated_model_filename = None
        
        # Add spectral regions
        self.init_sir(config_dict['spectral regions'])

        self.spectral_regions_dict = config_dict['spectral regions']
        
        # Read atmosphere
        if (self.atmosphere_type == 'MURAM'):
            if (self.rank == 0):
                self.logger.info('Using MURAM atmosphere')

                self.model_shape = tuple([int(k) for k in config_dict['atmosphere']['dimensions']])
                self.nx, self.ny, self.nz = self.model_shape
                self.deltaz = float(config_dict['atmosphere']['deltaz']) * np.arange(self.ny)
                if ('deltaxy' in config_dict['atmosphere']):
                    self.deltaxy = float(config_dict['atmosphere']['deltaxy'])
                self.T_file = config_dict['atmosphere']['temperature']
                self.logger.info(' - T file : {0}'.format(self.T_file))

                self.P_file = config_dict['atmosphere']['pressure']
                self.logger.info(' - P file : {0}'.format(self.P_file))

                self.rho_file = config_dict['atmosphere']['density']
                self.logger.info(' - rho file : {0}'.format(self.rho_file))

                if ('vz' in config_dict['atmosphere']):
                    self.vz_file = config_dict['atmosphere']['vz']
                    self.vz_type = 'vz'
                    self.logger.info(' - vz file : {0}'.format(self.vz_file))
                elif ('rho_vz' in config_dict['atmosphere']):
                    self.vz_file = config_dict['atmosphere']['rho_vz']
                    self.vz_type = 'rho_vz'
                    self.logger.info(' - rho_vz file : {0}'.format(self.vz_file))
                else:
                    raise Exception("You need to provide either vz or rho_vz")

                if ('vx' in config_dict['atmosphere']):
                    self.vx_file = config_dict['atmosphere']['vx']
                    self.logger.info(' - vx file : {0}'.format(self.vx_file))
                else:
                    self.vx_file = None
                if ('vy' in config_dict['atmosphere']):
                    self.vy_file = config_dict['atmosphere']['vy']
                    self.logger.info(' - vy file : {0}'.format(self.vy_file))
                else:
                    self.vy_file = None

                self.Bx_file = config_dict['atmosphere']['bx']
                self.By_file = config_dict['atmosphere']['by']
                self.Bz_file = config_dict['atmosphere']['bz']
                self.logger.info(' - Bx file : {0}'.format(self.Bx_file))
                self.logger.info(' - By file : {0}'.format(self.By_file))
                self.logger.info(' - Bz file : {0}'.format(self.Bz_file))

                if ('tau delta' in config_dict['atmosphere']):
                    self.tau_fine = float(config_dict['atmosphere']['tau delta'])
                    self.logger.info(' - tau axis will be interpolated to have delta={0}'.format(self.tau_fine))
                else:
                    self.tau_fine = 0.0
                
                if ('mux' in config_dict['atmosphere']):
                    self.mux = float(config_dict['atmosphere']['mux'])                    
                else:
                    self.mux = 1.0

                if ('muy' in config_dict['atmosphere']):
                    self.muy = float(config_dict['atmosphere']['muy'])
                else:
                    self.muy = 1.0

                if (self.mux < 1.0 or self.muy < 1.0):
                    self.need_slant = True

                    self.xangle = np.arccos(self.mux)
                    self.yangle = np.arccos(self.muy)

                    self.mu = np.cos(np.sqrt(self.xangle**2 + self.yangle**2))
        
                    self.deltaz_new = self.deltaz / np.abs(self.mu)
                    self.logger.info(f' Slating atmosphere to mux={self.mux} - muy={self.muy}')
                    self.logger.info(f' Equivalent mu={self.mu}')
                else:
                    self.need_slant = False
                                    
                if (self.mux < 1.0 or self.muy < 1.0):
                    self.need_slant = True
                    if (self.vx_file is None or self.vy_file is None):
                        raise Exception("For inclined rays you need to provide all velocity components")

                self.zeros = np.zeros(self.ny)                                            

                self.maximum_tau = float(config_dict['atmosphere']['maximum tau'])

                self.bx_multiplier = 1.0
                self.by_multiplier = 1.0
                self.bz_multiplier = 1.0
                self.vz_multiplier = 1.0

                if ('multipliers' in config_dict['atmosphere']):

                    if ('bx' in config_dict['atmosphere']['multipliers']):
                        self.bx_multiplier = float(config_dict['atmosphere']['multipliers']['bx'])
                        self.logger.info('Bx multiplier : {0}'.format(self.bx_multiplier))

                    if ('by' in config_dict['atmosphere']['multipliers']):
                        self.by_multiplier = float(config_dict['atmosphere']['multipliers']['by'])
                        self.logger.info('By multiplier : {0}'.format(self.by_multiplier))

                    if ('bz' in config_dict['atmosphere']['multipliers']):
                        self.bz_multiplier = float(config_dict['atmosphere']['multipliers']['bz'])
                        self.logger.info('Bz multiplier : {0}'.format(self.bz_multiplier))

                    if ('vz' in config_dict['atmosphere']['multipliers']):
                        self.vz_multiplier = float(config_dict['atmosphere']['multipliers']['vz'])
                        self.logger.info('vz multiplier : {0}'.format(self.vz_multiplier))

                            
    # def init_sir_external(self, spectral):
    #     """
    #     Initialize SIR for this synthesis
        
    #     Parameters
    #     ----------
    #     None
        
    #     Returns
    #     -------
    #     None
    
    #     """

    #     filename = os.path.join(os.path.dirname(__file__),'data/LINEAS')
    #     ff = open(filename, 'r')
    #     flines = ff.readlines()
    #     ff.close()
        
            
    #     f = open('malla.grid', 'w')
    #     f.write("IMPORTANT: a) All items must be separated by commas.                 \n")
    #     f.write("           b) The first six characters of the last line                \n")
    #     f.write("          in the header (if any) must contain the symbol ---       \n")
    #     f.write("\n")                                                                       
    #     f.write("Line and blends indices   :   Initial lambda     Step     Final lambda \n")
    #     f.write("(in this order)                    (mA)          (mA)         (mA)     \n")
    #     f.write("-----------------------------------------------------------------------\n")

    #     for k, v in spectral.items():            
    #         self.logger.info('Adding spectral regions {0}'.format(v['name']))

    #         left = float(v['wavelength range'][0])
    #         right = float(v['wavelength range'][1])
    #         n_lambda = int(v['n. wavelengths'])
            
    #         delta = (right - left) / n_lambda

    #         for i in range(len(v['spectral lines'])):
    #             for l in flines:
    #                 tmp = l.split()
    #                 index = int(tmp[0].split('=')[0])                    
    #                 if (index == int(v['spectral lines'][0])):
    #                     wvl = float(tmp[2])

    #         lines = ''
    #         n_lines = len(v['spectral lines'])
    #         for i in range(n_lines):
    #             lines += v['spectral lines'][i]
    #             if (i != n_lines - 1):
    #                 lines += ', '

    #         f.write("{0}            :  {1}, {2}, {3}\n".format(lines, 1e3*(left-wvl), 1e3*delta, 1e3*(right-wvl)))
    #     f.close()
        
    #     self.n_lambda_sir = sir_code.init_externalfile(1, filename)

    # def init_sir_agents_external(self):

    #     filename = os.path.join(os.path.dirname(__file__),'data/LINEAS')
    #     self.n_lambda_sir = sir_code.init_externalfile(1, filename)

    def init_sir(self, spectral):
        """
        Initialize SIR for this synthesis. This version does not make use of any external file, which might be
        not safe when running in MPI mode.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        """
        lines = []
        n_lines = 0

        elements = {'H':1,'HE':2,'LI':3,'BE':4,'B':5,'C':6,'N':7,'O':8,'F':9,'NE':10,
            'NA':11,'MG':12,'AL':13,'SI':14,'P':15,'S':16,'CL':17,'AR':18,'K':19,'CA':20,'SC':21,'TI':22,'V':23,'CR':24,
            'MN':25,'FE':26,'CO':27,'NI':28,'CU':29,'ZN':30,'GA':31,'GE':32,'AS':33,'SE':34,'BR':35,'KR':36,
            'RB':37,'SR':38,'Y':39,'ZR':40,'NB':41,'MO':42,'TC':43,'RU':44,'RH':45,'PD':46,'AG':47,'CD':48,'IN':49,
            'SN':50,'SB':51,'TE':52,'I':53,'XE':54,'CS':55,'BA':56,'LA':57,'CE':58,'PR':59,'ND':60,'PM':61,
            'SM':62,'EU':63,'GD':64,'TB':65,'DY':66,'HO':67,'ER':68,'TM':69,'YB':70,'LU':71,'HF':72,'TA':73,'W':74,
            'RE':75,'OS':76,'IR':77,'PT':78,'AU':79,'HG':80,'TL':81,'PB':82,'BI':83,'PO':84,'AT':85,'RN':86,
            'FR':87,'RA':88,'AC':89,'TH':90,'PA':91,'U':92}
        states = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6}

        for k, v in spectral.items(): 

            self.logger.info('Adding spectral regions {0}'.format(v['name']))
            
            n_lines += 1

            left = float(v['wavelength range'][0])
            right = float(v['wavelength range'][1])
            n_lambda = int(v['n. wavelengths'])
            
            delta = (right - left) / n_lambda

            nblend = len(v['spectral lines'])                                    
            
            lines = np.zeros(nblend, dtype=np.intc)
            atom = np.zeros(nblend, dtype=np.intc)
            istage = np.zeros(nblend, dtype=np.intc)
            wvl = np.zeros(nblend)
            zeff = np.zeros(nblend)
            energy = np.zeros(nblend)
            loggf = np.zeros(nblend)
            mult1 = np.zeros(nblend, dtype=np.intc)
            mult2 = np.zeros(nblend, dtype=np.intc)
            design1 = np.zeros(nblend, dtype=np.intc)
            design2 = np.zeros(nblend, dtype=np.intc)
            tam1 = np.zeros(nblend)
            tam2 = np.zeros(nblend)
            alfa = np.zeros(nblend)
            sigma = np.zeros(nblend)
            
            for i in range(nblend):            
                lines[i] = v['spectral lines'][i]
                for l in self.LINES:
                    tmp = l.split()
                    index = int(tmp[0].split('=')[0])

                    if (index == int(v['spectral lines'][i])):
                                                    
                        atom[i] = elements[tmp[0].split('=')[1]]
                        istage[i] = tmp[1]
                        wvl[i] = float(tmp[2])
                        zeff[i] = float(tmp[3])
                        energy[i] = float(tmp[4])
                        loggf[i] = float(tmp[5])
                        mult1[i] = int(tmp[6][:-1])
                        mult2[i] = int(tmp[8][:-1])
                        design1[i] = states[tmp[6][-1]]
                        design2[i] = states[tmp[8][-1]]
                        tam1[i] = float(tmp[7].split('-')[0])
                        tam2[i] = float(tmp[9].split('-')[0])
                        if (len(tmp) == 12):
                            alfa[i] = float(tmp[-2])
                            sigma[i] = float(tmp[-1])
                        else:
                            alfa[i] = 0.0
                            sigma[i] = 0.0
            
            lambda0 = 1e3*(left-wvl[0])
            lambda1 = 1e3*(right-wvl[0])

            sir_code.init(n_lines, nblend, lines, atom, istage, wvl, zeff, energy, loggf,
                mult1, mult2, design1, design2, tam1, tam2, alfa, sigma, lambda0, lambda1, n_lambda)
                
        self.n_lambda_sir = n_lambda

    def intpltau(self, newtau, oldtau, var):
        fX = interpolate.interp1d(oldtau, var, bounds_error=False, fill_value="extrapolate")
        return fX(newtau)

    def synth(self, z, T, P, rho, vz, Bx, By, Bz, interpolate_model=False):

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

        if (self.eos_type == 'MANCHA'):
             chi = (kappa * rho)[::-1]
        else:
             chi = kappa[::-1]

        tau = integ.cumtrapz(chi,x=z)
        ltau = np.log10(np.insert(tau, 0, 0.5*tau[0]))[::-1]

        ind = np.where(ltau < 2.0)[0]

        # Get electron pressure
        it0 = np.searchsorted(self.T_eos, log_T) - 1        
        it1 = it0 + 1
        ip0 = np.searchsorted(self.P_eos, log_P) - 1
        ip1 = ip0 + 1

        if (self.eos_type == 'MANCHA'):
            log_Pe = self.Pe_eos[ip0,it0] * (self.T_eos[it1] - log_T) * (self.P_eos[ip1] - log_P) + \
                self.Pe_eos[ip1,it0] * (log_T - self.T_eos[it0]) * (self.P_eos[ip1] - log_P) + \
                self.Pe_eos[ip0,it1] * (self.T_eos[it1] - log_T) * (log_P - self.P_eos[ip0]) + \
                self.Pe_eos[ip1,it1] * (log_T - self.T_eos[it0]) * (log_P - self.P_eos[ip0])
        else:
            log_Pe = self.Pe_eos[it0,ip0] * (self.T_eos[it1] - log_T) * (self.P_eos[ip1] - log_P) + \
                self.Pe_eos[it1,ip0] * (log_T - self.T_eos[it0]) * (self.P_eos[ip1] - log_P) + \
                self.Pe_eos[it0,ip1] * (self.T_eos[it1] - log_T) * (log_P - self.P_eos[ip0]) + \
                self.Pe_eos[it1,ip1] * (log_T - self.T_eos[it0]) * (log_P - self.P_eos[ip0])


        log_Pe /= ((self.T_eos[it1] - self.T_eos[it0]) * (self.P_eos[ip1] - self.P_eos[ip0]))

        if (self.tau_fine != 0.0):
            taufino = np.arange(np.min(ltau[ind]), np.max(ltau[ind]), self.tau_fine)[::-1]
            stokes, error = sir_code.synth(1, self.n_lambda_sir, taufino, self.intpltau(taufino, ltau[ind], T[ind]),
                10**self.intpltau(taufino, ltau[ind], log_Pe[ind]), self.intpltau(taufino, ltau[ind], self.zeros[ind]), 
                self.intpltau(taufino, ltau[ind], self.vz_multiplier*vz[ind]), self.intpltau(taufino, ltau[ind], self.bx_multiplier*Bx[ind]),
                self.intpltau(taufino, ltau[ind], self.by_multiplier*By[ind]), self.intpltau(taufino, ltau[ind], self.bz_multiplier*Bz[ind]), self.macroturbulence)

        else:

            stokes, error = sir_code.synth(1, self.n_lambda_sir, ltau[ind], T[ind], 10**log_Pe[ind], self.zeros[ind], self.vz_multiplier*vz[ind], 
                self.bx_multiplier*Bx[ind], self.by_multiplier*By[ind], self.bz_multiplier*Bz[ind], self.macroturbulence)        

        if (error != 0):
            stokes = -99.0 * np.ones_like(stokes)        

        # We want to interpolate the model to certain isotau surfaces
        if (interpolate_model):
            model = np.zeros((7,self.n_tau))

            model[0,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.deltaz[::-1])
            model[1,:] = self.intpltau(self.interpolated_tau, ltau[::-1], T[::-1])
            model[2,:] = np.exp(self.intpltau(self.interpolated_tau, ltau[::-1], np.log(P[::-1])))
            model[3,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.vz_multiplier * vz[::-1])
            model[4,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.bx_multiplier * Bx[::-1])
            model[5,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.by_multiplier * By[::-1])
            model[6,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.bz_multiplier * Bz[::-1])

            return stokes, model

        return stokes

    def synth2d(self, z, T, P, rho, vz, Bx, By, Bz, interpolate_model=False):

        n = T.shape[0]

        stokes_out = np.zeros((n,5,self.n_lambda_sir))

        if (interpolate_model):
            model_out = np.zeros((n,7,self.n_tau))

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

            if (self.eos_type == 'MANCHA'):
                chi = (kappa * rho[loop,:])[::-1]
            else:
                chi = kappa[::-1]
             
            tau = integ.cumtrapz(chi, x=z)
            ltau = np.log10(np.insert(tau, 0, 0.5*tau[0]))[::-1]

            ind = np.where(ltau < 2.0)[0]

            # Get electron pressure
            it0 = np.searchsorted(self.T_eos, log_T) - 1
            it1 = it0 + 1
            ip0 = np.searchsorted(self.P_eos, log_P) - 1
            ip1 = ip0 + 1

            if (self.eos_type == 'MANCHA'):
                log_Pe = self.Pe_eos[ip0,it0] * (self.T_eos[it1] - log_T) * (self.P_eos[ip1] - log_P) + \
                        self.Pe_eos[ip1,it0] * (log_T - self.T_eos[it0]) * (self.P_eos[ip1] - log_P) + \
                        self.Pe_eos[ip0,it1] * (self.T_eos[it1] - log_T) * (log_P - self.P_eos[ip0]) + \
                        self.Pe_eos[ip1,it1] * (log_T - self.T_eos[it0]) * (log_P - self.P_eos[ip0])
            else:
                log_Pe = self.Pe_eos[it0,ip0] * (self.T_eos[it1] - log_T) * (self.P_eos[ip1] - log_P) + \
                        self.Pe_eos[it1,ip0] * (log_T - self.T_eos[it0]) * (self.P_eos[ip1] - log_P) + \
                        self.Pe_eos[it0,ip1] * (self.T_eos[it1] - log_T) * (log_P - self.P_eos[ip0]) + \
                        self.Pe_eos[it1,ip1] * (log_T - self.T_eos[it0]) * (log_P - self.P_eos[ip0])

            log_Pe /= ((self.T_eos[it1] - self.T_eos[it0]) * (self.P_eos[ip1] - self.P_eos[ip0]))

            if (self.tau_fine != 0.0):
                taufino = np.arange(np.min(ltau[ind]), np.max(ltau[ind]), self.tau_fine)[::-1]
                stokes_out[loop,:,:], error = sir_code.synth(1, self.n_lambda_sir, taufino, self.intpltau(taufino, ltau[ind], T[loop,ind]),
                    10**self.intpltau(taufino, ltau[ind], log_Pe[ind]), self.intpltau(taufino, ltau[ind], self.zeros[ind]), 
                    self.intpltau(taufino, ltau[ind], self.vz_multiplier*vz[loop,ind]), self.intpltau(taufino, ltau[ind], self.bx_multiplier*Bx[loop,ind]),
                    self.intpltau(taufino, ltau[ind], self.by_multiplier*By[loop,ind]), self.intpltau(taufino, ltau[ind], self.bz_multiplier*Bz[loop,ind]), self.macroturbulence)
            else:
                stokes_out[loop,:,:], error = sir_code.synth(1, self.n_lambda_sir, ltau[ind], T[loop,ind], 10**log_Pe[ind], self.zeros[ind], 
                    self.vz_multiplier*vz[loop,ind], self.bx_multiplier*Bx[loop,ind], self.by_multiplier*By[loop,ind], self.bz_multiplier*Bz[loop,ind], self.macroturbulence)

            if (error != 0):
                stokes_out[loop,:,:] = -99.0

            # We want to interpolate the model to certain isotau surfaces
            if (interpolate_model):

                model_out[loop,0,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.deltaz[::-1])
                model_out[loop,1,:] = self.intpltau(self.interpolated_tau, ltau[::-1], T[loop,::-1])
                model_out[loop,2,:] = np.exp(self.intpltau(self.interpolated_tau, ltau[::-1], np.log(P[loop,::-1])))
                model_out[loop,3,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.vz_multiplier * vz[loop,::-1])
                model_out[loop,4,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.bx_multiplier * Bx[loop,::-1])
                model_out[loop,5,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.by_multiplier * By[loop,::-1])
                model_out[loop,6,:] = self.intpltau(self.interpolated_tau, ltau[::-1], self.bz_multiplier * Bz[loop,::-1])

        if (interpolate_model):
            return stokes_out, model_out
        else:
            return stokes_out
