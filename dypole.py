#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of functions to process multidimensional NMR datasets.

Created on Wed Jan 22 14:31:16 2020

@author: Henrik Bradtmüller - mail@bradtmueller.net - https://hbrmn.github.io/

split in subclasses for REDOR, RESPDOR etc

Functions
---------
process1d
    Does something
process2d
    Does another thing

'''

import csv
import numpy as np
import scipy.optimize as opt
import nmrglue as ng
import matplotlib.pyplot as plt
import palettable
from cycler import cycler


class Dataset:
    ''' Docstring

    '''
    
    ##### Initializations #####

    def __init__(self, path, name, vendor):
        self.path = path
        self.name = name
        self.vendor = vendor
        self.get_data()

        #Figure options
        fig_width_pt = 336.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = ((5)**(0.5)-1.0)/2.0              # Aesthetic ratio
        self.fig_width = fig_width_pt*inches_per_pt     # width in inches
        self.fig_height = self.fig_width*golden_mean    # height in inches
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(
            cycler('color',
                   palettable.cmocean.sequential.Thermal_20.mpl_colors)),
        # plt.rc('axes', prop_cycle=(cycler('color', 'k')),
                titlesize=11, labelsize=11)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('font', **{'family':'serif', 'serif':['Times New Roman']})

    def reset(self):
        del self.spec
        del self.area
        
    def get_data(self):
        ''' Fetches the dictionary and NMR data from Bruker binary data. 
            The data is then prepared by removing the group delay
            artifact if data was acquired digitally and attempting to repair
            2D data from possibly aborted 2D experiments.
        '''
        if self.vendor == 'bruker':
            self.dic, rawdata = ng.bruker.read(self.path)
            if 'acqu2s' in self.dic:
            # Check if rawdata has right shape
                if len(rawdata) != self.dic['acqu2s']['TD']:
                    # If not, try to repair the data array
                    aux = np.array([self.dic['acqu2s']['TD'],
                                    self.dic['acqus']['TD']])
                    self.rawdata = (
                        np.reshape(
                            rawdata,
                            (int((aux[0])),
                             int(aux[1]/2)))[0:self.dic['acqu2s']['TD'], :])
            else:
                self.rawdata = rawdata
    
            if self.dic['acqus']['DIGMOD'] != 0:
                # Removes the group delay artifact when recording digitally
                self.rawdata = ng.bruker.remove_digital_filter(self.dic, 
                                                               self.rawdata)
                # Zero fills the number of omitted initial points to value 2^n
                self.rawdata = ng.proc_base.zf(
                    self.rawdata, (self.rawdata.shape[self.rawdata.ndim-1]
                           - self.rawdata.shape[self.rawdata.ndim-1]))
                
            self.car_freq = self.dic['acqus']['SFO1']
            self.spec_width = self.dic['acqus']['SW_h']
            self.rel_off_freq = (self.dic['acqus']['O1']
                               - (self.dic['procs']['SF']
                                  - self.dic['acqus']['BF1'])*1e6)
        if self.vendor == 'varian':
            ''' Fetches the dictionary and NMR data from VNMR (Varian) 
                binary data.
            '''
            self.dic, rawdata = ng.varian.read(self.path)
            self.rawdata = rawdata[0]
            self.car_freq = np.float64(
                self.dic['procpar']['reffrq1']['values'][0])
            self.spec_width = np.float64(
                self.dic['procpar']['sw']['values'][0])
            self.rel_off_freq = (
                np.float64(self.dic['procpar']['sfrq']['values'][0])
                - self.car_freq)*1e6

    ##### Processing #####

    def process(self):
        
        '''Returns frequency and ppm scales of input dataset as well as
        the data array, data dictionary and transmitter offset value
        (SFO) in points
        
        Todo: Make user choose either a string of all inputs e.g. 2, 0.5, 3,
        or to choose following the script
        Fix trimming not yielding 2^n number
        '''
        def user_input(func):
            def inner(self, data):
                done = 0
                while done != 1:
                    plt.plot(data[self.index][0])
                    plt.show()
                    proc_data = func(self, data)
                    done = int(input('Selection OK? 1 - yes, 2 - no: '))
                return proc_data
            return inner
        
        @user_input
        def left_shift(self, data):
            left_shift = int(input('Enter number of points to left shift: '))
            if data.ndim == 1:
                data = data[left_shift:]
            else:
                data = data[:, left_shift:]
            plt.plot(data[self.index][0][0:25])
            plt.show()
            return data
        
        @user_input
        def trim(self, data):
            cutoff = float(input('Enter fraction (e.g. 0.5) of FID to be used: '))
            # Calculates the number of points after which FID is set to zeros
            trim = int(cutoff*self.rawdata.shape[self.rawdata.ndim-1])
    
            # if data.ndim == 1:
            #     data[trim:] = np.zeros(len(data[trim:]))
            # else:
            #     data[ :, trim:] = np.zeros(len(data[0, trim:]))
            if data.ndim == 1:
                data = data[0:trim]
            else:
                data = data[:, 0:trim]
            plt.plot(data[self.index][0])
            plt.show()
            return data
        
        @user_input          
        def zero_fill(self, data):
            zero_fill = int(input('Enter multiplier of number of points: '))
            data = ng.proc_base.zf_double(data, zero_fill-1)
            plt.plot(data[self.index][0])
            plt.show()
            return data
        
        @user_input   
        def apodize(self, data):
            exp_lb = int(input('Enter exponential line broadeing in Hz: '))
            # Calculates Line broadening (Topspin style) and
            # divides value by sw in Hz
            data = (
                ng.proc_base.em(data, lb=(exp_lb/(self.spec_width))))
            plt.plot(data[self.index][0])
            plt.show()
            return data
        
        def phase(self, spec):
            spec = ng.proc_autophase.autops(spec, "acme", p0=0, p1=0)
            plt.plot(spec[self.index][0])
            plt.show()
            return spec
        
        # def phase(self, spec):
        #     self.zero_order = 0
        #     self.first_order = 0
        #     done = 0
        #     while done != 1:
        #             spec = zero_phase(self, spec)
        #             spec = first_phase(self, spec)
        #             done = int(input('Selection OK? 1 - yes, 2 - no: '))
        #     return spec

        # @user_input  
        # def zero_phase(self, spec):
        #     self.zero_order = int(input('Enter zero order phasing: '))
        #     spec = ng.proc_base.ps(spec, p0=self.zero_order, 
        #                            p1=self.first_order)
        #     plt.plot(spec[self.index][0])
        #     plt.show()
        #     return spec
        
        # @user_input
        # def first_phase(self, spec):
        #     self.first_order = int(input('Enter first order phasing: '))
        #     spec = ng.proc_base.ps(spec, p0=self.zero_order, 
        #                            p1=self.first_order)
        #     plt.plot(spec[self.index][0])
        #     plt.show()
        #     return spec
            

        # def auto_phase(self, spec):
            
        # Begin data processing
        try:
            self.proc_data
        except: 
            proc = 1
        else: 
            proc = int(input('Re-process data?: 1 - yes, 2 - no: '))
            
        if proc == 1:
            data = self.rawdata
            
            self.index = np.where(data**2 == np.max(data**2))[0]
            plt.show()
            
            data = left_shift(self, data)
            
            data = trim(self, data)
            
            data = zero_fill(self, data)
            
            length = data.shape[data.ndim-1]
        
            # Create frequency axis
            self.freq_scale = ((np.arange((self.spec_width/2)+self.rel_off_freq,
                               (-self.spec_width/2)+self.rel_off_freq,
                               -self.spec_width/length)))
            
            # Create ppm axis
            self.ppm_scale = self.freq_scale/self.car_freq
            
            # Transmitter offset frequency in points
            # Selecting data size out of array
            self.sfo_point = (int(np.round(length/2) 
                                  + (self.rel_off_freq 
                                     / ((self.spec_width)/length)
                                     )))
        
            data = apodize(self, data)
            
            self.proc_data = data  
        
            # Fourier transform
            spec = ng.proc_base.fft(self.proc_data)
            
            # Normalize data - autophase works faster
            spec = spec/np.max(spec)
    
            spec = phase(self, spec)        
    
            spec = ng.proc_base.di(spec)  # Discard the imaginaries
            if self.vendor == 'bruker':
                spec = ng.proc_base.rev(spec) # Reverse the spectrum
            
            # spec = [(spec[i] -  bg_corr(self.freq_scale, spec[i], 6, 0.002)) 
            #         for i in range(self.rawdata.shape[0])]
            
            self.spec = spec


    def integrate(self, back_corr=False):
        
        '''Returns the sum of the datapoints in the specified intervals
        
        Parameters
        ----------
        data : numpy array of float
            Possible 2D dataset containing spectral intensities
        
        Returns
        -------
        area : numpy array of float
            Array of integrated intensities. Array has one column for SED data
            input and two columns for REDOR and RESPDOR, S and S0 respectively.
        
        '''
        self.process()
        
        try:
            self.area
        except: 
            integrate = 1
        else: 
            integrate = int(input('Re-integrate data?: 1 - yes, 2 - no: '))
            
        if integrate == 1:
            
            ppm_scale = self.ppm_scale
            spec = self.spec
            # ppm per point
            ppm_per_pt = ((np.abs(ppm_scale[-1])
                           + np.abs(ppm_scale[0])) / len(ppm_scale))
            ylim = np.max(spec)
            
            plt.plot(ppm_scale, spec[self.index[0]])
            plt.xlim(ppm_scale[int(len(ppm_scale)*0.33)], 
                     ppm_scale[int(len(ppm_scale)*0.66)])
            plt.ylim(-ylim*0.05, ylim*1.05)
            plt.show()
            
            done = 0
            while done != 1:
                x_low = int(input('Enter left bound for integration in ppm: '))
                x_high = int(input('Enter right bound for integration in ppm: '))
                min_int = int(self.sfo_point - x_low / ppm_per_pt)
                max_int = int(self.sfo_point - x_high / ppm_per_pt)
                
                plt.plot(ppm_scale, spec[self.index[0]])
                plt.hlines(0, np.max(ppm_scale), np.min(ppm_scale))
                plt.xlim(ppm_scale[int(min_int)-10], 
                         ppm_scale[int(max_int)+12])
                plt.ylim(-0.05, ylim)
                plt.vlines(x_low, 0, ylim)
                plt.vlines(x_high, 0, ylim)
                plt.show()
                done = int(input('Selection OK? 1 - yes, 2 - no: '))
            
    
            # Number of points in F2 dimension
            #numberDirectPoints = data['dictionary']['acqus']['TD']
            # Number of points in F1 dimension
            num_points = spec.shape[0]
            
            # Initialize area parameter
            if self.experiment == 'REDOR' or self.experiment ==  'RESPDOR':
                area = (np.ones(num_points).reshape(int(num_points/2), 2))
            elif self.experiment == 'SED' or self.experiment ==  'T1':
                area = np.ones(num_points)
            
            # Background correction and integration
            
            area = [np.sum(spec[i, min_int:max_int]) for i in range(num_points)
                    if self.experiment == 'SED' or self.experiment == 'T1']
            
            if self.experiment == 'REDOR' or self.experiment == 'RESPDOR':
                for i in range(num_points):
                    # Puts REDOR (S) and Echo (S0) intensities in same 
                    # variable but different columns
                    area[int(np.floor(i*0.5)), i%2] = (
                        np.sum(spec[i, min_int:max_int]))
            self.area = area
        
    ##### Exporting #####
    
    def export(self, xaxis, yaxis, xlab, ylab, data_type, result=None):

        exp_var = zip(xaxis, yaxis)
        with open(self.path + '\\' + self.name + data_type
                  + '.csv', 'w') as file:
            writer = csv.writer(file, delimiter='\t', lineterminator='\n')
            writer.writerow((xlab, ylab, result))
            for word in exp_var:
                writer.writerows([word])

    ##### Experiment evaluation #####

    def sed_fid(self, export=False):

        '''Returns the homonuclear dipole-dipole second moment of a spin-echo
        decay experiment based in the FID intensities.

        Parameters
        ----------
        export : string
            'full', 'zoom', exports SED curve showing either all data points
            or the ones selected for the linear fit + an additional 10.

        Todo:
            Works only if rawdata contains FIDs - make sure if input is from
            Bruker, that the FID will be used.
        Returns
        -------
        second_moment
            Homonuclear dipole-dipole second moment in units 1e6 rad²/s²
        '''
        self.experiment = 'SED'
        
        if self.vendor == 'varian':
            vdlist = (np.array(
                self.dic['procpar']['t1Xecho']['values']).astype(float))

        # time scale (2tau) in ms²
        time = (((vdlist*1e-3)*2)**2)

        # Calculates log(I/I0)
        sed_int = (np.abs(self.rawdata)).max(axis=2).reshape(len(time))
        sed_int = np.log(sed_int/sed_int[0])

        # Plot data
        plt.scatter(time, sed_int, color='w', edgecolors='k')
        plt.show()

        # User input loop to select fit range
        done = 0
        while done != 1:
            x_low = int(input('Enter first point in linear fit: '))
            x_high = int(input('Enter last point in linear fit: '))
            x = time[x_low:x_high]
            y = sed_int[x_low:x_high]
            plt.scatter(x, y, color='w', edgecolors='k')
            plt.show()
            done = int(input('Selection OK? 1 - yes, 2 - no: '))

        ########## Linear regression analysis
        def fit_func_norm(xaxis, slope, yintercept):
            return slope*xaxis+yintercept # linear function with y-intercept
        def fit_func(xaxis, slope):
            return slope*xaxis # linear function w/o y-intercept

        # Initial guess.
        # x0 = -100                    # Initial guess of curvature value
        # sigma = np.ones(fit_max[0][0]) # Std. deviation of y-data

        [popt, _] = (opt.curve_fit(fit_func_norm, x, y))

        # Normalization of data by y-intercept of first linear fit
        sed_int = (np.log(np.exp(sed_int)/
                          np.exp(fit_func_norm(0, popt[0], popt[1]))))
        y = sed_int[x_low:x_high]

        # Corrected linear fitting
        [popt2, _] = (opt.curve_fit(fit_func, x, y))

        # Since time is in ms², the M2 unit is 1e6 rad²s-²
        second_moment = -2*popt2[0]

        # Exporting data
        if export:
            # Experimental data
            self.export(time, sed_int, '2tau^2', 'ln(I/I0)', 'SED')
            # Fit
            scale = np.round(np.linspace(time[0], time[-1], 1000), decimals=4)
            fit = fit_func(scale, popt2[0])
            result = ('M2 = '+ str(np.round(second_moment, decimals=4))
                      + ' e6 rad^2s^-2')
            self.export(scale, fit, '2tau^2', 'ln(I/I0)', 'Fit', result)

            # Creating and saving SED plot
            fig = plt.figure(figsize=(self.fig_width, self.fig_height))

            plt.scatter(time, sed_int, color='w', edgecolors='k')
            plt.plot(scale, fit_func(scale, popt2[0]), color='k')
            
            if export == 'full':
                plt.xlim(-0.025, time[-2]+0.025)
                plt.ylim(sed_int[-2]-0.25, -0.025)
            elif export == 'zoom':
                plt.xlim(0, time[x_high+10])
                plt.ylim(sed_int[x_high+10]-0.25, -0.025)

            plt.xlabel(r'(2$\tau$)$^2$ / ms$^2$')
            plt.ylabel(r'ln(I / I$_{0}$)')
            fig.savefig(self.path + '\\' + self.name + '_SED.png',
                        format='png', dpi=300, bbox_inches='tight')
        return second_moment
    
    def t1_eval(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        self.experiment = 'T1'
        
        self.integrate()
        
        if self.vendor == 'varian':
            vdlist = (np.array(
                self.dic['procpar']['d2']['values']).astype(float))
            
        # time scale in s
        time = vdlist
        
        # normalized intensities
        t1_int = self.area/np.max(self.area)
        
        # Plot data
        plt.plot(time, t1_int)
        plt.show()
        
        def fit_func(time, amp, T1, beta):
            return amp*(1-np.exp(-(time/T1)**beta)) # saturation recovery equation
        
        [popt, _] = (opt.curve_fit(fit_func, time, t1_int, 
                                   bounds=(np.array([0, 0, 0]), 
                                           np.array([np.inf, np.inf, 1]))))
        
        amp, T1, beta = popt[0], popt[1], popt[2]
        
        # Exporting data

        # Experimental data
        self.export(time, t1_int, 'tau', 'I', 'T1-Satrec')
        # Fit
        scale = np.round(np.linspace(time[0], time[-1], 1000), decimals=4)
        fit = fit_func(scale, popt[0], popt[1])
        result = ('T1 = '+ str(np.round(T1, decimals=4)) + ' s' 
                  + 'beta = ' + str(np.round(beta, decimals=4)))
        self.export(scale, fit, 'tau', 'I', 'Fit', result)

        # Creating and saving SED plot
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))

        plt.plot(time, t1_int)
        plt.plot(scale, fit_func(scale, popt[0], popt[1]), '--', color='b')
        
        plt.xlabel(r'$\tau$ / s')
        plt.ylabel(r'I/I$_0$')
        fig.savefig(self.path + '\\' + self.name + '_T1.png',
                    format='png', dpi=300, bbox_inches='tight')
        return T1
        

def bg_corr(xaxis, yaxis, order, threshold):

    '''Returns the background corrected spectrum of the input data.

    Parameters
    ----------
    xaxis : float
        x axis
    yaxis : float
        y axis
    order : int
        order of the polynomioal function
    threshold : float
        Noise threshold to differentiate signals from noise.
        Values typically range from 0.1 to 0.0001

    Returns
    -------
    z : numpy array of float
    contains the background corrected dataset

    '''

    #Rescaling the data
    num_points = len(xaxis)
    i = np.argsort(xaxis)

    yaxis = yaxis[i]
    maxy = np.max(yaxis)
    dely = (maxy-np.min(yaxis))/2
    num_points_corr = (2 * (xaxis[:] - xaxis[num_points-1])
                     /(xaxis[num_points-1]-xaxis[0]) + 1)

    yaxis = (yaxis[:] - maxy) / dely + 1
    #Creating Vandermonde matrix
    const_p = np.arange(0, order+1, 1)
    #np.tile repeats arrays num_points_corr and const_p
    var_t = np.tile(num_points_corr,
                    (order+1, 1)).T ** np.tile(const_p, (num_points, 1))
    #analog to MATLAB's pins function
    tinv = np.linalg.pinv(np.matmul(var_t.T, var_t))
    tinv = np.matmul(tinv, var_t.T)
    #Initialisation (least-squares estimation)
    aux = np.matmul(tinv, yaxis)
    back_fun = np.matmul(var_t, aux)
    #Other variables
    alpha = 0.99 * 0.5
    it = 0
    zp = np.ones(num_points)

    #Fitting loop
    while (np.sum((back_fun-zp))**2)/(np.sum((zp))**2) > (1e-09):
        it += 1         #Iteration
        zp = back_fun          #Previous estimation
        res = yaxis - back_fun #Residual
        #### Add different functions atq, sh etc. here
        d = ((res*(2*alpha-1))*((res < threshold)*1)
             + (alpha*2*threshold-res) * ((res >= threshold)*1))
        aux = np.matmul(tinv, (yaxis+d))   #Polynomial coefficients a
        back_fun = np.matmul(var_t, aux)          #Polynomial

    #Rescaling
    j = np.argsort(i)
    back_fun = (back_fun[j]-1) * dely + maxy
    aux[1] = aux[1]-1
    aux = aux * dely
    return back_fun

#----------------------------------------------------------------------------#

Path = (r'C:\Users\HB\data_work\Projects\1_Crystallization_I\7Li_Satrec\60d') 
#         # + r'\210722-7Li-LS2-cryst_SEDLT.fid')
nmr_data = Dataset(Path, 'TEST', 'varian')
T1 = nmr_data.t1_eval()

# M2 = nmr_data.sed_fid(export='zoom')
# test_var = nmr_data.t1_eval()

#if __name__ == '__main__':
