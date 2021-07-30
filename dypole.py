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
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import palettable
import scipy.optimize as opt

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
            self.dic, self.rawdata = ng.varian.read(self.path)
            self.car_freq = np.float64(
                self.dic['procpar']['reffrq1']['values'][0])
            self.spec_width = np.float64(self.dic['procpar']['sw']['values'][0])
            self.rel_off_freq = (
                np.float64(self.dic['procpar']['sfrq']['values'][0])
                - self.car_freq)*1e6

    ##### Processing #####

    def process_data(self, left_shift=0, zero_fill=2, cutoff=0, exp_lb=0):
        '''Returns frequency and ppm scales of input dataset as well as
        the data array, data dictionary and transmitter offset value
        (SFO) in points

        Todo - make interactable to reduce arguments

        Parameters
        ----------
        left_shift :
        zero_fill :
        cutoff :
        exp_lb :
        '''
        data = self.rawdata()

        # Calculates the number of points after which FID is set to zeros
        trim = int(cutoff*data.shape[data.ndim-1])

        if data.ndim == 1:
            data = data[left_shift:]
            data[trim:] = np.zeros(len(data[trim:]))
        else:
            data = data[:, left_shift:]
            data[ :, trim:] = np.zeros(len(data[0, trim:]))

        # Zero fill
        data = ng.proc_base.zf_double(data, zero_fill)

        # Calculates Line broadening (Topspin style) and
        # divides value by sw in Hz
        data = (ng.proc_base.em(data, lb=(exp_lb/(self.spec_width))))

        # Create frequency axis
        freq = ((np.arange((self.spec_width/2)+self.rel_off_freq,
                           (-self.spec_width/2)+self.rel_off_freq,
                           -self.spec_width/data.shape[data.ndim-1])))
        # Create ppm axis
        ppm = freq/self.car_freq

        # Fourier transform
        spec = ng.proc_base.fft(data)

        # Normalize data - autophase works faster
        spec = spec/np.max(spec)

        return(freq, ppm, spec)

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

        # time scale in ms²
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

Path = (r'C:\Users\HB\data_work\Projects\1_Crystallization_I\7Li\LS2\SED' 
        + r'\210722-7Li-LS2-cryst_SEDLT.fid')
nmr_data = Dataset(Path, 'deleteMe', 'varian')

M2 = nmr_data.sed_fid(export='full')

#if __name__ == '__main__':
