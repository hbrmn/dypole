#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of functions to process multidimensional NMR datasets.

Created on Wed Jan 22 14:31:16 2020

@author: Henrik Bradtmüller - mail@bradtmueller.net - https://hbrmn.github.io/

- to do: make it so the REDOR functions call the process function

Functions
---------
process1d
    Does something
process2d
    Does another thing
    
"""

import nmrglue as ng
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import palettable


class Dataset:

    def __init__(self, path, name, vendor, **kwargs):
        self.path = path
        self.name = name
        self.vendor = vendor

    def plot_init(self):
        fig_width_pt = 336.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = ((5)**(0.5)-1.0)/2.0      # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        #Figure options
        matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
        +r"\usepackage{fontenc}\usepackage{siunitx}"
        plt.rc('text', usetex=True)
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
        if self.vendor == 'bruker':
            dic, rawdata = ng.bruker.read(self.path)
            self.dic = dic
            self.rawdata = rawdata
            self.carrierFreq = dic['acqus']['SFO1']
            self.specWidth = dic['acqus']['SW_h']
            self.relOffFreq = (dic['acqus']['O1']
                               - (dic['procs']['SF']-dic['acqus']['BF1'])*1e6)
        elif self.vendor == 'varian':
            dic, rawdata = ng.varian.read(self.path)
            self.dic = dic
            self.rawdata = rawdata
            self.carrierFreq = np.float64(
                dic['procpar']['reffrq1']['values'][0])
            self.specWidth = np.float64(dic['procpar']['sw']['values'][0])
            self.relOffFreq = (np.float64(dic['procpar']['sfrq']['values'][0])
                               - self.carrierFreq)*1e6

    def prep_data(self):
        if 'acqu2s' in self.dic:
            # checks if rawdata has right shape
            if len(self.rawdata) != self.dic['acqu2s']['TD']:
                # if not tries to repair the data array
                aux = np.array([self.dic['acqu2s']['TD'],
                                self.dic['acqus']['TD']])
                data = (
                    np.reshape(
                        self.rawdata,
                        (int((aux[0])),
                         int(aux[1]/2)))[0:self.dic['acqu2s']['TD'], :])

            if self.dic['acqus']['DIGMOD'] != 0:
                # Removes the group delay artifact when recording digitally
                data = ng.bruker.remove_digital_filter(self.dic, self.rawdata)
                data = ng.proc_base.zf(
                    data, (self.rawdata.shape[self.rawdata.ndim-1]
                           -data.shape[data.ndim-1]))
            else:
                data = self.rawdata
        
        return(data)

    def process_data(self, leftShift=0, zeroFill=2, cutoff=0, exp_lb=0):
        """Returns frequency and ppm scales of input dataset as well as 
        the data array, data dictionary and transmitter offset value 
        (SFO) in points
        
        Parameters
        ----------
        lshift
        """
        self.get_data()
        data = self.prep_data()
        
        # Calculates the number of points after which FID is set to zeros
        trim = int(cutoff*data.shape[data.ndim-1])    

        if data.ndim == 1:
            data = data[leftShift:]
            data[trim:] = np.zeros(len(data[trim:]))
        else:
            data = data[:, leftShift:]
            data[ :, trim:] = np.zeros(len(data[0, trim:]))
            
        # Zero fill
        data = ng.proc_base.zf_double(data, zeroFill)

        # Calculates Line broadening (Topspin style) and
        # divides value by sw in Hz
        data = (ng.proc_base.em(data, lb=(exp_lb/(self.specWidth))))

        # Create frequency axis
        freq = ((np.arange((self.specWidth/2)+self.relOffFreq,
                           (-self.specWidth/2)+self.relOffFreq,
                           -self.specWidth/data.shape[data.ndim-1])))
        # Create ppm axis
        ppm = freq/self.carrierFreq
        
        # Fourier transform
        spec = ng.proc_base.fft(data)
        spec = spec/np.max(spec) # Normalize data - autophase works faster
    
        return(freq, ppm, spec)

    def plot_1d(self):
        
        '''-'''

    def bg_corr(self, xaxis, yaxis, order, threshold):

        """Returns the background corrected spectrum of the input data.

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

        """
        #Rescaling the data
        numPoints = len(xaxis)
        i = np.argsort(xaxis)
        yaxis = yaxis[i]
        maxy = np.max(yaxis)
        dely = (maxy-np.min(yaxis))/2
        numPointsCorr = (2 * (xaxis[:] - xaxis[numPoints-1]) 
                         /(xaxis[numPoints-1]-xaxis[0]) + 1)
        yaxis = (yaxis[:] - maxy) / dely + 1
        #Creating Vandermonde matrix
        const_p = np.arange(0, order+1, 1)
        #np.tile repeats arrays numPointsCorr and const_p
        var_T = np.tile(numPointsCorr,
                        (order+1, 1)).T ** np.tile(const_p, (numPoints, 1))
        #analog to MATLAB's pins function
        Tinv = np.linalg.pinv(np.matmul(var_T.T, var_T))
        Tinv = np.matmul(Tinv, var_T.T)
        #Initialisation (least-squares estimation)
        a = np.matmul(Tinv, yaxis)
        backFun = np.matmul(var_T, a)
        #Other variables
        alpha = 0.99 * 0.5
        it = 0
        zp = np.ones(numPoints)
        #Fitting loop
        while (np.sum((backFun-zp))**2)/(np.sum((zp))**2) > (1e-09):
            it += 1         #Iteration
            zp = backFun          #Previous estimation
            res = yaxis - backFun #Residual
            #### Add different functions atq, sh etc. here
            d = ((res*(2*alpha-1))*((res < threshold)*1)
                 + (alpha*2*threshold-res) * ((res >= threshold)*1))
            a = np.matmul(Tinv, (yaxis+d))   #Polynomial coefficients a
            backFun = np.matmul(var_T, a)          #Polynomial
        #Rescaling
        j = np.argsort(i)
        backFun = (backFun[j]-1) * dely + maxy
        a[1] = a[1]-1
        a = a * dely
        return(backFun)

#-------------s------------------------------------------#

# path = r"C:\Users\edwu5ea1\Documents\sciebo\data\NMR Data Bruker\300MHz MS\nmr\MB2_PB2_Series\38\pdata\1"
path = r"C:\Users\HB\sciebo\data\NMR Data Bruker\600MHz SC\nmr\7Li_LiNbO3\1\pdata\1"
data = Dataset(path,'test','bruker')
# test = data.process()
test = data.plot_1d()






#if __name__ == '__main__':






