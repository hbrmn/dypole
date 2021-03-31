#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of functions to process multidimensional datasets.

Created on Wed Jan 22 14:31:16 2020
Modified May 25

@author: Henrik Bradtmüller

"""

import nmrglue as ng
import numpy as np


class Dataset:


    def __init__(self, path, name, vendor):

        self.path = path
        self.name = name
        self.vendor = vendor
        getData(self.path)

    def getData(self, path, vendor):
        if vendor == 'bruker':
            dic, rawdata = ng.bruker.read(path)
            self.dic = dic
            self.rawdata = rawdata
            self.carrierFreq = dic['acqus']['SFO1']
            self.specWidth = dic['acqus']['SW_h']
            self.relOffFreq = (dic['acqus']['O1']
                               - (dic['procs']['SF']-dic['acqus']['BF1'])*1e6)
        elif vendor == 'varian':
            dic, rawdata = ng.varian.read(path)
            self.dic = dic
            self.rawdata = rawdata
            self.carrierFreq = np.float64(dic['procpar']['reffrq1']['values'][0])
            self.specWidth = np.float64(dic['procpar']['sw']['values'][0])
            self.relOffFreq = (np.float64(dic['procpar']['sfrq']['values'][0])
                               - self.carrierFreq)*1e6

    def prepBrukerData(self):

        if 'acqu2s' in self.dic:
                # checks if rawdata has right shape
                if len(self.rawdata) != self.dic['acqu2s']['TD']:
                    # if not tries to repair the data array
                    aux = np.array([self.dic['acqu2s']['TD'],self.dic['acqus']['TD']])
                    self.rawdata = (
                        np.reshape(
                            self.rawdata,
                            (int((aux[0])),
                             int(aux[1]/2)))[0:self.dic['acqu2s']['TD'], :])

                if self.dic['acqus']['DIGMOD'] != 0:
                    # Removes the ominous group delay artifact when recording digitally
                    data = ng.bruker.remove_digital_filter(self.dic, self.rawdata)
                    data = ng.proc_base.zf(
                        data, (self.rawdata.shape[self.rawdata.ndim-1]-data.shape[data.ndim-1]))
                else:
                    data = self.rawdata
        return(data)

    def process(self):

        if self.vendor == 'bruker':
            # checks whether data is 2D


            if data.ndim == 1:
                data = data[proc_par['number_shift_points']:]
            else:
                data = data[:, proc_par['number_shift_points']:]



    def bgCorr(self, xaxis, yaxis, order, threshold):

        """Returns the background corrected spectrum of the input spectral data.

        Parameters
        ----------
        xaxis : float
            x axis
        yaxis : float
            y axis
        order : int
            order of the polynomioal function
        threshold : float
            Noise threshold to differentiate signals from noise. Values typically
            range from 0.1 to 0.0001

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
        numPointsCorr = 2 * (xaxis[:] - xaxis[numPoints-1]) / (xaxis[numPoints-1]-xaxis[0]) + 1
        yaxis = (yaxis[:] - maxy) / dely + 1
        #Creating Vandermonde matrix
        const_p = np.arange(0, order+1, 1)
        #np.tile repeats arrays numPointsCorr and const_p
        var_T = np.tile(numPointsCorr, (order+1, 1)).T ** np.tile(const_p, (numPoints, 1))
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
        while (np.sum((z-zp))**2)/(np.sum((zp))**2) > (1e-09):
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

#------------l------------------------------------------#

# path = r"C:\Users\edwu5ea1\Documents\sciebo\data\NMR Data Bruker\300MHz MS\nmr\MB2_PB2_Series\38\pdata\1"
path = r"/home/rm7c8/ownCloud/data/NMR Data Bruker/300MHz MS/nmr/MB2_PB2_Series/38/pdata/1"
data = Dataset(path,'test','bruker')
test = data.process()






#if __name__ == '__main__':






