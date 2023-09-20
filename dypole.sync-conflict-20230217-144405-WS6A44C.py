#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of functions to process multidimensional NMR datasets.

Created on Wed Jan 22 14:31:16 2020

@author: Henrik Bradtmüller - mail@bradtmueller.net - https://hbrmn.github.io/

ToDo: Include parabolic approximation for WRESPDOR curves:
    \Delta S / S_0 = 2/(3*\pi^2) M_2 (nTr)^2 - Anuraag Gaddam 2022

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
import scipy.special as ss
import nmrglue as ng
import matplotlib.pyplot as plt
import palettable
from cycler import cycler


class Dataset:
    ''' Docstring

    '''

    ##### Initializations #####

    def __init__(self, path, name, vendor,
                 ls=False, trim=False, zf=False, lb=False):
        self.path = path
        self.name = name
        self.vendor = vendor
        self.procpar = np.array([ls, trim, zf, lb])
        self.ls = ls
        self.trim = trim
        self.zf = zf
        self.lb = lb
        self.get_data()

        # Figure options
        fig_width_pt = 336.0  # From LaTeX: \showthe\columnwidth
        inches_per_pt = 1.0 / 72.27                    # pt to inch
        golden_mean = ((5)**(0.5) - 1.0) / 2.0         # Golden ratio
        self.fig_width = fig_width_pt * inches_per_pt  # width in inch
        self.fig_height = self.fig_width * golden_mean # height in inch
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(
            cycler('color',
                   palettable.cmocean.sequential.Thermal_20.mpl_colors)),
               # plt.rc('axes', prop_cycle=(cycler('color', 'k')),
               titlesize=11, labelsize=11)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

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
                    aux = ng.bruker.guess_shape(self.dic)

                    self.rawdata = np.reshape(
                        rawdata,(int((aux[0][1]+aux[0][0])*2),
                                 int(aux[0][2]/2)))[0:self.dic['acqu2s']['TD'],:]
                else:
                    self.rawdata = rawdata

            if self.dic['acqus']['DIGMOD'] != 0:
                # Removes the group delay artifact when recording digitally
                self.rawdata = ng.bruker.remove_digital_filter(self.dic,
                                                               self.rawdata)
                # Zero fills the number of omitted initial points to value 2^n
                self.rawdata = ng.proc_base.zf(
                    self.rawdata, (self.rawdata.shape[self.rawdata.ndim - 1]
                                   - self.rawdata.shape[self.rawdata.ndim - 1]))

            self.car_freq = self.dic['acqus']['SFO1']
            self.spec_width = self.dic['acqus']['SW_h']
            self.rel_off_freq = (self.dic['acqus']['O1']
                                 - (self.dic['procs']['SF']
                                    - self.dic['acqus']['BF1']) * 1e6)
        if self.vendor == 'varian':
            # Fetches the dictionary and NMR data from VNMR (Varian)
            # binary data.

            self.dic, rawdata = ng.varian.read(self.path)
            self.rawdata = rawdata[0]
            self.car_freq = np.float64(
                self.dic['procpar']['reffrq1']['values'][0])
            self.spec_width = np.float64(
                self.dic['procpar']['sw']['values'][0])
            self.rel_off_freq = (
                np.float64(self.dic['procpar']['sfrq']['values'][0])
                - self.car_freq) * 1e6

    def process(self):
        '''Returns frequency and ppm scales of input dataset as well as
        the data array, data dictionary and transmitter offset value
        (SFO) in points
        '''
        def input_wrapper(func):
            def inner(self, data):
                if self.procpar.any():
                    proc_data = func(self, data)
                else:
                    done = 0
                    while done != 1:
                        plt.plot(np.real(data[self.index][0]))
                        plt.show()
                        proc_data = func(self, data)
                        done = int(input('Selection OK? 1 - yes, 2 - no: '))
                return proc_data
            return inner

        @input_wrapper
        def left_shift(self, data):
            #Left-shift to FID max for integrative SED evaluation
            if self.experiment == 'SED':
                left_shift = [data[n,:].argmax() for n
                              in range(len(data))]
                data = [ng.proc_base.ls(data[n], left_shift[n]) for n
                        in range(len(data))]
                data = np.asarray(data)
            else:
                if self.ls:
                    left_shift = self.ls
                else:
                    left_shift = int(
                        input('Enter number of points to left shift: '))
                if data.ndim == 1:
                    data = data[left_shift:]
                else:
                    data = data[:, left_shift:]
                plt.plot(np.real(data[self.index][0][0:25]))
                plt.show()
            return data

        @input_wrapper
        def trim(self, data):
            if self.trim is not False:
                cutoff = self.trim
            else:
                cutoff = float(
                    input('Enter fraction (e.g. 0.5) of FID to be used: '))
            # Calculates the number of points after which FID is set to zeros
            trim = int(cutoff * self.rawdata.shape[self.rawdata.ndim - 1])

            # if data.ndim == 1:
            #     data[trim:] = np.zeros(len(data[trim:]))
            # else:
            #     data[ :, trim:] = np.zeros(len(data[0, trim:]))
            if data.ndim == 1:
                data = data[0:trim]
            else:
                data = data[:, 0:trim]
            plt.plot(np.real(data[self.index][0]))
            plt.show()
            return data

        @input_wrapper
        def zero_fill(self, data):
            if self.zf is not False:
                zero_fill = self.zf
                if self.zf==0:
                    zero_fill = int(
                    input('Multiplier cannot be zero, enter multiplier of number of points: '))
            else:
                zero_fill = int(
                    input('Enter multiplier of number of points: '))
            data = ng.proc_base.zf_double(data, zero_fill - 1)
            plt.plot(np.real(data[self.index][0]))
            plt.show()
            return data

        @input_wrapper
        def apodize(self, data):
            if self.lb is not False:
                exp_lb = self.lb
            else:
                exp_lb = int(input('Enter exponential line broadeing in Hz: '))
            # Calculates Line broadening (Topspin style) and
            # divides value by sw in Hz
            data = (
                ng.proc_base.em(data, lb=(exp_lb/(self.spec_width))))
            plt.plot(np.real(data[self.index][0]))
            plt.show()
            return data

        @input_wrapper
        def manual_phase(self, spec):
            self.zero_order = int(input('Enter zero order phasing: '))
            self.first_order = int(input('Enter first order phasing: '))
            spec = ng.proc_base.ps(spec, p0=self.zero_order,
                                    p1=self.first_order)
            plt.plot(spec[self.index][0])
            plt.show()
            return spec

        def auto_phase(self, spec):
            spec = ng.proc_autophase.autops(spec, "acme", p0=0, p1=0)
            plt.plot(np.real(spec[self.index][0]))
            plt.show()
            return spec

        @input_wrapper
        def backcorr(self, spec):
            poly_degree = int(input('Enter degree of polynomial: '))
            threshold = float(input('Enter threshold (e.g., 0.02): '))

            spec_corr = [(spec[i] -  bg_corr(self.freq_scale, spec[i],
                                         poly_degree, threshold))
                         for i in range(len(spec))]
            spec = np.asarray(spec_corr)

            plt.plot(spec[1,:])
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
            #find index of FID with maximum value among array
            self.index = np.where(data**2 == np.max(data**2))[0]
            plt.show()

            data = left_shift(self, data)

            data = trim(self, data)

            data = zero_fill(self, data)

            length = data.shape[data.ndim - 1]

            # Create frequency axis
            self.freq_scale = (
                (np.arange((self.spec_width/2) + self.rel_off_freq,
                           - self.spec_width/2 + self.rel_off_freq,
                           - self.spec_width/length)))

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

            phasing = int(input('Automatic Phasing?: 1 - yes, 2 - no, 3 - Abs: '))
            if phasing == 1:
                spec = auto_phase(self, spec)
            elif phasing == 3:
                spec = np.abs(spec)
            else:
                spec = manual_phase(self,spec)

            spec = ng.proc_base.di(spec)  # Discard the imaginaries
            if self.vendor == 'bruker':
                spec = ng.proc_base.rev(spec)  # Reverse the spectrum

            correction = int(input('Baseline Correction?: 1 - yes, 2 - no: '))
            if correction == 1:
                spec = backcorr(self, spec)

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
            ppm_per_pt = ((np.abs(ppm_scale[- 1])
                           + np.abs(ppm_scale[0])) / len(ppm_scale))
            ylim = np.max(spec)

            plt.plot(ppm_scale, spec[self.index[0]])
            plt.xlim(ppm_scale[int(len(ppm_scale) * 0.33)],
                     ppm_scale[int(len(ppm_scale) * 0.66)])
            plt.ylim(-ylim * 0.05, ylim * 1.05)
            plt.show()

            done = 0
            while done != 1:
                x_low = int(input('Enter left bound for integration in ppm: '))
                x_high = int(
                    input('Enter right bound for integration in ppm: '))
                min_int = int(self.sfo_point - x_low / ppm_per_pt)
                max_int = int(self.sfo_point - x_high / ppm_per_pt)

                plt.plot(ppm_scale, spec[self.index[0]])
                plt.hlines(0, np.max(ppm_scale), np.min(ppm_scale))
                plt.xlim(ppm_scale[int(min_int) - 10],
                         ppm_scale[int(max_int) + 12])
                plt.ylim(-0.05, ylim)
                plt.vlines(x_low, 0, ylim)
                plt.vlines(x_high, 0, ylim)
                plt.show()
                done = int(input('Selection OK? 1 - yes, 2 - no: '))

            # Number of points in F2 dimension
            #numberDirectPoints = self.dic['acqus']['TD']
            # Number of points in F1 dimension
            num_points = spec.shape[0]

            # Initialize area parameter
            # area = np.ones(num_points)

            # Background correction and integration

            area = np.array(
                [np.sum(spec[i, min_int:max_int]) for i in range(num_points)])

            if self.experiment == 'REDOR' or self.experiment == 'RESPDOR':
                area = area.reshape(int(num_points / 2), 2)

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

    def sed_eval(self, fid=False, export=False):
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
        if self.vendor == 'bruker':
            # vdlist = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
            #                   160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
            vdlist = np.linspace(5, 100, 20)
        # vdlist = vdlist

        # time scale (2tau) in ms²
        time = (((vdlist*1e-3)*2)**2)

        # Calculates log(I/I0)
        if fid:
            sed_int = (np.abs(self.rawdata)).max(axis=1).reshape(len(time))
        else:
            self.integrate()
            sed_int = self.area

        sed_int = np.log(sed_int/sed_int[0])

        # Removes missing experiments, yielding all zero FID's
        if np.any(np.isinf(sed_int)):
            max_index = np.where(np.isinf(sed_int))[0][0]
            time = time[0:max_index]
            sed_int = sed_int[0:max_index]
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

        # Linear regression analysis
        def fit_func_norm(xaxis, slope, yintercept):
            return slope*xaxis+yintercept  # linear function with y-intercept

        def fit_func(xaxis, slope):
            return slope*xaxis  # linear function w/o y-intercept

        # Initial guess.
        # x0 = -100                    # Initial guess of curvature value
        # sigma = np.ones(fit_max[0][0]) # Std. deviation of y-data

        [popt, _] = (opt.curve_fit(fit_func_norm, x, y))

        # Normalization of data by y-intercept of first linear fit
        sed_int = (np.log(np.exp(sed_int) /
                          np.exp(fit_func_norm(0, popt[0], popt[1]))))
        y = sed_int[x_low:x_high]

        # Corrected linear fitting
        [popt2, _] = (opt.curve_fit(fit_func, x, y))

        # Since time is in ms², the M2 unit is 1e6 rad²s-²
        second_moment = -2*popt2[0]

        # Exporting data
        if export:
            # Experimental data
            self.export(time, sed_int, '2tau^2 / ms^2',
                        'ln(I/I0)', 'SED')
            # Fit
            scale = np.round(np.linspace(time[0], time[-1], 1000),
                             decimals=4)
            fit = fit_func(scale, popt2[0])
            result = ('M2 = ' + str(np.round(second_moment, decimals=4))
                      + ' e6 rad^2s^-2')
            self.export(scale, fit, '2tau^2 / ms^2', 'ln(I/I0)', 'Fit',
                        result)

            # Creating and saving SED plot
            fig = plt.figure(figsize=(self.fig_width, self.fig_height))

            plt.scatter(time, sed_int, color='w', edgecolors='k')
            plt.plot(scale, fit_func(scale, popt2[0]), color='k')

            if export == 'full':
                plt.xlim(-0.005, 0.38)
                # plt.xlim(-0.005, time[-4]+0.025)
                plt.ylim(-3, 0.025)
                # plt.ylim(sed_int[-4]-0.25, 0.025)
            elif export == 'zoom':
                # plt.xlim(0, time[x_high+10])
                plt.xlim(-0.001, 0.041)
                # plt.ylim(sed_int[x_high]*2, 0.025)
                plt.ylim(-1.5, 0.025)

            plt.xlabel(r'(2$\tau$)$^2$ / ms$^2$')
            plt.ylabel(r'ln(I / I$_{0}$)')
            fig.savefig(self.path + '\\' + self.name + '_SED.png',
                        format='png', dpi=300, bbox_inches='tight')
        return second_moment

    def t1_eval(self, fid=False, fit_lim=False):
        '''

        Returns
        -------
        None.

        '''

        self.experiment = 'T1'

        if fid:
            t1_int = np.abs(self.rawdata).max(axis=1)
        else:
            self.integrate()
            t1_int = self.area

        if self.vendor == 'varian':
            vdlist = (np.array(
                self.dic['procpar']['d2']['values']).astype(float))

        if fit_lim == False:
            fit_lim = 0

        # time scale in s
        time = vdlist[0:len(t1_int)-fit_lim]

        # normalized intensities
        t1_int = self.area/np.max(self.area)
        t1_int = t1_int[0:len(t1_int)-fit_lim]

        # Plot data
        plt.plot(time, t1_int)
        plt.show()

        def mono_exp_fit(time, amp, T1, beta):
            return amp * (1 - np.exp(-(time/T1)**beta))  # saturation recovery

        def biexp_fit(time, amp1, T1a, beta1, amp2, T1b, beta2):
            return (amp1 * (1 - np.exp(-(time/T1a)**beta1))
                    + amp2 * (1 - np.exp(-(time/T1b)**beta2)))

        fit_type = int(input('Mono (1) or Bi (2) exponential fit?: '))

        if fit_type == 1:
            [popt, _] = (opt.curve_fit(mono_exp_fit, time, t1_int,
                                       bounds=(np.array([0, 0, 0.1]),
                                               np.array([np.inf, np.inf, 1]
                                                        ))))
            amp, T1, beta = popt[0], popt[1], popt[2]
        else:
            [popt, _] = (opt.curve_fit(biexp_fit, time, t1_int,
                                       bounds=(np.array([0, 0, 0.99999, 0, 0, 0.99999]),
                                               np.array([np.inf, np.inf, 1,
                                                         0.1, np.inf, 1]
                                                        ))))
            amp1, T1a, beta1, amp2, T1b, beta2 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]

        # Exporting data

        # Experimental data
        self.export(time, t1_int, 'tau / s', 'I', 'T1-Satrec')
        # Fit
        scale = np.round(np.linspace(time[0], time[-1], 1000),
                         decimals=4)
        if fit_type == 1:
            fit = mono_exp_fit(scale, amp, T1, beta)
            result = ('T1 = ' + str(np.round(T1, decimals=4)) + ' s, '
                      + 'beta = ' + str(np.round(beta, decimals=4)))
        else:
            fit = biexp_fit(scale, amp1, T1a, beta1, amp2, T1b, beta2)
            result = ('amp1 = ' + str(np.round(amp1, decimals=4))
                      + ', T1a = ' + str(np.round(T1a, decimals=4)) + ' s, '
                      + 'beta1 = ' + str(np.round(beta1, decimals=4))
                      + ', amp2 = ' + str(np.round(amp2, decimals=4))
                      + ', T1b = ' + str(np.round(T1b, decimals=4)) + ' s, '
                      + 'beta2 = ' + str(np.round(beta2, decimals=4)))


        self.export(scale, fit, 'tau / s', 'I', 'Fit', result)

        # Creating and saving T1 plot
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))

        plt.scatter(time, t1_int, color='w', edgecolors='k')
        plt.plot(scale, fit,
                 '--', color='k')
        if fit_type == 2:
            plt.plot(scale, mono_exp_fit(scale, amp1, T1a, beta1), '--', color='b')
            plt.plot(scale, mono_exp_fit(scale, amp2, T1b, beta2), '--', color='g')

        plt.xlabel(r'$t$ / s')
        plt.ylabel(r'I/I$_0$')
        # plt.axhline(y=1.0, color='k', linestyle='--')
        fig.savefig(self.path + '\\' + self.name + '_T1.png',
                    format='png', dpi=300, bbox_inches='tight')
        return result

    def t1rho_eval(self, fid=False, fit_lim=False):
        '''

        Returns
        -------
        None.

        '''

        self.experiment = 'T1rho'

        if fid:
            t1rho_int = np.abs(self.rawdata).max(axis=1)
        else:
            self.integrate()
            t1rho_int = self.area

        if self.vendor == 'varian':
            vdlist = (np.array(
                self.dic['procpar']['d2']['values']).astype(float))
        elif self.vendor == 'bruker':
            vdlist = np.array([2000, 2700, 3700, 5000, 6800, 9300, 12600, 17200, 23000,
                      31700, 43100, 58600, 79600, 108200, 147100, 200000]).astype(float)

        if fit_lim == False:
            fit_lim = 0

        # time scale in s
        time = vdlist[0:len(t1rho_int)-fit_lim] * 1e-6

        # normalized intensities
        t1rho_int = t1rho_int/np.max(t1rho_int)
        t1rho_int = t1rho_int[0:len(t1rho_int)-fit_lim]

        def mono_exp_fit(time, amp, T1, beta):
            return amp * (np.exp(-(time*T1)**(beta)))  # saturation recovery

        def biexp_fit(time, amp1, T1a, beta1, amp2, T1b, beta2):
            return (amp1 * (1 - np.exp(-(time/T1a)**beta1))
                    + amp2 * (1 - np.exp(-(time/T1b)**beta2)))


        [popt, _] = (opt.curve_fit(mono_exp_fit, time, t1rho_int,
                                   bounds=(np.array([0, 0, 0.1]),
                                           np.array([np.inf, np.inf, 1]
                                                    ))))
        amp, T1, beta = popt[0], popt[1], popt[2]


        # Exporting data

        # Experimental data
        self.export(time, t1rho_int, 'tau / s^-1', 'I', 'T1-Satrec')
        # Fit
        scale = np.round(np.linspace(time[0], time[-1], 1000),
                         decimals=4)
        fit = mono_exp_fit(scale, amp, T1, beta)
        result = ('T1rho = ' + str(np.round(T1, decimals=4)) + ' s^-1, '
                  + 'beta = ' + str(np.round(beta, decimals=4)))


        self.export(scale, fit, 'tau / s', 'I', 'Fit', result)

        # Creating and saving T1 plot
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))

        plt.scatter(time, t1rho_int, color='w', edgecolors='k')
        plt.plot(scale, fit,
                 '--', color='k')

        plt.xlabel(r'$t$ / ms')
        plt.ylabel(r'I/I$_0$')
        plt.ylim(-0.05, 1.05)
        # plt.axhline(y=1.0, color='k', linestyle='--')
        fig.savefig(self.path + '\\' + self.name + '_T1rho.png',
                    format='png', dpi=300, bbox_inches='tight')
        return result

    def sae_eval(self, fid=False, fit_lim=False):
         '''

         Returns
         -------
         None.

         '''

         self.experiment = 'SAE'

         if fid:
             sae_int = np.abs(self.rawdata).max(axis=1)
         else:
             self.integrate()
             sae_int = self.area


         if self.vendor == 'varian':
             vdlist = (np.array(
                 self.dic['procpar']['tau2']['values']).astype(float))

         # time scale in s
         time = vdlist*1e-6

         if fit_lim == False:
             fit_lim = len(time)

         fit_time = time[0:fit_lim]

         # normalized intensities
         sae_int = sae_int/sae_int[0]
         fit_int = sae_int[0:fit_lim]

         # Plot data
         plt.scatter(time, sae_int, color = 'w', edgecolor = 'k')
         plt.semilogx(time, sae_int, '--')
         plt.show()

         def mono_exp_fit(time, Tsae, beta):
             return (np.exp(-(time*Tsae)**beta))  # saturation recovery

         def Tsae_exp(time, A, Tsae, beta1, B):
             return (A * np.exp(-(time*Tsae))**beta1 + B)

         def T1sae_exp(time, T1sae, beta2):
             return (np.exp(-(time*T1sae)**beta2))

         def biexp_fit(time, A, Tsae, beta1, B, T1sae, beta2):
             return (Tsae_exp(time, A, Tsae, beta1, B) * T1sae_exp(time, T1sae, beta2))

         fit_type = int(input('Mono (1) or Bi (KWW) (2) exponential fit?: '))

         if fit_type == 1:
             [popt, _] = (opt.curve_fit(mono_exp_fit, fit_time, fit_int,
                                        bounds=(np.array([0.1, 0.1]),
                                                np.array([np.inf, 1]
                                                         ))))
             Tsae, beta = popt[0], popt[1]
         elif fit_type == 2:
             [popt, _] = (opt.curve_fit(biexp_fit, fit_time, fit_int,
                                        p0=[0.8, 466, 0.99, 0.2, 80, 0.17],
                                        bounds=(np.array([ 0, 0, 0, 0, 0, 0]),
                                                np.array([1, np.inf, 1, 1, np.inf, 1]
                                                         ))))
             A, Tsae, beta1, B, T1sae, beta2 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
             S_inf = B/(A+B)
         # elif fit_type == 3:


         # Exporting data



         # Experimental data
         self.export(time, sae_int, 'tau / s', 'I', 'SAE')
         # Fit
         scale = np.geomspace(time[0], time[-1], num=1000)
         if fit_type == 1:
             fit = mono_exp_fit(scale, Tsae, beta)
             result = ('Tsae-1 = ' + str(np.round(Tsae, decimals=4)) + ' Hz, '
                       + 'beta = ' + str(np.round(beta, decimals=4)))
         else:
             fit = biexp_fit(scale, A, Tsae, beta1, B, T1sae, beta2)
             result = ('S_inf = ' + str(np.round(S_inf, decimals=4))
                       + ', T1sae-1 = ' + str(np.round(Tsae, decimals=4)) + ' Hz, '
                       + 'beta1 = ' + str(np.round(beta1, decimals=4))
                       + ', T1sae-1 = ' + str(np.round(T1sae, decimals=4)) + ' Hz, '
                       + 'beta2 = ' + str(np.round(beta2, decimals=4)))


         self.export(scale, fit, 'tau / s', 'I', 'Fit', result)

         # Creating and saving T1 plot
         fig = plt.figure(figsize=(self.fig_width, self.fig_height))

         plt.scatter(time, sae_int, color = 'w', edgecolor = 'k')
         plt.semilogx(scale, fit, '-', color='k')

         y_min = sae_int[-1]*0.5

         if fit_type == 2:
             plt.loglog(scale, (Tsae_exp(scale, A, Tsae, beta1, 0)
                                * T1sae_exp(scale, T1sae, beta2)) + y_min, '--', color='g')
             plt.loglog(scale, (T1sae_exp(scale, T1sae, beta2) * B), '--', color='m')

         if fit_type == 1:
             plt.ylim(y_min*0.9, 1.1)
         else:
             plt.ylim(y_min*0.9, 1.2)
         # plt.xlim(1e-6, 100)
         plt.xlabel(r'mixing time $t_m$ / s')
         plt.ylabel(r'$S_2$ ($t_p$, $t_m$) / a.u.')
         fig.savefig(self.path + '\\' + self.name + '_SAE.png',
                     format='png', dpi=300, bbox_inches='tight')

         return result

    def respdor_eval(self, spin=False, fit_lim=False):
        """Returns the heterodipolar second moment value of a RESPDOR experiment
        and exports the deltaS/S0 data set together with a fitted bessel function.

        Parameters
        ----------

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.experiment = 'RESPDOR'

        self.fit_lim = fit_lim

        self.integrate()

        if self.vendor == 'bruker':
            spin_rate = self.dic['acqus']['CNST'][31]
        elif self.vendor == 'varian':
            spin_rate = self.dic['procpar']['srate']['values'][0]

        # Number of points in F1 dimension
        number_indirect_points = self.dic['acqu2s']['TD']

        respdor_int = np.round((self.area[:, 1]-self.area[:, 0])
                                / self.area[:, 1], decimals=4)  # calc DS
        respdor_int = np.insert(respdor_int, 0, 1e-10)
        loop_increment = (2*self.dic['acqus']['L'][1])

        # Builds the time scale in ms
        respdor_ntr = (np.round(np.arange(2/spin_rate,
                                          ((number_indirect_points/2)+0.1) *
                                          (loop_increment/spin_rate),
                                          ((loop_increment/spin_rate))) * 1e3,
                                decimals=2))
        respdor_ntr = np.insert(respdor_ntr, 0, 1e-10)

        # fit_max = np.where(respdor_int > fit_lim)
        fit_max = fit_lim

        # Bessel function

        def sat_rec(xaxis, dip_const):
            """Saturation-Based recoupling curve
            Todo: make curve between integration regions -> 0
            ----------
            xaxis : numpy array of float
                Time data as array
            dip_const : float
                Dipolar coupling constant in Hz.
            nat_abund : float
                Natural abundance of nonobserved nucleus.

            Returns
            -------
            float
                Bessel function value for given values of time and dipolar coupling.

            """
            spin_fact = 1 / (2 * spin + 1)

            def bessel_func(k):
                return ((4 * spin - 2 * (k - 1)) *
                        ss.jv(0.25, k * np.sqrt(2) * dip_const * xaxis) *
                        ss.jv(-0.25, k * np.sqrt(2) * dip_const * xaxis))

            return nat_abund * spin_fact * (
                2 * spin - (spin_fact * np.pi * np.sqrt(2) / 4) *
                sum([bessel_func(k) for k in range(1, int(2*spin+1))]))

        # RESPDOR analysis via Bessel function approach
        # after Goldbourt et al. doi: 10.1016/j.ssnmr.2018.04.001
        # x0 = 500    # Initial guess of dipolar coupling constant in Hertz
        nat_abund = 1  # Placeholder make abfragbar ##############
        # lower nat abundancy bound as ugly work around
        [dipole_const, res2] = (
            opt.curve_fit(
                sat_rec,
                respdor_ntr[0:fit_max]/1000,
                respdor_int[0:fit_max],
                bounds=(1, np.inf),
                p0=300))

        # Exporting data
        self.export(respdor_ntr, respdor_int, 'nTr / ms', 'DS/S0',
                    'RESPDOR')
        scale = np.round(np.linspace(respdor_ntr[0], respdor_ntr[-1], 1001),
                         decimals=4)
        fit = sat_rec(scale / 1000, dipole_const)

        result = ('dip_const = ' + str(np.round(dipole_const, decimals=4)) + ' Hz')

        self.export(scale, fit, 'NTr / ms', 'DS/S0', 'Fit', result)

        # Creating and saving RESPDOR plot
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))

        plt.scatter(respdor_ntr, respdor_int)
        plt.plot(scale, sat_rec(scale / 1e3, dipole_const),
                 '--', color='b')

        plt.xlabel(r'$nTr$ / ms')
        plt.ylabel(r'$\Delta$ S / S$_0$')
        plt.ylim(-0.05, 1.2)
        fig.savefig(self.path + '\\' + self.name + '_RESPDOR.png',
                    format='png', dpi=300, bbox_inches='tight')

        return dipole_const[0]

    def redor_eval(self, spin=False, fit_lim=False):
        """Returns the heterodipolar second moment value of a REDOR experiment
        and exports the deltaS/S0 data set together with a quadratic fit.

        Parameters
        ----------

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.experiment = 'REDOR'

        self.fit_lim = fit_lim

        self.integrate()

        if self.vendor == 'bruker':
            spin_rate = self.dic['acqus']['CNST'][31]
        elif self.vendor == 'varian':
            spin_rate = self.dic['procpar']['srate']['values'][0]

        # Number of points in F1 dimension
        number_indirect_points = self.dic['acqu2s']['TD']

        redor_int = np.round((self.area[:, 1]-self.area[:, 0])
                                / self.area[:, 1], decimals=4)  # calc DS
        redor_int = np.insert(redor_int, 0, 0)

        loop_increment = (self.dic['acqus']['L'][1])

        # Builds the time scale in ms
        redor_ntr = (np.round(np.arange(2/spin_rate,
                                          ((number_indirect_points/2)+0.1) *
                                          (loop_increment/spin_rate),
                                          ((loop_increment/spin_rate))) * 1e3,
                                decimals=3))
        redor_ntr = np.insert(redor_ntr, 0, 0)

        fit_max = np.where(redor_int > fit_lim)

        # REDOR analysis via quadratic function,
        # typically within dS/S0 regime < 0.2 for glasses

        def quad_func(xaxis, const):
            return const*xaxis**2

        # Initial guess of curvature value
        x0 = 10000
        # sigma = np.ones(fit_max[0][0]) # Std. deviation of y-data
        [res1, res2] = (opt.curve_fit(quad_func,
                                      redor_ntr[0:fit_max[0][0]],
                                      redor_int[0:fit_max[0][0]], x0))

        # Since redor_ntr is in ms, the M2 unit is 1e6 rad²s-²
        second_moment = (res1*(spin*(spin+1)*(np.pi**2)))

        # Exporting data
        self.export(redor_ntr, redor_int, 'nTr / ms', 'DS/S0', 'REDOR')

        scale = np.round(np.linspace(redor_ntr[0], redor_ntr[-1], 1001),
                         decimals=4)
        fit = quad_func(scale, res1)

        result = ('M2 = ' + str(np.round(second_moment, decimals=4))
                  + ' rad^2 s^-2')

        self.export(scale, fit, '$NT_r$ / ms', 'DS/S0', 'Fit', result)

        # Creating and saving REDOR plot
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))

        plt.scatter(redor_ntr, redor_int, color='w', edgecolor='k')
        plt.plot(scale, quad_func(scale, res1), '--', color='k')

        plt.xlabel(r'$nT_r$ / ms')
        plt.ylabel(r'$\Delta$S / S$_0$')
        plt.ylim(-0.05, 1.2)
        fig.savefig(self.path + '\\' + self.name + '_REDOR.png',
                    format='png', dpi=300, bbox_inches='tight')

        return second_moment

##### Global Functions #####

# Background correction


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

    # Rescaling the data
    num_points = len(xaxis)
    i = np.argsort(xaxis)

    yaxis = yaxis[i]
    maxy = np.max(yaxis)
    dely = (maxy-np.min(yaxis))/2
    num_points_corr = (2 * (xaxis[:] - xaxis[num_points-1])
                       / (xaxis[num_points-1]-xaxis[0]) + 1)

    yaxis = (yaxis[:] - maxy) / dely + 1
    # Creating Vandermonde matrix
    const_p = np.arange(0, order+1, 1)
    # np.tile repeats arrays num_points_corr and const_p
    var_t = np.tile(num_points_corr,
                    (order+1, 1)).T ** np.tile(const_p, (num_points, 1))
    # analog to MATLAB's pins function
    tinv = np.linalg.pinv(np.matmul(var_t.T, var_t))
    tinv = np.matmul(tinv, var_t.T)
    # Initialisation (least-squares estimation)
    aux = np.matmul(tinv, yaxis)
    back_fun = np.matmul(var_t, aux)
    # Other variables
    alpha = 0.99 * 0.5
    it = 0
    zp = np.ones(num_points)

    # Fitting loop
    while (np.sum((back_fun-zp))**2)/(np.sum((zp))**2) > (1e-09):
        it += 1  # Iteration
        zp = back_fun  # Previous estimation
        res = yaxis - back_fun  # Residual
        # Add different functions atq, sh etc. here
        d = ((res*(2*alpha-1))*((res < threshold)*1)
             + (alpha*2*threshold-res) * ((res >= threshold)*1))
        aux = np.matmul(tinv, (yaxis+d))  # Polynomial coefficients a
        back_fun = np.matmul(var_t, aux)  # Polynomial

    # Rescaling
    j = np.argsort(i)
    back_fun = (back_fun[j]-1) * dely + maxy
    aux[1] = aux[1]-1
    aux = aux * dely
    return back_fun



#----------------------------------------------------------------------------#

# SED
Path = (r"C:\Users\HB\data_work\Projects\1_SiO2-Li2O-Nb2O5\7Li_NMR\SED\LNS\20231402-7Li-Li44-SED-1.fid")

# # #         # + r'\210722-7Li-LS2-cryst_SEDLT.fid')
nmr_data = Dataset(Path, 'LNS44_1', 'varian')
M2 = nmr_data.sed_eval(export='zoom', fid=True)
# Fix that if predefined values are used it doesn't show the dialog for the other functions 4, 0.1, 4, 500

# result = Dataset.respdor_eval(nmr_data, spin = 9/2, fit_lim = 2)
# result  = nmr_data.sae_eval()

# SAE
# Path = (r"C:\Users\HB\data_work\Projects\1_Crystallization_I\LS2\7Li_SAE\20220503-7Li-LS2-cryst-SAE.fid")
# nmr_data = Dataset(Path, 'cryst-mono', 'varian')
# result  = nmr_data.sae_eval(fid=True, fit_lim=7)

# REDOR
# Path = (r"C:\Users\HB\sciebo\data\NMR Data Bruker\600MHz SC\nmr\PZABP\10\pdata\1")
# nmr_data = Dataset(Path, 'PZAB_AL6', 'bruker', 4, 1, 4, 100)
# result  = nmr_data.redor_eval(spin = 1/2, fit_lim = 0.3)

# # RESPDOR
# Path = (r"C:\Users\edwu5ea1\data_work\600MHz SC\nmr\7Li-93Nb-LNS\3\pdata\1")
# nmr_data = Dataset(Path, 'Test2', 'bruker', 2, 0.5, 4, 50)
# new_area = np.loadtxt(r'C:\Users\edwu5ea1\data_work\600MHz SC\nmr\7Li-93Nb-LNS\3\pdata\1\area.txt')
# nmr_data.area = new_area
# nmr_data.dic['acqu2s']['TD'] = 28
# result  = nmr_data.respdor_eval(spin = 9/2, fit_lim = 5)

# T1
# Path = (r"C:\Users\HB\data_work\Projects\1_Crystallization_I\LS2\7Li_Satrec\220602-7Li-LS2-25d_Satrec-static")
# nmr_data = Dataset(Path, '10d-biexp', 'varian', 4, 1, 4, 100)
# test_var = nmr_data.t1_eval()

# T1rho
# Path = (r"C:\Users\HB\sciebo\data\NMR Data Bruker\600MHz SC\nmr\7Li_LS2-spinlock\18\pdata\1")
# nmr_data = Dataset(Path, '60dFokin_10kHz', 'bruker')
# test_var = nmr_data.t1rho_eval(fid=True)

# if __name__ == '__main__':


