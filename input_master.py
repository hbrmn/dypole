#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:58:58 2020
Modified May 25
@author: Henrik Bradtmüller
To do:
    -Make Git Version

    -make different sorce data formats available

    -long term: make gauss fitting signals possible

    -make SED script - getting fid maximum and cutting all points up to
    that point - zerofill the difference

    -make RMS plot (xi² vs D) to include in graph

    -enable component fitting- make a fit with DMFIT or SIMPSON, import the
    lines in this script and it will optimize each lines intensity according
    to a cost function and in the end return the integral. Repeat for each
    spectrum and profit

    Include signal fitting to 2d Routine
        Czjek and Gaussian functions
"""
##### Package imports
import data_processing as dp

##### Script Inputs and processing parameters
# dataPath = r"/home/rm7c8/ownCloud/data/NMR Data Bruker/300MHz MS/nmr/ALW6/26/pdata/1"
# Path to raw data
# data_path = r"E:\DL\NMR_data_Henrik\Hssen_0Bi\2\pdata\1"
# for nu in range(21):
#     data_path = r"C:\Users\edwu5ea1\Documents\sciebo\data\NMR Data Bruker\300MHz MS\nmr\WURST_RESPDOR_rf-array-1200us\%s\pdata\1" % (nu+1)
#     file_name = 'W_RES_1200us_' + str(50-(2*nu)) + 'kHz'   # Output file name
data_path = r"C:\Users\HB\sciebo\data\NMR Data Bruker\600MHz SC\nmr\P-Nick_Dissolution\28\pdata\1"
file_name = 'PB3-P1-pre-B3'
debug = 1   # debug = 1 to show all spectra and integration limits, debug = 2 to display REDOR curve and fit
export = 1   # export = 1 to export redor curve and fits

# procPar: Parameters for data processing.
proc_par = {'vendor' : 'bruker',     # 'bruker' or 'varian'
            'ini_zero_order':20,      # Initial phase value for zero-order phasing
            'ini_first_order':0,# Initial phase value for first-order phasing
            'first_order':300,        # first order phase correction value
            'zero_order':-120,        # zero order phase correction value
            'line_broadening':150.0,   # Lorentzian broadening in Hz
            'zero_fill': 1,          # Zero filling to a total of value*td
            'cutoff' : 0.25,          # Cuts off FID by given factor. E.g., a value of 0.1 will use only 10% of the FID's points.
            'number_shift_points' : 0, # Number of points to shift the FID
            'auto_phase':1 ,       # 1 turns automatic phase correction on
            'effective_window':[-20,40]} #truncates data to work with in order to facilitate background correction etc

# bgc: Parameters for background correction
bgc_par = {'enabled': 1,          # 1 enabled, else disables
             'order': 7,            # defines or der of polynomial function
             'threshold': 0.02,  # threshold value for detecting what is signal and what is noise from data
             'function': 'ah',      # Cost function - for now only takes asymmetric huber function 'ah'.
             'effective_window':None} # Needs to be implemented for 2D experiments

# integPar: Parameters for signal integration.
integ_par = {'lowerIntegLimit':7,
            'upperIntegLimit': 18,
            'experiment':'REDOR',     # Choose between 'REDOR', 'RESPDOR, or 'SED'
            'effective_window':None} # Implementation of bgc window needs to be done

# evalPar: Parameters for analyzing the dephasing curves.
eval_par  = {'vendor' : 'bruker',    # 'bruker' or 'varian'
            'lim_par_fit': 0.1,      # Y-axis limit for parabola fit
            'limit_bes_fit': 0,      # REDOR points to be omitted from fit (counting backwards)
            'quant_number': 1/2,     # Spin quantum number of non-observed nucleus
            'nat_abund' : 1,         # Natural abundance of non-observed nucleus in case of shaped_redor
            'front_skip' : 0,        # Number of points to skip in the first points
            'fit_max' : 10,          # Maximum number of points to include in fit
            'xmax' : 0.04,           # Maximum of abscissa for plotting
            'ymax' : 0.1}            # Maximum of y-axis - must be a positive value

##### Calling scripts
##### Loading and processing 2D file and its parameters
[freq,ppm,spec,dic,sfo_point] = dp.process(data_path, proc_par, bgc_par, file_name, export, debug)
data = {'dictionary':dic, 'ppm_scale':ppm, 'freq_scale':freq,'spectra':spec, 'sfo_point':sfo_point}
##### Integrates S and S0 spectra and puts it in an array
area = dp.integration(data, integ_par, bgc_par, debug)
data['area'] = area

# # data['area'] = area[:,[1,0]]
# ##### REDOR analysis
# # D = dp.respdor_eval(data,eval_par,export,file_name,debug)
M2 = dp.redor_eval(data,eval_par,export,file_name,debug)

# M2 = dp.sed_eval(data_path,data, eval_par, export, file_name, debug)
#
