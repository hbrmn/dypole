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
"""
##### Package imports
import data_processing as dp

##### Script Inputs and processing parameters
# dataPath = r"/home/rm7c8/ownCloud/data/NMR Data Bruker/300MHz MS/nmr/ALW6/26/pdata/1"
# Path to raw data
data_path = r"E:\DL\NMR_data_Henrik\NaPO3_2105\2\pdata\1"
# data_path = r"C:\Users\edwu5ea1\Documents\sciebo\data\NMR Data Bruker\300MHz MS\nmr\ALW6\25\pdata\1"
file_name = 'NaPO3'   # Output file name
debug = 1   # debug = 1 to show all spectra and integration limits, debug = 2 to display REDOR curve and fit
export = 1   # export = 1 to export redor curve and fits

# procPar: Parameters for data processing.
proc_par = {'vendor' : 'bruker',     # 'bruker' or 'varian'
            'ini_zero_order':0,      # Initial phase value for zero-order phasing
            'ini_first_order':-100.0,   # Initial phase value for first-order phasing
            'first_order':-114,        # first order phase correction value
            'zero_order':-120,         # zero order phase correction value
            'line_broadening':50.0, # Lorentzian broadening in Hz
            'zero_fill': 1,          # Zero filling to a total of value*td
            'cutoff' : 1,          # Cuts off FID by given factor. E.g., a value of 0.1 will use only 10% of the FID's points.
            'number_shift_points' : 0, # Number of points to shift the FID
            'auto_phase':0 }         # 1 turns automatic phase correction on

# bgc: Parameters for background correction
bgc_par = {'enabled': 0,          # 1 enabled, else disables
             'order': 7,            # defines or der of polynomial function
             'threshold': 0.02,  # threshold value for detecting what is signal and what is noise from data
             'function': 'ah'}      # Cost function - for now only takes asymmetric huber function 'ah'.


##### Calling scripts
[freq,ppm,spec,dic,sfo_point] = dp.process(data_path, proc_par, bgc_par, file_name, export, debug)


