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
import M2_calc as m2

##### Script Inputs and processing parameters
# dataPath = r"/home/rm7c8/ownCloud/data/NMR Data Bruker/300MHz MS/nmr/ALW6/26/pdata/1"
# Path to raw data
data_path = r"C:\Users\HB\sciebo\data\NMR Data Bruker\600MHz SC\nmr\P-Nick_Dissolution\18\pdata\1"
# data_path = r"C:\Users\edwu5ea1\Documents\sciebo\data\NMR Data Bruker\300MHz MS\nmr\ALW6\25\pdata\1"
file_name = 'Spinner'   # Output file name
debug = 1   # debug = 1 to show all spectra and integration limits, debug = 2 to display REDOR curve and fit
export = 1   # export = 1 to export redor curve and fits

# procPar: Parameters for data processing.
proc_par = {'vendor' : 'bruker',     # 'bruker' or 'varian'
            'ini_zero_order':0,      # Initial phase value for zero-order phasing
            'ini_first_order':0,   # Initial phase value for first-order phasing
            'first_order':0,        # first order phase correction value
            'zero_order':25,         # zero order phase correction value
            'line_broadening':0, # Lorentzian broadening in Hz
            'zero_fill': 1,          # Zero filling to a total of value*td
            'cutoff' : 1,          # Cuts off FID by given factor. E.g., a value of 0.1 will use only 10% of the FID's points.
            'number_shift_points' : 0, # Number of points to shift the FID
            'auto_phase':0 ,       # 1 turns automatic phase correction on
            'effective_window':[-200,200]} #truncates data to work with in order to facilitate background correction etc

# bgc: Parameters for background correction
bgc_par = {'enabled':0,          # 1 enabled, else disables
             'order': 5,            # defines or der of polynomial function
             'threshold': 0.01,  # threshold value for detecting what is signal and what is noise from data
             'function': 'ah',      # Cost function - for now only takes asymmetric huber function 'ah'.
             'effective_window':[-50,60]}

##### Calling scripts
# [freq,ppm,spec,dic,sfo_point] = dp.process(data_path, proc_par, bgc_par, file_name, export, debug)

paths = [[r'C:\Users\HB\data_work\Projects\8_Nick_Followup\MAS\1H\PB1-P3-7d_350C_hz.dat',1, r'7 d'],
                [r'C:\Users\HB\data_work\Projects\8_Nick_Followup\MAS\1H\PB1-P3-pre_350C_hz.dat',0.32, r'Initial']]#,
                # [r'C:\Users\HB\data_work\Projects\8_Nick_Followup\MAS\1H\Spinner1_hz.dat',0.4, r'Background']]

[stackplot, diffplot] = dp.stackplot(paths)

second_moment = m2.DataSet(r'C:\Users\HB\sciebo\data\XRD Data\H3BO3\B-H_H3BO3.txt', '11B', '1H').second_moment()

# make plotting function when transferring to class-based code
# future: instead of outputting the spectra, output just the processing parameters and the raw data in order to be able to manipulate data later
# Probably will require some data strucutre...JSON
# need plotter function in class
# Add the possibility of building the difference plot from individual spectra or all of them