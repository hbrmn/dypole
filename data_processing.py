#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of functions to process multidimensional datasets.

Functions
----------
process
    Returns frequency and ppm scales of input dataset as well as the data
    array, data dictionary and transmitter offset value (SFO) in points
bgc
    Returns the background corrected spectrum of the input spectral data.
integration
    Returns the sum of the datapoints in the specified intervals
respdor_eval
    Returns the heterodipolar second moment value of a RESPDOR experiment and
    and exports the deltaS/S0 data set together with the fitted.
redor_eval
    Returns the heterodipolar second moment value of a REDOR experiment and
    and exports the deltaS/S0 data set together with its quadratic fit-function.
sed_eval
    Returns the homonuclear dipole-dipole second moment of an spin-echo
    decay experiment.

Created on Wed Jan 22 14:31:16 2020
Modified May 25

@author: Henrik Bradtmüller

"""
import csv
import numpy as np
import nmrglue as ng
import scipy.optimize as optimization
#from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import scipy.special as ss
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import palettable
#from matplotlib.widgets import Slider, Button, RadioButtons


def process(data_path, proc_par, bgc_par, file_name, export, debug):
    """Returns frequency and ppm scales of input dataset as well as the data
    array, data dictionary and transmitter offset value (SFO) in points

    Parameters
    ----------
    data_path : string
        path to r"..\\6\pdata\\1" folder
    proc_par : dictionary
        'ini_zero_order':-68,   # Initial phase value for zero-order phasing \n
        'ini_first_order':0.0,  # Initial phase value for first-order phasing \n
        'zero_order':-72,       # zero order phase correction value \n
        'line_broadening':0.0,  # Lorentzian broadening in Hz \n
        'zero_fill': 1,         # Zero filling to a total of value*td \n
        'cutoff' : 0.1,         # Cuts off FID by given factor. E.g., a value
                                of 0.1 will use only 10% of the FID's points.\n
        'number_shift_points' : 4 # Number of points to shift the FID \n
        'auto_phase':1          # 1 turns automatic phase correction on \n
    bgc_par : dictionary
        Should contain the following entries
        'enabled': 1,           # 1 enabled, else disables \n
        'order': 8,             # defines or der of polynomial function \n
        'threshold': 0.00002,   # threshold value for detecting what is signal
                                and what is noise from data \n
        'function': 'ah'        # Cost function -for now only takes asymmetric
                                huber function 'ah'. \n
    export : 1                  # 1 enabled, else disables\n
    debug : integer
       Can be either 0 or 1 while the latter shows the (first) spectrum of
       the (possibly 2D) Dataset

    Returns
    -------
    Returns frequency and ppm scales of input dataset as well as the data
    array, data dictionary and SFO in pt value \n
    freq : numpy array with frequency axis \n
    ppm : numpy array with ppm axis \n
    spec : numpy array with shape of 2D dataset - contains fourier transformed
            data\n
    dic : dictionary
    sfo_point :
    """
    # Loads the possibly two dimensional dataset
    if proc_par['vendor'] == 'bruker':
        
        dic, rawdata = ng.bruker.read(data_path)
        
        carrier_frequency = dic['acqus']['SFO1']  # Transmitter frequency
        # rel_car_freq = (carrier_frequency-dic['acqus']['BF1'])*1e6
        
        spectral_width = dic['acqus']['SW_h'] # spectral width
        # Relative transmitter offset to BF1: i.e. SFO1 - BF1
        
        const_o1 = dic['acqus']['O1']
        # TOPSPIN SR value. Spectrometer reference value: i.e. SF - BF1
        
        const_spec_ref = (dic['procs']['SF']-dic['acqus']['BF1'])*1e6
        # Calculates transmitter offset frequency relative to referenced 0 Hz/ppm
        
        rel_offset_frequency = const_o1 - const_spec_ref
        # Checks dimensionality
        
        if 'acqu2s' in dic:
            # checks if rawdata is of right shape. If not, manually create the
            # rawdata array and use the TD parameter of acqu2s
            # (actual number of recorded indirect points) to create dataset
            
            if len(rawdata) != dic['acqu2s']['TD']:
                # guess what real array shape should look like
                # Needs fixing, because it is not reliable
                # aux = ng.bruker.guess_shape(dic)
                aux = np.array([dic['acqu2s']['TD'],dic['acqus']['TD']])
                # reshaping rawdata accordingly. Missing experiments will be zeros
                rawdata = (
                    np.reshape(rawdata,
                               (int((aux[0])),
                                int(aux[1]/2)))[0:dic['acqu2s']['TD'], :])
                
        if dic['acqus']['DIGMOD'] != 0:
            # Removes the ominous group delay artifact when recording digitally
            data = ng.bruker.remove_digital_filter(dic, rawdata)
            # Zerofilling for the amount of points removed by previous step
            data = (ng.proc_base.zf(
                data, (rawdata.shape[rawdata.ndim-1]-data.shape[data.ndim-1])))
        else:
            data = rawdata
    elif proc_par['vendor'] == 'varian':
        dic, data = ng.varian.read(data_path)
        # Spectral width
        spectral_width = np.float64(dic['procpar']['sw']['values'][0])
        # Transmitter frequency
        carrier_frequency = np.float64(dic['procpar']['reffrq1']['values'][0])
        spectrometer_frequency = np.float64(dic['procpar']['sfrq']['values'][0])
        # Calculates transmitter offset frequency relative to referenced 0 Hz/ppm
        rel_offset_frequency = (spectrometer_frequency - carrier_frequency)*1e6
    # Shifting the FID by number_shift_points
    if data.ndim == 1:
        data = data[proc_par['number_shift_points']:]
    else:
        data = data[:, proc_par['number_shift_points']:]
    # zero-filling to a value 2^n
    data = (ng.proc_base.zf(
        data, proc_par['number_shift_points']))
##Data processing
    # Calculates the number of points after which FID is cut off (set to zeros)
    trim = int(proc_par['cutoff']*data.shape[data.ndim-1])
    # Sets FID points to zero after cutoff value
    if data.ndim == 1:
        data[trim:] = np.zeros(len(data[trim:]))
    else:
        data[ :, trim:] = np.zeros(len(data[0, trim:]))
    # Zero fill to 2*TD points
    spec = ng.proc_base.zf_double(data, proc_par['zero_fill'])
    # Calculates Line broadening (Topspin style) and divides value by sw in Hz
    spec = (ng.proc_base.em(spec, lb=(
        proc_par['line_broadening']/(spectral_width))))
    # Create frequency axis
    freq = ((np.arange((spectral_width/2)+rel_offset_frequency,
                       (-spectral_width/2)+rel_offset_frequency,
                       -spectral_width/spec.shape[spec.ndim-1])))
    ppm = freq/carrier_frequency # Create ppm axis
    spec = ng.proc_base.fft(spec) # Fourier transform
    spec = spec/np.max(spec) #normalize data to see if autophase works faster
    if proc_par['auto_phase'] == 1:
        #Automatic phase correction
        spec = ng.proc_autophase.autops(
            spec, "acme", p0=proc_par['ini_zero_order'], p1=proc_par['ini_first_order'])
    # elif proc_par['auto_phase'] == 2:
    #     endPhase = 0
    #     for spectra in spec
    #     while endPhase == 0:    
    #         zeroPhase = input('Set zero order phase value')
    #         firstPhase = input('Set first order phase value')
    #         spec = ng.proc_base.ps(spec, p0=float(zeroPhase), p1=float(firstPhase))
    #         plt.plot(ppm, spec[1, :])
    #         print('Are you satisfied with the phasing?')
    #         endPhase = input('Yes (1) or no (2)?')
    else:
        # Manual phase correction
        spec = ng.proc_base.ps(spec, p0=proc_par['zero_order'], p1=proc_par['first_order'])
    spec = ng.proc_base.di(spec)  # Discard the imaginaries
    if proc_par['vendor'] == 'bruker':
        spec = ng.proc_base.rev(spec) # Reverse the spectrum
    if spec.ndim == 1:
        ## Background correction
        if bgc_par['enabled'] == 1:
            ppm, spec = bgc(ppm, spec, bgc_par['effective_window'], bgc_par['order'],
                               bgc_par['threshold'], bgc_par['function'])
            freq = ppm * carrier_frequency
        if export == 1: #
            spec = spec/np.max(spec) # make normalizing on export optional
            export_var = zip(freq, spec)
            with open(file_name +'_hz.dat', 'w') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerow(('ti: ', file_name))
                writer.writerow(('##freq ', str(np.round(carrier_frequency, decimals=5))))
                for word in export_var:
                    writer.writerows([word])
    if debug == 1:
        # Using times font causes some unexplainable bug on my machine
        # so it is here substituted with serif
        plt.rc('font', **{'family':'serif', 'serif':['serif']})
        if data.ndim == 1:
            plt.plot(ppm, spec)
        else:
            plt.plot(ppm, spec[1,:])

        plt.xlim(proc_par['effective_window'])
        # plt.ylim(-0.05,1.1)
    # spectral center in points. Required for integration
    # (calculating ppm to pt without using any find function)
    sfo_point = (int(np.round(
        (spec.shape[spec.ndim-1]/2)
        +(rel_offset_frequency/ ((spectral_width)/spec.shape[spec.ndim-1])
          ))))
    return(freq, ppm, spec, dic, sfo_point)

########## background correction procedure
def bgc(xax, yax, window, order, threshold, function):
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
    function : string
        DESCRIPTION.

    Returns
    -------
    z : numpy array of float
    contains the background corrected dataset
    a : numpy array of float
    contains the polynomial used in the background correction
    it : int
    returns the number of iterations from the fitting process

    """
    #Rescaling the data
    if window:
        ppm_high = np.argmin(np.abs(xax-window[0]))
        ppm_low = np.argmin(np.abs(xax-window[1]))
        xaxis = xax[ppm_low:ppm_high]
        yaxis = yax[ppm_low:ppm_high]
    else:
        xaxis = xax
        yaxis = yax
    
    num_points = len(xaxis)
    i = np.argsort(xaxis)
    yaxis = yaxis[i]
    maxy = np.max(yaxis)
    dely = (maxy-np.min(yaxis))/2
    num_points_corr = 2 * (xaxis[:] - xaxis[num_points-1]) / (xaxis[num_points-1]-xaxis[0]) + 1
    yaxis = (yaxis[:] - maxy) / dely + 1
    #Creating Vandermonde matrix
    const_p = np.arange(0, order+1, 1)
    #np.tile repeats arrays num_points_corr and const_p
    var_T = np.tile(num_points_corr, (order+1, 1)).T ** np.tile(const_p, (num_points, 1))
    #analog to MATLAB's pins function
    Tinv = np.linalg.pinv(np.matmul(var_T.T, var_T))
    Tinv = np.matmul(Tinv, var_T.T)
    #Initialisation (least-squares estimation)
    a = np.matmul(Tinv, yaxis)
    z = np.matmul(var_T, a)
    #Other variables
    alpha = 0.99 * 0.5
    it = 0
    zp = np.ones(num_points)
    #Fitting loop
    while (np.sum((z-zp))**2)/(np.sum((zp))**2) > (1e-09):
        it += 1     #Iteration
        zp = z      #Previous estimation
        res = yaxis - z #Residual
        #### Add different functions atq, sh etc. here
        d = ((res*(2*alpha-1))*((res < threshold)*1)
             + (alpha*2*threshold-res) * ((res >= threshold)*1))
        a = np.matmul(Tinv, (yaxis+d))   #Polynomial coefficients a
        z = np.matmul(var_T, a)          #Polynomial
    #Rescaling
    j = np.argsort(i)
    z = (z[j]-1) * dely + maxy
    a[1] = a[1]-1
    a = a * dely
    if window:
        spec = yax[ppm_low:ppm_high] - z
        ppm = xax[ppm_low:ppm_high]
    else:
        spec = yax -z
        ppm = xax
    return(ppm, spec)

########## Baseline correction
def integration(data, integ_par, bgc_par, debug):
    """Returns the sum of the datapoints in the specified intervals

    Parameters
    ----------
    data : numpy array of float
        Possible 2D dataset containing spectral intensities
    integ_par : dictionary
        'lowerIntegLimit': -170.0 \n
        'upperIntegLimit': 120.0 \n
        'experiment':'SED'    # Choose between 'REDOR', 'RESPDOR, or 'SED'
    bgc_par : dictionary
        Should contain the following entries
        'enabled': 1,           # 1 enabled, else disables \n
        'order': 8,             # defines or der of polynomial function \n
        'threshold': 0.00002,   # threshold value for detecting what is signal
                                and what is noise from data \n
        'function': 'ah'        # Cost function -for now only takes asymmetric
                                huber function 'ah'. \n
    debug : integer
       Can be either 0 or 1 while the latter shows the (first) spectrum of
       the (possibly 2D) Dataset

    Returns
    -------
    area : numpy array of float
        Array of integrated intensities. Array has one column for SED data
        input and two columns for REDOR and RESPDOR - S and S0 respectively.

    """
    # ppm per point
    ppm_per_pt = ((np.abs(data['ppm_scale'][-1])
                   +np.abs(data['ppm_scale'][0])) / len(data['ppm_scale']))
    min_int = int(data['sfo_point'] - integ_par['lowerIntegLimit']/ppm_per_pt)
    max_int = int(data['sfo_point'] - integ_par['upperIntegLimit']/ppm_per_pt)
    # Number of points in F2 dimension
    #numberDirectPoints = data['dictionary']['acqus']['TD']
    # Number of points in F1 dimension
    number_indirect_points = data['spectra'].shape[0]
    # Initialize area parameter
    if integ_par['experiment'] == 'REDOR' or 'RESPDOR':
        area = (np.ones(number_indirect_points)
                .reshape(int(number_indirect_points/2), 2))
    elif integ_par['experiment'] == 'SED' or 'T1':
        area = np.ones(number_indirect_points)
    # Background correction and integration
    for i in range(number_indirect_points):
        #normalize data to see if autophase works faster
        data['spectra'] = data['spectra']/np.max(data['spectra'])
        if bgc_par['enabled'] == 1:
            data['spectra'][i, :] = (bgc(data['ppm_scale'],
                                            data['spectra'][i, :],
                                            bgc_par['effective_window'],
                                            bgc_par['order'],
                                            bgc_par['threshold'],
                                            bgc_par['function'])[1])
        # Normalization
        # Integration
        if integ_par['experiment'] == 'REDOR' or 'RESPDOR':
            # Puts REDOR and Echo intensities in same variable but different columns
            area[int(np.floor(i*0.5)), i%2] = (np.sum(data['spectra'][i, max_int:min_int]))
        elif integ_par['experiment'] == 'SED' or 'T1':
            area[i] = np.sum(data['spectra'][i, max_int:min_int])
        if debug == 1:
            # Using times font causes some unexplainable bug on my machine so
            # it is here substituted with serif
            plt.rc('font', **{'family':'serif', 'serif':['serif']})
            plt.plot(data['ppm_scale'][:], (data['spectra'][i, :]))
            plt.hlines(0, np.max(data['ppm_scale']), np.min(data['ppm_scale']))
            plt.xlim(integ_par['upperIntegLimit']+20,
                     integ_par['lowerIntegLimit']-20)
            plt.ylim(-0.05, 1.1)
            # For Debugging integration region
            plt.vlines(integ_par['lowerIntegLimit'], 0, 1)
            plt.vlines(integ_par['upperIntegLimit'], 0, 1)
    return area

########## Analysis of RESPDOR data
#correct formatting of area is required
def respdor_eval(data, eval_par, export, file_name, debug):
    """Returns the heterodipolar second moment value of a RESPDOR experiment and
    and exports the deltaS/S0 data set together with the fitted bessel function.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    eval_par : TYPE
        DESCRIPTION.
    export : TYPE
        DESCRIPTION.
    file_name : TYPE
        DESCRIPTION.
    debug : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if eval_par['vendor'] == 'bruker':
        spin_rate = data['dictionary']['acqus']['CNST'][31] # MAS spin rate
    elif eval_par['vendor'] == 'varian':
        spin_rate = data['dictionary']['procpar']['srate']['values'][0] # MAS spin rate
    # Number of points in F1 dimension
    number_indirect_points = data['dictionary']['acqu2s']['TD']
    redor_int = (np.round((data['area'][:, 1]-data['area'][:, 0])
                          /data['area'][:, 1], decimals=4))  # Calculates DS
    loop_increment = (2*data['dictionary']['acqus']['L'][1])
    # builds the time scale
    redor_ntr = (np.round
                 (np.arange(2/spin_rate,
                            ((number_indirect_points/2)+0.1)
                            *(loop_increment/spin_rate),
                            ((loop_increment/spin_rate)))* 1e3,
                  decimals=2))
    # REDOR analysis via Bessel function approach
    # after Goldbourt et al. doi: 10.1016/j.ssnm/r.2018.04.001
    # x0 = 500    # Initial guess of dipolar coupling constant in Hertz
    global nat_abund
    global quant_number
    nat_abund = eval_par['nat_abund']
    quant_number = eval_par['quant_number']
    # lower nat abundancy bound as ugly work around
    [res1, res2] = (optimization.curve_fit(func_bessel,
                                           redor_ntr[0:-1-eval_par['limit_bes_fit']]/1000,
                                           redor_int[0:-1-eval_par['limit_bes_fit']],
                                           bounds=([0],
                                                   [np.inf])))
    ########## Exporting data
    if export == 1: #Exporting REDOR points and Besselfunction to .csv files
        export_var = zip(redor_ntr, redor_int)
        with open(file_name +'_REDOR.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(('nTr/ms', 'S0-S/S0'))
            for word in export_var:
                writer.writerows([word])
        scale = np.round(np.linspace(redor_ntr[0], redor_ntr[-1], 1001), decimals=4)
        export_var2 = (zip(scale,
                           np.round(func_bessel(scale/1000, res1[0]), decimals=4)))
        with open(file_name +'_Bessel.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(('nTr/ms', 'S0-S/S0', 'D =',
                             str(np.round(res1[0], decimals=4))+' Hz'))
            for word in export_var2:
                writer.writerows([word])
        # #Exporting REDOR plot with Besselfunctions
        # insert os dephasing spectrum after 6 increments like in Goldbourt paper
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(redor_ntr, redor_int, edgecolors='k')
        plt.plot(scale, func_bessel(scale/1000, res1[0]))
        plt.xlim(0, 4)
        plt.ylim(-0.05, 1)
        plt.xlabel(r"NT$_{\mathrm{r}}$ / ms")
        plt.ylabel(r"$\Delta \mathrm{S} / \mathrm{S}_{0}$")
        fig.savefig(
            file_name + ".png", format='png', dpi=300, bbox_inches='tight')
    if debug == 2:
        ##### Plotting
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(redor_ntr, redor_int, edgecolors='k')
        scale = np.linspace(redor_ntr[0], redor_ntr[-1], 1000)
        plt.plot(scale, func_bessel(scale/1000, res1[0], eval_par['nat_abund']))
        plt.xlim(0, 4)
        plt.ylim(-0.05, 1)
        plt.xlabel(r"NT$_{\mathrm{r}}$ / ms")
        plt.ylabel(r"$\Delta \mathrm{S} / \mathrm{S}_{0}$")
    return res1[0]

########## Definition of Bessel function
def func_bessel(xaxis, dip_const):
    """Returns linearcombinations of Bessel function according to

    Parameters
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
    if quant_number == (3/2):
        return (nat_abund*0.25*
                (3 -(np.pi*np.sqrt(2))/16 *
                 ((6*ss.jv(0.25, 1*np.sqrt(2)*dip_const*xaxis)
                   *ss.jv(-0.25, 1*np.sqrt(2)*dip_const*xaxis))
                  + (4*ss.jv(0.25, 2*np.sqrt(2)*dip_const*xaxis)
                     *ss.jv(-0.25, 2*np.sqrt(2)*dip_const*xaxis))
                  + (2*ss.jv(0.25, 3*np.sqrt(2)*dip_const*xaxis)
                     *ss.jv(-0.25, 3*np.sqrt(2)*dip_const*xaxis)))))    
    elif quant_number == (5/2):
        return (nat_abund*(1/6)*
                (5 -(np.pi*np.sqrt(2))/24 *
                 ((10*ss.jv(0.25, 1*np.sqrt(2)*dip_const*xaxis)
                   *ss.jv(-0.25, 1*np.sqrt(2)*dip_const*xaxis))
                  + (8*ss.jv(0.25, 2*np.sqrt(2)*dip_const*xaxis)
                     *ss.jv(-0.25, 2*np.sqrt(2)*dip_const*xaxis))
                  + (6*ss.jv(0.25, 3*np.sqrt(2)*dip_const*xaxis)
                     *ss.jv(-0.25, 3*np.sqrt(2)*dip_const*xaxis))
                  + (4*ss.jv(0.25, 4*np.sqrt(2)*dip_const*xaxis)
                     *ss.jv(-0.25, 4*np.sqrt(2)*dip_const*xaxis))
                  + (2*ss.jv(0.25, 5*np.sqrt(2)*dip_const*xaxis)
                     *ss.jv(-0.25, 5*np.sqrt(2)*dip_const*xaxis)))))
########## Analysis of REDOR data
#correct formatting of area is required
def redor_eval(data, eval_par, export, file_name, debug):
    """Returns the heterodipolar second moment value of a REDOR experiment and
    and exports the deltaS/S0 data set together with its quadratic fit-function.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    eval_par : TYPE
        DESCRIPTION.
    export : TYPE
        DESCRIPTION.
    file_name : TYPE
        DESCRIPTION.
    debug : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if eval_par['vendor'] == 'bruker':
        spin_rate = data['dictionary']['acqus']['CNST'][31] # MAS spin rate
    elif eval_par['vendor'] == 'varian':
        spin_rate = np.float64(data['dictionary']['procpar']['srate']['values'][0]) # MAS spin rate
    # Number of points in F1 dimension
    number_indirect_points = data['spectra'].shape[0]
    # Calculates DS and adds 0 as first point of array
    redor_int = (np.round(np.insert((data['area'][:, 1]
                                     -data['area'][:, 0])/data['area'][:, 1],
                                    0, 0), decimals=4))
    # Calculate REDOR timescale as NTr in ms!
    redor_ntr = (
        np.round(
            np.arange(0,
                      (((number_indirect_points/2)+0.1))*(2/spin_rate),
                      (2/spin_rate))*1e3, decimals=3))
    # Find max value to be included in fit
    fit_max = np.where(redor_int > eval_par['lim_par_fit'])
    ########## REDOR analysis via quadratic function, typically within dS/S0 regime < 0.2
    def fit_func(xaxis, const):
        return const*xaxis*xaxis                           # Quadratic function
    # Initial guess.
    x0 = 10000                    # Initial guess of curvature value
    # sigma = np.ones(fit_max[0][0]) # Std. deviation of y-data
    [res1, res2] = (optimization.curve_fit(
        fit_func, redor_ntr[0:fit_max[0][0]], redor_int[0:fit_max[0][0]], x0))
    # Since redor_ntr is in ms, the M2 unit is 1e6 rad²s-²
    second_moment = (
        res1*(eval_par['quant_number']*(eval_par['quant_number']+1)*(np.pi**2)))

    ########## Exporting data
    if export == 1: #Exporting REDOR points and Besselfunction to .csv files
        export_var = zip(redor_ntr, redor_int)
        with open(file_name +'_REDOR.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(('nTr/ms', 'S0-S/S0'))
            for word in export_var:
                writer.writerows([word])
        scale = np.round(np.linspace(redor_ntr[0], redor_ntr[-1], 1000), decimals=4)
        export_var2 = zip(scale, fit_func(scale, res1))
        with open(file_name +'_fit.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(
                ('nTr/ms', 'S0-S/S0', 'M2 =',
                 str(np.round(second_moment, decimals=4))+' 1e6 rad^2s^-2'))
            for word in export_var2:
                writer.writerows([word])
        # #Exporting REDOR plot with Besselfunctions
        # insert os dephasing spectrum after 6 increments like in Goldbourt paper
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(redor_ntr, redor_int, edgecolors='k')
        # plt.scatter(redor_ntr, redor_int*1.7, edgecolors='k')
        # plt.plot(redor_ntr, fit_func(redor_ntr, res1))
        plt.plot(scale, fit_func(scale, res1), color='k')
        # plt.xlim(0, 4)
        plt.ylim(-0.05, 1.2)
        plt.xlabel(r"NT$_{\mathrm{r}}$ / ms")
        plt.ylabel(r"$\Delta \mathrm{S} / \mathrm{S}_{0}$")
        fig.savefig(file_name + ".png", format='png', dpi=300, bbox_inches='tight')
    if debug == 2:
        ##### Plotting
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(redor_ntr, redor_int, edgecolors='k')
        scale = np.linspace(redor_ntr[0], redor_ntr[-1], 1000)
        plt.plot(scale, fit_func(scale, res1), color='k')
        # plt.xlim(0, 4)
        plt.ylim(-0.05, 1)
        plt.xlabel(r"NT$_{\mathrm{r}}$ / ms")
        plt.ylabel(r"$\Delta \mathrm{S} / \mathrm{S}_{0}$")
    return second_moment
########## Analysis of spin echo decay (SED) data
#correct formatting of area is required
def sed_eval(data_path, data, eval_par, export, file_name, debug):
    """Returns the homonuclear dipole-dipole second moment of an spin-echo
    decay experiment.

    Parameters
    ----------
    data_path : string
        Path to the ...pdata\1 folder
    data : dictionary
        Contains sfo_point,area,dictionary,freq_scale,ppm_scale,spectra
    eval_par : dictionary
            'lim_par_fit': 0.2,        # Y-axis limit for parabola fit \n
            'limit_bes_fit': 0,        # REDOR points to be omitted from fit
                                       (counting backwards) \n
            'quant_number': 3/2,      # Spin quantum number of non-observed nucleus \n
            'natAbund' : 1,          # Natural abundance of non-observed
                                       nucleus in case of shaped_redor\n
            'front_skip' : 5,        # Number of points to skip in the first points \n
            'fit_max' : 18,          # Maximum number of points to include in fit \n
            'xmax' : 0.04,              # Maximum of abscissa for plotting \n
            'ymax' : 0.1}
    export : integer
        Can be 0 or 1.
    file_name : string
        Filename for export data
    debug : integer
        Can be 0, 1 or 2

    Returns
    -------
    second_moment
        Homonuclear dipole-dipole second moment in units 1e6 rad²/s²
    """
    # Opens vdlist, reads data, puts it in a list and transforms it to array
    vdlist = np.asarray(open(data_path.split('pdata')[0]+'vdlist', 'r')
                        .read().split('u'))
    # adding DE value to vdlist values is for the dead time correction of the spectrometer
    vdlist = vdlist[vdlist != ''].astype(np.float64)+data['dictionary']['acqus']['DE']
    # time scale in ms²
    time = (((vdlist*1e-3)*2)**2)[0:data['dictionary']['acqu2s']['TD']]
    # Calculates log(I/I0)
    sed_int = np.log(data['area']/np.max(data['area']))
    plt.rc('font', **{'family':'serif', 'serif':['serif']})
    ########## Linear regression analysis
    def fit_func_norm(xaxis, slope, yintercept):
        return slope*xaxis+yintercept      # linear function with y-intercept
    def fit_func(xaxis, slope):
        return slope*xaxis     # linear function w/o y-intercept
    # Initial guess.
    # x0 = -100                    # Initial guess of curvature value
    # sigma = np.ones(fit_max[0][0]) # Std. deviation of y-data
    [popt, pcov] = (optimization.curve_fit(
        fit_func_norm, time[eval_par['front_skip']+1:eval_par['fit_max']],
        sed_int[eval_par['front_skip']+1:eval_par['fit_max']]))
    # Normalization of data by y-intercept of first linear fit
    sed_int = np.log(np.exp(sed_int)/np.exp(fit_func_norm(0, popt[0], popt[1])))
    # Corrected linear fitting
    [popt2, pcov2] = (optimization.curve_fit(
        fit_func, time[eval_par['front_skip']+1:eval_par['fit_max']],
        sed_int[eval_par['front_skip']+1:eval_par['fit_max']]))
    # Since time is in ms², the M2 unit is 1e6 rad²s-²
    second_moment = -2*popt2[0]

    ########## Exporting data
    if export == 1: #Exporting REDOR points and Besselfunction to .csv files
        export_var = zip(time, sed_int)
        with open(file_name +'_SED.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(('2tau^2', 'ln(I/I0)'))
            for word in export_var:
                writer.writerows([word])
        scale = np.round(np.linspace(time[0], time[-1], 1000), decimals=4)
        export_var2 = zip(scale, fit_func(scale, popt2[0]))
        with open(file_name +'_fit.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(
                ('2tau^2', 'ln(I/I0)', 'M2 =',
                 str(np.round(second_moment, decimals=4))+' 1e6 rad^2s^-2'))
            for word in export_var2:
                writer.writerows([word])
        # #Exporting REDOR plot with Besselfunctions
        # insert os dephasing spectrum after 6 increments like in Goldbourt paper
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(time, sed_int, color='w', edgecolors='k')
        # plt.scatter(redor_ntr, redor_int*1.7, edgecolors='k')
        # plt.plot(redor_ntr, fit_func(redor_ntr, res))
        plt.plot(scale, fit_func(scale, popt2[0]), color='k')
        plt.xlim(0, eval_par['xmax'])
        plt.ylim(-eval_par['ymax'], 0.01)
        plt.xlabel(r"$(2\tau)^2$ / ms$^2$")
        plt.ylabel(r"ln($\mathrm{I} / \mathrm{I}_{0}$)")
        fig.savefig(file_name + ".png", format='png', dpi=300, bbox_inches='tight')
    if debug == 2:
        ##### Plotting
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.scatter(time, sed_int, edgecolors='k')
        scale = np.linspace(time[0], time[-1], 1000)
        plt.plot(scale, fit_func(scale, popt2[0]), color='k')
        plt.xlim(0, eval_par['xmax'])
        plt.ylim(-eval_par['ymax'], 0.01)
        plt.xlabel(r"$(2\tau^2)$ / ms$^2$")
        plt.ylabel(r"ln($\mathrm{I} / \mathrm{I}_{0}$)")
    return second_moment


# def t1_eval(data, eval_par, export, file_name, debug):
#     """Returns the T1 plot.

#     Parameters
#     ----------
#     data : TYPE
#         DESCRIPTION.
#     eval_par : TYPE
#         DESCRIPTION.
#     export : TYPE
#         DESCRIPTION.
#     file_name : TYPE
#         DESCRIPTION.
#     debug : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """

#     if eval_par['vendor'] == 'bruker':
#         # spin_rate = data['dictionary']['acqus']['CNST'][31] # MAS spin rate
#     elif eval_par['vendor'] == 'varian':
#         relax_delays = np.array(data['dictionary']['procpar']['d2']['values']) # MAS spin rate
#     # Number of points in F1 dimension
#     number_indirect_points = data['spectra'].shape[0]
#     # T1 fitting
#     # Find max value to be included in fit
#     fit_max = np.where(redor_int > eval_par['lim_par_fit'])
#     ########## REDOR analysis via quadratic function, typically within dS/S0 regime < 0.2
#     def fit_func(xaxis, const):
#         return const*xaxis*xaxis                           # Quadratic function
#     # Initial guess.
#     x0 = 10000                    # Initial guess of curvature value
#     # sigma = np.ones(fit_max[0][0]) # Std. deviation of y-data
#     [res1, res2] = (optimization.curve_fit(
#         fit_func, redor_ntr[0:fit_max[0][0]], redor_int[0:fit_max[0][0]], x0))
#     # Since redor_ntr is in ms, the M2 unit is 1e6 rad²s-²
#     second_moment = (
#         res1*(eval_par['quant_number']*(eval_par['quant_number']+1)*(np.pi**2)))

#     ########## Exporting data
#     if export == 1: #Exporting T1 points and fitfunction to .csv files
#         export_var = zip(redor_ntr, redor_int)
#         with open(file_name +'_REDOR.csv', 'w') as f:
#             writer = csv.writer(f, delimiter='\t', lineterminator='\n')
#             writer.writerow(('nTr/ms', 'S0-S/S0'))
#             for word in export_var:
#                 writer.writerows([word])
#         scale = np.round(np.linspace(redor_ntr[0], redor_ntr[-1], 1000), decimals=4)
#         export_var2 = zip(scale, fit_func(scale, res1))
#         with open(file_name +'_fit.csv', 'w') as f:
#             writer = csv.writer(f, delimiter='\t', lineterminator='\n')
#             writer.writerow(
#                 ('nTr/ms', 'S0-S/S0', 'M2 =',
#                   str(np.round(second_moment, decimals=4))+' 1e6 rad^2s^-2'))
#             for word in export_var2:
#                 writer.writerows([word])
#         # #Exporting REDOR plot with Besselfunctions
#         # insert os dephasing spectrum after 6 increments like in Goldbourt paper
#         fig = plt.figure(figsize=(fig_width, fig_height))
#         plt.scatter(redor_ntr, redor_int, edgecolors='k')
#         # plt.scatter(redor_ntr, redor_int*1.7, edgecolors='k')
#         # plt.plot(redor_ntr, fit_func(redor_ntr, res1))
#         plt.plot(scale, fit_func(scale, res1), color='k')
#         # plt.xlim(0, 4)
#         plt.ylim(-0.05, 1)
#         plt.xlabel(r"NT$_{\mathrm{r}}$ / ms")
#         plt.ylabel(r"$\Delta \mathrm{S} / \mathrm{S}_{0}$")
#         fig.savefig(file_name + ".png", format='png', dpi=600, bbox_inches='tight')
#     if debug == 2:
#         ##### Plotting
#         fig = plt.figure(figsize=(fig_width, fig_height))
#         plt.scatter(redor_ntr, redor_int, edgecolors='k')
#         scale = np.linspace(redor_ntr[0], redor_ntr[-1], 1000)
#         plt.plot(scale, fit_func(scale, res1), color='k')
#         # plt.xlim(0, 4)
#         plt.ylim(-0.05, 1)
#         plt.xlabel(r"NT$_{\mathrm{r}}$ / ms")
#         plt.ylabel(r"$\Delta \mathrm{S} / \mathrm{S}_{0}$")
#     return second_moment
########## Style definitions

def stackplot(paths):
    """Returns a stackplot of spectra within given list. For two spectra,
    additionally returns difference spectra.
    Parameters
    ----------
    path_list : LIST
    A list containing strings to the spectra in the (for now) dmfit format with their desired scaling factor and y-offset in percent.
    Todo - add sanity check for same nuclear resonance frequencies among spectra
    """
    # freqs = np.array()
    # ints = np.array()
    # DMFit data format
    imp = [np.split(np.genfromtxt(x[0], dtype=None, usecols = (0,1), 
                                  comments=None, encoding=None), [2], 0 ) 
           for x in paths]
    
    carrier = [np.float(imp[x][0][1,1]) for x in range(len(imp))]
    data = [imp[x][1].astype('float64') for x in range(len(imp))]
    names = [paths[x][2] for x in range(len(paths))]
    
    # Calculate frequency axis limits for truncation of datasets
    freq_limits = [np.min([data[x][0, 0] for x in range(len(data))]), 
                   np.max([data[x][-1, 0] for x in range(len(data))])]
    
    # Get the indices of the upper and lower bounds found in previous step for each spectrum
    iv = [[np.argmin(np.abs(data[x][:,0]-freq_limits[y]))
           for y in range(2)] 
          for x in range(len(data))]

    #####
    # For future - add spline interpolation if dwell-time is different between spectra    
    #####
    
    output = [np.stack((data[x][iv[x][0]:iv[x][1], 0], 
                        data[x][iv[x][0]:iv[x][1], 1]*paths[x][1]), axis=1) 
              for x in range(len(data))]
    
    # This is needed to make the output arrays even in case the approximate "index finder" comes up with different column lenghts
    [output[x].resize(np.min([iv[x][1]-iv[x][0] for x in range(len(iv))]), 2)
     for x in range(len(output))]
    
    # creating difference spectrum
    # output.append(np.array([output[0][:,0],(output[0][:,1]-np.sum([(output[x+1][:,1]) for x in range(len(output)-1)], axis=0))]).T)

    # names.append('Difference')
    # carrier.append(carrier[0])
    # data = [data for x < min]
    
    fig_width_pt = 336.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # 05Convert pt to inch
    golden_mean = ((5)**(0.5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    # Using times font causes some unexplainable bug on my machine so
    # it is here substituted with serif
    # Make Sliders for all spectra before plotting (scaling factors)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{fontenc}\usepackage{siunitx}"
    plt.rc('text', usetex=True)
    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=(cycler('color', palettable.tableau.Gray_5.mpl_colors) + 
                               cycler('linestyle', ['-', '-', '--', '--', '--'])),
                               titlesize=11,
                               labelsize=11)
    plt.rc('font', **{'family':'serif', 'serif':['serif']})
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                           
    [ax.plot(output[x][:,0]/carrier[x], output[x][:,1], label=names[x]) 
     for x in range(len(output))]

    ax.invert_xaxis()
    ax.set_yticks([])
    ax.set_xlim(40,-30)
    # ax.set_xlabel(r"$\delta$($^{1}$H) / ppm")
    ax.set_xlabel(r"$^{1}$H NMR shift / (ppm)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    handles,labels = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[1],handles[2]]
    # labels = [labels[0], labels[1],labels[2]]
    plt.legend(handles,labels,
          fontsize='11', frameon=False, loc=1,
          bbox_to_anchor=(0.6, 0.95, 0, 0),
          ncol=1, mode="expand", borderaxespad=0.)
    # plt.text(0.1, 0.30, "Glass", size=10,
    #          va="baseline", ha="left", multialignment="left",transform=ax.transAxes)

    plt.show()
    
    fig.savefig(names[0] + '.png', format='png', dpi=300, bbox_inches='tight')
    
    stackplot = 0
    diffplot = 0

    return stackplot, diffplot


fig_width_pt = 336.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = ((5)**(0.5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
#Figure options
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{fontenc}\usepackage{siunitx}"
plt.rc('text', usetex=True)
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', palettable.cmocean.sequential.Thermal_20.mpl_colors)),
# plt.rc('axes', prop_cycle=(cycler('color', 'k')),
        titlesize=11,
        labelsize=11)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', **{'family':'serif', 'serif':['Times']})

