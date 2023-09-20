# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:31:50 2022

@author: HB
"""

import csv
import numpy as np
import scipy.optimize as opt
import scipy.special as ss
import matplotlib.pyplot as plt


fig_width_pt = 336.0  # From LaTeX: \showthe\columnwidth
inches_per_pt = 1.0 / 72.27                    # pt to inch
golden_mean = ((5)**(0.5) - 1.0) / 2.0         # Golden ratio
fig_width = fig_width_pt * inches_per_pt  # width in inch
fig_height = fig_width * golden_mean # height in inch
plt.rc('lines', linewidth=1)
plt.rc('axes', titlesize=11, labelsize=11)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

name = 'LS2-15min'
path = r'C:\Users\HB\data_work\Projects\1_Crystallization_I\LS2\Thermal Analysis\Export_mW\ExpDat_LS2-15min.txt'
data = np.loadtxt(path, skiprows=36)

start_val = int(len(data)*0.75)
end_val = int(len(data)*0.88)

time = data[start_val:end_val, 1]*60
temp = data[start_val:end_val, 0]

DSC = data[start_val:end_val, 2]





# plt.plot(time, DSC)

def y1(time, a1):
    return(a1 * -1e-4 * time)

def y2(time, b2):
    return(b2 * -1e-4 * time)

def y1p2(time, a1, b2):
    return(y1(time, a1) + b2 * -1e-4 * time)

def y3(time, b3, m3, tau3):
    return(1 - (1/(1 + tau3 * np.exp(-(-b3 * 1e-2) * (time - (m3 * 1e2))))**(1 / tau3)))

def yrev(time, a1, b2, b3, m3, tau3):
    return(y1(time,a1) + (y2(time,b2) * y3(time, b3, m3, tau3)))

def y4(time, c, b4, m4, tau4):
    nominator = (c*1e2 * (-b4 * 1e-2) * np.exp(-(-b4 * 1e-2) * (time - (m4 * 1e2))))
    denominator = (1 + tau4 * np.exp(-(-b4 * 1e-2)* (time - (m4 * 1e2))))**((1 + tau4) / tau4)
    return(nominator / denominator)

def ytot(time, a1, b2, b3, b4, c, m3, m4, tau3, tau4, B):
    return((yrev(time, a1, b2, b3, m3, tau3) + y4(time, c, b4, m4, tau4)) + B)

### Debugging
# time = np.linspace(0, 1000, 101)

# a1 = 3.4
# b2 = 5.5

# tau3 = 0.1
# b3 = 1.5
# m3 = 24.2

# c = 0.77
# b4 = 4
# m4 = 25.13
# tau4 = 1

# B = -4

# plt.plot(y1(time, a1))
# plt.plot(y1p2(time, a1, b2))
# plt.plot(y3(time, b3, m3, tau3))
# plt.plot(yrev(time, a1, b2, b3, tau3, m3))
# plt.plot(y4(time, c, b4, m4, tau4))
# plt.plot(time, ytot(time, a1, b2, b3, b4, c, m3, m4, tau3, tau4, B))
### Debugging End

[popt, _] = (opt.curve_fit(ytot, time, DSC, 
                           p0=[3.4, 5.5, 1.5, 4, 0.77, 24, 25, 1, 1, -4], 
                           bounds=(
                               np.array([1, 1, 0, 0, 0, 18, 18, 0 , 0.3, -10]), 
                               np.array([100, 100, 10, np.inf, 100, 30, 30, 1, 1, 0]))))

fit = ytot(time, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
tg_comp = yrev(time, popt[0], popt[1], popt[2], popt[5], popt[7])
relax_comp = y4(time, popt[4], popt[3], popt[6], popt[8])

Tg_index = np.where(data[:,1] >= (popt[5]*100)/60)
Tg = data[:,0][Tg_index[0][0]]

Tr_index = np.where(data[:,1] >= (popt[6]*100)/60)
Tr = data[:,0][Tr_index[0][0]]

fig = plt.figure(figsize=(fig_width, fig_height))

plt.plot(temp, DSC, linewidth=3, color='k')
plt.plot(temp, fit, linestyle='dashed', color='c')

plt.axvline(x=Tg, linewidth=0.5, linestyle = '-',color = 'k')
plt.axvline(x=Tr, linewidth=0.5, linestyle = '-',color = 'k')

plt.text(Tg-14, -5, '$T_\mathrm{g}$ = ' + str(np.round(Tg, decimals=1)) + ' °C')
plt.text(Tr+1, -5, '$T_\mathrm{r}$ = ' + str(np.round(Tr, decimals=1)) + ' °C')

plt.plot(temp, tg_comp + popt[9], linestyle='dashed', color='b')
plt.text(482, -6.2, 'glass transition', fontsize=8, color='b')

plt.plot(temp, relax_comp + popt[9] -4.2, linestyle='dashed', color='g')
plt.text(438, -6.4, 'relaxation', fontsize=8, color='g')

plt.xlabel(r'Temperature / °C')
plt.ylabel(r'Heat Flow / mW')

fig.savefig(name + '_DSC.png',
            format='png', dpi=300, bbox_inches='tight')


