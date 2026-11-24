"""Relaxation and heteronuclear recoupling evaluations.

These functions are experimental: they are interactive, were written for
specific pulse programs and have not been tested beyond the data sets they
were developed for. Check the results before using them.

Dataset(path, name, vendor, ls, trim, zf, lb) takes the processing
parameters left shift, fraction of the FID kept, zero-filling multiplier and
exponential line broadening in Hz. Parameters left out are asked for.
"""

import dypole

# Saturation recovery T1 (Varian: delays in d2)
data = dypole.Dataset(r'path/to/Satrec.fid', 'sample', 'varian', 4, 1, 4, 100)
print(data.t1_eval())

# T1rho (spin lock); Bruker: vdlist in the experiment directory or delays=
data = dypole.Dataset(r'path/to/bruker/18', 'sample', 'bruker')
print(data.t1rho_eval(fid=True))

# SAE (Varian: mixing times in tau2)
data = dypole.Dataset(r'path/to/SAE.fid', 'sample', 'varian')
print(data.sae_eval(fid=True, fit_lim=7))

# REDOR, initial quadratic regime (Delta S / S0 < fit_lim)
data = dypole.Dataset(r'path/to/bruker/18', 'sample', 'bruker')
print(data.redor_eval(spin=1/2, fit_lim=0.2))

# RESPDOR, Bessel-function fit (Goldbourt et al.)
data = dypole.Dataset(r'path/to/bruker/21', 'sample', 'bruker', 4, 0.5, 4, 50)
print(data.respdor_eval(spin=9/2, fit_lim=3))
