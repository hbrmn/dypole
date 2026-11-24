"""Spin-echo decay (SED): homonuclear dipolar second moment M2.

Replace the paths with your own data. Varian/Agilent .fid directories with
the echo delays in t1Xecho (and t2Xecho) are read directly. For Bruker data,
pass the echo delays (in us) via ``delays`` or keep a vdlist file in the
experiment directory.
"""

import glob

import dypole

# Single data set, echo-aligned integration of the FIDs (recommended when
# the FIDs start before or on the echo top). fit_range=(first, last) selects
# the points of the linear fit; leave it out to choose interactively.
data = dypole.Dataset(r'path/to/20230828-7Li-SED.fid', 'sample', 'varian')
m2 = data.sed_eval(fid='echo', fit_range=(0, 8), export='zoom')
print(m2, data.m2_err)

# Repeated measurements of one sample: M2 with error bars
result = dypole.sed_replicates(sorted(glob.glob(r'path/to/*sample*.fid')),
                               fit_range=(0, 8))

# Series with different tau lists of one sample (all containing
# tau = 20 us), stitched into one decay curve
curve = dypole.sed_stitch(sorted(glob.glob(r'path/to/*sample*.fid')))

# Integrals from elsewhere (e.g. deconvolution), one value per FID
# import numpy as np
# data.custom_int = np.loadtxt(r'path/to/area.txt')
# m2 = data.sed_eval(fid='custom_int')
