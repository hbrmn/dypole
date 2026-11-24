"""Synthetic Varian Hahn-echo SED data sets for the tests.

Each FID contains a Gaussian echo whose top sits at
``offset + sw * 1 us * (tau - tau2)`` points and whose height decays as
``exp(-M2 / 2 * (2 tau)^2)`` (M2 in 1e6 rad^2/s^2, tau in ms), plus seeded
noise and a corrupted first point.
"""

import copy
import os

import nmrglue as ng
import numpy as np

SW = 1.0e6          # Hz, i.e. 1 point per us
NP = 128            # complex points per FID


def _param(name, values, template):
    entry = copy.deepcopy(template)
    entry['name'] = name
    entry['values'] = [str(v) for v in values]
    entry['basictype'] = '1'
    return entry


def write_sed(path, tau, tau2=None, m2=1.5, offset=6.0, width=8.0,
              noise=2e-3, seed=0, phase=0.4, aborted=0):
    """Writes a Varian .fid directory with one FID per echo delay.

    Parameters
    ----------
    path : str
        Directory to create.
    tau, tau2 : array of float
        Echo delays t1Xecho and t2Xecho in us (tau2 None: fixed at 0).
    aborted : int
        Number of FIDs at the end recorded with fewer scans (aborted run).
    """
    tau = np.asarray(tau, dtype=float)
    shift = tau - (0 if tau2 is None else np.asarray(tau2, dtype=float))
    rng = np.random.default_rng(seed)
    idx = np.arange(NP)
    data = np.empty((len(tau), NP), dtype=np.complex64)
    for i, (t, s) in enumerate(zip(tau, shift)):
        amp = np.exp(-m2 / 2 * (2 * t * 1e-3) ** 2)
        top = offset + SW * 1e-6 * s
        echo = amp * np.exp(-0.5 * ((idx - top) / width) ** 2)
        fid = echo * np.exp(1j * phase)
        fid = fid + noise * (rng.standard_normal(NP)
                             + 1j * rng.standard_normal(NP))
        fid[0] = 0.1 * fid[0]            # corrupted first point
        data[i] = fid * 1e5

    udic = {'ndim': 1, 0: {'size': NP, 'complex': True, 'encoding': 'direct',
                           'sw': SW, 'obs': 155.4, 'car': 0, 'label': 'Li7',
                           'time': True, 'freq': False}}
    dic = ng.varian.create_dic(udic)
    template = dic['procpar']['np']
    pp = dic['procpar']
    pp['sw'] = _param('sw', [SW], template)
    pp['reffrq1'] = _param('reffrq1', [155.4], template)
    pp['sfrq'] = _param('sfrq', [155.4], template)
    pp['t1Xecho'] = _param('t1Xecho', tau, template)
    if tau2 is not None:
        pp['t2Xecho'] = _param('t2Xecho', np.atleast_1d(tau2), template)
    dic['nblocks'] = len(tau)
    dic['np'] = 2 * NP
    blocks = []
    for i in range(len(tau)):
        bh = ng.varian.make_blockheader(dic, i + 1)
        bh['ctcount'] = 16 if i < len(tau) - aborted else 4
        blocks.append(bh)
    dic['blockheader'] = blocks
    os.makedirs(path, exist_ok=True)
    ng.varian.write(path, dic, data, overwrite=True)
    return path


TAU = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])


def moving_series(path, **kw):
    """Echo moves through the acquisition window (t2Xecho fixed)."""
    return write_sed(path, TAU, tau2=8.0, **kw)


def top_series(path, **kw):
    """Acquisition starts at the echo top (t2Xecho = t1Xecho)."""
    tau = np.array([20, 120, 160, 200, 300])
    return write_sed(path, tau, tau2=tau, **kw)
