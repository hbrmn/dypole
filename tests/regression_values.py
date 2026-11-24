"""Computes the SED results checked by test_sed.py.

Run as ``python tests/regression_values.py <module_dir> <work_dir>`` to print
the values as JSON for the dypole.py found in <module_dir>. Used to pin the
results of the code before the 0.1.0 cleanup.
"""

import builtins
import contextlib
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic  # noqa: E402


def make_data(work):
    paths = {}
    for seed in range(3):
        paths['rep%d' % seed] = synthetic.moving_series(
            os.path.join(work, 'rep%d.fid' % seed), seed=seed, aborted=1)
    paths['top'] = synthetic.top_series(os.path.join(work, 'top.fid'),
                                         seed=7)
    return paths


def compute(dypole, work):
    paths = make_data(work)
    out = {}

    data = dypole.Dataset(paths['rep0'], 'rep0', 'varian')
    out['echo_m2'] = data.sed_eval(fid='echo', fit_range=(0, None))
    out['echo_m2_err'] = data.m2_err
    out['echo_offset'] = float(data.echo_offset)

    data = dypole.Dataset(paths['rep1'], 'rep1', 'varian')
    out['fidmax_m2'] = data.sed_eval(fid=True, fit_range=(0, 8))

    # Spectral mode: processing parameters given, prompts answered
    answers = iter(['3', '2', '300', '-300', '1'])
    real_input = builtins.input
    builtins.input = lambda prompt='': next(answers)
    try:
        data = dypole.Dataset(paths['rep2'], 'rep2', 'varian',
                              ls='auto-smooth2', trim=0.9, zf=2, lb=100)
        out['spectral_m2'] = data.sed_eval(fit_range=(0, 8))
    finally:
        builtins.input = real_input

    rep = dypole.sed_replicates([paths['rep0'], paths['rep1'],
                                 paths['rep2']], fit_range=(0, None))
    out['replicates'] = {k: (np.asarray(v).tolist()) for k, v in rep.items()}

    st = dypole.sed_stitch([paths['rep0'], paths['rep1'], paths['top']])
    out['stitch'] = {k: float(st[k]) for k in
                     ('m2', 'm2_err', 'T', 'beta', 'T_err', 'beta_err',
                      'offset')}
    out['stitch_mean'] = st['mean'].tolist()
    return out


if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
    import dypole as module
    os.makedirs(sys.argv[2], exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        values = compute(module, sys.argv[2])
    print(json.dumps(values, indent=1))
