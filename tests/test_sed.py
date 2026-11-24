"""Tests for the SED evaluation, run with ``pytest`` from the repository."""

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import dypole  # noqa: E402
import regression_values  # noqa: E402
import synthetic  # noqa: E402


@pytest.fixture(scope='module')
def values(tmp_path_factory):
    return regression_values.compute(dypole, str(tmp_path_factory.mktemp('sed')))


def _compare(new, old, key=''):
    if isinstance(old, dict):
        assert set(new) == set(old), key
        for k in old:
            _compare(new[k], old[k], key + '/' + k)
    else:
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12,
                                   err_msg=key)


def test_matches_pre_cleanup_results(values):
    """The numbers are those of the code before the 0.1.0 cleanup."""
    with open(os.path.join(HERE, 'regression_baseline.json')) as file:
        baseline = json.load(file)
    _compare(json.loads(json.dumps(values)), baseline)


def test_recovers_synthetic_m2(values):
    # The synthetic echoes decay with M2 = 1.5e6 rad^2/s^2
    assert values['echo_m2'] == pytest.approx(1.5, abs=0.1)
    assert values['stitch']['m2'] == pytest.approx(1.5, abs=0.1)
    assert values['echo_offset'] == pytest.approx(6.0)


def test_aborted_fid_is_excluded(tmp_path):
    path = synthetic.moving_series(str(tmp_path / 'a.fid'), aborted=2)
    data = dypole.Dataset(path, 'a', 'varian')
    _, valid = data.echo_int(*_delays(data))
    assert valid.tolist() == [True] * 12 + [False] * 2


def _delays(data):
    tau, tau2 = data.echo_delays()
    return tau, 4, 10, 2, tau2


def test_export_writes_files(tmp_path):
    path = synthetic.moving_series(str(tmp_path / 'e.fid'))
    data = dypole.Dataset(path, 'e', 'varian')
    data.sed_eval(fid='echo', fit_range=(0, 8), export='zoom')
    for suffix in ('SED.csv', 'Fit.csv', '_SED.png'):
        assert os.path.isfile(os.path.join(path, 'e' + suffix))


def test_explicit_delays_match_procpar(tmp_path):
    path = synthetic.moving_series(str(tmp_path / 'd.fid'))
    m2 = dypole.Dataset(path, 'd', 'varian').sed_eval(fid=True,
                                                      fit_range=(0, 8))
    m2_delays = dypole.Dataset(path, 'd', 'varian').sed_eval(
        fid=True, fit_range=(0, 8), delays=synthetic.TAU)
    assert m2 == pytest.approx(m2_delays)


def test_frequency_axis_matches_spectrum(tmp_path, monkeypatch):
    # trim=0.45 and zf=2 give 114 points, for which a floating-point
    # arange of the frequency axis used to return 115
    path = synthetic.moving_series(str(tmp_path / 'f.fid'))
    answers = iter(['3', '2', '300', '-300', '1'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(answers))
    data = dypole.Dataset(path, 'f', 'varian', ls='auto-smooth2',
                          trim=0.45, zf=2, lb=100)
    m2 = data.sed_eval(fit_range=(0, 8))
    assert len(data.freq_scale) == data.spec.shape[-1] == 114
    assert len(data.ppm_scale) == data.spec.shape[-1]
    assert np.isfinite(m2)


def test_bruker_vdlist(tmp_path):
    expdir = tmp_path / '18'
    (expdir / 'pdata' / '1').mkdir(parents=True)
    (expdir / 'vdlist').write_text('5u\n10u\n0.5m\n2\n')
    data = dypole.Dataset.__new__(dypole.Dataset)
    data.path = str(expdir / 'pdata' / '1')
    np.testing.assert_allclose(data.bruker_delays(),
                               [5e-6, 10e-6, 5e-4, 2.0])
    data.path = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        data.bruker_delays()


def test_import_has_no_side_effects():
    out = subprocess.run([sys.executable, '-c', 'import dypole'],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ''
