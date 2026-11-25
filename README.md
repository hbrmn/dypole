# dypole

Python tools for evaluating solid-state NMR experiments that probe dipolar
couplings and relaxation, mainly in glasses. dypole reads Varian/Agilent and
Bruker data with [nmrglue](https://www.nmrglue.com/), processes the FIDs and
fits the decay or recoupling curves.

| Experiment | Function | Status |
|---|---|---|
| Spin-echo decay (SED), homonuclear M2 | `Dataset.sed_eval`, `sed_replicates`, `sed_stitch` | tested |
| REDOR (initial quadratic regime) | `Dataset.redor_eval` | experimental |
| RESPDOR (Bessel-function fit) | `Dataset.respdor_eval` | experimental |
| T1 saturation recovery, T1rho, SAE | `Dataset.t1_eval`, `t1rho_eval`, `sae_eval` | experimental |

**Status:** research code under active development. The interface is
interactive in places (it asks for processing parameters and fit ranges) and
may change between versions. If you use it for published work, cite the
archived version you used (see [Citing](#citing)).

## Installation

Python 3.8 or newer. dypole is not on PyPI, so `pip install dypole` does not
work; install it from GitHub instead.

A released version, e.g. 0.1.0 (recommended for reproducing published
results):

```
pip install git+https://github.com/hbrmn/dypole@v0.1.0
```

The latest development version:

```
pip install git+https://github.com/hbrmn/dypole
```

From a local copy, e.g. to modify the code or run the tests (`.` is the
repository folder):

```
git clone https://github.com/hbrmn/dypole
cd dypole
pip install .
```

Without installing: `pip install -r requirements.txt` and keep `dypole.py`
next to your scripts.

After installing, use it with `import dypole`; `dypole.__version__` shows the
installed version.

## Spin-echo decay (SED)

```python
import dypole

data = dypole.Dataset(r'path/to/sample-SED.fid', 'sample', 'varian')
m2 = data.sed_eval(fid='echo', fit_range=(0, 8), export='zoom')
print(m2, data.m2_err)   # M2 and its fit error in 1e6 rad^2/s^2
```

`sed_eval` fits ln(I/I0) against (2&tau;)&sup2; over the points
`fit_range=(first, last)` (asked for interactively if not given) and returns
the second moment M2 = &minus;2 &times; slope in 10&#8310; rad&sup2;/s&sup2;. The
intensity I can come from

- `fid='echo'`: **echo-aligned integration** of the FIDs (recommended when
  the FIDs start before or on the echo top). The echo position is modeled
  from the pulse timing as `offset + sw * 1 us * (t1Xecho - t2Xecho)` points,
  and every FID is integrated (phased, with linear interpolation) over the
  same window of `echo_npts` points starting `echo_start` points after its
  echo top. The first `echo_skip` points are treated as corrupted. FIDs of
  aborted Varian runs (fewer scans) are excluded automatically.
- `fid=True`: the FID maxima.
- `fid=False`: integrals of the processed spectra (interactive).
- `fid='custom_int'`: intensities set beforehand in `data.custom_int`.

Echo delays are read from `t1Xecho`/`t2Xecho` (Varian) or the `vdlist` file
(Bruker), or can be passed as `delays=` (in &mu;s).

Two helpers work on several data sets of one sample:

- `sed_replicates(paths, fit_range=...)` evaluates repeated measurements and
  returns the individual M2 values with mean, SD, SEM and the 95 % confidence
  interval.
- `sed_stitch(paths)` combines series measured with different &tau; lists
  (all containing &tau; = 20 &mu;s) into one decay curve and fits the initial
  Gaussian M2 and a compressed exponential to the whole curve.

More examples are in [`examples/`](examples).

## Other experiments

`redor_eval`, `respdor_eval`, `t1_eval`, `t1rho_eval` and `sae_eval` were
written for specific pulse programs and data sets and are interactive. Treat
their results with care and check them against an independent evaluation.
Known limitations:

- `redor_eval` converts the curvature to M2 with a factor S(S+1)&pi;&sup2;,
  which equals the standard 3&pi;&sup2;/4 only for S = 1/2.
- `respdor_eval` assumes 100 % natural abundance of the non-observed nucleus.
- Some Bruker parameters (spinning rate in `CNST31`, loop counter `L1`) are
  specific to the pulse programs they were written for.

## Tests

```
pip install pytest
pytest
```

The tests use synthetic Varian SED data sets and check that the SED results
are unchanged from the version used before the 0.1.0 cleanup.

## Citing

Each release is archived on Zenodo with its own DOI. Please cite the version
you used, e.g.

> H. Bradtmüller, dypole, version 0.1.0 (2026), Zenodo, https://doi.org/10.5281/zenodo.XXXXXXX

GitHub's "Cite this repository" button gives the same information from
[`CITATION.cff`](CITATION.cff).

## License

GNU General Public License v3.0 or later, see [LICENSE](LICENSE).
