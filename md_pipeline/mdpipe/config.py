"""Step 1 -- read the master input file and expand the compositional series.

The master input (YAML) is the single source of truth. This module turns it
into a list of "run" dictionaries, one per composition (and replica), each
containing everything the later steps need.
"""

import copy
import os

import numpy as np
import yaml

DEFAULTS = {
    'project': {'name': 'md_series', 'output_dir': 'runs', 'seed': 12345},
    'lammps': {'command': 'lmp', 'mpi_ranks': 1, 'mpi_launcher': 'mpirun -np {n}',
               'run_mode': 'local', 'slurm': {}},
    'system': {'target_atoms': 3000, 'min_distance': 1.6, 'replicas': 1},
    'simulation': {'timestep_fs': 1.0, 'thermo_every': 1000, 'stages': []},
    'analysis': {'r_max': 8.0, 'dr': 0.02, 'skip_frames': 0},
}


def _merge(base, new):
    """Recursive dict merge: values in `new` override `base`."""
    out = copy.deepcopy(base)
    for key, val in (new or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def _eval_expr(expr, x):
    """Evaluate an expression such as '1 - x' for the series variable x.
    Only arithmetic on x is allowed (no builtins)."""
    if isinstance(expr, (int, float)):
        return float(expr)
    return float(eval(str(expr), {'__builtins__': {}}, {'x': x}))


def _density_for(spec, x):
    """Starting density: a number, or [[x, rho], ...] points to interpolate."""
    if isinstance(spec, (int, float)):
        return float(spec)
    pts = np.array(spec, dtype=float)
    order = np.argsort(pts[:, 0])
    return float(np.interp(x, pts[order, 0], pts[order, 1]))


def expand_series(cfg):
    """Return a list of composition dicts: {name, x, oxides, density}."""
    comp = cfg['compositions']
    out = []
    if 'series' in comp:
        ser = comp['series']
        for x in ser['values']:
            oxides = {ox: _eval_expr(e, x) for ox, e in ser['oxides'].items()}
            out.append({
                'name': ser.get('name', 'x{x:.3f}').format(x=x),
                'x': float(x),
                'oxides': oxides,
                'density': _density_for(ser['density'], x),
            })
    for item in comp.get('list', []):
        out.append({
            'name': item['name'],
            'x': float(item.get('x', np.nan)),
            'oxides': {k: float(v) for k, v in item['oxides'].items()},
            'density': float(item['density']),
        })
    if not out:
        raise ValueError("No compositions defined: use 'series' and/or 'list'.")
    names = [c['name'] for c in out]
    if len(set(names)) != len(names):
        raise ValueError(f'Composition names must be unique: {names}')
    return out


def load_config(path):
    """Read the master input, fill defaults, load the potential file and
    build the list of runs."""
    with open(path) as fh:
        user = yaml.safe_load(fh)
    cfg = _merge(DEFAULTS, user)
    base = os.path.dirname(os.path.abspath(path))
    cfg['_base_dir'] = base

    pot_path = os.path.join(base, cfg['potential'])
    with open(pot_path) as fh:
        cfg['_potential'] = yaml.safe_load(fh)

    out_root = os.path.normpath(os.path.join(base, cfg['project']['output_dir']))
    runs = []
    nrep = int(cfg['system']['replicas'])
    for comp in expand_series(cfg):
        for rep in range(nrep):
            tag = comp['name'] if nrep == 1 else f"{comp['name']}_r{rep + 1}"
            runs.append({
                **comp,
                'tag': tag,
                'replica': rep + 1,
                'seed': int(cfg['project']['seed']) + 1000 * len(runs) + 7,
                'dir': os.path.join(out_root, tag),
            })
    cfg['_runs'] = runs
    cfg['_out_root'] = out_root
    return cfg
