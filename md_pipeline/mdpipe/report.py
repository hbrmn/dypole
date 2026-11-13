"""Step 6 -- collect the per-run results into series tables and plots.

Writes to <output_dir>/report/:
  summary.csv          one row per run, every scalar result as a column
  summary_mean.csv     replicas averaged (mean and std) per composition
  *.png                quantities vs composition, stacked g(r) and S(q)
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + ':'))
        else:
            out[key] = v
    return out


def _stack_plot(runs, fname, xcol, ylabel, path, offset):
    tables = [(r['tag'], pd.read_csv(os.path.join(r['dir'], 'analysis', fname)))
              for r in runs if os.path.exists(os.path.join(r['dir'], 'analysis', fname))]
    if not tables:
        return
    cols = [c for c in tables[0][1].columns if c != xcol]
    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        for k, (tag, df) in enumerate(tables):
            if col in df:
                ax.plot(df[xcol], df[col] + k * offset, lw=1, label=tag)
        ax.set_title(col)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ylabel)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_report(cfg):
    rep = os.path.join(cfg['_out_root'], 'report')
    os.makedirs(rep, exist_ok=True)
    rows, done = [], []
    for run in cfg['_runs']:
        f = os.path.join(run['dir'], 'analysis', 'summary.json')
        if not os.path.exists(f):
            print(f"[report] {run['tag']}: no analysis yet, skipped")
            continue
        with open(f) as fh:
            rows.append(_flatten(json.load(fh)))
        done.append(run)
    if not rows:
        print('[report] nothing to report')
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rep, 'summary.csv'), index=False)
    num = df.select_dtypes('number').drop(columns=['replica'], errors='ignore')
    grouped = num.groupby(df['name'], sort=False)
    mean = grouped.mean().add_suffix('')
    std = grouped.std().add_suffix(':std')
    summ = pd.concat([mean, std], axis=1)
    summ.to_csv(os.path.join(rep, 'summary_mean.csv'))

    # quantities vs composition
    xs = mean['x'] if mean['x'].notna().all() else pd.Series(np.arange(len(mean)), index=mean.index)
    xlabel = cfg['compositions'].get('series', {}).get('label', 'x') \
        if mean['x'].notna().all() else 'composition index'
    groups = {'density': ['density_g_cm3'], 'CN': 'CN_mean:', 'Qn': 'Qn:',
              'O_speciation': 'O_speciation:', 'bond_length': 'first_peak_A:',
              'bond_angle': 'angle_mean_deg:', 'diffusion': 'D_cm2_s:',
              'NMR_M2': 'M2_1e6_rad2_s2:'}
    for name, sel in groups.items():
        cols = sel if isinstance(sel, list) else [c for c in mean.columns if c.startswith(sel)]
        cols = [c for c in cols if c in mean.columns and mean[c].notna().any()]
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for c in cols:
            err = summ.get(c + ':std')
            ax.errorbar(xs, mean[c], yerr=err, marker='o', capsize=3,
                        label=c.split(':', 1)[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(name)
        if name == 'diffusion':
            ax.set_yscale('log')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(rep, f'{name}_vs_composition.png'), dpi=150)
        plt.close(fig)

    first = {}
    for run in done:      # one replica per composition for the stacked plots
        first.setdefault(run['name'], run)
    reps = list(first.values())
    _stack_plot(reps, 'rdf_partial.csv', 'r_A', 'g(r) + offset', os.path.join(rep, 'rdf_partial.png'), 2.0)
    _stack_plot(reps, 'sq_neutron.csv', 'q_invA', 'S(q) + offset', os.path.join(rep, 'sq.png'), 0.5)
    _stack_plot(reps, 'bond_angles.csv', 'angle_deg', 'P(angle)', os.path.join(rep, 'bond_angles.png'), 0.0)
    print(f'[report] wrote {rep}')
