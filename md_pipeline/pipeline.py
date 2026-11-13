#!/usr/bin/env python3
"""Automated MD pipeline for compositional series (LAMMPS).

Usage
-----
    python pipeline.py master_input.yaml all        # everything
    python pipeline.py master_input.yaml prepare    # structures + LAMMPS inputs
    python pipeline.py master_input.yaml run        # run / submit LAMMPS
    python pipeline.py master_input.yaml analyze    # analyse finished runs
    python pipeline.py master_input.yaml report     # series tables and plots
    python pipeline.py master_input.yaml status     # what is done?

Every step can be repeated: finished runs are skipped unless --force is given.
"""

import argparse
import os
import shutil

from mdpipe.analysis import analyse_run
from mdpipe.builder import build
from mdpipe.config import load_config
from mdpipe.lammps_input import write_input
from mdpipe.report import make_report
from mdpipe.runner import is_finished, run_all


def prepare(cfg, force=False):
    for run in cfg['_runs']:
        if os.path.exists(os.path.join(run['dir'], 'in.lammps')) and not force:
            print(f"[prepare] {run['tag']}: exists, skipping (use --force to rebuild)")
            continue
        meta = build(run, cfg)
        write_input(run['dir'], meta, cfg)
        mol = ', '.join(f'{k} {v:.2f}' for k, v in meta['real_mol_percent'].items())
        print(f"[prepare] {run['tag']}: {sum(meta['counts'].values())} atoms, "
              f"L = {meta['box_length']:.2f} A, mol%: {mol}")
    # keep a copy of the inputs next to the results (provenance)
    shutil.copy(cfg['_config_path'], os.path.join(cfg['_out_root'], 'master_input_used.yaml'))


def analyze(cfg, force=False):
    for run in cfg['_runs']:
        done = os.path.join(run['dir'], 'analysis', 'summary.json')
        if not is_finished(run['dir']):
            print(f"[analyze] {run['tag']}: simulation not finished, skipped")
            continue
        if os.path.exists(done) and not force:
            print(f"[analyze] {run['tag']}: already analysed, skipping")
            continue
        s = analyse_run(run['dir'], cfg)
        print(f"[analyze] {run['tag']}: density {s['density_g_cm3']:.3f} g/cm3, "
              f"CN {', '.join(f'{k} {v:.2f}' for k, v in s.get('CN_mean', {}).items())}")


def status(cfg):
    for run in cfg['_runs']:
        prep = os.path.exists(os.path.join(run['dir'], 'in.lammps'))
        fin = is_finished(run['dir'])
        ana = os.path.exists(os.path.join(run['dir'], 'analysis', 'summary.json'))
        print(f"{run['tag']:<20} prepared={prep!s:<6} simulated={fin!s:<6} analysed={ana}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('step', choices=['prepare', 'run', 'analyze', 'report', 'status', 'all'])
    ap.add_argument('--force', action='store_true', help='redo steps that are already done')
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['_config_path'] = args.config
    os.makedirs(cfg['_out_root'], exist_ok=True)

    if args.step in ('prepare', 'all'):
        prepare(cfg, args.force)
    if args.step in ('run', 'all'):
        run_all(cfg, args.force)
    if args.step in ('analyze', 'all'):
        analyze(cfg, args.force)
    if args.step in ('report', 'all'):
        make_report(cfg)
    if args.step == 'status':
        status(cfg)


if __name__ == '__main__':
    main()
