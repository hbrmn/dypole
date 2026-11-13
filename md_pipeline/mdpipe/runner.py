"""Step 4 -- run LAMMPS for every composition.

run_mode 'local' : run the jobs one after another on this machine.
run_mode 'slurm' : write a job script per run (and submit if submit: true).
run_mode 'none'  : only prepare the folders (e.g. to copy them to a cluster).

A run counts as finished when its log contains PIPELINE_FINISHED, so the
pipeline can be restarted at any time and will skip finished runs.
"""

import os
import subprocess


def is_finished(run_dir):
    log = os.path.join(run_dir, 'log.lammps')
    if not os.path.exists(log):
        return False
    with open(log, errors='ignore') as fh:
        return 'PIPELINE_FINISHED' in fh.read()


def lammps_command(cfg):
    lmp = cfg['lammps']
    n = int(lmp['mpi_ranks'])
    cmd = lmp['command']
    if n > 1:
        cmd = lmp['mpi_launcher'].format(n=n) + ' ' + cmd
    return cmd + ' -in in.lammps -log log.lammps'


def write_slurm(run, cfg):
    s = cfg['lammps'].get('slurm', {})
    lines = ['#!/bin/bash',
             f"#SBATCH --job-name={run['tag']}",
             f"#SBATCH --ntasks={cfg['lammps']['mpi_ranks']}",
             f"#SBATCH --time={s.get('time', '24:00:00')}"]
    for key in ('partition', 'account', 'mem'):
        if key in s:
            lines.append(f'#SBATCH --{key}={s[key]}')
    lines += s.get('extra_lines', [])
    launcher = s.get('launcher', 'srun')
    lines += [f'cd {run["dir"]}',
              f"{launcher} {cfg['lammps']['command']} -in in.lammps -log log.lammps"]
    path = os.path.join(run['dir'], 'job.slurm')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def run_all(cfg, force=False):
    mode = cfg['lammps']['run_mode']
    for run in cfg['_runs']:
        if is_finished(run['dir']) and not force:
            print(f"[run] {run['tag']}: already finished, skipping")
            continue
        if mode == 'local':
            print(f"[run] {run['tag']}: {lammps_command(cfg)}")
            with open(os.path.join(run['dir'], 'stdout.txt'), 'w') as out:
                res = subprocess.run(lammps_command(cfg), shell=True, cwd=run['dir'],
                                     stdout=out, stderr=subprocess.STDOUT)
            status = 'ok' if res.returncode == 0 and is_finished(run['dir']) else 'FAILED'
            print(f"[run] {run['tag']}: {status} (see {run['dir']}/log.lammps)")
        elif mode == 'slurm':
            job = write_slurm(run, cfg)
            if cfg['lammps'].get('slurm', {}).get('submit', False):
                subprocess.run(['sbatch', job], cwd=run['dir'], check=True)
                print(f"[run] {run['tag']}: submitted")
            else:
                print(f"[run] {run['tag']}: wrote {job}  (submit with: sbatch job.slurm)")
        elif mode == 'none':
            print(f"[run] {run['tag']}: prepared only")
        else:
            raise ValueError(f'Unknown run_mode {mode}')
