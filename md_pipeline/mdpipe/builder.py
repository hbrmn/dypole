"""Step 2 -- turn a composition into a starting structure.

oxide mol fractions -> number of atoms of each element -> cubic box from
density -> random, non-overlapping atom positions -> LAMMPS data file.
"""

import json
import os
import re

import numpy as np

from .elements import MASS

AVOGADRO = 6.02214076e23


def parse_formula(formula):
    """'Li2O' -> {'Li': 2, 'O': 1}"""
    out = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def atom_counts(oxides, target_atoms, charges):
    """Integer number of formula units per oxide so that the total number
    of atoms is close to `target_atoms`. Because whole oxide units are used,
    the box is always charge neutral."""
    total = sum(oxides.values())
    frac = {ox: f / total for ox, f in oxides.items() if f > 0}
    parsed = {ox: parse_formula(ox) for ox in frac}
    for ox, els in parsed.items():
        q = sum(n * charges[el] for el, n in els.items())
        if abs(q) > 1e-6:
            raise ValueError(f'{ox} is not neutral with the potential charges (q={q}).')
    atoms_per_unit = sum(frac[ox] * sum(parsed[ox].values()) for ox in frac)
    scale = target_atoms / atoms_per_unit
    units = {ox: max(1, int(round(frac[ox] * scale))) for ox in frac}
    counts = {}
    for ox, nu in units.items():
        for el, n in parsed[ox].items():
            counts[el] = counts.get(el, 0) + n * nu
    real = {ox: 100 * nu / sum(units.values()) for ox, nu in units.items()}
    return counts, units, real


def box_length(counts, density):
    """Cubic box edge (Angstrom) for mass density in g/cm^3."""
    mass_g = sum(MASS[el] * n for el, n in counts.items()) / AVOGADRO
    return (mass_g / density * 1e24) ** (1.0 / 3.0)


def random_positions(n, L, dmin, rng, max_tries=2000):
    """Random sequential insertion with a minimum distance, using a cell
    list so it stays fast for a few 10^4 atoms."""
    ncell = max(1, int(L // dmin))
    csize = L / ncell
    cells = {}
    pos = np.empty((n, 3))
    for i in range(n):
        for _ in range(max_tries):
            p = rng.random(3) * L
            c = (p // csize).astype(int) % ncell
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = ((c[0] + dx) % ncell, (c[1] + dy) % ncell, (c[2] + dz) % ncell)
                        for j in cells.get(key, ()):
                            d = pos[j] - p
                            d -= L * np.round(d / L)
                            if d @ d < dmin * dmin:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                pos[i] = p
                cells.setdefault(tuple(c), []).append(i)
                break
        else:
            raise RuntimeError('Could not place atoms: lower min_distance or density.')
    return pos


def element_order(counts):
    """Fixed type order: cations alphabetically, oxygen last."""
    return sorted([e for e in counts if e != 'O']) + (['O'] if 'O' in counts else [])


def write_data_file(path, counts, L, charges, rng, dmin):
    elements = element_order(counts)
    species = np.concatenate([[t + 1] * counts[el] for t, el in enumerate(elements)])
    rng.shuffle(species)
    pos = random_positions(len(species), L, dmin, rng)
    with open(path, 'w') as fh:
        fh.write('Random start structure written by mdpipe\n\n')
        fh.write(f'{len(species)} atoms\n{len(elements)} atom types\n\n')
        for ax in 'xyz':
            fh.write(f'0.0 {L:.6f} {ax}lo {ax}hi\n')
        fh.write('\nMasses\n\n')
        for t, el in enumerate(elements):
            fh.write(f'{t + 1} {MASS[el]}  # {el}\n')
        fh.write('\nAtoms # charge\n\n')
        for i, (t, p) in enumerate(zip(species, pos)):
            q = charges[elements[t - 1]]
            fh.write(f'{i + 1} {t} {q:.4f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    return elements


def build(run, cfg):
    """Create run directory, data file and meta.json for one run."""
    os.makedirs(run['dir'], exist_ok=True)
    charges = cfg['_potential']['charges']
    counts, units, real = atom_counts(run['oxides'], cfg['system']['target_atoms'], charges)
    L = box_length(counts, run['density'])
    rng = np.random.default_rng(run['seed'])
    elements = write_data_file(os.path.join(run['dir'], 'structure.data'),
                               counts, L, charges, rng, cfg['system']['min_distance'])
    meta = {k: run[k] for k in ('name', 'tag', 'x', 'replica', 'seed', 'oxides', 'density')}
    meta.update({'elements': elements, 'counts': counts, 'oxide_units': units,
                 'real_mol_percent': real, 'box_length': L})
    with open(os.path.join(run['dir'], 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    return meta
