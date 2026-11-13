"""Step 5 -- analyse the production trajectory of one run.

Pass 1 (all pairs up to r_max): partial RDFs, density, NMR second moments,
                                positions for the MSD.
Pass 2 (bonded pairs only):     coordination numbers, Qn, oxygen speciation,
                                bond-angle distributions.
Results go to <run>/analysis/ as CSV files plus one summary.json.
"""

import itertools
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from .builder import AVOGADRO
from .elements import MASS, NEUTRON_B

MU0_4PI = 1e-7            # T m / A
HBAR = 1.054571817e-34    # J s


# --------------------------------------------------------------------- I/O
def read_dump(path):
    """Yield (timestep, lo, L, ids, types, xyz) for each frame of a LAMMPS
    custom dump with columns id type xu yu zu (orthogonal box)."""
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith('ITEM: TIMESTEP'):
                continue
            step = int(fh.readline())
            fh.readline()
            n = int(fh.readline())
            fh.readline()
            b = np.array([fh.readline().split()[:2] for _ in range(3)], float)
            cols = fh.readline().split()[2:]
            data = np.loadtxt(fh, max_rows=n, ndmin=2)
            c = {name: k for k, name in enumerate(cols)}
            xyz = data[:, [c['xu'], c['yu'], c['zu']]]
            yield step, b[:, 0], b[:, 1] - b[:, 0], data[:, c['id']].astype(int), \
                data[:, c['type']].astype(int), xyz


def pair_key(a, b):
    return f'{a}-{b}'


def load_isotopes(path):
    """Read dypole's IsotopeProperties table -> {'7Li': (gamma, I, abundance)}."""
    iso = {}
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            f = line.rstrip('\n').split('\t')
            try:
                gamma = float(f[6]) * 1e7
                spin = float(f[4])
                ab = float(f[5]) / 100 if f[5] not in ('-', '') else 0.0
            except (ValueError, IndexError):
                continue
            iso[f'{f[3]}{f[1]}'] = (gamma, spin, ab)
    return iso


def element_of(isotope):
    return isotope.lstrip('0123456789')


# --------------------------------------------------------------- helpers
def periodic_pairs(xyz, lo, L, rcut):
    """All pairs i<j closer than rcut, with minimum-image vectors j - i."""
    w = np.mod(xyz - lo, L)
    w[w >= L] = 0.0
    tree = cKDTree(w, boxsize=L)
    ij = tree.query_pairs(rcut, output_type='ndarray')
    d = w[ij[:, 1]] - w[ij[:, 0]]
    d -= L * np.round(d / L)
    return ij, d, np.linalg.norm(d, axis=1)


def auto_cutoff(r, g, smooth=5, window=1.5, tol=0.1):
    """End of the first coordination shell.

    Finds the first peak (g > 1.2) and the minimum of g within `window` A
    after it, then returns the middle of the region around that minimum where
    g stays within `tol` of it. For a normal minimum (e.g. Li-O) this is the
    minimum itself; for strong bonds (e.g. Si-O) g(r) is ~0 over a wide range
    and the middle of that gap is far more stable than the noisy exact minimum."""
    gs = np.convolve(g, np.ones(smooth) / smooth, mode='same')
    peaks = [i for i in range(1, len(gs) - 1)
             if gs[i] > 1.2 and gs[i] >= gs[i - 1] and gs[i] >= gs[i + 1]]
    if not peaks:
        return None
    p = peaks[0]
    win = (r > r[p]) & (r < r[p] + window)
    if not win.any():
        return None
    iw = np.where(win)[0]
    imin = iw[np.argmin(gs[iw])]
    lo = hi = imin
    while lo - 1 > p and gs[lo - 1] <= gs[imin] + tol:
        lo -= 1
    while hi + 1 <= iw[-1] and gs[hi + 1] <= gs[imin] + tol:
        hi += 1
    return float(r[(lo + hi) // 2])


def lookup(d, a, b):
    return d.get(pair_key(a, b), d.get(pair_key(b, a)))


# ------------------------------------------------------------ main driver
def analyse_run(run_dir, cfg):
    ana = cfg['analysis']
    with open(os.path.join(run_dir, 'meta.json')) as fh:
        meta = json.load(fh)
    els = meta['elements']
    nel = len(els)
    out = os.path.join(run_dir, 'analysis')
    os.makedirs(out, exist_ok=True)
    traj = os.path.join(run_dir, 'traj.lammpstrj')
    dt_ps = cfg['simulation']['timestep_fs'] / 1000.0
    skip = int(ana.get('skip_frames', 0))

    rmax, dr = float(ana['r_max']), float(ana['dr'])
    nb = int(round(rmax / dr))
    edges = np.linspace(0, rmax, nb + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    upairs = list(itertools.combinations_with_replacement(range(nel), 2))
    hist = {p: np.zeros(nb) for p in upairs}

    m2_cfg = ana.get('nmr_m2', [])
    if m2_cfg:
        iso_path = os.path.join(cfg['_base_dir'], ana.get('isotope_table', '../IsotopeProperties'))
        isotopes = load_isotopes(iso_path)
    m2_sum = {k: 0.0 for k in range(len(m2_cfg))}

    # ---------------- pass 1
    nframes, vol_sum, mass_tot = 0, 0.0, None
    steps, positions, boxes, types = [], [], [], None
    for k, (step, lo, L, ids, typ, xyz) in enumerate(read_dump(traj)):
        if k < skip:
            continue
        t = typ - 1
        if types is None:
            types = t
            counts = np.bincount(t, minlength=nel)
            mass_tot = sum(MASS[e] * counts[i] for i, e in enumerate(els))
            if rmax > 0.5 * L.min():
                raise ValueError(f'r_max={rmax} exceeds half the box ({0.5 * L.min():.2f} A).')
        V = np.prod(L)
        ij, d, dist = periodic_pairs(xyz, lo, L, rmax)
        ti, tj = t[ij[:, 0]], t[ij[:, 1]]
        a, b = np.minimum(ti, tj), np.maximum(ti, tj)
        for (p, q) in upairs:
            m = (a == p) & (b == q)
            hist[(p, q)] += np.histogram(dist[m], bins=edges)[0] * V
        # NMR second moments: sum of r^-6 from each observed atom to partners
        for n_, m2 in enumerate(m2_cfg):
            eo = els.index(element_of(m2['observe']))
            ep = els.index(element_of(m2['partner']))
            sel = ((ti == eo) & (tj == ep)) | ((ti == ep) & (tj == eo))
            s6 = np.sum(dist[sel] ** -6.0) * (2 if eo == ep else 1)
            rho_p = counts[ep] / V
            tail = 4 * np.pi * rho_p / (3 * rmax ** 3)
            m2_sum[n_] += s6 / counts[eo] + tail
        steps.append(step)
        positions.append(xyz.copy())
        boxes.append((lo, L))
        vol_sum += V
        nframes += 1
    if nframes == 0:
        raise RuntimeError(f'No frames found in {traj}')
    Vm = vol_sum / nframes
    rho_num = counts / Vm   # atoms / A^3

    # partial g(r): hist was multiplied by V per frame, so divide by frames
    gr, cn_run = {}, {}
    for (p, q) in upairs:
        norm = counts[p] * counts[q] * shell * nframes
        h = hist[(p, q)] * (2.0 if p == q else 1.0)
        gr[(p, q)] = h / norm
    for p, q in itertools.product(range(nel), repeat=2):
        g = gr[(min(p, q), max(p, q))]
        cn_run[(p, q)] = np.cumsum(4 * np.pi * rho_num[q] * g * r ** 2 * dr)
    _write_table(os.path.join(out, 'rdf_partial.csv'), r,
                 {pair_key(els[p], els[q]): gr[(p, q)] for p, q in upairs}, 'r_A')
    _write_table(os.path.join(out, 'cn_running.csv'), r,
                 {pair_key(els[p], els[q]): cn_run[(p, q)]
                  for p, q in itertools.product(range(nel), repeat=2)}, 'r_A')

    summary = {'tag': meta['tag'], 'name': meta['name'], 'x': meta['x'],
               'replica': meta['replica'], 'n_frames': nframes,
               'real_mol_percent': meta['real_mol_percent'],
               'density_g_cm3': mass_tot / AVOGADRO / (Vm * 1e-24)}

    # --------------- cutoffs (user value or automatic first minimum)
    cations = [e for e in els if e != 'O']
    cn_pairs = ana.get('coordination', {}).get('pairs') or [f'{c}-O' for c in cations]
    user_cut = ana.get('coordination', {}).get('cutoffs', {}) or {}
    cutoffs = {}
    needed = set(cn_pairs)
    for ang in ana.get('bond_angles', []):
        e1, c, e2 = ang.split('-')
        needed |= {pair_key(c, e1), pair_key(c, e2)}
    for f in ana.get('qn', {}).get('formers', []):
        needed.add(pair_key(f, 'O'))
    for key in sorted(needed):
        a_, b_ = key.split('-')
        if a_ not in els or b_ not in els:
            continue
        cut = lookup(user_cut, a_, b_)
        if cut is None:
            i_, j_ = sorted((els.index(a_), els.index(b_)))
            cut = auto_cutoff(r, gr[(i_, j_)])
        cutoffs[key] = cut
    summary['cutoffs_A'] = cutoffs

    # first-peak positions (bond lengths)
    summary['first_peak_A'] = {}
    for p, q in upairs:
        key = pair_key(els[p], els[q])
        c = lookup(cutoffs, els[p], els[q]) or 3.5
        m = r < c
        if m.any() and gr[(p, q)][m].max() > 0:
            summary['first_peak_A'][key] = float(r[m][np.argmax(gr[(p, q)][m])])

    # ---------------- pass 2: bonded analysis on stored frames
    summary.update(_bonded_analysis(out, positions, boxes, types, els, cutoffs, cn_pairs, ana))

    # ---------------- neutron S(q)
    summary.update(_structure_factor(out, r, gr, upairs, els, counts, rho_num.sum(), ana))

    # ---------------- MSD / diffusion
    if ana.get('msd', True) and nframes > 4:
        summary.update(_msd(out, np.array(positions), types, els,
                            (np.array(steps) - steps[0]) * dt_ps))

    # ---------------- NMR second moments
    if m2_cfg:
        summary['M2_1e6_rad2_s2'] = {}
        for n_, m2 in enumerate(m2_cfg):
            gI, I, abI = isotopes[m2['observe']]
            gS, S, abS = isotopes[m2['partner']]
            s6 = m2_sum[n_] / nframes * 1e60      # A^-6 -> m^-6
            if m2['observe'] == m2['partner']:
                val = 0.6 * MU0_4PI ** 2 * gI ** 4 * HBAR ** 2 * I * (I + 1) * s6 * abI
            else:
                val = 4.0 / 15.0 * MU0_4PI ** 2 * gI ** 2 * gS ** 2 * HBAR ** 2 \
                    * S * (S + 1) * s6 * abS
            summary['M2_1e6_rad2_s2'][f"{m2['observe']}{{{m2['partner']}}}"] = val / 1e6

    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _bonded_analysis(out, positions, boxes, types, els, cutoffs, cn_pairs, ana):
    """Coordination numbers, Qn and bond angles from bonded neighbour lists."""
    nel = len(els)
    res = {}
    rc = max([c for c in cutoffs.values() if c] + [0.1])
    cn_hist = {k: np.zeros(17) for k in cn_pairs if all(e in els for e in k.split('-'))}
    formers = [f for f in ana.get('qn', {}).get('formers', []) if f in els]
    qn_hist = {f: np.zeros(7) for f in formers}
    ospec = np.zeros(5)
    angles = ana.get('bond_angles', [])
    abins = np.linspace(0, 180, 181)
    ahist = {a: np.zeros(180) for a in angles}

    cut_mat = np.zeros((nel, nel))
    for key, c in cutoffs.items():
        a, b = key.split('-')
        if c and a in els and b in els:
            cut_mat[els.index(a), els.index(b)] = cut_mat[els.index(b), els.index(a)] = c

    for xyz, (lo, L) in zip(positions, boxes):
        ij, d, dist = periodic_pairs(xyz, lo, L, rc)
        ti, tj = types[ij[:, 0]], types[ij[:, 1]]
        bonded = dist < cut_mat[ti, tj]
        ij, d, ti, tj = ij[bonded], d[bonded], ti[bonded], tj[bonded]
        # symmetric (directed) neighbour list
        src = np.concatenate([ij[:, 0], ij[:, 1]])
        dst = np.concatenate([ij[:, 1], ij[:, 0]])
        vec = np.concatenate([d, -d])
        ts, td = types[src], types[dst]
        N = len(types)
        for key in cn_hist:
            a, b = (els.index(e) for e in key.split('-'))
            m = (ts == a) & (td == b)
            cn = np.bincount(src[m], minlength=N)[types == a]
            cn_hist[key] += np.bincount(np.clip(cn, 0, 16), minlength=17)
        if formers:
            o = els.index('O')
            fidx = [els.index(f) for f in formers]
            m = (ts == o) & np.isin(td, fidx)
            nf = np.bincount(src[m], minlength=N)        # formers around each O
            ospec += np.bincount(np.clip(nf[types == o], 0, 4), minlength=5)
            is_bo = nf >= 2
            for f, fi in zip(formers, fidx):
                mm = (ts == fi) & (td == o) & is_bo[dst]
                nbo = np.bincount(src[mm], minlength=N)[types == fi]
                qn_hist[f] += np.bincount(np.clip(nbo, 0, 6), minlength=7)
        if angles:
            # neighbours of each atom as contiguous slices of a sorted list
            order = np.argsort(src, kind='stable')
            s_, v_, td_ = src[order], vec[order], td[order]
            starts = np.searchsorted(s_, np.arange(N))
            ends = np.searchsorted(s_, np.arange(N), side='right')
        for ang in angles:
            e1, c, e2 = (els.index(e) if e in els else -1 for e in ang.split('-'))
            if min(e1, c, e2) < 0:
                continue
            for atom in np.where(types == c)[0]:
                tv = td_[starts[atom]:ends[atom]]
                vv = v_[starts[atom]:ends[atom]]
                for i_, j_ in itertools.combinations(range(len(tv)), 2):
                    if {tv[i_], tv[j_]} == {e1, e2}:
                        cosang = vv[i_] @ vv[j_] / np.linalg.norm(vv[i_]) / np.linalg.norm(vv[j_])
                        deg = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                        ahist[ang][min(179, int(deg))] += 1

    # write CN distributions
    res['CN_mean'], res['CN_distribution'] = {}, {}
    with open(os.path.join(out, 'cn_distribution.csv'), 'w') as fh:
        fh.write('pair,' + ','.join(f'n{k}' for k in range(17)) + ',mean\n')
        for key, h in cn_hist.items():
            f = h / h.sum() if h.sum() else h
            mean = float(np.sum(np.arange(17) * f))
            res['CN_mean'][key] = mean
            res['CN_distribution'][key] = {int(k): float(v) for k, v in enumerate(f) if v > 0}
            fh.write(key + ',' + ','.join(f'{v:.5f}' for v in f) + f',{mean:.4f}\n')
    if formers:
        res['Qn'] = {}
        for f, h in qn_hist.items():
            frac = h / h.sum() if h.sum() else h
            res['Qn'][f] = {f'Q{k}': float(v) for k, v in enumerate(frac) if v > 0}
        fo = ospec / ospec.sum()
        res['O_speciation'] = {'free_O': float(fo[0]), 'NBO': float(fo[1]),
                               'BO': float(fo[2]), 'tricluster': float(fo[3] + fo[4])}
    if angles:
        centers = 0.5 * (abins[1:] + abins[:-1])
        dist = {a: (h / h.sum() if h.sum() else h) for a, h in ahist.items()}
        _write_table(os.path.join(out, 'bond_angles.csv'), centers, dist, 'angle_deg')
        res['angle_mean_deg'] = {a: float(np.sum(centers * p)) for a, p in dist.items() if p.sum()}
    return res


def _structure_factor(out, r, gr, upairs, els, counts, rho, ana):
    """Faber-Ziman partial and neutron-weighted total S(q), Lorch-damped."""
    q = np.linspace(0.3, float(ana.get('q_max', 20.0)), 600)
    dr = r[1] - r[0]
    rmax = r[-1] + dr / 2
    lorch = np.sinc(r / rmax)                 # np.sinc(x) = sin(pi x)/(pi x)
    c = counts / counts.sum()
    b = np.array([NEUTRON_B.get(e, 0.0) for e in els])
    kernel = np.sin(np.outer(q, r)) / np.outer(q, r)
    sq_part, total, gn = {}, np.zeros_like(q), np.zeros_like(r)
    for p, s in upairs:
        sij = 1 + 4 * np.pi * rho * (kernel @ (r ** 2 * (gr[(p, s)] - 1) * lorch)) * dr
        sq_part[pair_key(els[p], els[s])] = sij
        w = c[p] * c[s] * b[p] * b[s] * (1 if p == s else 2)
        total += w * sij
        gn += w * gr[(p, s)]
    norm = (c @ b) ** 2
    _write_table(os.path.join(out, 'sq_neutron.csv'), q,
                 {'S_total': total / norm, **sq_part}, 'q_invA')
    _write_table(os.path.join(out, 'gr_neutron_total.csv'), r, {'g_total': gn / norm}, 'r_A')
    return {}


def _msd(out, pos, types, els, t):
    """MSD with multiple time origins (COM drift removed) and D from a linear
    fit to the second half of the curve. Units: A^2, ps, cm^2/s."""
    pos = pos - pos.mean(axis=1, keepdims=True)
    nf = len(pos)
    lags = np.arange(1, nf // 2 + 1)
    msd = {e: np.zeros(len(lags)) for e in els}
    for k, lag in enumerate(lags):
        d2 = np.sum((pos[lag:] - pos[:-lag]) ** 2, axis=2).mean(axis=0)
        for i, e in enumerate(els):
            msd[e][k] = d2[types == i].mean()
    tl = t[lags]
    _write_table(os.path.join(out, 'msd.csv'), tl, msd, 't_ps')
    half = len(tl) // 2
    D = {}
    for e in els:
        slope = np.polyfit(tl[half:], msd[e][half:], 1)[0] if len(tl) - half >= 2 else np.nan
        D[e] = float(slope / 6.0 * 1e-4)
    return {'D_cm2_s': D}


def _write_table(path, x, cols, xname):
    keys = list(cols)
    with open(path, 'w') as fh:
        fh.write(','.join([xname] + keys) + '\n')
        for k in range(len(x)):
            fh.write(','.join([f'{x[k]:.5g}'] + [f'{cols[c][k]:.6g}' for c in keys]) + '\n')
