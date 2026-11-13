# mdpipe: an automated LAMMPS pipeline for compositional series

You put everything into **one master input file**: compositions, force field,
thermal history and the analyses you want. You get back **tables and plots
for the whole series**: partial RDFs, coordination numbers, Qn speciation,
bond-angle distributions, neutron S(q), density, diffusion coefficients and
NMR dipolar second moments.

This README is also a tutorial. It walks through how the pipeline is built,
one step at a time, so that you can change or extend it.

```
master_input.yaml
      │
      ▼
 ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
 │1 config  │──▶│2 builder │──▶│3 lammps_input│──▶│4 runner  │──▶│5 analysis │──▶│6 report  │
 │expand the│   │atoms, box│   │stages → in.  │   │local or  │   │RDF, CN,   │   │series CSV│
 │series    │   │data file │   │lammps        │   │SLURM     │   │Qn, S(q).. │   │+ plots   │
 └──────────┘   └──────────┘   └──────────────┘   └──────────┘   └───────────┘   └──────────┘
```

```
md_pipeline/
├── pipeline.py              # command-line driver (the only thing you run)
├── master_input.yaml        # production-style example: Li2O–SiO2 series
├── examples/quick_test.yaml # tiny version for testing (~3 min)
├── potentials/pedone.yaml   # force field definition
└── mdpipe/
    ├── config.py            # step 1
    ├── builder.py           # step 2
    ├── lammps_input.py      # step 3
    ├── runner.py            # step 4
    ├── analysis.py          # step 5
    ├── report.py            # step 6
    └── elements.py          # masses, neutron scattering lengths
```

---

## Step 0: Design principles (read this first)

These ideas matter more than any individual line of code:

1. **Single source of truth.** Everything that defines the study lives in
   `master_input.yaml`. Nothing is edited by hand in the run folders. A copy of
   the master input is stored next to the results (`master_input_used.yaml`),
   so every result can be traced back to its inputs.
2. **One folder per simulation.** `runs/<name>/` holds `structure.data`,
   `in.lammps`, `log.lammps`, `traj.lammpstrj`, `meta.json` and `analysis/`.
   This makes runs independent, so they can be sent to a cluster or rerun one at a time.
3. **Steps are separate and can be restarted.** `prepare → run → analyze → report`
   can each be called on their own. Finished work is detected and skipped:
   a run counts as finished when `log.lammps` contains `PIPELINE_FINISHED`.
   A crashed or interrupted series can be resumed by running the same command again.
4. **Simulation and analysis are decoupled.** The trajectory is the interface.
   You can change or add analyses later without rerunning a single MD step.
5. **Validate against an independent tool.** The production run also asks LAMMPS
   for its own RDF (`rdf_lammps.dat`). The Python analysis reproduces it
   (Si–O peak at 1.61 Å, Li–O coordination agrees to 0.001).

---

## Step 0b: Installation

```bash
pip install numpy scipy pandas matplotlib pyyaml
```

LAMMPS needs the **KSPACE** package (long-range Coulomb) and **EXTRA-PAIR**
(`pair_style pedone`, available since 2024). Options:

* On a cluster: `module load lammps` (check the version and packages).
* conda: `conda install -c conda-forge lammps`
* pip (fastest for testing): `pip install lammps mpich`. This also provides the `lmp` executable.

Check that the pair style is available:

```bash
lmp -h | grep -i pedone
```

---

## Step 1: The master input and expanding the series (`mdpipe/config.py`)

Open `master_input.yaml`. Its sections map one-to-one to the pipeline steps:

| section        | used by          | what it defines |
|----------------|------------------|-----------------|
| `project`      | all              | name, output folder, master random seed |
| `compositions` | config, builder  | the series |
| `system`       | builder          | number of atoms, replicas |
| `potential`    | lammps_input     | path to a force-field YAML |
| `lammps`       | runner           | executable, MPI ranks, local/SLURM |
| `simulation`   | lammps_input     | timestep and the list of thermal-history stages |
| `analysis`     | analysis, report | what to compute and how |

A compositional series is written as **oxide amounts as functions of `x`**:

```yaml
compositions:
  series:
    name: "LS_x{x:.2f}"
    values: [0.20, 0.25, 0.33, 0.40]
    oxides:
      Li2O: "x"
      SiO2: "1 - x"
    density: [[0.0, 2.20], [0.33, 2.35], [0.50, 2.33]]   # interpolated in x
```

`expand_series()` evaluates the expressions for every `x` and returns one
dictionary per composition. More complex series need no new code:

```yaml
    oxides:                     # constant 10% Al2O3, Li2O/SiO2 varied
      Li2O:  "x"
      Al2O3: "10"
      SiO2:  "90 - x"
```

You can also add arbitrary compositions under `list:`. `load_config()`
then multiplies the series by `replicas` and gives every run its own seed and folder.

> **Why replicas?** One quench of 3000 atoms is a single sample of a glass.
> Two or three independent quenches (different random starts and velocities)
> give you error bars. `report.py` averages them automatically.

---

## Step 2: From composition to starting structure (`mdpipe/builder.py`)

Four small physics problems:

1. **Parse formulas.** `parse_formula('Al2O3') → {'Al': 2, 'O': 3}`.
2. **Integer atom counts.** The number of *formula units* of each oxide is
   chosen so that the total is close to `target_atoms`. Because whole oxide
   units are used, **the box is always charge neutral**. With Pedone's partial
   charges, each oxide is neutral by itself, and the code checks this. Rounding
   changes the composition slightly, so the composition actually built is
   printed and stored in `meta.json` (`real_mol_percent`).
3. **Box size from density.**
   `L = (Σ nᵢ Mᵢ / N_A / ρ)^(1/3)`. This is only the *starting* density.
   NPT stages relax it, and the model's own density is a result.
4. **Random positions** with a minimum separation (`min_distance`, ~1.6 Å)
   to avoid overlapping atoms. This uses a cell list, so it is O(N).

The result is `structure.data` in LAMMPS `atom_style charge` format. The
atom types are ordered as cations alphabetically, then O last. The mapping is
written into the input-file header and into `meta.json`.

---

## Step 3: Force field and LAMMPS input (`potentials/*.yaml`, `mdpipe/lammps_input.py`)

### 3a. The force field as data

`potentials/pedone.yaml` holds the charges, cutoffs and per-pair parameters
of the Pedone potential:

U(r) = qᵢqⱼe²/r + D₀[(1 − e^(−a(r−r₀)))² − 1] + C₀/r¹²

It is a widely used, transferable potential for silicate, aluminosilicate and
phosphate glasses. The LAMMPS lines are templates:

```yaml
pair_style: "hybrid/overlay pedone {sr_cut} coul/long {coul_cut}"
global_coeffs: ["pair_coeff * * coul/long"]
pair_coeff: "pedone {0} {1} {2} {3}"
kspace_style: "pppm 1.0e-5"
```

**Swapping the potential:** copy the file, change the template lines and
parameters, and point `potential:` in the master input to it. For a Buckingham
potential (e.g. BKS or Teter) you would write
`pair_style: "buck/coul/long {sr_cut} {coul_cut}"` and
`pair_coeff: "{0} {1} {2}"`. Buckingham potentials need an extra short-range
repulsion against the "Buckingham catastrophe" at high temperature.

> ⚠️ Check every parameter against the original paper (Pedone et al.,
> *J. Phys. Chem. B* 110 (2006) 11780) before running production simulations.

### 3b. The thermal history as a list of stages

The classic melt-quench protocol is written in the master input as data:

```yaml
stages:
  - {type: minimize}                                   # relax the random start
  - {type: relax, T: 300, time_ps: 5}                  # nve/limit + Langevin: remove bad contacts safely
  - {type: ramp, T_start: 300, T_end: 4000, rate_K_per_ps: 100, ensemble: nvt}
  - {type: equilibrate, T: 4000, time_ps: 100, ensemble: nvt, name: melt}       # lose memory of the start
  - {type: ramp, T_start: 4000, T_end: 2000, rate_K_per_ps: 1, ensemble: nvt, name: quench_nvt}   # hot: fixed volume
  - {type: ramp, T_start: 2000, T_end: 300, rate_K_per_ps: 1, ensemble: npt, P_bar: 0, name: quench_npt}
  - {type: equilibrate, T: 300, time_ps: 100, ensemble: npt, P_bar: 0}          # density relaxes
  - {type: production, T: 300, time_ps: 100, ensemble: nvt, dump_every_ps: 1}   # the analysed run
```

Each `type` is a small Python function that returns LAMMPS commands
(`STAGES` dictionary in `lammps_input.py`). Look at the generated
`runs/<name>/in.lammps`; it is short and commented. Each stage ends with
`write_restart`, so a crashed run can be continued from the last stage.

The **production** stage additionally:
* dumps `id type xu yu zu`. Unwrapped coordinates are needed for the MSD; the
  analysis wraps them back into the box for the structural properties.
* runs LAMMPS' `compute rdf` + `fix ave/time` → `rdf_lammps.dat` (cross-check).
* writes temperature, pressure, energy and density per frame to `thermo_production.dat`.

**Physics choices:**

* **Melt temperature:** high enough that the liquid diffuses (check the MSD!),
  typically 3000–4500 K for silicates with Pedone.
* **Quench rate:** 1 K/ps is a common compromise. Faster quenches give a
  "hotter" glass: higher fictive temperature, more defects and broader Qn
  distributions. Make the rate a series parameter if your property is sensitive to it.
* **NPT below ~2000 K** lets the density emerge from the model. Compare it to
  experiment; this is your first validation. **Do not run NPT at P = 0 in the
  hot liquid:** with Pedone, an NS2 melt at 4000 K expands to 1.0–1.4 g/cm³
  (less than half the glass density) and can even cavitate. Keep the hot part
  in NVT at the experimental density.

**Which potential?** For alkali silicates, the common choices are:
* **Pedone (2006)** (`potentials/pedone.yaml`): Morse + r⁻¹² with partial charges
  and Ewald/PPPM. Transferable to Li, Na, K, Mg, Ca, Al, P… and a good default
  for structure and for comparisons across a series.
* **SHIK** (Sundararaman, Huang, Ispas, Kob, 2018–2020): Buckingham + r⁻²⁴
  with Wolf-truncated Coulomb (no Ewald, so faster). Fitted for alkali
  (alumino)silicates; very good densities and elastic properties across
  composition. The O charge depends on composition to keep the box neutral,
  which needs a small extension of `builder.py`.
* **Teter / Du & Cormack**: Buckingham with partial charges, widely used for
  sodium silicates. Needs a repulsive correction at short range for hot melts.
* **Machine-learned potentials** (DeePMD, ACE, MACE…): near-DFT accuracy but
  10–100× slower. Good for validating a classical potential on one composition.
* **Timestep:** 1 fs is safe for oxides with these potentials.

To add a new kind of stage (e.g. a pressure quench or a high-T diffusion
run), write one function `st_xyz(s, meta, cfg, dt)` returning a list of
lines and register it in `STAGES`.

---

## Step 4: Running (`mdpipe/runner.py`)

```yaml
lammps:
  command: lmp
  mpi_ranks: 4
  run_mode: local     # local | slurm | none
```

* `local` runs the compositions one after another (`mpirun -np 4 lmp -in in.lammps`).
* `slurm` writes `job.slurm` into each run folder. It also submits the jobs if
  `slurm: {submit: true}`. Partition, account, time and extra lines such as
  `module load ...` go into the `slurm` block.
* `none` only prepares the folders, e.g. to copy them to a cluster.

Before analysing, **look at the logs**. Did the temperature follow the
stages? Is the density converged at the end of the 300 K NPT stage? Is the
potential energy flat during production? `thermo_production.dat` makes the
last check easy.

Rough cost: the quick test (600 atoms, ~21 ps) takes ~75 s per composition on
one core. The default protocol (3000 atoms, ~4 ns, dominated by the 1 K/ps quench)
is roughly a day on one core per composition, or a few hours on 8–16 cores.
Plan your cluster time accordingly.

---

## Step 5: Analysis (`mdpipe/analysis.py`)

All results are averaged over the frames of the production trajectory and
written to `runs/<name>/analysis/`.

| output file | content |
|---|---|
| `rdf_partial.csv` | partial g_αβ(r) for all pairs |
| `cn_running.csv` | running coordination n_αβ(r) = 4πρ_β ∫ g_αβ r² dr |
| `cn_distribution.csv` | fraction of α atoms with n = 0, 1, 2… β neighbours, and the mean |
| `bond_angles.csv` | bond-angle distributions, e.g. O–Si–O, Si–O–Si |
| `sq_neutron.csv` | Faber–Ziman partial S_αβ(q) and the neutron-weighted total S(q) |
| `gr_neutron_total.csv` | neutron-weighted total g(r) (compare with diffraction) |
| `msd.csv` | mean-squared displacement per element |
| `summary.json` | all scalar results: density, cutoffs, CN, Qn, O speciation, D, M₂ … |

How each quantity is computed:

* **Neighbour search.** `scipy.spatial.cKDTree` with periodic `boxsize` gives
  all pairs within r_max in O(N log N). Distances use the minimum-image
  convention, so `r_max` must be smaller than half the box. The code checks this.
* **Partial RDF.** g_αβ(r) = V·h_αβ(r) / (N_α N_β · 4πr²dr). Pairs are counted
  once and doubled for α = β.
* **Coordination cutoffs.** If you do not give a cutoff in the master input,
  the code takes the end of the first shell: the first peak, then the middle of
  the minimum region after it. **For a series, check these values in
  `summary.csv`.** Consider fixing them in the master input
  (`coordination: cutoffs: {Li-O: 2.6}`) so that every composition uses the
  same definition.
* **Qn and oxygen speciation.** An O bonded to ≥ 2 network formers is bridging
  (BO). Q^n = fraction of formers with n BO. The fractions of free O, NBO, BO
  and triclusters are also reported. For aluminosilicates or phosphates, add
  Al or P to `qn: formers`.
* **S(q).** S_αβ(q) = 1 + 4πρ ∫ r²(g_αβ − 1) sin(qr)/(qr) · L(r) dr, where L is the Lorch
  window. The total uses coherent neutron lengths from `elements.py`. The
  low-q region is unreliable when r_max is small, so use big boxes for S(q).
* **MSD and D.** Multiple time origins, centre-of-mass drift removed, and
  D = slope/6 fitted to the second half of the curve. At 300 K over 100 ps you
  only see vibrations. For ion dynamics, add a production stage at higher T
  (several temperatures give an Arrhenius plot).
* **NMR dipolar second moments** (links to SED/REDOR in `dypole.py`). These are
  van Vleck sums over the structure, using γ, I and natural abundance from the
  repository's `IsotopeProperties` table:
  * homonuclear: M₂ = (3/5)(μ₀/4π)² γ⁴ħ² I(I+1) · a · ⟨Σ r⁻⁶⟩
  * heteronuclear (observe I, partner S): M₂ = (4/15)(μ₀/4π)² γ_I²γ_S²ħ² S(S+1) · a_S · ⟨Σ r⁻⁶⟩

  Sums run to r_max, plus an analytical tail correction 4πρ/(3r_max³). Output
  is in 10⁶ rad²/s². This is the classical van Vleck M₂. For SED on the central
  transition of a quadrupolar nucleus (e.g. ⁷Li), apply the scaling factor
  appropriate to your experiment before comparing.

---

## Step 6: The series report (`mdpipe/report.py`)

`runs/report/` contains:

* `summary.csv`: one row per run, with every scalar from `summary.json` as a
  column (e.g. `CN_mean:Li-O`, `Qn:Si:Q3`, `M2_1e6_rad2_s2:7Li{7Li}`). This is
  the table to load into Origin, Excel or pandas.
* `summary_mean.csv`: replicas averaged, `:std` columns for error bars.
* Plots against composition: `density_`, `CN_`, `Qn_`, `O_speciation_`,
  `bond_length_`, `bond_angle_`, `diffusion_`, `NMR_M2_vs_composition.png`.
* Stacked plots: `rdf_partial.png`, `sq.png`, `bond_angles.png`.

---

## Step 7: Using it

```bash
cd md_pipeline
# 1) smoke test: tiny boxes, short runs (~3 min), checks the whole chain
python pipeline.py examples/quick_test.yaml all
python pipeline.py examples/quick_test.yaml status
```

```bash
# 1b) a small real example: Na2O.2SiO2 glass, 450 atoms, ~4 min on 4 cores
python pipeline.py examples/sodium_silicate_quick.yaml all
```

```bash
# 2) the real series: edit master_input.yaml, then step by step
python pipeline.py master_input.yaml prepare   # inspect runs/*/in.lammps!
python pipeline.py master_input.yaml run       # local, or writes SLURM jobs
python pipeline.py master_input.yaml analyze   # after the jobs finished
python pipeline.py master_input.yaml report
```

Add `--force` to redo a step that has already been done, e.g. after changing
analysis settings: `python pipeline.py master_input.yaml analyze --force`.

---

## Step 8: Good practice and pitfalls

* **System size:** ≥ 3000 atoms for structure, more for S(q) at low q and for
  minority species (e.g. 1–2 mol% dopants give very few atoms, so check `counts` in `meta.json`).
* **Convergence:** the quench rate and the force field limit the accuracy more
  than statistics do. Validate the density and the neutron/X-ray S(q) against experiment
  for at least one composition.
* **Replicas** give realistic error bars; frame-to-frame scatter within one run does not.
* **Fixed cutoffs** across a series avoid artificial trends.
* **Version control** the master input and the potential file. Archive
  `runs/*/in.lammps`, `meta.json` and `analysis/`. Trajectories are large, so
  keep them outside git (see `.gitignore`).

## Step 9: Extending the pipeline

* **New analysis:** add a function in `analysis.py` that writes a CSV and returns
  a dict of scalars; merge it into `summary` in `analyse_run()`. The report picks
  up new scalars automatically. Ideas: ring statistics, Voronoi volumes, X-ray
  S(q) (needs atomic form factors), Li–Li clustering, cation–cation distances
  for REDOR, van Hove functions.
* **New stage types:** see Step 3b.
* **Other potentials:** another YAML file. For machine-learned potentials,
  the template lines change to the relevant `pair_style`.
* **Other parameters as series variables:** the same `x` mechanism can drive the
  quench rate or pressure. Make the stage values expressions and evaluate them
  in `lammps_input.py`, as `config.py` does for the oxides.
