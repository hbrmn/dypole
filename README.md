# dypole
 Collection of data processing functions for 1D and 2D NMR data.

 *Important*
 This code is still a work-in-progress and full of surprises (bugs) - be wary!

## To-do list
1. Create a UI for the master_input file or alternatively include data_processing it into the GUI of ssNake.
1. Put figure options and plotting into a function to call as needed
1. Create classes to optimize code
1. Make Inputscript return file with all used parameters for processing, that can be loaded to restore the session later.

## MD pipeline (`md_pipeline/`)
Automated LAMMPS melt-quench simulations for compositional series, driven by one
master input file and ending in RDFs, coordination numbers, Qn, S(q), diffusion
and NMR second moments. See [md_pipeline/README.md](md_pipeline/README.md) for a
step-by-step tutorial.
