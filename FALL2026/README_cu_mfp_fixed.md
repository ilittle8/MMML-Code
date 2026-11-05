# Copper phonon MFP fast-test workflow

## Files

- `run_cu_mfp_test_fixed.py` — generates/runs the Hessian and NEMD calculations, computes spectral transmission, fits MFP, and writes CSV/PNG diagnostics.
- `submit_cu_mfp_test_fixed.sh` — Slurm submission script for the Orion partition.

## Before submitting

1. Put both files and `Cu01.eam.alloy` in the same run directory.
2. Confirm that `lmp_urc` is the correct executable on the cluster. Override it with `LAMMPS_EXECUTABLE` if needed.
3. Confirm that the LAMMPS 02Aug23 build includes the PHONON package and supports the `dynamical_matrix` command. The job will stop at the Hessian step if that command is unavailable.
4. Confirm that the Python environment has NumPy, SciPy, pandas, and Matplotlib.

Submit with:

```bash
sbatch submit_cu_mfp_test_fixed.sh
```

## Main changes from the original script

- Hot reservoir is in the center; cold reservoirs are at both periodic ends.
- Uses a fixed-cell NVT equilibration rather than NPT so the geometric regions and sample length remain consistent.
- Uses an EAM-derived harmonic force-constant matrix from LAMMPS `dynamical_matrix`; it no longer uses arbitrary nearest-neighbor pairing or a single scalar spring constant.
- Keeps FFT time units in ps, consistent with LAMMPS `metal` units.
- Uses measured hot/cold reservoir temperatures rather than scaling the imposed temperature difference by box length.
- Fits using the unthermostatted branch length, requires at least three lengths, retains fit diagnostics, and does not clip MFPs at 100 nm.
- Compares the integrated spectral current against the thermostat power assigned to one of the two symmetric heat-flow branches.

## Important limitations

This is a diagnostic fast-test, not a converged bulk-copper result. Default `N_XY=4`, four lengths, 200 ps production, and five 40 ps spectral segments are intended to debug the workflow. Cross-sectional size, longitudinal length range, production duration, number of independent seeds, and spectral segment length need convergence testing.

The Hessian is calculated from a minimized 0 K configuration and applied to the 300 K NEMD trajectory. This is a harmonic-IFC spectral decomposition of an anharmonic trajectory; test sensitivity to the reference configuration and temperature before treating the spectrum as quantitative.

The result is the lattice/phonon contribution represented by the classical EAM model. It does not include electronic heat conduction in copper.

The script checks whether the integrated spectral current agrees with the thermostat power. A large mismatch means the spectral current normalization, force constants, sampling geometry, or statistics are not yet validated. Do not interpret an MFP fit as reliable when this diagnostic fails.
