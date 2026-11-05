#!/usr/bin/env python3
"""Frequency-dependent phonon MFP extraction for monatomic Cu (LAMMPS metal units).

Workflow
--------
1. Read the Cu EAM/alloy setfl header to obtain its reference lattice constant,
   mass, and radial cutoff.
2. For each cell length, use LAMMPS' PHONON-package dynamical_matrix command
   to calculate the mass-normalized harmonic Hessian for a slab around one
   transport cross-section. Convert it back to force constants.
3. Run periodic NEMD with a central hot reservoir and cold reservoirs at both
   ends. This avoids placing a hot and cold slab next to each other across the
   periodic boundary.
4. Compute the spectral heat current across one cross-section from the full
   3x3 interatomic force-constant blocks and the velocity cross spectra.
5. Normalize by the measured reservoir temperature difference and fit
   1/T(omega,L) = 1/M(omega) + L/[M(omega)*Lambda(omega)].

Important:
- This is a fast-test workflow, not a converged bulk-Cu calculation.
- The IFCs are the 0-K harmonic Hessian of the supplied EAM model. The
  frequency-resolved current method is therefore a harmonic-IFC decomposition
  of the 300-K NEMD trajectory.
- LAMMPS must be built with the PHONON package (for `dynamical_matrix`).
- Run this in a directory containing the EAM file.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import linregress

# -----------------------------------------------------------------------------
# User-adjustable settings
# -----------------------------------------------------------------------------
POTENTIAL_FILE = os.environ.get("CU_EAM_FILE", "Cu01.eam.alloy")
LAMMPS_EXECUTABLE = os.environ.get("LAMMPS_EXECUTABLE", "lmp_urc")

N_XY = 4
NZ_LIST = [24, 32, 40, 48]  # Fast-test lengths; increase for MFP convergence studies.

DT_PS = 0.001                 # 1 fs, LAMMPS metal time units (ps)
N_EQUIL_STEPS = 50_000        # 50 ps NVT equilibration at 300 K
N_NESS_STEPS = 50_000         # 50 ps NEMD steady-state establishment
N_PROD_STEPS = 200_000        # 200 ps NEMD production
DUMP_EVERY = 20               # 0.02 ps output interval; Nyquist = 25 THz
SEGMENT_FRAMES = 2_000        # 40 ps segments; 5 segments for the default run

T_HOT = 310.0
T_COLD = 290.0
LANGEVIN_DAMP_PS = 0.5
HESSIAN_DELTA_A = 1.0e-4      # Central finite-difference displacement (Angstrom)
MIN_SAMPLE_HALF_WIDTH_CELLS = 4.0

FREQ_MIN_THz = 0.10
FREQ_MAX_THz = 8.00
MIN_TRANSMISSION_FOR_FIT = 1.0e-8
MIN_FIT_POINTS = 3

KB_EV_PER_K = 8.617333262145e-5

# Thermostat regions in conventional-cell units.
HOT_HALF_WIDTH_CELLS = 1.0
COLD_THICKNESS_CELLS = 1.0


# -----------------------------------------------------------------------------
# EAM setfl metadata
# -----------------------------------------------------------------------------
def read_eam_alloy_metadata(path: str) -> Dict[str, float]:
    """Read mass, lattice constant, and cutoff from a DYNAMO setfl file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"EAM potential '{path}' was not found in {Path.cwd()}. "
            "Place Cu01.eam.alloy in the run directory or set CU_EAM_FILE."
        )

    lines = [line.strip() for line in p.read_text(errors="replace").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 5:
        raise ValueError(f"'{path}' does not look like an EAM/alloy setfl file.")

    # DYNAMO setfl files commonly begin with four comment lines, followed by
    # the element count and element-name line. Do not assume a fixed number
    # of comment lines: scan the header for a count/name pair.
    header = None
    for idx in range(min(20, len(lines) - 2)):
        fields = lines[idx].split()
        if not fields:
            continue
        try:
            candidate_count = int(fields[0])
        except ValueError:
            continue
        if not (1 <= candidate_count <= 20):
            continue

        # Most setfl files put `N_elements element1 ...` on one line.
        # Also accept files that put the names on the following line.
        if len(fields) - 1 >= candidate_count:
            names = fields[1 : 1 + candidate_count]
            data_line = idx + 1
        elif idx + 1 < len(lines):
            names = lines[idx + 1].split()
            data_line = idx + 2
        else:
            continue

        if len(names) == candidate_count and data_line + 1 < len(lines):
            header = (idx, candidate_count, names, data_line)
            break

    if header is None:
        raise ValueError(
            f"Could not locate the element-count/name lines in '{path}'. "
            "Expected a standard DYNAMO EAM/alloy setfl file."
        )

    header_line, n_elements, element_names, data_line = header
    # Standard setfl layout: each element has an atomic-data line followed
    # by a grid/cutoff line. This script uses the first (Cu) element.
    atomic_fields = lines[data_line].split()
    grid_fields = lines[data_line + 1].split()
    if len(atomic_fields) < 4 or len(grid_fields) < 5:
        raise ValueError(
            f"Could not read atomic metadata and cutoff from '{path}'. "
            "The supplied file may use a different EAM format."
        )

    atomic_mass = float(atomic_fields[1])
    lattice_constant = float(atomic_fields[2])
    lattice_type = atomic_fields[3].lower()
    cutoff = float(grid_fields[4])

    if element_names[0].lower() not in {"cu", "copper"}:
        raise ValueError(
            f"The first element in '{path}' is '{element_names[0]}', not Cu. "
            "This script is configured for a single-element Cu model."
        )
    if "fcc" not in lattice_type:
        raise ValueError(
            f"The EAM file reports lattice type '{atomic_fields[3]}'; this script "
            "currently assumes an fcc Cu reference lattice."
        )
    if atomic_mass <= 0 or lattice_constant <= 0 or cutoff <= 0:
        raise ValueError(f"Invalid mass/lattice/cutoff values in '{path}'.")

    return {
        "mass": atomic_mass,
        "a0": lattice_constant,
        "cutoff": cutoff,
    }


def get_sample_half_width_cells(cutoff_A: float, a0_A: float) -> float:
    """Include the full EAM harmonic coupling range across the sampling plane.

    For EAM, force-constant couplings can extend to pairs sharing a neighbor,
    so the slab is sized from approximately twice the EAM radial cutoff.
    """
    return max(
        MIN_SAMPLE_HALF_WIDTH_CELLS,
        2.0 * cutoff_A / a0_A + 0.5,
    )


def sample_geometry(n_z: int, a0_A: float, half_width_cells: float) -> Dict[str, float]:
    """Return thermostat/sample bounds and the unthermostatted path length."""
    z_cut_cells = n_z / 4.0 + 0.25  # Offset avoids cutting through an fcc atomic plane.
    sample_lo = z_cut_cells - half_width_cells
    sample_hi = z_cut_cells + half_width_cells
    hot_lo = n_z / 2.0 - HOT_HALF_WIDTH_CELLS
    cold_hi = COLD_THICKNESS_CELLS

    # Keep the harmonic sampling slab out of the thermostat groups.
    if sample_lo <= cold_hi + 0.05:
        raise ValueError(
            f"Nz={n_z}: sampling slab reaches the low-z cold reservoir. "
            "Increase NZ_LIST or reduce the cross-section sampling slab only "
            "after checking the EAM interaction range."
        )
    if sample_hi >= hot_lo - 0.05:
        raise ValueError(
            f"Nz={n_z}: sampling slab reaches the central hot reservoir. "
            "Increase NZ_LIST or reduce the cross-section sampling slab only "
            "after checking the EAM interaction range."
        )

    # Distance between the inner faces of the hot and cold thermostat regions.
    path_length_A = (hot_lo - cold_hi) * a0_A
    return {
        "z_cut_cells": z_cut_cells,
        "z_cut_A": z_cut_cells * a0_A,
        "sample_lo_cells": sample_lo,
        "sample_hi_cells": sample_hi,
        "path_length_A": path_length_A,
        "box_length_A": n_z * a0_A,
    }


# -----------------------------------------------------------------------------
# LAMMPS input generation
# -----------------------------------------------------------------------------
def generate_hessian_input(
    n_z: int,
    a0_A: float,
    mass: float,
    potential_file: str,
    geometry: Dict[str, float],
) -> Tuple[str, str, str]:
    input_file = f"in.cu_hessian_nz{n_z}.lmp"
    dynmat_file = f"cu_hessian_nz{n_z}.bin"
    ids_file = f"cu_sample_atoms_nz{n_z}.dump"

    script = f"""# Static EAM Hessian for Cu, Nz={n_z}
units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map array

lattice         fcc {a0_A:.12g}
region          box block 0 {N_XY} 0 {N_XY} 0 {n_z} units lattice
create_box      1 box
create_atoms    1 box
mass            1 {mass:.12g}

pair_style      eam/alloy
pair_coeff      * * {potential_file} Cu
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

thermo          1000
thermo_style    custom step pe press pxx pyy pzz
min_style       cg
minimize        1.0e-12 1.0e-12 10000 100000

region          reg_sample block INF INF INF INF {geometry['sample_lo_cells']:.8f} {geometry['sample_hi_cells']:.8f} units lattice
group           g_sample region reg_sample

# Group map is ordered by atom ID in LAMMPS dynamical_matrix.
write_dump      g_sample custom {ids_file} id x y z modify sort id

# LAMMPS PHONON package; binary output preserves double precision.
dynamical_matrix g_sample regular {HESSIAN_DELTA_A:.8g} file {dynmat_file} binary yes
"""
    Path(input_file).write_text(script)
    return input_file, dynmat_file, ids_file


def generate_nemd_input(
    n_z: int,
    a0_A: float,
    mass: float,
    potential_file: str,
    geometry: Dict[str, float],
) -> Tuple[str, str, str, str, str]:
    input_file = f"in.cu_nemd_nz{n_z}.lmp"
    dump_file = f"cu_velocities_nz{n_z}.lammpstrj"
    energy_file = f"cu_thermostat_energy_nz{n_z}.dat"
    temperature_file = f"cu_reservoir_temps_nz{n_z}.dat"
    log_file = f"log.cu_nemd_nz{n_z}.lammps"

    script = f"""# Periodic Cu NEMD: central hot slab, cold slabs at both ends
units           metal
dimension       3
boundary        p p p
atom_style      atomic

lattice         fcc {a0_A:.12g}
region          box block 0 {N_XY} 0 {N_XY} 0 {n_z} units lattice
create_box      1 box
create_atoms    1 box
mass            1 {mass:.12g}

pair_style      eam/alloy
pair_coeff      * * {potential_file} Cu
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# Match the reference coordinates used for the harmonic Hessian calculation.
min_style       cg
minimize        1.0e-12 1.0e-12 10000 100000

timestep        {DT_PS:.8g}

# Define fixed spatial groups before equilibration.
region          reg_hot block INF INF INF INF {n_z / 2.0 - HOT_HALF_WIDTH_CELLS:.8f} {n_z / 2.0 + HOT_HALF_WIDTH_CELLS:.8f} units lattice
region          reg_cold_lo block INF INF INF INF 0.0 {COLD_THICKNESS_CELLS:.8f} units lattice
region          reg_cold_hi block INF INF INF INF {n_z - COLD_THICKNESS_CELLS:.8f} {n_z:.8f} units lattice
region          reg_sample block INF INF INF INF {geometry['sample_lo_cells']:.8f} {geometry['sample_hi_cells']:.8f} units lattice

group           g_hot region reg_hot
group           g_cold_lo region reg_cold_lo
group           g_cold_hi region reg_cold_hi
group           g_cold union g_cold_lo g_cold_hi
group           g_sample region reg_sample

velocity        all create 300.0 12345 mom yes rot yes dist gaussian
fix             equil all nvt temp 300.0 300.0 0.1
thermo          5000
thermo_style    custom step temp pe ke etotal pxx pyy pzz
run             {N_EQUIL_STEPS}
unfix           equil

# NEMD production: hot in the middle, cold at both periodic ends.
fix             integrator all nve
fix             hot g_hot langevin {T_HOT:.8g} {T_HOT:.8g} {LANGEVIN_DAMP_PS:.8g} 54321 tally yes
fix             cold g_cold langevin {T_COLD:.8g} {T_COLD:.8g} {LANGEVIN_DAMP_PS:.8g} 98765 tally yes
compute         t_hot g_hot temp
compute         t_cold g_cold temp
thermo          5000
thermo_style    custom step temp c_t_hot c_t_cold f_hot f_cold

# Establish non-equilibrium steady state before opening the production dump.
run             {N_NESS_STEPS}
print           "PROD_START $(f_hot:%.16g) $(f_cold:%.16g)" file {energy_file} screen no

# Measure the actual reservoir temperature difference during production.
fix             tempstats all ave/time 100 10 1000 c_t_hot c_t_cold file {temperature_file}
dump            d_vel g_sample custom {DUMP_EVERY} {dump_file} id vx vy vz
dump_modify     d_vel sort id format float %20.16g
run             {N_PROD_STEPS}
undump          d_vel
unfix           tempstats
print           "PROD_END $(f_hot:%.16g) $(f_cold:%.16g)" append {energy_file} screen no
"""
    Path(input_file).write_text(script)
    return input_file, dump_file, energy_file, temperature_file, log_file


# -----------------------------------------------------------------------------
# LAMMPS execution and file readers
# -----------------------------------------------------------------------------
def run_lammps(input_file: str, log_file: str) -> None:
    exe = shlex.split(LAMMPS_EXECUTABLE)
    if not exe:
        raise ValueError("LAMMPS_EXECUTABLE is empty.")

    n_tasks = int(os.environ.get("SLURM_NTASKS", os.environ.get("MPI_NTASKS", "1")))
    if n_tasks < 1:
        raise ValueError("MPI task count must be at least 1.")

    if os.environ.get("SLURM_JOB_ID"):
        launcher = shlex.split(os.environ.get("MPI_LAUNCHER", "srun"))
        cmd = launcher + ["-n", str(n_tasks)] + exe
    elif n_tasks > 1:
        launcher = shlex.split(os.environ.get("MPI_LAUNCHER", "mpirun"))
        cmd = launcher + ["-np", str(n_tasks)] + exe
    else:
        cmd = exe

    cmd += ["-in", input_file, "-log", log_file]
    print("Executing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_first_dump_frame(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the first custom dump frame; return sorted IDs and xyz positions."""
    with open(path, "rt") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"No dump frame found in '{path}'.")
            if line.startswith("ITEM: TIMESTEP"):
                break
        f.readline()  # timestep
        if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Malformed dump header in '{path}'.")
        n_atoms = int(f.readline().strip())
        bounds_header = f.readline()
        if not bounds_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Missing BOX BOUNDS in '{path}'.")
        for _ in range(3):
            f.readline()
        atom_header = f.readline().strip().split()[2:]
        rows = [f.readline().split() for _ in range(n_atoms)]

    col = {name: i for i, name in enumerate(atom_header)}
    required = {"id", "x", "y", "z"}
    if not required.issubset(col):
        raise ValueError(f"Expected columns {required} in '{path}', got {atom_header}.")

    arr = np.asarray(rows, dtype=float)
    ids = arr[:, col["id"]].astype(np.int64)
    xyz = arr[:, [col["x"], col["y"], col["z"]]]
    order = np.argsort(ids)
    return ids[order], xyz[order]


def read_binary_hessian(path: str, n_atoms: int, mass: float) -> Tuple[np.ndarray, float]:
    """Read LAMMPS dynamical_matrix binary output and recover Phi (eV/A^2)."""
    raw = np.fromfile(path, dtype=np.float64)
    dimension = 3 * n_atoms
    expected = dimension * dimension
    if raw.size != expected:
        raise ValueError(
            f"Hessian file '{path}' has {raw.size} doubles; expected {expected} "
            f"for {n_atoms} sample atoms. Confirm that LAMMPS wrote binary output "
            "and that the PHONON package is enabled."
        )

    dynamical_matrix = raw.reshape(dimension, dimension)
    denom = np.linalg.norm(dynamical_matrix)
    symmetry_error = (
        np.linalg.norm(dynamical_matrix - dynamical_matrix.T) / denom
        if denom > 0 else np.inf
    )

    # LAMMPS regular dynamical_matrix output is Phi_ij/(sqrt(m_i*m_j)).
    # All atoms are Cu with the same mass, so multiply by m_Cu to recover Phi.
    phi_matrix = 0.5 * (dynamical_matrix + dynamical_matrix.T) * mass
    phi = phi_matrix.reshape(n_atoms, 3, n_atoms, 3)
    return phi, float(symmetry_error)


def read_velocity_dump(path: str, expected_ids: np.ndarray) -> np.ndarray:
    """Read a multi-frame `id vx vy vz` dump into [frame, atom, xyz]."""
    id_to_index = {int(atom_id): idx for idx, atom_id in enumerate(expected_ids)}
    frames: List[np.ndarray] = []

    with open(path, "rt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break
            if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed frame in '{path}'.")
            n_atoms = int(f.readline().strip())
            if n_atoms != len(expected_ids):
                raise ValueError(
                    f"Frame in '{path}' has {n_atoms} atoms; expected {len(expected_ids)}."
                )
            if not f.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing BOX BOUNDS in '{path}'.")
            for _ in range(3):
                f.readline()
            atom_header = f.readline().strip().split()[2:]
            col = {name: i for i, name in enumerate(atom_header)}
            required = {"id", "vx", "vy", "vz"}
            if not required.issubset(col):
                raise ValueError(f"Expected columns {required} in '{path}', got {atom_header}.")

            frame = np.empty((len(expected_ids), 3), dtype=np.float64)
            found = set()
            for _ in range(n_atoms):
                parts = f.readline().split()
                atom_id = int(parts[col["id"]])
                if atom_id not in id_to_index:
                    raise ValueError(f"Unexpected atom ID {atom_id} in '{path}'.")
                idx = id_to_index[atom_id]
                frame[idx, :] = [
                    float(parts[col["vx"]]),
                    float(parts[col["vy"]]),
                    float(parts[col["vz"]]),
                ]
                found.add(atom_id)
            if len(found) != len(expected_ids):
                raise ValueError(f"Incomplete atom set in a frame of '{path}'.")
            frames.append(frame)

    if not frames:
        raise ValueError(f"No velocity frames found in '{path}'.")
    return np.stack(frames, axis=0)


def read_thermostat_energy_change(path: str, production_time_ps: float) -> Tuple[float, float, float]:
    """Return signed hot/cold energy changes and hot-side power magnitude."""
    start = end = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("PROD_START"):
            vals = line.split()
            start = (float(vals[1]), float(vals[2]))
        elif line.startswith("PROD_END"):
            vals = line.split()
            end = (float(vals[1]), float(vals[2]))
    if start is None or end is None:
        raise ValueError(
            f"Could not find PROD_START/PROD_END energy records in '{path}'. "
            "Check that the Langevin fixes use `tally yes`."
        )
    delta_hot = end[0] - start[0]
    delta_cold = end[1] - start[1]
    hot_power_abs = abs(delta_hot) / production_time_ps
    return delta_hot, delta_cold, hot_power_abs


def read_measured_delta_t(path: str) -> float:
    """Read averaged hot/cold group temperatures from fix ave/time output."""
    try:
        arr = np.loadtxt(path, comments="#", ndmin=2)
    except Exception as exc:
        raise ValueError(f"Could not parse reservoir temperature file '{path}'.") from exc
    if arr.size == 0 or arr.shape[1] < 3:
        raise ValueError(
            f"Reservoir temperature file '{path}' does not contain step, T_hot, T_cold."
        )
    delta_t = float(np.mean(arr[:, 1] - arr[:, 2]))
    if not np.isfinite(delta_t) or abs(delta_t) < 1.0e-6:
        raise ValueError(f"Invalid measured reservoir temperature difference: {delta_t} K")
    return abs(delta_t)


# -----------------------------------------------------------------------------
# Spectral heat current and transmission
# -----------------------------------------------------------------------------
def compute_spectral_transmission(
    velocity_frames: np.ndarray,
    atom_ids: np.ndarray,
    atom_xyz: np.ndarray,
    phi: np.ndarray,
    z_cut_A: float,
    dt_dump_ps: float,
    delta_t_K: float,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Calculate cross-section spectral current using full 3x3 IFC blocks.

    The hot side is the +z side of the selected plane (toward the central hot
    thermostat); the cold side is the -z side (toward an end cold thermostat).
    The global sign is oriented so the integrated current is hot-to-cold.
    Frequency-resolved negative contributions are retained.
    """
    if phi.shape != (len(atom_ids), 3, len(atom_ids), 3):
        raise ValueError("Hessian atom count/order does not match the sample atom list.")
    if len(atom_xyz) != len(atom_ids):
        raise ValueError("Sample coordinate count does not match sample atom IDs.")

    cold_idx = np.flatnonzero(atom_xyz[:, 2] < z_cut_A)
    hot_idx = np.flatnonzero(atom_xyz[:, 2] > z_cut_A)
    if len(cold_idx) == 0 or len(hot_idx) == 0:
        raise ValueError("The sampling slab must contain atoms on both sides of z_cut.")

    # Phi[i,alpha,j,beta], i on hot side, j on cold side.
    phi_hot_cold = phi[np.ix_(hot_idx, np.arange(3), cold_idx, np.arange(3))]

    n_frames = velocity_frames.shape[0]
    n_segments = n_frames // SEGMENT_FRAMES
    if n_segments < 2:
        raise ValueError(
            f"Only {n_frames} frames were read; at least {2 * SEGMENT_FRAMES} are "
            "needed for two spectral segments. Increase N_PROD_STEPS or reduce "
            "SEGMENT_FRAMES."
        )
    n_used = n_segments * SEGMENT_FRAMES

    freqs_thz = rfftfreq(SEGMENT_FRAMES, d=dt_dump_ps)  # cycles/ps = THz
    omega_rad_ps = 2.0 * np.pi * freqs_thz
    q_segments = []
    segment_time_ps = SEGMENT_FRAMES * dt_dump_ps

    for seg in range(n_segments):
        lo = seg * SEGMENT_FRAMES
        hi = lo + SEGMENT_FRAMES
        v = velocity_frames[lo:hi].copy()

        # Remove residual drift independently for each atom and segment.
        v -= np.mean(v, axis=0, keepdims=True)
        vhat = rfft(v, axis=0) * dt_dump_ps  # (A/ps)*ps = A

        cross_spectrum = np.einsum(
            "fia,iajb,fjb->f",
            np.conjugate(vhat[:, hot_idx, :]),
            phi_hot_cold,
            vhat[:, cold_idx, :],
            optimize=True,
        )

        q_omega = np.zeros_like(freqs_thz, dtype=np.float64)
        positive = omega_rad_ps > 0.0
        q_omega[positive] = (
            -2.0
            * np.imag(cross_spectrum[positive])
            / (segment_time_ps * omega_rad_ps[positive])
        )
        q_segments.append(q_omega)

    q_omega_avg = np.mean(np.stack(q_segments, axis=0), axis=0)

    # Orient the complete spectrum to the known net hot-to-cold direction.
    # Do not take abs() frequency-by-frequency; negative spectral contributions
    # can be physical/statistical features of the decomposition.
    q_integral_eV_per_ps = float(np.trapz(q_omega_avg, x=omega_rad_ps) / (2.0 * np.pi))
    if q_integral_eV_per_ps < 0.0:
        q_omega_avg *= -1.0
        q_integral_eV_per_ps *= -1.0

    transmission = (2.0 * np.pi * q_omega_avg) / (KB_EV_PER_K * delta_t_K)
    used_segments = n_segments
    return freqs_thz, transmission, q_integral_eV_per_ps, used_segments


# -----------------------------------------------------------------------------
# MFP fit and plots
# -----------------------------------------------------------------------------
def extract_mfp(
    freqs_thz: np.ndarray,
    transmission_matrix: np.ndarray,
    path_lengths_A: np.ndarray,
) -> pd.DataFrame:
    n_freq = len(freqs_thz)
    mfp_nm = np.full(n_freq, np.nan)
    mode_count = np.full(n_freq, np.nan)
    r_squared = np.full(n_freq, np.nan)
    n_fit = np.zeros(n_freq, dtype=int)
    slope_A_inv = np.full(n_freq, np.nan)
    intercept = np.full(n_freq, np.nan)

    for k in range(n_freq):
        tvals = transmission_matrix[:, k]
        valid = np.isfinite(tvals) & (tvals > MIN_TRANSMISSION_FOR_FIT)
        if np.count_nonzero(valid) < MIN_FIT_POINTS:
            continue

        x_A = path_lengths_A[valid]
        y = 1.0 / tvals[valid]
        fit = linregress(x_A, y)
        if fit.slope > 0.0 and fit.intercept > 0.0:
            mfp_nm[k] = (fit.intercept / fit.slope) / 10.0
            mode_count[k] = 1.0 / fit.intercept
            r_squared[k] = fit.rvalue**2
            slope_A_inv[k] = fit.slope
            intercept[k] = fit.intercept
            n_fit[k] = np.count_nonzero(valid)

    return pd.DataFrame(
        {
            "Frequency_THz": freqs_thz,
            "MFP_nm": mfp_nm,
            "Mode_Count_Extrapolated": mode_count,
            "Fit_R2": r_squared,
            "N_Lengths_Used": n_fit,
            "Slope_per_Angstrom": slope_A_inv,
            "Intercept": intercept,
        }
    )


def save_plots(freqs_thz: np.ndarray, fit_df: pd.DataFrame) -> None:
    valid = np.isfinite(fit_df["MFP_nm"]) & (fit_df["MFP_nm"] > 0.0)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)

    if np.any(valid):
        ax.scatter(
            fit_df.loc[valid, "Frequency_THz"],
            fit_df.loc[valid, "MFP_nm"],
            s=18,
            alpha=0.55,
            edgecolors="none",
            label="Frequency-resolved fits",
        )
        bins = np.linspace(FREQ_MIN_THz, FREQ_MAX_THz, 31)
        work = fit_df.loc[valid, ["Frequency_THz", "MFP_nm"]].copy()
        work["bin"] = pd.cut(work["Frequency_THz"], bins=bins, include_lowest=True)
        med = work.groupby("bin", observed=False)["MFP_nm"].median()
        centers = 0.5 * (bins[:-1] + bins[1:])
        nonempty = med.notna().to_numpy()
        if np.any(nonempty):
            ax.plot(
                centers[nonempty],
                med.to_numpy()[nonempty],
                marker="o",
                linewidth=2.0,
                markersize=4,
                label="Binned median",
            )
        ax.set_yscale("log")
        ax.legend(frameon=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No frequency bins passed the positive-slope / positive-intercept fit checks",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_title("Cu Phonon MFP Spectrum (EAM IFC + Periodic NEMD Fast Test)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon mean free path (nm)")
    ax.set_xlim(FREQ_MIN_THz, FREQ_MAX_THz)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("cu_mfp_spectrum_fixed.png")
    plt.close(fig)

    # A second plot makes it easier to see whether the fitted transmission
    # actually varies monotonically with the sample length at each frequency.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.plot(freqs_thz, fit_df["Fit_R2"], linewidth=1.5)
    ax.set_xlim(FREQ_MIN_THz, FREQ_MAX_THz)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$R^2$ of $1/T$ vs. transport length")
    ax.set_title("MFP Fit Quality")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("cu_mfp_fit_quality.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    meta = read_eam_alloy_metadata(POTENTIAL_FILE)
    mass = meta["mass"]
    a0 = meta["a0"]
    cutoff = meta["cutoff"]
    sample_half_width = get_sample_half_width_cells(cutoff, a0)
    dt_dump_ps = DUMP_EVERY * DT_PS
    production_time_ps = N_PROD_STEPS * DT_PS
    delta_t_target = T_HOT - T_COLD

    print("Cu EAM metadata:")
    print(f"  mass = {mass:.6f} g/mol")
    print(f"  lattice constant = {a0:.6f} A")
    print(f"  EAM cutoff = {cutoff:.6f} A")
    print(f"  sample half-width = {sample_half_width:.3f} lattice cells")
    print(f"  dump interval = {dt_dump_ps:.5f} ps; Nyquist = {0.5 / dt_dump_ps:.2f} THz")
    print(f"  production duration = {production_time_ps:.2f} ps")

    results = []
    path_lengths_A: List[float] = []
    all_transmissions: List[np.ndarray] = []
    target_freqs = None
    run_labels: List[str] = []

    for n_z in NZ_LIST:
        geometry = sample_geometry(n_z, a0, sample_half_width)
        print(
            f"\n=== Nz={n_z}: box={geometry['box_length_A'] / 10.0:.3f} nm, "
            f"transport gap={geometry['path_length_A'] / 10.0:.3f} nm ===",
            flush=True,
        )

        # A. Full EAM harmonic IFC matrix for atoms in the sampling slab.
        hess_in, hess_file, ids_file = generate_hessian_input(
            n_z, a0, mass, POTENTIAL_FILE, geometry
        )
        run_lammps(hess_in, f"log.cu_hessian_nz{n_z}.lammps")
        sample_ids, sample_xyz = read_first_dump_frame(ids_file)
        phi, hessian_symmetry_error = read_binary_hessian(hess_file, len(sample_ids), mass)
        print(
            f"  sample atoms={len(sample_ids)}, Hessian relative asymmetry="
            f"{hessian_symmetry_error:.3e}",
            flush=True,
        )
        if hessian_symmetry_error > 5.0e-2:
            print(
                "  WARNING: Hessian asymmetry is >5%; check the LAMMPS PHONON build, "
                "finite-difference step, and sample atom ordering.",
                flush=True,
            )

        # B. Periodic NEMD trajectory.
        nemd_in, vel_file, energy_file, temp_file, log_file = generate_nemd_input(
            n_z, a0, mass, POTENTIAL_FILE, geometry
        )
        run_lammps(nemd_in, log_file)
        velocity_frames = read_velocity_dump(vel_file, sample_ids)
        delta_hot, delta_cold, hot_power_abs = read_thermostat_energy_change(
            energy_file, production_time_ps
        )
        delta_t_measured = read_measured_delta_t(temp_file)

        freqs, transmission, spectral_power, n_segments = compute_spectral_transmission(
            velocity_frames=velocity_frames,
            atom_ids=sample_ids,
            atom_xyz=sample_xyz,
            phi=phi,
            z_cut_A=geometry["z_cut_A"],
            dt_dump_ps=dt_dump_ps,
            delta_t_K=delta_t_measured,
        )

        # One of the two symmetric heat-flow branches carries approximately
        # half the total energy supplied to the central hot reservoir.
        branch_power = hot_power_abs / 2.0
        if branch_power > 0:
            rel_power_error = abs(spectral_power - branch_power) / branch_power
        else:
            rel_power_error = np.inf

        print(f"  measured reservoir DeltaT = {delta_t_measured:.3f} K (target {delta_t_target:.3f} K)")
        print(f"  thermostat energy changes: hot={delta_hot:.6e} eV, cold={delta_cold:.6e} eV")
        print(f"  branch power from hot bath = {branch_power:.6e} eV/ps")
        print(f"  integrated spectral current = {spectral_power:.6e} eV/ps")
        print(f"  spectral/thermostat power relative difference = {rel_power_error:.1%}")
        print(f"  averaged {n_segments} spectral segments from {velocity_frames.shape[0]} frames")
        if rel_power_error > 0.50:
            print(
                "  WARNING: spectral current does not yet match thermostat power within 50%; "
                "do not interpret the resulting MFP fit as validated.",
                flush=True,
            )

        mask = (freqs >= FREQ_MIN_THz) & (freqs <= FREQ_MAX_THz)
        freqs_band = freqs[mask]
        trans_band = transmission[mask]
        if target_freqs is None:
            target_freqs = freqs_band
        elif not np.allclose(target_freqs, freqs_band, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("Frequency grids differ between lengths; check dump/segment settings.")

        path_lengths_A.append(geometry["path_length_A"])
        all_transmissions.append(trans_band)
        run_labels.append(f"Nz{n_z}")

        # Save each length's raw spectrum and key diagnostics.
        pd.DataFrame(
            {
                "Frequency_THz": freqs_band,
                "Transmission": trans_band,
            }
        ).to_csv(f"cu_transmission_nz{n_z}.csv", index=False)

        results.append(
            {
                "Nz": n_z,
                "Box_Length_nm": geometry["box_length_A"] / 10.0,
                "Transport_Gap_nm": geometry["path_length_A"] / 10.0,
                "Sample_Atoms": len(sample_ids),
                "Hessian_Asymmetry": hessian_symmetry_error,
                "Measured_DeltaT_K": delta_t_measured,
                "Hot_Energy_Change_eV": delta_hot,
                "Cold_Energy_Change_eV": delta_cold,
                "Branch_Power_eV_per_ps": branch_power,
                "Spectral_Power_eV_per_ps": spectral_power,
                "Power_Relative_Difference": rel_power_error,
                "Spectral_Segments": n_segments,
                "Velocity_Frames": velocity_frames.shape[0],
            }
        )

        # Remove large intermediates/dumps before the next length.
        del velocity_frames, phi

    if len(path_lengths_A) < MIN_FIT_POINTS:
        raise RuntimeError(
            f"Only {len(path_lengths_A)} successful lengths; at least {MIN_FIT_POINTS} "
            "are required for a useful linear MFP fit."
        )

    path_lengths_A_array = np.asarray(path_lengths_A, dtype=float)
    transmission_matrix = np.stack(all_transmissions, axis=0)
    fit_df = extract_mfp(target_freqs, transmission_matrix, path_lengths_A_array)

    # Preserve the complete data, including frequencies where the fit was not
    # physically constrained. NaN marks an unresolved fit; no arbitrary 100-nm
    # clipping is applied.
    fit_df.to_csv("cu_mfp_results_fixed.csv", index=False)
    transmission_df = pd.DataFrame({"Frequency_THz": target_freqs})
    for label, column in zip(run_labels, transmission_matrix):
        transmission_df[f"Transmission_{label}"] = column
    transmission_df.to_csv("cu_transmission_all_lengths.csv", index=False)
    pd.DataFrame(results).to_csv("cu_mfp_run_diagnostics.csv", index=False)

    save_plots(target_freqs, fit_df)

    n_valid = int(np.isfinite(fit_df["MFP_nm"]).sum())
    print("\n=== COMPLETE ===")
    print(f"Valid MFP fits: {n_valid}/{len(fit_df)} frequency points")
    print("Saved:")
    print("  cu_mfp_results_fixed.csv")
    print("  cu_transmission_all_lengths.csv")
    print("  cu_mfp_run_diagnostics.csv")
    print("  cu_mfp_spectrum_fixed.png")
    print("  cu_mfp_fit_quality.png")
    print(f"Total wall time: {(time.time() - t0) / 60.0:.2f} min")

    if n_valid == 0:
        print(
            "WARNING: no frequencies passed the fit checks. Inspect the raw "
            "transmission CSVs and power diagnostics before changing fit thresholds."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            "LAMMPS failed. Inspect the corresponding log file. If the error says "
            "'Unknown command: dynamical_matrix', the installed LAMMPS executable "
            "was built without the PHONON package.",
            file=sys.stderr,
        )
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
