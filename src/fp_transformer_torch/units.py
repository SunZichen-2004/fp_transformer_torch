"""AKMA / picosecond unit helpers (matched to JAX ``water_pure_jaxmd_langevin``)."""

from __future__ import annotations

# OpenMM-derived: 1 fs in AKMA time units (dalton, Å, kcal/mol).
# Verified against JAX logs: 0.001 ps → dt_akma = 0.0204548282809.
FEMTOSECOND_TO_AKMA_TIME = 0.0204548282809

KB_KCAL_PER_MOL_K = 1.987204259e-3


def timestep_picosecond_to_akma(timestep_picosecond: float) -> float:
    """Convert BAOAB ``dt`` from picoseconds to AKMA time."""
    return float(timestep_picosecond) * 1000.0 * FEMTOSECOND_TO_AKMA_TIME


def friction_per_picosecond_to_akma(
    friction_per_picosecond: float,
    timestep_picosecond: float,
    timestep_akma: float,
) -> float:
    """Keep ``γ·dt`` invariant under the ps → AKMA map."""
    return (
        float(friction_per_picosecond)
        * float(timestep_picosecond)
        / float(timestep_akma)
    )
