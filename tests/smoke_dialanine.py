"""Smoke: short GAFF + torsion + torus BAOAB-FP run (local dialanine fixture)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fp_transformer_torch import (
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaffPotential,
    GaussianKernel,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance
from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates

SPONGE = (
    Path(__file__).resolve().parents[1] / "experiments" / "dialanine" / "dialanine"
)
TORSION = np.array([[5, 7, 9, 15], [7, 9, 15, 17]], dtype=np.int64)


@pytest.mark.skipif(not Path(str(SPONGE) + "_charge.txt").is_file(), reason="no fixture")
def test_smoke_dialanine_gaff():
    pot = GaffPotential(str(SPONGE))
    assert pot.num_atoms == 22
    base = torch.as_tensor(
        load_sponge_coordinates(SPONGE, pot.num_atoms), dtype=torch.float64
    )
    x = base.unsqueeze(0).repeat(4, 1, 1)
    system = EnergySystem(pot, x)
    cv = Torsion(TORSION)
    kernel = GaussianKernel(
        4,
        bandwidth=0.5,
        distance_function=torus_pairwise_distance,
        dtype=torch.float64,
    )
    integrator = BaoabFokkerPlanckIntegrator(
        system,
        cv,
        kernel,
        timestep=5e-4,
        langevin_friction=1.0,
        beta0=1.0,
        lambda_init=0.1,
        lambda_max=0.3,
        ess_threshold=0.8,
        min_lambda_step=0.05,
        max_lambda_step=0.15,
        num_lambda_trials=6,
        update_pace=5,
        mass=1.0,
    )
    state = integrator.run(10, seed=0, verbose=False)
    assert torch.isfinite(state.coordinate).all()
