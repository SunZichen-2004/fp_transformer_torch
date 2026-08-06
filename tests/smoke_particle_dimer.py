"""Smoke: short particle-dimer anneal run."""

import torch

from fp_transformer_torch import (
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaussianKernel,
    ParticleDimer,
    ParticlePairDistance,
)
from fp_transformer_torch.distance import pairwise_distance


def test_smoke_particle_dimer():
    pot = ParticleDimer()
    x = pot.init_positions().repeat(16, 1, 1)
    system = EnergySystem(pot, x)
    cv = ParticlePairDistance([0, 1])
    kernel = GaussianKernel(
        16, bandwidth=0.25, distance_function=pairwise_distance, dtype=torch.float64
    )
    integrator = BaoabFokkerPlanckIntegrator(
        system,
        cv,
        kernel,
        timestep=1e-4,
        langevin_friction=1.0,
        beta0=1.0,
        lambda_init=0.1,
        lambda_max=0.4,
        ess_threshold=0.8,
        min_lambda_step=0.05,
        max_lambda_step=0.2,
        num_lambda_trials=8,
        update_pace=5,
    )
    state = integrator.run(20, seed=0, verbose=False)
    assert torch.isfinite(state.coordinate).all()
    assert integrator.lambda_current >= 0.1
