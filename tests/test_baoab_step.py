import torch

from fp_transformer_torch import (
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaussianKernel,
    ParticleDimer,
    ParticlePairDistance,
)
from fp_transformer_torch.distance import pairwise_distance


def test_baoab_step_finite_no_nan():
    pot = ParticleDimer()
    x = pot.init_positions().repeat(8, 1, 1)
    system = EnergySystem(pot, x)
    cv = ParticlePairDistance([0, 1])
    kernel = GaussianKernel(
        8, bandwidth=0.2, distance_function=pairwise_distance, dtype=torch.float64
    )
    integrator = BaoabFokkerPlanckIntegrator(
        system,
        cv,
        kernel,
        timestep=1e-4,
        langevin_friction=1.0,
        beta0=1.0,
        production_lambda=0.2,
        update_pace=5,
    )
    state = integrator.init_state(seed=0)
    state = integrator.step(state, step_index=0)
    assert torch.isfinite(state.coordinate).all()
    assert torch.isfinite(state.momentum).all()
