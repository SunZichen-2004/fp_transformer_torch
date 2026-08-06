import numpy as np
import torch

from fp_transformer_torch import (
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaussianKernel,
    ParticleDimer,
    ParticlePairDistance,
)
from fp_transformer_torch.distance import pairwise_distance


def test_advance_lambda_keeps_coordinates():
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
        lambda_init=0.1,
        lambda_max=1.0,
        ess_threshold=0.5,
        min_lambda_step=0.05,
        max_lambda_step=0.3,
        num_lambda_trials=8,
        update_pace=100,
    )
    state = integrator.init_state(seed=1)
    state = integrator.run_stage(5, initial_state=state)
    before = state.coordinate.clone()
    density = integrator.estimate_collective_variable_density(state.coordinate)
    cv_np = cv(state.coordinate).detach().cpu().numpy()
    lam_k = integrator.lambda_current
    lam_next = integrator.choose_next_lambda(cv_np, density, lam_k)
    state2, w_u, w_n = integrator.advance_lambda(
        state, cv_np, density, lam_k, lam_next
    )
    assert torch.allclose(state2.coordinate, before)
    assert torch.allclose(state2.momentum, state.momentum)
    assert integrator.lambda_current == lam_next
    assert abs(integrator.kernel_grad_beta - lam_next * integrator.beta0) < 1e-12
    assert np.isclose(w_n.sum(), 1.0)
