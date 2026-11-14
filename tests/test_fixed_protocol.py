import numpy as np
import pytest
import torch
from fp_transformer_torch import (
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaussianKernel,
    ParticleDimer,
    ParticlePairDistance,
)
from fp_transformer_torch.distance import pairwise_distance
from fp_transformer_torch.integrator.protocol import run_fixed_protocol, validate_lambda


def test_healing_then_fixed_production_with_partial_last_chunk():
    class Driver:
        production_lambda = 0.1

        def sync_lambda_to_kernel_grad_beta(self):
            self.kernel_grad_beta = self.lambda_current * 2

    driver = Driver()
    calls = []
    frames = []

    def advance(state, steps):
        calls.append((steps, driver.lambda_current, driver.kernel_grad_beta))
        return state + steps

    state = run_fixed_protocol(
        driver,
        0,
        advance,
        25,
        10,
        [0.5, 0.2],
        7,
        lambda phase, step, lam, state: frames.append((phase, step, lam, state)),
    )
    assert state == 39
    assert calls == [
        (7, 0.5, 1.0),
        (7, 0.2, 0.4),
        (10, 0.1, 0.2),
        (10, 0.1, 0.2),
        (5, 0.1, 0.2),
    ]
    assert frames[2] == ("production", 0, 0.1, 14)
    assert frames[-1] == ("production", 25, 0.1, 39)
    assert driver.lambda_current == 0.1


@pytest.mark.parametrize("value", [0, -0.1, np.inf, np.nan])
def test_lambda_rejected_before_dynamics(value):
    with pytest.raises(ValueError):
        validate_lambda(value)


def test_real_baoab_protocol_remains_finite_and_fixed():
    potential = ParticleDimer()
    x = potential.init_positions().repeat(8, 1, 1)
    kernel = GaussianKernel(
        8, bandwidth=0.2, distance_function=pairwise_distance, dtype=torch.float64
    )
    driver = BaoabFokkerPlanckIntegrator(
        EnergySystem(potential, x),
        ParticlePairDistance([0, 1]),
        kernel,
        timestep=1e-4,
        langevin_friction=1,
        beta0=1,
        production_lambda=0.2,
        update_pace=5,
    )
    values = []
    final = driver.run(
        13,
        dump_every=5,
        heal_lambdas=[0.5],
        heal_steps=2,
        show_progress=False,
        on_frame=lambda phase, step, lam, state: values.append((phase, step, lam)),
    )
    assert values == [
        ("heal", 2, 0.5),
        ("production", 0, 0.2),
        ("production", 5, 0.2),
        ("production", 10, 0.2),
        ("production", 13, 0.2),
    ]
    assert torch.isfinite(final.coordinate).all()
    assert torch.isfinite(final.momentum).all()
