import torch

from fp_transformer_torch import ParticleDimer


def test_particle_dimer_energy_finite():
    pot = ParticleDimer()
    x = pot.init_positions().repeat(4, 1, 1)
    energy = pot.energy(x)
    assert energy.shape == (4, 1)
    assert torch.isfinite(energy).all()
    force = pot.force(x)
    assert force.shape == x.shape
    assert torch.isfinite(force).all()
