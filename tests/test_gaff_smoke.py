from pathlib import Path

import numpy as np
import pytest
import torch

from fp_transformer_torch import GaffPotential
from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates

SPONGE = (
    Path(__file__).resolve().parents[1] / "experiments" / "dialanine" / "dialanine"
)


@pytest.mark.skipif(not Path(str(SPONGE) + "_charge.txt").is_file(), reason="no fixture")
def test_gaff_energy_and_force():
    pot = GaffPotential(str(SPONGE))
    coords = torch.as_tensor(
        load_sponge_coordinates(SPONGE, pot.num_atoms), dtype=torch.float64
    ).unsqueeze(0)
    assert coords.shape[1] == pot.num_atoms
    energy = pot.energy(coords)
    assert energy.shape == (1, 1)
    assert torch.isfinite(energy).all()
    force = pot.force(coords)
    assert force.shape == coords.shape
    assert torch.isfinite(force).all()
