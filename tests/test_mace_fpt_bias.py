"""Smoke: MACE SumModel + FPTBias force shapes (needs MACE_torchsim + checkpoint)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

SPONGE = Path(__file__).resolve().parents[1] / "experiments" / "dialanine" / "dialanine"
PDB = Path(__file__).resolve().parents[1] / "experiments" / "dialanine" / "dialanine.pdb"
CACHE = Path.home() / ".cache" / "mace" / "MACE-OFF23_small.model"
TORSION = np.array([[5, 7, 9, 15], [7, 9, 15, 17]], dtype=np.int64)


@pytest.mark.skipif(not CACHE.is_file(), reason="MACE-OFF23_small.model missing")
@pytest.mark.skipif(not PDB.is_file(), reason="dialanine.pdb missing")
def test_mace_fpt_bias_sum_model_cpu():
    pytest.importorskip("torch_sim")
    # ase not required for this smoke (organic PDB parser is local)
    from torch_sim.models.interface import SumModel

    from fp_transformer_torch import (
        FPTBias,
        GaussianKernel,
        PerReplicaRescaleConfig,
        Torsion,
        load_mace_torch_sim_model,
    )
    from fp_transformer_torch.distance import torus_pairwise_distance
    from fp_transformer_torch.kernel.gaussian import PerReplicaRescaleConfig as PRC
    from fp_transformer_torch.potential.mace import (
        atomic_numbers_from_pdb_organic,
        batched_coordinates_to_sim_state,
    )
    from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates

    # Confirm fixed [2,4] defaults used by dialanine-style rescale.
    assert PRC().fixed_row_sum_min == 2
    assert PRC().fixed_row_sum_max == 4

    device = torch.device("cpu")
    dtype = torch.float64
    base = load_sponge_coordinates(SPONGE, num_atoms=22)
    coords = torch.as_tensor(base, dtype=dtype, device=device).unsqueeze(0).repeat(2, 1, 1)
    z = atomic_numbers_from_pdb_organic(PDB)
    assert int((z == 20).sum()) == 0  # no bogus Ca from ASE "CA"
    mass = torch.ones(22, dtype=dtype)

    mace = load_mace_torch_sim_model(
        "off-small", device=device, dtype=dtype, compute_stress=False
    )
    cv = Torsion(TORSION)
    kernel = GaussianKernel(
        2,
        bandwidth=0.5,
        distance_function=torus_pairwise_distance,
        device=device,
        dtype=dtype,
        rescale_per_replica=True,
        rescale_per_replica_config=PerReplicaRescaleConfig(
            enabled=True,
            use_normalized_kernel=False,
            normalize_row_sum_by_replicas=True,
            fixed_row_sum_min=2.0,
            fixed_row_sum_max=4.0,
        ),
    )
    beta0 = 1.0 / (8.617330337217213e-5 * 300.0)
    bias = FPTBias(
        cv,
        kernel,
        kernel_grad_beta=0.1 * beta0,
        device=device,
        dtype=dtype,
        n_atoms_per_system=22,
    )
    model = SumModel(mace, bias)
    state = batched_coordinates_to_sim_state(
        coords, atomic_numbers=z, masses=mass, pbc=False
    )
    out = model(state)
    assert out["energy"].shape == (2,)
    assert out["forces"].shape == (44, 3)
    assert torch.isfinite(out["forces"]).all()
