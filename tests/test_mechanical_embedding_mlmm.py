"""Unit tests for mechanical embedding ML/MM."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fp_transformer_torch.potential.mechanical_embedding_MLMM import (
    KE_EV,
    MechanicalEmbeddingMLMM,
    MLMMTopology,
    _delta_mic,
    build_tip3p_only_topology,
    make_water_dimer_state,
    tip3p_epsilon_ev,
)


def _cpu():
    return torch.device("cpu")


@pytest.fixture
def water_dimer():
    state, topo = make_water_dimer_state(box=40.0, dtype=torch.float64, device=_cpu())
    return state, topo


def test_tip3p_dimer_lj_matches_analytic(water_dimer):
    state, topo = water_dimer
    pos = state.positions
    cell = torch.eye(3, dtype=torch.float64) * 40.0
    e_lj = 0.0
    sig, eps = topo.sigma, topo.epsilon
    for i in range(3):
        for j in range(3, 6):
            r = float(torch.linalg.norm(_delta_mic(pos[i], pos[j], cell)))
            sij = 0.5 * (float(sig[i]) + float(sig[j]))
            eij = math.sqrt(max(float(eps[i]) * float(eps[j]), 0.0))
            if sij > 0 and eij > 0:
                x = sij / r
                e_lj += 4.0 * eij * (x**12 - x**6)

    model = MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        enable_bonded=False,
        enable_pme=False,
        enable_ml_coulomb_correction=False,
        enable_lj=True,
        cutoff=18.0,
        device=_cpu(),
        dtype=torch.float64,
    )
    out = model(state)
    assert abs(float(out["energy"]) - e_lj) < 1e-10


def test_tip3p_dimer_pme_near_coulomb(water_dimer):
    state, topo = water_dimer
    pos = state.positions
    cell = torch.eye(3, dtype=torch.float64) * 40.0
    q = topo.charges
    e_c = 0.0
    for i in range(3):
        for j in range(3, 6):
            r = float(torch.linalg.norm(_delta_mic(pos[i], pos[j], cell)))
            e_c += KE_EV * float(q[i] * q[j]) / r

    model = MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        enable_bonded=False,
        enable_lj=False,
        enable_pme=True,
        enable_ml_coulomb_correction=False,
        cutoff=18.0,
        pme_accuracy=1e-6,
        device=_cpu(),
        dtype=torch.float64,
    )
    e_pme = float(model(state)["energy"])
    # Large box: PME ≈ direct intermolecular Coulomb (exclusions remove intramolecular).
    assert abs(e_pme - e_c) / max(abs(e_c), 1e-8) < 5e-3


def test_tip3p_dimer_dsf_matches_torchsim_and_excludes(water_dimer):
    """DSF path matches torch_sim; Amber exclusions change the energy."""
    pytest.importorskip("nvalchemiops")
    import torch_sim as ts
    from torch_sim.models.electrostatics import DSFCoulombModel

    state, topo = water_dimer
    device = _cpu()
    dtype = torch.float64
    cutoff = 18.0
    alpha = 0.2

    model = MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        enable_bonded=False,
        enable_lj=False,
        electrostatics="dsf",
        dsf_alpha=alpha,
        enable_ml_coulomb_correction=False,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
    )
    e_excl = float(model(state)["energy"])
    forces = model(state)["forces"]
    assert math.isfinite(e_excl)
    assert forces.shape == state.positions.shape
    assert torch.isfinite(forces).all()

    # Disable exclusion mask → should match bare DSFCoulombModel
    model._exclude_mat_dsf.zero_()
    e_full = float(model(state)["energy"])
    st = ts.SimState(
        positions=state.positions,
        masses=state.masses,
        cell=state.cell if hasattr(state, "cell") else state.row_vector_cell,
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers,
        partial_charges=topo.charges,
    )
    e_ref = float(
        DSFCoulombModel(
            cutoff=cutoff,
            alpha=alpha,
            device=device,
            dtype=dtype,
            compute_stress=False,
        )(st)["energy"]
    )
    assert abs(e_full - e_ref) / max(abs(e_ref), 1e-8) < 1e-6
    # TIP3P intramolecular Coulomb is large; exclusions must matter.
    assert abs(e_excl - e_full) > 1.0


def test_invalid_electrostatics_raises(water_dimer):
    _, topo = water_dimer
    with pytest.raises(ValueError, match="electrostatics"):
        MechanicalEmbeddingMLMM(
            topo,
            None,
            electrostatics="ewald",
            enable_mace=False,
            device=_cpu(),
            dtype=torch.float64,
        )


def test_pme_vs_ewald_consistency():
    pytest.importorskip("nvalchemiops")
    import torch_sim as ts
    from torch_sim.models.electrostatics import EwaldModel, PMEModel

    dtype = torch.float64
    device = _cpu()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [4.5, 0.1, -0.2], [1.0, 3.0, 2.0], [6.0, 5.0, 4.0]],
        dtype=dtype,
        device=device,
    )
    q = torch.tensor([0.5, -0.5, 0.3, -0.3], dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device) * 18.0
    state = ts.SimState(
        positions=pos,
        masses=torch.ones(4, dtype=dtype, device=device),
        cell=cell.unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1, 1, 1], device=device),
        partial_charges=q,
    )
    pme = PMEModel(
        cutoff=8.0, accuracy=1e-5, device=device, dtype=dtype, compute_stress=False
    )
    ewald = EwaldModel(
        cutoff=8.0, accuracy=1e-5, device=device, dtype=dtype, compute_stress=False
    )
    ep = float(pme(state)["energy"])
    ee = float(ewald(state)["energy"])
    assert abs(ep - ee) / max(abs(ee), 1e-8) < 1e-3


def test_energy_partition_all_mm_vs_all_ml(water_dimer):
    state, topo = water_dimer
    # All MM
    topo_mm = MLMMTopology(
        ml_mask=torch.zeros(topo.n_atoms, dtype=torch.bool),
        charges=topo.charges,
        sigma=topo.sigma,
        epsilon=topo.epsilon,
        atomic_numbers=topo.atomic_numbers,
        masses=topo.masses,
        bond_idx=topo.bond_idx,
        bond_k=topo.bond_k,
        bond_r0=topo.bond_r0,
        angle_idx=topo.angle_idx,
        angle_k=topo.angle_k,
        angle_theta0=topo.angle_theta0,
        exclude_i=topo.exclude_i,
        exclude_j=topo.exclude_j,
        cell=topo.cell,
    )
    mm = MechanicalEmbeddingMLMM(
        topo_mm,
        None,
        enable_mace=False,
        cutoff=12.0,
        device=_cpu(),
        dtype=torch.float64,
    )
    e_mm = float(mm(state)["energy"])
    assert np.isfinite(e_mm)

    # All ML → without MACE, classical leftover should vanish
    topo_ml = MLMMTopology(
        ml_mask=torch.ones(topo.n_atoms, dtype=torch.bool),
        charges=topo.charges,
        sigma=topo.sigma,
        epsilon=topo.epsilon,
        atomic_numbers=topo.atomic_numbers,
        masses=topo.masses,
        bond_idx=topo.bond_idx,
        bond_k=topo.bond_k,
        bond_r0=topo.bond_r0,
        angle_idx=topo.angle_idx,
        angle_k=topo.angle_k,
        angle_theta0=topo.angle_theta0,
        exclude_i=topo.exclude_i,
        exclude_j=topo.exclude_j,
        cell=topo.cell,
    )
    ml = MechanicalEmbeddingMLMM(
        topo_ml,
        None,
        enable_mace=False,
        cutoff=12.0,
        device=_cpu(),
        dtype=torch.float64,
    )
    e_ml = float(ml(state)["energy"])
    assert abs(e_ml) < 1e-6


def test_forces_finite_difference(water_dimer):
    state, topo = water_dimer
    model = MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        cutoff=12.0,
        pme_accuracy=1e-5,
        device=_cpu(),
        dtype=torch.float64,
    )
    out0 = model(state)
    forces = out0["forces"].detach().cpu().numpy()
    pos0 = state.positions.detach().clone()
    eps = 1e-4
    # Check a few random components
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(8):
        i = int(rng.integers(0, pos0.shape[0]))
        k = int(rng.integers(0, 3))
        e = []
        for s in (+1.0, -1.0):
            pos = pos0.clone()
            pos[i, k] = pos0[i, k] + s * eps
            import torch_sim as ts

            st = ts.SimState(
                positions=pos,
                masses=state.masses,
                cell=state.cell if hasattr(state, "cell") else state.row_vector_cell,
                pbc=state.pbc,
                atomic_numbers=state.atomic_numbers,
                partial_charges=topo.charges,
            )
            e.append(float(model(st)["energy"]))
        fd = -(e[0] - e[1]) / (2 * eps)
        errs.append(abs(fd - forces[i, k]))
    # PME + LJ FD tolerance
    assert float(np.median(errs)) < 5e-2


def test_topology_npz_roundtrip(tmp_path, water_dimer):
    _, topo = water_dimer
    path = tmp_path / "topo.npz"
    topo.save_npz(path)
    loaded = MLMMTopology.load_npz(path)
    assert loaded.n_atoms == topo.n_atoms
    assert torch.allclose(loaded.charges, topo.charges)
    assert torch.equal(loaded.exclude_i, topo.exclude_i)


def test_build_tip3p_only_counts():
    topo = build_tip3p_only_topology(5)
    assert topo.n_atoms == 15
    assert int(topo.bond_idx.shape[0]) == 10  # 2 bonds × 5
    assert int(topo.angle_idx.shape[0]) == 5
    assert int(topo.exclude_i.numel()) == 15  # 3 excl × 5


def _skip_torchpme():
    pytest.importorskip("torchpme")


def _repeat_state(state, n: int, topo):
    import torch_sim as ts

    pos = state.positions.repeat(n, 1)
    mass = state.masses.repeat(n)
    z = state.atomic_numbers.repeat(n)
    n_per = int(state.positions.shape[0])
    system_idx = torch.arange(n, device=pos.device).repeat_interleave(n_per)
    cell = state.row_vector_cell
    if cell.shape[0] == 1:
        cell = cell.expand(n, 3, 3).contiguous()
    return ts.SimState(
        positions=pos,
        masses=mass,
        cell=cell,
        pbc=True,
        atomic_numbers=z,
        system_idx=system_idx,
        partial_charges=topo.charges.repeat(n),
    )


def test_tip3p_dimer_torchpme_near_coulomb(water_dimer):
    _skip_torchpme()
    state, topo = water_dimer
    pos = state.positions
    cell = torch.eye(3, dtype=torch.float64) * 40.0
    q = topo.charges
    e_c = 0.0
    for i in range(3):
        for j in range(3, 6):
            r = float(torch.linalg.norm(_delta_mic(pos[i], pos[j], cell)))
            e_c += KE_EV * float(q[i] * q[j]) / r

    model = MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        enable_bonded=False,
        enable_lj=False,
        electrostatics="torchpme",
        enable_ml_coulomb_correction=False,
        cutoff=18.0,
        pme_mesh_spacing=1.5,
        device=_cpu(),
        dtype=torch.float64,
    )
    e_pme = float(model(state)["energy"])
    assert abs(e_pme - e_c) / max(abs(e_c), 1e-8) < 5e-3


def test_torchpme_vs_nvalchemiops_water_dimer(water_dimer):
    _skip_torchpme()
    pytest.importorskip("nvalchemiops")
    state, topo = water_dimer
    kwargs = dict(
        mace_model=None,
        enable_mace=False,
        enable_bonded=False,
        enable_lj=False,
        enable_ml_coulomb_correction=False,
        cutoff=12.0,
        pme_mesh_spacing=1.5,
        device=_cpu(),
        dtype=torch.float64,
        compute_forces=True,
    )
    ref = MechanicalEmbeddingMLMM(topo, electrostatics="pme", **kwargs)
    new = MechanicalEmbeddingMLMM(topo, electrostatics="torchpme", **kwargs)
    out_ref = ref(state)
    out_new = new(state)
    e_ref = float(out_ref["energy"])
    e_new = float(out_new["energy"])
    assert abs(e_new - e_ref) / max(abs(e_ref), 1e-8) < 5e-3
    f_ref = out_ref["forces"]
    f_new = out_new["forces"]
    denom = float(f_ref.norm().clamp_min(1e-8))
    assert float((f_new - f_ref).norm()) / denom < 5e-3


def test_torchpme_batched_replicas_match(water_dimer):
    _skip_torchpme()
    state, topo = water_dimer
    model = MechanicalEmbeddingMLMM(
        topo,
        None,
        enable_mace=False,
        electrostatics="torchpme",
        enable_ml_coulomb_correction=False,
        cutoff=12.0,
        device=_cpu(),
        dtype=torch.float64,
    )
    out1 = model(state)
    state2 = _repeat_state(state, 2, topo)
    out2 = model(state2)
    assert out2["energy"].shape == (2,)
    assert torch.allclose(out2["energy"], out1["energy"].expand(2), atol=1e-6, rtol=1e-5)
    f1 = out1["forces"]
    f2 = out2["forces"].view(2, -1, 3)
    assert torch.allclose(f2[0], f1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(f2[1], f1, atol=1e-5, rtol=1e-5)


def test_torchpme_batched_matches_native_calculator(water_dimer):
    """Batched k-space must match per-system torchpme.Calculator.forward."""
    _skip_torchpme()
    from fp_transformer_torch.potential.torch_pme_model import (
        build_torchpme_calculator,
        torchpme_energy_forces,
        warp_free_neighbor_list,
    )

    state, topo = water_dimer
    n_rep = 3
    state_b = _repeat_state(state, n_rep, topo)
    cutoff = 12.0
    calc = build_torchpme_calculator(
        cutoff=cutoff,
        mesh_spacing=1.5,
        method="p3m",
        dtype=torch.float64,
        device=_cpu(),
        full_neighbor_list=False,
    )
    e_batch, f_batch = torchpme_energy_forces(
        state_b.positions,
        topo.charges.repeat(n_rep),
        state_b.row_vector_cell,
        state_b.system_idx,
        calc,
        cutoff=cutoff,
        compute_forces=True,
        nl_backend="pytorch",
        full_neighbor_list=False,
    )

    n_per = int(state.positions.shape[0])
    cell = state.row_vector_cell[0]
    q1 = topo.charges.reshape(-1, 1)
    periodic = torch.ones(3, dtype=torch.bool)
    e_ref = []
    f_ref = []
    for b in range(n_rep):
        pos = state_b.positions[b * n_per : (b + 1) * n_per].detach().requires_grad_(True)
        system_idx = torch.zeros(n_per, dtype=torch.long)
        edge_index, _, unit_shifts = warp_free_neighbor_list(
            pos.detach(),
            cell.unsqueeze(0),
            True,
            cutoff,
            system_idx,
            half=True,
            backend="pytorch",
        )
        from fp_transformer_torch.potential.torch_pme_model import _pair_distances

        dist = _pair_distances(
            pos, edge_index[0], edge_index[1], unit_shifts, cell, ortho_box=torch.diag(cell)
        )
        V = calc(q1, cell, pos, edge_index.t().contiguous(), dist, periodic)
        e = (q1 * V).sum()
        (grad,) = torch.autograd.grad(e, pos)
        e_ref.append(e.detach())
        f_ref.append((-grad).detach())
    e_ref = torch.stack(e_ref)
    f_ref = torch.cat(f_ref, dim=0)
    assert torch.allclose(e_batch, e_ref, atol=1e-8, rtol=1e-7)
    assert torch.allclose(f_batch, f_ref, atol=1e-7, rtol=1e-6)


def test_torchpme_warp_free_nl_matches_vesin(water_dimer):
    _skip_torchpme()
    pytest.importorskip("vesin")
    from fp_transformer_torch.potential.torch_pme_model import (
        pytorch_cutoff_neighbor_list,
        vesin_cutoff_neighbor_list,
    )

    state, _topo = water_dimer
    n = int(state.positions.shape[0])
    system_idx = torch.zeros(n, dtype=torch.long)
    kwargs = dict(
        positions=state.positions,
        cell=state.row_vector_cell,
        pbc=True,
        cutoff=12.0,
        system_idx=system_idx,
        half=True,
    )
    ei_t, _, sh_t = pytorch_cutoff_neighbor_list(**kwargs)
    ei_v, _, sh_v = vesin_cutoff_neighbor_list(**kwargs)

    def _pair_set(ei, sh):
        pairs = torch.stack((ei[0], ei[1], sh[:, 0], sh[:, 1], sh[:, 2]), dim=1)
        return {tuple(int(x) for x in row) for row in pairs.tolist()}

    assert _pair_set(ei_t, sh_t) == _pair_set(ei_v, sh_v)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs >= 2 CUDA devices",
)
def test_torchpme_shard_matches_single_gpu(water_dimer):
    _skip_torchpme()
    state, topo = water_dimer
    device0 = torch.device("cuda:0")
    topo0 = topo.to(device0, torch.float64)
    import torch_sim as ts

    n_rep = 4
    pos = state.positions.to(device0).repeat(n_rep, 1)
    mass = state.masses.to(device0).repeat(n_rep)
    z = state.atomic_numbers.to(device0).repeat(n_rep)
    n_per = int(state.positions.shape[0])
    system_idx = torch.arange(n_rep, device=device0).repeat_interleave(n_per)
    cell = (
        state.row_vector_cell.to(device0)
        .expand(n_rep, 3, 3)
        .contiguous()
    )
    batched = ts.SimState(
        positions=pos,
        masses=mass,
        cell=cell,
        pbc=True,
        atomic_numbers=z,
        system_idx=system_idx,
        partial_charges=topo0.charges.repeat(n_rep),
    )
    common = dict(
        enable_mace=False,
        electrostatics="torchpme",
        enable_ml_coulomb_correction=False,
        cutoff=12.0,
        dtype=torch.float64,
        compute_forces=True,
    )
    single = MechanicalEmbeddingMLMM(topo0, None, device=device0, **common)
    sharded = MechanicalEmbeddingMLMM(
        topo0,
        None,
        device=device0,
        mm_devices=["cuda:0", "cuda:1"],
        **common,
    )
    out_s = single(batched)
    out_m = sharded(batched)
    assert torch.allclose(out_s["energy"], out_m["energy"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_s["forces"], out_m["forces"], atol=1e-5, rtol=1e-5)


def test_compact_torsion_matches_full_molecule():
    from fp_transformer_torch.collective_variable.torsion import Torsion
    from fp_transformer_torch.integrator.sharded_baoab import compact_torsion, cv_atom_indices

    idx = torch.tensor([[4, 6, 8, 14], [6, 8, 14, 16]], dtype=torch.long)
    cv = Torsion(idx)
    cv_idx = cv_atom_indices(cv)
    compact = compact_torsion(cv, cv_idx)
    pos = torch.randn(5, 22, 3, dtype=torch.float64)
    full = cv(pos)
    sub = compact(pos[:, cv_idx, :])
    assert torch.allclose(full, sub, atol=1e-10, rtol=1e-10)


def test_wrap_to_box_keeps_centered_solute_intact():
    from fp_transformer_torch.collective_variable.torsion import Torsion
    from fp_transformer_torch.integrator.sharded_baoab import (
        compact_torsion,
        cv_atom_indices,
        unwrap_to_ref,
        wrap_to_box,
    )

    box = torch.tensor([18.0, 18.0, 18.0], dtype=torch.float64)
    idx = torch.tensor([[4, 6, 8, 14], [6, 8, 14, 16]], dtype=torch.long)
    cv = Torsion(idx)
    compact = compact_torsion(cv, cv_atom_indices(cv))
    pos = torch.zeros(1, 22, 3, dtype=torch.float64)
    pos[0] = torch.tensor([9.0, 8.2, 8.6])
    pos[0, 4] = torch.tensor([7.2, 9.1, 7.4])
    pos[0, 6] = torch.tensor([8.4, 8.0, 8.8])
    pos[0, 8] = torch.tensor([9.7, 7.2, 9.5])
    pos[0, 14] = torch.tensor([10.8, 8.6, 10.1])
    pos[0, 16] = torch.tensor([11.4, 9.8, 9.0])
    cv_idx = cv_atom_indices(cv)
    expected = compact(pos[:, cv_idx, :])
    assert expected.abs().max() > 0.2

    round_wrapped = pos - box * torch.round(pos / box)
    assert (round_wrapped[:, cv_idx, 0] < 0).any()
    assert not torch.allclose(compact(round_wrapped[:, cv_idx, :]), expected, atol=1e-5)

    wrapped = wrap_to_box(pos, box)
    assert torch.allclose(wrapped, pos)
    assert torch.allclose(compact(wrapped[:, cv_idx, :]), expected, atol=1e-10)

    split = wrap_to_box(pos + torch.tensor([10.0, 0.0, 0.0]), box)
    assert (split[:, cv_idx, 0] < 5).any() and (split[:, cv_idx, 0] > 13).any()
    recovered = unwrap_to_ref(split[:, cv_idx, :], box)
    assert torch.allclose(compact(recovered), expected, atol=1e-10)


def test_compact_torsion_pullback_matches_full():
    from fp_transformer_torch.collective_variable.torsion import Torsion
    from fp_transformer_torch.integrator.sharded_baoab import compact_torsion, cv_atom_indices
    from fp_transformer_torch.math.pullback import pullback_collective_variable_gradient

    idx = torch.tensor([[4, 6, 8, 14], [6, 8, 14, 16]], dtype=torch.long)
    cv = Torsion(idx)
    cv_idx = cv_atom_indices(cv)
    compact = compact_torsion(cv, cv_idx)
    pos = torch.randn(8, 22, 3, dtype=torch.float64)
    pos_full = pos.detach().requires_grad_(True)
    pos_sub = pos[:, cv_idx, :].detach().requires_grad_(True)
    cot = torch.randn(8, 2, 1, dtype=torch.float64)
    g_full = pullback_collective_variable_gradient(pos_full, cv, cot)
    g_sub = pullback_collective_variable_gradient(pos_sub, compact, cot)
    assert torch.allclose(g_full[:, cv_idx, :], g_sub, atol=1e-10, rtol=1e-10)
    mask = torch.ones(22, dtype=torch.bool)
    mask[cv_idx] = False
    assert torch.allclose(g_full[:, mask, :], torch.zeros_like(g_full[:, mask, :]))


def test_cv_force_index_add_writes_back():
    """Advanced-index add_ is a no-op; index_add_ must land on the source tensor."""
    n_local, n_per, n_cv = 3, 10, 2
    forces = torch.zeros(n_local * n_per, 3)
    extra = torch.arange(n_local * n_cv * 3, dtype=torch.float32).reshape(n_local, n_cv, 3)
    idx = torch.tensor([1, 4])
    view = forces.view(n_local, n_per, 3)
    lost = view.clone()
    lost[:, idx, :].add_(extra)
    assert torch.equal(lost, torch.zeros_like(lost))
    view.index_add_(1, idx, extra)
    assert torch.allclose(view[:, idx, :], extra)
    assert not torch.equal(forces, torch.zeros_like(forces))
