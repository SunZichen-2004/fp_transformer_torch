"""Warp-free batched PME/P3M via lab-cosmo ``torchpme`` (metal units: eV / Å).

Replica systems share a cell and atom count, so real-space pairs and the
reciprocal FFT are evaluated as one ``(B, …)`` batch (not a Python loop).
Multi-GPU work is replica sharding on CUDA streams in the caller
(``MechanicalEmbeddingMLMM``), not ``parallel_apply``.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import torchpme
except ImportError:  # pragma: no cover
    torchpme = None  # type: ignore[assignment]


EV_A = 14.399645478425667  # torchpme.prefactors.eV_A; matches MLMM KE_EV


def torchpme_available() -> bool:
    return torchpme is not None


def _require_torchpme() -> Any:
    if torchpme is None:
        raise ImportError(
            "torchpme is required for electrostatics='torchpme'. "
            "Install with: pip install torch-pme"
        )
    return torchpme


def build_torchpme_calculator(
    *,
    cutoff: float,
    mesh_spacing: float | None = 1.5,
    smearing: float | None = None,
    method: str = "p3m",
    interpolation_nodes: int = 4,
    full_neighbor_list: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
):
    """Build a ``P3MCalculator`` (default) or ``PMECalculator`` in eV·Å units."""
    tp = _require_torchpme()
    method = str(method).strip().lower()
    if method not in ("p3m", "pme"):
        raise ValueError(f"torchpme method must be 'p3m' or 'pme', got {method!r}")
    cut = float(cutoff)
    smear = float(smearing) if smearing is not None else cut / 5.0
    if smear <= 0.0:
        raise ValueError(f"smearing must be positive, got {smear}")
    spacing = float(mesh_spacing) if mesh_spacing is not None else 0.5 * smear
    if spacing <= 0.0:
        raise ValueError(f"mesh_spacing must be positive, got {spacing}")

    prefactor = float(getattr(tp.prefactors, "eV_A", EV_A))
    potential = tp.CoulombPotential(smearing=smear, prefactor=prefactor)
    cls = tp.P3MCalculator if method == "p3m" else tp.PMECalculator
    calc = cls(
        potential,
        mesh_spacing=spacing,
        interpolation_nodes=int(interpolation_nodes),
        full_neighbor_list=bool(full_neighbor_list),
    )
    if device is not None:
        calc = calc.to(device=torch.device(device), dtype=dtype)
    else:
        calc = calc.to(dtype=dtype)
    return calc


def _pbc_any(pbc) -> bool:
    if torch.is_tensor(pbc):
        return pbc.numel() > 0
    if isinstance(pbc, (tuple, list)):
        return any(bool(x) for x in pbc)
    return bool(pbc)


def _cell_rows(cell: torch.Tensor, n_systems: int) -> torch.Tensor:
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    if cell.shape[0] == 1 and n_systems > 1:
        return cell.expand(n_systems, 3, 3).contiguous()
    if int(cell.shape[0]) != n_systems:
        raise ValueError(
            f"cell batch {int(cell.shape[0])} != n_systems {n_systems}"
        )
    return cell


def _is_ortho_cell(cell: torch.Tensor) -> bool:
    c = cell[0] if cell.ndim == 3 else cell
    return bool(torch.allclose(c, torch.diag(torch.diag(c)), atol=1e-8))


def pytorch_cutoff_neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
    system_idx: torch.Tensor,
    *,
    half: bool = True,
    n_systems: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched MIC neighbor list matching ``torchsim_nl`` return layout.

    Returns
    -------
    edge_index
        ``(2, E)`` global atom indices.
    mapping
        ``(E,)`` system index of each pair.
    unit_shifts
        ``(E, 3)`` integer lattice translations such that
        ``r_i - r_j - unit_shifts @ cell`` is the minimum image.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3), got {tuple(positions.shape)}")
    n_atoms = int(positions.shape[0])
    if n_systems is None:
        n_systems = int(system_idx.max().item()) + 1 if n_atoms else 0
    if n_systems <= 0:
        empty_e = positions.new_empty(2, 0, dtype=torch.long)
        empty_m = positions.new_empty(0, dtype=torch.long)
        empty_s = positions.new_empty(0, 3, dtype=torch.long)
        return empty_e, empty_m, empty_s
    if n_atoms % n_systems != 0:
        raise ValueError("unequal atoms per system")
    n_per = n_atoms // n_systems
    if not _pbc_any(pbc):
        raise ValueError("pytorch_cutoff_neighbor_list requires PBC")

    cell_b = _cell_rows(cell.to(device=positions.device, dtype=positions.dtype), n_systems)
    pos = positions.view(n_systems, n_per, 3)
    cutoff_sq = float(cutoff) * float(cutoff)
    # Chunk along replica dim to cap (B, N, N, 3) memory.
    elem = max(int(positions.element_size()), 1)
    max_b = max(1, int(1_200_000_000 / max(n_per * n_per * 3 * elem, 1)))
    max_b = min(n_systems, max(1, max_b))

    edges_i: list[torch.Tensor] = []
    edges_j: list[torch.Tensor] = []
    maps: list[torch.Tensor] = []
    shifts: list[torch.Tensor] = []
    ortho = _is_ortho_cell(cell_b)
    eye = torch.eye(n_per, dtype=torch.bool, device=positions.device)

    for lo in range(0, n_systems, max_b):
        hi = min(n_systems, lo + max_b)
        p = pos[lo:hi]
        b = hi - lo
        disp = p[:, :, None, :] - p[:, None, :, :]
        c0 = cell_b[lo]
        if ortho:
            box = torch.diag(c0)
            raw_shift = torch.round(disp / box.clamp_min(1e-12))
            disp = disp - raw_shift * box
        else:
            inv = torch.linalg.inv(c0)
            frac = disp @ inv
            raw_shift = torch.round(frac)
            disp = (frac - raw_shift) @ c0
        r2 = (disp * disp).sum(dim=-1)
        keep = r2 < cutoff_sq
        if half:
            keep = keep & torch.triu(torch.ones_like(eye), diagonal=1)
        else:
            keep = keep & ~eye
        keep = keep.expand(b, n_per, n_per)
        bb, ii, jj = torch.where(keep)
        edges_i.append(bb.to(torch.long) * n_per + ii + lo * n_per)
        edges_j.append(bb.to(torch.long) * n_per + jj + lo * n_per)
        maps.append(bb.to(torch.long) + lo)
        shifts.append(raw_shift[bb, ii, jj].to(dtype=torch.long))

    if not edges_i:
        empty_e = positions.new_empty(2, 0, dtype=torch.long)
        empty_m = positions.new_empty(0, dtype=torch.long)
        empty_s = positions.new_empty(0, 3, dtype=torch.long)
        return empty_e, empty_m, empty_s
    i = torch.cat(edges_i)
    j = torch.cat(edges_j)
    mapping = torch.cat(maps)
    unit_shifts = torch.cat(shifts, dim=0)
    edge_index = torch.stack((i, j), dim=0)
    return edge_index, mapping, unit_shifts


def vesin_cutoff_neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
    system_idx: torch.Tensor,
    *,
    half: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-system ``vesin.NeighborList`` (CPU). Same return layout as PyTorch NL."""
    try:
        from vesin import NeighborList
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vesin is required for nl_backend='vesin'") from exc
    if not _pbc_any(pbc):
        raise ValueError("vesin_cutoff_neighbor_list requires PBC")

    n_atoms = int(positions.shape[0])
    n_systems = int(system_idx.max().item()) + 1 if n_atoms else 0
    n_per = n_atoms // max(n_systems, 1)
    cell_b = _cell_rows(cell.detach().cpu().to(dtype=torch.float64), n_systems)
    pos_cpu = positions.detach().cpu().to(dtype=torch.float64).view(n_systems, n_per, 3)
    nl = NeighborList(cutoff=float(cutoff), full_list=not half)
    e_i: list[torch.Tensor] = []
    e_j: list[torch.Tensor] = []
    maps: list[torch.Tensor] = []
    shs: list[torch.Tensor] = []
    for b in range(n_systems):
        pts = pos_cpu[b].numpy()
        box = cell_b[b].numpy()
        ii, jj, ss = nl.compute(pts, box, True, quantities="ijS")
        if len(ii) == 0:
            continue
        off = b * n_per
        e_i.append(torch.as_tensor(ii, dtype=torch.long) + off)
        e_j.append(torch.as_tensor(jj, dtype=torch.long) + off)
        maps.append(torch.full((int(len(ii)),), b, dtype=torch.long))
        shs.append(torch.as_tensor(ss, dtype=torch.long))
    device = positions.device
    if not e_i:
        return (
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 3, dtype=torch.long, device=device),
        )
    edge_index = torch.stack((torch.cat(e_i), torch.cat(e_j)), dim=0).to(device=device)
    mapping = torch.cat(maps).to(device=device)
    unit_shifts = torch.cat(shs, dim=0).to(device=device)
    return edge_index, mapping, unit_shifts


def warp_free_neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
    system_idx: torch.Tensor,
    *,
    half: bool = True,
    backend: str = "pytorch",
    n_systems: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = str(backend).strip().lower()
    if mode == "vesin":
        return vesin_cutoff_neighbor_list(
            positions, cell, pbc, cutoff, system_idx, half=half
        )
    if mode in ("pytorch", "torch", "auto"):
        return pytorch_cutoff_neighbor_list(
            positions, cell, pbc, cutoff, system_idx, half=half, n_systems=n_systems
        )
    raise ValueError(f"unknown nl backend {backend!r}")


def warmup_torchpme_device(device: torch.device | str, dtype: torch.dtype) -> None:
    """Prime ``torch.linalg`` / FFT lazy wrappers on ``device`` (main thread)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    with torch.cuda.device(dev):
        eye = torch.eye(3, device=dev, dtype=dtype)
        torch.linalg.inv_ex(eye)
        torch.linalg.inv(eye)
        torch.linalg.det(eye)
        torch.linalg.norm(eye, dim=-1)
        dummy = torch.zeros(2, 1, 8, 8, 8, device=dev, dtype=dtype)
        hat = torch.fft.rfftn(dummy, dim=(-3, -2, -1))
        torch.fft.irfftn(hat, s=(8, 8, 8), dim=(-3, -2, -1))


def _pair_distances(
    positions: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: torch.Tensor,
    *,
    ortho_box: torch.Tensor | None,
) -> torch.Tensor:
    if i.numel() == 0:
        return positions.new_zeros(0)
    if ortho_box is not None:
        shift_vec = unit_shifts.to(dtype=positions.dtype) * ortho_box.unsqueeze(0)
    else:
        c = cell[0] if cell.ndim == 3 else cell
        shift_vec = unit_shifts.to(dtype=positions.dtype) @ c
    disp = positions[i] - positions[j] - shift_vec
    return torch.linalg.norm(disp, dim=-1).clamp_min(1e-12)


def _ensure_kspace(calculator, cell: torch.Tensor) -> None:
    if getattr(calculator, "_fpt_kspace_ready", False):
        return
    from torchpme.lib.kvectors import get_ns_mesh

    ns = get_ns_mesh(cell, calculator.mesh_spacing)
    calculator.mesh_interpolator.update(cell, ns)
    calculator.kspace_filter.update(cell, ns)
    calculator._fpt_ns = (
        int(ns[0].item()),
        int(ns[1].item()),
        int(ns[2].item()),
    )
    calculator._fpt_kspace_ready = True


def _kspace_potential_batched(
    calculator,
    charges: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Reciprocal-space ``V_i`` for ``B`` identical cells.

    ``charges`` / ``positions`` are ``(B, N, C)`` / ``(B, N, 3)``.
    Returns ``(B * N, C)`` (same 1/2 convention as ``PMECalculator._compute_kspace``).
    Charge is smeared onto a ``(B, C, nx, ny, nz)`` mesh and FFT'd once.
    """
    batch, n_per, n_channels = charges.shape
    pos_flat = positions.reshape(batch * n_per, 3)
    q_flat = charges.reshape(batch * n_per, n_channels)
    _ensure_kspace(calculator, cell)
    interp = calculator.mesh_interpolator
    filt = calculator.kspace_filter
    interp.compute_weights(pos_flat)

    nx, ny, nz = calculator._fpt_ns
    device = pos_flat.device
    b_idx = torch.arange(batch, device=device, dtype=torch.long).repeat_interleave(n_per)
    wprod = (
        interp.interpolation_weights[interp.x_shifts, :, 0]
        * interp.interpolation_weights[interp.y_shifts, :, 1]
        * interp.interpolation_weights[interp.z_shifts, :, 2]
    )
    b_exp = b_idx.unsqueeze(0).expand_as(interp.x_indices)

    rho = pos_flat.new_zeros(batch, n_channels, nx, ny, nz)
    for ch in range(n_channels):
        rho[:, ch].index_put_(
            (b_exp, interp.x_indices, interp.y_indices, interp.z_indices),
            q_flat[:, ch] * wprod,
            accumulate=True,
        )

    mesh_hat = torch.fft.rfftn(rho, norm=filt._fft_norm, dim=(-3, -2, -1))
    pot_mesh = torch.fft.irfftn(
        mesh_hat * filt._kfilter,
        s=(nx, ny, nz),
        norm=filt._ifft_norm,
        dim=(-3, -2, -1),
    )

    interpolated = q_flat.new_empty(q_flat.shape)
    mesh_ch = pot_mesh.unbind(dim=1)
    for ch, mesh_b in enumerate(mesh_ch):
        sampled = mesh_b[b_exp, interp.x_indices, interp.y_indices, interp.z_indices]
        interpolated[:, ch] = (sampled * wprod).sum(dim=0)

    ivolume = torch.abs(torch.linalg.det(cell)).pow(-1)
    interpolated = interpolated * ivolume
    interpolated = interpolated - q_flat * calculator.potential.self_contribution()
    q_tot = q_flat.view(batch, n_per, n_channels).sum(dim=1)
    interpolated = interpolated - (
        2 * calculator.potential.background_correction() * q_tot[b_idx] * ivolume
    )
    return interpolated / 2


def torchpme_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    system_idx: torch.Tensor,
    calculator,
    *,
    cutoff: float,
    compute_forces: bool,
    edge_index: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    mapping: torch.Tensor | None = None,
    ortho_box: torch.Tensor | None = None,
    pbc=True,
    nl_backend: str = "pytorch",
    full_neighbor_list: bool = False,
    n_systems: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batched torchpme energy ``(n_systems,)`` and optional forces.

    Real space uses the full neighbor list at once; k-space is one batched FFT
    over replicas. ``V_i`` already contains the Coulomb 1/2, so
    ``E = sum_i q_i V_i``.
    """
    del mapping
    n_atoms = int(positions.shape[0])
    if n_systems is None:
        n_systems = int(system_idx.max().item()) + 1 if n_atoms else 0
    dtype = positions.dtype
    device = positions.device
    if n_systems == 0:
        z = torch.zeros(0, device=device, dtype=dtype)
        return z, (torch.zeros_like(positions) if compute_forces else None)
    if n_atoms % n_systems != 0:
        raise ValueError("unequal atoms per system")
    n_per = n_atoms // n_systems
    cell_b = _cell_rows(cell.to(device=device, dtype=dtype), n_systems)
    q = charges.reshape(-1, 1).to(device=device, dtype=dtype)
    if int(q.shape[0]) != n_atoms:
        raise ValueError(f"charges length {q.shape[0]} != n_atoms {n_atoms}")

    pos = positions
    if compute_forces:
        pos = positions.detach().requires_grad_(True)

    if edge_index is None or unit_shifts is None:
        edge_index, _, unit_shifts = warp_free_neighbor_list(
            pos.detach(),
            cell_b,
            pbc,
            cutoff,
            system_idx,
            half=not full_neighbor_list,
            backend=nl_backend,
            n_systems=n_systems,
        )

    box = ortho_box
    if box is None and _is_ortho_cell(cell_b):
        box = torch.diag(cell_b[0])

    smear = getattr(getattr(calculator, "potential", None), "smearing", None)
    if torch.is_tensor(smear) and (smear.device != device or smear.dtype != dtype):
        calculator = calculator.to(device=device, dtype=dtype)

    dist = _pair_distances(
        pos, edge_index[0], edge_index[1], unit_shifts, cell_b[0], ortho_box=box
    )
    neighbor_indices = edge_index.t().contiguous()
    v_sr = calculator._compute_rspace(q, neighbor_indices, dist)
    v_lr = _kspace_potential_batched(
        calculator,
        q.view(n_systems, n_per, 1),
        pos.view(n_systems, n_per, 3),
        cell_b[0],
    )
    potential = v_sr + v_lr
    e_atom = (q * potential).sum(dim=-1)
    energy = torch.zeros(n_systems, device=device, dtype=dtype)
    energy.scatter_add_(0, system_idx.long(), e_atom)

    forces = None
    if compute_forces:
        (grad,) = torch.autograd.grad(energy.sum(), pos, retain_graph=False)
        forces = (-grad).detach()
        energy = energy.detach()
    else:
        energy = energy.detach()
    return energy, forces
