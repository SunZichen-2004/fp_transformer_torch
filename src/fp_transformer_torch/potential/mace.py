"""MACE foundation-model wrappers for torch-sim / EnergySystem."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from .base import Potential

# Prefer TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD for older MACE checkpoints.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

DEFAULT_MACE_CACHE = Path.home() / ".cache" / "mace"

# ASE maps PDB atom name "CA" → Calcium; for peptides use first-letter organic rule.
_ORGANIC_TWO_LETTER = {
    "CL": 17,
    "BR": 35,
    "NA": 11,
    "MG": 12,
    "ZN": 30,
    "FE": 26,
}
_ORGANIC_ONE_LETTER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "I": 53,
}


def atomic_numbers_from_pdb_organic(path: Path | str) -> torch.Tensor:
    """Parse atomic numbers from a PDB using organic / peptide atom-name rules.

    Unlike ASE, ``CA`` / ``CB`` / … are treated as carbon (not calcium).
    """
    path = Path(path)
    zs: list[int] = []
    for line in path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        # Xponge / whitespace PDB: ATOM serial name ...
        # Standard fixed-width: cols 13-16 name, 77-78 element.
        name = parts[2]
        if len(line) >= 78 and line[76:78].strip():
            elem = line[76:78].strip().upper()
        else:
            elem = name.strip().upper()
        if len(elem) >= 2 and elem[:2] in _ORGANIC_TWO_LETTER:
            zs.append(_ORGANIC_TWO_LETTER[elem[:2]])
        elif elem[:1] in _ORGANIC_ONE_LETTER:
            zs.append(_ORGANIC_ONE_LETTER[elem[:1]])
        else:
            raise ValueError(f"cannot map PDB atom name {name!r} to element in {path}")
    if not zs:
        raise ValueError(f"no ATOM/HETATM records in {path}")
    return torch.as_tensor(zs, dtype=torch.long)

MACE_MODELS: dict[str, dict[str, Any]] = {
    "off-small": {
        "path": DEFAULT_MACE_CACHE / "MACE-OFF23_small.model",
        "head": None,
        "description": "MACE-OFF23 small (organic / vacuum)",
    },
    "off-medium": {
        "path": DEFAULT_MACE_CACHE / "MACE-OFF23_medium.model",
        "head": None,
        "description": "MACE-OFF23 medium",
    },
    "off-large": {
        "path": DEFAULT_MACE_CACHE / "MACE-OFF23_large.model",
        "head": None,
        "description": "MACE-OFF23 large",
    },
    "mh-0": {
        "path": DEFAULT_MACE_CACHE / "macemh0model",
        "head": "omol",
        "description": "MACE-MH-0 (omol head)",
    },
}


def resolve_mace_checkpoint(
    model: str | Path,
    *,
    cache_dir: Path | str | None = None,
) -> tuple[Path, str | None]:
    """Resolve ``(checkpoint_path, head)`` for a named model or raw path.

    Named keys: ``off-small``, ``off-medium``, ``off-large``, ``mh-0``.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_MACE_CACHE
    key = str(model).strip().lower().replace("_", "-")
    # aliases
    aliases = {
        "off": "off-small",
        "mace-off": "off-small",
        "mace-off-small": "off-small",
        "mace-off23-small": "off-small",
        "mace-mh-0": "mh-0",
        "macemh0": "mh-0",
        "mh0": "mh-0",
    }
    key = aliases.get(key, key)

    if key in MACE_MODELS:
        meta = MACE_MODELS[key]
        path = Path(meta["path"])
        if cache_dir is not None:
            path = cache / path.name
        if not path.is_file():
            raise FileNotFoundError(
                f"MACE checkpoint missing for '{key}': {path}. "
                f"Expected under {cache}."
            )
        return path, meta["head"]

    path = Path(os.path.expanduser(str(model)))
    if not path.is_file():
        raise FileNotFoundError(f"MACE checkpoint not found: {path}")
    # Infer head for mh-0 style filenames.
    head = "omol" if "mh0" in path.name.lower().replace("-", "").replace("_", "") else None
    return path, head


def load_mace_torch_sim_model(
    model: str | Path = "off-small",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    compute_forces: bool = True,
    compute_stress: bool = False,
    head: str | None = ...,  # type: ignore[assignment]
    cache_dir: Path | str | None = None,
    **kwargs: Any,
):
    """Load a ``torch_sim``-compatible ``MaceModel`` (OFF / MH-0).

    Args:
        model: Named key (``off-small``, ``mh-0``, ...) or checkpoint path.
        head: Override head. Default ``...`` means use the named-model default
            (``None`` for OFF, ``\"omol\"`` for MH-0). Pass ``None`` explicitly
            to disable.
        cache_dir: Optional override for ``~/.cache/mace``.
    """
    try:
        from torch_sim.models.mace import MaceModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch_sim / mace are required. Activate conda env MACE_torchsim."
        ) from exc

    path, default_head = resolve_mace_checkpoint(model, cache_dir=cache_dir)
    resolved_head = default_head if head is ... else head
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    raw = torch.load(str(path), map_location=device, weights_only=False)
    return MaceModel(
        model=raw,
        device=device,
        dtype=dtype,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
        head=resolved_head,
        **kwargs,
    )


def batched_coordinates_to_sim_state(
    coordinate: torch.Tensor,
    *,
    atomic_numbers: torch.Tensor,
    masses: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    pbc: bool = False,
    charge: float | torch.Tensor = 0.0,
    spin: float | torch.Tensor = 1.0,
):
    """Build a batched ``SimState`` from coordinates shaped ``(B, N, 3)``."""
    import torch_sim as ts

    if coordinate.ndim != 3 or coordinate.shape[-1] != 3:
        raise ValueError(f"coordinate must be (B, N, 3), got {tuple(coordinate.shape)}")
    batch, num_atoms, _ = coordinate.shape
    device, dtype = coordinate.device, coordinate.dtype

    positions = coordinate.reshape(batch * num_atoms, 3)
    z = torch.as_tensor(atomic_numbers, device=device, dtype=torch.long).reshape(-1)
    if z.numel() == num_atoms:
        z = z.repeat(batch)
    elif z.numel() != batch * num_atoms:
        raise ValueError(
            f"atomic_numbers length {z.numel()} incompatible with B={batch}, N={num_atoms}"
        )

    if masses is None:
        mass = torch.ones(batch * num_atoms, device=device, dtype=dtype)
    else:
        mass = torch.as_tensor(masses, device=device, dtype=dtype).reshape(-1)
        if mass.numel() == num_atoms:
            mass = mass.repeat(batch)
        elif mass.numel() != batch * num_atoms:
            raise ValueError(
                f"masses length {mass.numel()} incompatible with B={batch}, N={num_atoms}"
            )

    if cell is None:
        box = 50.0
        cell = torch.eye(3, device=device, dtype=dtype) * box
        cell = cell.unsqueeze(0).expand(batch, 3, 3).contiguous()
    else:
        cell = torch.as_tensor(cell, device=device, dtype=dtype)
        if cell.shape == (3, 3):
            cell = cell.unsqueeze(0).expand(batch, 3, 3).contiguous()
        elif cell.shape != (batch, 3, 3):
            raise ValueError(f"cell must be (3,3) or (B,3,3), got {tuple(cell.shape)}")

    system_idx = torch.arange(batch, device=device).repeat_interleave(num_atoms)

    if isinstance(charge, (int, float)):
        total_charge = torch.full((batch,), float(charge), device=device, dtype=dtype)
    else:
        total_charge = torch.as_tensor(charge, device=device, dtype=dtype).reshape(batch)

    if isinstance(spin, (int, float)):
        total_spin = torch.full((batch,), float(spin), device=device, dtype=dtype)
    else:
        total_spin = torch.as_tensor(spin, device=device, dtype=dtype).reshape(batch)

    return ts.SimState(
        positions=positions,
        masses=mass,
        cell=cell,
        pbc=pbc,
        atomic_numbers=z,
        system_idx=system_idx,
        total_charge=total_charge,
        total_spin=total_spin,
    )


def sim_state_to_batched_coordinates(state) -> torch.Tensor:
    """``SimState`` positions → ``(n_systems, n_atoms_per_system, 3)``.

    Requires equal atom counts across systems.
    """
    n_systems = int(state.n_systems)
    n_atoms = int(state.n_atoms)
    if n_atoms % n_systems != 0:
        raise ValueError(
            f"unequal atoms per system: n_atoms={n_atoms}, n_systems={n_systems}"
        )
    n_per = n_atoms // n_systems
    return state.positions.view(n_systems, n_per, 3)


class MacePotential(Potential):
    """``Potential`` adapter around a torch-sim ``MaceModel``.

    Energies / forces are in **metal units** (eV, eV/Å). Coordinates in Å.
    """

    def __init__(
        self,
        model: str | Path | Any = "off-small",
        *,
        atomic_numbers: torch.Tensor | list[int],
        masses: torch.Tensor | list[float] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        cell_length: float = 50.0,
        pbc: bool = False,
        charge: float = 0.0,
        spin: float = 1.0,
        compute_stress: bool = False,
        head: str | None = ...,  # type: ignore[assignment]
        cache_dir: Path | str | None = None,
        mace_model: Any | None = None,
    ):
        if mace_model is None:
            self.mace_model = load_mace_torch_sim_model(
                model,
                device=device,
                dtype=dtype,
                compute_forces=True,
                compute_stress=compute_stress,
                head=head,
                cache_dir=cache_dir,
            )
        else:
            self.mace_model = mace_model

        self.device = self.mace_model.device
        self.dtype = self.mace_model.dtype
        self.atomic_numbers = torch.as_tensor(
            atomic_numbers, dtype=torch.long, device=self.device
        ).reshape(-1)
        self.num_atoms = int(self.atomic_numbers.numel())
        if masses is None:
            self.masses = torch.ones(
                self.num_atoms, dtype=self.dtype, device=self.device
            )
        else:
            self.masses = torch.as_tensor(
                masses, dtype=self.dtype, device=self.device
            ).reshape(-1)
            if int(self.masses.numel()) != self.num_atoms:
                raise ValueError("masses length must match atomic_numbers")
        self.cell_length = float(cell_length)
        self.pbc = bool(pbc)
        self.charge = float(charge)
        self.spin = float(spin)

    def to_torch_sim_model(self):
        """Return the underlying ``MaceModel``."""
        return self.mace_model

    def energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        """Return energies shaped ``(batch, 1)`` in eV."""
        coordinate = coordinate.to(device=self.device, dtype=self.dtype)
        state = batched_coordinates_to_sim_state(
            coordinate,
            atomic_numbers=self.atomic_numbers,
            masses=self.masses,
            cell=torch.eye(3, device=self.device, dtype=self.dtype) * self.cell_length,
            pbc=self.pbc,
            charge=self.charge,
            spin=self.spin,
        )
        out = self.mace_model(state)
        energy = out["energy"].reshape(-1, 1)
        return energy.detach()

    def force(self, coordinate: torch.Tensor) -> torch.Tensor:
        """Return forces shaped ``(batch, n_atoms, 3)`` in eV/Å."""
        coordinate = coordinate.to(device=self.device, dtype=self.dtype)
        batch = int(coordinate.shape[0])
        state = batched_coordinates_to_sim_state(
            coordinate,
            atomic_numbers=self.atomic_numbers,
            masses=self.masses,
            cell=torch.eye(3, device=self.device, dtype=self.dtype) * self.cell_length,
            pbc=self.pbc,
            charge=self.charge,
            spin=self.spin,
        )
        out = self.mace_model(state)
        forces = out["forces"].detach().view(batch, self.num_atoms, 3)
        return forces
