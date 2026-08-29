"""GAFF as a torch-sim ``ModelInterface`` (metal units: eV / eV·Å⁻¹)."""

from __future__ import annotations

from typing import Any

import torch

from .gaff import GaffPotential

try:
    from torch_sim.models.interface import ModelInterface as _ModelInterface
except ImportError:  # pragma: no cover
    _ModelInterface = torch.nn.Module  # type: ignore[misc, assignment]

# 1 eV = 23.060547830619027 kcal/mol (same as mechanical_embedding_MLMM)
KCAL_PER_MOL_TO_EV = 1.0 / 23.060547830619027


def _n_atoms_per_system(state, n_atoms_per_system: int | None) -> int:
    n_systems = int(state.n_systems)
    n_atoms = int(state.n_atoms)
    if n_atoms_per_system is not None:
        n_per = int(n_atoms_per_system)
        if n_per * n_systems != n_atoms:
            raise ValueError(
                f"n_atoms_per_system={n_per} incompatible with "
                f"n_atoms={n_atoms}, n_systems={n_systems}"
            )
        return n_per
    if n_atoms % n_systems != 0:
        raise ValueError("unequal atoms per system in SimState")
    return n_atoms // n_systems


class GaffTorchSimModel(_ModelInterface):
    """Wrap ``GaffPotential`` for ``torch_sim.integrate`` / ``SumModel``.

    SPONGE/GAFF energies are kcal·mol⁻¹; outputs are scaled to **eV** so MetaD
    / Langevin metal units match MACE experiments.
    """

    def __init__(
        self,
        sponge_prefix: str,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float64,
        n_atoms_per_system: int | None = None,
        energy_scale: float = KCAL_PER_MOL_TO_EV,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ):
        if _ModelInterface is torch.nn.Module:
            raise ImportError(
                "torch_sim is required. Activate conda env MACE_torchsim."
            )
        super().__init__()
        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = bool(compute_forces)
        self._compute_stress = bool(compute_stress)
        self._memory_scales_with = "n_atoms"
        self.energy_scale = float(energy_scale)
        self.gaff = GaffPotential(str(sponge_prefix))
        self.n_atoms_per_system = (
            int(n_atoms_per_system)
            if n_atoms_per_system is not None
            else int(self.gaff.num_atoms)
        )
        if self.n_atoms_per_system != int(self.gaff.num_atoms):
            raise ValueError(
                f"n_atoms_per_system={self.n_atoms_per_system} != "
                f"GAFF num_atoms={self.gaff.num_atoms}"
            )
        self._last_metrics: dict[str, float] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, value: bool) -> None:  # noqa: FBT001
        if value:
            raise NotImplementedError("GaffTorchSimModel does not compute stress")
        self._compute_stress = False

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @compute_forces.setter
    def compute_forces(self, value: bool) -> None:  # noqa: FBT001
        self._compute_forces = bool(value)

    @property
    def memory_scales_with(self):
        return self._memory_scales_with

    def forward(self, state: Any, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_per = _n_atoms_per_system(state, self.n_atoms_per_system)
        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        coordinate = coordinate.detach().requires_grad_(True)
        # GAFF → kcal/mol; scale to eV for metal-unit Langevin / MetaD
        energy_kcal = self.gaff.energy(coordinate).reshape(-1)
        energy = energy_kcal * self.energy_scale
        if not self._compute_forces:
            return {"energy": energy.detach()}
        (grad_x,) = torch.autograd.grad(
            energy.sum(), coordinate, create_graph=False, retain_graph=False
        )
        forces = (-grad_x).reshape(n_systems * n_per, 3).detach()
        energy_out = energy.detach()
        self._last_metrics = {
            "gaff_energy_mean_eV": float(energy_out.mean().item()),
            "gaff_energy_max_eV": float(energy_out.max().item()),
            "force_rms": float(torch.sqrt(torch.mean(forces * forces)).item()),
        }
        return {"energy": energy_out, "forces": forces}
