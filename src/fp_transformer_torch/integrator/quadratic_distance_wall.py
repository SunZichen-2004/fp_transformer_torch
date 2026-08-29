"""Quadratic upper-wall restraint on selected pair distances (torch-sim model)."""

from __future__ import annotations

from typing import Sequence

import torch

try:
    from torch_sim.models.interface import ModelInterface as _ModelInterface
except ImportError:  # pragma: no cover
    _ModelInterface = torch.nn.Module  # type: ignore[misc, assignment]


class QuadraticDistanceWall(_ModelInterface):
    """Metal-unit wall: ``U = ½ k Σ_p relu(d_p - d_max)²`` (eV, Å).

    Applied independently to each listed atom pair. Soft upper bound only
    (no force when ``d ≤ d_max``).
    """

    def __init__(
        self,
        pairs: Sequence[tuple[int, int] | Sequence[int]],
        *,
        d_max: float = 3.5,
        k: float = 20.0,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        n_atoms_per_system: int | None = None,
    ):
        if _ModelInterface is torch.nn.Module:
            raise ImportError(
                "torch_sim is required for QuadraticDistanceWall. "
                "Activate conda env MACE_torchsim."
            )
        super().__init__()
        if not pairs:
            raise ValueError("need at least one atom pair")
        self.pairs = [tuple(int(x) for x in p) for p in pairs]
        for p in self.pairs:
            if len(p) != 2:
                raise ValueError(f"pair must have two indices, got {p}")
        self.d_max = float(d_max)
        self.k = float(k)
        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"
        self.n_atoms_per_system = n_atoms_per_system

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
            raise NotImplementedError("QuadraticDistanceWall does not compute stress")
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

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_atoms = int(state.n_atoms)
        if self.n_atoms_per_system is not None:
            n_per = int(self.n_atoms_per_system)
            if n_per * n_systems != n_atoms:
                raise ValueError(
                    f"n_atoms_per_system={n_per} incompatible with "
                    f"n_atoms={n_atoms}, n_systems={n_systems}"
                )
        else:
            if n_atoms % n_systems != 0:
                raise ValueError("unequal atoms per system in SimState")
            n_per = n_atoms // n_systems

        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        coordinate = coordinate.detach().requires_grad_(True)

        energy = torch.zeros(n_systems, device=self._device, dtype=self._dtype)
        for i, j in self.pairs:
            d = torch.linalg.norm(coordinate[:, i, :] - coordinate[:, j, :], dim=-1)
            excess = torch.relu(d - self.d_max)
            energy = energy + 0.5 * self.k * excess * excess

        forces = torch.zeros_like(coordinate)
        if float(energy.sum().item()) != 0.0 or energy.requires_grad:
            (grad,) = torch.autograd.grad(
                energy.sum(), coordinate, create_graph=False, allow_unused=False
            )
            # F = -∇U
            forces = (-grad).detach()

        return {
            "energy": energy.detach(),
            "forces": forces.reshape(n_atoms, 3),
        }


class QuadraticDistanceLowerWall(_ModelInterface):
    """Metal-unit lower / nonbond wall: ``U = ½ k Σ_p relu(r_min - d_p)²`` (eV, Å).

    Soft repulsion when atoms approach closer than ``r_min`` (prevents bonding).
    """

    def __init__(
        self,
        pairs: Sequence[tuple[int, int] | Sequence[int]],
        *,
        r_min: float = 2.4,
        k: float = 4.33641,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        n_atoms_per_system: int | None = None,
    ):
        if _ModelInterface is torch.nn.Module:
            raise ImportError(
                "torch_sim is required for QuadraticDistanceLowerWall. "
                "Activate conda env MACE_torchsim."
            )
        super().__init__()
        if not pairs:
            raise ValueError("need at least one atom pair")
        self.pairs = [tuple(int(x) for x in p) for p in pairs]
        for p in self.pairs:
            if len(p) != 2:
                raise ValueError(f"pair must have two indices, got {p}")
        self.r_min = float(r_min)
        self.k = float(k)
        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"
        self.n_atoms_per_system = n_atoms_per_system

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
            raise NotImplementedError(
                "QuadraticDistanceLowerWall does not compute stress"
            )
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

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_atoms = int(state.n_atoms)
        if self.n_atoms_per_system is not None:
            n_per = int(self.n_atoms_per_system)
            if n_per * n_systems != n_atoms:
                raise ValueError(
                    f"n_atoms_per_system={n_per} incompatible with "
                    f"n_atoms={n_atoms}, n_systems={n_systems}"
                )
        else:
            if n_atoms % n_systems != 0:
                raise ValueError("unequal atoms per system in SimState")
            n_per = n_atoms // n_systems

        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        coordinate = coordinate.detach().requires_grad_(True)

        energy = torch.zeros(n_systems, device=self._device, dtype=self._dtype)
        for i, j in self.pairs:
            d = torch.linalg.norm(coordinate[:, i, :] - coordinate[:, j, :], dim=-1)
            deficit = torch.relu(self.r_min - d)
            energy = energy + 0.5 * self.k * deficit * deficit

        forces = torch.zeros_like(coordinate)
        if float(energy.sum().item()) != 0.0 or energy.requires_grad:
            (grad,) = torch.autograd.grad(
                energy.sum(), coordinate, create_graph=False, allow_unused=False
            )
            forces = (-grad).detach()

        return {
            "energy": energy.detach(),
            "forces": forces.reshape(n_atoms, 3),
        }


class SoftminDistanceUpperWall(_ModelInterface):
    """Upper wall on softmin of two pair distances (eV, Å).

    ``d_softmin = -1/a ln(exp(-a d1) + exp(-a d2))``
    ``V = ½ k [max(0, d_softmin - d_wall)]²``

    Defaults match Claisen rearrangement: ``a=10 Å⁻¹``, ``d_wall=2.7 Å``.
    """

    def __init__(
        self,
        pair1: tuple[int, int] | Sequence[int],
        pair2: tuple[int, int] | Sequence[int],
        *,
        a: float = 10.0,
        d_wall: float = 2.7,
        k: float = 10.0,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        n_atoms_per_system: int | None = None,
    ):
        if _ModelInterface is torch.nn.Module:
            raise ImportError(
                "torch_sim is required for SoftminDistanceUpperWall. "
                "Activate conda env MACE_torchsim."
            )
        super().__init__()
        p1 = tuple(int(x) for x in pair1)
        p2 = tuple(int(x) for x in pair2)
        if len(p1) != 2 or len(p2) != 2:
            raise ValueError(f"each pair must have two indices, got {p1}, {p2}")
        self.pair1 = p1
        self.pair2 = p2
        self.a = float(a)
        self.d_wall = float(d_wall)
        self.k = float(k)
        if self.a <= 0.0:
            raise ValueError(f"a must be positive, got {self.a}")
        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = False
        self._memory_scales_with = "n_atoms"
        self.n_atoms_per_system = n_atoms_per_system

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
            raise NotImplementedError(
                "SoftminDistanceUpperWall does not compute stress"
            )
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

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        n_systems = int(state.n_systems)
        n_atoms = int(state.n_atoms)
        if self.n_atoms_per_system is not None:
            n_per = int(self.n_atoms_per_system)
            if n_per * n_systems != n_atoms:
                raise ValueError(
                    f"n_atoms_per_system={n_per} incompatible with "
                    f"n_atoms={n_atoms}, n_systems={n_systems}"
                )
        else:
            if n_atoms % n_systems != 0:
                raise ValueError("unequal atoms per system in SimState")
            n_per = n_atoms // n_systems

        coordinate = state.positions.view(n_systems, n_per, 3).to(
            device=self._device, dtype=self._dtype
        )
        coordinate = coordinate.detach().requires_grad_(True)

        i1, j1 = self.pair1
        i2, j2 = self.pair2
        d1 = torch.linalg.norm(coordinate[:, i1, :] - coordinate[:, j1, :], dim=-1)
        d2 = torch.linalg.norm(coordinate[:, i2, :] - coordinate[:, j2, :], dim=-1)
        # softmin via logsumexp for numerical stability
        d_softmin = -torch.logsumexp(
            torch.stack((-self.a * d1, -self.a * d2), dim=-1), dim=-1
        ) / self.a
        excess = torch.relu(d_softmin - self.d_wall)
        energy = 0.5 * self.k * excess * excess

        forces = torch.zeros_like(coordinate)
        if float(energy.sum().item()) != 0.0 or energy.requires_grad:
            (grad,) = torch.autograd.grad(
                energy.sum(), coordinate, create_graph=False, allow_unused=False
            )
            forces = (-grad).detach()

        return {
            "energy": energy.detach(),
            "forces": forces.reshape(n_atoms, 3),
        }
