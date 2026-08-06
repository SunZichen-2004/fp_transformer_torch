"""GAFF potential from SPONGE parameter files."""

from __future__ import annotations

import os
from typing import Optional

import torch

from .base import Potential
from .gaff_utils import GaffForceFieldEnergy, convert_sponge_input_to_dict


class GaffPotential(Potential):
    """Tight-sized GAFF energy (no NMAX padding; parameters stay unbatched)."""

    def __init__(self, sponge_prefix: str, num_atoms: Optional[int] = None):
        required = [
            f"{sponge_prefix}_charge.txt",
            f"{sponge_prefix}_LJ.txt",
            f"{sponge_prefix}_bond.txt",
            f"{sponge_prefix}_angle.txt",
        ]
        for path in required:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required parameter file not found: {path}")

        raw = convert_sponge_input_to_dict(sponge_prefix)
        self.num_atoms = (
            int(num_atoms) if num_atoms is not None else int(len(raw["C"]))
        )
        if self.num_atoms <= 0:
            raise ValueError(f"Invalid number of atoms: {self.num_atoms}")

        self.parameters = {
            "C": torch.as_tensor(raw["C"][: self.num_atoms], dtype=torch.float64),
            "S": torch.as_tensor(raw["S"][: self.num_atoms], dtype=torch.float64),
            "E": torch.as_tensor(raw["E"][: self.num_atoms], dtype=torch.float64),
            "Ex": torch.as_tensor(
                raw["Ex"][: self.num_atoms, : self.num_atoms], dtype=torch.float64
            ),
            "Ba": torch.as_tensor(raw["Ba"], dtype=torch.long),
            "Bb": torch.as_tensor(raw["Bb"], dtype=torch.long),
            "Bk": torch.as_tensor(raw["Bk"], dtype=torch.float64),
            "Br": torch.as_tensor(raw["Br"], dtype=torch.float64),
            "Aa": torch.as_tensor(raw["Aa"], dtype=torch.long),
            "Ab": torch.as_tensor(raw["Ab"], dtype=torch.long),
            "Ac": torch.as_tensor(raw["Ac"], dtype=torch.long),
            "Ak": torch.as_tensor(raw["Ak"], dtype=torch.float64),
            "At": torch.as_tensor(raw["At"], dtype=torch.float64),
            "Da": torch.as_tensor(raw["Da"], dtype=torch.long),
            "Db": torch.as_tensor(raw["Db"], dtype=torch.long),
            "Dc": torch.as_tensor(raw["Dc"], dtype=torch.long),
            "Dd": torch.as_tensor(raw["Dd"], dtype=torch.long),
            "Dn": torch.as_tensor(raw["Dn"], dtype=torch.long),
            "Dk": torch.as_tensor(raw["Dk"], dtype=torch.float64),
            "Dp": torch.as_tensor(raw["Dp"], dtype=torch.float64),
            "Na": torch.as_tensor(raw["Na"], dtype=torch.long),
            "Nb": torch.as_tensor(raw["Nb"], dtype=torch.long),
            "Nlf": torch.as_tensor(raw["Nlf"], dtype=torch.float64),
            "Nqf": torch.as_tensor(raw["Nqf"], dtype=torch.float64),
        }
        self.energy_calculator = GaffForceFieldEnergy(num_atoms=self.num_atoms)
        self._device_params: dict[str, torch.Tensor] | None = None
        self._device_key: tuple[str, torch.dtype] | None = None

    def _params_on(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        key = (str(device), dtype)
        if self._device_params is not None and self._device_key == key:
            return self._device_params
        index_keys = {
            "Ba",
            "Bb",
            "Aa",
            "Ab",
            "Ac",
            "Da",
            "Db",
            "Dc",
            "Dd",
            "Na",
            "Nb",
            "Dn",
        }
        out: dict[str, torch.Tensor] = {}
        for name, param in self.parameters.items():
            tensor = param.to(device=device)
            if name in index_keys:
                out[name] = tensor.to(dtype=torch.long)
            else:
                out[name] = tensor.to(dtype=dtype)
        self._device_params = out
        self._device_key = key
        return out

    def energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        if coordinate.ndim != 3 or coordinate.shape[-1] != 3:
            raise ValueError(
                f"expected (batch, num_atoms, 3), got {tuple(coordinate.shape)}"
            )
        if coordinate.shape[1] != self.num_atoms:
            raise ValueError(
                f"Expected {self.num_atoms} atoms, got {coordinate.shape[1]}"
            )
        p = self._params_on(coordinate.device, coordinate.dtype)
        energy = self.energy_calculator(
            coordinate,
            p["C"],
            p["E"],
            p["S"],
            p["Ex"],
            p["Ba"],
            p["Bb"],
            p["Bk"],
            p["Br"],
            p["Aa"],
            p["Ab"],
            p["Ac"],
            p["Ak"],
            p["At"],
            p["Da"],
            p["Db"],
            p["Dc"],
            p["Dd"],
            p["Dn"],
            p["Dk"],
            p["Dp"],
            p["Na"],
            p["Nb"],
            p["Nlf"],
            p["Nqf"],
        )
        return energy.reshape(-1, 1)
