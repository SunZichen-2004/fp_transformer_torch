"""Noé particle-dimer toy potential (PyTorch)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import Potential


@dataclass(frozen=True)
class ParticleDimerParams:
    num_solvent: int = 36
    epsilon: float = 1.0
    rm: float = 1.1
    dimer_slope: float = -0.5
    dimer_a: float = 25.0
    dimer_b: float = 10.0
    dimer_dmid: float = 1.5
    dimer_k: float = 20.0
    box_halfsize: float = 3.0
    box_k: float = 100.0
    grid_k: float = 0.0


class ParticleDimer(Potential):
    def __init__(self, params: ParticleDimerParams | None = None, **overrides):
        base = ParticleDimerParams() if params is None else params
        data = {**base.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        # accept legacy key
        if "nsolvent" in overrides and overrides["nsolvent"] is not None:
            data["num_solvent"] = overrides["nsolvent"]
        self.params = ParticleDimerParams(
            **{k: data[k] for k in ParticleDimerParams.__dataclass_fields__}
        )
        self.num_particles = int(self.params.num_solvent) + 2
        eye = torch.eye(self.num_particles, dtype=torch.float64)
        mask = torch.ones((self.num_particles, self.num_particles), dtype=torch.float64) - eye
        mask[0, 1] = 0.0
        mask[1, 0] = 0.0
        self.mask_matrix = mask
        self.grid = self.init_positions(self.params.dimer_dmid)

    def init_positions(
        self, dimer_distance: float | None = None, scaling_factor: float = 1.05
    ) -> torch.Tensor:
        dmid = (
            float(self.params.dimer_dmid)
            if dimer_distance is None
            else float(dimer_distance)
        )
        num_solvent = int(self.params.num_solvent)
        half = float(self.params.box_halfsize)
        sqrtn = int(round(math.sqrt(num_solvent)))
        if sqrtn * sqrtn != num_solvent:
            raise ValueError(f"num_solvent={num_solvent} must be a perfect square")
        locs = torch.linspace(-half - 1.0, half + 1.0, sqrtn + 2, dtype=torch.float64)[
            1:-1
        ]
        ii, jj = torch.meshgrid(locs, locs, indexing="ij")
        solvent = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=-1)
        dimer = torch.tensor([[-0.5 * dmid, 0.0], [0.5 * dmid, 0.0]], dtype=torch.float64)
        coords = torch.cat([dimer, solvent], dim=0).unsqueeze(0)
        return scaling_factor * coords

    @staticmethod
    def as_batch_particles_dim(coordinate: torch.Tensor) -> torch.Tensor:
        if coordinate.ndim == 2 and coordinate.shape[-1] == 2:
            return coordinate.unsqueeze(0)
        if coordinate.ndim == 2:
            batch = coordinate.shape[0]
            return coordinate.reshape(batch, -1, 2)
        if coordinate.ndim == 3 and coordinate.shape[-1] == 2:
            return coordinate
        raise ValueError(f"bad coordinate shape {tuple(coordinate.shape)}")

    def lj_energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = self.as_batch_particles_dim(coordinate)
        diff = coordinate[:, :, None, :] - coordinate[:, None, :, :]
        distance_squared = torch.sum(diff * diff, dim=-1)
        mask = self.mask_matrix.to(device=coordinate.device, dtype=coordinate.dtype)
        distance_squared = distance_squared + (1.0 - mask)
        relative = (self.params.rm**2) / distance_squared
        relative = relative * mask
        return 0.5 * self.params.epsilon * torch.sum(relative**6, dim=(-2, -1))

    def dimer_energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = self.as_batch_particles_dim(coordinate)
        p0 = coordinate[:, 0, :]
        p1 = coordinate[:, 1, :]
        energy_dx = self.params.dimer_k * (p0[:, 0] + p1[:, 0]) ** 2
        energy_dy = self.params.dimer_k * (p0[:, 1] ** 2 + p1[:, 1] ** 2)
        distance = torch.linalg.norm(p0 - p1, dim=-1)
        d0 = 2.0 * (distance - self.params.dimer_dmid)
        d2 = d0 * d0
        d4 = d2 * d2
        interaction = (
            self.params.dimer_slope * d0 - self.params.dimer_a * d2 + self.params.dimer_b * d4
        )
        return energy_dx + energy_dy + interaction

    def box_energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = self.as_batch_particles_dim(coordinate)
        half = self.params.box_halfsize
        k = self.params.box_k
        d_left = -(coordinate[..., 0] + half)
        d_right = coordinate[..., 0] - half
        d_down = -(coordinate[..., 1] + half)
        d_up = coordinate[..., 1] - half
        return (
            torch.sum((torch.sign(d_left) + 1.0) * k * d_left**2, dim=-1)
            + torch.sum((torch.sign(d_right) + 1.0) * k * d_right**2, dim=-1)
            + torch.sum((torch.sign(d_down) + 1.0) * k * d_down**2, dim=-1)
            + torch.sum((torch.sign(d_up) + 1.0) * k * d_up**2, dim=-1)
        )

    def grid_energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = self.as_batch_particles_dim(coordinate)
        grid = self.grid.to(device=coordinate.device, dtype=coordinate.dtype)
        if grid.ndim == 3 and grid.shape[0] == 1:
            grid = grid[0]
        diff2 = (coordinate - grid) ** 2
        return torch.sum(
            self.params.grid_k * (self.params.rm**2 * diff2) ** 6, dim=(-2, -1)
        )

    def energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        coordinate = self.as_batch_particles_dim(coordinate)
        if coordinate.shape[-2] != self.num_particles:
            raise ValueError(
                f"Expected {self.num_particles} particles, got {coordinate.shape[-2]}"
            )
        total = (
            self.lj_energy(coordinate)
            + self.dimer_energy(coordinate)
            + self.box_energy(coordinate)
            + self.grid_energy(coordinate)
        )
        return total[:, None]
