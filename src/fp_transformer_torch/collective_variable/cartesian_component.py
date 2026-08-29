"""Select one Cartesian component as a 1D collective variable."""

from __future__ import annotations

import torch

from .base import CollectiveVariable


class CartesianComponent(CollectiveVariable):
    """``coordinate[:, particle, dim]`` → ``(batch, 1, 1)``.

    For the 2D double-well toy, use ``particle_index=0``, ``dim_index=0`` (``x1``).
    """

    def __init__(self, particle_index: int = 0, dim_index: int = 0):
        self.particle_index = int(particle_index)
        self.dim_index = int(dim_index)

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        if coordinate.ndim != 3:
            raise ValueError(
                f"expected (batch, n_particles, dim), got {tuple(coordinate.shape)}"
            )
        i = self.particle_index
        j = self.dim_index
        if i < 0 or i >= coordinate.shape[1]:
            raise IndexError(f"particle_index={i} out of range for {tuple(coordinate.shape)}")
        if j < 0 or j >= coordinate.shape[2]:
            raise IndexError(f"dim_index={j} out of range for {tuple(coordinate.shape)}")
        out = coordinate[:, i, j][:, None, None]
        return out[0] if squeeze else out
