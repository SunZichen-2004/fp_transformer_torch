"""Dihedral-angle collective variables (torus)."""

from __future__ import annotations

import torch

from ..math.dihedral import dihedral_angle
from .base import CollectiveVariable


class Torsion(CollectiveVariable):
    """Compute dihedrals for index rows of shape ``(num_torsions, 4)``."""

    def __init__(self, torsion_indices: torch.Tensor | list):
        indices = torch.as_tensor(torsion_indices, dtype=torch.long)
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError(
                f"torsion_indices must be (M, 4), got {tuple(indices.shape)}"
            )
        self.torsion_indices = indices

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        """Return angles shaped ``(..., num_torsions, 1)`` in radians."""
        if coordinate.ndim < 2 or coordinate.shape[-1] != 3:
            raise ValueError(
                f"Torsion expects [..., n_atoms, 3], got {tuple(coordinate.shape)}"
            )
        indices = self.torsion_indices.to(device=coordinate.device)
        leading = coordinate.shape[:-2]
        flat = coordinate.reshape((-1, coordinate.shape[-2], 3))
        point_a = flat[:, indices[:, 0], :]
        point_b = flat[:, indices[:, 1], :]
        point_c = flat[:, indices[:, 2], :]
        point_d = flat[:, indices[:, 3], :]
        # (batch, M, 3) -> flatten for batched dihedral
        batch_size, num_torsions, _ = point_a.shape
        angles = dihedral_angle(
            point_a.reshape(-1, 3),
            point_b.reshape(-1, 3),
            point_c.reshape(-1, 3),
            point_d.reshape(-1, 3),
        ).reshape(batch_size, num_torsions)
        angles = angles.reshape(*leading, num_torsions)
        return angles.unsqueeze(-1)
