"""One-dimensional particle-pair distance CV."""

from __future__ import annotations

import torch

from .base import CollectiveVariable


class ParticlePairDistance(CollectiveVariable):
    """Distance between two particles → ``(batch, 1, 1)``."""

    def __init__(self, select_index=(0, 1)):
        indices = torch.as_tensor(select_index, dtype=torch.long)
        if indices.ndim != 1 or int(indices.numel()) != 2:
            raise ValueError(f"expect two particle indices, got {select_index}")
        self.select_index = indices

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        if coordinate.ndim != 3:
            raise ValueError(
                f"expected (batch, n_particles, dim), got {tuple(coordinate.shape)}"
            )
        indices = self.select_index.to(device=coordinate.device)
        selected = coordinate[:, indices, :]
        distance = torch.linalg.norm(selected[:, 0, :] - selected[:, 1, :], dim=-1)
        out = distance[:, None, None]
        return out[0] if squeeze else out
