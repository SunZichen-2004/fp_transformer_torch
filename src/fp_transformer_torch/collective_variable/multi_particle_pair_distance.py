"""Multi-pair Euclidean distance CV → ``(batch, n_pairs, 1)``."""

from __future__ import annotations

from typing import Sequence

import torch

from .base import CollectiveVariable
from .particle_pair_distance import ParticlePairDistance


class MultiParticlePairDistance(CollectiveVariable):
    """Stack several :class:`ParticlePairDistance` CVs along axis 1."""

    def __init__(self, pairs: Sequence[tuple[int, int] | Sequence[int]]):
        if not pairs:
            raise ValueError("need at least one atom pair")
        self.pairs = [ParticlePairDistance(tuple(p)) for p in pairs]

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        parts = [cv(coordinate) for cv in self.pairs]  # each (B,1,1)
        out = torch.cat(parts, dim=1)
        return out[0] if squeeze else out
