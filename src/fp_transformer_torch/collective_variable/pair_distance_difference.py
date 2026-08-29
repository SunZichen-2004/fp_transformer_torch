"""1D CV: ``ξ = d(i₁,j₁) − d(i₂,j₂)`` → ``(batch, 1, 1)``."""

from __future__ import annotations

from typing import Sequence

import torch

from .base import CollectiveVariable
from .multi_particle_pair_distance import MultiParticlePairDistance


class PairDistanceDifference(CollectiveVariable):
    """``ξ = d1 − d2`` from two atom pairs (Claisen: breaking − forming)."""

    def __init__(self, pairs: Sequence[tuple[int, int] | Sequence[int]]):
        if len(pairs) != 2:
            raise ValueError(f"need exactly 2 pairs, got {len(pairs)}")
        self.pairs = [tuple(p) for p in pairs]
        self._dist = MultiParticlePairDistance(self.pairs)

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        d = self._dist(coordinate)  # (B, 2, 1)
        out = d[:, 0:1, :] - d[:, 1:2, :]
        return out[0] if squeeze else out
