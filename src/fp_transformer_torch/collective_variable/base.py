"""Base collective variable interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class CollectiveVariable(ABC):
    @abstractmethod
    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        """Map Cartesian coordinates to CV tensor ``(batch, ..., feature_dim)``."""
