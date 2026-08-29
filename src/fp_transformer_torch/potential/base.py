"""Potential energy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Potential(ABC):
    @abstractmethod
    def energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        """Return potential energy with shape ``(batch, 1)``."""

    def __call__(self, coordinate: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        del beta
        return self.energy(coordinate)

    def force(self, coordinate: torch.Tensor) -> torch.Tensor:
        """``F = -∇U`` via autograd."""
        variable = coordinate.detach().requires_grad_(True)
        energy = self.energy(variable).sum()
        (gradient,) = torch.autograd.grad(energy, variable, create_graph=False)
        return (-gradient).detach()
