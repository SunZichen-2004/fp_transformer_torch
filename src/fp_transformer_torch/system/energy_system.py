"""Batch coordinates + potential energy wrapper."""

from __future__ import annotations

import torch

from ..potential.base import Potential


class EnergySystem:
    def __init__(self, potential: Potential, coordinate: torch.Tensor):
        self.potential = potential
        self.coordinate = coordinate

    def energy(self, coordinate: torch.Tensor | None = None) -> torch.Tensor:
        coordinate = self.coordinate if coordinate is None else coordinate
        return self.potential.energy(coordinate)

    def force(self, coordinate: torch.Tensor | None = None) -> torch.Tensor:
        coordinate = self.coordinate if coordinate is None else coordinate
        return self.potential.force(coordinate)

    def potential_gradient(
        self, coordinate: torch.Tensor, beta: float | torch.Tensor
    ) -> torch.Tensor:
        """``β ∇U`` (same sign convention as JAX ``_system_grad_times_beta``)."""
        variable = coordinate.detach().requires_grad_(True)
        energy = self.potential.energy(variable)
        # energy (B,1) → sum for scalar, get per-replica via grad
        (gradient,) = torch.autograd.grad(energy.sum(), variable, create_graph=False)
        return (beta * gradient).detach()
