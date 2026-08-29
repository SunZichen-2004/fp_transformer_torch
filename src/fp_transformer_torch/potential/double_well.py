"""Classic 2D double-well toy potential.

Slow reaction coordinate ``x1`` crosses a high barrier; ``x2`` is a stiff
harmonic (fast) degree of freedom:

.. math::

    V(x_1, x_2) = h (x_1^2 - 1)^2 + (1/2) k x_2^2
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import Potential


@dataclass(frozen=True)
class DoubleWellParams:
    """Energy / length units are arbitrary (usually set ``k_B T = 1``)."""

    h: float = 5.0
    """Barrier height at ``(0, 0)``."""

    k: float = 10.0
    """Harmonic spring constant for the fast coordinate ``x2``."""


class DoubleWell2D(Potential):
    """2D double well: high barrier along ``x1``, fast oscillator along ``x2``.

    .. math::

        V(x_1, x_2) = h (x_1^2 - 1)^2 + \\frac{1}{2} k x_2^2
    """

    def __init__(self, params: DoubleWellParams | None = None, **overrides):
        base = DoubleWellParams() if params is None else params
        data = {**base.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        self.params = DoubleWellParams(
            **{k: data[k] for k in DoubleWellParams.__dataclass_fields__}
        )

    @staticmethod
    def as_batch_xy(coordinate: torch.Tensor) -> torch.Tensor:
        """Accept ``(2,)``, ``(B, 2)``, or ``(B, 1, 2)`` → ``(B, 2)``."""
        x = coordinate
        if x.ndim == 1:
            if x.shape[0] != 2:
                raise ValueError(f"expected (2,), got {tuple(x.shape)}")
            return x.reshape(1, 2)
        if x.ndim == 2 and x.shape[-1] == 2:
            return x
        if x.ndim == 3 and x.shape[-2:] == (1, 2):
            return x.reshape(x.shape[0], 2)
        if x.ndim == 3 and x.shape[-1] == 2:
            # (B, n_atoms, 2) with n_atoms=1
            if x.shape[-2] != 1:
                raise ValueError(f"expected one particle, got {tuple(x.shape)}")
            return x.reshape(x.shape[0], 2)
        raise ValueError(f"bad coordinate shape {tuple(x.shape)}")

    def energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        xy = self.as_batch_xy(coordinate)
        x1 = xy[:, 0]
        x2 = xy[:, 1]
        h = float(self.params.h)
        k = float(self.params.k)
        v = h * (x1 * x1 - 1.0) ** 2 + 0.5 * k * (x2 * x2)
        return v[:, None]

    def minima(self) -> list[tuple[float, float]]:
        return [(-1.0, 0.0), (1.0, 0.0)]

    def saddle(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def barrier_height(self) -> float:
        return float(self.params.h)
