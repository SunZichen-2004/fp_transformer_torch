"""Soft-distance switching CVs (rational bond switches).

Implements the standard rational form (numerically stable via geometric sums)::

    c(d) = (1 - (d / d_ref)^n) / (1 - (d / d_ref)^m)

with default ``n=6``, ``m=8``. Equivalent to
``Σ_{k=0}^{n-1} x^k / Σ_{k=0}^{m-1} x^k`` where ``x = d / d_ref``,
which is continuous at ``d = d_ref`` (limit ``n/m``).

For Diels–Alder style 2D CVs::

    s1 = c1 + c2 + c3
    s2 = c2 - c3
"""

from __future__ import annotations

from typing import Sequence

import torch

from .base import CollectiveVariable


def soft_switch(
    distance: torch.Tensor,
    d_ref: float,
    *,
    n: int = 6,
    m: int = 8,
) -> torch.Tensor:
    """Rational soft switch ``(1-(d/d_ref)^n)/(1-(d/d_ref)^m)``."""
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be >= 1, got n={n}, m={m}")
    if float(d_ref) <= 0.0:
        raise ValueError(f"d_ref must be > 0, got {d_ref}")
    x = distance / float(d_ref)
    # Stable rewrite of (1-x^n)/(1-x^m) for all x ≥ 0.
    num = torch.ones_like(x)
    den = torch.ones_like(x)
    x_pow = torch.ones_like(x)
    for k in range(1, max(n, m)):
        x_pow = x_pow * x
        if k < n:
            num = num + x_pow
        if k < m:
            den = den + x_pow
    return num / den.clamp_min(1e-12)


class SoftDistance(CollectiveVariable):
    """Per-pair soft distances → ``(batch, n_pairs, 1)``."""

    def __init__(
        self,
        pairs: Sequence[tuple[int, int] | Sequence[int]],
        *,
        d_ref: float = 2.0,
        n: int = 6,
        m: int = 8,
    ):
        if not pairs:
            raise ValueError("need at least one atom pair")
        self.pairs = [tuple(int(x) for x in p) for p in pairs]
        for p in self.pairs:
            if len(p) != 2:
                raise ValueError(f"pair must have two indices, got {p}")
        self.d_ref = float(d_ref)
        self.n = int(n)
        self.m = int(m)

    def distances(self, coordinate: torch.Tensor) -> torch.Tensor:
        """Raw Euclidean distances → ``(batch, n_pairs)``."""
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        if coordinate.ndim != 3:
            raise ValueError(
                f"expected (batch, n_particles, dim), got {tuple(coordinate.shape)}"
            )
        dists = []
        for i, j in self.pairs:
            dists.append(
                torch.linalg.norm(coordinate[:, i, :] - coordinate[:, j, :], dim=-1)
            )
        out = torch.stack(dists, dim=-1)
        return out[0] if squeeze else out

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        d = self.distances(coordinate)
        c = soft_switch(d, self.d_ref, n=self.n, m=self.m)
        out = c.unsqueeze(-1)  # (B, n_pairs, 1)
        return out[0] if squeeze else out


class SoftDistanceS12(CollectiveVariable):
    """DA-style 2D CV: ``s1=c1+c2+c3``, ``s2=c2-c3`` → ``(batch, 2, 1)``.

    Requires exactly three pairs ``(d1, d2, d3)``.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[int, int] | Sequence[int]],
        *,
        d_ref: float = 2.0,
        n: int = 6,
        m: int = 8,
    ):
        if len(pairs) != 3:
            raise ValueError(f"SoftDistanceS12 needs exactly 3 pairs, got {len(pairs)}")
        self.soft = SoftDistance(pairs, d_ref=d_ref, n=n, m=m)
        self.pairs = self.soft.pairs
        self.d_ref = self.soft.d_ref
        self.n = self.soft.n
        self.m = self.soft.m

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        c = self.soft(coordinate).squeeze(-1)  # (B, 3)
        s1 = c[:, 0] + c[:, 1] + c[:, 2]
        s2 = c[:, 1] - c[:, 2]
        out = torch.stack([s1, s2], dim=-1).unsqueeze(-1)  # (B, 2, 1)
        return out[0] if squeeze else out
