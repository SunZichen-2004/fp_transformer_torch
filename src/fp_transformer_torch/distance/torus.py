"""Torus (circular) pairwise distance for angular CVs."""

from __future__ import annotations

import torch


def torus_pairwise_distance(collective_variable: torch.Tensor) -> torch.Tensor:
    """Return ``(batch, batch)`` torus distance ``2 - 2 mean(cos Δ)``.

    Matches the JAX ``torus_pairwise_distance`` used for dihedral CVs.
    """
    if collective_variable.ndim != 3:
        raise ValueError(
            f"expected (batch, atoms, dim), got {tuple(collective_variable.shape)}"
        )
    batch_size, num_atoms, spatial_dim = collective_variable.shape
    flat = collective_variable.reshape(batch_size, num_atoms * spatial_dim)
    cosine = torch.cos(flat)
    sine = torch.sin(flat)
    dots = cosine @ cosine.T + sine @ sine.T
    return 2.0 - 2.0 * (dots / float(num_atoms * spatial_dim))
