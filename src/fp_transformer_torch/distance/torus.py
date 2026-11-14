"""Torus (circular) pairwise distance for angular CVs."""

from __future__ import annotations

import torch


def torus_pairwise_distance(
    collective_variable: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(batch, batch)`` torus distance ``2 - 2 mean(cos Δ)``.

    An optional reference batch gives a rectangular query-by-center matrix.
    Matches the JAX ``torus_pairwise_distance`` used for dihedral CVs.
    """
    if collective_variable.ndim != 3:
        raise ValueError(
            f"expected (batch, atoms, dim), got {tuple(collective_variable.shape)}"
        )
    batch_size, num_atoms, spatial_dim = collective_variable.shape
    flat = collective_variable.reshape(batch_size, num_atoms * spatial_dim)
    if reference is not None and (
        reference.ndim != 3 or reference.shape[1:] != collective_variable.shape[1:]
    ):
        raise ValueError("reference must have the same CV feature shape as the query")
    cosine = torch.cos(flat)
    sine = torch.sin(flat)
    center_cosine = cosine if reference is None else torch.cos(reference.flatten(1))
    center_sine = sine if reference is None else torch.sin(reference.flatten(1))
    dots = cosine @ center_cosine.T + sine @ center_sine.T
    return 2.0 - 2.0 * (dots / float(num_atoms * spatial_dim))
