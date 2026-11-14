"""Euclidean pairwise distance for CV tensors shaped ``(batch, atoms, dim)``."""

from __future__ import annotations

import torch


def pairwise_distance(
    collective_variable: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(batch, batch)`` mean-squared Euclidean distance / ``(A*D)``.

    An optional reference batch gives a rectangular query-by-center matrix.
    Matches the JAX ``pairwise_distance`` convention used by the Gaussian kernel.
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
    centers = flat if reference is None else reference.flatten(1)
    norm_squared = torch.sum(flat * flat, dim=1)
    center_norm_squared = (
        norm_squared if reference is None else torch.sum(centers * centers, dim=1)
    )
    distance_squared = (
        norm_squared[:, None] + center_norm_squared[None, :] - 2.0 * (flat @ centers.T)
    )
    distance_squared = torch.clamp(distance_squared, min=0.0)
    return distance_squared / float(num_atoms * spatial_dim)
