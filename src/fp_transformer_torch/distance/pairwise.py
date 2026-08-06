"""Euclidean pairwise distance for CV tensors shaped ``(batch, atoms, dim)``."""

from __future__ import annotations

import torch


def pairwise_distance(collective_variable: torch.Tensor) -> torch.Tensor:
    """Return ``(batch, batch)`` mean-squared Euclidean distance / ``(A*D)``.

    Matches the JAX ``pairwise_distance`` convention used by the Gaussian kernel.
    """
    if collective_variable.ndim != 3:
        raise ValueError(
            f"expected (batch, atoms, dim), got {tuple(collective_variable.shape)}"
        )
    batch_size, num_atoms, spatial_dim = collective_variable.shape
    flat = collective_variable.reshape(batch_size, num_atoms * spatial_dim)
    norm_squared = torch.sum(flat * flat, dim=1)
    distance_squared = (
        norm_squared[:, None] + norm_squared[None, :] - 2.0 * (flat @ flat.T)
    )
    distance_squared = torch.clamp(distance_squared, min=0.0)
    return distance_squared / float(num_atoms * spatial_dim)
