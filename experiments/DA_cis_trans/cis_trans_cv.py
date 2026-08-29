"""IMDA cis/trans 2D CVs (paper HLDA + cis/trans torsion product).

Distances follow the original PLUMED CV (nm); Cartesian input is Angstrom.
"""

from __future__ import annotations

import torch

from fp_transformer_torch.collective_variable.base import CollectiveVariable

# Forming DA bonds (0-based) for quadratic walls — wall still uses Angstrom.
PAIR_D4 = (7, 16)  # paper 8–17
PAIR_D6 = (18, 0)  # paper 19–1
WALL_PAIRS = (PAIR_D4, PAIR_D6)


def distance(positions: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Atom indices are 0-based. Return distance in **nm** (positions in Å)."""
    rij = positions[..., i, :] - positions[..., j, :]
    d_angstrom = torch.linalg.norm(rij, dim=-1)
    return d_angstrom / 10.0


def vector_torsion(
    v1: torch.Tensor,
    axis: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """PLUMED-like torsion from VECTOR1 / AXIS / VECTOR2 → radians in [-π, π]."""
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)

    v1_perp = v1 - (v1 * axis).sum(dim=-1, keepdim=True) * axis
    v2_perp = v2 - (v2 * axis).sum(dim=-1, keepdim=True) * axis

    v1_perp = v1_perp / torch.linalg.norm(v1_perp, dim=-1, keepdim=True).clamp_min(eps)
    v2_perp = v2_perp / torch.linalg.norm(v2_perp, dim=-1, keepdim=True).clamp_min(eps)

    cos_phi = (v1_perp * v2_perp).sum(dim=-1)
    sin_phi = (axis * torch.linalg.cross(v1_perp, v2_perp, dim=-1)).sum(dim=-1)
    return torch.atan2(sin_phi, cos_phi)


def compute_cv_parts(positions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Full paper CV breakdown (cv1, cv2, c8, c17, …)."""
    d1 = distance(positions, 0, 3)  # 1-4
    d2 = distance(positions, 3, 5)  # 4-6
    d3 = distance(positions, 5, 7)  # 6-8
    d4 = distance(positions, 7, 16)  # 8-17
    d5 = distance(positions, 16, 18)  # 17-19
    d6 = distance(positions, 18, 0)  # 19-1

    x = (
        2.074238e-02 * d1
        + 5.398011e-02 * d2
        - 7.989440e-03 * d3
        + 4.247663e-01 * d4
        + 1.854460e-02 * d5
        + 9.032285e-01 * d6
    )
    cv1 = (4.0 / (1.0 + torch.exp((x - 0.35) / 0.1)) + 1.0) * (x - 0.35)

    vector1_c8 = positions[..., 5, :] - positions[..., 7, :]  # 8→6
    axis_c8 = positions[..., 7, :] - positions[..., 16, :]  # 17→8
    vector2_c8 = positions[..., 9, :] - positions[..., 7, :]  # 8→10
    c8 = vector_torsion(vector1_c8, axis_c8, vector2_c8)

    vector1_c17 = positions[..., 15, :] - positions[..., 16, :]  # 17→16
    axis_c17 = positions[..., 7, :] - positions[..., 16, :]  # 17→8
    vector2_c17 = positions[..., 18, :] - positions[..., 16, :]  # 17→19
    c17 = vector_torsion(vector1_c17, axis_c17, vector2_c17)

    cv2 = c8 * c17
    return {
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "d5": d5,
        "d6": d6,
        "x": x,
        "cv1": cv1,
        "cv2": cv2,
        "c8": c8,
        "c17": c17,
    }


def compute_cv(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    parts = compute_cv_parts(positions)
    return parts["cv1"], parts["cv2"]


class DACisTransCV(CollectiveVariable):
    """Return ``(batch, 2, 1)`` with paper ``[cv1, cv2]``."""

    n_cv = 2

    def components(self, coordinate: torch.Tensor) -> dict[str, torch.Tensor]:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        if coordinate.ndim != 3 or coordinate.shape[-1] != 3:
            raise ValueError(
                f"expected (batch, n_atoms, 3), got {tuple(coordinate.shape)}"
            )
        out = compute_cv_parts(coordinate)
        if squeeze:
            return {k: v[0] for k, v in out.items()}
        return out

    def __call__(self, coordinate: torch.Tensor) -> torch.Tensor:
        squeeze = coordinate.ndim == 2
        if squeeze:
            coordinate = coordinate.unsqueeze(0)
        cv1, cv2 = compute_cv(coordinate)
        stacked = torch.stack([cv1, cv2], dim=-1).unsqueeze(-1)  # (B, 2, 1)
        return stacked[0] if squeeze else stacked
