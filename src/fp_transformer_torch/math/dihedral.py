"""Dihedral (torsion) angle in radians."""

from __future__ import annotations

import torch


def dihedral_angle(
    point_a: torch.Tensor,
    point_b: torch.Tensor,
    point_c: torch.Tensor,
    point_d: torch.Tensor,
) -> torch.Tensor:
    """Stable atan2 dihedral for points with last dim 3.

    Accepts leading batch dims; returns angle with the same leading shape.
    """
    eps = 1e-12
    bond_ab = point_b - point_a
    bond_bc = point_c - point_b
    bond_cd = point_d - point_c

    normal_abc = torch.cross(bond_ab, bond_bc, dim=-1)
    normal_bcd = torch.cross(bond_bc, bond_cd, dim=-1)

    bond_bc_unit = bond_bc / torch.clamp(
        torch.linalg.norm(bond_bc, dim=-1, keepdim=True), min=eps
    )
    normal_abc_unit = normal_abc / torch.clamp(
        torch.linalg.norm(normal_abc, dim=-1, keepdim=True), min=eps
    )
    normal_bcd_unit = normal_bcd / torch.clamp(
        torch.linalg.norm(normal_bcd, dim=-1, keepdim=True), min=eps
    )

    sin_term = torch.sum(
        torch.cross(normal_abc_unit, normal_bcd_unit, dim=-1) * bond_bc_unit, dim=-1
    )
    cos_term = torch.sum(normal_abc_unit * normal_bcd_unit, dim=-1)
    return torch.atan2(sin_term, cos_term)


def vector_torsion(
    vector1: torch.Tensor,
    axis: torch.Tensor,
    vector2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """PLUMED-style torsion from VECTOR1 / AXIS / VECTOR2 (radians in [-π, π]).

    Angle between the projections of ``vector1`` and ``vector2`` onto the plane
    orthogonal to ``axis``.
    """
    axis_u = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    v1_perp = vector1 - (vector1 * axis_u).sum(dim=-1, keepdim=True) * axis_u
    v2_perp = vector2 - (vector2 * axis_u).sum(dim=-1, keepdim=True) * axis_u
    v1_perp = v1_perp / torch.linalg.norm(v1_perp, dim=-1, keepdim=True).clamp_min(eps)
    v2_perp = v2_perp / torch.linalg.norm(v2_perp, dim=-1, keepdim=True).clamp_min(eps)
    cos_phi = (v1_perp * v2_perp).sum(dim=-1)
    sin_phi = (axis_u * torch.linalg.cross(v1_perp, v2_perp, dim=-1)).sum(dim=-1)
    return torch.atan2(sin_phi, cos_phi)
