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
