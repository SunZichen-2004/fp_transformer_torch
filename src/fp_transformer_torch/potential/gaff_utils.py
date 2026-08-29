"""SPONGE GAFF parameter IO and vectorized energy (PyTorch port)."""

from __future__ import annotations

import os

import numpy as np
import torch


def parse_num_file(path: str) -> list[list[float]]:
    with open(path, "r") as handle:
        return [
            [float(x.strip()) for x in line.split()]
            for line in handle.readlines()
            if line.strip() != ""
        ]


def convert_sponge_input_to_dict(prefix: str) -> dict[str, np.ndarray]:
    charge = np.loadtxt(prefix + "_charge.txt")[1:]
    n_atoms = len(charge)
    exclude = np.ones((n_atoms, n_atoms), dtype=np.int32) - np.identity(
        n_atoms, dtype=np.int32
    )
    exclude_items = parse_num_file(prefix + "_exclude.txt")
    for atom_i in range(n_atoms):
        for atom_j in exclude_items[1 + atom_i][1:]:
            atom_j = int(atom_j)
            exclude[atom_i, atom_j] = exclude[atom_j, atom_i] = 0

    lj_items = parse_num_file(prefix + "_LJ.txt")
    n_lj_types = int(lj_items[0][1])
    lj_items = lj_items[1:]
    c12 = np.zeros([n_lj_types, n_lj_types])
    c6 = np.zeros([n_lj_types, n_lj_types])
    for type_i in range(n_lj_types):
        for type_j in range(type_i + 1):
            c12[type_i][type_j] = c12[type_j][type_i] = lj_items[type_i][type_j]
    lj_items = lj_items[n_lj_types:]
    for type_i in range(n_lj_types):
        for type_j in range(type_i + 1):
            c6[type_i][type_j] = c6[type_j][type_i] = lj_items[type_i][type_j]
    lj_items = lj_items[n_lj_types:]
    diag_c12 = np.diag(c12)
    diag_c6 = np.diag(c6)
    lj_sigma = (diag_c12 / np.maximum(diag_c6, 1e-12)) ** (1.0 / 6.0)
    lj_epsilon = diag_c6 / (4.0 * np.maximum(lj_sigma, 1e-12) ** 6)
    type_index = [int(x[0]) for x in lj_items]
    lj_sigma = lj_sigma[type_index]
    lj_epsilon = lj_epsilon[type_index]

    bond_items = np.array(parse_num_file(prefix + "_bond.txt")[1:])
    angle_items = np.array(parse_num_file(prefix + "_angle.txt")[1:])

    if os.path.exists(prefix + "_dihedral.txt"):
        dihedral_terms = np.array(parse_num_file(prefix + "_dihedral.txt")[1:])
        dihedral = {
            "a": dihedral_terms[:, 0].astype(np.int32),
            "b": dihedral_terms[:, 1].astype(np.int32),
            "c": dihedral_terms[:, 2].astype(np.int32),
            "d": dihedral_terms[:, 3].astype(np.int32),
            "p": dihedral_terms[:, 4].astype(np.int32),
            "k": dihedral_terms[:, 5].astype(np.float32),
            "p0": dihedral_terms[:, 6].astype(np.float32),
        }
    else:
        dihedral = {
            k: np.zeros(0, dtype=np.int32 if k not in ("k", "p0") else np.float32)
            for k in ("a", "b", "c", "d", "p", "k", "p0")
        }
        dihedral["k"] = np.zeros(0, dtype=np.float32)
        dihedral["p0"] = np.zeros(0, dtype=np.float32)
        dihedral["p"] = np.zeros(0, dtype=np.int32)

    if os.path.exists(prefix + "_nb14.txt"):
        nb14_terms = np.array(parse_num_file(prefix + "_nb14.txt")[1:])
        nb14 = {
            "a": nb14_terms[:, 0].astype(np.int32),
            "b": nb14_terms[:, 1].astype(np.int32),
            "lj_f": nb14_terms[:, 2].astype(np.float32),
            "q_f": nb14_terms[:, 3].astype(np.float32),
        }
    else:
        nb14 = {
            "a": np.zeros(0, dtype=np.int32),
            "b": np.zeros(0, dtype=np.int32),
            "lj_f": np.zeros(0, dtype=np.float32),
            "q_f": np.zeros(0, dtype=np.float32),
        }

    return {
        "C": charge.astype(np.float32),
        "S": lj_sigma.astype(np.float32),
        "E": lj_epsilon.astype(np.float32),
        "Ba": bond_items[:, 0].astype(np.int32),
        "Bb": bond_items[:, 1].astype(np.int32),
        "Bk": bond_items[:, 2].astype(np.float32),
        "Br": bond_items[:, 3].astype(np.float32),
        "Aa": angle_items[:, 0].astype(np.int32),
        "Ab": angle_items[:, 1].astype(np.int32),
        "Ac": angle_items[:, 2].astype(np.int32),
        "Ak": angle_items[:, 3].astype(np.float32),
        "At": angle_items[:, 4].astype(np.float32),
        "Da": dihedral["a"],
        "Db": dihedral["b"],
        "Dc": dihedral["c"],
        "Dd": dihedral["d"],
        "Dk": dihedral["k"],
        "Dp": dihedral["p0"],
        "Dn": dihedral["p"],
        "Na": nb14["a"],
        "Nb": nb14["b"],
        "Nlf": nb14["lj_f"],
        "Nqf": nb14["q_f"],
        "Ex": exclude.astype(np.float32),
    }


class GaffForceFieldEnergy:
    """Batched GAFF energy with *unbatched* parameter tensors (no B-replication)."""

    def __init__(self, num_atoms: int, eps: float = 1e-12):
        self.num_atoms = int(num_atoms)
        self.eps = eps
        eye = 1.0 - torch.eye(self.num_atoms, dtype=torch.float64)
        self.temp_mask = eye.unsqueeze(0)

    @staticmethod
    def _clip_unit(values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, -0.999999, 0.999999)

    def __call__(
        self,
        coordinates: torch.Tensor,
        charge: torch.Tensor,
        lj_epsilon: torch.Tensor,
        lj_sigma: torch.Tensor,
        exclude: torch.Tensor,
        bond_a: torch.Tensor,
        bond_b: torch.Tensor,
        bond_k: torch.Tensor,
        bond_r0: torch.Tensor,
        angle_a: torch.Tensor,
        angle_b: torch.Tensor,
        angle_c: torch.Tensor,
        angle_k: torch.Tensor,
        angle_theta0: torch.Tensor,
        dihedral_a: torch.Tensor,
        dihedral_b: torch.Tensor,
        dihedral_c: torch.Tensor,
        dihedral_d: torch.Tensor,
        dihedral_n: torch.Tensor,
        dihedral_k: torch.Tensor,
        dihedral_phi0: torch.Tensor,
        nb14_a: torch.Tensor,
        nb14_b: torch.Tensor,
        nb14_lj_factor: torch.Tensor,
        nb14_charge_factor: torch.Tensor,
    ) -> torch.Tensor:
        dtype = coordinates.dtype
        device = coordinates.device
        mask = self.temp_mask.to(device=device, dtype=dtype)

        # Nonbonded pairs: (B, N, N)
        disp = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        squared = torch.sum(disp * disp, dim=-1)
        squared = torch.where(
            squared < self.eps, torch.full_like(squared, 1e3), squared
        )
        inverse_squared = mask / squared + 1e-6
        distance = torch.sqrt(squared)

        charge_1d = charge.reshape(-1)
        eps_1d = lj_epsilon.reshape(-1)
        sig_1d = lj_sigma.reshape(-1)
        exclude_2d = exclude.reshape(self.num_atoms, self.num_atoms)

        eps_ij = torch.sqrt(eps_1d[:, None] * eps_1d[None, :])
        sigma = 0.5 * (sig_1d[:, None] + sig_1d[None, :])
        sigma6 = sigma**6
        sigma12 = sigma6 * sigma6
        r6 = inverse_squared**3
        r12 = r6 * r6
        pair_lj = 4.0 * eps_ij * (sigma12 * r12 - sigma6 * r6)
        pair_lj_m = pair_lj * exclude_2d
        energy = 0.5 * torch.sum(pair_lj_m, dim=(1, 2))

        pair_q = (charge_1d[:, None] * charge_1d[None, :]) / distance
        pair_q_m = pair_q * exclude_2d
        energy = energy + 0.5 * torch.sum(pair_q_m, dim=(1, 2))

        # Bonds: gather (B, n_bonds, 3)
        if bond_k.numel() > 0:
            ra = coordinates[:, bond_a]
            rb = coordinates[:, bond_b]
            bond_len = torch.linalg.norm(ra - rb, dim=-1) - bond_r0
            energy = energy + torch.sum(bond_k * bond_len * bond_len, dim=-1)

        # Angles
        if angle_k.numel() > 0:
            va = coordinates[:, angle_a] - coordinates[:, angle_b]
            vc = coordinates[:, angle_c] - coordinates[:, angle_b]
            cos_theta = self._clip_unit(
                torch.sum(va * vc, dim=-1)
                / (
                    torch.linalg.norm(va, dim=-1) * torch.linalg.norm(vc, dim=-1)
                    + self.eps
                )
            )
            dtheta = torch.arccos(cos_theta) - angle_theta0
            energy = energy + torch.sum(angle_k * dtheta * dtheta, dim=-1)

        # Dihedrals (same convention as prior SPONGE/GAFF port)
        if dihedral_k.numel() > 0:
            v1 = coordinates[:, dihedral_a] - coordinates[:, dihedral_b]
            v2 = coordinates[:, dihedral_b] - coordinates[:, dihedral_c]
            v3 = coordinates[:, dihedral_c] - coordinates[:, dihedral_d]
            v1 = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + self.eps)
            v2 = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + self.eps)
            v3 = v3 / (torch.linalg.norm(v3, dim=-1, keepdim=True) + self.eps)
            a_norm = torch.cross(v2, v1, dim=-1)
            b_norm = torch.cross(v3, v2, dim=-1)
            cross_ab = torch.cross(a_norm, b_norm, dim=-1)
            sin_phi = torch.sum(cross_ab * v2, dim=-1)
            cos_phi = torch.sum(a_norm * b_norm, dim=-1)
            active = dihedral_k > 1e-9
            sin_phi = torch.where(
                active, self._clip_unit(sin_phi), torch.ones_like(sin_phi)
            )
            cos_phi = torch.where(
                active, self._clip_unit(cos_phi), torch.zeros_like(cos_phi)
            )
            phi = torch.atan2(-sin_phi, cos_phi)
            dn = dihedral_n.to(dtype=dtype)
            dphi = dn * phi - dihedral_phi0
            energy = energy + torch.sum(
                dihedral_k * (1.0 + torch.cos(dphi)), dim=-1
            )

        # NB14 corrections on the dense pair table
        if nb14_lj_factor.numel() > 0:
            energy = energy + torch.sum(
                nb14_lj_factor * pair_lj[:, nb14_a, nb14_b]
                + nb14_charge_factor * pair_q[:, nb14_a, nb14_b],
                dim=-1,
            )
        return energy
