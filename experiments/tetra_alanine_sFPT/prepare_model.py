#!/usr/bin/env python3
"""Build vacuum ACE–ALA–ALA–ALA–NME (tetra-alanine) like dialanine.

Same force field as ``experiments/dialanine``: Xponge Amber ff14SB residue
library → SPONGE parameter files, energy via ``GaffPotential``.

CVs: all backbone φ/ψ (3 φ + 3 ψ). ``backbone_cv.txt`` is 1-based, ordered
φ1, ψ1, φ2, ψ2, φ3, ψ3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

Z_FROM_MASS = (
    (2.0, 1),
    (13.0, 6),
    (15.5, 7),
    (17.5, 8),
    (33.0, 16),
)
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def z_from_mass(mass: float) -> int:
    for cutoff, z in Z_FROM_MASS:
        if float(mass) < cutoff:
            return z
    return 6


def flatten_atoms(mol) -> None:
    mol.atoms = []
    for res in mol.residues:
        mol.atoms.extend(res.atoms)
    mol.atom_index.clear()
    mol.atom_index.update({atom: i for i, atom in enumerate(mol.atoms)})


def atom_index_1based(mol, residue, name: str) -> int:
    return int(mol.atom_index[residue.name2atom(name)]) + 1


def backbone_phi_psi_indices(mol) -> tuple[list[list[int]], list[str]]:
    """φ = C(prev)–N–CA–C, ψ = N–CA–C–N(next) for each internal ALA."""
    residues = list(mol.residues)
    rows: list[list[int]] = []
    names: list[str] = []
    ala_idx = 0
    for i, res in enumerate(residues):
        if str(res.name) != "ALA":
            continue
        ala_idx += 1
        prev_res, next_res = residues[i - 1], residues[i + 1]
        phi = [
            atom_index_1based(mol, prev_res, "C"),
            atom_index_1based(mol, res, "N"),
            atom_index_1based(mol, res, "CA"),
            atom_index_1based(mol, res, "C"),
        ]
        psi = [
            atom_index_1based(mol, res, "N"),
            atom_index_1based(mol, res, "CA"),
            atom_index_1based(mol, res, "C"),
            atom_index_1based(mol, next_res, "N"),
        ]
        rows.extend([phi, psi])
        names.extend([f"phi{ala_idx}", f"psi{ala_idx}"])
    if len(rows) != 6:
        raise RuntimeError(f"expected 3 φ + 3 ψ, got {names}")
    return rows, names


def write_xyz(path: Path, coords: np.ndarray, zs: list[int], comment: str) -> None:
    with Path(path).open("w") as fh:
        fh.write(f"{coords.shape[0]}\n{comment}\n")
        for z, (x, y, zc) in zip(zs, coords):
            fh.write(f"{SYMBOL.get(int(z), 'C')} {x:.6f} {y:.6f} {zc:.6f}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR)
    p.add_argument("--prefix-name", type=str, default="tetra_alanine")
    args = p.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / args.prefix_name

    import Xponge
    import Xponge.forcefield.amber.ff14sb  # noqa: F401 — loads ACE/ALA/NME

    ace = Xponge.ResidueType.get_type("ACE")
    ala = Xponge.ResidueType.get_type("ALA")
    nme = Xponge.ResidueType.get_type("NME")
    mol = ace + ala + ala + ala + nme
    mol.name = "tetra_alanine"

    pdb_path = out / f"{args.prefix_name}.pdb"
    Xponge.save_pdb(mol, str(pdb_path))
    Xponge.save_sponge_input(mol, prefix=args.prefix_name, dirname=str(out))
    flatten_atoms(mol)

    n_atoms = len(mol.atoms)
    masses = np.asarray([float(atom.mass) for atom in mol.atoms], dtype=np.float64)
    zs = [z_from_mass(m) for m in masses]
    coords = np.asarray(
        [[float(atom.x), float(atom.y), float(atom.z)] for atom in mol.atoms],
        dtype=np.float64,
    )

    mass_path = Path(str(prefix) + "_mass.txt")
    if not mass_path.is_file():
        with mass_path.open("w") as fh:
            fh.write(f"{n_atoms}\n")
            for m in masses:
                fh.write(f"{m:.6f}\n")

    with Path(str(prefix) + "_atomic_number.txt").open("w") as fh:
        for z in zs:
            fh.write(f"{z}\n")

    write_xyz(out / f"{args.prefix_name}.xyz", coords, zs, "tetra_alanine ACE-(ALA)3-NME")

    cv_rows, cv_names = backbone_phi_psi_indices(mol)
    cv_path = out / "backbone_cv.txt"
    with cv_path.open("w") as fh:
        fh.write("# 1-based  φ1 ψ1 φ2 ψ2 φ3 ψ3   (C-N-CA-C / N-CA-C-N)\n")
        for row, name in zip(cv_rows, cv_names):
            fh.write(" ".join(str(i) for i in row) + f"  # {name}\n")

    seq = [str(res.name) for res in mol.residues]
    print(f"[prep] seq={' '.join(seq)}  n_atoms={n_atoms}", flush=True)
    print(f"[prep] pdb={pdb_path}", flush=True)
    print(f"[prep] sponge prefix={prefix}_*", flush=True)
    print(f"[prep] CVs ({len(cv_names)}):", flush=True)
    for name, row in zip(cv_names, cv_rows):
        print(f"       {name}: {row}", flush=True)

    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from fp_transformer_torch import GaffPotential, Torsion
    from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates

    pot = GaffPotential(str(prefix))
    xyz = load_sponge_coordinates(prefix, num_atoms=pot.num_atoms)
    x = torch.as_tensor(xyz[None, ...], dtype=torch.float64)
    e = float(pot.energy(x).reshape(-1)[0].item())
    tors = np.asarray(cv_rows, dtype=np.int64) - 1
    cv = Torsion(tors)(x).detach().cpu().numpy().reshape(-1)
    print(f"[prep] E0={e:.4f} kcal/mol  n_atoms(GAFF)={pot.num_atoms}", flush=True)
    print(
        "       "
        + "  ".join(f"{n}={v:.3f}" for n, v in zip(cv_names, cv)),
        flush=True,
    )
    print("[prep] done", flush=True)


if __name__ == "__main__":
    main()
