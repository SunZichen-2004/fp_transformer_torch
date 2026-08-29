#!/usr/bin/env python3
"""Prepare a solvated Claisen (allyl phenyl ether) box for ML/MM.

Outputs
-------
* ``box.xyz`` / ``box.pdb`` — solute + TIP3P waters
* ``cell.txt`` — 3×3 cell (Å)
* ``topology.npz`` — merged :class:`MLMMTopology`
* ``ml_atom_indices.txt`` — 0-based solute atom indices (ML region)
* optional SPONGE-like solute param stub if ``--sponge-prefix`` is omitted

Water packing is a simple cubic lattice (no Packmol dependency). For production
boxes prefer an external equilibrated PDB and pass ``--coords`` + ``--n-waters``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from fp_transformer_torch.potential.gaff_utils import (  # noqa: E402
    convert_sponge_input_to_dict,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    SPONGE_CHARGE_SCALE,
    TIP3P_PARAMS,
    build_topology_from_sponge_and_tip3p,
)

DEFAULT_XYZ = EXPERIMENT_DIR / "allyl_phenyl_ether.xyz"

# Element fallback LJ (GAFF-ish, kcal/mol) for stub solute parameters
_ELEMENT_LJ = {
    "H": (2.471, 0.0157),
    "C": (3.3997, 0.0860),
    "N": (3.2500, 0.1700),
    "O": (3.0669, 0.2104),
    "F": (3.1181, 0.0610),
    "S": (3.5636, 0.2500),
    "CL": (3.4709, 0.2650),
    "BR": (3.6000, 0.3200),
}
_ELEMENT_MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "S": 32.06,
    "CL": 35.45,
    "BR": 79.904,
}
_ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "CL": 17, "BR": 35}
_ELEMENT_Q = {"H": 0.06, "C": -0.08, "N": -0.40, "O": -0.40, "F": -0.20, "S": -0.20, "CL": -0.20, "BR": -0.10}


def load_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    symbols: list[str] = []
    coords = np.zeros((n, 3), dtype=np.float64)
    for i, line in enumerate(lines[2 : 2 + n]):
        parts = line.split()
        symbols.append(parts[0].upper())
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return symbols, coords


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str = "") -> None:
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            fh.write(f"{s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


def write_pdb(
    path: Path,
    symbols: list[str],
    coords: np.ndarray,
    *,
    n_solute: int,
    cell: np.ndarray,
) -> None:
    with open(path, "w") as fh:
        a, b, c = np.linalg.norm(cell, axis=1)
        fh.write(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n")
        serial = 1
        for i, (s, (x, y, z)) in enumerate(zip(symbols, coords)):
            if i < n_solute:
                res = "LIG"
                name = f"{s}{i+1}"[:4]
                chain = "A"
                resid = 1
            else:
                w = (i - n_solute) // 3
                k = (i - n_solute) % 3
                res = "WAT"
                name = ("O", "H1", "H2")[k]
                chain = "W"
                resid = w + 1
            fh.write(
                f"ATOM  {serial:5d} {name:4s} {res:3s} {chain}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {s:>2s}\n"
            )
            serial += 1
        fh.write("END\n")


def tip3p_molecule(origin: np.ndarray) -> np.ndarray:
    oh = float(TIP3P_PARAMS["OH_r0"])
    ang = math.radians(float(TIP3P_PARAMS["HOH_theta0_deg"]))
    hx = oh * math.sin(ang / 2.0)
    hz = oh * math.cos(ang / 2.0)
    return np.array(
        [
            origin,
            origin + np.array([hx, 0.0, hz]),
            origin + np.array([-hx, 0.0, hz]),
        ],
        dtype=np.float64,
    )


def pack_tip3p_lattice(
    solute_coords: np.ndarray,
    box: float,
    *,
    density: float = 0.033,  # waters / Å³ ≈ 1 g/cm³
    min_dist: float = 2.2,
    seed: int = 0,
) -> np.ndarray:
    """Place waters on a cubic lattice, skipping sites too close to the solute."""
    rng = np.random.default_rng(seed)
    volume = box**3
    target = max(1, int(density * volume))
    spacing = box / max(1, int(round(box * density ** (1.0 / 3.0))))
    # refine spacing from target
    n_edge = max(1, int(math.floor(box / max(spacing, 2.8))))
    spacing = box / n_edge
    origins: list[np.ndarray] = []
    solute_com = solute_coords.mean(axis=0)
    # center solute in box
    shift = np.array([box / 2.0, box / 2.0, box / 2.0]) - solute_com
    solute = solute_coords + shift

    for ix in range(n_edge):
        for iy in range(n_edge):
            for iz in range(n_edge):
                o = np.array(
                    [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing],
                    dtype=np.float64,
                )
                dmin = np.linalg.norm(solute - o[None, :], axis=1).min()
                if dmin < min_dist:
                    continue
                # small random jitter
                o = o + rng.normal(0.0, 0.05, size=3)
                o = np.mod(o, box)
                origins.append(o)
                if len(origins) >= target:
                    break
            if len(origins) >= target:
                break
        if len(origins) >= target:
            break

    waters = np.concatenate([tip3p_molecule(o) for o in origins], axis=0) if origins else np.zeros((0, 3))
    return solute, waters, len(origins)


def write_sponge_stub(prefix: Path, symbols: list[str], coords: np.ndarray) -> None:
    """Write minimal SPONGE GAFF-like files for a vacuum organic solute."""
    n = len(symbols)
    qs = np.array([_ELEMENT_Q.get(s, 0.0) for s in symbols], dtype=np.float64)
    # neutralize
    qs -= qs.mean()
    q_sponge = qs * SPONGE_CHARGE_SCALE

    # connectivity by distance
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            rij = float(np.linalg.norm(coords[i] - coords[j]))
            maxr = 1.20 if "H" in (symbols[i], symbols[j]) else 1.70
            if rij < maxr:
                bonds.append((i, j))

    # angles from bond graph
    graph = [[] for _ in range(n)]
    for a, b in bonds:
        graph[a].append(b)
        graph[b].append(a)
    angles: list[tuple[int, int, int]] = []
    for b in range(n):
        nbrs = graph[b]
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                angles.append((nbrs[ii], b, nbrs[jj]))

    # LJ types per element
    uniq = []
    type_of = []
    for s in symbols:
        if s not in uniq:
            uniq.append(s)
        type_of.append(uniq.index(s))
    n_types = len(uniq)
    # SPONGE LJ: C12 then C6 lower triangles; A = 4εσ^12, B = 4εσ^6
    c12 = np.zeros((n_types, n_types))
    c6 = np.zeros((n_types, n_types))
    for i, si in enumerate(uniq):
        si_sig, si_eps = _ELEMENT_LJ.get(si, (3.5, 0.1))
        for j, sj in enumerate(uniq):
            sj_sig, sj_eps = _ELEMENT_LJ.get(sj, (3.5, 0.1))
            sig = 0.5 * (si_sig + sj_sig)
            eps = math.sqrt(si_eps * sj_eps)
            c12[i, j] = 4.0 * eps * sig**12
            c6[i, j] = 4.0 * eps * sig**6

    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(prefix) + "_charge.txt", "w") as fh:
        fh.write(f"{n}\n")
        for q in q_sponge:
            fh.write(f"{q:.6f}\n")
    with open(str(prefix) + "_mass.txt", "w") as fh:
        fh.write(f"{n}\n")
        for s in symbols:
            fh.write(f"{_ELEMENT_MASS.get(s, 12.0):.6f}\n")
    with open(str(prefix) + "_atomic_number.txt", "w") as fh:
        for s in symbols:
            fh.write(f"{_ELEMENT_Z.get(s, 6)}\n")
    with open(str(prefix) + "_LJ.txt", "w") as fh:
        fh.write(f"{n} {n_types}\n\n")
        for i in range(n_types):
            fh.write("   " + "   ".join(f"{c12[i, j]:.7e}" for j in range(i + 1)) + " \n")
        fh.write("\n")
        for i in range(n_types):
            fh.write("   " + "   ".join(f"{c6[i, j]:.7e}" for j in range(i + 1)) + " \n")
        fh.write("\n")
        for t in type_of:
            fh.write(f"{t}\n")
    with open(str(prefix) + "_bond.txt", "w") as fh:
        fh.write(f"{len(bonds)}\n")
        for a, b in bonds:
            r0 = float(np.linalg.norm(coords[a] - coords[b]))
            fh.write(f"{a} {b} 300.000000 {r0:.6f}\n")
    with open(str(prefix) + "_angle.txt", "w") as fh:
        fh.write(f"{len(angles)}\n")
        for a, b, c in angles:
            va = coords[a] - coords[b]
            vc = coords[c] - coords[b]
            cos_t = float(np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-12))
            cos_t = max(-1.0, min(1.0, cos_t))
            theta = math.acos(cos_t)
            fh.write(f"{a} {b} {c} 50.000000 {theta:.6f}\n")
    # exclude: 1-2 and 1-3
    exclude = [set() for _ in range(n)]
    for a, b in bonds:
        exclude[a].add(b)
        exclude[b].add(a)
    for a, b, c in angles:
        exclude[a].add(c)
        exclude[c].add(a)
    with open(str(prefix) + "_exclude.txt", "w") as fh:
        fh.write(f"{n}\n")
        for i in range(n):
            js = sorted(exclude[i])
            fh.write(f"{len(js)}")
            for j in js:
                fh.write(f" {j}")
            fh.write("\n")
    with open(str(prefix) + "_coordinate.txt", "w") as fh:
        fh.write(f"{n}\n")
        for x, y, z in coords:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz", type=Path, default=DEFAULT_XYZ)
    p.add_argument("--sponge-prefix", type=Path, default=None,
                   help="Existing SPONGE solute prefix; if omitted, write element stub.")
    p.add_argument("--box", type=float, default=32.0, help="Cubic box length (Å)")
    p.add_argument("--density", type=float, default=0.033)
    p.add_argument("--min-dist", type=float, default=2.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=EXPERIMENT_DIR / "mlmm_box")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    symbols, coords = load_xyz(Path(args.xyz))
    n_solute = len(symbols)

    if args.sponge_prefix is None:
        sponge = out / "solute_sponge" / "solute"
        write_sponge_stub(sponge, symbols, coords)
        print(f"[prep] wrote SPONGE stub under {sponge}_*", flush=True)
    else:
        sponge = Path(args.sponge_prefix)
        # validate
        convert_sponge_input_to_dict(str(sponge))

    solute, waters, n_waters = pack_tip3p_lattice(
        coords, float(args.box), density=float(args.density),
        min_dist=float(args.min_dist), seed=int(args.seed),
    )
    water_sym = []
    for _ in range(n_waters):
        water_sym.extend(["O", "H", "H"])
    all_sym = list(symbols) + water_sym
    all_xyz = np.concatenate([solute, waters], axis=0) if n_waters else solute

    cell = np.eye(3, dtype=np.float64) * float(args.box)
    topo = build_topology_from_sponge_and_tip3p(
        str(sponge), n_waters=n_waters, solute_is_ml=True, cell=cell,
    )
    # fix solute Z/mass from symbols if stub
    zs = np.array([_ELEMENT_Z.get(s, 6) for s in symbols] + [8, 1, 1] * n_waters)
    masses = np.array(
        [_ELEMENT_MASS.get(s, 12.0) for s in symbols]
        + [TIP3P_PARAMS["OW_mass"], TIP3P_PARAMS["HW_mass"], TIP3P_PARAMS["HW_mass"]] * n_waters
    )
    topo.atomic_numbers = torch.as_tensor(zs, dtype=torch.long)
    topo.masses = torch.as_tensor(masses, dtype=torch.float64)

    write_xyz(out / "box.xyz", all_sym, all_xyz, comment=f"Claisen ML/MM box L={args.box} n_wat={n_waters}")
    write_pdb(out / "box.pdb", all_sym, all_xyz, n_solute=n_solute, cell=cell)
    np.savetxt(out / "cell.txt", cell)
    topo.save_npz(out / "topology.npz")
    np.savetxt(out / "ml_atom_indices.txt", np.arange(n_solute, dtype=np.int64), fmt="%d")
    with open(out / "README.txt", "w") as fh:
        fh.write(
            "Solvated Claisen ML/MM box\n"
            f"box={args.box} Å  n_solute={n_solute}  n_waters={n_waters}\n"
            "ML region = solute atoms 0..n_solute-1\n"
            "MM = TIP3P water (flexible)\n"
            "Solute MM params: SPONGE prefix (stub or user-provided). "
            "Replace stub with antechamber/SPONGE GAFF export for production.\n"
        )
    print(
        f"[prep] done  atoms={len(all_sym)}  waters={n_waters}  out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
