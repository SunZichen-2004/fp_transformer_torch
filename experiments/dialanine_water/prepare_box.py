#!/usr/bin/env python3
"""Build a small solvated dialanine box: GAFF (SPONGE) + TIP3P, full classical MM.

Uses existing ``experiments/dialanine/dialanine_*`` parameters with
``solute_is_ml=False`` so all solute bonded/NB14 terms are kept (no MACE).

Default cubic box ~18 Å (tight padding around ACE-ALA-NME).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DIAL = EXP.parent / "dialanine"
MACE_EXP = EXP.parent / "rearrangement_mace"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MACE_EXP))

from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    TIP3P_PARAMS,
    build_topology_from_sponge_and_tip3p,
)
from prepare_mlmm_box import (  # noqa: E402
    load_xyz,
    pack_tip3p_lattice,
    write_pdb,
    write_xyz,
)

DEFAULT_XYZ = DIAL / "dialanine.xyz"
DEFAULT_SPONGE = DIAL / "dialanine"
DEFAULT_OUT = EXP / "mm_box"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz", type=Path, default=DEFAULT_XYZ)
    p.add_argument("--sponge-prefix", type=Path, default=DEFAULT_SPONGE)
    p.add_argument(
        "--box",
        type=float,
        default=18.0,
        help="Cubic box length (Å); keep small (≥ ~2× solute extent)",
    )
    p.add_argument("--density", type=float, default=0.033, help="waters / Å³")
    p.add_argument("--min-dist", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    symbols, coords = load_xyz(Path(args.xyz))
    n_solute = len(symbols)
    ext = coords.max(axis=0) - coords.min(axis=0)
    print(
        f"[prep] solute N={n_solute}  extent={ext}  box={args.box} Å",
        flush=True,
    )
    if float(args.box) < float(np.max(ext)) + 4.0:
        print(
            f"[warn] box={args.box} may be tight vs extent max={float(np.max(ext)):.2f}",
            flush=True,
        )

    sponge = Path(args.sponge_prefix)
    solute, waters, n_waters = pack_tip3p_lattice(
        coords,
        float(args.box),
        density=float(args.density),
        min_dist=float(args.min_dist),
        seed=int(args.seed),
    )
    water_sym: list[str] = []
    for _ in range(n_waters):
        water_sym.extend(["O", "H", "H"])
    all_sym = list(symbols) + water_sym
    all_xyz = np.concatenate([solute, waters], axis=0) if n_waters else solute
    cell = np.eye(3, dtype=np.float64) * float(args.box)

    topo = build_topology_from_sponge_and_tip3p(
        str(sponge),
        n_waters=n_waters,
        solute_is_ml=False,  # full classical GAFF solute
        cell=cell,
    )
    # Prefer organic PDB Z / mass files when present
    z_path = Path(str(sponge) + "_atomic_number.txt")
    if z_path.is_file():
        zs_sol = np.loadtxt(z_path, dtype=np.int64).reshape(-1)[:n_solute]
    else:
        from prepare_mlmm_box import _ELEMENT_Z

        zs_sol = np.array([_ELEMENT_Z.get(s.upper(), 6) for s in symbols])
    mass_path = Path(str(sponge) + "_mass.txt")
    if mass_path.is_file():
        m = np.loadtxt(mass_path, dtype=np.float64).reshape(-1)
        if int(m[0]) == n_solute and m.size == n_solute + 1:
            m = m[1:]
        ms_sol = m[:n_solute]
    else:
        from prepare_mlmm_box import _ELEMENT_MASS

        ms_sol = np.array([_ELEMENT_MASS.get(s.upper(), 12.0) for s in symbols])

    zs = np.concatenate([zs_sol, np.array([8, 1, 1] * n_waters, dtype=np.int64)])
    ms = np.concatenate(
        [
            ms_sol,
            np.array(
                [TIP3P_PARAMS["OW_mass"], TIP3P_PARAMS["HW_mass"], TIP3P_PARAMS["HW_mass"]]
                * n_waters,
                dtype=np.float64,
            ),
        ]
    )
    topo.atomic_numbers = torch.as_tensor(zs, dtype=torch.long)
    topo.masses = torch.as_tensor(ms, dtype=torch.float64)

    write_xyz(
        out / "box.xyz",
        all_sym,
        all_xyz,
        comment=f"dialanine GAFF+TIP3P L={args.box} n_wat={n_waters}",
    )
    write_pdb(out / "box.pdb", all_sym, all_xyz, n_solute=n_solute, cell=cell)
    np.savetxt(out / "cell.txt", cell)
    topo.save_npz(out / "topology.npz")
    # CV file (1-based φ/ψ) — same as vacuum dialanine
    cv_src = DIAL / "backbone_cv.txt"
    if cv_src.is_file():
        (out / "backbone_cv.txt").write_text(cv_src.read_text())

    with open(out / "README.txt", "w") as fh:
        fh.write(
            "Solvated dialanine — full classical MM (GAFF + TIP3P)\n"
            f"box={args.box} Å  n_solute={n_solute}  n_waters={n_waters}  "
            f"n_atoms={len(all_sym)}\n"
            f"solute SPONGE: {sponge}\n"
            "solute_is_ml=False (all bonded/NB14 kept)\n"
            "MechanicalEmbeddingMLMM with enable_mace=False\n"
        )
    print(
        f"[prep] done  atoms={len(all_sym)}  waters={n_waters}  "
        f"ml_mask.any={bool(topo.ml_mask.any())}  out={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
