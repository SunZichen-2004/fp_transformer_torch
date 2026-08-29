#!/usr/bin/env python3
"""Generate GAFF + TPACM4 charges for the Claisen solute via Xponge.

Writes a SPONGE prefix, then rebuilds ``topology.npz`` for an existing
``mlmm_box`` (keeps ``box.xyz`` / water packing unchanged).

Requires conda env with Xponge (e.g. MACE_torchsim)::

    python experiments/rearrangement_mace/prepare_solute_gaff_xponge.py \\
        --box-dir experiments/rearrangement_water_metadynamics/mlmm_box

Charge method: TPACM4 (GAFF-compatible empirical). RESP needs a QM backend;
Gasteiger needs rdkit (not installed in MACE_torchsim).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MACE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_MACE))

from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    TIP3P_PARAMS,
    build_topology_from_sponge_and_tip3p,
)
from prepare_mlmm_box import (  # noqa: E402
    _ELEMENT_MASS,
    _ELEMENT_Z,
    load_xyz,
)

DEFAULT_XYZ = ROOT / "experiments/rearrangement_mace/allyl_phenyl_ether.xyz"
DEFAULT_BOX = ROOT / "experiments/rearrangement_water_metadynamics/mlmm_box"


def xponge_gaff_to_sponge(
    xyz: Path,
    sponge_prefix: Path,
    *,
    charge_method: str = "TPACM4",
    total_charge: float = 0.0,
    resname: str = "CLAI",
) -> None:
    """Run Xponge GAFF typing + charge assignment → SPONGE input files."""
    from Xponge import Molecule
    from Xponge.forcefield.amber.gaff import (  # noqa: F401 — loads gaff.dat
        Get_Assignment_From_Xyz,
        Save_SPONGE_Input,
    )

    assign = Get_Assignment_From_Xyz(str(xyz))
    assign.Determine_Ring_And_Bond_Type()
    # TPACM4 must run before GAFF Determine_Atom_Type (typing conflict → all-zero q).
    method = charge_method.strip().upper()
    if method == "TPACM4":
        assign.Calculate_Charge("TPACM4", charge=int(round(total_charge)))
    elif method == "GASTEIGER":
        assign.Calculate_Charge("GASTEIGER")
    elif method == "RESP":
        assign.Calculate_Charge("RESP")
    else:
        raise ValueError(f"unsupported charge method {charge_method!r}")
    assign.Determine_Atom_Type("gaff")

    sponge_prefix = Path(sponge_prefix)
    sponge_prefix.parent.mkdir(parents=True, exist_ok=True)
    rt = assign.To_ResidueType(name=resname)
    mol = Molecule(resname)
    mol.Add_Residue(rt)
    Save_SPONGE_Input(mol, str(sponge_prefix))

    # Extra files our topology loader uses when present
    symbols, _ = load_xyz(xyz)
    with open(str(sponge_prefix) + "_atomic_number.txt", "w") as fh:
        for s in symbols:
            fh.write(f"{_ELEMENT_Z.get(s.upper(), 6)}\n")

    q = np.asarray(assign.charge, dtype=np.float64)
    types = [str(assign.atom_types[i]).split()[-1] for i in range(assign.atom_numbers)]
    print(f"[xponge] GAFF types: {types}", flush=True)
    print(f"[xponge] charges (e): {np.round(q, 4).tolist()}  sum={q.sum():.3e}", flush=True)
    print(f"[xponge] wrote SPONGE prefix {sponge_prefix}_*", flush=True)


def rebuild_topology(box_dir: Path, sponge_prefix: Path, xyz: Path) -> None:
    box_dir = Path(box_dir)
    old = np.load(box_dir / "topology.npz")
    n_ml = int(old["ml_mask"].sum())
    n_wat = (int(old["charges"].shape[0]) - n_ml) // 3
    cell = np.loadtxt(box_dir / "cell.txt", dtype=np.float64).reshape(3, 3)

    topo = build_topology_from_sponge_and_tip3p(
        str(sponge_prefix),
        n_waters=n_wat,
        solute_is_ml=True,
        cell=cell,
    )
    symbols, _ = load_xyz(xyz)
    zs = [_ELEMENT_Z.get(s.upper(), 6) for s in symbols] + [8, 1, 1] * n_wat
    ms = (
        [_ELEMENT_MASS.get(s.upper(), 12.0) for s in symbols]
        + [
            float(TIP3P_PARAMS["OW_mass"]),
            float(TIP3P_PARAMS["HW_mass"]),
            float(TIP3P_PARAMS["HW_mass"]),
        ]
        * n_wat
    )
    topo.atomic_numbers = torch.as_tensor(zs, dtype=torch.long)
    topo.masses = torch.as_tensor(ms, dtype=torch.float64)
    topo.save_npz(box_dir / "topology.npz")

    with open(box_dir / "README.txt", "w") as fh:
        fh.write(
            "Solvated Claisen ML/MM box\n"
            f"box={cell[0, 0]} Å  n_solute={n_ml}  n_waters={n_wat}\n"
            "ML region = solute atoms 0..n_solute-1\n"
            "MM = TIP3P water (flexible)\n"
            f"Solute MM params: Xponge GAFF + TPACM4 ({sponge_prefix.name}_*).\n"
            "Mechanical embedding: solute bonded/nb14 dropped; charges/LJ kept for ML–MM.\n"
        )
    print(
        f"[topo] updated {box_dir / 'topology.npz'}  "
        f"ML q sum={float(topo.charges[:n_ml].sum()):.3e}  "
        f"sigma[O]={float(topo.sigma[6]):.4f}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz", type=Path, default=DEFAULT_XYZ)
    p.add_argument(
        "--box-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_BOX, ROOT / "experiments/rearrangement_water_fpt/mlmm_box"],
        help="mlmm_box directories to refresh (coordinates kept)",
    )
    p.add_argument("--charge-method", type=str, default="TPACM4")
    p.add_argument("--total-charge", type=float, default=0.0)
    args = p.parse_args()

    boxes = [Path(b) for b in args.box_dir]
    primary = boxes[0]
    sponge = primary / "solute_gaff" / "solute"
    xponge_gaff_to_sponge(
        Path(args.xyz),
        sponge,
        charge_method=args.charge_method,
        total_charge=float(args.total_charge),
    )

    for box in boxes:
        box.mkdir(parents=True, exist_ok=True)
        dst = box / "solute_gaff"
        if dst.resolve() != sponge.parent.resolve():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(sponge.parent, dst)
        rebuild_topology(box, dst / "solute", Path(args.xyz))


if __name__ == "__main__":
    main()
