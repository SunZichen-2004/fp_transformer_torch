#!/usr/bin/env python3
"""Profile MechanicalEmbeddingMLMM term costs (PME default).

Reports mean ms / forward for MACE, neighbor list, PME, exclusion, LJ, bonded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from fp_transformer_torch import load_sharded_mace_torch_sim_model  # noqa: E402
from fp_transformer_torch.potential.mace import batched_coordinates_to_sim_state  # noqa: E402
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402


def _load_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--box-dir",
        type=Path,
        default=ROOT
        / "experiments/rearrangement_water_metadynamics/mlmm_box",
    )
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--replicas", type=int, default=1024)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--pme-cutoff", type=float, default=9.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument(
        "--electrostatics",
        type=str,
        default=None,
        choices=["pme", "dsf", "torchpme"],
        help="Default: pme on 1 GPU, torchpme when --n-gpus>1.",
    )
    p.add_argument(
        "--torchpme-method", type=str, default="p3m", choices=["p3m", "pme"]
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--no-mace", action="store_true")
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this profile")
    mm_devices = parse_cuda_devices(args.devices or args.n_gpus)
    device = mm_devices[0]
    if args.electrostatics is None:
        args.electrostatics = "torchpme" if len(mm_devices) > 1 else "pme"

    box = Path(args.box_dir)
    topo = MLMMTopology.load_npz(box / "topology.npz").to(device, dtype)
    cell = np.loadtxt(box / "cell.txt", dtype=np.float64).reshape(3, 3)
    _, coords = _load_xyz(box / "box.xyz")
    n_per = coords.shape[0]
    mace_devices = mm_devices
    print(
        f"profile ML/MM  N={n_per}  R={args.replicas}  elec={args.electrostatics}  "
        f"cutoff={args.pme_cutoff}  mesh={args.pme_mesh_spacing}  dtype={args.dtype}  "
        f"device={device}  mm_gpus={len(mm_devices)}",
        flush=True,
    )

    mace_model = None
    if not args.no_mace:
        mace_model = load_sharded_mace_torch_sim_model(
            args.mace_model,
            devices=mace_devices,
            dtype=dtype,
            compute_forces=True,
            compute_stress=False,
            cache_dir=args.mace_cache,
        )
        print(f"MACE devices={[str(d) for d in mace_devices]}", flush=True)

    mm_kw = dict(
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        pme_accuracy=1e-4,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        torchpme_method=str(args.torchpme_method),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        enable_mace=not args.no_mace,
        mm_devices=mm_devices if len(mm_devices) > 1 else None,
    )
    mlmm = MechanicalEmbeddingMLMM(topo, mace_model=mace_model, **mm_kw)
    print(f"MM workers={len(mlmm._mm_workers)} (0 → all MM on {device})", flush=True)

    batch = np.repeat(coords[None, ...], args.replicas, axis=0)
    coordinate = torch.as_tensor(batch, dtype=dtype, device=device)
    state = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=topo.atomic_numbers.to(device),
        masses=topo.masses.to(device=device, dtype=dtype),
        cell=cell,
        pbc=True,
        charge=0.0,
        spin=1.0,
    )

    ms = mlmm.profile_forward(state, warmup=args.warmup, repeats=args.repeats)
    total = ms["forward_total"]
    print("\n=== mean ms / call (CUDA sync) ===", flush=True)
    order = [
        "mace",
        "mm_sharded",
        "neighbor_list",
        "pme_or_dsf",
        "pme_exclusion",
        "ml_coulomb_corr",
        "lj",
        "bonded",
        "forward_total",
    ]
    for k in order:
        v = ms[k]
        pct = 100.0 * v / total if total > 0 and k != "forward_total" else None
        if pct is None:
            print(f"{k:18s}  {v:8.2f} ms   ({1000.0/v:.2f} it/s equiv)", flush=True)
        else:
            print(f"{k:18s}  {v:8.2f} ms   ({pct:5.1f}% of forward)", flush=True)

    parts = {k: ms[k] for k in order if k != "forward_total"}
    slowest = max(parts, key=parts.get)
    print(
        f"\nSLOWEST: {slowest} = {parts[slowest]:.2f} ms "
        f"({100.0 * parts[slowest] / max(total, 1e-9):.1f}% of forward)",
        flush=True,
    )


if __name__ == "__main__":
    main()
