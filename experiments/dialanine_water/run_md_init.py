#!/usr/bin/env python3
"""Unbiased NVT Langevin MD for solvated dialanine → sFPT init ensemble.

GAFF + TIP3P via ``MechanicalEmbeddingMLMM(enable_mace=False)``. Starts from
``mm_box/box.xyz``, dumps full-box frames after equilibration discard, then
evenly subsamples into ``init_coords_{n_init}.npy`` for sFPT.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import load_xyz, plot_phi_psi  # noqa: E402
from fp_transformer_torch import Torsion  # noqa: E402
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)

DEFAULT_BOX = EXPERIMENT_DIR / "mm_box"
DEFAULT_CV = DEFAULT_BOX / "backbone_cv.txt"
N_SOLUTE = 22


def load_torsion_indices(path: Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        chunk: list[int] = []
        for part in parts:
            try:
                chunk.append(int(part) - 1)
            except ValueError:
                continue
            if len(chunk) == 4:
                rows.append(chunk)
                chunk = []
    if not rows:
        raise ValueError(f"no torsion indices in {path}")
    return np.asarray(rows, dtype=np.int64)


def _attach_charges(state, charges: torch.Tensor, n_replicas: int):
    import torch_sim as ts

    extras = dict(state.system_extras)
    atom_extras = dict(state.atom_extras)
    atom_extras["partial_charges"] = charges.repeat(n_replicas)
    return ts.SimState(
        positions=state.positions,
        masses=state.masses,
        cell=state.row_vector_cell,
        pbc=True,
        atomic_numbers=state.atomic_numbers,
        system_idx=state.system_idx,
        _system_extras=extras,
        _atom_extras=atom_extras,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=Path, default=DEFAULT_CV)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--dump-every", type=int, default=3)
    p.add_argument("--discard-steps", type=int, default=2000)
    p.add_argument("--n-init", type=int, default=1024)
    p.add_argument("--chunk-steps", type=int, default=100, help="integrate chunk size")
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument("--electrostatics", type=str, default="pme", choices=["pme", "dsf"])
    p.add_argument("--dsf-alpha", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "md_init",
    )
    args = p.parse_args()

    import torch_sim as ts

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    box = Path(args.box_dir)
    topo_path = box / "topology.npz"
    if not topo_path.is_file():
        raise FileNotFoundError(f"missing {topo_path}; run prepare_box.py first")

    symbols, xyz_coords = load_xyz(box / "box.xyz")
    topo = MLMMTopology.load_npz(topo_path).to(device, dtype)
    cell = torch.as_tensor(
        np.loadtxt(box / "cell.txt").reshape(3, 3), dtype=dtype, device=device
    )
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(f"xyz N={xyz_coords.shape[0]} != topology N={num_atoms}")

    tors = load_torsion_indices(Path(args.cv_file))[:2]
    cv_fn = Torsion(tors)

    mm = MechanicalEmbeddingMLMM(
        topo,
        mace_model=None,
        enable_mace=False,
        enable_ml_coulomb_correction=False,
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        dsf_alpha=float(args.dsf_alpha),
        pme_accuracy=1e-4,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    coordinate = torch.as_tensor(xyz_coords[None, ...], dtype=dtype, device=device)
    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    state = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
    )
    state = _attach_charges(state, topo.charges.to(device=device, dtype=dtype), 1)

    n_steps = int(args.steps)
    dump_every = max(1, int(args.dump_every))
    discard = max(0, int(args.discard_steps))
    chunk = max(1, int(args.chunk_steps))

    frames: list[np.ndarray] = []
    dihedrals: list[np.ndarray] = []
    print(
        f"[md] dialanine-water  N={num_atoms}  steps={n_steps}  "
        f"dump_every={dump_every}  discard={discard}  device={device}  "
        f"box={float(cell[0, 0]):.1f} Å",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={tors.tolist()}", flush=True)

    def _integrate(n: int):
        nonlocal state
        if n <= 0:
            return
        # chunk long stretches to keep progress logs responsive
        left = int(n)
        while left > 0:
            take = min(chunk, left)
            state = ts.integrate(
                state,
                mm,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=int(take),
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                pbar=False,
            )
            left -= take

    done = 0
    if discard > 0:
        eq = min(discard, n_steps)
        print(f"[md] equilibrating {eq} steps (no dump)...", flush=True)
        _integrate(eq)
        done = eq
        print(f"[md] equilibration done step={done}", flush=True)

    while done < n_steps:
        n_run = min(dump_every, n_steps - done)
        _integrate(n_run)
        done += n_run
        coords = sim_state_to_batched_coordinates(state)
        frames.append(coords[0].detach().cpu().numpy().astype(np.float32))
        cv = cv_fn(coords).detach().cpu().numpy().reshape(-1)
        dihedrals.append(cv.astype(np.float64))
        if (
            len(frames) == 1
            or done % max(1, n_steps // 10) < dump_every
            or done == n_steps
        ):
            print(f"[md] step={done}/{n_steps}  n_frames={len(frames)}", flush=True)

    if not frames:
        raise RuntimeError("no frames dumped; check discard/dump-every/steps")

    traj = np.stack(frames, axis=0)
    dih = np.stack(dihedrals, axis=0)
    traj_path = out / f"md_traj_s{args.seed}_{n_steps}.npy"
    dih_path = out / f"md_dihedral_s{args.seed}_{n_steps}.npy"
    np.save(traj_path, traj)
    np.save(dih_path, dih)
    print(f"[md] wrote {traj_path}  shape={traj.shape}", flush=True)

    n_init = int(args.n_init)
    if traj.shape[0] < n_init:
        raise RuntimeError(
            f"only {traj.shape[0]} frames after discard; need n_init={n_init}. "
            f"Increase --steps or lower --dump-every/--discard-steps."
        )
    idx = np.linspace(0, traj.shape[0] - 1, n_init, dtype=np.int64)
    init = traj[idx]
    init_path = EXPERIMENT_DIR / f"init_coords_{n_init}.npy"
    np.save(init_path, init.astype(np.float32))
    np.save(out / f"init_coords_{n_init}.npy", init.astype(np.float32))
    np.save(out / f"init_dihedral_{n_init}.npy", dih[idx].astype(np.float64))
    print(
        f"[md] wrote init ensemble {init_path}  shape={init.shape}  "
        f"idx[0..3]={idx[:4].tolist()}",
        flush=True,
    )
    plot_phi_psi(
        dih[idx],
        out / f"init_phi_psi_{n_init}.png",
        title=f"MD init φ–ψ  N={n_init}",
    )
    # solute-only convenience plot of last frame CV already covered
    print("[md] done", flush=True)


if __name__ == "__main__":
    main()
