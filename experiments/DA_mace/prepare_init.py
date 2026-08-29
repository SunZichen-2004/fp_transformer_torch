#!/usr/bin/env python3
"""DA (tropone + dimethylfulvene) prep: FIRE min → NVT MD → FPT init.

MACE-MH-0, charge 0, vacuum. Soft-distance 2D CV ``(s1,s2)`` with pairs
2–15 / 3–13 / 7–24 (1-based). Quadratic wall on raw distances > ``wall_d_max``.
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

from io_utils import (  # noqa: E402
    atomic_numbers_from_symbols,
    load_xyz,
    masses_from_symbols,
    save_stage_artifacts,
)
from fp_transformer_torch import (  # noqa: E402
    QuadraticDistanceWall,
    SoftDistanceS12,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402

DEFAULT_XYZ = EXPERIMENT_DIR / "DA_tropone_dimethylfulvene.xyz"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
DEFAULT_MASS = EXPERIMENT_DIR / "da_mass.txt"
DEFAULT_INIT_NPY = EXPERIMENT_DIR / "init_coords_1024.npy"


def load_pair_indices(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        ints: list[int] = []
        for part in parts:
            try:
                ints.append(int(part) - 1)
            except ValueError:
                continue
        if len(ints) >= 2:
            rows.append((ints[0], ints[1]))
    if len(rows) != 3:
        raise ValueError(f"need exactly 3 pairs in {path}, got {len(rows)}")
    return rows


def load_masses(path: Path, num_atoms: int) -> np.ndarray:
    mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
        mass_vals = mass_vals[1:]
    if mass_vals.size < num_atoms:
        raise ValueError(f"mass file too short: {path}")
    return mass_vals[:num_atoms].astype(np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--xyz", type=str, default=str(DEFAULT_XYZ))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--init-npy", type=str, default=str(DEFAULT_INIT_NPY))
    p.add_argument("--replicas", type=int, default=1024)
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--md-steps", type=int, default=2000)
    p.add_argument("--md-dump-every", type=int, default=200)
    p.add_argument("--fpt-init-from-last-dumps", type=int, default=3)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--cell-length", type=float, default=40.0)
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--d-ref", type=float, default=2.0)
    p.add_argument("--soft-n", type=int, default=6)
    p.add_argument("--soft-m", type=int, default=8)
    p.add_argument("--wall-d-max", type=float, default=4.5)
    p.add_argument("--wall-k", type=float, default=20.0)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--fpt-init-out",
        type=Path,
        default=EXPERIMENT_DIR / "init_coords_md_1024.npy",
    )
    p.add_argument("--max-xyz-frames", type=int, default=None)
    return p


def _cv_numpy(cv_fn, coords_t: torch.Tensor) -> np.ndarray:
    return cv_fn(coords_t).detach().cpu().numpy().reshape(-1, 2)


def main() -> None:
    args = build_parser().parse_args()
    import torch_sim as ts
    from torch_sim.models.interface import SumModel

    primary = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if args.devices:
        mace_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mace_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mace_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    out = Path(
        args.output
        or (
            EXPERIMENT_DIR
            / "results"
            / f"da_prepare_{args.mace_model}_g{len(mace_devices)}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "s1_s2").mkdir(parents=True, exist_ok=True)

    symbols, _ = load_xyz(Path(args.xyz))
    atomic_numbers = atomic_numbers_from_symbols(symbols)
    num_atoms = int(atomic_numbers.shape[0])
    if Path(args.mass_file).is_file():
        masses = load_masses(Path(args.mass_file), num_atoms)
    else:
        masses = masses_from_symbols(symbols)

    pairs = load_pair_indices(Path(args.cv_file))
    print(
        f"device={device}  mace={args.mace_model}  charge={args.charge}  "
        f"n_gpus={len(mace_devices)}  atoms={num_atoms}  pairs={pairs}  "
        f"d_ref={args.d_ref}",
        flush=True,
    )

    loaded = np.load(Path(args.init_npy))
    if loaded.ndim != 3 or loaded.shape[1] != num_atoms:
        raise ValueError(f"bad init npy {args.init_npy}: {loaded.shape}")
    loaded = loaded[: args.replicas]
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

    cv_fn = SoftDistanceS12(
        pairs, d_ref=float(args.d_ref), n=int(args.soft_n), m=int(args.soft_m)
    )
    save_stage_artifacts(
        out,
        "stage_geom_init",
        loaded.astype(np.float32),
        _cv_numpy(cv_fn, coordinate),
        symbols,
        title="DA geom init",
        max_xyz_frames=args.max_xyz_frames,
    )

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mace_model, wall)
    print("[init] SumModel(MACE, QuadraticDistanceWall)", flush=True)

    mass_t = torch.as_tensor(masses, dtype=dtype, device=device)
    z_t = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
    cell = torch.eye(3, device=device, dtype=dtype) * float(args.cell_length)
    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=False,
        charge=float(args.charge),
        spin=float(args.spin),
    )

    def _never_converge(state, last_energy):  # noqa: ARG001
        return torch.zeros(state.n_systems, dtype=torch.bool, device=state.device)

    print(f"[min] FIRE max_steps={args.min_steps}", flush=True)
    state_min = ts.optimize(
        state0,
        model,
        optimizer=ts.Optimizer.fire,
        max_steps=int(args.min_steps),
        steps_between_swaps=max(1, min(5, int(args.min_steps))),
        convergence_fn=_never_converge,
        autobatcher=False,
        pbar=True,
    )
    coords_min = sim_state_to_batched_coordinates(state_min)
    save_stage_artifacts(
        out,
        "stage_min",
        coords_min.detach().cpu().numpy(),
        _cv_numpy(cv_fn, coords_min),
        symbols,
        title=f"DA after FIRE min ({args.min_steps} steps)",
        max_xyz_frames=args.max_xyz_frames,
    )

    state = batched_coordinates_to_sim_state(
        coords_min.detach(),
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=False,
        charge=float(args.charge),
        spin=float(args.spin),
    )

    dump_every = max(1, int(args.md_dump_every))
    md_steps = int(args.md_steps)
    n_chunks = (md_steps + dump_every - 1) // dump_every
    dumps: list[np.ndarray] = []
    done = 0
    print(
        f"[md] NVT Langevin T={args.temperature} K  dt={args.timestep} ps  "
        f"γ={args.langevin_friction}/ps  steps={md_steps}",
        flush=True,
    )
    for _ in range(n_chunks):
        n_this = min(dump_every, md_steps - done)
        state = ts.integrate(
            state,
            model,
            integrator=ts.Integrator.nvt_langevin,
            n_steps=n_this,
            temperature=float(args.temperature),
            timestep=float(args.timestep),
            gamma=float(args.langevin_friction),
            autobatcher=False,
            pbar=False,
        )
        done += n_this
        coords = sim_state_to_batched_coordinates(state)
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
        dumps.append(coords_np)
        tag = f"stage_md_step{done:05d}"
        save_stage_artifacts(
            out,
            tag,
            coords_np,
            _cv_numpy(cv_fn, coords),
            symbols,
            title=f"DA MD step {done}",
            max_xyz_frames=args.max_xyz_frames,
        )
        print(f"[md] step={done}", flush=True)

    traj = np.stack(dumps, axis=0)
    np.save(out / "md_traj.npy", traj)

    k = min(max(1, int(args.fpt_init_from_last_dumps)), traj.shape[0])
    pool = traj[-k:].reshape(-1, num_atoms, 3)
    idx = np.linspace(0, pool.shape[0] - 1, args.replicas).astype(np.int64)
    fpt_init = pool[idx].astype(np.float32)
    np.save(out / "init_coords_for_fpt.npy", fpt_init)
    fpt_out = Path(args.fpt_init_out)
    fpt_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(fpt_out, fpt_init)
    save_stage_artifacts(
        out,
        "stage_fpt_init",
        fpt_init,
        _cv_numpy(cv_fn, torch.as_tensor(fpt_init, dtype=dtype, device=device)),
        symbols,
        title=f"DA FPT init (last {k} MD dumps)",
        max_xyz_frames=args.max_xyz_frames,
    )
    print(f"[done] FPT init → {fpt_out}  shape={fpt_init.shape}", flush=True)


if __name__ == "__main__":
    main()
