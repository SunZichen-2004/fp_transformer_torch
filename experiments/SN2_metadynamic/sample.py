#!/usr/bin/env python3
"""Vacuum SN2 (Cl- + CH3Cl) + MACE + 2D distance CV + well-tempered MetaD.

Same force field / CV / walls as ``experiments/SN2_mace``:
  d1 = d(Cl_nuc, C), d2 = d(Cl_lg, C); QuadraticDistanceWall d_max=3.5 Å.
Charge −1, T=300 K, multi-walker shared bias (like rearrangement_metadynamic).
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
SN2_MACE = EXPERIMENT_DIR.parent / "SN2_mace"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SN2_MACE))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import plot_d1_d2, symbols_from_z  # noqa: E402
from fp_transformer_torch import (  # noqa: E402
    MultiParticlePairDistance,
    QuadraticDistanceWall,
    TorchSimWelltemperMetadynamic,
    WellTemperedMetadynamicsBias,
    load_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    atomic_numbers_from_pdb_organic,
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)

DEFAULT_PDB = SN2_MACE / "sn2.pdb"
DEFAULT_CV_FILE = SN2_MACE / "cv_pairs.txt"
DEFAULT_MASS = SN2_MACE / "sn2_mass.txt"
_DEFAULT_MD = SN2_MACE / "init_coords_md_1024.npy"
_DEFAULT_GEOM = SN2_MACE / "init_coords_1024.npy"
DEFAULT_INIT_NPY = _DEFAULT_MD if _DEFAULT_MD.is_file() else _DEFAULT_GEOM
KB_EV_PER_K = 8.617330337217213e-5


def load_pair_indices(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ints: list[int] = []
        for part in line.replace(",", " ").split():
            try:
                ints.append(int(part) - 1)
            except ValueError:
                continue
        if len(ints) >= 2:
            rows.append((ints[0], ints[1]))
    if len(rows) != 2:
        raise ValueError(f"need 2 pairs in {path}, got {len(rows)}")
    return rows


def load_masses(path: Path, num_atoms: int) -> np.ndarray:
    mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
        mass_vals = mass_vals[1:]
    if mass_vals.size < num_atoms:
        raise ValueError(f"mass file too short: {path}")
    return mass_vals[:num_atoms].astype(np.float64)


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "SN2",
    append: bool = False,
    max_frames: int | None = None,
) -> None:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"coords must be (F,N,3) or (N,3), got {arr.shape}")
    if max_frames is not None:
        arr = arr[: int(max_frames)]
    n_atoms = int(arr.shape[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.is_file() else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for i, frame in enumerate(arr):
            fh.write(f"{n_atoms}\n")
            fh.write(f"{comment_prefix} frame={i}\n")
            for sym, (x, y, z) in zip(symbols, frame, strict=True):
                fh.write(f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--pdb", type=str, default=str(DEFAULT_PDB))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=str(DEFAULT_INIT_NPY))
    p.add_argument("--max-xyz-frames", type=int, default=None)
    p.add_argument("--cell-length", type=float, default=30.0)
    p.add_argument("--charge", type=float, default=-1.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=3.5)
    p.add_argument("--wall-k", type=float, default=20.0)
    p.add_argument("--metad-sigma", type=str, default="0.1,0.1", help="σ for (d1,d2) in Å")
    p.add_argument("--metad-height", type=float, default=0.02)
    p.add_argument("--metad-bias-factor", type=float, default=30.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument("--metad-max-hills", type=int, default=200000)
    p.add_argument("--fes-d1-min", type=float, default=1.0)
    p.add_argument("--fes-d1-max", type=float, default=4.0)
    p.add_argument("--fes-d2-min", type=float, default=1.0)
    p.add_argument("--fes-d2-max", type=float, default=4.0)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma

    from torch_sim.models.interface import SumModel

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print(
        f"SN2 MetaD  device={device}  mace={args.mace_model}  "
        f"R={args.replicas}  steps={args.steps}  T={args.temperature}",
        flush=True,
    )

    atomic_numbers = (
        atomic_numbers_from_pdb_organic(Path(args.pdb)).cpu().numpy().astype(np.int64)
    )
    num_atoms = int(atomic_numbers.shape[0])
    masses = load_masses(Path(args.mass_file), num_atoms)
    symbols = symbols_from_z(atomic_numbers)
    pairs = load_pair_indices(Path(args.cv_file))
    print(
        f"[system] N={num_atoms} Z={atomic_numbers.tolist()} charge={args.charge} "
        f"CV pairs(0-based)={pairs}",
        flush=True,
    )

    init_path = Path(args.init_npy) if args.init_npy else None
    if init_path is not None and init_path.is_file():
        loaded = np.load(init_path)
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(f"[init] {init_path.name} → R={args.replicas}", flush=True)
    else:
        raise FileNotFoundError(
            f"need --init-npy (expected {DEFAULT_INIT_NPY}); run sn2_prepare.slurm first"
        )
    if loaded.shape[1] != num_atoms:
        raise ValueError(
            f"init npy n_atoms={loaded.shape[1]} != pdb n_atoms={num_atoms}"
        )
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

    mace_model = load_mace_torch_sim_model(
        args.mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    cv_fn = MultiParticlePairDistance(pairs)
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    print(
        f"[restrain] quadratic wall d_max={args.wall_d_max} Å  k={args.wall_k} eV/Å²",
        flush=True,
    )
    sigma = _parse_floats(args.metad_sigma)
    bias = WellTemperedMetadynamicsBias(
        cv_fn,
        n_cv=2,
        sigma=sigma,
        height=float(args.metad_height),
        bias_factor=float(args.metad_bias_factor),
        temperature=float(args.temperature),
        kB=float(args.kB),
        max_hills=int(args.metad_max_hills),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mace_model, bias, wall)
    driver = TorchSimWelltemperMetadynamic(
        model,
        bias,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    print(
        f"[init] WT-MetaD σ={sigma} Å  w0={args.metad_height} γ={args.metad_bias_factor} "
        f"pace={args.metad_deposit_pace}",
        flush=True,
    )

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

    out = Path(
        args.output
        or (EXPERIMENT_DIR / "results" / f"sn2_metad_{args.mace_model}_r{args.replicas}")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "live").mkdir(parents=True, exist_ok=True)
    traj = out / "trajectory.xyz"
    if traj.is_file():
        traj.unlink()

    cv0 = cv_fn(coordinate).detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    hist_cv = [cv0]
    hist_t = [np.zeros(cv0.shape[0])]

    write_xyz(traj, coordinate.detach().cpu().numpy(), symbols, comment_prefix="step=0")
    np.save(out / "trajectory_cv.npy", cv0)

    g1 = np.linspace(args.fes_d1_min, args.fes_d1_max, 49)
    g2 = np.linspace(args.fes_d2_min, args.fes_d2_max, 49)

    def _all():
        return np.concatenate(hist_cv), np.concatenate(hist_t)

    def _live(done: int):
        cv_a, t_a = _all()
        plot_d1_d2(
            cv_a,
            out / "live" / "d1_d2.png",
            title=f"live d1–d2 step={done}",
            d_max=float(args.wall_d_max),
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(
                t_a, cv_a[:, 0], s=4, alpha=0.35, c="C0",
                label=r"$d_1$ Cl$_{\mathrm{nuc}}$–C", linewidths=0,
            )
            ax.scatter(
                t_a, cv_a[:, 1], s=4, alpha=0.35, c="C1",
                label=r"$d_2$ Cl$_{\mathrm{lg}}$–C", linewidths=0,
            )
            ax.set_xlabel("MD step")
            ax.set_ylabel("distance / Å")
            ax.legend(frameon=False, markerscale=3)
            ax.set_title(f"d1/d2 vs t step={done}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "live" / "d12_vs_t.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] d-vs-t: {exc}", flush=True)

        F = bias.free_energy_grid(g1, g2)
        np.savez(out / "live" / "fes.npz", d1=g1, d2=g2, F=F)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.contourf(g1, g2, F - np.nanmin(F), levels=30, cmap="viridis")
            ax.set_xlabel(r"$d_1$ / Å")
            ax.set_ylabel(r"$d_2$ / Å")
            ax.set_title(f"FES step={done}")
            fig.colorbar(im, ax=ax, label="F / eV")
            fig.tight_layout()
            fig.savefig(out / "live" / "fes_d1_d2.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] FES: {exc}", flush=True)
        np.savez(
            out / "live" / "hills.npz",
            centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
            heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
            n_hills=bias.n_hills,
        )

    _live(0)

    def on_dump(done, state, metrics):
        coords = sim_state_to_batched_coordinates(state)
        cv = cv_fn(coords).detach().cpu().numpy()
        if cv.ndim == 3:
            cv = cv.reshape(cv.shape[0], -1)
        hist_cv.append(cv)
        hist_t.append(np.full(cv.shape[0], float(done)))
        write_xyz(
            traj,
            coords.detach().cpu().numpy(),
            symbols,
            comment_prefix=f"step={done}",
            append=True,
            max_frames=args.max_xyz_frames,
        )
        cv_a, _ = _all()
        np.save(out / "trajectory_cv.npy", cv_a)
        print(
            f"[dump] step={done} n_hills={int(metrics.get('n_hills', 0))}  "
            f"d1 med={np.median(cv[:, 0]):.3f} d2 med={np.median(cv[:, 1]):.3f}",
            flush=True,
        )
        _live(done)

    final = driver.run(
        state0,
        int(args.steps),
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_dump=on_dump,
        dump_every=int(args.dump_every),
    )
    fc = sim_state_to_batched_coordinates(final)
    fcv = cv_fn(fc).detach().cpu().numpy()
    if fcv.ndim == 3:
        fcv = fcv.reshape(fcv.shape[0], -1)
    hist_cv.append(fcv)
    hist_t.append(np.full(fcv.shape[0], float(args.steps)))
    write_xyz(traj, fc.detach().cpu().numpy(), symbols, comment_prefix="final", append=True)
    np.savez(
        out / "final_hills.npz",
        centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
        heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
        n_hills=bias.n_hills,
    )
    _live(int(args.steps))
    print(f"done out={out}", flush=True)


if __name__ == "__main__":
    main()
