#!/usr/bin/env python3
"""Allyl phenyl ether (Claisen) + MACE + 2D distance CV + well-tempered MetaD.

CVs (Å): d1=d(O7,C8), d2=d(C1,C10). Multi-walker shared bias;
trajectory + live d1–d2 / FES plots.
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
MACE_EXP = EXPERIMENT_DIR.parent / "rearrangement_mace"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MACE_EXP))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import (  # noqa: E402
    atomic_numbers_from_symbols,
    load_xyz,
    masses_from_symbols,
    plot_cv12,
    write_xyz,
)
from fp_transformer_torch import (  # noqa: E402
    MultiParticlePairDistance,
    QuadraticDistanceWall,
    SoftminDistanceUpperWall,
    TorchSimWelltemperMetadynamic,
    WellTemperedMetadynamicsBias,
    load_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)

DEFAULT_XYZ = EXPERIMENT_DIR / "allyl_phenyl_ether.xyz"
DEFAULT_CV_FILE = MACE_EXP / "cv_pairs.txt"
DEFAULT_MASS = MACE_EXP / "mass.txt"
KB_EV_PER_K = 8.617330337217213e-5


def load_pair_indices(path: Path) -> list[tuple[int, int]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ints = []
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


def load_masses(path: Path, num_atoms: int, symbols: list[str]) -> np.ndarray:
    if path.is_file():
        vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if int(vals[0]) == num_atoms and vals.size == num_atoms + 1:
            vals = vals[1:]
        return vals[:num_atoms].astype(np.float64)
    return masses_from_symbols(symbols)


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--xyz", type=str, default=str(DEFAULT_XYZ))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--max-xyz-frames", type=int, default=None)
    p.add_argument("--cell-length", type=float, default=40.0)
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--softmin-a", type=float, default=10.0, help="softmin a in 1/Å")
    p.add_argument("--softmin-d-wall", type=float, default=2.7, help="softmin wall in Å")
    p.add_argument("--softmin-k", type=float, default=10.0, help="softmin wall k in eV/Å²")
    p.add_argument("--metad-sigma", type=str, default="0.1,0.2", help="σ for (d1,d2) in Å")
    p.add_argument("--metad-height", type=float, default=0.02)
    p.add_argument("--metad-bias-factor", type=float, default=30.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument(
        "--metad-max-hills",
        type=int,
        default=200000,
        help="hill buffer cap; <=0 means unlimited (grow on demand)",
    )
    p.add_argument(
        "--resume-hills",
        type=str,
        default=None,
        help="load prior hills.npz and continue depositing (FES from hills)",
    )
    p.add_argument(
        "--live-dir",
        type=str,
        default="live",
        help="subdir under --output for fes/hills/plots (e.g. live_rerun)",
    )
    p.add_argument("--fes-d1-min", type=float, default=1.0)
    p.add_argument("--fes-d1-max", type=float, default=4.0)
    p.add_argument("--fes-d2-min", type=float, default=1.0)
    p.add_argument("--fes-d2-max", type=float, default=6.0)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    return p


def _load_last_xyz_frames(path: Path, n_frames: int) -> tuple[list[str], np.ndarray]:
    """Return ``(symbols, coords[n_frames,N,3])`` from the end of a multi-frame XYZ."""
    text = Path(path).read_text().splitlines()
    frames: list[np.ndarray] = []
    symbols: list[str] | None = None
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        try:
            nat = int(line)
        except ValueError:
            i += 1
            continue
        i += 2  # skip comment
        syms: list[str] = []
        xyz = np.zeros((nat, 3), dtype=np.float64)
        for a in range(nat):
            parts = text[i + a].split()
            syms.append(parts[0])
            xyz[a] = [float(parts[1]), float(parts[2]), float(parts[3])]
        if symbols is None:
            symbols = syms
        frames.append(xyz)
        i += nat
    if not frames or symbols is None:
        raise ValueError(f"no frames in {path}")
    take = frames[-int(n_frames) :]
    if len(take) < n_frames:
        # tile last frame if short
        while len(take) < n_frames:
            take.append(take[-1].copy())
    return symbols, np.stack(take[-n_frames:], axis=0).astype(np.float64)


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
        f"Claisen MetaD  device={device}  mace={args.mace_model}  "
        f"R={args.replicas}  steps={args.steps}",
        flush=True,
    )

    symbols, xyz_coords = load_xyz(Path(args.xyz))
    atomic_numbers = atomic_numbers_from_symbols(symbols)
    num_atoms = len(symbols)
    masses = load_masses(Path(args.mass_file), num_atoms, symbols)
    pairs = load_pair_indices(Path(args.cv_file))
    print(f"[CV] distances (d1,d2) pairs(0-based)={pairs} Å", flush=True)

    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(f"[init] from npy {args.init_npy} → R={args.replicas}", flush=True)
    elif args.resume_hills:
        # Prefer last walkers from the prior trajectory next to hills
        hills_p = Path(args.resume_hills)
        cand_traj = hills_p.parent.parent / "trajectory.xyz"
        if cand_traj.is_file():
            _, loaded = _load_last_xyz_frames(cand_traj, int(args.replicas))
            print(
                f"[init] resume from last {args.replicas} frames of {cand_traj}",
                flush=True,
            )
        else:
            loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
            print(f"[init] resume hills but no traj; box xyz × {args.replicas}", flush=True)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        print(f"[init] exact xyz × {args.replicas} (no noise)", flush=True)
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

    mace_model = load_mace_torch_sim_model(
        args.mace_model, device=device, dtype=dtype,
        compute_forces=True, compute_stress=False, cache_dir=args.mace_cache,
    )
    cv_fn = MultiParticlePairDistance(pairs)
    wall = QuadraticDistanceWall(
        pairs, d_max=float(args.wall_d_max), k=float(args.wall_k),
        device=device, dtype=dtype, n_atoms_per_system=num_atoms,
    )
    if len(pairs) < 2:
        raise ValueError(f"softmin restraint needs ≥2 CV pairs, got {pairs}")
    softmin_wall = SoftminDistanceUpperWall(
        pairs[0], pairs[1],
        a=float(args.softmin_a), d_wall=float(args.softmin_d_wall),
        k=float(args.softmin_k), device=device, dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    print(
        f"[restrain] softmin upper wall  a={args.softmin_a}/Å  "
        f"d_wall={args.softmin_d_wall} Å  k={args.softmin_k} eV/Å²  "
        f"pairs={pairs[0]},{pairs[1]}",
        flush=True,
    )
    sigma = _parse_floats(args.metad_sigma)
    bias = WellTemperedMetadynamicsBias(
        cv_fn, n_cv=2, sigma=sigma, height=float(args.metad_height),
        bias_factor=float(args.metad_bias_factor), temperature=float(args.temperature),
        kB=float(args.kB), max_hills=int(args.metad_max_hills),
        device=device, dtype=dtype, n_atoms_per_system=num_atoms,
    )
    if args.resume_hills:
        n_loaded = bias.load_hills(args.resume_hills)
        print(
            f"[resume] loaded {n_loaded} hills from {args.resume_hills}  "
            f"max_hills={'unlimited' if int(args.metad_max_hills) <= 0 else args.metad_max_hills}",
            flush=True,
        )
    model = SumModel(mace_model, bias, wall, softmin_wall)
    driver = TorchSimWelltemperMetadynamic(
        model, bias, timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    print(
        f"[init] WT-MetaD σ={sigma} Å  w0={args.metad_height} γ={args.metad_bias_factor}  "
        f"n_hills0={bias.n_hills}",
        flush=True,
    )

    mass_t = torch.as_tensor(masses, dtype=dtype, device=device)
    z_t = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
    cell = torch.eye(3, device=device, dtype=dtype) * float(args.cell_length)
    state0 = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell,
        pbc=False, charge=float(args.charge), spin=float(args.spin),
    )

    out = Path(
        args.output
        or (EXPERIMENT_DIR / "results" / f"claisen_metad_{args.mace_model}_r{args.replicas}")
    )
    out.mkdir(parents=True, exist_ok=True)
    live = out / str(args.live_dir)
    live.mkdir(parents=True, exist_ok=True)
    traj_name = "trajectory_rerun.xyz" if args.resume_hills else "trajectory.xyz"
    traj = out / traj_name
    if traj.is_file():
        traj.unlink()

    cv0 = cv_fn(coordinate).detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    hist_cv = [cv0]
    hist_t = [np.zeros(cv0.shape[0])]

    write_xyz(traj, coordinate.detach().cpu().numpy(), symbols, comment_prefix="step=0")
    cv_traj_path = out / (
        "trajectory_cv_rerun.npy" if args.resume_hills else "trajectory_cv.npy"
    )
    np.save(cv_traj_path, cv0)

    g1 = np.linspace(args.fes_d1_min, args.fes_d1_max, 49)
    g2 = np.linspace(args.fes_d2_min, args.fes_d2_max, 49)

    def _all():
        return np.concatenate(hist_cv), np.concatenate(hist_t)

    def _live(done: int):
        cv_a, t_a = _all()
        plot_cv12(cv_a, live / "d1_d2.png", title=f"live d1–d2 step={done}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(t_a, cv_a[:, 0], s=4, alpha=0.35, c="C0", label="d1 O7–C8", linewidths=0)
            ax.scatter(t_a, cv_a[:, 1], s=4, alpha=0.35, c="C1", label="d2 C1–C10", linewidths=0)
            ax.set_xlabel("MD step")
            ax.set_ylabel("distance / Å")
            ax.legend(frameon=False, markerscale=3)
            ax.set_title(f"d1/d2 vs t step={done}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(live / "d12_vs_t.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] d-vs-t: {exc}", flush=True)
        F = bias.free_energy_grid(g1, g2)
        np.savez(live / "fes.npz", d1=g1, d2=g2, F=F)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.contourf(g1, g2, F - np.nanmin(F), levels=30, cmap="viridis")
            ax.set_xlabel(r"$d_1$ / Å")
            ax.set_ylabel(r"$d_2$ / Å")
            ax.set_title(f"FES step={done}  n_hills={bias.n_hills}")
            fig.colorbar(im, ax=ax, label="F / eV")
            fig.tight_layout()
            fig.savefig(live / "fes_d1_d2.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] FES: {exc}", flush=True)
        np.savez(
            live / "hills.npz",
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
            traj, coords.detach().cpu().numpy(), symbols,
            comment_prefix=f"step={done}", append=True, max_frames=args.max_xyz_frames,
        )
        cv_a, _ = _all()
        np.save(cv_traj_path, cv_a)
        print(
            f"[dump] step={done} n_hills={int(metrics.get('n_hills',0))}  "
            f"d1 med={np.median(cv[:,0]):.3f} d2 med={np.median(cv[:,1]):.3f}",
            flush=True,
        )
        _live(done)

    final = driver.run(
        state0, int(args.steps), verbose=args.verbose, verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress, on_dump=on_dump, dump_every=int(args.dump_every),
    )
    fc = sim_state_to_batched_coordinates(final)
    fcv = cv_fn(fc).detach().cpu().numpy()
    if fcv.ndim == 3:
        fcv = fcv.reshape(fcv.shape[0], -1)
    hist_cv.append(fcv)
    hist_t.append(np.full(fcv.shape[0], float(args.steps)))
    write_xyz(traj, fc.detach().cpu().numpy(), symbols, comment_prefix="final", append=True)
    final_hills_name = (
        "final_hills_rerun.npz" if args.resume_hills else "final_hills.npz"
    )
    np.savez(
        out / final_hills_name,
        centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
        heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
        n_hills=bias.n_hills,
    )
    _live(int(args.steps))
    print(f"done out={out}  live={live}", flush=True)


if __name__ == "__main__":
    main()
