#!/usr/bin/env python3
"""IMDA cis/trans + MACE + paper 2D CV + well-tempered MetaD (shared bias).

CVs: cv1 (HLDA), cv2 = c8*c17. Saves trajectory; live scatter + FES overwritten
in ``live/`` (no per-dump scatter PNGs).
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

from cis_trans_cv import DACisTransCV, WALL_PAIRS  # noqa: E402
from io_utils import (  # noqa: E402
    CV_LABELS,
    atomic_numbers_from_symbols,
    load_xyz,
    masses_from_symbols,
    plot_fes_projection,
    plot_live_scatters,
    write_xyz,
)
from fp_transformer_torch import (  # noqa: E402
    QuadraticDistanceWall,
    TorchSimWelltemperMetadynamic,
    WellTemperedMetadynamicsBias,
    load_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)

DEFAULT_XYZ = EXPERIMENT_DIR / "reactant.xyz"
DEFAULT_MASS = EXPERIMENT_DIR / "mass.txt"
KB_EV_PER_K = 8.617330337217213e-5


def load_masses(path: Path, num_atoms: int, symbols: list[str]) -> np.ndarray:
    if path.is_file():
        mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
            mass_vals = mass_vals[1:]
        if mass_vals.size < num_atoms:
            raise ValueError(f"mass file too short: {path}")
        return mass_vals[:num_atoms].astype(np.float64)
    return masses_from_symbols(symbols)


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def _cv_pack(cv_fn: DACisTransCV, coords: torch.Tensor):
    parts = cv_fn.components(coords)
    cv = torch.stack([parts["cv1"], parts["cv2"]], dim=-1)
    return (
        cv.detach().cpu().numpy(),
        parts["c8"].detach().cpu().numpy(),
        parts["c17"].detach().cpu().numpy(),
    )


def _grid_from_samples(vals: np.ndarray, pad: float, n: int = 61) -> np.ndarray:
    lo = float(np.min(vals)) - pad
    hi = float(np.max(vals)) + pad
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -1.0, 1.0
    return np.linspace(lo, hi, n)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--xyz", type=str, default=str(DEFAULT_XYZ))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=4)
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
    p.add_argument(
        "--metad-sigma",
        type=str,
        default="0.2,0.3",
        help="σ for (cv1, cv2)",
    )
    p.add_argument("--metad-height", type=float, default=0.02)
    p.add_argument("--metad-bias-factor", type=float, default=10.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument("--metad-max-hills", type=int, default=200000)
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
        f"method=metad2d  device={device}  mace={args.mace_model}  "
        f"replicas={args.replicas}  steps={args.steps}  T={args.temperature} K",
        flush=True,
    )

    symbols, xyz_coords = load_xyz(Path(args.xyz))
    atomic_numbers = atomic_numbers_from_symbols(symbols)
    num_atoms = int(atomic_numbers.shape[0])
    masses = load_masses(Path(args.mass_file), num_atoms, symbols)

    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        if loaded.ndim != 3 or loaded.shape[1] != num_atoms:
            raise ValueError(f"bad init npy: {loaded.shape}")
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(f"[init] subsampled init-npy → B={args.replicas}", flush=True)
    else:
        base = xyz_coords.astype(np.float64)
        loaded = np.repeat(base[None, ...], args.replicas, axis=0)
        print(f"[init] exact xyz × {args.replicas} (no noise)", flush=True)
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

    mace_model = load_mace_torch_sim_model(
        args.mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    print(f"[init] MACE loaded on {device}", flush=True)

    cv_fn = DACisTransCV()
    cv0, c8_0, c17_0 = _cv_pack(cv_fn, coordinate)
    print(
        f"[init] cv1 med={np.median(cv0[:,0]):.3f}  cv2 med={np.median(cv0[:,1]):.3f}  "
        f"(cv2<0 cis, >0 trans)",
        flush=True,
    )

    wall = QuadraticDistanceWall(
        list(WALL_PAIRS),
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
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
        f"[init] WT-MetaD 2D shared bias  σ={sigma}  w0={args.metad_height} eV  "
        f"γ={args.metad_bias_factor}  deposit_pace={args.metad_deposit_pace}",
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
        or (
            EXPERIMENT_DIR
            / "results"
            / f"da_cis_trans_metad_{args.mace_model}_r{args.replicas}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "live").mkdir(parents=True, exist_ok=True)
    traj_xyz = out / "trajectory.xyz"
    if traj_xyz.is_file():
        traj_xyz.unlink()

    hist_cv: list[np.ndarray] = [cv0]
    hist_c8: list[np.ndarray] = [np.asarray(c8_0).reshape(-1)]
    hist_c17: list[np.ndarray] = [np.asarray(c17_0).reshape(-1)]

    write_xyz(
        traj_xyz,
        coordinate.detach().cpu().numpy(),
        symbols,
        comment_prefix="IMDA-MetaD step=0",
        max_frames=args.max_xyz_frames,
        append=False,
    )
    np.save(out / "trajectory_cv.npy", cv0.astype(np.float64))

    def _all_cv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(hist_cv, axis=0),
            np.concatenate(hist_c8, axis=0),
            np.concatenate(hist_c17, axis=0),
        )

    def _update_live_plots(done: int) -> None:
        cv_all, c8_all, c17_all = _all_cv()
        title = f"IMDA-MetaD live (n={cv_all.shape[0]}, step={done})"
        plot_live_scatters(out, cv_all, c8_all, c17_all, title=title)

        g1 = _grid_from_samples(cv_all[:, 0], 0.2)
        g2 = _grid_from_samples(cv_all[:, 1], 0.5)
        F = bias.free_energy_grid(g1, g2)
        plot_fes_projection(
            g1,
            g2,
            F,
            out / "live" / "fes_cv1_cv2.png",
            title=f"WT-MetaD F(cv1,cv2)  step={done}",
            xlabel=CV_LABELS[0],
            ylabel=CV_LABELS[1],
        )
        np.savez(
            out / "live" / "hills.npz",
            centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
            heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
            sigma=bias.sigma.detach().cpu().numpy(),
            n_hills=bias.n_hills,
            bias_factor=bias.bias_factor,
        )
        np.savez(out / "live" / "free_energy_grid.npz", cv1=g1, cv2=g2, F=F)

    _update_live_plots(0)

    def on_dump(done: int, state, metrics: dict) -> None:
        coords = sim_state_to_batched_coordinates(state)
        cv, c8, c17 = _cv_pack(cv_fn, coords)
        hist_cv.append(cv)
        hist_c8.append(np.asarray(c8).reshape(-1))
        hist_c17.append(np.asarray(c17).reshape(-1))

        write_xyz(
            traj_xyz,
            coords.detach().cpu().numpy(),
            symbols,
            comment_prefix=f"IMDA-MetaD step={done}",
            max_frames=args.max_xyz_frames,
            append=True,
        )
        cv_all, _, _ = _all_cv()
        np.save(out / "trajectory_cv.npy", cv_all.astype(np.float64))
        np.save(out / f"step{done:06d}_coords.npy", coords.detach().cpu().numpy())

        print(
            f"[dump] step={done}  T_kin={metrics.get('kinetic_temperature_mean', float('nan')):.1f} K  "
            f"n_hills={int(metrics.get('n_hills', 0))}  "
            f"cv1 med={np.median(cv[:,0]):.3f}  cv2 med={np.median(cv[:,1]):.3f}",
            flush=True,
        )
        _update_live_plots(done)

    final_state = driver.run(
        state0,
        int(args.steps),
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_dump=on_dump,
        dump_every=int(args.dump_every),
    )
    final_coords = sim_state_to_batched_coordinates(final_state)
    final_cv, final_c8, final_c17 = _cv_pack(cv_fn, final_coords)
    hist_cv.append(final_cv)
    hist_c8.append(np.asarray(final_c8).reshape(-1))
    hist_c17.append(np.asarray(final_c17).reshape(-1))
    write_xyz(
        traj_xyz,
        final_coords.detach().cpu().numpy(),
        symbols,
        comment_prefix="IMDA-MetaD final",
        max_frames=args.max_xyz_frames,
        append=True,
    )
    cv_all, _, _ = _all_cv()
    np.save(out / "trajectory_cv.npy", cv_all.astype(np.float64))
    np.savez(
        out / "final_hills.npz",
        centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
        heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
        sigma=bias.sigma.detach().cpu().numpy(),
        n_hills=bias.n_hills,
        bias_factor=bias.bias_factor,
    )
    _update_live_plots(int(args.steps))
    print(f"done  out={out}  live plots in {out / 'live'}", flush=True)


if __name__ == "__main__":
    main()
