#!/usr/bin/env python3
"""DA + MACE-MH-0 + soft-distance 2D CV + well-tempered MetaD.

Same system / SoftDistanceS12 CV as ``experiments/DA_mace``, plus nonbond
lower walls (C6–C24, C3–C18, O8–C19, r_min=2.4 Å) to block wrong bonding.
Multi-walker MetaD shares one bias. Saves trajectory; live s1–s2 and d1/d2/d3–t.
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
    write_xyz,
)
from fp_transformer_torch import (  # noqa: E402
    QuadraticDistanceLowerWall,
    QuadraticDistanceWall,
    SoftDistanceS12,
    TorchSimWelltemperMetadynamic,
    WellTemperedMetadynamicsBias,
    load_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)

DEFAULT_XYZ = EXPERIMENT_DIR / "DA_tropone_dimethylfulvene.xyz"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
DEFAULT_NONBOND = EXPERIMENT_DIR / "nonbond_pairs.txt"
DEFAULT_MASS = EXPERIMENT_DIR / "da_mass.txt"
_DEFAULT_MD = EXPERIMENT_DIR / "init_coords_md_1024.npy"
_DEFAULT_GEOM = EXPERIMENT_DIR / "init_coords_1024.npy"
DEFAULT_INIT_NPY = _DEFAULT_MD if _DEFAULT_MD.is_file() else _DEFAULT_GEOM

KB_EV_PER_K = 8.617330337217213e-5
KCAL_PER_MOL_TO_EV = 1.0 / 23.06054783061903  # ≈ 0.043364 eV


def load_pair_indices(path: Path, *, expect: int | None = None) -> list[tuple[int, int]]:
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
    if expect is not None and len(rows) != expect:
        raise ValueError(f"need exactly {expect} pairs in {path}, got {len(rows)}")
    if not rows:
        raise ValueError(f"no pairs in {path}")
    return rows


def load_masses(path: Path, num_atoms: int) -> np.ndarray:
    mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
        mass_vals = mass_vals[1:]
    if mass_vals.size < num_atoms:
        raise ValueError(f"mass file too short: {path}")
    return mass_vals[:num_atoms].astype(np.float64)


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def _raw_distances(cv_fn: SoftDistanceS12, coords: torch.Tensor) -> np.ndarray:
    """Return raw Å distances (B, 3) for CV pairs d1,d2,d3."""
    return cv_fn.soft.distances(coords).detach().cpu().numpy()


def _plot_live_s1_s2(path: Path, cv: np.ndarray, *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skip s1-s2 plot: {exc}", flush=True)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cv, dtype=np.float64).reshape(-1, 2)
    plt.figure(figsize=(5, 5))
    plt.scatter(arr[:, 0], arr[:, 1], c="C0", s=3.0, alpha=0.45, linewidths=0)
    plt.xlabel(r"$s_1=c_1+c_2+c_3$")
    plt.ylabel(r"$s_2=c_2-c_3$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[plot] wrote {path}", flush=True)


def _plot_live_d_vs_t(
    path: Path,
    times: np.ndarray,
    d_hist: np.ndarray,
    *,
    title: str,
) -> None:
    """One figure: d1, d2, d3 vs time (all walkers as scatter)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skip d-vs-t plot: {exc}", flush=True)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    d = np.asarray(d_hist, dtype=np.float64)
    # d: (n_samples, 3) aligned with times
    if d.ndim != 2 or d.shape[0] != t.shape[0] or d.shape[1] < 3:
        print(f"[warn] bad d_hist shape {d.shape} vs times {t.shape}", flush=True)
        return
    plt.figure(figsize=(7, 4.5))
    for i, (lab, col) in enumerate(
        (("d1 (2–15)", "C0"), ("d2 (3–13)", "C1"), ("d3 (7–24)", "C2"))
    ):
        plt.scatter(t, d[:, i], c=col, s=4.0, alpha=0.35, linewidths=0, label=lab)
    plt.xlabel("MD step")
    plt.ylabel(r"distance / Å")
    plt.title(title)
    plt.legend(markerscale=3, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[plot] wrote {path}", flush=True)


def _plot_fes(path: Path, s1g, s2g, F, *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] skip FES plot: {exc}", flush=True)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.contourf(s1g, s2g, F - np.nanmin(F), levels=30, cmap="viridis")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="F / eV")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--xyz", type=str, default=str(DEFAULT_XYZ))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--nonbond-file", type=str, default=str(DEFAULT_NONBOND))
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
    p.add_argument(
        "--init-npy",
        type=str,
        default=str(DEFAULT_INIT_NPY) if Path(str(DEFAULT_INIT_NPY)).is_file() else None,
    )
    p.add_argument("--max-xyz-frames", type=int, default=None)
    p.add_argument("--cell-length", type=float, default=40.0)
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--d-ref", type=float, default=1.7)
    p.add_argument("--soft-n", type=int, default=6)
    p.add_argument("--soft-m", type=int, default=8)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--nonbond-r-min", type=float, default=2.4, help="Å")
    p.add_argument(
        "--nonbond-k-kcal",
        type=float,
        default=100.0,
        help="κ in kcal/mol/Å² for lower wall",
    )
    p.add_argument("--metad-sigma", type=str, default="0.05,0.05")
    p.add_argument("--metad-height", type=float, default=0.02, help="w0 in eV")
    p.add_argument("--metad-bias-factor", type=float, default=30.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument("--metad-max-hills", type=int, default=200000)
    p.add_argument("--fes-s1-min", type=float, default=0.0)
    p.add_argument("--fes-s1-max", type=float, default=3.0)
    p.add_argument("--fes-s2-min", type=float, default=-1.0)
    p.add_argument("--fes-s2-max", type=float, default=1.0)
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
        f"method=metad  device={device}  mace={args.mace_model}  "
        f"replicas={args.replicas}  steps={args.steps}  T={args.temperature} K  "
        f"d_ref={args.d_ref}",
        flush=True,
    )

    symbols, _ = load_xyz(Path(args.xyz))
    atomic_numbers = atomic_numbers_from_symbols(symbols)
    num_atoms = int(atomic_numbers.shape[0])
    masses = load_masses(Path(args.mass_file), num_atoms)
    pairs = load_pair_indices(Path(args.cv_file), expect=3)
    nonbond = load_pair_indices(Path(args.nonbond_file))
    print(
        f"[CV] SoftDistanceS12 pairs(0-based)={pairs}  "
        f"c=(1-(d/d_ref)^n)/(1-(d/d_ref)^m)  d_ref={args.d_ref} n={args.soft_n} m={args.soft_m}",
        flush=True,
    )
    print(
        f"[CV] s1=c1+c2+c3  s2=c2-c3   "
        f"(1-based pairs 2–15, 3–13, 7–24)",
        flush=True,
    )
    print(
        f"[nonbond] pairs(0-based)={nonbond}  "
        f"(1-based C6–C24, C3–C18, O8–C19)  r_min={args.nonbond_r_min} Å  "
        f"k={args.nonbond_k_kcal} kcal/mol/Å²",
        flush=True,
    )

    if not args.init_npy:
        raise ValueError("--init-npy is required")
    loaded = np.load(Path(args.init_npy))
    if loaded.ndim != 3 or loaded.shape[1] != num_atoms:
        raise ValueError(f"bad init npy: {loaded.shape}")
    if args.replicas > loaded.shape[0]:
        raise ValueError(f"replicas={args.replicas} > B={loaded.shape[0]}")
    idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
    loaded = loaded[idx]
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    print(f"[init] subsampled init-npy → B={args.replicas}", flush=True)

    mace_model = load_mace_torch_sim_model(
        args.mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    print(f"[init] MACE loaded on {device}", flush=True)

    cv_fn = SoftDistanceS12(
        pairs, d_ref=float(args.d_ref), n=int(args.soft_n), m=int(args.soft_m)
    )
    cv0 = cv_fn(coordinate).detach().cpu().numpy().reshape(-1, 2)
    d0 = _raw_distances(cv_fn, coordinate)
    print(
        f"[init] s1[min/med/max]={cv0[:,0].min():.3f}/{np.median(cv0[:,0]):.3f}/{cv0[:,0].max():.3f}  "
        f"s2[min/med/max]={cv0[:,1].min():.3f}/{np.median(cv0[:,1]):.3f}/{cv0[:,1].max():.3f}",
        flush=True,
    )

    upper_wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    k_ev = float(args.nonbond_k_kcal) * KCAL_PER_MOL_TO_EV
    lower_wall = QuadraticDistanceLowerWall(
        nonbond,
        r_min=float(args.nonbond_r_min),
        k=k_ev,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    print(
        f"[wall] upper k={args.wall_k} eV/Å² d_max={args.wall_d_max} Å; "
        f"lower k={k_ev:.4f} eV/Å² ({args.nonbond_k_kcal} kcal/mol/Å²) r_min={args.nonbond_r_min} Å",
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
    model = SumModel(mace_model, bias, upper_wall, lower_wall)
    driver = TorchSimWelltemperMetadynamic(
        model,
        bias,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    print(
        f"[init] WT-MetaD shared bias  σ={sigma}  w0={args.metad_height} eV  "
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
            / f"da_metad_{args.mace_model}_r{args.replicas}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "live").mkdir(parents=True, exist_ok=True)
    traj_xyz = out / "trajectory.xyz"
    if traj_xyz.is_file():
        traj_xyz.unlink()

    # history: each dump appends all walkers
    hist_cv: list[np.ndarray] = [cv0]
    hist_d: list[np.ndarray] = [d0]
    hist_t: list[np.ndarray] = [np.zeros(cv0.shape[0], dtype=np.float64)]

    write_xyz(
        traj_xyz,
        coordinate.detach().cpu().numpy(),
        symbols,
        comment_prefix="DA-MetaD step=0",
        max_frames=args.max_xyz_frames,
    )
    np.save(out / "trajectory_cv.npy", cv0.astype(np.float64))
    np.save(out / "trajectory_d.npy", d0.astype(np.float64))

    s1g = np.linspace(float(args.fes_s1_min), float(args.fes_s1_max), 61)
    s2g = np.linspace(float(args.fes_s2_min), float(args.fes_s2_max), 41)

    def _all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(hist_cv, axis=0),
            np.concatenate(hist_d, axis=0),
            np.concatenate(hist_t, axis=0),
        )

    def _update_live(done: int) -> None:
        cv_all, d_all, t_all = _all()
        _plot_live_s1_s2(
            out / "live" / "s1_s2.png",
            cv_all,
            title=f"DA-MetaD live s1–s2 (n={cv_all.shape[0]}, step={done})",
        )
        _plot_live_d_vs_t(
            out / "live" / "d123_vs_t.png",
            t_all,
            d_all,
            title=f"d1/d2/d3 vs time (step={done})",
        )
        F = bias.free_energy_grid(s1g, s2g)
        np.savez(out / "live" / "free_energy_grid.npz", s1=s1g, s2=s2g, F=F)
        _plot_fes(
            out / "live" / "fes_s1_s2.png",
            s1g,
            s2g,
            F,
            title=f"WT-MetaD F(s)  γ={args.metad_bias_factor}  step={done}",
        )
        np.savez(
            out / "live" / "hills.npz",
            centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
            heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
            sigma=bias.sigma.detach().cpu().numpy(),
            n_hills=bias.n_hills,
            bias_factor=bias.bias_factor,
        )

    _update_live(0)

    def on_dump(done: int, state, metrics: dict) -> None:
        coords = sim_state_to_batched_coordinates(state)
        cv = cv_fn(coords).detach().cpu().numpy().reshape(-1, 2)
        d = _raw_distances(cv_fn, coords)
        hist_cv.append(cv)
        hist_d.append(d)
        hist_t.append(np.full(cv.shape[0], float(done), dtype=np.float64))

        write_xyz(
            traj_xyz,
            coords.detach().cpu().numpy(),
            symbols,
            comment_prefix=f"DA-MetaD step={done}",
            max_frames=args.max_xyz_frames,
            append=True,
        )
        cv_all, d_all, _ = _all()
        np.save(out / "trajectory_cv.npy", cv_all.astype(np.float64))
        np.save(out / "trajectory_d.npy", d_all.astype(np.float64))
        np.save(out / f"step{done:06d}_coords.npy", coords.detach().cpu().numpy())

        print(
            f"[dump] step={done}  T_kin={metrics.get('kinetic_temperature_mean', float('nan')):.1f} K  "
            f"n_hills={int(metrics.get('n_hills', 0))}  "
            f"s1 med={np.median(cv[:,0]):.3f}  s2 med={np.median(cv[:,1]):.3f}",
            flush=True,
        )
        _update_live(done)

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
    final_cv = cv_fn(final_coords).detach().cpu().numpy().reshape(-1, 2)
    final_d = _raw_distances(cv_fn, final_coords)
    hist_cv.append(final_cv)
    hist_d.append(final_d)
    hist_t.append(np.full(final_cv.shape[0], float(args.steps), dtype=np.float64))
    write_xyz(
        traj_xyz,
        final_coords.detach().cpu().numpy(),
        symbols,
        comment_prefix="DA-MetaD final",
        max_frames=args.max_xyz_frames,
        append=True,
    )
    cv_all, d_all, _ = _all()
    np.save(out / "trajectory_cv.npy", cv_all.astype(np.float64))
    np.save(out / "trajectory_d.npy", d_all.astype(np.float64))
    np.savez(
        out / "final_hills.npz",
        centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
        heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
        sigma=bias.sigma.detach().cpu().numpy(),
        n_hills=bias.n_hills,
        bias_factor=bias.bias_factor,
    )
    _update_live(int(args.steps))
    print(f"done  out={out}  live={out / 'live'}", flush=True)


if __name__ == "__main__":
    main()
