#!/usr/bin/env python3
"""Dialanine (GAFF/SPONGE) + φ/ψ torsion CV + multi-walker WT-MetaD.

Same force field / backbone torsions as ``experiments/dialanine``. Metal units
(eV / Å / ps) via ``GaffTorchSimModel`` (kcal→eV). Shared bias; φ/ψ are
periodic with period 2π.
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
DIALANINE_DIR = EXPERIMENT_DIR.parent / "dialanine"
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from fp_transformer_torch import (  # noqa: E402
    GaffTorchSimModel,
    TorchSimWelltemperMetadynamic,
    Torsion,
    WellTemperedMetadynamicsBias,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    atomic_numbers_from_pdb_organic,
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.sponge_io import (  # noqa: E402
    load_sponge_coordinates,
)

DEFAULT_SPONGE = DIALANINE_DIR / "dialanine"
DEFAULT_CV_FILE = DIALANINE_DIR / "backbone_cv.txt"
DEFAULT_PDB = DIALANINE_DIR / "dialanine.pdb"
DEFAULT_MASS = DIALANINE_DIR / "dialanine_mass.txt"
DEFAULT_INIT_NPY = DIALANINE_DIR / "init_coords_2048.npy"

KB_EV_PER_K = 8.617330337217213e-5
TWO_PI = 2.0 * np.pi


def load_torsion_indices(path: Path) -> np.ndarray:
    """Load torsion quadruples; file is 1-based (same as dialanine FPT)."""
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


def load_masses(path: Path, num_atoms: int) -> np.ndarray:
    mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
        mass_vals = mass_vals[1:]
    if mass_vals.size < num_atoms:
        raise ValueError(f"mass file too short: {path}")
    return mass_vals[:num_atoms].astype(np.float64)


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def _symbols_from_z(z: np.ndarray) -> list[str]:
    table = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S"}
    return [table.get(int(x), f"Z{int(x)}") for x in z]


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "",
    append: bool = False,
) -> None:
    """Write one or many XYZ frames. ``coords`` is ``(N,3)`` or ``(B,N,3)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        frames = arr[None, ...]
    elif arr.ndim == 3:
        frames = arr
    else:
        raise ValueError(f"coords must be (N,3) or (B,N,3), got {arr.shape}")
    with path.open(mode) as f:
        for i, frame in enumerate(frames):
            n = frame.shape[0]
            f.write(f"{n}\n")
            f.write(f"{comment_prefix} frame={i}\n")
            for a in range(n):
                s = symbols[a] if a < len(symbols) else "X"
                x, y, z = frame[a]
                f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")


def plot_phi_psi(
    collective_variable: np.ndarray,
    save_path: Path,
    *,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return

    cv = np.asarray(collective_variable, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(-1, cv.shape[-1])
    if cv.ndim != 2 or cv.shape[1] < 2:
        print(f"[warn] need (N,≥2) for phi-psi, got {cv.shape}", flush=True)
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(cv[:, 0], cv[:, 1], c="C0", s=2.0, alpha=0.5, linewidths=0)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel(r"$\phi$ / rad")
    plt.ylabel(r"$\psi$ / rad")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=str, default=str(DEFAULT_SPONGE))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--pdb", type=str, default=str(DEFAULT_PDB))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001, help="ps")
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0, help="1/ps")
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=str(DEFAULT_INIT_NPY))
    p.add_argument("--cell-length", type=float, default=50.0)
    p.add_argument("--metad-sigma", type=str, default="0.35,0.35", help="rad")
    p.add_argument("--metad-height", type=float, default=0.02, help="eV")
    p.add_argument("--metad-bias-factor", type=float, default=15.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument(
        "--metad-max-hills",
        type=int,
        default=0,
        help="<=0 means unlimited hill buffer",
    )
    p.add_argument("--resume-hills", type=str, default=None)
    p.add_argument("--live-dir", type=str, default="live")
    p.add_argument("--fes-phi-min", type=float, default=-np.pi)
    p.add_argument("--fes-phi-max", type=float, default=np.pi)
    p.add_argument("--fes-psi-min", type=float, default=-np.pi)
    p.add_argument("--fes-psi-max", type=float, default=np.pi)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument(
        "--dtype", type=str, default="float64", choices=["float32", "float64"]
    )
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

    prefix = Path(args.sponge_prefix)
    gaff_model = GaffTorchSimModel(
        str(prefix),
        device=device,
        dtype=dtype,
    )
    num_atoms = int(gaff_model.gaff.num_atoms)

    torsion_indices = load_torsion_indices(Path(args.cv_file))
    torsion_indices = torsion_indices[torsion_indices.max(axis=1) < num_atoms]
    if torsion_indices.shape[0] < 2:
        raise ValueError("need ≥2 backbone torsions (φ, ψ)")
    torsion_indices = torsion_indices[:2]
    cv_fn = Torsion(torsion_indices)

    z_t = atomic_numbers_from_pdb_organic(args.pdb).to(device=device)
    if int(z_t.numel()) != num_atoms:
        raise ValueError(f"pdb Z={z_t.numel()} != GAFF N={num_atoms}")
    mass_np = load_masses(Path(args.mass_file), num_atoms)
    mass_t = torch.as_tensor(mass_np, dtype=dtype, device=device)
    symbols = _symbols_from_z(z_t.detach().cpu().numpy())

    print(
        f"Dialanine GAFF MetaD  device={device}  dtype={dtype}  "
        f"N={num_atoms}  R={args.replicas}  steps={args.steps}  T={args.temperature} K",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={torsion_indices.tolist()}", flush=True)

    init_path = Path(args.init_npy) if args.init_npy else None
    if init_path is not None and init_path.is_file():
        loaded = np.load(init_path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"bad init npy {init_path}: {loaded.shape}")
        if loaded.shape[1] != num_atoms:
            raise ValueError(
                f"init n_atoms={loaded.shape[1]} != GAFF n_atoms={num_atoms}"
            )
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(
            f"[init] subsampled {args.replicas}/{np.load(init_path).shape[0]} "
            f"from {init_path}",
            flush=True,
        )
    else:
        base = load_sponge_coordinates(prefix, num_atoms=num_atoms)
        loaded = np.repeat(base[None, ...], args.replicas, axis=0)
        print(f"[init] SPONGE coords × {args.replicas}", flush=True)

    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

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
        period=TWO_PI,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    if args.resume_hills:
        n_loaded = bias.load_hills(args.resume_hills)
        print(f"[resume] loaded {n_loaded} hills from {args.resume_hills}", flush=True)

    model = SumModel(gaff_model, bias)
    driver = TorchSimWelltemperMetadynamic(
        model,
        bias,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    print(
        f"[init] WT-MetaD σ={sigma} rad  w0={args.metad_height} eV  "
        f"γ={args.metad_bias_factor}  period=2π  n_hills0={bias.n_hills}",
        flush=True,
    )

    cell = torch.eye(3, device=device, dtype=dtype) * float(args.cell_length)
    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=False,
        charge=0.0,
        spin=1.0,
    )

    out = Path(
        args.output
        or (
            EXPERIMENT_DIR
            / "results"
            / f"dialanine_metad_r{args.replicas}_s{args.seed}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    live = out / str(args.live_dir)
    live.mkdir(parents=True, exist_ok=True)
    traj = out / "trajectory.xyz"
    if traj.is_file():
        traj.unlink()

    cv0 = cv_fn(coordinate).detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    hist_cv = [cv0]
    hist_t = [np.zeros(cv0.shape[0])]

    write_xyz(traj, coordinate.detach().cpu().numpy(), symbols, comment_prefix="step=0")
    cv_traj_path = out / "trajectory_cv.npy"
    np.save(cv_traj_path, cv0)
    np.save(out / "init_coords.npy", coordinate.detach().cpu().numpy().astype(np.float64))

    g_phi = np.linspace(args.fes_phi_min, args.fes_phi_max, 73)
    g_psi = np.linspace(args.fes_psi_min, args.fes_psi_max, 73)

    def _all():
        return np.concatenate(hist_cv), np.concatenate(hist_t)

    def _live(done: int):
        cv_a, _ = _all()
        plot_phi_psi(cv_a, live / "phi_psi.png", title=f"live φ–ψ step={done}")
        F = bias.free_energy_grid(g_phi, g_psi)
        np.savez(live / "fes.npz", phi=g_phi, psi=g_psi, F=F)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.contourf(
                g_phi, g_psi, F - np.nanmin(F), levels=30, cmap="viridis"
            )
            ax.set_xlabel(r"$\phi$ / rad")
            ax.set_ylabel(r"$\psi$ / rad")
            ax.set_title(f"FES step={done}  n_hills={bias.n_hills}")
            fig.colorbar(im, ax=ax, label="F / eV")
            fig.tight_layout()
            fig.savefig(live / "fes_phi_psi.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] FES plot: {exc}", flush=True)
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
        # keep trajectory size manageable: write replica 0 only each dump
        write_xyz(
            traj,
            coords[0].detach().cpu().numpy(),
            symbols,
            comment_prefix=f"step={done}",
            append=True,
        )
        cv_a, _ = _all()
        np.save(cv_traj_path, cv_a)
        print(
            f"[dump] step={done} n_hills={int(metrics.get('n_hills', 0))}  "
            f"φ med={np.median(cv[:, 0]):.3f} ψ med={np.median(cv[:, 1]):.3f}",
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
    write_xyz(traj, fc[0].detach().cpu().numpy(), symbols, comment_prefix="final", append=True)
    np.savez(
        out / "final_hills.npz",
        centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
        heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
        n_hills=bias.n_hills,
    )
    np.save(out / "final_coords.npy", fc.detach().cpu().numpy().astype(np.float64))
    _live(int(args.steps))
    print(f"done out={out}  live={live}", flush=True)


if __name__ == "__main__":
    main()
