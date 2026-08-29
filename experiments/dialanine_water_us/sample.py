#!/usr/bin/env python3
"""Solvated dialanine (GAFF+TIP3P) 2D umbrella sampling on φ/ψ.

Batched multi-window US: ``replicas = n_phi × n_ψ`` (default 16×16 = 256).
Harmonic κ default 100 kcal mol⁻¹ rad⁻² with period 2π (torus minimum image).

Init: prefer ``--init-npy`` from ``prepare_init.py`` (MetaD solute frames +
TIP3P re-solvation). Short clash relax under umbrella, then production.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WATER_DIR = Path(__file__).resolve().parents[1] / "dialanine_water"
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WATER_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import load_xyz, plot_phi_psi, write_xyz  # noqa: E402
from fp_transformer_torch import (  # noqa: E402
    TorchSimUmbrellaSampling,
    Torsion,
    UmbrellaBias,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)

DEFAULT_BOX = WATER_DIR / "mm_box"
DEFAULT_CV = DEFAULT_BOX / "backbone_cv.txt"
DEFAULT_INIT = EXPERIMENT_DIR / "init_us2d_r256.npy"
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
KCAL_TO_EV = 1.0 / EV_TO_KCAL
N_SOLUTE = 22
TWO_PI = 2.0 * math.pi


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


def make_phi_psi_centers(n_phi: int, n_psi: int) -> np.ndarray:
    edges_phi = np.linspace(-np.pi, np.pi, int(n_phi) + 1, dtype=np.float64)
    edges_psi = np.linspace(-np.pi, np.pi, int(n_psi) + 1, dtype=np.float64)
    phi = 0.5 * (edges_phi[:-1] + edges_phi[1:])
    psi = 0.5 * (edges_psi[:-1] + edges_psi[1:])
    gg = np.stack(np.meshgrid(phi, psi, indexing="xy"), axis=-1)
    return gg.reshape(-1, 2)


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


def _cv_np(cv_fn, coordinate: torch.Tensor) -> np.ndarray:
    cv = cv_fn(coordinate).detach().cpu().numpy()
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    return np.asarray(cv, dtype=np.float64)


def _torus_delta(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = s - c
    return d - TWO_PI * np.round(d / TWO_PI)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV))
    p.add_argument("--n-phi", type=int, default=16)
    p.add_argument("--n-psi", type=int, default=16)
    p.add_argument(
        "--replicas",
        type=int,
        default=None,
        help="override n_phi*n_psi (must equal product if both set)",
    )
    p.add_argument("--steps", type=int, default=100000, help="production steps (100 ps @ 1 fs)")
    p.add_argument("--relax-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=str(DEFAULT_INIT))
    p.add_argument("--centers-npy", type=str, default=None)
    p.add_argument(
        "--kappa-kcal",
        type=float,
        default=100.0,
        help="κ in kcal/mol/rad² (applied to both φ and ψ)",
    )
    p.add_argument("--kappa", type=float, default=None, help="override κ in eV/rad²")
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument("--electrostatics", type=str, default="pme", choices=["pme", "dsf"])
    p.add_argument("--dsf-alpha", type=float, default=0.2)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument("--max-xyz-frames", type=int, default=16)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    return p


def _run_nvt(state, model, *, steps, temperature, timestep, gamma, update_pace, label):
    import torch_sim as ts

    steps = int(steps)
    if steps <= 0:
        return state
    remaining = steps
    done = 0
    pace = max(1, int(update_pace))
    while remaining > 0:
        chunk = min(pace, remaining)
        state = ts.integrate(
            system=state,
            model=model,
            integrator=ts.Integrator.nvt_langevin,
            n_steps=chunk,
            temperature=float(temperature),
            timestep=float(timestep),
            gamma=float(gamma),
            pbar=False,
        )
        remaining -= chunk
        done += chunk
        if done == steps or done % max(pace * 5, 100) == 0:
            print(f"[{label}] step={done}/{steps}", flush=True)
    return state


def main() -> None:
    args = build_parser().parse_args()
    if args.traj_full:
        args.traj_solute_only = False

    from torch_sim.models.interface import SumModel

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    n_phi = int(args.n_phi)
    n_psi = int(args.n_psi)
    n_win = n_phi * n_psi
    if args.replicas is not None:
        if int(args.replicas) != n_win:
            raise ValueError(
                f"--replicas={args.replicas} != n_phi*n_psi={n_win}; "
                "set --n-phi/--n-psi instead"
            )

    box = Path(args.box_dir)
    topo_path = box / "topology.npz"
    if not topo_path.is_file():
        raise FileNotFoundError(f"missing {topo_path}")

    symbols, xyz_coords = load_xyz(box / "box.xyz")
    topo = MLMMTopology.load_npz(topo_path).to(device, dtype)
    cell = torch.as_tensor(
        np.loadtxt(box / "cell.txt").reshape(3, 3), dtype=dtype, device=device
    )
    num_atoms = topo.n_atoms
    solute_symbols = symbols[:N_SOLUTE]

    if args.centers_npy and Path(args.centers_npy).is_file():
        centers = np.load(Path(args.centers_npy)).astype(np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"centers must be (B,2), got {centers.shape}")
        n_win = int(centers.shape[0])
    else:
        centers = make_phi_psi_centers(n_phi, n_psi)

    kappa = (
        float(args.kappa)
        if args.kappa is not None
        else float(args.kappa_kcal) * KCAL_TO_EV
    )
    kappa_kcal = kappa * EV_TO_KCAL

    tors = load_torsion_indices(Path(args.cv_file))[:2]
    cv_fn = Torsion(tors)

    out = Path(
        args.output
        or (
            EXPERIMENT_DIR
            / "results"
            / f"dialanine_water_us2d_r{n_win}_k{kappa_kcal:.0f}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "live").mkdir(parents=True, exist_ok=True)

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
    print(
        f"Dialanine-water US-2D  device={device}  N={num_atoms}  "
        f"windows={n_win} ({n_phi}×{n_psi})  steps={args.steps}  "
        f"κ={kappa_kcal:.4g} kcal/mol/rad²  elec={args.electrostatics}",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={tors.tolist()}  period=2π", flush=True)

    init_path = Path(args.init_npy) if args.init_npy else None
    if init_path is not None and init_path.is_file():
        loaded = np.load(init_path)
        if loaded.shape[0] < n_win:
            raise ValueError(f"init-npy has {loaded.shape[0]} < {n_win} windows")
        if loaded.shape[0] != n_win:
            print(
                f"[warn] init has {loaded.shape[0]} frames; using first {n_win}",
                flush=True,
            )
            loaded = loaded[:n_win]
        if loaded.shape[1] != num_atoms:
            raise ValueError(
                f"init atoms {loaded.shape[1]} != topology N={num_atoms}"
            )
        init_coords = loaded.astype(np.float32)
        print(f"[init] loaded {init_path} → {init_coords.shape}", flush=True)
    else:
        raise FileNotFoundError(
            f"need --init-npy (run prepare_init.py); missing {init_path}"
        )

    np.save(out / "init_coords.npy", init_coords)
    np.save(out / "umbrella_centers.npy", centers.astype(np.float64))

    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    coordinate = torch.as_tensor(init_coords, dtype=dtype, device=device)

    umbrella = UmbrellaBias(
        cv_fn,
        centers,
        kappa=kappa,
        period=TWO_PI,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mm, umbrella)

    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=True,
    )
    state0 = _attach_charges(state0, topo.charges.to(device=device, dtype=dtype), n_win)

    cv0 = _cv_np(cv_fn, coordinate)
    d0 = _torus_delta(cv0, centers)
    print(
        f"[init-cv] |Δ|_torus mean/max="
        f"{np.sqrt((d0 * d0).sum(-1)).mean():.4f}/"
        f"{np.sqrt((d0 * d0).sum(-1)).max():.4f} rad",
        flush=True,
    )
    plot_phi_psi(cv0, out / "live" / "init_phi_psi.png", title="US init φ–ψ")

    if int(args.relax_steps) > 0:
        print(f"[relax] {args.relax_steps} steps under umbrella", flush=True)
        state0 = _run_nvt(
            state0,
            model,
            steps=int(args.relax_steps),
            temperature=float(args.temperature),
            timestep=float(args.timestep),
            gamma=float(args.langevin_friction),
            update_pace=int(args.update_pace),
            label="relax",
        )
        coordinate = sim_state_to_batched_coordinates(state0)
        np.save(
            out / "relax_coords.npy",
            coordinate.detach().cpu().numpy().astype(np.float32),
        )
        cv_r = _cv_np(cv_fn, coordinate)
        np.save(out / "relax_cv.npy", cv_r.astype(np.float32))
        plot_phi_psi(cv_r, out / "live" / "relax_phi_psi.png", title="US relax φ–ψ")

    driver = TorchSimUmbrellaSampling(
        model,
        umbrella,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        update_pace=int(args.update_pace),
    )

    traj = out / "trajectory.xyz"
    if traj.is_file():
        traj.unlink()

    kap_arr = umbrella.kappa.detach().cpu().numpy()
    np.save(out / "umbrella_kappa.npy", kap_arr)
    with open(out / "run_info.txt", "w", encoding="utf-8") as fh:
        fh.write("cv=phi,psi\n")
        fh.write(f"replicas={n_win}\n")
        fh.write(f"n_phi={n_phi}\n")
        fh.write(f"n_psi={n_psi}\n")
        fh.write(f"steps={args.steps}\n")
        fh.write(f"timestep_ps={args.timestep}\n")
        fh.write(f"total_time_ps={args.steps * args.timestep}\n")
        fh.write(f"relax_steps={args.relax_steps}\n")
        fh.write(f"temperature_K={args.temperature}\n")
        fh.write(f"kappa_eV_per_rad2={kappa}\n")
        fh.write(f"kappa_kcal_per_mol_rad2={kappa_kcal}\n")
        fh.write("period=2*pi\n")
        fh.write(f"torsions={tors.tolist()}\n")
        fh.write(f"init_npy={init_path}\n")
        fh.write(f"electrostatics={args.electrostatics}\n")

    def _traj_coords(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if args.traj_solute_only:
            return coords[..., :N_SOLUTE, :], solute_symbols
        return coords, symbols

    cv0 = _cv_np(cv_fn, coordinate)
    driver.cv_trajectory.append(cv0.astype(np.float64))
    driver.cv_times_ps.append(0.0)
    c0, s0 = _traj_coords(coordinate.detach().cpu().numpy())
    write_xyz(traj, c0, s0, comment_prefix="step=0")
    np.save(out / "trajectory_cv.npy", cv0[None, ...].astype(np.float32))
    np.save(out / "trajectory_cv_times_ps.npy", np.array([0.0]))

    def _live(done: int, cv_frame: np.ndarray | None = None):
        times, cv_traj = driver.stacked_cv_trajectory()
        np.save(out / "trajectory_cv.npy", cv_traj.astype(np.float32))
        np.save(out / "trajectory_cv_times_ps.npy", times.astype(np.float64))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            last = cv_frame if cv_frame is not None else cv_traj[-1]
            plot_phi_psi(
                last,
                out / "live" / "phi_psi.png",
                title=f"US-2D step={done}",
            )
            # overlay centers
            fig, ax = plt.subplots(figsize=(5.2, 5))
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="k",
                s=10,
                marker="x",
                label="centers",
            )
            ax.scatter(
                last[:, 0],
                last[:, 1],
                c="C3",
                s=8,
                alpha=0.7,
                linewidths=0,
                label="replicas",
            )
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_xlabel(r"$\phi$ / rad")
            ax.set_ylabel(r"$\psi$ / rad")
            ax.set_title(f"US-2D step={done}")
            ax.legend(frameon=False, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "live" / "windows_phi_psi.png", dpi=140)
            plt.close(fig)

            if cv_traj.shape[0] >= 2:
                delta = _torus_delta(
                    cv_traj.reshape(cv_traj.shape[0], n_win, 2),
                    centers[None, :, :],
                )
                rms = np.sqrt(np.mean(delta * delta, axis=-1))
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(times, rms.mean(axis=1), c="C0", label="mean |Δ|_torus")
                ax.fill_between(
                    times,
                    rms.min(axis=1),
                    rms.max(axis=1),
                    color="C0",
                    alpha=0.2,
                    label="min–max",
                )
                ax.set_xlabel("t / ps")
                ax.set_ylabel(r"$|\mathbf{s}-\mathbf{s}_0|_{\mathrm{torus}}$ / rad")
                ax.set_title(f"umbrella deviation step={done}")
                ax.grid(True, alpha=0.3)
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(out / "live" / "rms_dev_vs_t.png", dpi=140)
                plt.close(fig)
        except Exception as exc:
            print(f"[warn] live plot: {exc}", flush=True)

    _live(0, cv0)
    print(
        f"[prod] start  R={n_win}  steps={args.steps}  "
        f"({args.steps * args.timestep:g} ps)",
        flush=True,
    )

    def on_dump(done, state, metrics):
        coords = sim_state_to_batched_coordinates(state)
        cv = _cv_np(cv_fn, coords)
        tc, ts_ = _traj_coords(coords.detach().cpu().numpy())
        write_xyz(
            traj,
            tc,
            ts_,
            comment_prefix=f"step={done}",
            append=True,
            max_frames=args.max_xyz_frames,
        )
        print(
            f"[dump] step={done}  umb_E={metrics.get('umbrella_energy_mean', float('nan')):.4f}  "
            f"rms_dev={metrics.get('umbrella_rms_dev_mean', float('nan')):.3f}  "
            f"φ med={np.median(cv[:, 0]):.3f} ψ med={np.median(cv[:, 1]):.3f}",
            flush=True,
        )
        _live(done, cv)

    final = driver.run(
        state0,
        int(args.steps),
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_dump=on_dump,
        dump_every=int(args.dump_every),
        record_cv=True,
    )
    fc = sim_state_to_batched_coordinates(final)
    fcv = _cv_np(cv_fn, fc)
    tc, ts_ = _traj_coords(fc.detach().cpu().numpy())
    write_xyz(traj, tc, ts_, comment_prefix="final", append=True)
    np.save(out / "final_coords.npy", fc.detach().cpu().numpy().astype(np.float32))
    np.save(out / "final_cv.npy", fcv.astype(np.float32))
    times, cv_traj = driver.stacked_cv_trajectory()
    np.save(out / "trajectory_cv.npy", cv_traj.astype(np.float32))
    np.save(out / "trajectory_cv_times_ps.npy", times.astype(np.float64))
    np.savez(
        out / "window_stats.npz",
        centers=centers,
        kappa=kap_arr,
        mean_cv=cv_traj.mean(axis=0),
        std_cv=cv_traj.std(axis=0),
        n_frames=cv_traj.shape[0],
        times_ps=times,
    )
    _live(int(args.steps), fcv)
    print(f"done out={out}", flush=True)


if __name__ == "__main__":
    main()
