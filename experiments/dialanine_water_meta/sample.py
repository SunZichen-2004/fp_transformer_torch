#!/usr/bin/env python3
"""Solvated dialanine (GAFF + TIP3P) + φ/ψ WT-MetaD (multi-walker, shared bias).

Full classical MM via ``MechanicalEmbeddingMLMM(enable_mace=False)``.
Box: ``../dialanine_water/mm_box`` (run ``../dialanine_water/prepare_box.py``).
φ/ψ are periodic with period 2π. Metal units (eV / Å / ps).
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
WATER_DIR = EXPERIMENT_DIR.parent / "dialanine_water"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WATER_DIR))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import load_xyz, plot_phi_psi, write_xyz  # noqa: E402
from fp_transformer_torch import (  # noqa: E402
    TorchSimWelltemperMetadynamic,
    Torsion,
    WellTemperedMetadynamicsBias,
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
KB_EV_PER_K = 8.617330337217213e-5
TWO_PI = 2.0 * np.pi
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


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


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


def _load_last_xyz_frames(path: Path, n_frames: int) -> np.ndarray:
    text = Path(path).read_text().splitlines()
    frames: list[np.ndarray] = []
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
        i += 2
        xyz = np.zeros((nat, 3), dtype=np.float64)
        for a in range(nat):
            parts = text[i + a].split()
            xyz[a] = [float(parts[1]), float(parts[2]), float(parts[3])]
        frames.append(xyz)
        i += nat
    if not frames:
        raise ValueError(f"no frames in {path}")
    take = frames[-int(n_frames) :]
    while len(take) < n_frames:
        take.append(take[-1].copy())
    return np.stack(take[-n_frames:], axis=0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument(
        "--pre-md-steps",
        type=int,
        default=0,
        help="unbiased NVT Langevin steps (no MetaD hills) before WT-MetaD",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true")
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument("--electrostatics", type=str, default="pme", choices=["pme", "dsf"])
    p.add_argument("--dsf-alpha", type=float, default=0.2)
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
    p.add_argument("--max-xyz-frames", type=int, default=None)
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
    if args.traj_full:
        args.traj_solute_only = False

    from torch_sim.models.interface import SumModel

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    box = Path(args.box_dir)
    xyz_path = box / "box.xyz"
    topo_path = box / "topology.npz"
    cell_path = box / "cell.txt"
    if not topo_path.is_file():
        raise FileNotFoundError(f"missing {topo_path}; run dialanine_water/prepare_box.py")

    symbols, xyz_coords = load_xyz(xyz_path)
    topo = MLMMTopology.load_npz(topo_path).to(device, dtype)
    cell = torch.as_tensor(np.loadtxt(cell_path).reshape(3, 3), dtype=dtype, device=device)
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(f"xyz N={xyz_coords.shape[0]} != topology N={num_atoms}")

    tors = load_torsion_indices(Path(args.cv_file))[:2]
    cv_fn = Torsion(tors)
    solute_symbols = symbols[:N_SOLUTE]

    print(
        f"Dialanine-water MetaD  device={device}  N={num_atoms}  "
        f"R={args.replicas}  steps={args.steps}  elec={args.electrostatics}  "
        f"cutoff={args.pme_cutoff}  box={float(cell[0, 0]):.1f} Å",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={tors.tolist()}", flush=True)

    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(f"[init] from npy {args.init_npy}", flush=True)
    elif args.resume_hills:
        hills_p = Path(args.resume_hills)
        cand_traj = hills_p.parent.parent / "trajectory.xyz"
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        if cand_traj.is_file():
            frames = _load_last_xyz_frames(cand_traj, int(args.replicas))
            if frames.shape[-2] == num_atoms:
                loaded = frames
                print(f"[init] resume: full-box frames from {cand_traj}", flush=True)
            else:
                print(
                    f"[init] resume: traj solute-only (N={frames.shape[-2]}); "
                    f"using box xyz × {args.replicas}",
                    flush=True,
                )
        else:
            print(f"[init] resume hills; box xyz × {args.replicas}", flush=True)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        print(f"[init] box xyz × {args.replicas}", flush=True)
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

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

    model = SumModel(mm, bias)
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

    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    state0 = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
    )
    state0 = _attach_charges(
        state0, topo.charges.to(device=device, dtype=dtype), int(args.replicas)
    )

    out = Path(
        args.output
        or (
            EXPERIMENT_DIR
            / "results"
            / f"dialanine_water_metad_r{args.replicas}_s{args.seed}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    live = out / str(args.live_dir)
    live.mkdir(parents=True, exist_ok=True)
    traj_name = "trajectory_rerun.xyz" if args.resume_hills else "trajectory.xyz"
    traj = out / traj_name
    if traj.is_file():
        traj.unlink()

    def _traj_coords(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if args.traj_solute_only:
            return coords[..., :N_SOLUTE, :], solute_symbols
        return coords, symbols

    cv0 = cv_fn(coordinate).detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    hist_cv = [cv0]
    hist_t = [np.zeros(cv0.shape[0])]

    c0, s0 = _traj_coords(coordinate.detach().cpu().numpy())
    write_xyz(traj, c0, s0, comment_prefix="step=0")
    cv_traj_path = out / (
        "trajectory_cv_rerun.npy" if args.resume_hills else "trajectory_cv.npy"
    )
    np.save(cv_traj_path, cv0)
    np.save(
        out / ("init_coords_rerun.npy" if args.resume_hills else "init_coords.npy"),
        coordinate.detach().cpu().numpy().astype(np.float32),
    )

    if int(args.pre_md_steps) > 0:
        import time as _time

        import torch_sim as ts

        n_pre = int(args.pre_md_steps)
        print(
            f"[pre-md] unbiased NVT Langevin  steps={n_pre}  dt={args.timestep}  "
            f"γ={args.langevin_friction}  (no MetaD hills)",
            flush=True,
        )
        remaining = n_pre
        done_pre = 0
        t_pre = _time.perf_counter()
        chunk = 50
        while remaining > 0:
            c = min(chunk, remaining)
            state0 = ts.integrate(
                system=state0,
                model=mm,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=c,
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                pbar=False,
            )
            remaining -= c
            done_pre += c
            if done_pre % 500 == 0 or remaining == 0:
                elapsed = max(_time.perf_counter() - t_pre, 1e-12)
                t_per = None
                if hasattr(state0, "calc_temperature"):
                    t_per = state0.calc_temperature()
                tmean = (
                    float(t_per.detach().mean().item()) if t_per is not None else float("nan")
                )
                print(
                    f"[pre-md] step={done_pre}/{n_pre}  T_kin mean={tmean:.2f} K  "
                    f"{done_pre / elapsed:.2f} it/s",
                    flush=True,
                )
        coordinate = sim_state_to_batched_coordinates(state0)
        np.save(
            out / "init_coords_from_pre_md.npy",
            coordinate.detach().cpu().numpy().astype(np.float32),
        )
        cv_pre = cv_fn(coordinate).detach().cpu().numpy()
        if cv_pre.ndim == 3:
            cv_pre = cv_pre.reshape(cv_pre.shape[0], -1)
        hist_cv.append(cv_pre)
        hist_t.append(np.full(cv_pre.shape[0], float(n_pre)))
        tc, ts_ = _traj_coords(coordinate.detach().cpu().numpy())
        write_xyz(traj, tc, ts_, comment_prefix=f"pre-md step={n_pre}", append=True)
        plot_phi_psi(
            cv_pre,
            live / "phi_psi_after_pre_md.png",
            title=f"after pre-md {n_pre} steps",
        )
        print(
            f"[pre-md] done  φ med={np.median(cv_pre[:, 0]):.3f}  "
            f"ψ med={np.median(cv_pre[:, 1]):.3f}  "
            f"saved {out / 'init_coords_from_pre_md.npy'}",
            flush=True,
        )

    g_phi = np.linspace(args.fes_phi_min, args.fes_phi_max, 73)
    g_psi = np.linspace(args.fes_psi_min, args.fes_psi_max, 73)

    def _all():
        return np.concatenate(hist_cv), np.concatenate(hist_t)

    def _live(done: int):
        cv_a, _ = _all()
        plot_phi_psi(cv_a, live / "phi_psi.png", title=f"live φ–ψ step={done}")
        try:
            F = bias.free_energy_grid(g_phi, g_psi)
            np.savez(live / "fes.npz", phi=g_phi, psi=g_psi, F=F)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.contourf(g_phi, g_psi, F - np.nanmin(F), levels=30, cmap="viridis")
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
        tc, ts_ = _traj_coords(coords.detach().cpu().numpy())
        write_xyz(
            traj,
            tc,
            ts_,
            comment_prefix=f"step={done}",
            append=True,
            max_frames=args.max_xyz_frames,
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
    tc, ts_ = _traj_coords(fc.detach().cpu().numpy())
    write_xyz(traj, tc, ts_, comment_prefix="final", append=True)
    hills_out = "final_hills_rerun.npz" if args.resume_hills else "final_hills.npz"
    coords_out = "final_coords_rerun.npy" if args.resume_hills else "final_coords.npy"
    np.savez(
        out / hills_out,
        centers=bias._centers[: bias.n_hills].detach().cpu().numpy(),
        heights=bias._heights[: bias.n_hills].detach().cpu().numpy(),
        n_hills=bias.n_hills,
    )
    np.save(out / coords_out, fc.detach().cpu().numpy().astype(np.float32))
    _live(int(args.steps))
    print(f"done out={out}  live={live}", flush=True)


if __name__ == "__main__":
    main()
