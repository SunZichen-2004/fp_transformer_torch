#!/usr/bin/env python3
"""Solvated Claisen + ML/MM + 1D umbrella sampling on ξ = d1 − d2.

  d1 = d(O7, C8)  (breaking),  d2 = d(C1, C10)  (forming)
  ξ  = d1 − d2

Protocol
--------
1. **Relax** (1 replica): plain NVT (walls only).
2. **Steered init** (1 replica): from post-relax ξ, hop the harmonic center
   every ``hop_ps`` to the next 1D window (nearest → last, reverse-fill).
3. **Production** (R replicas): batched 1D US.

Default windows: ξ ∈ [−3.5, 2.5] Å, κ = 100 kcal mol⁻¹ Å⁻².
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

from io_utils import load_xyz, write_xyz  # noqa: E402
from fp_transformer_torch import (  # noqa: E402
    MultiParticlePairDistance,
    PairDistanceDifference,
    QuadraticDistanceWall,
    SoftminDistanceUpperWall,
    TorchSimUmbrellaSampling,
    UmbrellaBias,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402

DEFAULT_BOX = EXPERIMENT_DIR / "mlmm_box"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
KCAL_TO_EV = 1.0 / EV_TO_KCAL


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
        raise ValueError(f"need exactly 2 pairs in {path}, got {len(rows)}")
    return rows


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


def _cv_np(cv_fn, coordinate: torch.Tensor) -> np.ndarray:
    """Return ``(B, n_cv)``."""
    cv = cv_fn(coordinate).detach().cpu().numpy()
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], -1)
    return np.asarray(cv, dtype=np.float64)


def nearest_window_index(xi: float, centers: np.ndarray) -> int:
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    return int(np.argmin((c - float(xi)) ** 2))


def resolve_kappa_eV(args) -> float:
    if args.kappa is not None:
        vals = _parse_floats(args.kappa)
        if len(vals) != 1:
            raise ValueError(f"1D US: --kappa needs 1 float (eV/Å²), got {vals}")
        return float(vals[0])
    vals = _parse_floats(args.kappa_kcal)
    if len(vals) != 1:
        raise ValueError(f"1D US: --kappa-kcal needs 1 float, got {vals}")
    return float(vals[0]) * KCAL_TO_EV


def make_xi_centers(xi_min: float, xi_max: float, n_windows: int) -> np.ndarray:
    """Evenly spaced 1D centers ``(n_windows, 1)``."""
    n_windows = int(n_windows)
    if n_windows < 2:
        raise ValueError("n_windows must be >= 2")
    xi = np.linspace(float(xi_min), float(xi_max), n_windows, dtype=np.float64)
    return xi.reshape(-1, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--xyz", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--cell-file", type=str, default=None)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument(
        "--steer-start-npy",
        type=str,
        default=None,
        help="Skip relax; steered hop starts from this frame (N,3) or (B,N,3).",
    )
    p.add_argument(
        "--steer-start-index",
        type=int,
        default=0,
        help="If --steer-start-npy is (B,N,3), use this replica/window index.",
    )
    p.add_argument("--relax-steps", type=int, default=2000)
    p.add_argument("--hop-ps", type=float, default=0.2)
    p.add_argument("--skip-init", action="store_true")
    p.add_argument("--max-xyz-frames", type=int, default=32)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true")
    p.add_argument("--pme-cutoff", type=float, default=9.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument(
        "--electrostatics", type=str, default="pme", choices=["pme", "dsf"]
    )
    p.add_argument("--dsf-alpha", type=float, default=0.2)
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--softmin-a", type=float, default=10.0)
    p.add_argument("--softmin-d-wall", type=float, default=2.4)
    p.add_argument("--softmin-k", type=float, default=10.0)
    p.add_argument(
        "--kappa-kcal",
        type=str,
        default="100.0",
        help="harmonic κ in kcal/mol/Å² for ξ (default 100)",
    )
    p.add_argument(
        "--kappa",
        type=str,
        default=None,
        help="override κ in eV/Å²; else use --kappa-kcal",
    )
    p.add_argument(
        "--xi-min",
        type=float,
        default=-3.5,
        help="first window center for ξ=d1−d2 (Å); reactant ~−3.5",
    )
    p.add_argument(
        "--xi-max",
        type=float,
        default=2.5,
        help="last window center for ξ=d1−d2 (Å); product ~+2.5",
    )
    p.add_argument("--centers-npy", type=str, default=None, help="override centers (B,1)")
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--n-gpus", type=int, default=4)
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


def steered_window_init(
    *,
    state,
    mlmm,
    wall,
    softmin_wall,
    cv_fn,
    centers: np.ndarray,
    kappa: float,
    device,
    dtype,
    num_atoms: int,
    temperature: float,
    timestep: float,
    gamma: float,
    hop_ps: float,
    update_pace: int,
    out: Path,
) -> tuple[np.ndarray, np.ndarray, object]:
    from torch_sim.models.interface import SumModel

    n_win = int(centers.shape[0])
    centers_1d = centers.reshape(-1)
    hop_steps = max(1, int(round(float(hop_ps) / float(timestep))))
    coords = sim_state_to_batched_coordinates(state)
    xi0 = float(_cv_np(cv_fn, coords)[0, 0])
    i0 = nearest_window_index(xi0, centers_1d)
    print(
        f"[steer] start ξ={xi0:.3f} Å → nearest window={i0} "
        f"center={centers_1d[i0]:.3f}  hop={hop_ps} ps ({hop_steps} steps)",
        flush=True,
    )

    order = list(range(i0, n_win)) + list(range(i0 - 1, -1, -1))
    init_coords = np.zeros((n_win, coords.shape[1], 3), dtype=np.float32)
    init_xi = np.zeros(n_win, dtype=np.float64)
    filled = np.zeros(n_win, dtype=bool)

    umbrella = UmbrellaBias(
        cv_fn,
        centers[i0 : i0 + 1],
        kappa=kappa,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mlmm, umbrella, wall, softmin_wall)
    xi_hist = [xi0]
    idx_hist = [i0]

    for k, iw in enumerate(order):
        umbrella.set_centers(centers[iw : iw + 1])
        state = _run_nvt(
            state,
            model,
            steps=hop_steps,
            temperature=temperature,
            timestep=timestep,
            gamma=gamma,
            update_pace=update_pace,
            label=f"steer-w{iw:02d}",
        )
        coords = sim_state_to_batched_coordinates(state)
        xi = float(_cv_np(cv_fn, coords)[0, 0])
        frame = coords[0].detach().cpu().numpy().astype(np.float32)
        if not filled[iw]:
            init_coords[iw] = frame
            init_xi[iw] = xi
            filled[iw] = True
        xi_hist.append(xi)
        idx_hist.append(iw)
        print(
            f"[steer] {k + 1}/{len(order)}  window={iw}  "
            f"ξ={xi:.3f}  target={centers_1d[iw]:.3f}  "
            f"|Δ|={abs(xi - centers_1d[iw]):.3f}",
            flush=True,
        )

    if not bool(np.all(filled)):
        raise RuntimeError(f"steered init missed windows: {np.where(~filled)[0].tolist()}")

    np.save(out / "init_coords.npy", init_coords)
    np.save(out / "init_cv.npy", init_xi.astype(np.float32))
    np.savez(
        out / "steer_log.npz",
        order=np.asarray(order, dtype=np.int64),
        start_window=i0,
        xi_start=xi0,
        xi_hist=np.asarray(xi_hist, dtype=np.float64),
        window_hist=np.asarray(idx_hist, dtype=np.int64),
        centers=centers,
        hop_ps=float(hop_ps),
        hop_steps=hop_steps,
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(centers_1d, np.zeros_like(centers_1d), "k|", ms=10, label="windows")
        ax.plot(xi_hist, np.arange(len(xi_hist)), "o-", c="C1", ms=3, lw=1, label="steer")
        ax.scatter(init_xi, np.zeros_like(init_xi) - 0.5, c=np.arange(n_win), cmap="turbo", s=18)
        ax.set_xlabel(r"$\xi = d_1 - d_2$ / Å")
        ax.set_ylabel("steer step / window marker")
        ax.set_title("1D steered US init")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "live" / "steer_path.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] steer plot: {exc}", flush=True)

    print(f"[steer] wrote init_coords.npy shape={init_coords.shape}", flush=True)
    return init_coords, init_xi, state


def main() -> None:
    args = build_parser().parse_args()
    if args.traj_full:
        args.traj_solute_only = False

    primary = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if args.devices:
        mace_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mace_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mace_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    from torch_sim.models.interface import SumModel

    box = Path(args.box_dir)
    xyz_path = Path(args.xyz or (box / "box.xyz"))
    topo_path = Path(args.topology or (box / "topology.npz"))
    cell_path = Path(args.cell_file or (box / "cell.txt"))
    if not topo_path.is_file():
        raise FileNotFoundError(f"missing {topo_path}")

    symbols, xyz_coords = load_xyz(xyz_path)
    topo = MLMMTopology.load_npz(topo_path).to(device, dtype)
    cell = torch.as_tensor(
        np.loadtxt(cell_path, dtype=np.float64).reshape(3, 3), dtype=dtype, device=device
    )
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(f"xyz N={xyz_coords.shape[0]} != topology N={num_atoms}")

    n_solute = int(topo.ml_mask.sum().item())
    solute_symbols = symbols[:n_solute]
    pairs = load_pair_indices(Path(args.cv_file))
    n_win = int(args.replicas)

    if args.centers_npy and Path(args.centers_npy).is_file():
        centers = np.load(Path(args.centers_npy)).astype(np.float64)
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        if centers.shape != (n_win, 1):
            raise ValueError(f"centers must be (B,1)=({n_win},1), got {centers.shape}")
    else:
        centers = make_xi_centers(args.xi_min, args.xi_max, n_win)

    kappa = resolve_kappa_eV(args)
    kappa_kcal = kappa * EV_TO_KCAL

    out = Path(
        args.output
        or (EXPERIMENT_DIR / "results" / f"claisen_water_us1d_{args.mace_model}_r{n_win}")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "live").mkdir(parents=True, exist_ok=True)

    print(
        f"Claisen-water US-1D  device={device}  mace={args.mace_model}  "
        f"N={num_atoms} ML={n_solute}  windows={n_win}  prod_steps={args.steps}  "
        f"T={args.temperature} K  mace_gpus={len(mace_devices)}",
        flush=True,
    )
    print(f"[CV] ξ=d1−d2  pairs(0-based)={pairs}", flush=True)
    print(
        f"[US] κ={kappa_kcal:.4g} kcal/mol/Å²  ({kappa:.6g} eV/Å²)  "
        f"ξ ∈ [{centers[0, 0]:.3f}, {centers[-1, 0]:.3f}] Å",
        flush=True,
    )

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    mlmm = MechanicalEmbeddingMLMM(
        topo,
        mace_model=mace_model,
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        dsf_alpha=float(args.dsf_alpha),
        pme_accuracy=1e-4,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        mm_devices=None,
    )

    # Umbrella CV is 1D ξ; walls still act on the two bond distances.
    cv_fn = PairDistanceDifference(pairs)
    dist2 = MultiParticlePairDistance(pairs)
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    softmin_wall = SoftminDistanceUpperWall(
        pairs[0],
        pairs[1],
        a=float(args.softmin_a),
        d_wall=float(args.softmin_d_wall),
        k=float(args.softmin_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )

    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)

    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        if loaded.shape[0] < n_win:
            raise ValueError(f"init-npy has {loaded.shape[0]} frames < {n_win}")
        if loaded.shape[0] != n_win:
            idx = np.linspace(0, loaded.shape[0] - 1, n_win).astype(np.int64)
            loaded = loaded[idx]
        init_coords = loaded.astype(np.float32)
        print(f"[init] loaded {args.init_npy} → {init_coords.shape}", flush=True)
        np.save(out / "init_coords.npy", init_coords)
    elif args.skip_init:
        init_coords = np.repeat(xyz_coords[None, ...], n_win, axis=0).astype(np.float32)
        print(f"[init] skip steer; box xyz × {n_win}", flush=True)
        np.save(out / "init_coords.npy", init_coords)
    else:
        if args.steer_start_npy:
            loaded = np.load(Path(args.steer_start_npy))
            if loaded.ndim == 3:
                idx = int(args.steer_start_index)
                if idx < 0 or idx >= loaded.shape[0]:
                    raise ValueError(
                        f"steer-start-index {idx} out of range for {loaded.shape}"
                    )
                frame = loaded[idx]
                print(
                    f"[steer-start] {args.steer_start_npy}[{idx}] shape={frame.shape}",
                    flush=True,
                )
            elif loaded.ndim == 2:
                frame = loaded
                print(
                    f"[steer-start] {args.steer_start_npy} shape={frame.shape}",
                    flush=True,
                )
            else:
                raise ValueError(
                    f"steer-start-npy must be (N,3) or (B,N,3), got {loaded.shape}"
                )
            if frame.shape != xyz_coords.shape:
                raise ValueError(
                    f"steer-start atoms {frame.shape} != box xyz {xyz_coords.shape}"
                )
            coord1 = torch.as_tensor(frame[None, ...], dtype=dtype, device=device)
            state1 = batched_coordinates_to_sim_state(
                coord1,
                atomic_numbers=z_t,
                masses=mass_t,
                cell=cell,
                pbc=True,
                charge=float(args.charge),
                spin=float(args.spin),
            )
            state1 = _attach_charges(state1, topo.charges, 1)
            d12 = _cv_np(dist2, coord1)[0]
            print(
                f"[steer-start] skip relax  d1={d12[0]:.3f} d2={d12[1]:.3f} "
                f"ξ={float(d12[0] - d12[1]):.3f}",
                flush=True,
            )
            np.save(out / "steer_start_coords.npy", frame.astype(np.float32))
            np.save(
                out / "steer_start_cv.npy",
                np.asarray([[d12[0] - d12[1]]], dtype=np.float32),
            )
        else:
            coord1 = torch.as_tensor(xyz_coords[None, ...], dtype=dtype, device=device)
            state1 = batched_coordinates_to_sim_state(
                coord1,
                atomic_numbers=z_t,
                masses=mass_t,
                cell=cell,
                pbc=True,
                charge=float(args.charge),
                spin=float(args.spin),
            )
            state1 = _attach_charges(state1, topo.charges, 1)
            d12 = _cv_np(dist2, coord1)[0]
            xi_box = float(d12[0] - d12[1])
            print(
                f"[relax] plain MD steps={args.relax_steps}  "
                f"box d1={d12[0]:.3f} d2={d12[1]:.3f} ξ={xi_box:.3f}",
                flush=True,
            )
            relax_model = SumModel(mlmm, wall, softmin_wall)
            state1 = _run_nvt(
                state1,
                relax_model,
                steps=int(args.relax_steps),
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                update_pace=int(args.update_pace),
                label="relax",
            )
            relax_t = sim_state_to_batched_coordinates(state1)
            np.save(
                out / "relax_coords.npy",
                relax_t.detach().cpu().numpy().astype(np.float32),
            )
            np.save(out / "relax_cv.npy", _cv_np(cv_fn, relax_t).astype(np.float32))

        init_coords, _, _ = steered_window_init(
            state=state1,
            mlmm=mlmm,
            wall=wall,
            softmin_wall=softmin_wall,
            cv_fn=cv_fn,
            centers=centers,
            kappa=kappa,
            device=device,
            dtype=dtype,
            num_atoms=num_atoms,
            temperature=float(args.temperature),
            timestep=float(args.timestep),
            gamma=float(args.langevin_friction),
            hop_ps=float(args.hop_ps),
            update_pace=int(args.update_pace),
            out=out,
        )

    coordinate = torch.as_tensor(init_coords, dtype=dtype, device=device)
    umbrella = UmbrellaBias(
        cv_fn,
        centers,
        kappa=kappa,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mlmm, umbrella, wall, softmin_wall)
    driver = TorchSimUmbrellaSampling(
        model,
        umbrella,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        update_pace=int(args.update_pace),
    )

    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=True,
        charge=float(args.charge),
        spin=float(args.spin),
    )
    state0 = _attach_charges(state0, topo.charges, n_win)

    traj = out / "trajectory.xyz"
    if traj.is_file():
        traj.unlink()

    np.save(out / "umbrella_centers.npy", centers.astype(np.float64))
    kap_arr = umbrella.kappa.detach().cpu().numpy()
    np.save(out / "umbrella_kappa.npy", kap_arr)
    with open(out / "run_info.txt", "w", encoding="utf-8") as fh:
        fh.write("cv=xi=d1-d2\n")
        fh.write(f"replicas={n_win}\n")
        fh.write(f"steps={args.steps}\n")
        fh.write(f"timestep_ps={args.timestep}\n")
        fh.write(f"total_time_ps={args.steps * args.timestep}\n")
        fh.write(f"temperature_K={args.temperature}\n")
        fh.write(f"kappa_eV_per_A2={kappa}\n")
        fh.write(f"kappa_kcal_per_mol_A2={kappa_kcal}\n")
        fh.write(f"xi_min={float(centers[0, 0])}\n")
        fh.write(f"xi_max={float(centers[-1, 0])}\n")
        fh.write(f"relax_steps={args.relax_steps}\n")
        fh.write(f"hop_ps={args.hop_ps}\n")
        fh.write(f"mace={args.mace_model}\n")
        fh.write(f"n_gpus={len(mace_devices)}\n")
        fh.write(f"pairs={pairs}\n")

    def _traj_coords(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if args.traj_solute_only:
            return coords[..., :n_solute, :], solute_symbols
        return coords, symbols

    cv0 = _cv_np(cv_fn, coordinate)
    driver.cv_trajectory.append(cv0.astype(np.float64))
    driver.cv_times_ps.append(0.0)
    c0, s0 = _traj_coords(coordinate.detach().cpu().numpy())
    write_xyz(traj, c0, s0, comment_prefix="step=0")
    np.save(out / "trajectory_cv.npy", cv0[None, ...].astype(np.float32))
    np.save(out / "trajectory_cv_times_ps.npy", np.array([0.0]))

    centers_1d = centers.reshape(-1)

    def _live(done: int, cv_frame: np.ndarray | None = None):
        times, cv_traj = driver.stacked_cv_trajectory()
        np.save(out / "trajectory_cv.npy", cv_traj.astype(np.float32))
        np.save(out / "trajectory_cv_times_ps.npy", times.astype(np.float64))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            last = (cv_frame if cv_frame is not None else cv_traj[-1]).reshape(-1)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.plot(centers_1d, np.arange(n_win), "k.--", lw=1, ms=4, label="centers")
            ax.scatter(
                last,
                np.arange(n_win),
                c=np.arange(n_win),
                cmap="turbo",
                s=22,
                alpha=0.9,
                linewidths=0,
                label="replicas",
            )
            ax.set_xlabel(r"$\xi = d_1 - d_2$ / Å")
            ax.set_ylabel("window index")
            ax.set_title(f"1D US step={done}")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(out / "live" / "windows_xi.png", dpi=140)
            plt.close(fig)

            if cv_traj.shape[0] >= 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                delta = cv_traj.reshape(cv_traj.shape[0], -1) - centers_1d[None, :]
                rms = np.abs(delta)
                ax.plot(times, rms.mean(axis=1), c="C0", label="mean |ξ−ξ0|")
                ax.fill_between(
                    times,
                    rms.min(axis=1),
                    rms.max(axis=1),
                    color="C0",
                    alpha=0.2,
                    label="min–max",
                )
                ax.set_xlabel("t / ps")
                ax.set_ylabel(r"$|\xi-\xi_0|$ / Å")
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
            f"ξ med={np.median(cv[:, 0]):.3f}",
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
