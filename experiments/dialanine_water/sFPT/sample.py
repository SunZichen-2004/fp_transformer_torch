#!/usr/bin/env python3
"""Solvated dialanine (GAFF + TIP3P) + φ/ψ torsion CV + torch-sim sFPT.

Full classical MM via ``MechanicalEmbeddingMLMM(enable_mace=False)``.
Box: ``../mm_box`` (run ``../prepare_box.py`` first). Metal units (eV / Å / ps).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
WATER_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WATER_DIR))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from io_utils import load_xyz, plot_phi_psi, write_xyz  # noqa: E402
from fp_transformer_torch import (  # noqa: E402
    FPTBias,
    GaussianKernel,
    PerReplicaRescaleConfig,
    ShardedBaoabFokkerPlanckIntegrator,
    TorchSimFokkerPlanckIntegrator,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (  # noqa: E402
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402

DEFAULT_BOX = WATER_DIR / "mm_box"
DEFAULT_CV = DEFAULT_BOX / "backbone_cv.txt"
KB_EV_PER_K = 8.617330337217213e-5
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


def initial_bandwidth(cv: torch.Tensor) -> float:
    distance = torus_pairwise_distance(cv)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    return median if np.isfinite(median) and median > 0 else 0.5


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=0.2)
    p.add_argument("--ess-threshold", type=float, default=0.97)
    p.add_argument("--min-lambda-step", type=float, default=0.001)
    p.add_argument("--max-lambda-step", type=float, default=0.02)
    p.add_argument("--num-lambda-trials", type=int, default=10)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--pre-md-steps", type=int, default=100)
    p.add_argument("--heal-lambdas", type=str, default="0.1,0.05")
    p.add_argument("--heal-steps", type=int, default=100)
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument(
        "--electrostatics",
        type=str,
        default=None,
        choices=["pme", "dsf", "torchpme"],
        help="MM electrostatics. Default: pme on 1 GPU, torchpme when --n-gpus>1.",
    )
    p.add_argument(
        "--torchpme-method",
        type=str,
        default="p3m",
        choices=["p3m", "pme"],
        help="Calculator inside electrostatics=torchpme (default P3M).",
    )
    p.add_argument("--dsf-alpha", type=float, default=0.2)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated MM GPUs, e.g. 0,1,2,3 (overrides --n-gpus)",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Shard classical MM across this many GPUs (replica-parallel)",
    )
    p.add_argument(
        "--integrator",
        type=str,
        default="auto",
        choices=["auto", "torchsim", "sharded"],
        help="auto: torch-sim on 1 GPU, sharded BAOAB when --n-gpus>1. "
        "sharded: always in-repo BAOAB (use this to debug on 1 GPU).",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=20)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument(
        "--max-stages",
        type=int,
        default=None,
        help="Stop annealing after this many production stages (heal not counted)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.steps is not None:
        args.stage_steps = int(args.steps)
    heal = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))

    from torch_sim.models.interface import SumModel
    import torch_sim as ts

    primary = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if args.devices:
        mm_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mm_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mm_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.electrostatics is None:
        args.electrostatics = "torchpme" if len(mm_devices) > 1 else "pme"
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

    mm_kw = dict(
        mace_model=None,
        enable_mace=False,
        enable_ml_coulomb_correction=False,
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        dsf_alpha=float(args.dsf_alpha),
        pme_accuracy=1e-4,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        torchpme_method=str(args.torchpme_method),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )
    if len(mm_devices) > 1:
        mm_kw["mm_devices"] = mm_devices
    mm = MechanicalEmbeddingMLMM(topo, **mm_kw)
    n_mm = len(mm._mm_workers) if mm._mm_workers else 1
    print(
        f"Dialanine-water sFPT  device={device}  N={num_atoms}  "
        f"R={args.replicas}  λ0={args.lambda_init}  elec={args.electrostatics}"
        + (f"/{args.torchpme_method}" if args.electrostatics == "torchpme" else "")
        + f"  cutoff={args.pme_cutoff}  box={float(cell[0, 0]):.1f} Å  "
        f"mm_gpus={n_mm}  devices={[str(d) for d in mm_devices]}",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={tors.tolist()}", flush=True)

    out = Path(
        args.output
        or (EXPERIMENT_DIR / "results" / f"dialanine_water_sfpt_r{args.replicas}_s{args.seed}")
    )
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "phi_psi"
    plots.mkdir(parents=True, exist_ok=True)

    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)

    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        coordinate = torch.as_tensor(loaded[idx], dtype=dtype, device=device)
        print(f"[init] from {args.init_npy}", flush=True)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        if args.pre_md_steps > 0:
            print(f"[pre-md] {args.pre_md_steps} unbiased NVT steps", flush=True)
            state_md = batched_coordinates_to_sim_state(
                coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
            )
            state_md = _attach_charges(state_md, topo.charges.to(device=device, dtype=dtype), args.replicas)
            state_md = ts.integrate(
                state_md,
                mm,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=int(args.pre_md_steps),
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                pbar=False,
            )
            coordinate = sim_state_to_batched_coordinates(state_md).to(dtype=dtype)
            np.save(out / "init_coords_from_pre_md.npy", coordinate.detach().cpu().numpy())

    cv0 = cv_fn(coordinate)
    bw0 = initial_bandwidth(cv0)
    print(f"[init] bw={bw0:.4g}  R={args.replicas}", flush=True)
    plot_phi_psi(cv0.detach().cpu().numpy(), plots / "stage_init_phi_psi.png", title="init")

    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    rescale = PerReplicaRescaleConfig(
        enabled=True,
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=0.5,
        adaptive_band_low=args.rescale_adaptive_band_low,
        adaptive_band_high=args.rescale_adaptive_band_high,
        bandwidth_min=args.rescale_bw_min,
        bandwidth_max=args.rescale_bw_max,
        global_pull=0.02,
        fixed_row_sum_min=float(args.rescale_fixed_row_sum_min),
        fixed_row_sum_max=float(args.rescale_fixed_row_sum_max),
        use_normalized_kernel=False,
        normalize_row_sum_by_replicas=False,
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bw0,
        distance_function=torus_pairwise_distance,
        rescale_per_replica=True,
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )
    driver_kw = dict(
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        beta0=beta0,
        lambda_init=float(args.lambda_init),
        lambda_max=float(args.lambda_max),
        ess_threshold=float(args.ess_threshold),
        min_lambda_step=float(args.min_lambda_step),
        max_lambda_step=float(args.max_lambda_step),
        num_lambda_trials=int(args.num_lambda_trials),
        update_pace=int(args.update_pace),
        device=device,
        dtype=dtype,
    )
    use_sharded = args.integrator == "sharded" or (
        args.integrator == "auto" and n_mm > 1
    )
    if use_sharded:
        print(
            f"[integrator] sharded BAOAB  replicas resident on {n_mm} GPUs; "
            "primary receives CV atoms only",
            flush=True,
        )
        driver = ShardedBaoabFokkerPlanckIntegrator(
            mm, cv_fn, kernel, **driver_kw
        )
    else:
        fpt_bias = FPTBias(
            cv_fn,
            kernel,
            kernel_grad_beta=float(args.lambda_init) * float(beta0),
            device=device,
            dtype=dtype,
            use_kernel_gradient=True,
            update_pace=int(args.update_pace),
            n_atoms_per_system=num_atoms,
        )
        model = SumModel(mm, fpt_bias)
        driver = TorchSimFokkerPlanckIntegrator(
            model, fpt_bias, cv_fn, kernel, **driver_kw
        )

    state0 = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
    )
    state0 = _attach_charges(state0, topo.charges.to(device=device, dtype=dtype), args.replicas)

    def on_stage_end(stage_i: int, lam: float, st, phase: str = "anneal") -> None:
        coords = sim_state_to_batched_coordinates(st)
        cv = cv_fn(coords).detach().cpu().numpy()
        if cv.ndim == 3:
            cv = cv.reshape(cv.shape[0], -1)
        pref = "heal" if phase == "heal" else "stage"
        if phase == "pre_md":
            pref = "pre_md"
        tag = f"{pref}{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")
        np.save(out / f"{tag}_dihedral.npy", cv)
        np.save(
            out / f"{tag}_bandwidth.npy",
            driver.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        plot_phi_psi(cv, plots / f"{tag}_phi_psi.png", title=tag)
        print(f"[{phase}-end] {tag}", flush=True)

    final = driver.run(
        state0,
        int(args.stage_steps),
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_stage_end=on_stage_end,
        heal_lambdas=heal or None,
        heal_steps=int(args.heal_steps) if heal else None,
        max_stages=int(args.max_stages) if args.max_stages is not None else None,
    )
    fc = sim_state_to_batched_coordinates(final)
    np.save(out / "final_coords.npy", fc.detach().cpu().numpy())
    write_xyz(
        out / "final_solute.xyz",
        fc[0, :N_SOLUTE].detach().cpu().numpy(),
        solute_symbols,
        comment_prefix="final",
    )
    print(f"done out={out}", flush=True)


if __name__ == "__main__":
    main()
