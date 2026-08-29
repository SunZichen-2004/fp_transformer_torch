#!/usr/bin/env python3
"""Vacuum SN2 (Cl- + CH3Cl) + MACE-MH-0 + 2D distance CV + torch-sim BAOAB-FP.

Atom order in ``sn2.pdb`` / ``init_coords_*.npy``:
  0 Cl_nuc, 1 C, 2–4 H, 5 Cl_lg

CVs: d(Cl_nuc–C), d(Cl_lg–C). Quadratic upper wall for each when d > 3.5 Å.
Total charge −1 (omol head). Metal units: eV / Å / amu / ps.
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

from io_utils import save_stage_artifacts, symbols_from_z  # noqa: E402
from fp_transformer_torch import (  # noqa: E402
    FPTBias,
    GaussianKernel,
    MultiParticlePairDistance,
    PerReplicaRescaleConfig,
    QuadraticDistanceWall,
    TorchSimFokkerPlanckIntegrator,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.mace import (  # noqa: E402
    atomic_numbers_from_pdb_organic,
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.sharded_mace import (  # noqa: E402
    parse_cuda_devices,
)

DEFAULT_PDB = EXPERIMENT_DIR / "sn2.pdb"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
DEFAULT_MASS = EXPERIMENT_DIR / "sn2_mass.txt"
_DEFAULT_MD = EXPERIMENT_DIR / "init_coords_md_1024.npy"
_DEFAULT_GEOM = EXPERIMENT_DIR / "init_coords_1024.npy"
DEFAULT_INIT_NPY = _DEFAULT_MD if _DEFAULT_MD.is_file() else _DEFAULT_GEOM

KB_EV_PER_K = 8.617330337217213e-5


def initial_bandwidth_from_collective_variable(
    collective_variable: torch.Tensor,
) -> float:
    distance = pairwise_distance(collective_variable)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    if not np.isfinite(median) or median <= 0.0:
        return 0.5
    return median


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
    if not rows:
        raise ValueError(f"no pair indices in {path}")
    return rows


def load_masses(path: Path, num_atoms: int) -> np.ndarray:
    mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
        mass_vals = mass_vals[1:]
    if mass_vals.size < num_atoms:
        raise ValueError(f"mass file too short: {path}")
    return mass_vals[:num_atoms].astype(np.float64)


def atomic_numbers_from_pdb(path: Path) -> np.ndarray:
    return atomic_numbers_from_pdb_organic(path).cpu().numpy().astype(np.int64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mace-model",
        type=str,
        default="mh-0",
        help="Named model (mh-0 / off-small / ...) or checkpoint path",
    )
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--pdb", type=str, default=str(DEFAULT_PDB))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=1024)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--timestep",
        type=float,
        default=0.001,
        help="BAOAB dt in picoseconds (torch_sim metal units)",
    )
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument(
        "--langevin-friction",
        type=float,
        default=1.0,
        help="Langevin γ in 1/ps",
    )
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--lambda-init", type=float, default=0.05)
    p.add_argument("--lambda-max", type=float, default=0.5)
    p.add_argument("--ess-threshold", type=float, default=0.9)
    p.add_argument("--min-lambda-step", type=float, default=0.05)
    p.add_argument("--max-lambda-step", type=float, default=0.2)
    p.add_argument("--num-lambda-trials", type=int, default=10)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--noise-scale", type=float, default=0.0)
    p.add_argument(
        "--init-npy",
        type=str,
        default=str(DEFAULT_INIT_NPY) if Path(DEFAULT_INIT_NPY).is_file() else None,
    )
    p.add_argument(
        "--max-xyz-frames",
        type=int,
        default=None,
        help="Cap XYZ frames per stage (default: all replicas).",
    )
    p.add_argument(
        "--pre-eq-steps",
        type=int,
        default=200,
        help="NVT steps with FP bias disabled before FPT (thermalize velocities).",
    )
    p.add_argument("--cell-length", type=float, default=30.0)
    p.add_argument("--charge", type=float, default=-1.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=3.5)
    p.add_argument(
        "--wall-k",
        type=float,
        default=20.0,
        help="Quadratic wall stiffness (eV/Å²) for d > d_max",
    )
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Primary device for SimState / FPTBias (e.g. cuda or cuda:0)",
    )
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated CUDA devices for sharded MACE",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs for MACE replica sharding",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--cv-grad-clip-norm",
        type=float,
        default=0.05,
        help="Clip CV-space kernel gradient by global norm. Set <=0 to disable.",
    )
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=4.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument("--rew-lambda-init", type=float, default=None)
    p.add_argument("--rew-lambda-max", type=float, default=None)
    p.add_argument("--rew-ess-threshold", type=float, default=None)
    p.add_argument("--rew-min-lambda-step", type=float, default=None)
    p.add_argument("--rew-max-lambda-step", type=float, default=None)
    p.add_argument("--rew-num-lambda-trials", type=int, default=None)
    p.add_argument("--rew-anneal-ref-temperature", type=float, default=None)
    return p


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.steps is not None:
        args.stage_steps = args.steps
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    if args.rew_lambda_init is not None:
        args.lambda_init = args.rew_lambda_init
    if args.rew_lambda_max is not None:
        args.lambda_max = args.rew_lambda_max
    if args.rew_ess_threshold is not None:
        args.ess_threshold = args.rew_ess_threshold
    if args.rew_min_lambda_step is not None:
        args.min_lambda_step = args.rew_min_lambda_step
    if args.rew_max_lambda_step is not None:
        args.max_lambda_step = args.rew_max_lambda_step
    if args.rew_num_lambda_trials is not None:
        args.num_lambda_trials = args.rew_num_lambda_trials
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    temp = args.temperature
    if args.rew_anneal_ref_temperature is not None:
        temp = args.rew_anneal_ref_temperature
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    return args


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    primary = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if args.devices:
        mace_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mace_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mace_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    from torch_sim.models.interface import SumModel

    print(
        f"device={device}  dtype={args.dtype}  beta0={args.beta0:.6g}  "
        f"mace={args.mace_model}  T={args.temperature} K  "
        f"charge={args.charge}  spin={args.spin}  "
        f"n_gpus={len(mace_devices)}  mace_devices={[str(d) for d in mace_devices]}"
    )
    print(
        f"[baoab time] dt={args.timestep} ps  γ={args.langevin_friction} /ps  "
        f"γ·dt={args.langevin_friction * args.timestep:.6g}"
    )

    atomic_numbers = atomic_numbers_from_pdb(Path(args.pdb))
    num_atoms = int(atomic_numbers.shape[0])
    masses = load_masses(Path(args.mass_file), num_atoms)
    symbols = symbols_from_z(atomic_numbers)
    print(f"SN2 num_atoms={num_atoms}  Z={atomic_numbers.tolist()}")
    print(f"[init] using init-npy={args.init_npy}", flush=True)

    pairs = load_pair_indices(Path(args.cv_file))
    for i, j in pairs:
        if i < 0 or j < 0 or i >= num_atoms or j >= num_atoms:
            raise ValueError(f"pair ({i},{j}) out of range for n_atoms={num_atoms}")
    print(f"CV pairs (0-based)={pairs}  wall d_max={args.wall_d_max} k={args.wall_k}")

    if not args.init_npy:
        raise ValueError("--init-npy is required for SN2_mace")
    path = Path(os.path.abspath(os.path.expanduser(args.init_npy)))
    loaded = np.load(path)
    if loaded.ndim != 3 or loaded.shape[-1] != 3:
        raise ValueError(f"bad init npy {path}: {loaded.shape}")
    if loaded.shape[1] != num_atoms:
        raise ValueError(
            f"init npy n_atoms={loaded.shape[1]} != pdb n_atoms={num_atoms}"
        )
    n_loaded = int(loaded.shape[0])
    if args.replicas > n_loaded:
        raise ValueError(f"--replicas={args.replicas} > init npy B={n_loaded} ({path})")
    if args.replicas < n_loaded:
        print(
            f"[init] taking first {args.replicas} of {n_loaded} frames from {path}",
            flush=True,
        )
        loaded = loaded[: args.replicas]
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    if args.noise_scale > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        noise = torch.randn(coordinate.shape, dtype=dtype, generator=generator).to(
            device
        )
        coordinate = coordinate + args.noise_scale * noise

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    n_mace_gpus = getattr(mace_model, "n_gpus", 1)
    print(
        f"[init] MACE loaded  r_max={getattr(mace_model, 'r_max', '?')}  "
        f"n_gpus={n_mace_gpus}",
        flush=True,
    )

    collective_variable = MultiParticlePairDistance(pairs)
    collective0 = collective_variable(coordinate)
    bandwidth0 = initial_bandwidth_from_collective_variable(collective0)
    cv0 = collective0.detach().cpu().numpy().reshape(-1, 2)
    print(
        f"[init] bw_init={bandwidth0:.6g}  replicas={args.replicas}  "
        f"d1[min/med/max]={cv0[:, 0].min():.3f}/{np.median(cv0[:, 0]):.3f}/{cv0[:, 0].max():.3f}  "
        f"d2[min/med/max]={cv0[:, 1].min():.3f}/{np.median(cv0[:, 1]):.3f}/{cv0[:, 1].max():.3f}"
    )

    rescale = PerReplicaRescaleConfig(
        enabled=bool(args.rescale_per_replica),
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=args.rescale_adaptive_ref_quantile,
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
    print(
        f"[init] rescale metric=unnormalized_K_row_sum  "
        f"mode=adaptive_quantile  "
        f"band=[{args.rescale_adaptive_band_low},{args.rescale_adaptive_band_high}]"
        f"*median clipped to "
        f"[{rescale.fixed_row_sum_min},{rescale.fixed_row_sum_max}]  "
        f"bw_clip=[{rescale.bandwidth_min},{rescale.bandwidth_max}]"
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bandwidth0,
        distance_function=pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )

    cv_clip = (
        None
        if args.cv_grad_clip_norm is None or args.cv_grad_clip_norm <= 0.0
        else float(args.cv_grad_clip_norm)
    )
    kernel_grad_beta0 = float(args.lambda_init) * float(args.beta0)
    fpt_bias = FPTBias(
        collective_variable,
        kernel,
        kernel_grad_beta=kernel_grad_beta0,
        device=device,
        dtype=dtype,
        collective_variable_grad_clip=cv_clip,
        update_pace=args.update_pace,
        n_atoms_per_system=num_atoms,
    )
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mace_model, fpt_bias, wall)
    print(
        f"[init] SumModel(MACE, FPTBias, QuadraticDistanceWall)  "
        f"cv_grad_clip_norm={cv_clip}  λ_init={args.lambda_init}"
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

    integrator = TorchSimFokkerPlanckIntegrator(
        model,
        fpt_bias,
        collective_variable,
        kernel,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        beta0=float(args.beta0),
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

    out = args.output or (
        EXPERIMENT_DIR / "results" / f"sn2_mace_{args.mace_model}_{args.replicas}"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "d1_d2").mkdir(parents=True, exist_ok=True)

    def _stage_tag(stage_i: int, lam: float) -> str:
        return f"stage{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i: int, lam: float, st) -> None:
        coords = sim_state_to_batched_coordinates(st)
        coords_np = coords.detach().cpu().numpy()
        cv_np = collective_variable(coords).detach().cpu().numpy().reshape(-1, 2)
        tag = _stage_tag(stage_i, lam)
        np.save(
            out / f"{tag}_bandwidth.npy",
            integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        metrics = integrator._last_step_metrics
        metrics.update(integrator.kinetic_temperature_stats(st))
        print(
            f"[stage-end] {tag}  "
            f"T_kin mean/std/min/max="
            f"{metrics.get('kinetic_temperature_mean', float('nan')):.2f}/"
            f"{metrics.get('kinetic_temperature_std', float('nan')):.2f}/"
            f"{metrics.get('kinetic_temperature_min', float('nan')):.2f}/"
            f"{metrics.get('kinetic_temperature_max', float('nan')):.2f} K  "
            f"(target={args.temperature:.2f} K)  "
            f"d1[min/med/max]={cv_np[:,0].min():.3f}/{np.median(cv_np[:,0]):.3f}/{cv_np[:,0].max():.3f}  "
            f"d2[min/med/max]={cv_np[:,1].min():.3f}/{np.median(cv_np[:,1]):.3f}/{cv_np[:,1].max():.3f}  "
            f"bw[min/med/max]="
            f"{metrics.get('bandwidth_min', float('nan')):.4g}/"
            f"{metrics.get('bandwidth_median', float('nan')):.4g}/"
            f"{metrics.get('bandwidth_max', float('nan')):.4g}",
            flush=True,
        )
        save_stage_artifacts(
            out,
            tag,
            coords_np,
            cv_np,
            symbols,
            title=f"SN2-MACE d1–d2  {tag}",
            d_max=float(args.wall_d_max),
            max_xyz_frames=args.max_xyz_frames,
        )

    save_stage_artifacts(
        out,
        "stage_init",
        coordinate.detach().cpu().numpy(),
        cv0,
        symbols,
        title="SN2-MACE d1–d2  init",
        d_max=float(args.wall_d_max),
        max_xyz_frames=args.max_xyz_frames,
    )

    if int(args.pre_eq_steps) > 0:
        import torch_sim as ts

        print(
            f"[pre-eq] {args.pre_eq_steps} NVT steps with FP bias OFF "
            f"(MACE + wall only)",
            flush=True,
        )
        fpt_bias.use_kernel_gradient = False
        state0 = ts.integrate(
            state0,
            model,
            integrator=ts.Integrator.nvt_langevin,
            n_steps=int(args.pre_eq_steps),
            temperature=float(args.temperature),
            timestep=float(args.timestep),
            gamma=float(args.langevin_friction),
            autobatcher=False,
            pbar=True,
        )
        fpt_bias.use_kernel_gradient = True
        coords_eq = sim_state_to_batched_coordinates(state0)
        cv_eq = collective_variable(coords_eq).detach().cpu().numpy().reshape(-1, 2)
        tstats = integrator.kinetic_temperature_stats(state0)
        print(
            f"[pre-eq] done  T_kin mean/std/min/max="
            f"{tstats.get('kinetic_temperature_mean', float('nan')):.2f}/"
            f"{tstats.get('kinetic_temperature_std', float('nan')):.2f}/"
            f"{tstats.get('kinetic_temperature_min', float('nan')):.2f}/"
            f"{tstats.get('kinetic_temperature_max', float('nan')):.2f} K",
            flush=True,
        )
        save_stage_artifacts(
            out,
            "stage_preeq",
            coords_eq.detach().cpu().numpy(),
            cv_eq,
            symbols,
            title=f"SN2 after pre-eq ({args.pre_eq_steps} steps, no FP)",
            d_max=float(args.wall_d_max),
            max_xyz_frames=args.max_xyz_frames,
        )

    final_state = integrator.run(
        state0,
        args.stage_steps,
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_stage_end=on_stage_end,
    )
    final_coords = sim_state_to_batched_coordinates(final_state)
    final_cv = collective_variable(final_coords).detach().cpu().numpy()
    np.save(
        out / "final_bandwidth.npy",
        integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
    )
    save_stage_artifacts(
        out,
        "stage_final",
        final_coords.detach().cpu().numpy(),
        final_cv,
        symbols,
        title=f"SN2-MACE d1–d2  final λ={integrator.lambda_current:.4g}",
        d_max=float(args.wall_d_max),
        max_xyz_frames=args.max_xyz_frames,
    )
    print(
        f"wrote {out / 'stage_final_coords.npy'}  "
        f"cv0_shape={tuple(collective0.shape)}  lambda={integrator.lambda_current}"
    )


if __name__ == "__main__":
    main()
