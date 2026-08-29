#!/usr/bin/env python3
"""Vacuum DA (tropone + dimethylfulvene) + MACE-MH-0 + soft-distance FPT.

Two CV modes (same atom pairs 1-based: 2–15, 3–13, 7–24)::

  --cv-mode s12   : s1=c1+c2+c3, s2=c2-c3   (same as DA_mace_meta)
  --cv-mode c123  : cv=(c1,c2,c3); plots also include derived s1–s2

Nonbond lower walls (same as DA_mace_meta): C6–C24, C3–C18, O8–C20,
``r_min=2.4 Å``, ``k=100 kcal/mol/Å²``.
Metal units: eV / Å / amu / ps.
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
    save_stage_artifacts,
)
from fp_transformer_torch import (  # noqa: E402
    FPTBias,
    GaussianKernel,
    PerReplicaRescaleConfig,
    QuadraticDistanceLowerWall,
    QuadraticDistanceWall,
    SoftDistance,
    SoftDistanceS12,
    TorchSimFokkerPlanckIntegrator,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402

DEFAULT_XYZ = EXPERIMENT_DIR / "DA_tropone_dimethylfulvene.xyz"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
DEFAULT_NONBOND = EXPERIMENT_DIR / "nonbond_pairs.txt"
DEFAULT_MASS = EXPERIMENT_DIR / "da_mass.txt"
_DEFAULT_MD = EXPERIMENT_DIR / "init_coords_md_1024.npy"
_DEFAULT_GEOM = EXPERIMENT_DIR / "init_coords_1024.npy"
DEFAULT_INIT_NPY = _DEFAULT_MD if _DEFAULT_MD.is_file() else _DEFAULT_GEOM

KB_EV_PER_K = 8.617330337217213e-5
KCAL_PER_MOL_TO_EV = 1.0 / 23.06054783061903


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


def load_pair_indices(path: Path, *, expect: int | None = 3) -> list[tuple[int, int]]:
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cv-mode",
        type=str,
        default="s12",
        choices=["s12", "c123"],
        help="s12: (s1,s2); c123: (c1,c2,c3) with extra s1–s2 plots",
    )
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--xyz", type=str, default=str(DEFAULT_XYZ))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--nonbond-file", type=str, default=str(DEFAULT_NONBOND))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=1024)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=1.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--lambda-init", type=float, default=0.1)
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
        default=str(DEFAULT_INIT_NPY) if Path(str(DEFAULT_INIT_NPY)).is_file() else None,
    )
    p.add_argument("--max-xyz-frames", type=int, default=None)
    p.add_argument("--heal-lambdas", type=str, default=None)
    p.add_argument("--heal-steps", type=int, default=100)
    p.add_argument("--cell-length", type=float, default=40.0)
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--d-ref", type=float, default=1.7)
    p.add_argument("--soft-n", type=int, default=6)
    p.add_argument("--soft-m", type=int, default=8)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--nonbond-r-min", type=float, default=2.4)
    p.add_argument("--nonbond-k-kcal", type=float, default=100.0)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cv-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=4.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=5.0)
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
    heal: list[float] = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    args.heal_lambda_list = heal
    return args


def _cv_stats(cv: np.ndarray, mode: str) -> str:
    names = ("s1", "s2") if mode == "s12" else ("c1", "c2", "c3")
    bits = []
    for i, name in enumerate(names):
        if i >= cv.shape[1]:
            break
        col = cv[:, i]
        bits.append(
            f"{name}[min/med/max]={col.min():.3f}/{np.median(col):.3f}/{col.max():.3f}"
        )
    return "  ".join(bits)


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
    cv_mode = str(args.cv_mode).lower()

    from torch_sim.models.interface import SumModel

    print(
        f"device={device}  dtype={args.dtype}  beta0={args.beta0:.6g}  "
        f"mace={args.mace_model}  T={args.temperature} K  charge={args.charge}  "
        f"d_ref={args.d_ref}  n_gpus={len(mace_devices)}  cv_mode={cv_mode}",
        flush=True,
    )

    symbols, _ = load_xyz(Path(args.xyz))
    atomic_numbers = atomic_numbers_from_symbols(symbols)
    num_atoms = int(atomic_numbers.shape[0])
    masses = load_masses(Path(args.mass_file), num_atoms)
    pairs = load_pair_indices(Path(args.cv_file), expect=3)
    nonbond = load_pair_indices(Path(args.nonbond_file), expect=None)
    print(
        f"DA num_atoms={num_atoms}  CV pairs(0-based)={pairs}  "
        f"(1-based 2–15, 3–13, 7–24)",
        flush=True,
    )
    print(
        f"[nonbond] pairs(0-based)={nonbond}  r_min={args.nonbond_r_min} Å  "
        f"k={args.nonbond_k_kcal} kcal/mol/Å²",
        flush=True,
    )
    print(f"[init] using init-npy={args.init_npy}", flush=True)

    if not args.init_npy:
        raise ValueError("--init-npy is required")
    path = Path(os.path.abspath(os.path.expanduser(args.init_npy)))
    loaded = np.load(path)
    if loaded.ndim != 3 or loaded.shape[1] != num_atoms:
        raise ValueError(f"bad init npy {path}: {loaded.shape}")
    if args.replicas > loaded.shape[0]:
        raise ValueError(f"--replicas={args.replicas} > B={loaded.shape[0]}")
    loaded = loaded[: args.replicas]
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )
    print(f"[init] MACE loaded  n_gpus={getattr(mace_model, 'n_gpus', 1)}", flush=True)

    if cv_mode == "s12":
        collective_variable = SoftDistanceS12(
            pairs, d_ref=float(args.d_ref), n=int(args.soft_n), m=int(args.soft_m)
        )
        print(
            f"[CV] SoftDistanceS12  s1=c1+c2+c3  s2=c2-c3  "
            f"d_ref={args.d_ref} n={args.soft_n} m={args.soft_m}",
            flush=True,
        )
    else:
        collective_variable = SoftDistance(
            pairs, d_ref=float(args.d_ref), n=int(args.soft_n), m=int(args.soft_m)
        )
        print(
            f"[CV] SoftDistance 3D  cv=(c1,c2,c3)  "
            f"d_ref={args.d_ref} n={args.soft_n} m={args.soft_m}",
            flush=True,
        )

    collective0 = collective_variable(coordinate)
    bandwidth0 = initial_bandwidth_from_collective_variable(collective0)
    cv0 = collective0.detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    print(
        f"[init] bw_init={bandwidth0:.6g}  replicas={args.replicas}  {_cv_stats(cv0, cv_mode)}",
        flush=True,
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
    fpt_bias = FPTBias(
        collective_variable,
        kernel,
        kernel_grad_beta=float(args.lambda_init) * float(args.beta0),
        device=device,
        dtype=dtype,
        collective_variable_grad_clip=cv_clip,
        update_pace=args.update_pace,
        n_atoms_per_system=num_atoms,
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
    model = SumModel(mace_model, fpt_bias, upper_wall, lower_wall)
    print(
        f"[init] SumModel(MACE, FPTBias, UpperWall, NonbondLower)  "
        f"λ_init={args.lambda_init}  cv_clip={cv_clip}  "
        f"lower k={k_ev:.4f} eV/Å² r_min={args.nonbond_r_min}",
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

    out = Path(
        args.output
        or (
            EXPERIMENT_DIR
            / "results"
            / f"da_mace_{cv_mode}_{args.mace_model}_{args.replicas}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    def _stage_tag(stage_i: int, lam: float, *, phase: str) -> str:
        prefix = "heal" if phase == "heal" else "stage"
        return f"{prefix}{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i: int, lam: float, st, phase: str = "anneal") -> None:
        coords = sim_state_to_batched_coordinates(st)
        coords_np = coords.detach().cpu().numpy()
        cv_np = collective_variable(coords).detach().cpu().numpy()
        if cv_np.ndim == 3:
            cv_np = cv_np.reshape(cv_np.shape[0], -1)
        tag = _stage_tag(stage_i, lam, phase=phase)
        np.save(
            out / f"{tag}_bandwidth.npy",
            integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        metrics = integrator._last_step_metrics
        metrics.update(integrator.kinetic_temperature_stats(st))
        print(
            f"[stage-end/{phase}] {tag}  "
            f"T_kin mean={metrics.get('kinetic_temperature_mean', float('nan')):.2f} K  "
            f"{_cv_stats(cv_np, cv_mode)}",
            flush=True,
        )
        save_stage_artifacts(
            out,
            tag,
            coords_np,
            cv_np,
            symbols,
            title=f"DA-MACE {cv_mode}  {phase} {tag}",
            cv_mode=cv_mode,
            max_xyz_frames=args.max_xyz_frames,
        )

    save_stage_artifacts(
        out,
        "stage_init",
        coordinate.detach().cpu().numpy(),
        cv0,
        symbols,
        title=f"DA-MACE {cv_mode}  init",
        cv_mode=cv_mode,
        max_xyz_frames=args.max_xyz_frames,
    )

    if args.heal_lambda_list:
        fpt_bias.set_kernel_grad_beta(
            float(args.heal_lambda_list[0]) * float(args.beta0)
        )
        print(
            f"[init] heal schedule λ={args.heal_lambda_list}  "
            f"heal_steps={args.heal_steps}  then anneal λ_init={args.lambda_init}",
            flush=True,
        )

    final_state = integrator.run(
        state0,
        args.stage_steps,
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_stage_end=on_stage_end,
        heal_lambdas=args.heal_lambda_list or None,
        heal_steps=int(args.heal_steps) if args.heal_lambda_list else None,
    )
    final_coords = sim_state_to_batched_coordinates(final_state)
    final_cv = collective_variable(final_coords).detach().cpu().numpy()
    if final_cv.ndim == 3:
        final_cv = final_cv.reshape(final_cv.shape[0], -1)
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
        title=f"DA-MACE {cv_mode}  final λ={integrator.lambda_current:.4g}",
        cv_mode=cv_mode,
        max_xyz_frames=args.max_xyz_frames,
    )
    print(f"done  out={out}  lambda={integrator.lambda_current}", flush=True)


if __name__ == "__main__":
    main()
