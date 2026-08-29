#!/usr/bin/env python3
"""Allyl phenyl ether (Claisen) + MACE + 2D distance CV + FPT.

CVs (1-based, Angstrom)::

    cv1 = d1 = d(O7, C8)   breaking
    cv2 = d2 = d(C1, C10)  forming

Pre-MD from exact xyz (no noise) → FPT init.
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
    masses_from_symbols,
    save_stage_artifacts,
)
from fp_transformer_torch import (  # noqa: E402
    FPTBias,
    GaussianKernel,
    PerReplicaRescaleConfig,
    QuadraticDistanceWall,
    SoftminDistanceUpperWall,
    MultiParticlePairDistance,
    TorchSimFokkerPlanckIntegrator,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.mace import (  # noqa: E402
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices  # noqa: E402

DEFAULT_XYZ = EXPERIMENT_DIR / "allyl_phenyl_ether.xyz"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
DEFAULT_MASS = EXPERIMENT_DIR / "mass.txt"
KB_EV_PER_K = 8.617330337217213e-5


def initial_bandwidth_from_collective_variable(cv: torch.Tensor) -> float:
    distance = pairwise_distance(cv)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    return median if np.isfinite(median) and median > 0 else 0.5


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


def load_masses(path: Path, num_atoms: int, symbols: list[str]) -> np.ndarray:
    if path.is_file():
        vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if int(vals[0]) == num_atoms and vals.size == num_atoms + 1:
            vals = vals[1:]
        return vals[:num_atoms].astype(np.float64)
    return masses_from_symbols(symbols)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--xyz", type=str, default=str(DEFAULT_XYZ))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=1024)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=0.5)
    p.add_argument("--ess-threshold", type=float, default=0.97)
    p.add_argument("--min-lambda-step", type=float, default=0.001)
    p.add_argument("--max-lambda-step", type=float, default=0.002)
    p.add_argument("--num-lambda-trials", type=int, default=10)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--pre-md-steps", type=int, default=200)
    p.add_argument("--pre-md-dump-every", type=int, default=50)
    p.add_argument("--max-xyz-frames", type=int, default=None)
    p.add_argument("--heal-lambdas", type=str, default="0.1,0.05,0.02")
    p.add_argument("--heal-steps", type=int, default=100)
    p.add_argument("--cell-length", type=float, default=40.0)
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--softmin-a", type=float, default=10.0, help="softmin a in 1/Å")
    p.add_argument("--softmin-d-wall", type=float, default=2.7, help="softmin wall in Å")
    p.add_argument("--softmin-k", type=float, default=10.0, help="softmin wall k in eV/Å²")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=20)
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


def _resolve(args):
    if args.steps is not None:
        args.stage_steps = args.steps
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    for src, dst in [
        ("rew_lambda_init", "lambda_init"),
        ("rew_lambda_max", "lambda_max"),
        ("rew_ess_threshold", "ess_threshold"),
        ("rew_min_lambda_step", "min_lambda_step"),
        ("rew_max_lambda_step", "max_lambda_step"),
        ("rew_num_lambda_trials", "num_lambda_trials"),
    ]:
        v = getattr(args, src)
        if v is not None:
            setattr(args, dst, v)
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    temp = args.temperature
    if args.rew_anneal_ref_temperature is not None:
        temp = args.rew_anneal_ref_temperature
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    heal = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    args.heal_lambda_list = heal
    return args


def main() -> None:
    args = _resolve(build_parser().parse_args())
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

    import torch_sim as ts
    from torch_sim.models.interface import SumModel

    print(
        f"Claisen FPT  device={device}  mace={args.mace_model}  "
        f"n_gpus={len(mace_devices)}  λ_init={args.lambda_init}",
        flush=True,
    )

    symbols, xyz_coords = load_xyz(Path(args.xyz))
    atomic_numbers = atomic_numbers_from_symbols(symbols)
    num_atoms = int(atomic_numbers.shape[0])
    masses = load_masses(Path(args.mass_file), num_atoms, symbols)
    pairs = load_pair_indices(Path(args.cv_file))
    print(
        f"atoms={num_atoms}  pairs(0-based)={pairs}  "
        f"(1-based O7–C8, C1–C10)",
        flush=True,
    )

    # write mass if missing
    if not Path(args.mass_file).is_file():
        with open(args.mass_file, "w") as fh:
            fh.write(f"{num_atoms}\n")
            for m in masses:
                fh.write(f"{m:.6f}\n")

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
    )

    cv_fn = MultiParticlePairDistance(pairs)
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    if len(pairs) < 2:
        raise ValueError(f"softmin restraint needs ≥2 CV pairs, got {pairs}")
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
    print(
        f"[restrain] softmin upper wall  a={args.softmin_a}/Å  "
        f"d_wall={args.softmin_d_wall} Å  k={args.softmin_k} eV/Å²  "
        f"pairs={pairs[0]},{pairs[1]}",
        flush=True,
    )

    out = Path(
        args.output
        or (EXPERIMENT_DIR / "results" / f"claisen_fpt_{args.mace_model}_{args.replicas}")
    )
    out.mkdir(parents=True, exist_ok=True)

    mass_t = torch.as_tensor(masses, dtype=dtype, device=device)
    z_t = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
    cell = torch.eye(3, device=device, dtype=dtype) * float(args.cell_length)

    if args.init_npy:
        loaded = np.load(Path(args.init_npy))[: args.replicas]
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[init] init-npy B={coordinate.shape[0]}", flush=True)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[pre-md] exact xyz × {args.replicas}, steps={args.pre_md_steps}", flush=True)
        md_model = SumModel(mace_model, wall, softmin_wall)
        state_md = batched_coordinates_to_sim_state(
            coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell,
            pbc=False, charge=float(args.charge), spin=float(args.spin),
        )
        done = 0
        dump_every = max(1, int(args.pre_md_dump_every))
        md_steps = max(0, int(args.pre_md_steps))
        dumps = []
        while done < md_steps:
            n_this = min(dump_every, md_steps - done)
            state_md = ts.integrate(
                state_md, md_model, integrator=ts.Integrator.nvt_langevin,
                n_steps=n_this, temperature=float(args.temperature),
                timestep=float(args.timestep), gamma=float(args.langevin_friction),
                autobatcher=False, pbar=False,
            )
            done += n_this
            dumps.append(sim_state_to_batched_coordinates(state_md).detach().cpu().numpy())
            print(f"[pre-md] step={done}", flush=True)
        if dumps:
            np.save(out / "pre_md_traj.npy", np.stack(dumps, axis=0))
            coordinate = torch.as_tensor(dumps[-1], dtype=dtype, device=device)
        np.save(out / "init_coords_from_pre_md.npy", coordinate.detach().cpu().numpy())

    collective0 = cv_fn(coordinate)
    bw0 = initial_bandwidth_from_collective_variable(collective0)
    cv0 = collective0.detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    print(
        f"[init] bw={bw0:.4g}  d1 med={np.median(cv0[:,0]):.3f} Å  "
        f"d2 med={np.median(cv0[:,1]):.3f} Å",
        flush=True,
    )

    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
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
        args.replicas, bandwidth=bw0, distance_function=pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale, device=device, dtype=dtype,
    )
    cv_clip = None if args.cv_grad_clip_norm <= 0 else float(args.cv_grad_clip_norm)
    fpt_bias = FPTBias(
        cv_fn, kernel,
        kernel_grad_beta=float(args.lambda_init) * float(args.beta0),
        device=device, dtype=dtype,
        collective_variable_grad_clip=cv_clip,
        update_pace=args.update_pace, n_atoms_per_system=num_atoms,
    )
    model = SumModel(mace_model, fpt_bias, wall, softmin_wall)

    state0 = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell,
        pbc=False, charge=float(args.charge), spin=float(args.spin),
    )
    integrator = TorchSimFokkerPlanckIntegrator(
        model, fpt_bias, cv_fn, kernel,
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
        device=device, dtype=dtype,
    )

    def _tag(i, lam, phase):
        pref = "heal" if phase == "heal" else "stage"
        return f"{pref}{i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i, lam, st, phase="anneal"):
        coords = sim_state_to_batched_coordinates(st)
        cv = cv_fn(coords).detach().cpu().numpy()
        if cv.ndim == 3:
            cv = cv.reshape(cv.shape[0], -1)
        tag = _tag(stage_i, lam, phase)
        np.save(out / f"{tag}_bandwidth.npy", integrator.gaussian_kernel.bandwidth.detach().cpu().numpy())
        print(
            f"[stage/{phase}] {tag}  d1 med={np.median(cv[:,0]):.3f}  d2 med={np.median(cv[:,1]):.3f}",
            flush=True,
        )
        save_stage_artifacts(
            out, tag, coords.detach().cpu().numpy(), cv, symbols,
            title=f"Claisen-FPT {phase} {tag}",
            max_xyz_frames=args.max_xyz_frames,
        )

    save_stage_artifacts(
        out, "stage_init", coordinate.detach().cpu().numpy(), cv0, symbols,
        title="Claisen-FPT init", max_xyz_frames=args.max_xyz_frames,
    )
    if args.heal_lambda_list:
        fpt_bias.set_kernel_grad_beta(float(args.heal_lambda_list[0]) * float(args.beta0))
        print(f"[init] heal λ={args.heal_lambda_list}", flush=True)

    final = integrator.run(
        state0, args.stage_steps, verbose=args.verbose, verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress, on_stage_end=on_stage_end,
        heal_lambdas=args.heal_lambda_list or None,
        heal_steps=int(args.heal_steps) if args.heal_lambda_list else None,
    )
    fc = sim_state_to_batched_coordinates(final)
    fcv = cv_fn(fc).detach().cpu().numpy()
    if fcv.ndim == 3:
        fcv = fcv.reshape(fcv.shape[0], -1)
    save_stage_artifacts(
        out, "stage_final", fc.detach().cpu().numpy(), fcv, symbols,
        title=f"Claisen-FPT final λ={integrator.lambda_current:.4g}",
        max_xyz_frames=args.max_xyz_frames,
    )
    print(f"done out={out} λ={integrator.lambda_current}", flush=True)


if __name__ == "__main__":
    main()
