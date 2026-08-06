#!/usr/bin/env python3
"""Particle-dimer + pair-distance CV + BAOAB-FP λ annealing (no resampling)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fp_transformer_torch import (  # noqa: E402
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaussianKernel,
    ParticleDimer,
    ParticlePairDistance,
    PerReplicaRescaleConfig,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicas", type=int, default=32)
    p.add_argument("--stage-steps", type=int, default=50)
    p.add_argument("--steps", type=int, default=None, help="Alias for --stage-steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=1e-4)
    p.add_argument("--baoab-dt", type=float, default=None, help="Alias for --timestep")
    p.add_argument("--langevin-friction", type=float, default=1.0)
    p.add_argument(
        "--baoab-langevin-gamma",
        type=float,
        default=None,
        help="Alias for --langevin-friction",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--kB", type=float, default=1.0)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--ess-threshold", type=float, default=0.95)
    p.add_argument("--min-lambda-step", type=float, default=0.05)
    p.add_argument("--max-lambda-step", type=float, default=0.5)
    p.add_argument("--num-lambda-trials", type=int, default=20)
    p.add_argument("--update-pace", type=int, default=10)
    p.add_argument("--noise-scale", type=float, default=0.02)
    p.add_argument("--init-coords", type=str, default=None)
    p.add_argument("--dimer-init", type=str, default="extended")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    # CLI aliases matching JAX scripts
    p.add_argument("--rew-lambda-init", type=float, default=None)
    p.add_argument("--rew-lambda-max", type=float, default=None)
    p.add_argument("--rew-ess-threshold", type=float, default=None)
    p.add_argument("--rew-min-lambda-step", type=float, default=None)
    p.add_argument("--rew-max-lambda-step", type=float, default=None)
    p.add_argument("--rew-num-lambda-trials", type=int, default=None)
    p.add_argument("--rew-anneal-ref-temperature", type=float, default=None)
    p.add_argument("--rew-debug-reweight", action="store_true")
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
        if temp is None:
            temp = 1.0
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    return args


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    print(f"device={device}  beta0={args.beta0}  stage_steps={args.stage_steps}")

    potential = ParticleDimer()
    n_atoms = potential.num_particles

    if args.init_coords:
        path = Path(os.path.abspath(os.path.expanduser(args.init_coords)))
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[1] != n_atoms or loaded.shape[2] != 2:
            raise ValueError(f"bad init coords {path}: {loaded.shape}")
        if loaded.shape[0] != args.replicas:
            print(
                f"[init] --replicas={args.replicas} overridden by loaded B={loaded.shape[0]}"
            )
            args.replicas = int(loaded.shape[0])
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[init] loaded {path} shape={tuple(coordinate.shape)}")
    else:
        # wells: compact / extended via energy scan on init lattice
        d_comp, d_ext = 0.97, 2.03  # approximate Noé wells; init_positions uses dmid default
        if args.dimer_init == "extended":
            d0 = d_ext
        elif args.dimer_init == "compact":
            d0 = d_comp
        else:
            d0 = float(potential.params.dimer_dmid)
        base = potential.init_positions(d0)
        coordinate = base.repeat(args.replicas, 1, 1).to(device=device, dtype=dtype)
        if args.noise_scale > 0:
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            noise = torch.randn(coordinate.shape, dtype=dtype, generator=generator).to(
                device
            )
            coordinate = coordinate + args.noise_scale * noise

    energy_system = EnergySystem(potential, coordinate)
    collective_variable = ParticlePairDistance([0, 1])
    collective0 = collective_variable(coordinate)
    flat = collective0.reshape(args.replicas, -1)
    diffs = flat[:, None, :] - flat[None, :, :]
    dist2 = torch.sum(diffs * diffs, dim=-1)
    triu = torch.triu_indices(args.replicas, args.replicas, offset=1)
    median = torch.median(dist2[triu[0], triu[1]]).clamp(min=1e-8).sqrt()

    rescale = PerReplicaRescaleConfig(
        enabled=bool(args.rescale_per_replica),
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=args.rescale_adaptive_ref_quantile,
        adaptive_band_low=args.rescale_adaptive_band_low,
        adaptive_band_high=args.rescale_adaptive_band_high,
        bandwidth_min=args.rescale_bw_min,
        bandwidth_max=args.rescale_bw_max,
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=float(median),
        distance_function=pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )

    integrator = BaoabFokkerPlanckIntegrator(
        energy_system,
        collective_variable,
        kernel,
        timestep=args.timestep,
        langevin_friction=args.langevin_friction,
        beta0=args.beta0,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
        ess_threshold=args.ess_threshold,
        min_lambda_step=args.min_lambda_step,
        max_lambda_step=args.max_lambda_step,
        num_lambda_trials=args.num_lambda_trials,
        mass=1.0,
        boltzmann_constant=float(args.kB),
        update_pace=args.update_pace,
        device=device,
        dtype=dtype,
    )
    state = integrator.run(
        args.stage_steps, seed=args.seed, verbose=args.verbose, verbose_pace=args.verbose_pace
    )
    out = args.output or (
        Path(__file__).resolve().parent / "results" / f"particle_dimer_{args.replicas}"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "final_coordinate.npy", state.coordinate.detach().cpu().numpy())
    np.save(
        out / "final_bandwidth.npy",
        integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
    )
    print(f"wrote {out / 'final_coordinate.npy'}  lambda={integrator.lambda_current}")


# late import for expanduser path
import os  # noqa: E402


if __name__ == "__main__":
    main()
