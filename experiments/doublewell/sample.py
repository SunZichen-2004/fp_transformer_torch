#!/usr/bin/env python3
"""2D double-well + CV=x1 + BAOAB-FP λ annealing (sFPT, no resampling).

Reduced units (same convention as ``particle_dimer``):
  - ``k_B = 1``, ``T = 1`` → ``β0 = 1``
  - ``m = 1``
  - potential ``V = h(x1²−1)² + ½ k x2²`` with default ``h=5``, ``k=10``
    (barrier = 5 k_B T)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fp_transformer_torch import (  # noqa: E402
    BaoabFokkerPlanckIntegrator,
    CartesianComponent,
    DoubleWell2D,
    DoubleWellParams,
    EnergySystem,
    GaussianKernel,
    PerReplicaRescaleConfig,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicas", type=int, default=512)
    p.add_argument("--stage-steps", type=int, default=200)
    p.add_argument("--steps", type=int, default=None, help="Alias for --stage-steps")
    p.add_argument("--seed", type=int, default=0)
    # Integrator (reduced units; stiffer than particle-dimer wells → slightly larger dt ok)
    p.add_argument("--timestep", type=float, default=1e-3)
    p.add_argument("--baoab-dt", type=float, default=None, help="Alias for --timestep")
    p.add_argument("--langevin-friction", type=float, default=1.0)
    p.add_argument(
        "--baoab-langevin-gamma",
        type=float,
        default=None,
        help="Alias for --langevin-friction",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kB", type=float, default=1.0)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--mass", type=float, default=1.0)
    # Potential
    p.add_argument("--barrier-h", type=float, default=5.0)
    p.add_argument("--fast-k", type=float, default=10.0)
    # λ annealing
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--ess-threshold", type=float, default=0.95)
    p.add_argument("--min-lambda-step", type=float, default=0.01)
    p.add_argument("--max-lambda-step", type=float, default=0.03)
    p.add_argument("--num-lambda-trials", type=int, default=20)
    p.add_argument("--update-pace", type=int, default=10)
    # Init near left well A=(-1,0)
    p.add_argument("--init-well", type=str, default="left", choices=("left", "right", "both"))
    p.add_argument("--noise-scale", type=float, default=0.05)
    p.add_argument("--init-coords", type=str, default=None)
    p.add_argument(
        "--pre-md-steps",
        type=int,
        default=0,
        help="Unbiased Langevin steps before sFPT anneal (kernel force off)",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=50)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--cv-grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip CV-space kernel gradient by global norm. Set <=0 to disable.",
    )
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-target-mode", type=str, default="adaptive_quantile")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    # CLI aliases matching particle_dimer / JAX scripts
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
        args.temperature = float(temp)
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    return args


def _init_coordinates(
    *,
    replicas: int,
    init_well: str,
    noise_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Shape ``(R, 1, 2)`` near well(s)."""
    if init_well == "left":
        centers = [(-1.0, 0.0)]
    elif init_well == "right":
        centers = [(1.0, 0.0)]
    else:
        centers = [(-1.0, 0.0), (1.0, 0.0)]
    coords = []
    for i in range(replicas):
        cx, cy = centers[i % len(centers)]
        coords.append([cx, cy])
    base = torch.as_tensor(coords, dtype=dtype, device=device).reshape(replicas, 1, 2)
    if noise_scale > 0:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        noise = torch.randn(base.shape, dtype=dtype, generator=gen).to(device)
        base = base + float(noise_scale) * noise
    return base


def plot_x1_x2(xy: np.ndarray, path: Path, *, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.55, c="#1f77b4", edgecolors="none")
    ax.axvline(0.0, color="0.6", ls=":", lw=0.8)
    ax.plot([-1.0, 1.0], [0.0, 0.0], "k*", ms=8, zorder=5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    print(
        f"device={device}  kB={args.kB}  T={args.temperature}  β0={args.beta0:.6g}  "
        f"dt={args.timestep}  γ={args.langevin_friction}  m={args.mass}  "
        f"h={args.barrier_h}  k_fast={args.fast_k}  "
        f"λ0={args.lambda_init}→{args.lambda_max}  R={args.replicas}  "
        f"stage_steps={args.stage_steps}",
        flush=True,
    )

    potential = DoubleWell2D(
        DoubleWellParams(h=float(args.barrier_h), k=float(args.fast_k))
    )

    if args.init_coords:
        path = Path(os.path.abspath(os.path.expanduser(args.init_coords)))
        loaded = np.load(path)
        if loaded.ndim == 2 and loaded.shape[-1] == 2:
            loaded = loaded.reshape(loaded.shape[0], 1, 2)
        if loaded.ndim != 3 or loaded.shape[1] != 1 or loaded.shape[2] != 2:
            raise ValueError(f"bad init coords {path}: {loaded.shape}")
        if loaded.shape[0] != args.replicas:
            print(
                f"[init] --replicas={args.replicas} overridden by loaded B={loaded.shape[0]}",
                flush=True,
            )
            args.replicas = int(loaded.shape[0])
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[init] loaded {path} shape={tuple(coordinate.shape)}", flush=True)
    else:
        coordinate = _init_coordinates(
            replicas=int(args.replicas),
            init_well=str(args.init_well),
            noise_scale=float(args.noise_scale),
            seed=int(args.seed),
            device=device,
            dtype=dtype,
        )
        print(
            f"[init] well={args.init_well}  noise={args.noise_scale}  "
            f"shape={tuple(coordinate.shape)}",
            flush=True,
        )

    energy_system = EnergySystem(potential, coordinate)
    collective_variable = CartesianComponent(particle_index=0, dim_index=0)  # x1
    collective0 = collective_variable(coordinate)
    flat = collective0.reshape(args.replicas, -1)
    diffs = flat[:, None, :] - flat[None, :, :]
    dist2 = torch.sum(diffs * diffs, dim=-1)
    triu = torch.triu_indices(args.replicas, args.replicas, offset=1)
    median = torch.median(dist2[triu[0], triu[1]]).clamp(min=1e-8).sqrt()
    print(f"[kernel] median CV bandwidth seed={float(median):.4g}", flush=True)

    # Same bandwidth rescale as dialanine_water: unnormalized kernel row-sum,
    # clipped to fixed_row_sum_[min,max] (legacy JAX path).
    rescale = PerReplicaRescaleConfig(
        enabled=bool(args.rescale_per_replica),
        target_mode=str(args.rescale_target_mode),
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
        f"[kernel] rescale mode={args.rescale_target_mode}  "
        f"fixed_row_sum=[{args.rescale_fixed_row_sum_min:g},{args.rescale_fixed_row_sum_max:g}]  "
        f"unnormalized row-sum",
        flush=True,
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

    cv_clip = (
        None
        if args.cv_grad_clip_norm is None or args.cv_grad_clip_norm <= 0.0
        else float(args.cv_grad_clip_norm)
    )
    print(f"[init] cv_grad_clip_norm={cv_clip}", flush=True)
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
        mass=float(args.mass),
        boltzmann_constant=float(args.kB),
        update_pace=args.update_pace,
        collective_variable_grad_clip=cv_clip,
        device=device,
        dtype=dtype,
    )

    out = args.output or (
        Path(__file__).resolve().parent
        / "results"
        / f"doublewell_r{args.replicas}_s{args.seed}"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "x1_x2"
    plots_dir.mkdir(parents=True, exist_ok=True)

    xy0 = coordinate.detach().cpu().numpy().reshape(-1, 2)
    plot_x1_x2(xy0, plots_dir / "stage_init.png", title="double-well init")

    def _stage_tag(stage_i: int, lam: float, *, phase: str = "anneal") -> str:
        if phase == "pre_md":
            pref = "pre_md"
        elif phase == "heal":
            pref = "heal"
        else:
            pref = "stage"
        return f"{pref}{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i: int, lam: float, st, phase: str = "anneal") -> None:
        xy = st.coordinate.detach().cpu().numpy().reshape(-1, 2)
        cv = collective_variable(st.coordinate).detach().cpu().numpy().reshape(-1)
        tag = _stage_tag(stage_i, lam, phase=phase)
        np.save(out / f"{tag}_coords.npy", st.coordinate.detach().cpu().numpy())
        np.save(out / f"{tag}_x1.npy", cv)
        np.save(
            out / f"{tag}_bandwidth.npy",
            integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        plot_x1_x2(
            xy,
            plots_dir / f"{tag}.png",
            title=rf"double-well  {tag}  $\lambda$={lam:g}",
        )
        metrics = integrator._last_step_metrics
        print(
            f"[{phase}-end] {tag}  "
            f"x1[min/med/max]={cv.min():.3f}/{np.median(cv):.3f}/{cv.max():.3f}  "
            f"T_kin={metrics.get('kinetic_temperature_mean', float('nan')):.3g}  "
            f"bw[med]={metrics.get('bandwidth_median', float('nan')):.4g}",
            flush=True,
        )

    state = integrator.init_state(seed=int(args.seed))

    pre_md_steps = max(0, int(args.pre_md_steps))
    if pre_md_steps > 0:
        print(
            f"[pre-md] {pre_md_steps} unbiased Langevin steps "
            f"(use_kernel_gradient=False)",
            flush=True,
        )
        saved_use_kg = bool(integrator.use_kernel_gradient)
        saved_pace = int(integrator.update_pace)
        integrator.use_kernel_gradient = False
        # Skip bandwidth rescale while kernel force is off.
        integrator.update_pace = pre_md_steps + 1
        state = integrator.run_stage(
            pre_md_steps,
            initial_state=state,
            verbose=bool(args.verbose),
            verbose_pace=int(args.verbose_pace),
            show_progress=not bool(args.no_progress),
        )
        integrator.use_kernel_gradient = saved_use_kg
        integrator.update_pace = saved_pace
        integrator.sync_lambda_to_kernel_grad_beta()
        np.save(
            out / "init_coords_from_pre_md.npy",
            state.coordinate.detach().cpu().numpy(),
        )
        on_stage_end(0, 0.0, state, phase="pre_md")
        print("[pre-md] done → FPT anneal", flush=True)

    state = integrator.run(
        int(args.stage_steps),
        initial_state=state,
        seed=int(args.seed),
        verbose=bool(args.verbose),
        verbose_pace=int(args.verbose_pace),
        show_progress=not bool(args.no_progress),
        on_stage_end=on_stage_end,
    )
    np.save(out / "final_coordinate.npy", state.coordinate.detach().cpu().numpy())
    np.save(
        out / "final_bandwidth.npy",
        integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
    )
    final_cv = collective_variable(state.coordinate).detach().cpu().numpy().reshape(-1)
    np.save(out / "final_x1.npy", final_cv)
    plot_x1_x2(
        state.coordinate.detach().cpu().numpy().reshape(-1, 2),
        plots_dir / "final.png",
        title=rf"final  $\lambda$={integrator.lambda_current:g}",
    )
    print(
        f"wrote {out}  lambda={integrator.lambda_current:.6g}  "
        f"x1[min/med/max]={final_cv.min():.3f}/{np.median(final_cv):.3f}/{final_cv.max():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
