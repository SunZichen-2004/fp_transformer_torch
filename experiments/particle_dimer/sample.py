#!/usr/bin/env python3
"""Particle-dimer + pair-distance CV + BAOAB-FP λ annealing (no resampling).

Reduced units: ``k_B = 1``, default ``T = 1`` → ``β0 = 1``. Kernel force uses
``F(1+λ)/λ`` (``kernel_grad_beta = λ/(1+λ) · β0``).
"""

from __future__ import annotations

import argparse
import csv
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
    EnergySystem,
    GaussianKernel,
    ParticleDimer,
    ParticlePairDistance,
    PerReplicaRescaleConfig,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicas", type=int, default=128)
    p.add_argument("--stage-steps", type=int, default=100)
    p.add_argument("--steps", type=int, default=None, help="Alias for --stage-steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=1e-4)
    p.add_argument("--baoab-dt", type=float, default=None, help="Alias for --timestep")
    p.add_argument("--langevin-friction", type=float, default=10.0)
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
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--ess-threshold", type=float, default=0.96)
    p.add_argument("--min-lambda-step", type=float, default=1e-12)
    p.add_argument("--max-lambda-step", type=float, default=1e-12)
    p.add_argument("--num-lambda-trials", type=int, default=20)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--noise-scale", type=float, default=0.05)
    p.add_argument("--init-coords", type=str, default=None)
    p.add_argument("--dimer-init", type=str, default="extended")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=50)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--heal-lambdas", type=str, default="0.5,0.1")
    p.add_argument("--heal-steps", type=int, default=200)
    p.add_argument(
        "--pre-md-steps",
        type=int,
        default=10000,
        help="Max unbiased Langevin steps before heal/anneal (kernel force off).",
    )
    p.add_argument(
        "--pre-md-chunk",
        type=int,
        default=1000,
        help="Check T_kin every this many pre-MD steps.",
    )
    p.add_argument(
        "--pre-md-tol",
        type=float,
        default=0.15,
        help="Stop pre-MD early when |T_kin/T - 1| < this.",
    )
    p.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Stop MD (including heal) after this many seconds.",
    )
    p.add_argument("--max-stages", type=int, default=None)
    p.add_argument(
        "--kernel-grad-beta-mode",
        type=str,
        default="lambda_over_one_plus_lambda",
        choices=(
            "lambda",
            "lambda_over_one_plus_lambda",
            "f_over_lambda",
            "f_one_plus_lambda_over_lambda",
        ),
    )
    p.add_argument(
        "--cv-grad-clip-norm",
        type=float,
        default=2.0,
        help="Clip CV-space kernel gradient by global norm. Set <=0 to disable.",
    )
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=15.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=25.0)
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
    heal = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    args.heal_lambda_list = heal
    return args


def plot_distance_hist(d: np.ndarray, path: Path, *, title: str, t_target: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.hist(d, bins=40, range=(0.5, 2.8), color="#1f77b4", alpha=0.75, edgecolor="none")
    ax.axvline(0.97, color="0.35", ls="--", lw=0.8)
    ax.axvline(2.03, color="0.35", ls="--", lw=0.8)
    ax.set_xlabel(r"dimer distance $d$")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.5, 2.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    print(
        f"device={device}  threads={n_threads}  kB={args.kB}  T={args.temperature}  "
        f"β0={args.beta0:.6g}  dt={args.timestep}  γ={args.langevin_friction}  "
        f"λ0={args.lambda_init}→{args.lambda_max}  R={args.replicas}  "
        f"stage_steps={args.stage_steps}  heal={args.heal_lambda_list}  "
        f"kgb_mode={args.kernel_grad_beta_mode}",
        flush=True,
    )

    potential = ParticleDimer()
    n_atoms = potential.num_particles

    if args.init_coords:
        path = Path(os.path.abspath(os.path.expanduser(args.init_coords)))
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[1] != n_atoms or loaded.shape[2] != 2:
            raise ValueError(f"bad init coords {path}: {loaded.shape}")
        if loaded.shape[0] != args.replicas:
            if int(loaded.shape[0]) < int(args.replicas):
                raise ValueError(
                    f"init B={loaded.shape[0]} < --replicas={args.replicas}"
                )
            idx = np.linspace(0, loaded.shape[0] - 1, int(args.replicas)).astype(
                np.int64
            )
            print(
                f"[init] subsampled {path} {tuple(loaded.shape)} → "
                f"R={args.replicas} (linspace)",
                flush=True,
            )
            loaded = loaded[idx]
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[init] loaded {path} shape={tuple(coordinate.shape)}", flush=True)
    else:
        d_comp, d_ext = 0.97, 2.03
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
        print(
            f"[init] dimer={args.dimer_init} d0={d0:.3f} noise={args.noise_scale} "
            f"shape={tuple(coordinate.shape)}",
            flush=True,
        )

    energy_system = EnergySystem(potential, coordinate)
    collective_variable = ParticlePairDistance([0, 1])
    collective0 = collective_variable(coordinate)
    flat = collective0.reshape(args.replicas, -1)
    diffs = flat[:, None, :] - flat[None, :, :]
    dist2 = torch.sum(diffs * diffs, dim=-1)
    triu = torch.triu_indices(args.replicas, args.replicas, offset=1)
    median = torch.median(dist2[triu[0], triu[1]]).clamp(min=1e-8).sqrt()
    print(f"[kernel] median CV bandwidth seed={float(median):.4g}", flush=True)

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
        f"[kernel] rescale adaptive_quantile  "
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
        kernel_grad_beta_mode=str(args.kernel_grad_beta_mode),
        device=device,
        dtype=dtype,
    )
    print(
        f"[init] cv_grad_clip_norm={cv_clip}  "
        f"kernel_grad_beta={integrator.kernel_grad_beta:.6g} "
        f"(mode={integrator.kernel_grad_beta_mode})",
        flush=True,
    )

    out = args.output or (
        Path(__file__).resolve().parent
        / "results"
        / f"particle_dimer_r{args.replicas}_s{args.seed}"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "d_hist"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "stage_metrics.csv"
    metrics_rows: list[dict[str, object]] = []

    def _stage_tag(stage_i: int, lam: float, *, phase: str = "anneal") -> str:
        if phase == "pre_md":
            pref = "pre_md"
        elif phase == "heal":
            pref = "heal"
        else:
            pref = "stage"
        return f"{pref}{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i: int, lam: float, st, phase: str = "anneal") -> None:
        d = collective_variable(st.coordinate).detach().cpu().numpy().reshape(-1)
        tag = _stage_tag(stage_i, lam, phase=phase)
        np.save(out / f"{tag}_coords.npy", st.coordinate.detach().cpu().numpy())
        np.save(out / f"{tag}_d.npy", d)
        np.save(
            out / f"{tag}_bandwidth.npy",
            integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        plot_distance_hist(
            d,
            plots_dir / f"{tag}.png",
            title=rf"particle-dimer  {tag}  $\lambda$={lam:g}",
            t_target=float(args.temperature),
        )
        metrics = dict(integrator._last_step_metrics)
        t_kin = float(metrics.get("kinetic_temperature_mean", float("nan")))
        t_std = float(metrics.get("kinetic_temperature_std", float("nan")))
        row = {
            "phase": phase,
            "stage": int(stage_i),
            "lambda": float(lam),
            "kernel_grad_beta": float(integrator.kernel_grad_beta),
            "T_kin_mean": t_kin,
            "T_kin_std": t_std,
            "T_target": float(args.temperature),
            "d_min": float(d.min()),
            "d_med": float(np.median(d)),
            "d_max": float(d.max()),
            "n_compact": int(np.sum(d < 1.5)),
            "n_extended": int(np.sum(d >= 1.5)),
            "bw_median": float(metrics.get("bandwidth_median", float("nan"))),
        }
        metrics_rows.append(row)
        with metrics_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(
            f"[{phase}-end] {tag}  "
            f"d[min/med/max]={d.min():.3f}/{np.median(d):.3f}/{d.max():.3f}  "
            f"T_kin={t_kin:.3g}±{t_std:.2g} (target={args.temperature:g})  "
            f"n_comp/ext={row['n_compact']}/{row['n_extended']}",
            flush=True,
        )

    d0 = collective_variable(coordinate).detach().cpu().numpy().reshape(-1)
    plot_distance_hist(
        d0,
        plots_dir / "stage_init.png",
        title="particle-dimer init",
        t_target=float(args.temperature),
    )
    np.save(out / "stage_init_d.npy", d0)

    state = integrator.init_state(seed=int(args.seed))
    pre_md_steps = max(0, int(args.pre_md_steps))
    if pre_md_steps > 0:
        chunk = max(1, int(args.pre_md_chunk))
        tol = float(args.pre_md_tol)
        t_target = float(args.temperature)
        print(
            f"[pre-md] up to {pre_md_steps} unbiased Langevin steps "
            f"(use_kernel_gradient=False), chunk={chunk}, "
            f"stop when |T_kin/T-1|<{tol:g} (T={t_target:g})",
            flush=True,
        )
        saved_use_kg = bool(integrator.use_kernel_gradient)
        saved_pace = int(integrator.update_pace)
        integrator.use_kernel_gradient = False
        integrator.update_pace = chunk + 1
        done = 0
        chunk_i = 0
        while done < pre_md_steps:
            n_this = min(chunk, pre_md_steps - done)
            state = integrator.run_stage(
                n_this,
                initial_state=state,
                verbose=bool(args.verbose),
                verbose_pace=int(args.verbose_pace),
                show_progress=not bool(args.no_progress),
            )
            done += n_this
            t_kin = float(
                integrator._last_step_metrics.get(
                    "kinetic_temperature_mean", float("nan")
                )
            )
            rel = abs(t_kin / t_target - 1.0) if t_target > 0 else float("inf")
            print(
                f"[pre-md] steps={done}/{pre_md_steps}  "
                f"T_kin={t_kin:.4g}  target={t_target:g}  |ΔT|/T={rel:.3f}",
                flush=True,
            )
            on_stage_end(chunk_i, 0.0, state, phase="pre_md")
            chunk_i += 1
            if rel <= tol:
                print(
                    f"[pre-md] temperature within {tol:.0%}; stopping pre-MD",
                    flush=True,
                )
                break
        else:
            print(
                f"[pre-md] reached max {pre_md_steps} steps "
                f"(T_kin still {t_kin:.4g} vs {t_target:g})",
                flush=True,
            )
        integrator.use_kernel_gradient = saved_use_kg
        integrator.update_pace = saved_pace
        integrator.sync_lambda_to_kernel_grad_beta()
        np.save(
            out / "init_coords_from_pre_md.npy",
            state.coordinate.detach().cpu().numpy(),
        )
        print("[pre-md] done → heal / FPT anneal", flush=True)

    state = integrator.run(
        int(args.stage_steps),
        initial_state=state,
        seed=int(args.seed),
        verbose=bool(args.verbose),
        verbose_pace=int(args.verbose_pace),
        show_progress=not bool(args.no_progress),
        on_stage_end=on_stage_end,
        heal_lambdas=args.heal_lambda_list or None,
        heal_steps=int(args.heal_steps) if args.heal_lambda_list else None,
        max_stages=None if args.max_stages is None else int(args.max_stages),
        max_runtime_sec=(
            None if args.max_runtime_sec is None else float(args.max_runtime_sec)
        ),
    )
    np.save(out / "final_coordinate.npy", state.coordinate.detach().cpu().numpy())
    np.save(
        out / "final_bandwidth.npy",
        integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
    )
    final_d = collective_variable(state.coordinate).detach().cpu().numpy().reshape(-1)
    np.save(out / "final_d.npy", final_d)
    plot_distance_hist(
        final_d,
        plots_dir / "final.png",
        title=rf"final  $\lambda$={integrator.lambda_current:g}",
        t_target=float(args.temperature),
    )
    print(
        f"wrote {out}  lambda={integrator.lambda_current:.6g}  "
        f"d[min/med/max]={final_d.min():.3f}/{np.median(final_d):.3f}/{final_d.max():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
