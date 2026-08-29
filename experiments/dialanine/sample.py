#!/usr/bin/env python3
"""Dialanine (GAFF/SPONGE) + torsion CV + torus kernel + BAOAB-FP annealing."""

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

from fp_transformer_torch import (  # noqa: E402
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaffPotential,
    GaussianKernel,
    PerReplicaRescaleConfig,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.sponge_io import (  # noqa: E402
    load_sponge_coordinates,
)
from fp_transformer_torch.units import (  # noqa: E402
    KB_KCAL_PER_MOL_K,
    friction_per_picosecond_to_akma,
    timestep_picosecond_to_akma,
)

DEFAULT_SPONGE = EXPERIMENT_DIR / "dialanine"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "backbone_cv.txt"
DEFAULT_INIT_NPY = EXPERIMENT_DIR / "init_coords_2048.npy"


def initial_bandwidth_from_collective_variable(
    collective_variable: torch.Tensor,
) -> float:
    """Median pairwise torus distance (same recipe as JAX ``init_bandwidth``)."""
    distance = torus_pairwise_distance(collective_variable)
    upper = torch.triu(
        torch.ones_like(distance, dtype=torch.bool), diagonal=1
    )
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    if not np.isfinite(median) or median <= 0.0:
        return 0.5
    return median


def load_torsion_indices(path: Path) -> np.ndarray:
    """Load torsion quadruples; file is 1-based (same as JAX ``sample_xponge``)."""
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


def plot_phi_psi(
    collective_variable: np.ndarray,
    save_path: Path,
    *,
    title: str,
    x_name: str = "phi",
    y_name: str = "psi",
) -> None:
    """Scatter φ–ψ (radians) for dialanine backbone torsions."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return

    cv = np.asarray(collective_variable, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], cv.shape[1])
    if cv.shape[1] < 2:
        print(f"[warn] need ≥2 CVs for phi-psi, got shape {cv.shape}", flush=True)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(cv[:, 0], cv[:, 1], c="C3", s=2.0, alpha=0.6, linewidths=0)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel(f"{x_name} (rad)")
    plt.ylabel(f"{y_name} (rad)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=str, default=str(DEFAULT_SPONGE))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--timestep",
        type=float,
        default=0.001,
        help="BAOAB dt in picoseconds (converted to AKMA internally)",
    )
    p.add_argument("--baoab-dt", type=float, default=None, help="Alias of --timestep (ps)")
    p.add_argument(
        "--langevin-friction",
        type=float,
        default=1.0,
        help="Langevin γ in 1/ps (converted to AKMA internally)",
    )
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_KCAL_PER_MOL_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=0.5)
    p.add_argument("--ess-threshold", type=float, default=0.9)
    p.add_argument("--min-lambda-step", type=float, default=0.05)
    p.add_argument("--max-lambda-step", type=float, default=0.2)
    p.add_argument("--num-lambda-trials", type=int, default=10)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help="Coordinate jitter (Å). Keep 0 when using a diverse --init-npy.",
    )
    p.add_argument(
        "--init-npy",
        type=str,
        default=str(DEFAULT_INIT_NPY) if DEFAULT_INIT_NPY.is_file() else None,
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--cv-grad-clip-norm",
        type=float,
        default=0.05,
        help="Clip CV-space kernel gradient by global norm (JAX-compatible). "
        "Needed for GAFF stability at small λ; set <=0 to disable.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile GAFF energy (CUDA; first steps pay compile cost).",
    )
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=1.0)
    p.add_argument(
        "--rescale-fixed-row-sum-min",
        type=float,
        default=2.0,
        help="Absolute clip on adaptive band for unnormalized row_sum metric",
    )
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument("--rew-lambda-init", type=float, default=None)
    p.add_argument("--rew-lambda-max", type=float, default=None)
    p.add_argument("--rew-ess-threshold", type=float, default=None)
    p.add_argument("--rew-min-lambda-step", type=float, default=None)
    p.add_argument("--rew-max-lambda-step", type=float, default=None)
    p.add_argument("--rew-num-lambda-trials", type=int, default=None)
    p.add_argument("--rew-anneal-ref-temperature", type=float, default=None)
    p.add_argument("--rew-debug-reweight", action="store_true")
    p.add_argument(
        "--pre-md-steps",
        type=int,
        default=0,
        help="Unbiased Langevin steps before FP (kernel force off). 0 disables.",
    )
    p.add_argument(
        "--heal-lambdas",
        type=str,
        default=None,
        help="Comma-separated λ heal schedule before anneal, e.g. 0.1,0.05",
    )
    p.add_argument(
        "--heal-steps",
        type=int,
        default=500,
        help="Steps per heal λ (no ESS reweight between heal values)",
    )
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


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    prefix = Path(args.sponge_prefix)

    dt_ps = float(args.timestep)
    gamma_ps = float(args.langevin_friction)
    dt_akma = timestep_picosecond_to_akma(dt_ps)
    gamma_akma = friction_per_picosecond_to_akma(gamma_ps, dt_ps, dt_akma)

    print(f"device={device}  beta0={args.beta0:.6g}  sponge={prefix}")
    print(
        f"[baoab time] dt={dt_ps} ps → dt_akma={dt_akma:.12g}; "
        f"γ={gamma_ps} /ps → γ_akma={gamma_akma:.12g}; "
        f"γ·dt={gamma_ps * dt_ps:.6g}"
    )

    potential = GaffPotential(str(prefix))
    print(f"GAFF num_atoms={potential.num_atoms}")
    if args.compile and device.type == "cuda":
        try:
            potential.energy_calculator = torch.compile(
                potential.energy_calculator, fullgraph=False, dynamic=False
            )
            print("[init] torch.compile enabled for GAFF energy")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] torch.compile failed: {exc}")
    torsion_indices = load_torsion_indices(Path(args.cv_file))
    torsion_indices = torsion_indices[torsion_indices.max(axis=1) < potential.num_atoms]
    if torsion_indices.size == 0:
        raise ValueError("no torsion indices valid for this GAFF system")

    if args.init_npy:
        path = Path(os.path.abspath(os.path.expanduser(args.init_npy)))
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"bad init npy {path}: {loaded.shape}")
        if loaded.shape[1] != potential.num_atoms:
            raise ValueError(
                f"init npy n_atoms={loaded.shape[1]} != GAFF n_atoms={potential.num_atoms}"
            )
        if loaded.shape[0] != args.replicas:
            print(
                f"[init] --replicas={args.replicas} overridden by loaded B={loaded.shape[0]}"
            )
            args.replicas = int(loaded.shape[0])
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    else:
        base = load_sponge_coordinates(prefix, num_atoms=potential.num_atoms)
        coordinate = (
            torch.as_tensor(base, dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(args.replicas, 1, 1)
        )
        if args.noise_scale > 0:
            print(
                f"[warn] applying coordinate noise_scale={args.noise_scale} Å "
                "(bonded systems should use 0)"
            )
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            noise = torch.randn(coordinate.shape, dtype=dtype, generator=generator).to(
                device
            )
            coordinate = coordinate + args.noise_scale * noise

    energy_system = EnergySystem(potential, coordinate)
    collective_variable = Torsion(torsion_indices)
    collective0 = collective_variable(coordinate)
    bandwidth0 = initial_bandwidth_from_collective_variable(collective0)
    print(f"[init] bw_init={bandwidth0:.6g}  replicas={args.replicas}")

    # Unnormalized Gaussian K; metric = raw row_sum (no /B); adaptive band clipped to [2,4].
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
        f"bw_clip=[{rescale.bandwidth_min},{rescale.bandwidth_max}]  "
        f"global_pull={rescale.global_pull}"
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bandwidth0,
        distance_function=torus_pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )

    mass_path = Path(str(prefix) + "_mass.txt")
    if mass_path.is_file():
        mass_vals = np.loadtxt(mass_path, dtype=np.float64).reshape(-1)
        if (
            int(mass_vals[0]) == potential.num_atoms
            and mass_vals.size == potential.num_atoms + 1
        ):
            mass_vals = mass_vals[1:]
        mass = torch.as_tensor(
            mass_vals[: potential.num_atoms], dtype=dtype, device=device
        ).view(1, -1, 1)
    else:
        mass = 1.0

    cv_clip = (
        None
        if args.cv_grad_clip_norm is None or args.cv_grad_clip_norm <= 0.0
        else float(args.cv_grad_clip_norm)
    )
    integrator = BaoabFokkerPlanckIntegrator(
        energy_system,
        collective_variable,
        kernel,
        timestep=dt_akma,
        langevin_friction=gamma_akma,
        beta0=args.beta0,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
        ess_threshold=args.ess_threshold,
        min_lambda_step=args.min_lambda_step,
        max_lambda_step=args.max_lambda_step,
        num_lambda_trials=args.num_lambda_trials,
        mass=mass,
        boltzmann_constant=float(args.kB),
        update_pace=args.update_pace,
        collective_variable_grad_clip=cv_clip,
        device=device,
        dtype=dtype,
    )
    print(f"[init] cv_grad_clip_norm={cv_clip}")
    out = args.output or (EXPERIMENT_DIR / "results" / f"dialanine_{args.replicas}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "phi_psi"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _stage_tag(stage_i: int, lam: float, *, phase: str = "anneal") -> str:
        pref = "heal" if phase == "heal" else "stage"
        if phase == "pre_md":
            pref = "pre_md"
        return f"{pref}{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")

    def on_stage_end(stage_i: int, lam: float, st, phase: str = "anneal") -> None:
        cv_np = (
            collective_variable(st.coordinate).detach().cpu().numpy().reshape(-1, 2)
        )
        tag = _stage_tag(stage_i, lam, phase=phase)
        np.save(out / f"{tag}_coords.npy", st.coordinate.detach().cpu().numpy())
        np.save(out / f"{tag}_dihedral.npy", cv_np)
        np.save(
            out / f"{tag}_bandwidth.npy",
            integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
        )
        metrics = integrator._last_step_metrics
        print(
            f"[{phase}-end] {tag}  T_kin={metrics.get('kinetic_temperature_mean', float('nan')):.2f}K  "
            f"bw[min/med/max]="
            f"{metrics.get('bandwidth_min', float('nan')):.4g}/"
            f"{metrics.get('bandwidth_median', float('nan')):.4g}/"
            f"{metrics.get('bandwidth_max', float('nan')):.4g}",
            flush=True,
        )
        plot_phi_psi(
            cv_np,
            plots_dir / f"{tag}_phi_psi.png",
            title=f"dialanine φ–ψ  {tag}",
        )

    # Initial distribution
    plot_phi_psi(
        collective0.detach().cpu().numpy(),
        plots_dir / "stage_init_phi_psi.png",
        title="dialanine φ–ψ  init",
    )

    state = integrator.init_state(seed=args.seed)

    # Pre-MD: unbiased Langevin (kernel force ~0 via huge kernel_grad_beta)
    pre_md_steps = max(0, int(args.pre_md_steps))
    if pre_md_steps > 0:
        print(
            f"[pre-md] {pre_md_steps} unbiased Langevin steps "
            f"(kernel_grad_beta → 1e12)",
            flush=True,
        )
        saved_kgb = float(integrator.kernel_grad_beta)
        integrator.kernel_grad_beta = 1e12
        state = integrator.run_stage(
            pre_md_steps,
            initial_state=state,
            verbose=args.verbose,
            verbose_pace=args.verbose_pace,
            show_progress=not args.no_progress,
        )
        integrator.kernel_grad_beta = saved_kgb
        np.save(
            out / "init_coords_from_pre_md.npy",
            state.coordinate.detach().cpu().numpy(),
        )
        on_stage_end(0, 0.0, state, phase="pre_md")
        print("[pre-md] done → FPT init", flush=True)

    if args.heal_lambda_list:
        print(
            f"[init] heal λ={args.heal_lambda_list}  heal_steps={args.heal_steps}  "
            f"then anneal λ_init={args.lambda_init}",
            flush=True,
        )

    state = integrator.run(
        args.stage_steps,
        initial_state=state,
        seed=args.seed,
        verbose=args.verbose,
        verbose_pace=args.verbose_pace,
        show_progress=not args.no_progress,
        on_stage_end=on_stage_end,
        heal_lambdas=args.heal_lambda_list or None,
        heal_steps=int(args.heal_steps) if args.heal_lambda_list else None,
    )
    np.save(out / "final_coordinate.npy", state.coordinate.detach().cpu().numpy())
    np.save(
        out / "final_bandwidth.npy",
        integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
    )
    final_cv = collective_variable(state.coordinate).detach().cpu().numpy()
    np.save(out / "final_dihedral.npy", final_cv)
    plot_phi_psi(
        final_cv,
        plots_dir / "stage_final_phi_psi.png",
        title=f"dialanine φ–ψ  final λ={integrator.lambda_current:.4g}",
    )
    print(
        f"wrote {out / 'final_coordinate.npy'}  "
        f"cv0_shape={tuple(collective0.shape)}  lambda={integrator.lambda_current}"
    )


if __name__ == "__main__":
    main()
