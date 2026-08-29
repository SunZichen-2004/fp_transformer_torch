#!/usr/bin/env python3
"""Tetra-alanine (vacuum GAFF/SPONGE) + 3D φ CV + torus kernel + BAOAB-FP.

Sequence ACE–ALA–ALA–ALA–NME. CVs: φ1, φ2, φ3 (no ψ).
Fixed λ (no heal / no ESS annealing). Dump coords / CV / bandwidth every
``--dump-every`` steps for ``--total-time`` ps.

Kernel / rescale recipe matches vacuum dialanine sFPT used in
``dialanine_main_picture`` (unnormalized row-sum band [2, 4], γ=10 /ps).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from io_utils import load_torsion_indices, plot_phi_psi  # noqa: E402
from fes_window import (  # noqa: E402
    DEFAULT_META_CSV,
    PHI_PAIRS,
    kde_1d_window,
    kde_2d_window,
    load_meta_phi23,
    load_remd_phi3,
    plot_fes_pairs,
    remd_1d_fes,
    remd_pair_fes,
)
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

DEFAULT_SPONGE = EXPERIMENT_DIR / "tetra_alanine"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "backbone_phi.txt"
DEFAULT_INIT_NPY = EXPERIMENT_DIR / "init_coords_4096.npy"
DEFAULT_REMD = EXPERIMENT_DIR / "remd_280_400_n64_90605"


def initial_bandwidth_from_collective_variable(
    collective_variable: torch.Tensor,
) -> float:
    distance = torus_pairwise_distance(collective_variable)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    if not np.isfinite(median) or median <= 0.0:
        return 0.5
    return median


def select_phi_torsions(torsion_indices: np.ndarray) -> np.ndarray:
    """Keep 3 φ rows. A 6-row φψ file is subsampled as rows 0,2,4."""
    n = int(torsion_indices.shape[0])
    if n == 3:
        return torsion_indices
    if n == 6:
        return torsion_indices[0:6:2]
    raise ValueError(f"expected 3 φ or 6 φψ torsions, got {n}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=str, default=str(DEFAULT_SPONGE))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=4096)
    p.add_argument(
        "--total-time",
        type=float,
        default=500.0,
        help="Production length in ps (dt × n_steps)",
    )
    p.add_argument(
        "--dump-every",
        type=int,
        default=200,
        help="MD steps between dumps of coords / CV / bandwidth",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001, help="BAOAB dt in ps")
    p.add_argument("--baoab-dt", type=float, default=None, help="Alias of --timestep")
    p.add_argument(
        "--langevin-friction",
        type=float,
        default=10.0,
        help="Langevin γ in 1/ps",
    )
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_KCAL_PER_MOL_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument(
        "--lambda",
        dest="lambda_fixed",
        type=float,
        default=0.02,
        help="Fixed FP λ (no annealing)",
    )
    p.add_argument("--lambda-init", type=float, default=None, help="Alias of --lambda")
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--noise-scale", type=float, default=0.0)
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
    p.add_argument("--cv-grad-clip-norm", type=float, default=0.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=1e-3)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument("--plot-every", type=int, default=1, help="Scatter every N dumps; 0=off")
    p.add_argument("--remd-dir", type=Path, default=DEFAULT_REMD)
    p.add_argument(
        "--meta-fes",
        type=Path,
        default=DEFAULT_META_CSV if DEFAULT_META_CSV.is_file() else None,
        help="MetaD 2D F(φ2,φ3) CSV (kcal/mol); also used for 1D φ2/φ3",
    )
    p.add_argument("--fes-every-ps", type=float, default=10.0)
    p.add_argument("--fes-window-ps", type=float, default=10.0)
    p.add_argument("--fes-vmax", type=float, default=16.0)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--kde-chunk", type=int, default=2048)
    return p


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    if args.lambda_init is not None:
        args.lambda_fixed = args.lambda_init
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    return args


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    prefix = Path(args.sponge_prefix)
    lam = float(args.lambda_fixed)
    dt_ps = float(args.timestep)
    dump_every = max(1, int(args.dump_every))
    n_total = int(round(float(args.total_time) / dt_ps))
    if n_total < dump_every:
        raise ValueError(
            f"total steps {n_total} < dump-every {dump_every} "
            f"(total-time={args.total_time} ps, dt={dt_ps} ps)"
        )
    n_stages = n_total // dump_every
    n_run = n_stages * dump_every

    dt_akma = timestep_picosecond_to_akma(dt_ps)
    gamma_ps = float(args.langevin_friction)
    gamma_akma = friction_per_picosecond_to_akma(gamma_ps, dt_ps, dt_akma)

    print(f"device={device}  beta0={args.beta0:.6g}  sponge={prefix}")
    print(
        f"[baoab time] dt={dt_ps} ps → dt_akma={dt_akma:.12g}; "
        f"γ={gamma_ps} /ps → γ_akma={gamma_akma:.12g}; γ·dt={gamma_ps * dt_ps:.6g}"
    )
    print(
        f"[run] fixed λ={lam:g}  total={n_run} steps ({n_run * dt_ps:g} ps)  "
        f"dump_every={dump_every}  n_stages={n_stages}  R={args.replicas}",
        flush=True,
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
    torsion_indices = select_phi_torsions(torsion_indices)
    print(f"[CV] 3φ (0-based)={torsion_indices.tolist()}", flush=True)

    if args.init_npy:
        path = Path(os.path.abspath(os.path.expanduser(args.init_npy)))
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"bad init npy {path}: {loaded.shape}")
        if loaded.shape[1] != potential.num_atoms:
            raise ValueError(
                f"init npy n_atoms={loaded.shape[1]} != GAFF n_atoms={potential.num_atoms}"
            )
        if loaded.shape[0] < int(args.replicas):
            raise ValueError(
                f"init npy B={loaded.shape[0]} < --replicas={args.replicas}"
            )
        if loaded.shape[0] > int(args.replicas):
            idx = np.linspace(0, loaded.shape[0] - 1, int(args.replicas), dtype=np.int64)
            loaded = loaded[idx]
            print(
                f"[init] subsampled {path} → B={loaded.shape[0]}",
                flush=True,
            )
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        args.replicas = int(coordinate.shape[0])
    else:
        base = load_sponge_coordinates(prefix, num_atoms=potential.num_atoms)
        coordinate = (
            torch.as_tensor(base, dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(args.replicas, 1, 1)
        )
        if args.noise_scale > 0:
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
    print(
        f"[init] bw_init={bandwidth0:.6g}  replicas={args.replicas}  "
        f"cv_shape={tuple(collective0.shape)}",
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
        lambda_init=lam,
        lambda_max=lam,
        ess_threshold=1.0,
        min_lambda_step=0.0,
        max_lambda_step=0.0,
        num_lambda_trials=1,
        mass=mass,
        boltzmann_constant=float(args.kB),
        update_pace=args.update_pace,
        collective_variable_grad_clip=cv_clip,
        device=device,
        dtype=dtype,
    )
    integrator.lambda_current = lam
    integrator.sync_lambda_to_kernel_grad_beta()
    print(f"[init] cv_grad_clip_norm={cv_clip}  fixed λ={lam:g} (no anneal)")

    out = args.output or (
        EXPERIMENT_DIR / "results" / f"tetra_alanine_phi3_r{args.replicas}"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "phi_phi"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fes_dir = out / "fes_2d"
    fes_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "stage_log.csv"

    kT = float(args.kB) * float(args.temperature)
    fes_ax = np.linspace(-np.pi, np.pi, int(args.grid_n))
    n_win = max(1, int(round(float(args.fes_window_ps) / (dump_every * dt_ps))))
    n_fes_every = max(1, int(round(float(args.fes_every_ps) / (dump_every * dt_ps))))
    window: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=n_win)
    remd_phi = load_remd_phi3(Path(args.remd_dir), temperature=float(args.temperature))
    remd_fes = remd_pair_fes(remd_phi, ax=fes_ax, kT=kT)
    remd_1d = remd_1d_fes(remd_phi, ax=fes_ax, kT=kT)
    meta_2d = meta_1d = None
    if args.meta_fes is not None and Path(args.meta_fes).is_file():
        meta_2d, meta_1d = load_meta_phi23(Path(args.meta_fes), ax=fes_ax, kT=kT)
    print(
        f"[fes] every {args.fes_every_ps:g} ps  window={args.fes_window_ps:g} ps "
        f"({n_win} dumps)  grid={args.grid_n}  vmax={args.fes_vmax}",
        flush=True,
    )

    def _stage_tag(stage_i: int) -> str:
        return f"stage{stage_i:04d}_lambda{lam:.6f}".replace(".", "p")

    def dump_stage(stage_i: int, st) -> None:
        cv_np = collective_variable(st.coordinate).detach().cpu().numpy()
        if cv_np.ndim == 3:
            cv_np = cv_np.reshape(cv_np.shape[0], -1)
        bw_np = integrator.gaussian_kernel.bandwidth.detach().cpu().numpy()
        tag = _stage_tag(stage_i)
        np.save(out / f"{tag}_coords.npy", st.coordinate.detach().cpu().numpy())
        np.save(out / f"{tag}_dihedral.npy", cv_np)
        np.save(out / f"{tag}_bandwidth.npy", bw_np)
        metrics = integrator._last_step_metrics
        t_kin = metrics.get("kinetic_temperature_mean", float("nan"))
        bw_min = metrics.get("bandwidth_min", float("nan"))
        bw_med = metrics.get("bandwidth_median", float("nan"))
        bw_max = metrics.get("bandwidth_max", float("nan"))
        t_ps = (stage_i + 1) * dump_every * dt_ps
        print(
            f"[dump] {tag}  t={t_ps:.3g} ps  T_kin={t_kin:.2f}K  "
            f"bw[min/med/max]={bw_min:.4g}/{bw_med:.4g}/{bw_max:.4g}",
            flush=True,
        )
        new_file = not log_path.is_file()
        with log_path.open("a", newline="") as fh:
            wr = csv.writer(fh)
            if new_file:
                wr.writerow(
                    ["stage", "t_ps", "lambda", "T_kin", "bw_min", "bw_med", "bw_max"]
                )
            wr.writerow(
                [stage_i, t_ps, lam, t_kin, bw_min, bw_med, bw_max]
            )
        plot_n = int(args.plot_every)
        if plot_n > 0 and (stage_i % plot_n == 0 or stage_i == n_stages - 1):
            plot_phi_psi(
                cv_np,
                plots_dir / f"{tag}_phi.png",
                title=rf"tetra-alanine 3φ  {tag}",
            )
        window.append((cv_np, bw_np))
        if (stage_i + 1) % n_fes_every == 0 and len(window) >= n_win:
            rows = list(window)
            f_sfpt = {}
            for i, j, ii, jj in PHI_PAIRS:
                f_sfpt[(i, j)] = kde_2d_window(
                    rows,
                    i=i,
                    j=j,
                    ax=fes_ax,
                    lam=lam,
                    kT=kT,
                    device=device,
                    chunk=int(args.kde_chunk),
                )
                print(f"[fes] t={t_ps:g} ps  (φ{ii},φ{jj})", flush=True)
            f1_sfpt = np.stack(
                [
                    kde_1d_window(
                        rows,
                        i=i,
                        ax=fes_ax,
                        lam=lam,
                        kT=kT,
                        device=device,
                        chunk=int(args.kde_chunk),
                    )
                    for i in range(3)
                ]
            )
            np.savez_compressed(
                fes_dir / f"fes_2d_t{t_ps:04.0f}ps.npz",
                ax=fes_ax,
                t_ps=np.float64(t_ps),
                lam=np.float64(lam),
                F_phi1_phi2=f_sfpt[(0, 1)],
                F_phi1_phi3=f_sfpt[(0, 2)],
                F_phi2_phi3=f_sfpt[(1, 2)],
                F_remd_phi1_phi2=remd_fes[(0, 1)],
                F_remd_phi1_phi3=remd_fes[(0, 2)],
                F_remd_phi2_phi3=remd_fes[(1, 2)],
                F1_phi1=f1_sfpt[0],
                F1_phi2=f1_sfpt[1],
                F1_phi3=f1_sfpt[2],
                F1_remd_phi1=remd_1d[0],
                F1_remd_phi2=remd_1d[1],
                F1_remd_phi3=remd_1d[2],
                **(
                    {
                        "F_meta_phi2_phi3": meta_2d,
                        "F1_meta_phi1": meta_1d[0],
                        "F1_meta_phi2": meta_1d[1],
                        "F1_meta_phi3": meta_1d[2],
                    }
                    if meta_2d is not None
                    else {}
                ),
            )
            vs = "sFPT vs REMD" + (" vs MetaD" if meta_2d is not None else "")
            plot_fes_pairs(
                fes_dir / f"fes_2d_t{t_ps:04.0f}ps.png",
                ax=fes_ax,
                f_sfpt=f_sfpt,
                f_remd=remd_fes,
                f1_sfpt=f1_sfpt,
                f1_remd=remd_1d,
                f_meta=meta_2d,
                f1_meta=meta_1d,
                vmax=float(args.fes_vmax),
                title=rf"{vs}  last {args.fes_window_ps:g} ps  "
                rf"($t={t_ps:g}$ ps, $\lambda={lam:g}$)",
            )

    plot_phi_psi(
        collective0.detach().cpu().numpy(),
        plots_dir / "stage_init_phi.png",
        title="tetra-alanine 3φ  init",
    )

    state = integrator.init_state(seed=args.seed)
    for stage_i in range(n_stages):
        state = integrator.run_stage(
            dump_every,
            initial_state=state,
            verbose=args.verbose,
            verbose_pace=args.verbose_pace,
            show_progress=False,
        )
        dump_stage(stage_i, state)

    np.save(out / "final_coordinate.npy", state.coordinate.detach().cpu().numpy())
    np.save(
        out / "final_bandwidth.npy",
        integrator.gaussian_kernel.bandwidth.detach().cpu().numpy(),
    )
    final_cv = collective_variable(state.coordinate).detach().cpu().numpy()
    if final_cv.ndim == 3:
        final_cv = final_cv.reshape(final_cv.shape[0], -1)
    np.save(out / "final_dihedral.npy", final_cv)
    plot_phi_psi(
        final_cv,
        plots_dir / "stage_final_phi.png",
        title=f"tetra-alanine 3φ  final λ={lam:.4g}",
    )
    print(
        f"wrote {out / 'final_coordinate.npy'}  "
        f"cv0_shape={tuple(collective0.shape)}  lambda={lam}  stages={n_stages}"
    )


if __name__ == "__main__":
    main()
