#!/usr/bin/env python3
"""Particle-dimer 1D WT-MetaD (dimer distance CV) on CPU.

Uses BAOAB Langevin (kernel force off). Hills are deposited into a shared
well-tempered bias ``V(d)`` and added to the dimer potential. Initial
coordinates should be an sFPT pre-MD snapshot ``(B, 38, 2)``.
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
    WellTemperedMetadynamicsBias,
)
from fp_transformer_torch.distance import pairwise_distance  # noqa: E402
from fp_transformer_torch.potential.base import Potential  # noqa: E402


def _cv_matrix(collective: torch.Tensor) -> torch.Tensor:
    if collective.ndim == 3:
        return collective.squeeze(-1)
    if collective.ndim == 2:
        return collective
    raise ValueError(f"unexpected CV shape {tuple(collective.shape)}")


class DimerPlusMetadPotential(Potential):
    """``U_tot = U_dimer + V_bias(d)`` with shared WT hills."""

    def __init__(
        self,
        dimer: ParticleDimer,
        bias: WellTemperedMetadynamicsBias,
        collective_variable: ParticlePairDistance,
    ):
        self.dimer = dimer
        self.bias = bias
        self.collective_variable = collective_variable

    def energy(self, coordinate: torch.Tensor) -> torch.Tensor:
        u = self.dimer.energy(coordinate)
        s = _cv_matrix(self.collective_variable(coordinate))
        v = self.bias.bias_potential_cv(s)
        return u + v.reshape(-1, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=1e-4)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kB", type=float, default=1.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--init-coords", type=str, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=50)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metad-sigma", type=float, default=0.1)
    p.add_argument("--metad-height", type=float, default=0.1)
    p.add_argument("--metad-bias-factor", type=float, default=15.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument("--metad-max-hills", type=int, default=0)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument("--fes-grid", type=int, default=257)
    p.add_argument("--d-min", type=float, default=0.5)
    p.add_argument("--d-max", type=float, default=2.8)
    return p


def subsample_coords(loaded: np.ndarray, n: int) -> np.ndarray:
    b = int(loaded.shape[0])
    if n == b:
        return loaded
    if n > b:
        raise ValueError(f"replicas={n} > available init B={b}")
    idx = np.linspace(0, b - 1, n).astype(np.int64)
    return loaded[idx]


def plot_hist_and_fes(
    d: np.ndarray,
    d_grid: np.ndarray,
    fes: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    ax.hist(d, bins=40, range=(0.5, 2.8), color="#1f77b4", alpha=0.75, edgecolor="none")
    ax.axvline(0.97, color="0.35", ls="--", lw=0.8)
    ax.axvline(2.03, color="0.35", ls="--", lw=0.8)
    ax.set_xlabel(r"dimer distance $d$")
    ax.set_ylabel("count")
    ax.set_xlim(0.5, 2.8)
    ax = axes[1]
    fes_shift = fes - np.nanmin(fes)
    ax.plot(d_grid, fes_shift, color="#d62728", lw=1.4)
    ax.axvline(0.97, color="0.35", ls="--", lw=0.8)
    ax.axvline(2.03, color="0.35", ls="--", lw=0.8)
    ax.set_xlabel(r"dimer distance $d$")
    ax.set_ylabel(r"$F(d)-min$")
    ax.set_xlim(0.5, 2.8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fes_1d(bias: WellTemperedMetadynamicsBias, d_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = torch.as_tensor(
        d_grid.reshape(-1, 1), device=bias.sigma.device, dtype=bias.sigma.dtype
    )
    with torch.no_grad():
        v = bias.bias_potential_cv(s).detach().cpu().numpy()
    gamma = float(bias.bias_factor)
    fes = -(gamma / (gamma - 1.0)) * v
    return v.astype(np.float64), fes.astype(np.float64)


def save_hills(bias: WellTemperedMetadynamicsBias, path: Path) -> None:
    n = int(bias.n_hills)
    np.savez_compressed(
        path,
        centers=bias._centers[:n].detach().cpu().numpy() if n else np.zeros((0, bias.n_cv)),
        heights=bias._heights[:n].detach().cpu().numpy() if n else np.zeros((0,)),
        n_hills=n,
        sigma=bias.sigma.detach().cpu().numpy(),
        height0=np.float64(bias.height0),
        bias_factor=np.float64(bias.bias_factor),
        temperature=np.float64(bias.temperature),
        kB=np.float64(bias.kB),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma

    n_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float64
    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print(
        f"method=WT-MetaD  device={device}  threads={n_threads}  "
        f"kB={args.kB}  T={args.temperature}  β0={beta0:.6g}  "
        f"dt={args.timestep}  γ_fric={args.langevin_friction}  "
        f"R={args.replicas}  steps={args.steps}  "
        f"σ={args.metad_sigma}  w0={args.metad_height}  "
        f"bias_factor={args.metad_bias_factor}  "
        f"deposit_pace={args.metad_deposit_pace}",
        flush=True,
    )

    dimer = ParticleDimer()
    n_atoms = dimer.num_particles
    init_path = Path(os.path.abspath(os.path.expanduser(args.init_coords)))
    loaded = np.load(init_path)
    if loaded.ndim != 3 or loaded.shape[1] != n_atoms or loaded.shape[2] != 2:
        raise ValueError(f"bad init coords {init_path}: {loaded.shape}")
    coords_np = subsample_coords(loaded, int(args.replicas))
    coordinate = torch.as_tensor(coords_np, dtype=dtype, device=device)
    print(
        f"[init] {init_path}  loaded={tuple(loaded.shape)} → "
        f"subsampled={tuple(coordinate.shape)}",
        flush=True,
    )

    cv = ParticlePairDistance([0, 1])
    bias = WellTemperedMetadynamicsBias(
        cv,
        n_cv=1,
        sigma=float(args.metad_sigma),
        height=float(args.metad_height),
        bias_factor=float(args.metad_bias_factor),
        temperature=float(args.temperature),
        kB=float(args.kB),
        max_hills=int(args.metad_max_hills),
        device=device,
        dtype=dtype,
    )
    potential = DimerPlusMetadPotential(dimer, bias, cv)
    energy_system = EnergySystem(potential, coordinate)

    dummy_kernel = GaussianKernel(
        args.replicas,
        bandwidth=1.0,
        distance_function=pairwise_distance,
        rescale_per_replica=False,
        device=device,
        dtype=dtype,
    )
    integrator = BaoabFokkerPlanckIntegrator(
        energy_system,
        cv,
        dummy_kernel,
        timestep=args.timestep,
        langevin_friction=args.langevin_friction,
        beta0=beta0,
        lambda_init=1.0,
        lambda_max=1.0,
        ess_threshold=0.96,
        min_lambda_step=1.0,
        max_lambda_step=1.0,
        num_lambda_trials=2,
        mass=float(args.mass),
        boltzmann_constant=float(args.kB),
        update_pace=max(int(args.steps) + 1, 1),
        use_kernel_gradient=False,
        device=device,
        dtype=dtype,
    )

    out = args.output or (
        Path(__file__).resolve().parent
        / "results"
        / f"particle_dimer_meta_r{args.replicas}_T{args.temperature}_s{args.seed}"
    )
    out = Path(out)
    plots_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    np.save(out / "init_coords.npy", coords_np)

    d_grid = np.linspace(float(args.d_min), float(args.d_max), int(args.fes_grid))
    metrics_path = out / "stage_metrics.csv"
    metrics_rows: list[dict[str, object]] = []
    cv_chunks: list[np.ndarray] = []

    def dump(step: int, state) -> None:
        d = cv(state.coordinate).detach().cpu().numpy().reshape(-1)
        cv_chunks.append(d.copy())
        v, fes = fes_1d(bias, d_grid)
        tag = f"step{step:07d}"
        np.save(out / f"{tag}_coords.npy", state.coordinate.detach().cpu().numpy())
        np.save(out / f"{tag}_d.npy", d)
        np.savez_compressed(out / f"{tag}_fes.npz", d=d_grid, V=v, F=fes)
        save_hills(bias, out / "hills.npz")
        plot_hist_and_fes(
            d,
            d_grid,
            fes,
            plots_dir / f"{tag}.png",
            title=rf"WT-MetaD  {tag}  T={args.temperature:g}  R={args.replicas}",
        )
        t_kin = float(
            integrator._last_step_metrics.get("kinetic_temperature_mean", float("nan"))
        )
        t_std = float(
            integrator._last_step_metrics.get("kinetic_temperature_std", float("nan"))
        )
        row = {
            "step": int(step),
            "n_hills": int(bias.n_hills),
            "T_kin_mean": t_kin,
            "T_kin_std": t_std,
            "T_target": float(args.temperature),
            "d_min": float(d.min()),
            "d_med": float(np.median(d)),
            "d_max": float(d.max()),
            "n_compact": int(np.sum(d < 1.5)),
            "n_extended": int(np.sum(d >= 1.5)),
            "fes_span": float(np.nanmax(fes) - np.nanmin(fes)),
        }
        metrics_rows.append(row)
        with metrics_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(
            f"[dump] {tag}  n_hills={bias.n_hills}  "
            f"d[min/med/max]={d.min():.3f}/{np.median(d):.3f}/{d.max():.3f}  "
            f"T_kin={t_kin:.3g}±{t_std:.2g}  "
            f"n_comp/ext={row['n_compact']}/{row['n_extended']}",
            flush=True,
        )

    state = integrator.init_state(seed=int(args.seed))
    dump(0, state)

    remaining = int(args.steps)
    done = 0
    deposit_pace = max(1, int(args.metad_deposit_pace))
    dump_every = max(deposit_pace, int(args.dump_every))
    while remaining > 0:
        n_this = min(deposit_pace, remaining)
        state = integrator.run_stage(
            n_this,
            initial_state=state,
            verbose=bool(args.verbose),
            verbose_pace=int(args.verbose_pace),
            show_progress=not bool(args.no_progress),
        )
        dep = bias.deposit(state.coordinate)
        done += n_this
        remaining -= n_this
        if done % dump_every == 0 or remaining == 0:
            print(
                f"[deposit] step={done}  n_hills={int(dep['n_hills'])}  "
                f"w_mean={dep['deposit_height_mean']:.4g}  "
                f"V_mean={dep['bias_energy_at_deposit_mean']:.4g}",
                flush=True,
            )
            dump(done, state)

    np.save(out / "final_coordinate.npy", state.coordinate.detach().cpu().numpy())
    np.save(out / "cv_snapshots.npy", np.stack(cv_chunks, axis=0))
    v, fes = fes_1d(bias, d_grid)
    np.savez_compressed(out / "fes_final.npz", d=d_grid, V=v, F=fes)
    save_hills(bias, out / "hills.npz")
    print(f"[done] steps={done}  n_hills={bias.n_hills}  out={out}", flush=True)


if __name__ == "__main__":
    main()
