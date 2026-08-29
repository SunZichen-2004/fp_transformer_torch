#!/usr/bin/env python3
"""Tetra-alanine sFPT vs REMD: 6D torus-KDE FES, then Boltzmann project.

Reconstruction (sFPT)
  1. Evaluate the **6D** torus Gaussian KDE (same kernel as sampling) on a
     uniform periodic grid in (φ1, ψ1, φ2, ψ2, φ3, ψ3).
  2. One density per stage; a group (or last-frac) **averages P**, then
     ``F_6 = −kT ln P / λ``.
  3. 1D / 2D FES = Boltzmann projection of ``F_6``
     ``F(q) = −kT ln ∫ exp(−β F_6(q, r)) dr``.

REMD (300 K replica) uses the **same 6D grid**; ``P_6`` is a periodic
histogram (unbiased, λ = 1) then the same Boltzmann projection.
For λ = 1 this is identical to the 1D/2D marginal histogram — do **not**
KDE in the subspace, and do **not** use a 6D KDE with a median bandwidth
(that oversmooths in 6D).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from fp_transformer_torch.kernel.gaussian import (  # noqa: E402
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)
from fp_transformer_torch.units import KB_KCAL_PER_MOL_K  # noqa: E402
from plot_fes_compare import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    FE_LABEL,
    _draw_fes2d_on_ax,
    _rdbu_r_white_over,
    nature_rcparams,
    relative_fes,
)
from plot_fes_rmse_vs_time import rmse_1d  # noqa: E402

STAGE_RE = re.compile(r"stage(\d+)", re.I)
LAMBDA_RE = re.compile(r"lambda(\d+)p(\d+)", re.I)
COLOR_REMD = COLOR_META
N_CV = 6
PHI_IDX = (0, 2, 4)
# (cv_i, cv_j, 1-based i, 1-based j)
PHI_PAIRS = ((0, 2, 1, 2), (0, 4, 1, 3), (2, 4, 2, 3))
PSI_IDX = (1, 3, 5)

DEFAULT_FPT = EXP / "results" / "tetra_alanine_sfpt_r1024_s0_88562"
DEFAULT_REMD = EXP / "remd_280_400_n64_90605"


def parse_lambda(path: Path) -> float | None:
    m = LAMBDA_RE.search(path.name)
    if m is None:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def parse_stage(path: Path) -> int | None:
    m = STAGE_RE.search(path.name)
    return int(m.group(1)) if m else None


def discover_dihedral_stages(
    data_dir: Path,
    *,
    lambda_target: float,
    lambda_atol: float,
) -> list[tuple[int, Path, Path, float]]:
    rows: list[tuple[int, Path, Path, float]] = []
    for dih in sorted(data_dir.glob("stage[0-9]*_lambda*_dihedral.npy")):
        idx = parse_stage(dih)
        lam = parse_lambda(dih)
        if idx is None or lam is None:
            continue
        if abs(float(lam) - float(lambda_target)) > float(lambda_atol):
            continue
        bw = dih.with_name(dih.name.replace("_dihedral.npy", "_bandwidth.npy"))
        if not bw.is_file():
            continue
        rows.append((idx, dih, bw, float(lam)))
    rows.sort(key=lambda r: r[0])
    return rows


def as_cv6(cv: np.ndarray) -> np.ndarray:
    arr = np.asarray(cv, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[1] != N_CV:
        raise ValueError(f"expected (N, 6) dihedrals, got {arr.shape}")
    return arr


def to_kernel_cv(cv: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(cv, dtype=np.float64))
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    if t.ndim == 3:
        t = t.reshape(t.shape[0], -1)
    if t.ndim != 2:
        raise ValueError(f"expected CV (N, d), got {tuple(t.shape)}")
    return t.reshape(t.shape[0], t.shape[1], 1)


def torus_cross_distance(grid_cv: torch.Tensor, sample_cv: torch.Tensor) -> torch.Tensor:
    m = int(grid_cv.shape[0])
    n = int(sample_cv.shape[0])
    g = grid_cv.reshape(m, -1)
    s = sample_cv.reshape(n, -1)
    feat = float(g.shape[1])
    dots = torch.cos(g) @ torch.cos(s).T + torch.sin(g) @ torch.sin(s).T
    return 2.0 - 2.0 * (dots / feat)


def density_on_grid_torus(
    grid_xy: np.ndarray,
    sample_cv: np.ndarray,
    bandwidth: np.ndarray,
    *,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    samples = to_kernel_cv(sample_cv).to(device=device)
    bw = torch.as_tensor(bandwidth, dtype=torch.float64, device=device).reshape(-1)
    n = float(samples.shape[0])
    m = int(grid_xy.shape[0])
    feature_dim = float(samples.shape[-2] * samples.shape[-1])
    out = np.empty((m,), dtype=np.float64)
    for start in range(0, m, int(chunk)):
        stop = min(start + int(chunk), m)
        grid = to_kernel_cv(grid_xy[start:stop]).to(device=device)
        dist = torus_cross_distance(grid, samples)
        log_k = (-0.5 * dist) / bandwidth_column_squared(bw) + gaussian_kernel_log_normalization(
            bw, feature_dim
        )
        log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
        out[start:stop] = torch.exp(log_p).detach().cpu().numpy()
    return out


def median_torus_bandwidth(sample_cv: np.ndarray, *, max_n: int = 2048) -> float:
    x = np.asarray(sample_cv, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] > int(max_n):
        rng = np.random.default_rng(0)
        x = x[rng.choice(x.shape[0], int(max_n), replace=False)]
    t = to_kernel_cv(x)
    dist = torus_cross_distance(t, t)
    upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    vals = dist[upper]
    pos = vals[vals > 0]
    if pos.numel() == 0:
        return 0.2
    med = float(torch.median(pos).item())
    return med if np.isfinite(med) and med > 0 else 0.2


def make_torus_grid_6d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Periodic 6D grid, endpoint=False so ±π is not double-counted."""
    ax = np.linspace(-np.pi, np.pi, int(n), endpoint=False)
    meshes = np.meshgrid(*([ax] * N_CV), indexing="ij")
    grid = np.stack([m.ravel() for m in meshes], axis=-1)
    return ax, grid


def fes_from_density_6d(p: np.ndarray, *, kT: float, lam: float, n: int) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, None)
    f = -float(kT) * np.log(p) / float(lam)
    return relative_fes(f, None).reshape((int(n),) * N_CV)


def boltzmann_project(f6: np.ndarray, keep_axes: tuple[int, ...], *, kT: float) -> np.ndarray:
    """Boltzmann-project 6D F onto ``keep_axes`` (equal cell weights)."""
    other = tuple(i for i in range(N_CV) if i not in keep_axes)
    beta = 1.0 / float(kT)
    a = -beta * np.asarray(f6, dtype=np.float64)
    order = list(keep_axes) + list(other)
    a = np.transpose(a, order)
    lead = a.shape[: len(keep_axes)]
    flat = a.reshape(*lead, -1)
    m = np.max(flat, axis=-1, keepdims=True)
    logz = np.squeeze(m, axis=-1) + np.log(np.sum(np.exp(flat - m), axis=-1))
    return relative_fes((-float(kT)) * logz, None)


def upsample_periodic_1d(y: np.ndarray, ax_in: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    yp = np.concatenate([y, y[:1]])
    ax = np.concatenate([ax_in, ax_in[:1] + 2.0 * np.pi])
    axo = np.linspace(-np.pi, np.pi, int(n_out))
    xq = np.mod(axo - ax[0], 2.0 * np.pi) + ax[0]
    return axo, np.interp(xq, ax, yp)


def upsample_periodic_2d(f_xy: np.ndarray, ax_in: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """``f_xy[i, j] = F(ax[i], ax[j])`` → fine grid, contourf-ready ``(ny, nx)``."""
    fp = np.pad(f_xy, ((0, 1), (0, 1)), mode="wrap")
    ax = np.concatenate([ax_in, ax_in[:1] + 2.0 * np.pi])
    interp = RegularGridInterpolator((ax, ax), fp, bounds_error=False, fill_value=None)
    axo = np.linspace(-np.pi, np.pi, int(n_out))
    gx, gy = np.meshgrid(axo, axo, indexing="xy")
    qx = np.mod(gx.ravel() - ax[0], 2.0 * np.pi) + ax[0]
    qy = np.mod(gy.ravel() - ax[0], 2.0 * np.pi) + ax[0]
    # interpolator takes (axis0, axis1) = (x, y) matching f_xy[i,j]
    vals = interp(np.stack([qx, qy], axis=-1)).reshape(gx.shape)
    return axo, relative_fes(vals, None)


def load_remd_300k(
    remd_dir: Path,
    *,
    temperature: float,
    discard_frac: float,
) -> tuple[np.ndarray, float, int]:
    temps = np.load(remd_dir / "temperatures.npy").astype(np.float64)
    dih = np.load(remd_dir / "traj_dihedral.npy")
    if dih.ndim != 3 or dih.shape[-1] != N_CV:
        raise ValueError(f"unexpected REMD dihedral shape {dih.shape}")
    i_t = int(np.argmin(np.abs(temps - float(temperature))))
    t_slot = float(temps[i_t])
    n0 = int(np.floor(float(discard_frac) * dih.shape[0]))
    samples = np.asarray(dih[n0:, i_t, :], dtype=np.float64)
    print(
        f"[remd] {remd_dir.name}  T_slot={t_slot:.3f} K (idx={i_t})  "
        f"frames {n0}:{dih.shape[0]} → {samples.shape[0]}",
        flush=True,
    )
    meta_path = remd_dir / "remd_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        print(
            f"[remd] steps={meta.get('steps')}  dt={meta.get('baoab_dt_ps')} ps  "
            f"n_replicas={meta.get('n_replicas')}",
            flush=True,
        )
    return samples, t_slot, i_t


def histogram_density_6d(samples: np.ndarray, ax: np.ndarray) -> np.ndarray:
    """Periodic 6D histogram on the same torus bins as the KDE grid (λ=1 / REMD).

    Boltzmann-projecting ``F=−kT ln P_6`` is then identical to the 1D/2D
    marginal histogram; empty bins contribute 0.
    """
    n = int(ax.shape[0])
    width = 2.0 * np.pi / float(n)
    x = np.mod(np.asarray(samples, dtype=np.float64) + np.pi, 2.0 * np.pi)
    idx = np.floor(x / width).astype(np.int64) % n
    flat = np.ravel_multi_index(tuple(idx.T), (n,) * N_CV)
    counts = np.zeros((n,) * N_CV, dtype=np.float64)
    np.add.at(counts.ravel(), flat, 1.0)
    total = float(counts.sum())
    if total <= 0.0:
        raise RuntimeError("REMD 6D histogram is empty")
    return (counts / total).ravel()


def kde_stage_6d(
    dih_p: Path,
    bw_p: Path,
    grid: np.ndarray,
    *,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    cv = as_cv6(np.load(dih_p))
    bw = np.load(bw_p)
    return density_on_grid_torus(grid, cv, bw, chunk=chunk, device=device)


def plot_rmse_vs_time(out: Path, t: np.ndarray, rmse: np.ndarray, *, ylabel: str) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for i in range(3):
        ax.plot(
            t, rmse[:, i], color=colors[i], lw=1.4, marker="o", ms=2.5,
            label=rf"$\phi_{i+1}$",
        )
    ax.plot(t, rmse.mean(axis=1), color="0.15", lw=1.8, label="mean", zorder=3)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def plot_1d_phi_overlay(
    out: Path,
    xi: np.ndarray,
    f_sfpt: np.ndarray,
    f_remd: np.ndarray,
    *,
    vmax: float,
    title: str,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.15), sharey=True)
    for i, ax in enumerate(axes):
        y_r = relative_fes(f_remd[i], vmax if vmax > 0 else None)
        y_s = relative_fes(f_sfpt[i], vmax if vmax > 0 else None)
        ax.plot(xi, y_r, color=COLOR_REMD, lw=1.5, ls="--", label="REMD", zorder=2)
        ax.plot(xi, y_s, color=COLOR_SFPT, lw=1.6, label="sFPT KDE", zorder=3)
        ax.set_xlabel(rf"$\phi_{i+1}$ / rad")
        ax.set_xlim(-np.pi, np.pi)
        if vmax > 0:
            ax.set_ylim(0.0, float(vmax) * 1.02)
        ax.tick_params(which="both", top=False, right=False)
        if i == 0:
            ax.set_ylabel(FE_LABEL)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def plot_2d_pair(
    out: Path,
    x: np.ndarray,
    y: np.ndarray,
    f_sfpt: np.ndarray,
    f_remd: np.ndarray,
    *,
    vmax: float,
    n_levels: int,
    xlabel: str,
    ylabel: str,
    title_a: str,
    title_b: str,
) -> None:
    plt.rcParams.update(nature_rcparams())
    levels = np.linspace(0.0, float(vmax), int(n_levels))
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = _rdbu_r_white_over()
    fig = plt.figure(figsize=(7.4, 3.5))
    gs = gridspec.GridSpec(
        1, 4, figure=fig, width_ratios=[1.0, 1.0, 0.06, 0.02],
        wspace=0.18, left=0.08, right=0.92, bottom=0.16, top=0.88,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    im = _draw_fes2d_on_ax(
        ax0, x, y, relative_fes(f_sfpt, None),
        levels=levels, norm=norm, cmap=cmap, vmax=vmax,
        panel=None, title=title_a, xlabel=xlabel, ylabel=ylabel,
    )
    _draw_fes2d_on_ax(
        ax1, x, y, relative_fes(f_remd, None),
        levels=levels, norm=norm, cmap=cmap, vmax=vmax,
        panel=None, title=title_b, xlabel=xlabel, ylabel=ylabel, show_ylabel=False,
    )
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(FE_LABEL, labelpad=8)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def plot_rama_grid(
    out: Path,
    x: np.ndarray,
    y: np.ndarray,
    f_sfpt: np.ndarray,
    f_remd: np.ndarray,
    *,
    vmax: float,
    n_levels: int,
) -> None:
    plt.rcParams.update(nature_rcparams())
    levels = np.linspace(0.0, float(vmax), int(n_levels))
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = _rdbu_r_white_over()
    fig = plt.figure(figsize=(7.6, 10.2))
    gs = gridspec.GridSpec(
        3, 3, figure=fig, width_ratios=[1.0, 1.0, 0.06],
        wspace=0.16, hspace=0.28, left=0.10, right=0.90, bottom=0.06, top=0.96,
    )
    im = None
    for i in range(3):
        ax_s = fig.add_subplot(gs[i, 0])
        ax_r = fig.add_subplot(gs[i, 1])
        im = _draw_fes2d_on_ax(
            ax_s, x, y, relative_fes(f_sfpt[i], None),
            levels=levels, norm=norm, cmap=cmap, vmax=vmax,
            panel=None, title=rf"sFPT  $\phi_{i+1}$–$\psi_{i+1}$",
            xlabel=rf"$\phi_{i+1}$ / rad", ylabel=rf"$\psi_{i+1}$ / rad",
        )
        _draw_fes2d_on_ax(
            ax_r, x, y, relative_fes(f_remd[i], None),
            levels=levels, norm=norm, cmap=cmap, vmax=vmax,
            panel=None, title=rf"REMD  $\phi_{i+1}$–$\psi_{i+1}$",
            xlabel=rf"$\phi_{i+1}$ / rad", ylabel=rf"$\psi_{i+1}$ / rad",
            show_ylabel=False,
        )
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(FE_LABEL, labelpad=8)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def plot_phi_ij_grid(
    out: Path,
    x: np.ndarray,
    y: np.ndarray,
    f_sfpt: np.ndarray,
    f_remd: np.ndarray,
    *,
    vmax: float,
    n_levels: int,
    pairs: tuple = PHI_PAIRS,
) -> None:
    plt.rcParams.update(nature_rcparams())
    levels = np.linspace(0.0, float(vmax), int(n_levels))
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = _rdbu_r_white_over()
    fig = plt.figure(figsize=(7.6, 10.2))
    gs = gridspec.GridSpec(
        3, 3, figure=fig, width_ratios=[1.0, 1.0, 0.06],
        wspace=0.16, hspace=0.28, left=0.10, right=0.90, bottom=0.06, top=0.96,
    )
    im = None
    for k, (_ci, _cj, i, j) in enumerate(pairs):
        ax_s = fig.add_subplot(gs[k, 0])
        ax_r = fig.add_subplot(gs[k, 1])
        im = _draw_fes2d_on_ax(
            ax_s, x, y, relative_fes(f_sfpt[k], None),
            levels=levels, norm=norm, cmap=cmap, vmax=vmax,
            panel=None, title=rf"sFPT  $\phi_{i}$–$\phi_{j}$",
            xlabel=rf"$\phi_{i}$ / rad", ylabel=rf"$\phi_{j}$ / rad",
        )
        _draw_fes2d_on_ax(
            ax_r, x, y, relative_fes(f_remd[k], None),
            levels=levels, norm=norm, cmap=cmap, vmax=vmax,
            panel=None, title=rf"REMD  $\phi_{i}$–$\phi_{j}$",
            xlabel=rf"$\phi_{i}$ / rad", ylabel=rf"$\phi_{j}$ / rad",
            show_ylabel=False,
        )
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(FE_LABEL, labelpad=8)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def project_phi_1d(f6: np.ndarray, ax: np.ndarray, *, kT: float, n_plot: int) -> tuple[np.ndarray, np.ndarray]:
    f1 = np.stack([boltzmann_project(f6, (PHI_IDX[i],), kT=kT) for i in range(3)], axis=0)
    xi, f_plot = upsample_periodic_1d(f1[0], ax, n_plot)
    out = np.zeros((3, n_plot), dtype=np.float64)
    out[0] = f_plot
    for i in range(1, 3):
        _, out[i] = upsample_periodic_1d(f1[i], ax, n_plot)
    return xi, out


def project_2d_pair(
    f6: np.ndarray,
    ax: np.ndarray,
    keep: tuple[int, int],
    *,
    kT: float,
    n_plot: int,
) -> tuple[np.ndarray, np.ndarray]:
    f2 = boltzmann_project(f6, keep, kT=kT)
    return upsample_periodic_2d(f2, ax, n_plot)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dir", type=Path, default=DEFAULT_FPT)
    p.add_argument("--remd-dir", type=Path, default=DEFAULT_REMD)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--lambda-target", type=float, default=0.08)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=1.0)
    p.add_argument("--last-frac", type=float, default=0.5)
    p.add_argument("--remd-discard-frac", type=float, default=0.2)
    p.add_argument("--n-grid-6d", type=int, default=10, help="bins per CV on the 6D torus grid")
    p.add_argument("--n-plot-1d", type=int, default=129)
    p.add_argument("--n-plot-2d", type=int, default=80)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--rmse-vmax", type=float, default=8.0)
    p.add_argument("--n-levels", type=int, default=21)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    fpt = Path(args.fpt_dir).resolve()
    remd_dir = Path(args.remd_dir).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else fpt / "convergence"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    kT = float(KB_KCAL_PER_MOL_K) * float(args.temperature)
    rmse_vmax = float(args.rmse_vmax) if float(args.rmse_vmax) > 0 else None
    n6 = int(args.n_grid_6d)
    ax6, grid6 = make_torus_grid_6d(n6)
    print(
        f"[init] device={device}  kT={kT:.4g} kcal/mol  6D grid n={n6} "
        f"({grid6.shape[0]} pts)  out={out}",
        flush=True,
    )

    remd_cv, t_slot, i_t = load_remd_300k(
        remd_dir,
        temperature=float(args.temperature),
        discard_frac=float(args.remd_discard_frac),
    )
    print(
        f"[remd] 6D histogram on n={n6} torus grid  N={remd_cv.shape[0]}  "
        f"(λ=1 → project ≡ 1D/2D marginal hist)",
        flush=True,
    )
    p_remd = histogram_density_6d(remd_cv, ax6)
    n_occ = int(np.count_nonzero(p_remd))
    print(
        f"[remd] occupied bins={n_occ}/{p_remd.size}  "
        f"Pmax={float(p_remd.max()):.3g}",
        flush=True,
    )
    f6_remd = fes_from_density_6d(p_remd, kT=kT, lam=1.0, n=n6)
    xi, f_remd_phi = project_phi_1d(f6_remd, ax6, kT=kT, n_plot=int(args.n_plot_1d))
    ax2, f_remd_phi12 = project_2d_pair(
        f6_remd, ax6, (0, 2), kT=kT, n_plot=int(args.n_plot_2d)
    )
    f_remd_phij = np.zeros((len(PHI_PAIRS), int(args.n_plot_2d), int(args.n_plot_2d)), dtype=np.float64)
    for k, (ci, cj, i, j) in enumerate(PHI_PAIRS):
        _, f_remd_phij[k] = project_2d_pair(
            f6_remd, ax6, (ci, cj), kT=kT, n_plot=int(args.n_plot_2d)
        )
        print(
            f"[remd] projected Fmax  φ{i}–φ{j}={float(np.nanmax(f_remd_phij[k])):.3g}",
            flush=True,
        )
    f_remd_rama = np.zeros((3, int(args.n_plot_2d), int(args.n_plot_2d)), dtype=np.float64)
    for i, keep in enumerate(((0, 1), (2, 3), (4, 5))):
        _, f_remd_rama[i] = project_2d_pair(
            f6_remd, ax6, keep, kT=kT, n_plot=int(args.n_plot_2d)
        )
        print(
            f"[remd] projected Fmax  φ{i+1}={float(np.nanmax(f_remd_phi[i])):.3g}  "
            f"φ{i+1}ψ{i+1}={float(np.nanmax(f_remd_rama[i])):.3g}",
            flush=True,
        )
    print(f"[remd] projected Fmax φ1–φ2={float(np.nanmax(f_remd_phi12)):.3g}", flush=True)

    rows = discover_dihedral_stages(
        fpt, lambda_target=float(args.lambda_target), lambda_atol=float(args.lambda_atol)
    )
    if not rows:
        raise RuntimeError(f"no λ≈{args.lambda_target:g} stages in {fpt}")
    print(
        f"[sfpt] {len(rows)} stages  {rows[0][0]}–{rows[-1][0]}  "
        f"λ≈{rows[0][3]:g}…{rows[-1][3]:g}",
        flush=True,
    )

    gsize = int(args.group_size)
    n_groups = len(rows) // gsize
    if n_groups < 1:
        raise RuntimeError(f"need ≥{gsize} stages for one group, got {len(rows)}")
    n_keep = max(1, int(np.ceil(float(args.last_frac) * len(rows))))
    i_last0 = len(rows) - n_keep

    t = np.zeros(n_groups, dtype=np.float64)
    rmse = np.zeros((n_groups, 3), dtype=np.float64)
    p_last = np.zeros(grid6.shape[0], dtype=np.float64)
    lam_last = 0.0
    n_last_acc = 0

    for ig in range(n_groups):
        chunk_rows = rows[ig * gsize : (ig + 1) * gsize]
        p_acc = np.zeros(grid6.shape[0], dtype=np.float64)
        lam_acc = 0.0
        for j, (idx, dih_p, bw_p, lam) in enumerate(chunk_rows):
            p = kde_stage_6d(dih_p, bw_p, grid6, chunk=int(args.chunk), device=device)
            p_acc += p
            lam_acc += float(lam)
            global_i = ig * gsize + j
            if global_i >= i_last0:
                p_last += p
                lam_last += float(lam)
                n_last_acc += 1
        p_g = p_acc / float(len(chunk_rows))
        lam_g = lam_acc / float(len(chunk_rows))
        f6 = fes_from_density_6d(p_g, kT=kT, lam=lam_g, n=n6)
        _, f1 = project_phi_1d(f6, ax6, kT=kT, n_plot=int(args.n_plot_1d))
        t[ig] = float(ig + 1) * float(args.dt_group)
        for i in range(3):
            rmse[ig, i] = rmse_1d(
                f1[i], f_remd_phi[i], vmax=rmse_vmax, xi=xi,
                xi_lo=float(xi.min()), xi_hi=float(xi.max()),
            )
        if ig == 0 or (ig + 1) % 5 == 0 or ig + 1 == n_groups:
            print(
                f"[rmse] group {ig+1}/{n_groups}  t={t[ig]:.2f} ps  "
                f"RMSE φ={rmse[ig, 0]:.3f}/{rmse[ig, 1]:.3f}/{rmse[ig, 2]:.3f}  "
                f"mean={rmse[ig].mean():.3f}",
                flush=True,
            )

    leftover = rows[n_groups * gsize :]
    for j, (_idx, dih_p, bw_p, lam) in enumerate(leftover):
        global_i = n_groups * gsize + j
        if global_i < i_last0:
            continue
        p = kde_stage_6d(dih_p, bw_p, grid6, chunk=int(args.chunk), device=device)
        p_last += p
        lam_last += float(lam)
        n_last_acc += 1

    plot_rmse_vs_time(
        out / "fes_rmse_phi_1d_vs_time.png",
        t,
        rmse,
        ylabel=r"RMSE vs REMD $F(\phi)$ / kcal mol$^{-1}$",
    )

    if n_last_acc != n_keep:
        print(f"[warn] last-frac accumulated {n_last_acc} != n_keep={n_keep}", flush=True)
    p_last /= float(max(n_last_acc, 1))
    lam_last /= float(max(n_last_acc, 1))
    f6_last = fes_from_density_6d(p_last, kT=kT, lam=lam_last, n=n6)
    last_rows = rows[-n_keep:]
    print(
        f"[last-frac] stages {last_rows[0][0]}–{last_rows[-1][0]}  n={n_last_acc}  λ={lam_last:g}",
        flush=True,
    )

    _, f_last_phi = project_phi_1d(f6_last, ax6, kT=kT, n_plot=int(args.n_plot_1d))
    rmse_last = np.array(
        [
            rmse_1d(
                f_last_phi[i], f_remd_phi[i], vmax=rmse_vmax, xi=xi,
                xi_lo=float(xi.min()), xi_hi=float(xi.max()),
            )
            for i in range(3)
        ],
        dtype=np.float64,
    )
    print(f"[last-frac] RMSE φ={rmse_last}  mean={rmse_last.mean():.4f}", flush=True)
    plot_1d_phi_overlay(
        out / "fes_1d_phi_last_frac_vs_remd.png",
        xi,
        f_last_phi,
        f_remd_phi,
        vmax=float(args.vmax),
        title=(
            rf"6D KDE → $F(\phi)$  last {float(args.last_frac):.0%}  "
            rf"stage {last_rows[0][0]}–{last_rows[-1][0]}  RMSE={rmse_last.mean():.3g}"
        ),
    )

    _, f_last_phi12 = project_2d_pair(
        f6_last, ax6, (0, 2), kT=kT, n_plot=int(args.n_plot_2d)
    )
    plot_2d_pair(
        out / "fes_2d_phi1_phi2_last_frac_vs_remd.png",
        ax2, ax2, f_last_phi12, f_remd_phi12,
        vmax=float(args.vmax), n_levels=int(args.n_levels),
        xlabel=r"$\phi_1$ / rad", ylabel=r"$\phi_2$ / rad",
        title_a="sFPT 6D→2D", title_b="REMD 6D→2D",
    )

    f_last_rama = np.zeros_like(f_remd_rama)
    for i, keep in enumerate(((0, 1), (2, 3), (4, 5))):
        _, f_last_rama[i] = project_2d_pair(
            f6_last, ax6, keep, kT=kT, n_plot=int(args.n_plot_2d)
        )
        plot_2d_pair(
            out / f"fes_2d_phi{i+1}_psi{i+1}_last_frac_vs_remd.png",
            ax2, ax2, f_last_rama[i], f_remd_rama[i],
            vmax=float(args.vmax), n_levels=int(args.n_levels),
            xlabel=rf"$\phi_{i+1}$ / rad", ylabel=rf"$\psi_{i+1}$ / rad",
            title_a="sFPT 6D→2D", title_b="REMD 6D→2D",
        )
    plot_rama_grid(
        out / "fes_2d_phi_psi_all_last_frac_vs_remd.png",
        ax2, ax2, f_last_rama, f_remd_rama,
        vmax=float(args.vmax), n_levels=int(args.n_levels),
    )

    f_last_phij = np.zeros_like(f_remd_phij)
    for k, (ci, cj, i, j) in enumerate(PHI_PAIRS):
        _, f_last_phij[k] = project_2d_pair(
            f6_last, ax6, (ci, cj), kT=kT, n_plot=int(args.n_plot_2d)
        )
        plot_2d_pair(
            out / f"fes_2d_phi{i}_phi{j}_last_frac_vs_remd.png",
            ax2, ax2, f_last_phij[k], f_remd_phij[k],
            vmax=float(args.vmax), n_levels=int(args.n_levels),
            xlabel=rf"$\phi_{i}$ / rad", ylabel=rf"$\phi_{j}$ / rad",
            title_a="sFPT 6D→2D", title_b="REMD 6D→2D",
        )
    plot_phi_ij_grid(
        out / "fes_2d_phi_ij_all_last_frac_vs_remd.png",
        ax2, ax2, f_last_phij, f_remd_phij,
        vmax=float(args.vmax), n_levels=int(args.n_levels),
    )

    np.savez_compressed(
        out / "convergence_phi_remd.npz",
        ax_6d=ax6,
        xi=xi,
        ax_2d=ax2,
        t=t,
        rmse_phi=rmse,
        F1d_sfpt_last=f_last_phi,
        F1d_remd=f_remd_phi,
        F2d_phi1_phi2_sfpt=f_last_phi12,
        F2d_phi1_phi2_remd=f_remd_phi12,
        F2d_rama_sfpt=f_last_rama,
        F2d_rama_remd=f_remd_rama,
        F2d_phi_ij_sfpt=f_last_phij,
        F2d_phi_ij_remd=f_remd_phij,
        F6_sfpt_last=f6_last,
        F6_remd=f6_remd,
        rmse_last=rmse_last,
        last_frac=np.float64(args.last_frac),
        stage_lo=np.int64(last_rows[0][0]),
        stage_hi=np.int64(last_rows[-1][0]),
        n_stages_last=np.int64(n_last_acc),
        n_grid_6d=np.int64(n6),
        lambda_target=np.float64(args.lambda_target),
        lambda_last=np.float64(lam_last),
        temperature=np.float64(args.temperature),
        remd_T_slot=np.float64(t_slot),
        remd_T_index=np.int64(i_t),
        remd_method=np.asarray("histogram_6d"),
        group_size=np.int64(gsize),
        dt_group=np.float64(args.dt_group),
        kT=np.float64(kT),
        fpt_dir=str(fpt),
        remd_dir=str(remd_dir),
    )
    print(
        f"[done] groups={n_groups}  last RMSE φ={rmse_last}  mean={rmse_last.mean():.4f}",
        flush=True,
    )
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
