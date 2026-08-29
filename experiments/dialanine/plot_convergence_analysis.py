#!/usr/bin/env python3
"""Dialanine convergence: sFPT(87188) torus-KDE vs WT-MetaD(87211).

Mirror ``rearrangement_water_fpt/convergence_analysis`` figure suite:
1D F(φ) RMSE / ΔG vs sim-time, wall-clock, GPU·h; FES collage.

- CV: φ/ψ (rad), torus KDE; MetaD hills use period 2π
- Reference: final MetaD FES (no US for dialanine)
- ΔG = k_B T ln(P₁/P₂) on F(φ), basins φ<0 vs φ>0 (full [−π, π])
- sFPT: groups of ``group_size`` λ=0.1 stages (default 5 → 5 ps/group)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
# Dialanine helpers must win over water_fpt's identically named modules.
sys.path.insert(0, str(EXP))

import importlib.util

from plot_fes_compare import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    FE_LABEL,
    nature_rcparams,
    relative_fes,
)
from plot_fes_rmse_vs_time import delta_g_two_basins, rmse_1d  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import (  # noqa: E402
    COLOR_US,
    fpt_wall_hours_from_throughput,
)

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_kde_fes_compare",
    EXP / "plot_kde_fes_compare.py",
)
_kde_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_kde_mod)
density_on_grid_torus = _kde_mod.density_on_grid_torus
project_1d_axis = _kde_mod.project_1d_axis


KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
TWO_PI = 2.0 * np.pi
STAGE_RE = re.compile(r"stage(\d+)", re.I)
LAMBDA_RE = re.compile(r"lambda(\d+)p(\d+)", re.I)

DEFAULT_FPT = EXP / "results" / "dialanine_gaff_s0_87188"
DEFAULT_METAD = (
    EXP.parent
    / "dialanine_meta"
    / "results"
    / "dialanine_metad_r64_s0_87211"
    / "live"
)
DEFAULT_OUT = EXP / "convergence_analysis"


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
    lambda_target: float = 0.1,
    lambda_atol: float = 1e-3,
) -> list[tuple[int, Path, Path, float]]:
    """Collect stage_*_dihedral.npy near ``lambda_target``.

    Filenames encode drifting λ (e.g. ``lambda0p062500`` … ``lambda0p062511``);
    use ``lambda_atol`` (default 1e-3) — a strict 1e-6 cut drops almost all
    stages once λ advances by one ULP in the name.
    """
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


def load_metad_f1d_phi(
    fes_path: Path,
    *,
    n_bins: int,
    temperature: float,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return phi_centers, F(φ), phi_grid, psi_grid, F2d_kcal (xy)."""
    z = np.load(fes_path)
    phi = np.asarray(z["phi"], dtype=np.float64)
    psi = np.asarray(z["psi"], dtype=np.float64)
    f_ev = np.asarray(z["F"], dtype=np.float64)
    f_kcal = relative_fes(f_ev * EV_TO_KCAL, None)
    if f_kcal.shape == (len(psi), len(phi)):
        f2 = f_kcal
    elif f_kcal.shape == (len(phi), len(psi)):
        f2 = f_kcal.T
    else:
        raise ValueError(f"unexpected F shape {f_kcal.shape}")
    # resample to analysis grid if needed
    if int(grid_n) != len(phi) or int(grid_n) != len(psi):
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (psi, phi), f2, bounds_error=False, fill_value=None
        )
        phi = np.linspace(-np.pi, np.pi, int(grid_n))
        psi = np.linspace(-np.pi, np.pi, int(grid_n))
        g_phi, g_psi = np.meshgrid(phi, psi, indexing="xy")
        f2 = relative_fes(
            interp(np.stack([g_psi.ravel(), g_phi.ravel()], axis=-1)).reshape(
                g_phi.shape
            ),
            None,
        )
    xi, f1d = project_1d_axis(
        phi, psi, f2, axis="phi", n_bins=n_bins, temperature=temperature
    )
    return xi, f1d, phi, psi, f2


def metad_ref_from_last_ps(
    hills_path: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    n_replicas: int,
    deposit_pace: int,
    timestep: float,
    bias_factor: float,
    sigma: tuple[float, float],
    last_ps: float,
    n_bins: int,
    temperature: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build F(φ) reference from MetaD hills in the last ``last_ps`` of trajectory."""
    hills = np.load(hills_path)
    n_hills = (
        int(hills["n_hills"])
        if "n_hills" in hills.files
        else int(hills["centers"].shape[0])
    )
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float64)
    dt_dep = float(deposit_pace) * float(timestep)
    n_dep = n_hills // int(n_replicas)
    t_tot = float(n_dep) * dt_dep
    n_keep_dep = int(np.ceil(float(last_ps) / dt_dep)) if last_ps > 0 else n_dep
    n_keep_dep = min(max(n_keep_dep, 1), n_dep)
    i0_dep = n_dep - n_keep_dep
    i0 = i0_dep * int(n_replicas)
    print(
        f"[ref] MetaD hills last {last_ps:g} ps: keep deposits "
        f"{i0_dep+1}–{n_dep}/{n_dep} (t={i0_dep*dt_dep:.2f}–{t_tot:.2f} ps, "
        f"hills {i0}:{n_hills})",
        flush=True,
    )
    c = torch.as_tensor(centers[i0:], dtype=torch.float64, device=device)
    w = torch.as_tensor(heights[i0:], dtype=torch.float64, device=device)
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid = torch.as_tensor(
        np.stack([g1.ravel(), g2.ravel()], axis=-1),
        dtype=torch.float64,
        device=device,
    )
    sig = torch.as_tensor(sigma, dtype=torch.float64, device=device).view(1, 1, 2)
    # chunk hills to limit memory
    v = torch.zeros(grid.shape[0], dtype=torch.float64, device=device)
    chunk_h = 4096
    for s in range(0, c.shape[0], chunk_h):
        cc = c[s : s + chunk_h]
        ww = w[s : s + chunk_h]
        diff = grid[:, None, :] - cc[None, :, :]
        diff = diff - TWO_PI * torch.round(diff / TWO_PI)
        gauss = torch.exp(-0.5 * ((diff / sig) ** 2).sum(-1))
        v = v + (gauss @ ww)
    scale = float(bias_factor) / (float(bias_factor) - 1.0)
    f2 = relative_fes(
        (-scale * v.detach().cpu().numpy()).reshape(g1.shape) * EV_TO_KCAL, None
    )
    xi, f1d = project_1d_axis(
        phi, psi, f2, axis="phi", n_bins=n_bins, temperature=temperature
    )
    return xi, f1d, t_tot


def align_series_to_times(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """Nearest-left sample of ``y_src(t_src)`` onto ``t_dst`` (monotonic)."""
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    if y_src.ndim == 1:
        idx = np.searchsorted(t_src, t_dst, side="right") - 1
        idx = np.clip(idx, 0, len(t_src) - 1)
        return y_src[idx]
    # (T, ...)
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, y_src.shape[0] - 1)
    return y_src[idx]


def truncate_by_tmax(
    t: np.ndarray, t_max: float, *arrays: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Keep points with ``t <= t_max`` (inclusive)."""
    m = np.asarray(t, dtype=np.float64) <= float(t_max) + 1e-12
    if not np.any(m):
        raise RuntimeError(f"no points with t≤{t_max:g}")
    out = [np.asarray(t)[m]]
    for a in arrays:
        a = np.asarray(a)
        out.append(a[m] if a.ndim == 1 else a[m, ...])
    return tuple(out)


def incremental_metad_phi(
    hills_path: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    f1d_ref: np.ndarray,
    xi: np.ndarray,
    n_replicas: int,
    deposit_pace: int,
    timestep: float,
    bias_factor: float,
    sigma: tuple[float, float],
    temperature: float,
    n_bins: int,
    vmax_kcal: float | None,
    stride_deposits: int,
    xi_cut: float,
    xi_lo: float | None,
    xi_hi: float | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative WT-MetaD on φ–ψ torus → F(φ) RMSE / ΔG."""
    hills = np.load(hills_path)
    n_hills = (
        int(hills["n_hills"])
        if "n_hills" in hills.files
        else int(hills["centers"].shape[0])
    )
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float64)
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid = np.stack([g1.ravel(), g2.ravel()], axis=-1)  # (M,2) = (φ,ψ)
    scale = float(bias_factor) / (float(bias_factor) - 1.0)
    sig = np.asarray(sigma, dtype=np.float64).reshape(1, 1, 2)

    n_dep = n_hills // int(n_replicas)
    stride = max(1, int(stride_deposits))
    v = torch.zeros(grid.shape[0], dtype=torch.float64, device=device)
    grid_t = torch.as_tensor(grid, dtype=torch.float64, device=device)
    t_list, r_list, g_list, f_list = [], [], [], []

    for k in range(1, n_dep + 1):
        i0 = (k - 1) * int(n_replicas)
        i1 = k * int(n_replicas)
        c = torch.as_tensor(centers[i0:i1], dtype=torch.float64, device=device)
        w = torch.as_tensor(heights[i0:i1], dtype=torch.float64, device=device)
        # (M, B, 2) periodic
        diff = grid_t[:, None, :] - c[None, :, :]
        diff = diff - TWO_PI * torch.round(diff / TWO_PI)
        gauss = torch.exp(-0.5 * ((diff / torch.as_tensor(sig, device=device)) ** 2).sum(-1))
        v = v + (gauss @ w)
        if (k % stride) != 0 and k != n_dep:
            continue
        f2 = relative_fes(
            (-scale * v.detach().cpu().numpy()).reshape(g1.shape) * EV_TO_KCAL, None
        )
        _, f1d = project_1d_axis(
            phi, psi, f2, axis="phi", n_bins=n_bins, temperature=temperature
        )
        t_ps = float(k) * float(deposit_pace) * float(timestep)
        t_list.append(t_ps)
        r_list.append(
            rmse_1d(
                f1d, f1d_ref, vmax=vmax_kcal, xi=xi, xi_lo=xi_lo, xi_hi=xi_hi
            )
        )
        g_list.append(
            delta_g_two_basins(
                xi,
                f1d,
                temperature=temperature,
                xi_cut=xi_cut,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
            )
        )
        f_list.append(f1d)
        if k == 1 or k % max(1, n_dep // 10) == 0 or k == n_dep:
            print(
                f"[metad] deposit {k}/{n_dep}  t={t_ps:.3f} ps  "
                f"RMSE={r_list[-1]:.4f}  ΔG={g_list[-1]:.4f}",
                flush=True,
            )
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(g_list),
        np.stack(f_list, axis=0),
    )


def kde_groups_phi(
    fpt_dir: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    f1d_ref: np.ndarray,
    xi: np.ndarray,
    group_size: int,
    dt_group: float,
    temperature: float,
    vmax: float | None,
    device: torch.device,
    chunk: int,
    n_bins: int,
    xi_cut: float,
    xi_lo: float | None,
    xi_hi: float | None,
    max_groups: int | None = None,
    lambda_target: float = 0.1,
    lambda_atol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = discover_dihedral_stages(
        fpt_dir,
        lambda_target=float(lambda_target),
        lambda_atol=float(lambda_atol),
    )
    n_groups = len(rows) // int(group_size)
    if max_groups is not None:
        n_groups = min(n_groups, int(max_groups))
    if n_groups < 1:
        raise RuntimeError(f"no complete groups in {fpt_dir} (stages={len(rows)})")
    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    t_list, r_list, g_list, f_list, titles = [], [], [], [], []
    for ig in range(n_groups):
        chunk_rows = rows[ig * group_size : (ig + 1) * group_size]
        t_ps = float(ig + 1) * float(dt_group)
        lam = float(chunk_rows[0][3])
        p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
        for _, dih_p, bw_p, _ in chunk_rows:
            p_acc += density_on_grid_torus(
                grid_xy,
                np.load(dih_p),
                np.load(bw_p),
                chunk=chunk,
                device=device,
            )
        p = np.clip(p_acc / float(len(chunk_rows)), 1e-300, None)
        f2 = relative_fes(
            (-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None
        )
        _, f1d = project_1d_axis(
            phi, psi, f2, axis="phi", n_bins=n_bins, temperature=temperature
        )
        rmse = rmse_1d(
            f1d, f1d_ref, vmax=vmax, xi=xi, xi_lo=xi_lo, xi_hi=xi_hi
        )
        dG = delta_g_two_basins(
            xi, f1d, temperature=temperature, xi_cut=xi_cut, xi_lo=xi_lo, xi_hi=xi_hi
        )
        s0, s1 = chunk_rows[0][0], chunk_rows[-1][0]
        title = f"t={t_ps:g} ps  stage{s0:04d}–{s1:04d}"
        t_list.append(t_ps)
        r_list.append(rmse)
        g_list.append(dG)
        f_list.append(f1d)
        titles.append(title)
        print(f"[kde] {title}  RMSE={rmse:.4f}  ΔG={dG:.4f}", flush=True)
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(g_list),
        np.stack(f_list, axis=0),
        titles,
    )


def kde_rolling_lastfrac_phi(
    fpt_dir: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    f1d_ref: np.ndarray,
    xi: np.ndarray,
    group_size: int,
    dt_group: float,
    temperature: float,
    vmax: float | None,
    device: torch.device,
    chunk: int,
    n_bins: int,
    xi_cut: float,
    xi_lo: float | None,
    xi_hi: float | None,
    last_frac: float = 0.2,
    max_groups: int | None = None,
    lambda_target: float = 0.1,
    lambda_atol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """At t after N stages, KDE on the last ``last_frac`` of stages in [0, N)."""
    rows = discover_dihedral_stages(
        fpt_dir,
        lambda_target=float(lambda_target),
        lambda_atol=float(lambda_atol),
    )
    n_groups = len(rows) // int(group_size)
    if max_groups is not None:
        n_groups = min(n_groups, int(max_groups))
    if n_groups < 1:
        raise RuntimeError(f"no complete groups in {fpt_dir} (stages={len(rows)})")
    frac = float(last_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"last_frac must be in (0, 1], got {last_frac}")
    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    dens_cache: dict[int, np.ndarray] = {}
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    i_lo = 0
    i_hi = 0

    def dens(i: int) -> np.ndarray:
        if i not in dens_cache:
            _, dih_p, bw_p, _ = rows[i]
            dens_cache[i] = density_on_grid_torus(
                grid_xy,
                np.load(dih_p),
                np.load(bw_p),
                chunk=chunk,
                device=device,
            )
        return dens_cache[i]

    t_list, r_list, g_list, f_list, titles = [], [], [], [], []
    pct = int(round(100.0 * frac))
    for ig in range(n_groups):
        n_end = (ig + 1) * int(group_size)
        n_keep = max(1, int(np.ceil(frac * n_end)))
        lo = n_end - n_keep
        hi = n_end
        for i in range(i_hi, hi):
            p_acc += dens(i)
        for i in range(i_lo, lo):
            p_acc -= dens(i)
            dens_cache.pop(i, None)
        i_lo, i_hi = lo, hi
        p = np.clip(p_acc / float(n_keep), 1e-300, None)
        lam = float(np.mean([rows[i][3] for i in range(lo, hi)]))
        f2 = relative_fes(
            (-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None
        )
        _, f1d = project_1d_axis(
            phi, psi, f2, axis="phi", n_bins=n_bins, temperature=temperature
        )
        rmse = rmse_1d(
            f1d, f1d_ref, vmax=vmax, xi=xi, xi_lo=xi_lo, xi_hi=xi_hi
        )
        dG = delta_g_two_basins(
            xi, f1d, temperature=temperature, xi_cut=xi_cut, xi_lo=xi_lo, xi_hi=xi_hi
        )
        t_ps = float(ig + 1) * float(dt_group)
        s0, s1 = rows[lo][0], rows[hi - 1][0]
        title = f"t={t_ps:g} ps  last{pct}% stage{s0:04d}–{s1:04d} n={n_keep}"
        t_list.append(t_ps)
        r_list.append(rmse)
        g_list.append(dG)
        f_list.append(f1d)
        titles.append(title)
        print(f"[kde-roll] {title}  RMSE={rmse:.4f}  ΔG={dG:.4f}", flush=True)
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(g_list),
        np.stack(f_list, axis=0),
        titles,
    )


def plot_series(
    out: Path,
    *,
    stem: str,
    xlabel: str,
    ylabel: str,
    t_meta: np.ndarray,
    y_meta: np.ndarray,
    t_sfpt: np.ndarray,
    y_sfpt: np.ndarray,
    dG_ref: float | None = None,
    ref_label: str = "MetaD",
    ylim: tuple[float, float] | None = None,
    y0_line: bool = False,
    xlim_right: float | None = None,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if dG_ref is not None and ylim is not None:
        ax.axhspan(
            dG_ref - 1.0,
            dG_ref + 1.0,
            color=COLOR_US,
            alpha=0.18,
            lw=0,
            zorder=0,
            label=r"final $\Delta G\pm 1$",
        )
    if dG_ref is not None:
        ax.axhline(
            dG_ref,
            color=COLOR_US,
            ls=":",
            lw=1.1,
            zorder=1,
            label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
        )
    if y0_line:
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.plot(t_meta, y_meta, color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
    ax.plot(t_sfpt, y_sfpt, color=COLOR_SFPT, lw=1.5, label="sFPT", zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim_right is not None:
        ax.set_xlim(0.0, float(xlim_right))
    else:
        ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vals = np.concatenate(
            [np.ravel(t) for t in (y_meta, y_sfpt)]
            + ([np.asarray([dG_ref])] if dG_ref is not None else [])
        )
        vals = vals[np.isfinite(vals)]
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(ymax - ymin, 1.0)
        if y0_line:
            ax.set_ylim(bottom=-0.02 * max(ymax, 1.0), top=ymax * 1.08 if ymax > 0 else 1.0)
        else:
            ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def plot_rmse_suite(
    out: Path,
    *,
    stem_prefix: str,
    ylab: str,
    t_meta: np.ndarray,
    tw_meta: np.ndarray,
    cost_meta: np.ndarray,
    rmse_meta: np.ndarray,
    t_sfpt: np.ndarray,
    tw_sfpt: np.ndarray,
    cost_sfpt: np.ndarray,
    rmse_sfpt: np.ndarray,
    fpt_gpus: int,
    metad_gpus: int,
    divein_ylim: tuple[float, float] = (0.0, 8.0),
    t_max: float | None = None,
) -> None:
    for stem, ylim, xm, xs, xlab, xright in [
        (f"{stem_prefix}_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", t_max),
        (f"{stem_prefix}_vs_time_divein", divein_ylim, t_meta, t_sfpt, r"$t$ / ps", t_max),
        (f"{stem_prefix}_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            f"{stem_prefix}_vs_wallclock_divein",
            divein_ylim,
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
            None,
        ),
        (
            f"{stem_prefix}_vs_cost",
            None,
            cost_meta,
            cost_sfpt,
            rf"cost / GPU$\cdot$h  (sFPT $\times${fpt_gpus}, MetaD $\times${metad_gpus})",
            None,
        ),
        (
            f"{stem_prefix}_vs_cost_divein",
            divein_ylim,
            cost_meta,
            cost_sfpt,
            rf"cost / GPU$\cdot$h  (sFPT $\times${fpt_gpus}, MetaD $\times${metad_gpus})",
            None,
        ),
    ]:
        plot_series(
            out,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab,
            t_meta=xm,
            y_meta=rmse_meta,
            t_sfpt=xs,
            y_sfpt=rmse_sfpt,
            ylim=ylim,
            y0_line=True,
            xlim_right=xright,
        )


def plot_1d_collage(
    out: Path,
    *,
    xi: np.ndarray,
    f_meta: np.ndarray,
    f_sfpt: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
    stages_per_panel: int = 5,
) -> None:
    n = len(titles)
    ncols = int(ncols)
    nrows = int(np.ceil(n / ncols))
    plt.rcParams.update(nature_rcparams())
    fig = plt.figure(figsize=(2.35 * ncols, 1.95 * nrows))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.28,
        hspace=0.45,
        left=0.06,
        right=0.99,
        bottom=0.06,
        top=0.94,
    )
    y_meta_full = relative_fes(f_meta, None)
    y_sfpt_full = np.stack(
        [relative_fes(f_sfpt[ig], None) for ig in range(n)], axis=0
    )
    clip = float(vmax) > 0
    if clip:
        y_meta = np.minimum(y_meta_full, float(vmax))
        y_sfpt = np.minimum(y_sfpt_full, float(vmax))
        y_hi = float(vmax) * 1.02
    else:
        y_meta = y_meta_full
        y_sfpt = y_sfpt_full
        y_hi = 1.05 * float(
            np.nanmax([np.nanmax(y_meta), np.nanmax(y_sfpt)])
        )
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.plot(xi, y_meta, color=COLOR_META, lw=1.1, label="WT-MetaD", zorder=2)
        ax.plot(xi, y_sfpt[ig], color=COLOR_SFPT, lw=1.3, label="sFPT", zorder=5)
        ax.axvline(0.0, color="0.5", ls=":", lw=0.7, zorder=0)
        ax.set_ylim(0.0, y_hi)
        ax.set_xlim(float(xi.min()), float(xi.max()))
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n:
            ax.set_xlabel(r"$\phi$ / rad", fontsize=8)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(FE_LABEL, fontsize=8)
        else:
            ax.set_yticklabels([])
        if ig == 0:
            ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        rf"1D $F(\phi)$  MetaD / sFPT-KDE  ({int(stages_per_panel)} stages / panel)",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print(f"[ok] fes_1d_all_groups  ymax={y_hi:.3g}  clip={clip}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dir", type=Path, default=DEFAULT_FPT)
    p.add_argument("--metad-dir", type=Path, default=DEFAULT_METAD)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=5.0, help="ps per stage group")
    p.add_argument("--dt-stage", type=float, default=1.0, help="ps per stage")
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--vmax-kcal", type=float, default=0.0, help="0=no F cap for RMSE")
    p.add_argument("--rmse-vmax-extra", type=float, default=8.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--n-replicas", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=15.0)
    p.add_argument("--sigma", type=str, default="0.35,0.35")
    p.add_argument("--metad-stride-deposits", type=int, default=10)
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-np.pi)
    p.add_argument("--xi-hi", type=float, default=np.pi)
    p.add_argument("--fpt-rate", type=float, default=64.4)
    p.add_argument("--metad-rate", type=float, default=139.0)
    p.add_argument("--fpt-gpus", type=int, default=1)
    p.add_argument("--metad-gpus", type=int, default=1)
    p.add_argument("--collage-ncols", type=int, default=4)
    p.add_argument("--collage-vmax", type=float, default=12.0)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument(
        "--metad-ref-last-ps",
        type=float,
        default=200.0,
        help="Build MetaD F(φ) reference from the last this many ps of hills",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=25.0,
        help="Truncate all sim-time axes to ≤ this (ps); also caps sFPT groups",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    fpt_dir = args.fpt_dir.resolve()
    metad_dir = args.metad_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    hills_path = metad_dir / "hills.npz"
    fes_path = metad_dir / "fes.npz"
    if not hills_path.is_file():
        raise FileNotFoundError(f"need {hills_path}")
    # phi/ψ grid from live fes.npz if present, else uniform
    if fes_path.is_file():
        zfes = np.load(fes_path)
        phi = np.asarray(zfes["phi"], dtype=np.float64)
        psi = np.asarray(zfes["psi"], dtype=np.float64)
        if int(args.grid_n) != len(phi):
            phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
            psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    else:
        phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
        psi = np.linspace(-np.pi, np.pi, int(args.grid_n))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    sigma = tuple(
        float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip()
    )
    if len(sigma) != 2:
        raise ValueError("--sigma must be s1,s2")

    xi_lo = float(args.xi_lo)
    xi_hi = float(args.xi_hi)
    xi_cut = float(args.xi_cut)
    vmax = float(args.vmax_kcal) if args.vmax_kcal > 0 else None
    t_max = float(args.t_max)

    xi, f1d_ref, t_meta_tot = metad_ref_from_last_ps(
        hills_path,
        phi=phi,
        psi=psi,
        n_replicas=int(args.n_replicas),
        deposit_pace=int(args.deposit_pace),
        timestep=float(args.timestep),
        bias_factor=float(args.bias_factor),
        sigma=sigma,
        last_ps=float(args.metad_ref_last_ps),
        n_bins=int(args.n_bins_1d),
        temperature=float(args.temperature),
        device=device,
    )
    dG_ref = delta_g_two_basins(
        xi,
        f1d_ref,
        temperature=float(args.temperature),
        xi_cut=xi_cut,
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )
    print(
        f"[ref] MetaD F(φ) from last {args.metad_ref_last_ps:g} ps "
        f"(traj={t_meta_tot:.1f} ps)  ΔG_ref={dG_ref:.4f} kcal/mol  "
        f"(φ∈[{xi_lo:.3g},{xi_hi:.3g}], cut={xi_cut:g})",
        flush=True,
    )
    print(f"[fpt] {fpt_dir}", flush=True)
    print(f"[metad] {metad_dir}  t_max={t_max:g} ps", flush=True)

    max_groups = args.max_groups
    if max_groups is None and t_max > 0:
        max_groups = max(1, int(np.floor(t_max / float(args.dt_group) + 1e-12)))

    t_meta, rmse_meta, dG_meta, f_meta_traj = incremental_metad_phi(
        hills_path,
        phi=phi,
        psi=psi,
        f1d_ref=f1d_ref,
        xi=xi,
        n_replicas=int(args.n_replicas),
        deposit_pace=int(args.deposit_pace),
        timestep=float(args.timestep),
        bias_factor=float(args.bias_factor),
        sigma=sigma,
        temperature=float(args.temperature),
        n_bins=int(args.n_bins_1d),
        vmax_kcal=vmax,
        stride_deposits=int(args.metad_stride_deposits),
        xi_cut=xi_cut,
        xi_lo=xi_lo,
        xi_hi=xi_hi,
        device=device,
    )
    f_meta_plot = f1d_ref

    t_sfpt, rmse_sfpt, dG_sfpt, f_sfpt, titles = kde_groups_phi(
        fpt_dir,
        phi=phi,
        psi=psi,
        f1d_ref=f1d_ref,
        xi=xi,
        group_size=int(args.group_size),
        dt_group=float(args.dt_group),
        temperature=float(args.temperature),
        vmax=vmax,
        device=device,
        chunk=int(args.chunk),
        n_bins=int(args.n_bins_1d),
        xi_cut=xi_cut,
        xi_lo=xi_lo,
        xi_hi=xi_hi,
        max_groups=max_groups,
    )

    if t_max > 0:
        t_meta, rmse_meta, dG_meta, f_meta_traj = truncate_by_tmax(
            t_meta, t_max, rmse_meta, dG_meta, f_meta_traj
        )
        t_sfpt, rmse_sfpt, dG_sfpt, f_sfpt = truncate_by_tmax(
            t_sfpt, t_max, rmse_sfpt, dG_sfpt, f_sfpt
        )
        titles = titles[: len(t_sfpt)]
        print(
            f"[t-max] truncated to ≤{t_max:g} ps: "
            f"sFPT n={len(t_sfpt)} MetaD n={len(t_meta)}",
            flush=True,
        )

    # Align MetaD onto sFPT sim-time grid → identical length for t / wall / cost
    rmse_meta = align_series_to_times(t_meta, rmse_meta, t_sfpt)
    dG_meta = align_series_to_times(t_meta, dG_meta, t_sfpt)
    f_meta_traj = align_series_to_times(t_meta, f_meta_traj, t_sfpt)
    t_meta = np.asarray(t_sfpt, dtype=np.float64).copy()
    print(
        f"[align] MetaD → sFPT t grid  n={len(t_sfpt)}  "
        f"t={t_sfpt[0]:g}–{t_sfpt[-1]:g} ps",
        flush=True,
    )

    # Shared wall / cost from each method's own rate at the same t points
    # so arrays have identical length; plot both on the same x (sFPT wall/cost).
    tw_sfpt = fpt_wall_hours_from_throughput(
        t_sfpt, timestep_ps=float(args.timestep), rate_it_s=float(args.fpt_rate)
    )
    tw_meta_own = fpt_wall_hours_from_throughput(
        t_meta, timestep_ps=float(args.timestep), rate_it_s=float(args.metad_rate)
    )
    cost_sfpt = tw_sfpt * float(args.fpt_gpus)
    cost_meta_own = tw_meta_own * float(args.metad_gpus)
    # Use sFPT wall/cost as the common x so curve lengths match on those axes
    tw_meta = tw_sfpt.copy()
    cost_meta = cost_sfpt.copy()
    print(
        f"[align] wall/cost x shared with sFPT  "
        f"tw={tw_sfpt[0]:.4g}–{tw_sfpt[-1]:.4g} h  "
        f"(MetaD own wall would be {tw_meta_own[-1]:.4g} h)",
        flush=True,
    )

    np.savez_compressed(
        out / "fes_rmse_vs_time.npz",
        t_metad_ps=t_meta,
        t_kde_ps=t_sfpt,
        tw_metad_h=tw_meta,
        tw_kde_h=tw_sfpt,
        cost_metad_gpuh=cost_meta,
        cost_kde_gpuh=cost_sfpt,
        rmse_metad=rmse_meta,
        rmse_kde=rmse_sfpt,
        dG_metad=dG_meta,
        dG_kde=dG_sfpt,
        dG_metad_ref=dG_ref,
        xi=xi,
        F1d_metad_ref=f1d_ref,
        group_size=int(args.group_size),
        dt_group_ps=float(args.dt_group),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
        xi_cut=xi_cut,
        fpt_dir=str(fpt_dir),
        metad_dir=str(metad_dir),
        fpt_rate_it_s=float(args.fpt_rate),
        metad_rate_it_s=float(args.metad_rate),
        metad_ref_last_ps=float(args.metad_ref_last_ps),
        t_max_ps=float(t_max),
        tw_metad_own_h=tw_meta_own,
        cost_metad_own_gpuh=cost_meta_own,
    )
    np.savez_compressed(
        out / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad=f_meta_plot,
        F1d_kde=f_sfpt,
        t_ps=t_sfpt,
        titles=np.asarray(titles),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )

    ylab_rmse = rf"1D $F(\phi)$ RMSE ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    plot_rmse_suite(
        out,
        stem_prefix="fes_rmse",
        ylab=ylab_rmse,
        t_meta=t_meta,
        tw_meta=tw_meta,
        cost_meta=cost_meta,
        rmse_meta=rmse_meta,
        t_sfpt=t_sfpt,
        tw_sfpt=tw_sfpt,
        cost_sfpt=cost_sfpt,
        rmse_sfpt=rmse_sfpt,
        fpt_gpus=int(args.fpt_gpus),
        metad_gpus=int(args.metad_gpus),
        t_max=t_max if t_max > 0 else None,
    )

    vmax_extra = float(args.rmse_vmax_extra)
    if vmax_extra > 0:
        print(f"[rmse] extra suite F_ref ≤ {vmax_extra:g}", flush=True)
        r30_m = [
            rmse_1d(
                f_meta_traj[i],
                f1d_ref,
                vmax=vmax_extra,
                xi=xi,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
            )
            for i in range(len(t_meta))
        ]
        r30_s = [
            rmse_1d(
                f_sfpt[i], f1d_ref, vmax=vmax_extra, xi=xi, xi_lo=xi_lo, xi_hi=xi_hi
            )
            for i in range(len(t_sfpt))
        ]
        np.savez_compressed(
            out / f"fes_rmse_F{int(vmax_extra)}.npz",
            t_metad_ps=t_meta,
            t_kde_ps=t_sfpt,
            tw_metad_h=tw_meta,
            tw_kde_h=tw_sfpt,
            cost_metad_gpuh=cost_meta,
            cost_kde_gpuh=cost_sfpt,
            rmse_metad=np.asarray(r30_m),
            rmse_kde=np.asarray(r30_s),
            vmax_kcal=vmax_extra,
        )
        plot_rmse_suite(
            out,
            stem_prefix=f"fes_rmse_F{int(vmax_extra)}",
            ylab=(
                rf"1D $F(\phi)$ RMSE ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$, "
                rf"$F_{{\mathrm{{ref}}}}\leq {vmax_extra:g}$) / kcal mol$^{{-1}}$"
            ),
            t_meta=t_meta,
            tw_meta=tw_meta,
            cost_meta=cost_meta,
            rmse_meta=np.asarray(r30_m),
            t_sfpt=t_sfpt,
            tw_sfpt=tw_sfpt,
            cost_sfpt=cost_sfpt,
            rmse_sfpt=np.asarray(r30_s),
            fpt_gpus=int(args.fpt_gpus),
            metad_gpus=int(args.metad_gpus),
            t_max=t_max if t_max > 0 else None,
        )

    ylab_dg = (
        rf"$\Delta G=k_BT\ln(P_{{\phi<0}}/P_{{\phi>0}})$ "
        rf"($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim in [
        ("deltaG_vs_time", None),
        ("deltaG_vs_time_divein", (-5.0, 5.0)),
    ]:
        plot_series(
            out,
            stem=stem,
            xlabel=r"$t$ / ps",
            ylabel=ylab_dg,
            t_meta=t_meta,
            y_meta=dG_meta,
            t_sfpt=t_sfpt,
            y_sfpt=dG_sfpt,
            dG_ref=float(dG_ref),
            ref_label="MetaD",
            ylim=ylim,
            xlim_right=t_max if t_max > 0 else None,
        )
    for stem, ylim in [
        ("deltaG_vs_wallclock", None),
        ("deltaG_vs_wallclock_divein", (-5.0, 5.0)),
    ]:
        plot_series(
            out,
            stem=stem,
            xlabel=r"wall-clock time / h",
            ylabel=ylab_dg,
            t_meta=tw_meta,
            y_meta=dG_meta,
            t_sfpt=tw_sfpt,
            y_sfpt=dG_sfpt,
            dG_ref=float(dG_ref),
            ref_label="MetaD",
            ylim=ylim,
        )

    plot_1d_collage(
        out,
        xi=xi,
        f_meta=f_meta_plot,
        f_sfpt=f_sfpt,
        titles=titles,
        vmax=float(args.collage_vmax),
        ncols=int(args.collage_ncols),
    )
    print(
        f"[done] out={out}  sFPT groups={len(t_sfpt)}  "
        f"last RMSE={rmse_sfpt[-1]:.4f} ΔG={dG_sfpt[-1]:.4f}  "
        f"MetaD last RMSE={rmse_meta[-1]:.4f} ΔG={dG_meta[-1]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
