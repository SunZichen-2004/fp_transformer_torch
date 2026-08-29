#!/usr/bin/env python3
"""Tetra-alanine 3φ convergence: 3 sFPT seeds vs 3 WT-MetaD seeds.

ΔG = k_B T ln(P_A / P_B),  A = {φ1<0, φ2<0, φ3<0}.
Reference = last WT-MetaD FES (seed-averaged). Curves are mean ± std.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
META_EXP = EXP.parent / "tetra_alanine_meta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(EXP))

from fes_window import density_on_grid_torus, relative_fes  # noqa: E402
from plot_fes_compare import FE_LABEL, nature_rcparams  # noqa: E402
from plot_fes_rmse_vs_time import rmse_1d  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import COLOR_US, fpt_wall_hours_from_throughput  # noqa: E402

COLOR_META = "#2166AC"
COLOR_SFPT = "#E67E22"
KT_KCAL = 0.00198720425864083 * 300.0
EV_TO_KCAL = 23.06054783061903
TWO_PI = 2.0 * np.pi
T_RE = re.compile(r"fes_2d_t(\d+)ps\.npz$")
STAGE_RE = re.compile(r"stage(\d+)", re.I)

DEFAULT_FPT_DIRS = [
    EXP / "results" / "tetra_alanine_phi3_r1024_s1_94120",
    EXP / "results" / "tetra_alanine_phi3_r1024_s2_94121",
    EXP / "results" / "tetra_alanine_phi3_r1024_s3_94122",
]
DEFAULT_META_DIRS = [
    META_EXP / "results" / "tetra_alanine_metad_bspline_r64_s0_94242",
    META_EXP / "results" / "tetra_alanine_metad_bspline_r64_s1_94243",
    META_EXP / "results" / "tetra_alanine_metad_bspline_r64_s2_94244",
]
DEFAULT_OUT = EXP / "convergence_analysis"

# 94120-family ~88.8 it/s.  3φ B-spline MetaD ~144 it/s (H800, R=64).
SFPT_IT_S = 88.8
META_IT_S = 144.0


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def discover_fes_npz(fes_dir: Path) -> list[Path]:
    rows = []
    for p in fes_dir.glob("fes_2d_t*ps.npz"):
        m = T_RE.search(p.name)
        if m:
            rows.append((int(m.group(1)), p))
    rows.sort()
    return [p for _, p in rows]


def discover_stages(data_dir: Path) -> list[tuple[int, Path, Path]]:
    rows = []
    for dih in sorted(data_dir.glob("stage[0-9]*_lambda*_dihedral.npy")):
        m = STAGE_RE.search(dih.name)
        if m is None:
            continue
        bw = dih.with_name(dih.name.replace("_dihedral.npy", "_bandwidth.npy"))
        if not bw.is_file():
            continue
        rows.append((int(m.group(1)), dih, bw))
    rows.sort(key=lambda r: r[0])
    return rows


def delta_g_octant(F3: np.ndarray, ax: np.ndarray, *, kT: float) -> float:
    """ΔG = kT ln(P_A / P_other), A = all three φ < 0."""
    f = relative_fes(F3)
    g1, g2, g3 = np.meshgrid(ax, ax, ax, indexing="ij")
    w = np.exp(-np.clip(f, 0.0, 80.0) / float(kT))
    mask = (g1 < 0.0) & (g2 < 0.0) & (g3 < 0.0)
    z_a = float(np.sum(w[mask]))
    z_b = float(np.sum(w[~mask]))
    z_a = max(z_a, 1e-300)
    z_b = max(z_b, 1e-300)
    return float(kT) * float(np.log(z_a / z_b))


def project_1d_from_3d(F3: np.ndarray, *, kT: float) -> np.ndarray:
    """Boltzmann-project F(φ1,φ2,φ3) → F(φi), i=0,1,2."""
    w = np.exp(-np.clip(relative_fes(F3), 0.0, 80.0) / float(kT))
    out = np.zeros((3, w.shape[0]), dtype=np.float64)
    out[0] = relative_fes(-float(kT) * np.log(np.clip(w.sum(axis=(1, 2)), 1e-300, None)))
    out[1] = relative_fes(-float(kT) * np.log(np.clip(w.sum(axis=(0, 2)), 1e-300, None)))
    out[2] = relative_fes(-float(kT) * np.log(np.clip(w.sum(axis=(0, 1)), 1e-300, None)))
    return out


def interp_1d(ax_src: np.ndarray, f_src: np.ndarray, ax_dst: np.ndarray) -> np.ndarray:
    """Periodic linear interpolation on (−π, π)."""
    x = np.concatenate([ax_src - TWO_PI, ax_src, ax_src + TWO_PI])
    y = np.concatenate([f_src, f_src, f_src])
    return relative_fes(np.interp(ax_dst, x, y))


def mean_std(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0)
    std = np.nanstd(stack, axis=0, ddof=1)
    std = np.where(n <= 1, 0.0, std)
    return mean, std


def interp_series(t_src, y_src, t_dst) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    out = np.full(t_dst.shape, np.nan, dtype=np.float64)
    if t_src.size == 0:
        return out
    m = (t_dst + 1e-9 >= t_src[0]) & (t_dst - 1e-9 <= t_src[-1])
    if np.any(m):
        out[m] = np.interp(t_dst[m], t_src, y_src)
    return out


def stack_mean_std(
    series: list[tuple[np.ndarray, np.ndarray]], dt: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_max = max(float(t[-1]) for t, _ in series)
    t_common = np.arange(dt, t_max + 0.5 * dt, dt)
    stacked = np.stack([interp_series(t, y, t_common) for t, y in series], axis=0)
    mean, std = mean_std(stacked)
    return t_common, mean, std


def boltzmann_mean(fs: list[np.ndarray], *, kT: float = KT_KCAL) -> np.ndarray:
    w = np.mean(
        [np.exp(-np.clip(relative_fes(f), 0.0, 80.0) / float(kT)) for f in fs],
        axis=0,
    )
    return relative_fes(-float(kT) * np.log(np.clip(w, 1e-300, None)))


def snapshot_meta(run_dir: Path, snap_root: Path) -> Path:
    live = run_dir / "live"
    dst = snap_root / f"snap_{run_dir.name}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("fes_2d.npz", "hills.npz"):
        src = live / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst / name)
    return dst


def kde_3d_window(
    rows: list[tuple[np.ndarray, np.ndarray]],
    *,
    ax: np.ndarray,
    lam: float,
    kT: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    g1, g2, g3 = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)
    p_acc = np.zeros(grid.shape[0], dtype=np.float64)
    for cv, bw in rows:
        p_acc += density_on_grid_torus(
            grid, np.asarray(cv, dtype=np.float64), np.asarray(bw), chunk=chunk, device=device
        )
    p = np.clip(p_acc / float(len(rows)), 1e-300, None)
    return relative_fes((-np.log(p) / float(lam) * float(kT)).reshape(g1.shape))


def hist_3d_window(
    cvs: list[np.ndarray],
    *,
    ax: np.ndarray,
    lam: float,
    kT: float,
    smooth: float,
) -> np.ndarray:
    n = int(ax.shape[0])
    edges = np.linspace(-np.pi, np.pi, n + 1)
    pts = wrap_pi(np.concatenate([np.asarray(c, dtype=np.float64) for c in cvs], axis=0))
    counts, _ = np.histogramdd(pts, bins=[edges, edges, edges])
    counts = np.asarray(counts, dtype=np.float64)
    if float(smooth) > 0.0:
        counts = gaussian_filter(counts, sigma=float(smooth), mode="wrap")
    tot = float(np.sum(counts))
    p = np.clip(counts / max(tot, 1e-300), 1e-300, None)
    return relative_fes(-np.log(p) / float(lam) * float(kT))


def load_sfpt_1d(fes_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return t_ps (T,), ax (G,), F1 (T, 3, G), lam."""
    paths = discover_fes_npz(fes_dir)
    if not paths:
        raise FileNotFoundError(f"no fes_2d_t*ps.npz in {fes_dir}")
    t, f1 = [], []
    ax = lam = None
    for p in paths:
        z = np.load(p)
        t.append(float(z["t_ps"]))
        f1.append(
            np.stack(
                [
                    relative_fes(z["F1_phi1"]),
                    relative_fes(z["F1_phi2"]),
                    relative_fes(z["F1_phi3"]),
                ],
                axis=0,
            )
        )
        ax = np.asarray(z["ax"], dtype=np.float64)
        lam = float(z["lam"])
    return np.asarray(t), ax, np.stack(f1, axis=0), float(lam)


def incremental_metad_3d(
    hills_path: Path,
    *,
    ax: np.ndarray,
    n_replicas: int,
    deposit_pace: int,
    timestep: float,
    bias_factor: float,
    sigma: tuple[float, float, float],
    stride_deposits: int,
    kT_kcal: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return t_ps (T,), F1 (T, 3, n), dG (T,) from cumulative WT-MetaD hills."""
    hills = np.load(hills_path)
    n_hills = int(hills["n_hills"]) if "n_hills" in hills.files else int(hills["centers"].shape[0])
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float64)
    n = int(ax.shape[0])
    g1, g2, g3 = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)
    scale = float(bias_factor) / (float(bias_factor) - 1.0)
    sig = torch.as_tensor(sigma, dtype=torch.float64, device=device).view(1, 1, 3)
    n_dep = n_hills // int(n_replicas)
    stride = max(1, int(stride_deposits))
    v = torch.zeros(grid.shape[0], dtype=torch.float64, device=device)
    grid_t = torch.as_tensor(grid, dtype=torch.float64, device=device)
    t_list, f_list, g_list = [], [], []
    print(
        f"[metad] hills={n_hills}  deposits={n_dep}  grid={n}^3  stride={stride}  device={device}",
        flush=True,
    )
    chunk_h = 512
    k_prev = 0
    snapshots = list(range(stride, n_dep + 1, stride))
    if not snapshots or snapshots[-1] != n_dep:
        snapshots.append(n_dep)
    for k in snapshots:
        i0 = k_prev * int(n_replicas)
        i1 = k * int(n_replicas)
        for s in range(i0, i1, chunk_h):
            e = min(s + chunk_h, i1)
            c = torch.as_tensor(centers[s:e], dtype=torch.float64, device=device)
            w = torch.as_tensor(heights[s:e], dtype=torch.float64, device=device)
            diff = grid_t[:, None, :] - c[None, :, :]
            diff = diff - TWO_PI * torch.round(diff / TWO_PI)
            gauss = torch.exp(-0.5 * ((diff / sig) ** 2).sum(-1))
            v = v + (gauss @ w)
        f3 = relative_fes(
            (-scale * v.detach().cpu().numpy()).reshape(n, n, n) * EV_TO_KCAL
        )
        t_ps = float(k) * float(deposit_pace) * float(timestep)
        t_list.append(t_ps)
        f_list.append(project_1d_from_3d(f3, kT=kT_kcal))
        g_list.append(delta_g_octant(f3, ax, kT=kT_kcal))
        k_prev = k
        if k == snapshots[0] or k == snapshots[-1] or k in snapshots[:: max(1, len(snapshots) // 10)]:
            print(
                f"[metad] deposit {k}/{n_dep}  t={t_ps:.2f} ps  ΔG={g_list[-1]:.4f}",
                flush=True,
            )
    return np.asarray(t_list), np.stack(f_list, axis=0), np.asarray(g_list)


def align_left(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    return y_src[idx]


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
    e_meta: np.ndarray | None = None,
    e_sfpt: np.ndarray | None = None,
    dG_ref: float | None = None,
    ref_label: str = "final FES",
    ylim: tuple[float, float] | None = None,
    y0_line: bool = False,
    xlim_right: float | None = None,
    extra_lines: list[tuple[np.ndarray, np.ndarray, str, str, str, float]] | None = None,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if dG_ref is not None and ylim is not None:
        ax.axhspan(
            dG_ref - 1.0, dG_ref + 1.0, color=COLOR_US, alpha=0.18, lw=0, zorder=0,
            label=r"final $\Delta G\pm 1$",
        )
    if dG_ref is not None:
        ax.axhline(
            dG_ref, color=COLOR_US, ls=":", lw=1.1, zorder=1,
            label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
        )
    if y0_line:
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    if e_meta is not None:
        ax.fill_between(
            t_meta, y_meta - e_meta, y_meta + e_meta,
            color=COLOR_META, alpha=0.22, lw=0, zorder=1,
        )
    ax.plot(t_meta, y_meta, color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
    if e_sfpt is not None:
        ax.fill_between(
            t_sfpt, y_sfpt - e_sfpt, y_sfpt + e_sfpt,
            color=COLOR_SFPT, alpha=0.25, lw=0, zorder=3,
        )
    ax.plot(t_sfpt, y_sfpt, color=COLOR_SFPT, lw=1.5, label="sFPT", zorder=4)
    if extra_lines:
        for xt, yt, color, ls, lab, lw in extra_lines:
            ax.plot(xt, yt, color=color, ls=ls, lw=lw, label=lab, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim_right is not None:
        ax.set_xlim(0.0, float(xlim_right))
    else:
        ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def plot_rmse_three_phi(
    out: Path,
    *,
    stem: str,
    xlabel: str,
    t_meta: np.ndarray,
    rmse_meta: np.ndarray,
    t_sfpt: np.ndarray,
    rmse_sfpt: np.ndarray,
    e_meta: np.ndarray | None = None,
    e_sfpt: np.ndarray | None = None,
    ylim: tuple[float, float] | None,
    xlim_right: float | None,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharey=True)
    for i, ax in enumerate(axes):
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
        if e_meta is not None:
            ax.fill_between(
                t_meta, rmse_meta[:, i] - e_meta[:, i], rmse_meta[:, i] + e_meta[:, i],
                color=COLOR_META, alpha=0.22, lw=0, zorder=1,
            )
        ax.plot(t_meta, rmse_meta[:, i], color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
        if e_sfpt is not None:
            ax.fill_between(
                t_sfpt, rmse_sfpt[:, i] - e_sfpt[:, i], rmse_sfpt[:, i] + e_sfpt[:, i],
                color=COLOR_SFPT, alpha=0.25, lw=0, zorder=3,
            )
        ax.plot(t_sfpt, rmse_sfpt[:, i], color=COLOR_SFPT, lw=1.5, label="sFPT", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_title(rf"$F(\phi_{i + 1})$", fontsize=11)
        ax.set_xlim(0.0, float(xlim_right) if xlim_right is not None else None)
        if xlim_right is None:
            ax.set_xlim(left=0.0)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(which="both", top=False, right=False)
        if i == 0:
            ax.set_ylabel(
                r"1D $F(\phi)$ RMSE vs MetaD / kcal mol$^{-1}$"
            )
            ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def plot_1d_collage(
    out: Path,
    *,
    stem: str,
    xi: np.ndarray,
    f_ref: np.ndarray,
    f_meta: np.ndarray,
    f_sfpt: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
    phi_idx: int,
    f_meta_std: np.ndarray | None = None,
    f_sfpt_std: np.ndarray | None = None,
) -> None:
    n = len(titles)
    ncols = int(ncols)
    nrows = int(np.ceil(n / ncols))
    plt.rcParams.update(nature_rcparams())
    fig = plt.figure(figsize=(2.35 * ncols, 1.95 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, wspace=0.28, hspace=0.45,
        left=0.06, right=0.99, bottom=0.06, top=0.94,
    )
    y_ref = np.minimum(relative_fes(f_ref), float(vmax))
    y_hi = float(vmax) * 1.02
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ym = np.minimum(relative_fes(f_meta[ig]), float(vmax))
        ys = np.minimum(relative_fes(f_sfpt[ig]), float(vmax))
        if f_meta_std is not None:
            ylo = relative_fes(f_meta[ig] - f_meta_std[ig])
            yhi = relative_fes(f_meta[ig] + f_meta_std[ig])
            ax.fill_between(xi, ylo, yhi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
        ax.plot(xi, ym, color=COLOR_META, lw=1.1, label="WT-MetaD", zorder=2)
        if f_sfpt_std is not None:
            ylo = relative_fes(f_sfpt[ig] - f_sfpt_std[ig])
            yhi = relative_fes(f_sfpt[ig] + f_sfpt_std[ig])
            ax.fill_between(xi, ylo, yhi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
        ax.plot(xi, ys, color=COLOR_SFPT, lw=1.3, label="sFPT", zorder=5)
        ax.plot(xi, y_ref, color=COLOR_US, lw=1.2, ls="--", label="MetaD ref", zorder=6)
        ax.axvline(0.0, color="0.5", ls=":", lw=0.7, zorder=0)
        ax.set_ylim(0.0, y_hi)
        ax.set_xlim(float(xi.min()), float(xi.max()))
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n:
            ax.set_xlabel(rf"$\phi_{phi_idx}$ / rad", fontsize=8)
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
        rf"1D $F(\phi_{phi_idx})$  MetaD / sFPT / MetaD ref  (3 seeds)",
        fontsize=11, y=0.995,
    )
    fig.savefig(out / f"{stem}.png", dpi=160)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dirs", type=Path, nargs="+", default=None)
    p.add_argument("--fpt-dir", type=Path, default=None)
    p.add_argument("--metad-dirs", type=Path, nargs="+", default=None)
    p.add_argument("--metad-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-3d", type=int, default=25)
    p.add_argument("--dg-method", choices=("kde", "hist"), default="hist")
    p.add_argument("--hist-smooth", type=float, default=1.2)
    p.add_argument("--window-ps", type=float, default=5.0)
    p.add_argument("--dump-ps", type=float, default=0.2)
    p.add_argument("--vmax-rmse", type=float, default=16.0)
    p.add_argument("--vmax-1d", type=float, default=16.0)
    p.add_argument("--n-replicas-meta", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=15.0)
    p.add_argument("--sigma", type=str, default="0.35,0.35,0.35")
    p.add_argument("--stride-ps", type=float, default=5.0)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--collage-every", type=int, default=2, help="keep every k-th sFPT window in collage")
    p.add_argument("--collage-ncols", type=int, default=5)
    p.add_argument("--t-max", type=float, default=None)
    p.add_argument("--divein-t", type=float, default=200.0)
    p.add_argument("--sfpt-it-s", type=float, default=SFPT_IT_S)
    p.add_argument("--meta-it-s", type=float, default=META_IT_S)
    p.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="replot from fes_rmse_vs_time.npz (skip KDE / hills)",
    )
    args = p.parse_args()
    fpt_dirs = list(args.fpt_dirs) if args.fpt_dirs else (
        [args.fpt_dir] if args.fpt_dir is not None else list(DEFAULT_FPT_DIRS)
    )
    meta_dirs = list(args.metad_dirs) if args.metad_dirs else (
        [args.metad_dir] if args.metad_dir is not None else list(DEFAULT_META_DIRS)
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _interp_f1d(t_src, f_src, t_dst):
        out_f = np.full((len(t_dst), f_src.shape[1], f_src.shape[2]), np.nan)
        for i in range(f_src.shape[1]):
            for g in range(f_src.shape[2]):
                out_f[:, i, g] = interp_series(t_src, f_src[:, i, g], t_dst)
        return out_f

    def _interp_rmse(t_src, r_src, t_dst):
        out_r = np.full((len(t_dst), r_src.shape[1]), np.nan)
        for i in range(r_src.shape[1]):
            out_r[:, i] = interp_series(t_src, r_src[:, i], t_dst)
        return out_r

    def _rmse_curve(f1, f1_ref, ax, vmax):
        r = np.zeros((f1.shape[0], 3), dtype=np.float64)
        for k in range(f1.shape[0]):
            for i in range(3):
                r[k, i] = rmse_1d(f1[k, i], f1_ref[i], vmax=float(vmax), xi=ax)
        return r

    if args.from_npz is not None:
        z = np.load(args.from_npz, allow_pickle=True)
        t_sfpt = np.asarray(z["t_kde_ps"], dtype=np.float64)
        t_meta = np.asarray(z["t_metad_ps"], dtype=np.float64)
        t_dg = np.asarray(z["t_dg_kde_ps"], dtype=np.float64)
        ax1 = np.asarray(z["ax"], dtype=np.float64)
        f1_sfpt = np.asarray(z["F1d_kde"], dtype=np.float64)
        f1_meta = np.asarray(z["F1d_metad"], dtype=np.float64)
        f1_sfpt_std = np.asarray(z["F1d_kde_std"]) if "F1d_kde_std" in z.files else np.zeros_like(f1_sfpt)
        f1_meta_std = np.asarray(z["F1d_metad_std"]) if "F1d_metad_std" in z.files else np.zeros_like(f1_meta)
        dG_sfpt = np.asarray(z["dG_kde"], dtype=np.float64)
        dG_meta = np.asarray(z["dG_metad"], dtype=np.float64)
        e_dg_s = np.asarray(z["dG_kde_std"]) if "dG_kde_std" in z.files else np.zeros_like(dG_sfpt)
        e_dg_m = np.asarray(z["dG_metad_std"]) if "dG_metad_std" in z.files else np.zeros_like(dG_meta)
        rmse_sfpt = np.asarray(z["rmse_kde"], dtype=np.float64)
        rmse_meta = np.asarray(z["rmse_metad"], dtype=np.float64)
        e_rmse_s = np.asarray(z["rmse_kde_std"]) if "rmse_kde_std" in z.files else np.zeros_like(rmse_sfpt)
        e_rmse_m = np.asarray(z["rmse_metad_std"]) if "rmse_metad_std" in z.files else np.zeros_like(rmse_meta)
        rmse_sfpt_avg = np.asarray(z["rmse_kde_avg"]) if "rmse_kde_avg" in z.files else rmse_sfpt.mean(axis=1)
        rmse_meta_avg = np.asarray(z["rmse_metad_avg"]) if "rmse_metad_avg" in z.files else rmse_meta.mean(axis=1)
        e_rmse_s_avg = np.asarray(z["rmse_kde_avg_std"]) if "rmse_kde_avg_std" in z.files else np.zeros_like(rmse_sfpt_avg)
        e_rmse_m_avg = np.asarray(z["rmse_metad_avg_std"]) if "rmse_metad_avg_std" in z.files else np.zeros_like(rmse_meta_avg)
        lam = float(z["lam"]) if "lam" in z.files else 0.05
        f1_ref = np.asarray(z["F1d_ref"], dtype=np.float64) if "F1d_ref" in z.files else f1_meta[-1].copy()
        dG_ref = float(z["dG_ref"]) if "dG_ref" in z.files else float(dG_meta[-1])
        print(
            f"[from-npz] {args.from_npz}  sFPT t={t_sfpt[-1]:g}  "
            f"MetaD t={t_meta[-1]:g}  ΔG_ref={dG_ref:.4f} (last MetaD mean)",
            flush=True,
        )
    else:
        device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        print(f"[device] {device}  n_sFPT={len(fpt_dirs)}  n_Meta={len(meta_dirs)}", flush=True)
        sig = tuple(float(x) for x in str(args.sigma).split(","))
        if len(sig) != 3:
            raise ValueError("--sigma needs 3 values")
        n_win = max(1, int(round(float(args.window_ps) / float(args.dump_ps))))
        n3 = int(args.n_3d)
        _edges = np.linspace(-np.pi, np.pi, n3 + 1)
        ax3 = 0.5 * (_edges[:-1] + _edges[1:])
        dt_dep = float(args.deposit_pace) * float(args.timestep)
        stride_dep = max(1, int(round(float(args.stride_ps) / dt_dep)))

        sfpt_1d, sfpt_dg, lam, ax1 = [], [], 0.05, None
        for d in fpt_dirs:
            t1, ax1, f1, lam_i = load_sfpt_1d(Path(d) / "fes_2d")
            lam = float(lam_i)
            print(
                f"[sfpt-1d] {Path(d).name}  {len(t1)} windows  t={t1[0]:g}–{t1[-1]:g} ps",
                flush=True,
            )
            stages = discover_stages(Path(d))
            n_groups = len(stages) // n_win
            print(
                f"[sfpt-3d] {Path(d).name} stages={len(stages)} windows={n_groups}",
                flush=True,
            )
            t_list, g_list = [], []
            for ig in range(n_groups):
                chunk_rows = stages[ig * n_win : (ig + 1) * n_win]
                t_ps = float(ig + 1) * float(args.window_ps)
                if args.dg_method == "kde":
                    pairs = [(np.load(dih), np.load(bw)) for _, dih, bw in chunk_rows]
                    f3 = kde_3d_window(
                        pairs, ax=ax3, lam=lam, kT=KT_KCAL, device=device, chunk=int(args.chunk)
                    )
                else:
                    cvs = [np.load(dih) for _, dih, _ in chunk_rows]
                    f3 = hist_3d_window(
                        cvs, ax=ax3, lam=lam, kT=KT_KCAL, smooth=float(args.hist_smooth)
                    )
                t_list.append(t_ps)
                g_list.append(delta_g_octant(f3, ax3, kT=KT_KCAL))
                if ig == 0 or (ig + 1) % max(1, n_groups // 10) == 0 or ig == n_groups - 1:
                    print(
                        f"[sfpt-3d] {Path(d).name} t={t_ps:g} ps  ΔG={g_list[-1]:.4f}",
                        flush=True,
                    )
            sfpt_1d.append((t1, f1))
            sfpt_dg.append((np.asarray(t_list), np.asarray(g_list)))

        snap_root = out / "meta_snaps"
        last_f1, last_g = [], []
        meta_1d, meta_dg = [], []
        for d in meta_dirs:
            snap = snapshot_meta(Path(d), snap_root)
            hills = snap / "hills.npz"
            print(f"[meta] {Path(d).name} hills={hills}", flush=True)
            t_m, f1_ax3, g_m = incremental_metad_3d(
                hills,
                ax=ax3,
                n_replicas=int(args.n_replicas_meta),
                deposit_pace=int(args.deposit_pace),
                timestep=float(args.timestep),
                bias_factor=float(args.bias_factor),
                sigma=sig,
                stride_deposits=stride_dep,
                kT_kcal=KT_KCAL,
                device=device,
            )
            f1_m = np.stack(
                [
                    np.stack([interp_1d(ax3, f1_ax3[k, i], ax1) for i in range(3)], axis=0)
                    for k in range(f1_ax3.shape[0])
                ],
                axis=0,
            )
            meta_1d.append((t_m, f1_m))
            meta_dg.append((t_m, g_m))
            last_f1.append(f1_m[-1])
            last_g.append(float(g_m[-1]))
            print(
                f"[meta] {Path(d).name} t={t_m[-1]:g} ps  ΔG_last={g_m[-1]:.4f}",
                flush=True,
            )

        f1_ref = boltzmann_mean(last_f1)
        dG_ref = float(np.mean(last_g))
        print(f"[ref] last WT-MetaD mean  ΔG_ref={dG_ref:.4f}  seeds={last_g}", flush=True)

        t_sfpt = np.arange(
            float(args.window_ps),
            max(float(t[-1]) for t, _ in sfpt_1d) + 0.5 * float(args.window_ps),
            float(args.window_ps),
        )
        t_meta = np.arange(
            float(args.stride_ps),
            max(float(t[-1]) for t, _ in meta_1d) + 0.5 * float(args.stride_ps),
            float(args.stride_ps),
        )

        f1s_stack = np.stack([_interp_f1d(t, f, t_sfpt) for t, f in sfpt_1d], axis=0)
        f1m_stack = np.stack([_interp_f1d(t, f, t_meta) for t, f in meta_1d], axis=0)
        f1_sfpt, f1_sfpt_std = mean_std(f1s_stack)
        f1_meta, f1_meta_std = mean_std(f1m_stack)

        rmse_s_stack = np.stack(
            [_interp_rmse(t, _rmse_curve(f, f1_ref, ax1, args.vmax_rmse), t_sfpt) for t, f in sfpt_1d],
            axis=0,
        )
        rmse_m_stack = np.stack(
            [_interp_rmse(t, _rmse_curve(f, f1_ref, ax1, args.vmax_rmse), t_meta) for t, f in meta_1d],
            axis=0,
        )
        rmse_sfpt, e_rmse_s = mean_std(rmse_s_stack)
        rmse_meta, e_rmse_m = mean_std(rmse_m_stack)
        rmse_sfpt_avg, e_rmse_s_avg = mean_std(np.nanmean(rmse_s_stack, axis=2))
        rmse_meta_avg, e_rmse_m_avg = mean_std(np.nanmean(rmse_m_stack, axis=2))

        t_dg, dG_sfpt, e_dg_s = stack_mean_std(sfpt_dg, dt=float(args.window_ps))
        t_meta_g, dG_meta, e_dg_m = stack_mean_std(meta_dg, dt=float(args.stride_ps))
        t_meta_for_f1 = t_meta
        t_meta = t_meta_g

        np.savez_compressed(
            out / "fes_rmse_vs_time.npz",
            t_kde_ps=t_sfpt,
            t_metad_ps=t_meta_for_f1,
            t_dg_kde_ps=t_dg,
            t_dg_metad_ps=t_meta,
            rmse_kde=rmse_sfpt,
            rmse_kde_std=e_rmse_s,
            rmse_metad=rmse_meta,
            rmse_metad_std=e_rmse_m,
            rmse_kde_avg=rmse_sfpt_avg,
            rmse_kde_avg_std=e_rmse_s_avg,
            rmse_metad_avg=rmse_meta_avg,
            rmse_metad_avg_std=e_rmse_m_avg,
            dG_kde=dG_sfpt,
            dG_kde_std=e_dg_s,
            dG_metad=dG_meta,
            dG_metad_std=e_dg_m,
            dG_ref=np.float64(dG_ref),
            dG_metad_ref=np.float64(dG_ref),
            dG_ref_seeds=np.asarray(last_g, dtype=np.float64),
            ax=ax1,
            F1d_kde=f1_sfpt,
            F1d_kde_std=f1_sfpt_std,
            F1d_metad=f1_meta,
            F1d_metad_std=f1_meta_std,
            F1d_ref=f1_ref,
            lam=np.float64(lam),
            n_3d=np.int64(args.n_3d),
            dg_method=np.asarray(args.dg_method),
            n_sfpt=np.int64(len(fpt_dirs)),
            n_meta=np.int64(len(meta_dirs)),
        )
        t_meta_rmse = t_meta_for_f1
        # fall through with both t_meta (ΔG) and t_meta_rmse (FES/RMSE)

    if args.from_npz is not None:
        t_meta_rmse = t_meta
        t_meta_dg = t_meta
        if "t_dg_metad_ps" in np.load(args.from_npz).files:
            t_meta_dg = np.asarray(np.load(args.from_npz)["t_dg_metad_ps"], dtype=np.float64)
        t_meta = t_meta_dg
    else:
        t_meta_rmse = t_meta_for_f1

    if args.t_max is not None:
        tmax = float(args.t_max) + 1e-12
        m_s = t_sfpt <= tmax
        m_m = t_meta_rmse <= tmax
        m_g = t_dg <= tmax
        m_mg = t_meta <= tmax
        t_sfpt, f1_sfpt, f1_sfpt_std, rmse_sfpt, e_rmse_s = (
            t_sfpt[m_s], f1_sfpt[m_s], f1_sfpt_std[m_s], rmse_sfpt[m_s], e_rmse_s[m_s]
        )
        rmse_sfpt_avg, e_rmse_s_avg = rmse_sfpt_avg[m_s], e_rmse_s_avg[m_s]
        t_meta_rmse, f1_meta, f1_meta_std, rmse_meta, e_rmse_m = (
            t_meta_rmse[m_m], f1_meta[m_m], f1_meta_std[m_m], rmse_meta[m_m], e_rmse_m[m_m]
        )
        rmse_meta_avg, e_rmse_m_avg = rmse_meta_avg[m_m], e_rmse_m_avg[m_m]
        t_dg, dG_sfpt, e_dg_s = t_dg[m_g], dG_sfpt[m_g], e_dg_s[m_g]
        t_meta, dG_meta, e_dg_m = t_meta[m_mg], dG_meta[m_mg], e_dg_m[m_mg]

    tw_sfpt = fpt_wall_hours_from_throughput(
        t_sfpt, timestep_ps=float(args.timestep), rate_it_s=float(args.sfpt_it_s)
    )
    tw_meta = fpt_wall_hours_from_throughput(
        t_meta, timestep_ps=float(args.timestep), rate_it_s=float(args.meta_it_s)
    )
    tw_dg = fpt_wall_hours_from_throughput(
        t_dg, timestep_ps=float(args.timestep), rate_it_s=float(args.sfpt_it_s)
    )
    tw_meta_rmse = fpt_wall_hours_from_throughput(
        t_meta_rmse, timestep_ps=float(args.timestep), rate_it_s=float(args.meta_it_s)
    )
    cost_sfpt, cost_meta, cost_dg = tw_sfpt, tw_meta_rmse, tw_dg

    ylab_dg = (
        r"$\Delta G=k_{\mathrm{B}}T\ln(P_A/P_B)$"
        r"  $A=\{\phi_{1,2,3}<0\}$ / kcal mol$^{-1}$"
    )
    t_right = float(args.t_max) if args.t_max is not None else None
    for stem, ylim, xm, ym, em, xs, ys, es, xlab, xright in [
        ("deltaG_vs_time", None, t_meta, dG_meta, e_dg_m, t_dg, dG_sfpt, e_dg_s, r"$t$ / ps", t_right),
        (
            "deltaG_vs_time_divein",
            (dG_ref - 3.0, dG_ref + 3.0),
            t_meta, dG_meta, e_dg_m, t_dg, dG_sfpt, e_dg_s, r"$t$ / ps", float(args.divein_t),
        ),
        (
            "deltaG_vs_wallclock",
            None, tw_meta, dG_meta, e_dg_m, tw_dg, dG_sfpt, e_dg_s, r"wall-clock time / h", None,
        ),
        (
            "deltaG_vs_wallclock_divein",
            (dG_ref - 3.0, dG_ref + 3.0),
            tw_meta, dG_meta, e_dg_m, tw_dg, dG_sfpt, e_dg_s, r"wall-clock time / h", None,
        ),
    ]:
        plot_series(
            out, stem=stem, xlabel=xlab, ylabel=ylab_dg,
            t_meta=xm, y_meta=ym, e_meta=em, t_sfpt=xs, y_sfpt=ys, e_sfpt=es,
            dG_ref=dG_ref, ref_label="MetaD", ylim=ylim, xlim_right=xright,
        )

    ylab_rmse = r"1D $F(\phi)$ RMSE vs MetaD (mean of $\phi_{1,2,3}$) / kcal mol$^{-1}$"
    for stem, ylim, xm, ym, em, xs, ys, es, xlab, xright in [
        ("fes_rmse_vs_time", None, t_meta_rmse, rmse_meta_avg, e_rmse_m_avg, t_sfpt, rmse_sfpt_avg, e_rmse_s_avg, r"$t$ / ps", t_right),
        (
            "fes_rmse_vs_time_divein", (0.0, 3.0),
            t_meta_rmse, rmse_meta_avg, e_rmse_m_avg, t_sfpt, rmse_sfpt_avg, e_rmse_s_avg, r"$t$ / ps", float(args.divein_t),
        ),
        (
            "fes_rmse_vs_wallclock", None,
            tw_meta_rmse, rmse_meta_avg, e_rmse_m_avg, tw_sfpt, rmse_sfpt_avg, e_rmse_s_avg, r"wall-clock time / h", None,
        ),
        (
            "fes_rmse_vs_wallclock_divein", (0.0, 3.0),
            tw_meta_rmse, rmse_meta_avg, e_rmse_m_avg, tw_sfpt, rmse_sfpt_avg, e_rmse_s_avg, r"wall-clock time / h", None,
        ),
        (
            "fes_rmse_vs_cost", None,
            cost_meta, rmse_meta_avg, e_rmse_m_avg, cost_sfpt, rmse_sfpt_avg, e_rmse_s_avg,
            r"cost / GPU$\cdot$h", None,
        ),
        (
            "fes_rmse_vs_cost_divein", (0.0, 3.0),
            cost_meta, rmse_meta_avg, e_rmse_m_avg, cost_sfpt, rmse_sfpt_avg, e_rmse_s_avg,
            r"cost / GPU$\cdot$h", None,
        ),
    ]:
        plot_series(
            out, stem=stem, xlabel=xlab, ylabel=ylab_rmse,
            t_meta=xm, y_meta=ym, e_meta=em, t_sfpt=xs, y_sfpt=ys, e_sfpt=es,
            ylim=ylim, y0_line=True, xlim_right=xright,
        )

    plot_rmse_three_phi(
        out, stem="fes_rmse_vs_time_phi123", xlabel=r"$t$ / ps",
        t_meta=t_meta_rmse, rmse_meta=rmse_meta, e_meta=e_rmse_m,
        t_sfpt=t_sfpt, rmse_sfpt=rmse_sfpt, e_sfpt=e_rmse_s,
        ylim=None, xlim_right=t_right,
    )
    plot_rmse_three_phi(
        out, stem="fes_rmse_vs_time_phi123_divein", xlabel=r"$t$ / ps",
        t_meta=t_meta_rmse, rmse_meta=rmse_meta, e_meta=e_rmse_m,
        t_sfpt=t_sfpt, rmse_sfpt=rmse_sfpt, e_sfpt=e_rmse_s,
        ylim=(0.0, 3.0), xlim_right=float(args.divein_t),
    )

    step = max(1, int(args.collage_every))
    idx = np.arange(0, t_sfpt.shape[0], step)
    titles = [f"t={t_sfpt[i]:g} ps" for i in idx]
    f_meta_al = align_left(t_meta_rmse, f1_meta, t_sfpt)
    f_meta_al_std = align_left(t_meta_rmse, f1_meta_std, t_sfpt)
    for i in range(3):
        stem = "fes_1d_all_groups" if i == 0 else f"fes_1d_phi{i + 1}_all_groups"
        plot_1d_collage(
            out, stem=stem, xi=ax1, f_ref=f1_ref[i],
            f_meta=f_meta_al[idx, i], f_sfpt=f1_sfpt[idx, i],
            f_meta_std=f_meta_al_std[idx, i], f_sfpt_std=f1_sfpt_std[idx, i],
            titles=titles, vmax=float(args.vmax_1d),
            ncols=int(args.collage_ncols), phi_idx=i + 1,
        )

    print(
        f"[done] ΔG_ref={dG_ref:.4f}  sFPT last ΔG={dG_sfpt[-1]:.4f}  "
        f"MetaD last ΔG={dG_meta[-1]:.4f}  "
        f"RMSE_sfpt={rmse_sfpt[-1]}  RMSE_meta={rmse_meta[-1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
