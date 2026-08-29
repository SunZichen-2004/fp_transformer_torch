#!/usr/bin/env python3
"""Tetra-alanine main figure: 1×4 panels a–d from 3 sFPT + 3 WT-MetaD seeds.

a/b: Boltzmann-mean F(φ2, φ3) of each method's latest window.
c:   F(φ3) mean ± std over seeds.
d:   ΔG(t) mean ± std; reference = last MetaD ΔG averaged over seeds.
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments"
SFP = EXP / "tetra_alanine_sFPT"
OUT = SFP / "main_picture"
sys.path.insert(0, str(SFP))

from fes_window import (  # noqa: E402
    FE_LABEL,
    PI_LABELS,
    PI_TICKS,
    REMD_PLOT_SMOOTH_2D,
    SFPT_SMOOTH_2D,
    fes_cmap,
    relative_fes,
    smooth_fes_torus,
)

SFPT_DIRS = [
    SFP / "results" / "tetra_alanine_phi3_r1024_s1_94120",
    SFP / "results" / "tetra_alanine_phi3_r1024_s2_94121",
    SFP / "results" / "tetra_alanine_phi3_r1024_s3_94122",
]
META_DIRS = [
    EXP / "tetra_alanine_meta" / "results" / "tetra_alanine_metad_bspline_r64_s0_94242",
    EXP / "tetra_alanine_meta" / "results" / "tetra_alanine_metad_bspline_r64_s1_94243",
    EXP / "tetra_alanine_meta" / "results" / "tetra_alanine_metad_bspline_r64_s2_94244",
]

COLOR_META = "#2166AC"
COLOR_SFPT = "#E67E22"
COLOR_US = "#2CA02C"
KT_KCAL = 0.00198720425864083 * 300.0
VMAX = 16.0
REF_BAND = 0.5
T_RE = re.compile(r"fes_2d_t(\d+)ps\.npz$")
DG_TITLE = r"$\Delta G=k_{\mathrm{B}}T\ln(P_A/P_B)$"


def nature_rcparams() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }


def latest_sfpt_npz(run_or_fes: Path) -> Path:
    fes_dir = run_or_fes / "fes_2d" if (run_or_fes / "fes_2d").is_dir() else run_or_fes
    cands = []
    for p in fes_dir.glob("fes_2d_t*ps.npz"):
        m = T_RE.search(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        raise FileNotFoundError(f"no fes_2d_t*ps.npz in {fes_dir}")
    cands.sort()
    return cands[-1][1]


def meta_live(run_dir: Path) -> Path:
    p = run_dir / "live" / "fes_2d.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def tag_title(
    ax, letter: str, title: str, *, letter_x: float = -0.08, title_fontsize: float = 17
) -> None:
    ax.text(
        letter_x, 1.04, f"{letter}", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=18, fontweight="bold", clip_on=False,
    )
    ax.set_title(
        title if title else " ",
        loc="center",
        pad=8.0,
        fontsize=title_fontsize,
        fontweight="normal",
    )


def _wrap_1d(xi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([xi, xi[:1] + 2.0 * np.pi]), np.concatenate([y, y[:1]])


def fes_1d_from_2d(F: np.ndarray, *, kT: float, axis: int) -> np.ndarray:
    w = np.exp(-np.clip(relative_fes(F), 0.0, 80.0) / float(kT))
    p = np.clip(w.sum(axis=0 if axis == 0 else 1), 1e-300, None)
    return relative_fes(-float(kT) * np.log(p))


def f_phi3_meta(f_pairs: dict[tuple[int, int], np.ndarray], *, kT: float) -> np.ndarray:
    a = fes_1d_from_2d(f_pairs[(0, 2)], kT=kT, axis=1)
    b = fes_1d_from_2d(f_pairs[(1, 2)], kT=kT, axis=1)
    p = np.mean(
        [np.exp(-np.clip(a, 0.0, 80.0) / float(kT)), np.exp(-np.clip(b, 0.0, 80.0) / float(kT))],
        axis=0,
    )
    return relative_fes(-float(kT) * np.log(np.clip(p, 1e-300, None)))


def fes_full_square(grid: np.ndarray, F: np.ndarray, *, sigma: float, n_fine: int = 361):
    """Periodic smooth, then interpolate onto a closed [−π, π] square."""
    from scipy.interpolate import RegularGridInterpolator

    z = np.minimum(smooth_fes_torus(grid, F, sigma=sigma), float(VMAX))
    if np.isclose(grid[-1], grid[0] + 2.0 * np.pi):
        ax_ext, z_ext = grid, z
    else:
        ax_ext = np.concatenate([grid, grid[:1] + 2.0 * np.pi])
        z_ext = np.pad(z, ((0, 1), (0, 1)), mode="wrap")
    interp = RegularGridInterpolator(
        (ax_ext, ax_ext), z_ext, bounds_error=False, fill_value=None
    )
    fine = np.linspace(-np.pi, np.pi, int(n_fine))
    gy, gx = np.meshgrid(fine, fine, indexing="ij")
    zf = interp(np.stack([gy.ravel(), gx.ravel()], axis=-1)).reshape(gy.shape)
    return fine, relative_fes(zf)


def boltzmann_mean(fs: list[np.ndarray], *, kT: float = KT_KCAL) -> np.ndarray:
    w = np.mean(
        [np.exp(-np.clip(relative_fes(f), 0.0, 80.0) / float(kT)) for f in fs],
        axis=0,
    )
    return relative_fes(-float(kT) * np.log(np.clip(w, 1e-300, None)))


def interp_1d(ax_src: np.ndarray, f_src: np.ndarray, ax_dst: np.ndarray) -> np.ndarray:
    two_pi = 2.0 * np.pi
    x = np.concatenate([ax_src - two_pi, ax_src, ax_src + two_pi])
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
    m = (t_dst + 1e-9 >= t_src[0]) & (t_dst - 1e-9 <= t_src[-1])
    if np.any(m):
        out[m] = np.interp(t_dst[m], t_src, y_src)
    return out


def draw_fes2d_z(ax, xx, z, *, vmax: float, title: str, letter: str, show_ylabel: bool):
    z = np.minimum(relative_fes(z), float(vmax))
    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    im = ax.contourf(xx, xx, z, levels=levels, cmap=cmap, norm=norm, extend="max")
    ax.contour(
        xx, xx, np.clip(z, 0.0, float(vmax)),
        levels=levels[::3], colors="0.45", linewidths=0.35, alpha=0.45,
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.margins(0.0)
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_yticks(PI_TICKS)
    ax.set_xlabel(r"$\phi_2$ / rad")
    ax.set_aspect("auto")
    ax.tick_params(which="both", top=False, right=False)
    if show_ylabel:
        ax.set_yticklabels(PI_LABELS)
        ax.set_ylabel(r"$\phi_3$ / rad")
        tag_title(ax, letter, title)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        tag_title(ax, letter, title, letter_x=0.02)
    return im


def draw_fes2d(ax, grid, F, *, vmax: float, title: str, letter: str, sigma: float, show_ylabel: bool):
    xx, z = fes_full_square(grid, F, sigma=sigma)
    z = np.minimum(z, float(vmax))
    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    im = ax.contourf(xx, xx, z, levels=levels, cmap=cmap, norm=norm, extend="max")
    ax.contour(
        xx, xx, np.clip(z, 0.0, float(vmax)),
        levels=levels[::3], colors="0.45", linewidths=0.35, alpha=0.45,
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.margins(0.0)
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_yticks(PI_TICKS)
    ax.set_xlabel(r"$\phi_2$ / rad")
    ax.set_aspect("auto")
    ax.tick_params(which="both", top=False, right=False)
    if show_ylabel:
        ax.set_yticklabels(PI_LABELS)
        ax.set_ylabel(r"$\phi_3$ / rad")
        tag_title(ax, letter, title)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        tag_title(ax, letter, title, letter_x=0.02)
    return im


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


def compute_sfpt_dg(
    run_dir: Path,
    *,
    window_ps: float = 5.0,
    dump_ps: float = 0.2,
    n_3d: int = 25,
    hist_smooth: float = 1.2,
    lam: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    from plot_convergence_phi3 import delta_g_octant, discover_stages, hist_3d_window

    n_win = max(1, int(round(float(window_ps) / float(dump_ps))))
    stages = discover_stages(run_dir)
    n_groups = len(stages) // n_win
    edges = np.linspace(-np.pi, np.pi, int(n_3d) + 1)
    ax3 = 0.5 * (edges[:-1] + edges[1:])
    t_list, g_list = [], []
    print(
        f"[sfpt-dg] {run_dir.name} stages={len(stages)} windows={n_groups}",
        flush=True,
    )
    for ig in range(n_groups):
        chunk = stages[ig * n_win : (ig + 1) * n_win]
        cvs = [np.load(dih) for _, dih, _ in chunk]
        f3 = hist_3d_window(
            cvs, ax=ax3, lam=float(lam), kT=KT_KCAL, smooth=float(hist_smooth)
        )
        t_list.append(float(ig + 1) * float(window_ps))
        g_list.append(delta_g_octant(f3, ax3, kT=KT_KCAL))
        if ig == 0 or (ig + 1) % max(1, n_groups // 10) == 0 or ig == n_groups - 1:
            print(
                f"[sfpt-dg] {run_dir.name} t={t_list[-1]:g} ps  ΔG={g_list[-1]:.4f}",
                flush=True,
            )
    return np.asarray(t_list, dtype=np.float64), np.asarray(g_list, dtype=np.float64)


def compute_meta_dg(
    hills_path: Path,
    *,
    device,
    n_3d: int = 25,
    n_replicas: int = 64,
    deposit_pace: int = 50,
    timestep: float = 0.001,
    bias_factor: float = 15.0,
    sigma: tuple[float, float, float] = (0.35, 0.35, 0.35),
    stride_ps: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    from plot_convergence_phi3 import incremental_metad_3d

    edges = np.linspace(-np.pi, np.pi, int(n_3d) + 1)
    ax3 = 0.5 * (edges[:-1] + edges[1:])
    dt_dep = float(deposit_pace) * float(timestep)
    stride_dep = max(1, int(round(float(stride_ps) / dt_dep)))
    t_meta, _, dG_meta = incremental_metad_3d(
        hills_path,
        ax=ax3,
        n_replicas=int(n_replicas),
        deposit_pace=int(deposit_pace),
        timestep=float(timestep),
        bias_factor=float(bias_factor),
        sigma=sigma,
        stride_deposits=stride_dep,
        kT_kcal=KT_KCAL,
        device=device,
    )
    return np.asarray(t_meta, dtype=np.float64), np.asarray(dG_meta, dtype=np.float64)


def stack_mean_std(
    series: list[tuple[np.ndarray, np.ndarray]], dt: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_max = max(float(t[-1]) for t, _ in series)
    t_common = np.arange(dt, t_max + 0.5 * dt, dt)
    stacked = np.stack([interp_series(t, y, t_common) for t, y in series], axis=0)
    mean, std = mean_std(stacked)
    return t_common, mean, std


def load_latest_2d_1d(
    meta_fes: list[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ax1 = np.linspace(-np.pi, np.pi, 73)
    f23_s, f23_m, f3_s, f3_m = [], [], [], []
    xx = None
    for d in SFPT_DIRS:
        p = latest_sfpt_npz(d)
        z = np.load(p)
        ax = np.asarray(z["ax"], dtype=np.float64)
        xx, z23 = fes_full_square(ax, np.asarray(z["F_phi2_phi3"]), sigma=SFPT_SMOOTH_2D)
        f23_s.append(z23)
        f3_s.append(interp_1d(ax, relative_fes(np.asarray(z["F1_phi3"])), ax1))
        print(f"[2d] sFPT {d.name} {p.name} t={float(z['t_ps']):g} ps", flush=True)
    meta_paths = meta_fes if meta_fes is not None else [meta_live(d) for d in META_DIRS]
    for d, p in zip(META_DIRS, meta_paths):
        m = np.load(p)
        ax = np.asarray(m["ax"], dtype=np.float64)
        _, z23 = fes_full_square(
            ax, np.asarray(m["F_phi2_phi3"]), sigma=REMD_PLOT_SMOOTH_2D
        )
        f23_m.append(z23)
        f_pairs = {
            (0, 2): np.asarray(m["F_phi1_phi3"]),
            (1, 2): np.asarray(m["F_phi2_phi3"]),
        }
        f3_m.append(interp_1d(ax, f_phi3_meta(f_pairs, kT=KT_KCAL), ax1))
        print(f"[2d] Meta {d.name} {p} t={float(m['t_ps']):g} ps", flush=True)
    assert xx is not None
    return (
        xx,
        boltzmann_mean(f23_s),
        boltzmann_mean(f23_m),
        np.stack(f3_s, axis=0),
        np.stack(f3_m, axis=0),
    )


def compute_all_dg(
    *,
    device,
    n_3d: int,
    cache: Path,
    hills: list[Path],
) -> dict[str, np.ndarray]:
    sfpt_series = []
    for d in SFPT_DIRS:
        z = np.load(latest_sfpt_npz(d))
        lam = float(z["lam"]) if "lam" in z.files else 0.05
        t, g = compute_sfpt_dg(d, n_3d=n_3d, lam=lam)
        sfpt_series.append((t, g))
    meta_series = []
    last_g = []
    for d, hp in zip(META_DIRS, hills):
        print(f"[meta-dg] {d.name} hills={hp}", flush=True)
        t, g = compute_meta_dg(hp, device=device, n_3d=n_3d)
        meta_series.append((t, g))
        last_g.append(float(g[-1]))
        print(
            f"[meta-dg] {d.name} t={t[-1]:g} ps  ΔG_last={g[-1]:.4f}",
            flush=True,
        )
    t_s, y_s, e_s = stack_mean_std(sfpt_series)
    t_m, y_m, e_m = stack_mean_std(meta_series)
    dG_ref = float(np.mean(last_g))
    payload = {
        "t_sfpt": t_s,
        "dG_sfpt_mean": y_s,
        "dG_sfpt_std": e_s,
        "t_meta": t_m,
        "dG_meta_mean": y_m,
        "dG_meta_std": e_m,
        "dG_ref": np.asarray(dG_ref),
        "dG_ref_seeds": np.asarray(last_g, dtype=np.float64),
    }
    for i, (t, g) in enumerate(sfpt_series):
        payload[f"t_sfpt_{i}"] = t
        payload[f"dG_sfpt_{i}"] = g
    for i, (t, g) in enumerate(meta_series):
        payload[f"t_meta_{i}"] = t
        payload[f"dG_meta_{i}"] = g
    np.savez_compressed(cache, **payload)
    print(f"[ok] cache {cache}  ΔG_ref={dG_ref:.4f}  seeds={last_g}", flush=True)
    return payload


def load_dg_cache(cache: Path) -> dict[str, np.ndarray]:
    z = np.load(cache)
    return {k: z[k] for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-3d", type=int, default=25)
    p.add_argument("--from-npz", type=Path, default=None)
    p.add_argument(
        "--cache",
        type=Path,
        default=OUT / "multi_seed_cache.npz",
    )
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(nature_rcparams())

    if args.from_npz is not None:
        xx, f23_s, f23_m, f3s, f3m = load_latest_2d_1d()
        dg = load_dg_cache(args.from_npz)
    else:
        import torch

        device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        print(f"[device] {device}", flush=True)
        snap_root = OUT / "meta_snaps"
        snaps = [snapshot_meta(d, snap_root) for d in META_DIRS]
        xx, f23_s, f23_m, f3s, f3m = load_latest_2d_1d(
            meta_fes=[s / "fes_2d.npz" for s in snaps]
        )
        dg = compute_all_dg(
            device=device,
            n_3d=int(args.n_3d),
            cache=args.cache,
            hills=[s / "hills.npz" for s in snaps],
        )
    f3s_mean, f3s_std = mean_std(f3s)
    f3m_mean, f3m_std = mean_std(f3m)
    ax1 = np.linspace(-np.pi, np.pi, 73)

    t_sfpt = np.asarray(dg["t_sfpt"], dtype=np.float64)
    t_meta = np.asarray(dg["t_meta"], dtype=np.float64)
    dG_sfpt = np.asarray(dg["dG_sfpt_mean"], dtype=np.float64)
    dG_meta = np.asarray(dg["dG_meta_mean"], dtype=np.float64)
    e_sfpt = np.asarray(dg["dG_sfpt_std"], dtype=np.float64)
    e_meta = np.asarray(dg["dG_meta_std"], dtype=np.float64)
    dG_ref = float(np.asarray(dg["dG_ref"]))

    fig_w, fig_h = 14.0, 4.80
    fig = plt.figure(figsize=(fig_w, fig_h))
    h = 0.54
    w = h * fig_h / fig_w
    y0 = 0.16
    x0 = 0.062
    gap_ab = 0.022
    gap_cbar = 0.006
    cbar_w = 0.011
    gap_bc = 0.100
    gap_cd = 0.055
    xa = x0
    xb = xa + w + gap_ab
    xbar = xb + w + gap_cbar
    xc = xbar + cbar_w + gap_bc
    xd = xc + w + gap_cd
    ax_a = fig.add_axes([xa, y0, w, h])
    ax_b = fig.add_axes([xb, y0, w, h])
    ax_c = fig.add_axes([xc, y0, w, h])
    ax_d = fig.add_axes([xd, y0, w, h])

    im = draw_fes2d_z(
        ax_a, xx, f23_s, vmax=VMAX, title="sFPT", letter="a", show_ylabel=True,
    )
    draw_fes2d_z(
        ax_b, xx, f23_m, vmax=VMAX, title="WT-MetaD", letter="b", show_ylabel=False,
    )

    xm, ym = _wrap_1d(ax1, f3m_mean)
    _, ym_lo = _wrap_1d(ax1, f3m_mean - f3m_std)
    _, ym_hi = _wrap_1d(ax1, f3m_mean + f3m_std)
    xs, ys = _wrap_1d(ax1, f3s_mean)
    _, ys_lo = _wrap_1d(ax1, f3s_mean - f3s_std)
    _, ys_hi = _wrap_1d(ax1, f3s_mean + f3s_std)
    ax_c.fill_between(xm, ym_lo, ym_hi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
    ax_c.plot(xm, ym, color=COLOR_META, lw=1.8, label="WT-MetaD", zorder=2)
    ax_c.fill_between(xs, ys_lo, ys_hi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
    ax_c.plot(xs, ys, color=COLOR_SFPT, lw=2.1, label="sFPT", zorder=4)
    ax_c.axvline(0.0, color="0.55", ls=":", lw=1.0, zorder=0)
    ax_c.set_xlim(-np.pi, np.pi)
    ax_c.set_xticks(PI_TICKS)
    ax_c.set_xticklabels(PI_LABELS)
    ax_c.set_ylim(0.0, VMAX)
    ax_c.set_yticks([0, 4, 8, 12, 16])
    ax_c.set_xlabel(r"$\phi_3$ / rad")
    ax_c.set_ylabel(FE_LABEL)
    tag_title(ax_c, "c", r"$F(\phi_3)$")
    ax_c.tick_params(which="both", top=False, right=False)

    ax_d.axhspan(
        dG_ref - REF_BAND, dG_ref + REF_BAND,
        color=COLOR_US, alpha=0.18, lw=0, zorder=0,
    )
    ax_d.axhline(dG_ref, color=COLOR_US, ls=":", lw=1.6, zorder=1)
    t_meta_ns = np.asarray(t_meta, dtype=np.float64) / 1000.0
    t_sfpt_ns = np.asarray(t_sfpt, dtype=np.float64) / 1000.0
    ax_d.fill_between(
        t_meta_ns, dG_meta - e_meta, dG_meta + e_meta,
        color=COLOR_META, alpha=0.22, lw=0, zorder=1,
    )
    ax_d.plot(t_meta_ns, dG_meta, color=COLOR_META, lw=1.8, zorder=2)
    ax_d.fill_between(
        t_sfpt_ns, dG_sfpt - e_sfpt, dG_sfpt + e_sfpt,
        color=COLOR_SFPT, alpha=0.25, lw=0, zorder=3,
    )
    ax_d.plot(t_sfpt_ns, dG_sfpt, color=COLOR_SFPT, lw=2.1, zorder=4)
    ax_d.set_xlabel(r"$t$ / ns")
    ax_d.set_ylabel(r"$\Delta G$ / kcal mol$^{-1}$")
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_ylim(-1.5, 2.5)
    ax_d.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax_d.tick_params(which="both", top=False, right=False)
    tag_title(ax_d, "d", DG_TITLE, title_fontsize=13)

    fig.legend(
        handles=[
            Line2D([0], [0], color=COLOR_META, lw=2.4, label="WT-MetaD"),
            Line2D([0], [0], color=COLOR_SFPT, lw=2.6, label="sFPT"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=2,
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        borderaxespad=0.0,
        columnspacing=1.2,
    )

    cax = fig.add_axes([xbar, y0, cbar_w, h])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(r"kcal mol$^{-1}$", fontsize=14, labelpad=3)
    cb.ax.tick_params(labelsize=13, length=3.2, width=1.0)
    cb.set_ticks(np.linspace(0.0, VMAX, 5))

    png = OUT / "tetra_alanine_main_picture.png"
    pdf = OUT / "tetra_alanine_main_picture.pdf"
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0.08)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0.08)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)
    print(f"[d] ΔG_ref={dG_ref:.4f}  ±{REF_BAND:g}  n_sFPT={len(SFPT_DIRS)}  n_Meta={len(META_DIRS)}", flush=True)


if __name__ == "__main__":
    main()
