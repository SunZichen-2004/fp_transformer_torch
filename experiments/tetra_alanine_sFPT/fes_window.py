"""Live 2D (φi, φj) + 1D φ FES: torus-KDE (panel-a recipe) vs REMD histogram."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.colors import Normalize

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from fp_transformer_torch.kernel.gaussian import (
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)

PHI_PAIRS = ((0, 1, 1, 2), (0, 2, 1, 3), (1, 2, 2, 3))
PI_TICKS = (-np.pi, 0.0, np.pi)
PI_LABELS = (r"$-\pi$", r"$0$", r"$\pi$")
FE_LABEL = r"free energy / kcal mol$^{-1}$"
FES_CMAP = "RdBu_r"
COLOR_SFPT = "#e67e22"
COLOR_REMD = "#2166ac"
COLOR_META = "#1a7f37"
REMD_HIST_SMOOTH_2D = 1.8
REMD_HIST_SMOOTH_1D = 1.8
REMD_PLOT_SMOOTH_2D = 1.3
SFPT_SMOOTH_2D = 1.1
DEFAULT_META_CSV = (
    Path(__file__).resolve().parent / "results" / "reference_FES_matrix_kcal.csv"
)
STAGE_DIH_RE = re.compile(r"stage(\d+)_.*_dihedral\.npy$", re.I)


def relative_fes(f: np.ndarray) -> np.ndarray:
    return np.asarray(f, dtype=np.float64) - np.nanmin(f)


def to_kernel_cv(cv: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(cv, dtype=np.float64))
    if t.ndim == 1:
        t = t.reshape(-1, 1)
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


def kde_2d_window(
    rows: list[tuple[np.ndarray, np.ndarray]],
    *,
    i: int,
    j: int,
    ax: np.ndarray,
    lam: float,
    kT: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    g1, g2 = np.meshgrid(ax, ax, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    for cv, bw in rows:
        pair = np.stack([cv[:, i], cv[:, j]], axis=1)
        p_acc += density_on_grid_torus(
            grid_xy, pair, bw, chunk=chunk, device=device
        )
    p = np.clip(p_acc / float(len(rows)), 1e-300, None)
    return relative_fes((-np.log(p) / float(lam) * float(kT)).reshape(g1.shape))


def kde_1d_window(
    rows: list[tuple[np.ndarray, np.ndarray]],
    *,
    i: int,
    ax: np.ndarray,
    lam: float,
    kT: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    grid = np.asarray(ax, dtype=np.float64).reshape(-1, 1)
    p_acc = np.zeros(grid.shape[0], dtype=np.float64)
    for cv, bw in rows:
        sample = np.asarray(cv[:, i], dtype=np.float64).reshape(-1, 1)
        p_acc += density_on_grid_torus(
            grid, sample, bw, chunk=chunk, device=device
        )
    p = np.clip(p_acc / float(len(rows)), 1e-300, None)
    return relative_fes(-np.log(p) / float(lam) * float(kT))


def _fes_from_counts(counts: np.ndarray, *, kT: float) -> np.ndarray:
    tot = float(np.sum(counts))
    f = np.full(np.shape(counts), np.nan, dtype=np.float64)
    if tot <= 0.0:
        return f
    pos = np.asarray(counts, dtype=np.float64) > 0.0
    f[pos] = -float(kT) * np.log(np.asarray(counts, dtype=np.float64)[pos] / tot)
    return relative_fes(f)


def hist_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ax: np.ndarray,
    kT: float,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    edges = np.linspace(float(ax[0]), float(ax[-1]), int(ax.shape[0]) + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    counts = np.asarray(counts, dtype=np.float64)
    if float(smooth_sigma) > 0.0:
        counts = gaussian_filter(counts, sigma=float(smooth_sigma), mode="wrap")
    return _fes_from_counts(counts, kT=kT).T


def hist_1d(
    x: np.ndarray,
    *,
    ax: np.ndarray,
    kT: float,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    edges = np.linspace(float(ax[0]), float(ax[-1]), int(ax.shape[0]) + 1)
    counts, _ = np.histogram(np.asarray(x, dtype=np.float64), bins=edges)
    counts = np.asarray(counts, dtype=np.float64)
    if float(smooth_sigma) > 0.0:
        counts = gaussian_filter1d(counts, sigma=float(smooth_sigma), mode="wrap")
    return _fes_from_counts(counts, kT=kT)


def smooth_fes_torus(ax: np.ndarray, F: np.ndarray, *, sigma: float = 1.1) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    z = relative_fes(F)
    z = np.nan_to_num(z, nan=np.nanmax(z) if np.isfinite(z).any() else 0.0)
    return relative_fes(gaussian_filter(z, sigma=float(sigma), mode="wrap"))


def smooth_fes_1d(F: np.ndarray, *, sigma: float = 1.5) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    z = relative_fes(F)
    z = np.nan_to_num(z, nan=np.nanmax(z) if np.isfinite(z).any() else 0.0)
    return relative_fes(gaussian_filter1d(z, sigma=float(sigma), mode="wrap"))


def fes_cmap():
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(FES_CMAP).copy()
    cmap.set_over("white")
    cmap.set_bad("white")
    return cmap


def load_remd_phi3(
    remd_dir: Path,
    *,
    temperature: float = 300.0,
    discard_frac: float = 0.2,
) -> np.ndarray:
    temps = np.load(remd_dir / "temperatures.npy").astype(np.float64)
    dih = np.load(remd_dir / "traj_dihedral.npy")
    i_t = int(np.argmin(np.abs(temps - float(temperature))))
    n0 = int(np.floor(float(discard_frac) * dih.shape[0]))
    samples = np.asarray(dih[n0:, i_t, :], dtype=np.float64)
    if samples.shape[-1] == 6:
        samples = samples[:, 0:6:2]
    elif samples.shape[-1] != 3:
        raise ValueError(f"REMD dihedral last dim {samples.shape[-1]}, expected 3 or 6")
    print(
        f"[remd] T_slot={temps[i_t]:.3f} K idx={i_t}  "
        f"frames {n0}:{dih.shape[0]} → {samples.shape}",
        flush=True,
    )
    return samples


def remd_pair_fes(
    remd_phi3: np.ndarray,
    *,
    ax: np.ndarray,
    kT: float,
    smooth_sigma: float = REMD_HIST_SMOOTH_2D,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for i, j, _, _ in PHI_PAIRS:
        out[(i, j)] = hist_2d(
            remd_phi3[:, i], remd_phi3[:, j], ax=ax, kT=kT, smooth_sigma=smooth_sigma
        )
    return out


def remd_1d_fes(
    remd_phi3: np.ndarray,
    *,
    ax: np.ndarray,
    kT: float,
    smooth_sigma: float = REMD_HIST_SMOOTH_1D,
) -> np.ndarray:
    out = np.zeros((3, int(ax.shape[0])), dtype=np.float64)
    for i in range(3):
        out[i] = hist_1d(remd_phi3[:, i], ax=ax, kT=kT, smooth_sigma=smooth_sigma)
    return out


def _interp_periodic_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    x2 = np.concatenate([x[:-1] - 2.0 * np.pi, x, x[1:] + 2.0 * np.pi])
    y2 = np.concatenate([y[:-1], y, y[1:]])
    return np.interp(np.asarray(xq, dtype=np.float64), x2, y2)


def load_meta_phi23(
    path: Path, *, ax: np.ndarray, kT: float
) -> tuple[np.ndarray, np.ndarray]:
    """MetaD 2D F(φ2, φ3) CSV → grid F and Boltzmann 1D F(φ2), F(φ3).

    CSV: header ``other_cv_rad, x0, x1, ...`` (φ2); row labels are φ3.
    """
    import csv
    from scipy.interpolate import RegularGridInterpolator

    rows = list(csv.reader(Path(path).open()))
    if len(rows) < 3:
        raise ValueError(f"meta FES csv too short: {path}")
    x = np.array([float(v) for v in rows[0][1:]], dtype=np.float64)
    y = np.array([float(r[0]) for r in rows[1:]], dtype=np.float64)
    F = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float64)
    if F.shape != (y.shape[0], x.shape[0]):
        raise ValueError(f"meta FES shape {F.shape} != ({y.size}, {x.size})")
    if y[0] > y[-1]:
        y = y[::-1].copy()
        F = F[::-1, :]
    F = relative_fes(F)
    interp = RegularGridInterpolator(
        (y, x), F, bounds_error=False, fill_value=None
    )
    g1, g2 = np.meshgrid(ax, ax, indexing="xy")
    Fi = interp(np.stack([g2.ravel(), g1.ravel()], axis=-1)).reshape(g1.shape)
    Fi = relative_fes(Fi)
    w = np.exp(-np.clip(F, 0.0, 80.0) / float(kT))
    p2 = np.clip(w.sum(axis=0), 1e-300, None)
    p3 = np.clip(w.sum(axis=1), 1e-300, None)
    f1 = np.full((3, int(ax.shape[0])), np.nan, dtype=np.float64)
    f1[1] = relative_fes(_interp_periodic_1d(x, -float(kT) * np.log(p2), ax))
    f1[2] = relative_fes(_interp_periodic_1d(y, -float(kT) * np.log(p3), ax))
    print(f"[meta] {path}  grid x={x.size} y={y.size} → {Fi.shape}", flush=True)
    return Fi, f1


def _wrap_1d(xi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([xi, xi[:1] + 2.0 * np.pi]), np.concatenate([y, y[:1]])


def plot_fes_pairs(
    out_png: Path,
    *,
    ax: np.ndarray,
    f_sfpt: dict[tuple[int, int], np.ndarray],
    f_remd: dict[tuple[int, int], np.ndarray],
    f1_sfpt: np.ndarray,
    f1_remd: np.ndarray,
    vmax: float,
    title: str,
    f_meta: np.ndarray | None = None,
    f1_meta: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    has_meta = f_meta is not None
    n_map = 3 if has_meta else 2
    fig = plt.figure(figsize=(11.4 if has_meta else 8.8, 13.6))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[3.05, 0.92],
        hspace=0.18, left=0.08, right=0.92, bottom=0.05, top=0.94,
    )
    wr = [1.0] * n_map + [0.055]
    gs2 = outer[0].subgridspec(
        3, n_map + 1, width_ratios=wr, wspace=0.16, hspace=0.32,
    )
    gs1 = outer[1].subgridspec(1, 3, wspace=0.32)
    im = None

    def _draw_2d(a, F, *, sigma: float, ylab_on: bool, xlab: str, ylab: str, title_s: str):
        nonlocal im
        z = np.minimum(smooth_fes_torus(ax, F, sigma=sigma), float(vmax))
        im = a.contourf(ax, ax, z, levels=levels, cmap=cmap, norm=norm, extend="max")
        a.contour(
            ax, ax, np.clip(z, 0.0, float(vmax)),
            levels=levels[::3], colors="0.45", linewidths=0.35, alpha=0.45,
        )
        a.set_xlim(-np.pi, np.pi)
        a.set_ylim(-np.pi, np.pi)
        a.set_xticks(PI_TICKS)
        a.set_xticklabels(PI_LABELS)
        a.set_yticks(PI_TICKS)
        a.set_yticklabels(PI_LABELS if ylab_on else [])
        a.set_xlabel(xlab)
        if ylab_on:
            a.set_ylabel(ylab)
        a.set_title(title_s, fontsize=11)
        a.set_aspect("equal")
        a.tick_params(which="both", top=False, right=False)

    for row, (i, j, ii, jj) in enumerate(PHI_PAIRS):
        xlab = rf"$\phi_{ii}$ / rad"
        ylab = rf"$\phi_{jj}$ / rad"
        panels = [
            (f_sfpt[(i, j)], "sFPT", SFPT_SMOOTH_2D),
            (f_remd[(i, j)], "REMD", REMD_PLOT_SMOOTH_2D),
        ]
        for col, (F, lab, sigma) in enumerate(panels):
            a = fig.add_subplot(gs2[row, col])
            _draw_2d(
                a, F, sigma=sigma, ylab_on=(col == 0),
                xlab=xlab, ylab=ylab, title_s=(lab if row == 0 else ""),
            )
        if has_meta:
            a = fig.add_subplot(gs2[row, 2])
            if (i, j) == (1, 2):
                _draw_2d(
                    a, f_meta, sigma=SFPT_SMOOTH_2D, ylab_on=False,
                    xlab=xlab, ylab=ylab, title_s=("MetaD" if row == 0 else ""),
                )
            else:
                a.set_xlim(-np.pi, np.pi)
                a.set_ylim(-np.pi, np.pi)
                a.set_xticks(PI_TICKS)
                a.set_xticklabels(PI_LABELS)
                a.set_yticks(PI_TICKS)
                a.set_yticklabels([])
                a.set_xlabel(xlab)
                a.set_aspect("equal")
                a.tick_params(which="both", top=False, right=False)
                a.set_title("MetaD" if row == 0 else "", fontsize=11)
                a.text(
                    0.5, 0.5, "n/a\n" r"(no $\phi_1$)",
                    transform=a.transAxes, ha="center", va="center",
                    color="0.45", fontsize=10,
                )
                for spine in a.spines.values():
                    spine.set_color("0.8")
        fig.text(
            0.02, 0.82 - row * 0.215,
            rf"$(\phi_{ii},\phi_{jj})$",
            rotation=90, va="center", ha="center", fontsize=11,
        )
    cax = fig.add_subplot(gs2[:, n_map])
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(FE_LABEL)

    f1s = np.asarray(f1_sfpt, dtype=np.float64)
    f1r = np.asarray(f1_remd, dtype=np.float64)
    f1m = None if f1_meta is None else np.asarray(f1_meta, dtype=np.float64)
    for i in range(3):
        a = fig.add_subplot(gs1[0, i])
        yr = relative_fes(f1r[i])
        ys = relative_fes(f1s[i])
        xr, yr = _wrap_1d(ax, yr)
        xs, ys = _wrap_1d(ax, ys)
        a.plot(xr, yr, color=COLOR_REMD, lw=1.55, ls="--", label="REMD", zorder=2)
        a.plot(xs, ys, color=COLOR_SFPT, lw=1.7, label="sFPT", zorder=3)
        if f1m is not None and np.isfinite(f1m[i]).any():
            ym = relative_fes(f1m[i])
            xm, ym = _wrap_1d(ax, ym)
            a.plot(xm, ym, color=COLOR_META, lw=1.65, ls="-.", label="MetaD", zorder=4)
        a.set_xlim(-np.pi, np.pi)
        a.set_ylim(0.0, float(vmax) * 1.02)
        a.set_xticks(PI_TICKS)
        a.set_xticklabels(PI_LABELS)
        a.set_xlabel(rf"$\phi_{i + 1}$ / rad")
        if i == 0:
            a.set_ylabel(FE_LABEL)
        if i == (1 if has_meta else 0):
            a.legend(frameon=False, fontsize=8, loc="upper right")
        a.set_title(rf"$F(\phi_{i + 1})$", fontsize=11)
        a.tick_params(which="both", top=False, right=False)
        a.axvline(0.0, color="0.8", lw=0.6, ls=":", zorder=1)

    fig.suptitle(title, fontsize=12)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fes] wrote {out_png}", flush=True)


def _npz_2d_dict(z) -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 1): np.asarray(z["F_phi1_phi2"], dtype=np.float64),
        (0, 2): np.asarray(z["F_phi1_phi3"], dtype=np.float64),
        (1, 2): np.asarray(z["F_phi2_phi3"], dtype=np.float64),
    }


def _infer_dump_dt_ps(run_dir: Path) -> float:
    log = run_dir / "stage_log.csv"
    if log.is_file():
        with log.open() as fh:
            header = fh.readline()
            row = fh.readline().strip().split(",")
        if len(row) >= 2:
            return float(row[1])
        del header
    return 0.2


def _infer_window_ps(fes_dir: Path) -> float:
    times = sorted(
        int(p.stem.replace("fes_2d_t", "").replace("ps", ""))
        for p in fes_dir.glob("fes_2d_t*ps.npz")
    )
    if len(times) >= 2:
        return float(times[1] - times[0])
    if times:
        return float(times[0])
    return 5.0


def load_window_rows(
    run_dir: Path, *, t_ps: float, window_ps: float, dump_dt_ps: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_win = max(1, int(round(float(window_ps) / float(dump_dt_ps))))
    last = int(round(float(t_ps) / float(dump_dt_ps))) - 1
    want = set(range(last - n_win + 1, last + 1))
    found: dict[int, tuple[Path, Path]] = {}
    for dih in run_dir.glob("stage*_dihedral.npy"):
        m = STAGE_DIH_RE.search(dih.name)
        if m is None:
            continue
        si = int(m.group(1))
        if si not in want:
            continue
        bw = dih.with_name(dih.name.replace("_dihedral.npy", "_bandwidth.npy"))
        if bw.is_file():
            found[si] = (dih, bw)
    missing = sorted(want - set(found))
    if missing:
        raise FileNotFoundError(
            f"{run_dir}: missing dumps for t={t_ps:g} ps window: {missing[:8]}..."
        )
    rows = []
    for si in sorted(found):
        cv = np.load(found[si][0])
        if cv.ndim == 3:
            cv = cv.reshape(cv.shape[0], -1)
        rows.append((cv, np.load(found[si][1])))
    return rows


def replot_npz(
    npz_path: Path,
    *,
    run_dir: Path,
    remd_2d: dict[tuple[int, int], np.ndarray],
    remd_1d: np.ndarray,
    window_ps: float,
    dump_dt_ps: float,
    kT: float,
    vmax: float,
    device: torch.device,
    chunk: int,
    force: bool = False,
    meta_2d: np.ndarray | None = None,
    meta_1d: np.ndarray | None = None,
) -> bool:
    z = np.load(npz_path)
    t_ps = float(z["t_ps"])
    lam = float(z["lam"])
    ax = np.asarray(z["ax"], dtype=np.float64)
    png = npz_path.with_suffix(".png")
    has_1d = "F1_phi1" in z.files
    has_meta = "F_meta_phi2_phi3" in z.files
    if (
        not force
        and has_1d
        and (meta_2d is None or has_meta)
        and png.is_file()
        and png.stat().st_mtime >= npz_path.stat().st_mtime
    ):
        return False
    if has_1d:
        f1 = np.stack(
            [
                np.asarray(z["F1_phi1"], dtype=np.float64),
                np.asarray(z["F1_phi2"], dtype=np.float64),
                np.asarray(z["F1_phi3"], dtype=np.float64),
            ]
        )
    else:
        rows = load_window_rows(
            run_dir, t_ps=t_ps, window_ps=window_ps, dump_dt_ps=dump_dt_ps
        )
        f1 = np.stack(
            [
                kde_1d_window(
                    rows, i=i, ax=ax, lam=lam, kT=kT, device=device, chunk=chunk
                )
                for i in range(3)
            ]
        )
    f_sfpt = _npz_2d_dict(z)
    save_kw = dict(
        ax=ax,
        t_ps=np.float64(t_ps),
        lam=np.float64(lam),
        F_phi1_phi2=f_sfpt[(0, 1)],
        F_phi1_phi3=f_sfpt[(0, 2)],
        F_phi2_phi3=f_sfpt[(1, 2)],
        F_remd_phi1_phi2=remd_2d[(0, 1)],
        F_remd_phi1_phi3=remd_2d[(0, 2)],
        F_remd_phi2_phi3=remd_2d[(1, 2)],
        F1_phi1=f1[0],
        F1_phi2=f1[1],
        F1_phi3=f1[2],
        F1_remd_phi1=remd_1d[0],
        F1_remd_phi2=remd_1d[1],
        F1_remd_phi3=remd_1d[2],
    )
    if meta_2d is not None:
        save_kw["F_meta_phi2_phi3"] = meta_2d
        save_kw["F1_meta_phi1"] = meta_1d[0]
        save_kw["F1_meta_phi2"] = meta_1d[1]
        save_kw["F1_meta_phi3"] = meta_1d[2]
    np.savez_compressed(npz_path, **save_kw)
    vs = "sFPT vs REMD" + (" vs MetaD" if meta_2d is not None else "")
    plot_fes_pairs(
        png,
        ax=ax,
        f_sfpt=f_sfpt,
        f_remd=remd_2d,
        f1_sfpt=f1,
        f1_remd=remd_1d,
        f_meta=meta_2d,
        f1_meta=meta_1d,
        vmax=float(vmax),
        title=(
            rf"{vs}  last {window_ps:g} ps  "
            rf"($t={t_ps:g}$ ps, $\lambda={lam:g}$)"
        ),
    )
    return True


def replot_run(
    run_dir: Path,
    *,
    remd_dir: Path,
    temperature: float = 300.0,
    vmax: float = 16.0,
    kB: float = 0.00198720425864083,
    chunk: int = 2048,
    device: str = "cpu",
    force: bool = False,
    meta_csv: Path | None = None,
) -> int:
    fes_dir = run_dir / "fes_2d"
    npzs = sorted(fes_dir.glob("fes_2d_t*ps.npz"))
    if not npzs:
        print(f"[fes] no npz in {fes_dir}", flush=True)
        return 0
    dump_dt = _infer_dump_dt_ps(run_dir)
    window_ps = _infer_window_ps(fes_dir)
    kT = float(kB) * float(temperature)
    ax0 = np.asarray(np.load(npzs[0])["ax"], dtype=np.float64)
    remd_phi = load_remd_phi3(remd_dir, temperature=temperature)
    remd_2d = remd_pair_fes(remd_phi, ax=ax0, kT=kT)
    remd_1d = remd_1d_fes(remd_phi, ax=ax0, kT=kT)
    meta_2d = meta_1d = None
    if meta_csv is not None and Path(meta_csv).is_file():
        meta_2d, meta_1d = load_meta_phi23(Path(meta_csv), ax=ax0, kT=kT)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    n_ok = 0
    for p in npzs:
        try:
            if replot_npz(
                p,
                run_dir=run_dir,
                remd_2d=remd_2d,
                remd_1d=remd_1d,
                window_ps=window_ps,
                dump_dt_ps=dump_dt,
                kT=kT,
                vmax=vmax,
                device=dev,
                chunk=chunk,
                force=force,
                meta_2d=meta_2d,
                meta_1d=meta_1d,
            ):
                n_ok += 1
        except Exception as exc:
            print(f"[fes] skip {p.name}: {exc}", flush=True)
    return n_ok


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="Replot tetra-alanine 2D+1D FES")
    p.add_argument("--run-dir", type=Path, required=True, action="append")
    p.add_argument(
        "--remd-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "remd_280_400_n64_90605",
    )
    p.add_argument("--vmax", type=float, default=16.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--force", action="store_true")
    p.add_argument("--watch", type=float, default=0.0, help="Rescan every N seconds")
    p.add_argument("--meta-fes", type=Path, default=DEFAULT_META_CSV)
    args = p.parse_args()
    force = bool(args.force)
    meta = Path(args.meta_fes) if args.meta_fes else None
    while True:
        n = 0
        for run in args.run_dir:
            n += replot_run(
                Path(run),
                remd_dir=Path(args.remd_dir),
                temperature=float(args.temperature),
                vmax=float(args.vmax),
                chunk=int(args.chunk),
                device=str(args.device),
                force=force,
                meta_csv=meta,
            )
        force = False
        if float(args.watch) <= 0.0:
            break
        print(f"[fes] watch sleep {args.watch:g}s  redrew={n}", flush=True)
        time.sleep(float(args.watch))


if __name__ == "__main__":
    main()

