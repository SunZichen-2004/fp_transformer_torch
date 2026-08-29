#!/usr/bin/env python3
"""Compare MLE FES (FPT) vs well-tempered MetaD FES on (d1, d2).

Also projects a 1D FES along ξ = d2 − d1. After evaluation, grids are saved to
``.npz`` so later runs can use ``--from-grid`` without reloading the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fp_transformer_torch.reconstruct_FES.train_max_likelihood_torch import (  # noqa: E402
    evaluate_free_energy,
    load_free_energy_model,
)

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
NATURE_DOUBLE_COL_MM = 180.0


def nature_rcparams() -> dict[str, object]:
    """Double-column figure: sans-serif, enlarged type, outward ticks."""
    mm = 1.0 / 25.4
    double_col_in = NATURE_DOUBLE_COL_MM * mm
    return {
        "figure.figsize": (double_col_in, double_col_in * 0.62),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.25,
        "ytick.major.size": 3.25,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    }


def kt_to_kcal(f_kt: np.ndarray, temperature: float) -> np.ndarray:
    return f_kt * (KB_EV_PER_K * float(temperature) * EV_TO_KCAL)


def ev_to_kcal(f_ev: np.ndarray) -> np.ndarray:
    return f_ev * EV_TO_KCAL


def kT_kcal(temperature: float) -> float:
    return KB_EV_PER_K * float(temperature) * EV_TO_KCAL


def relative_fes(f: np.ndarray, vmax: float | None = None) -> np.ndarray:
    out = np.asarray(f, dtype=np.float64) - np.nanmin(f)
    if vmax is not None and float(vmax) > 0:
        out = np.clip(out, 0.0, float(vmax))
    return out


def project_fes_1d_d2_minus_d1(
    d1: np.ndarray,
    d2: np.ndarray,
    f_kcal: np.ndarray,
    *,
    temperature: float,
    n_bins: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann projection of 2D FES onto ξ = d2 − d1 (kcal/mol, min=0).

    ``F(ξ) = −kT ln ∫ exp(−β F(d1,d2)) δ(ξ − (d2−d1)) dd1 dd2``
    approximated by binning the rectangular grid (equal cell weights).
    """
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    xi = (g2 - g1).ravel()
    f = np.asarray(f_kcal, dtype=np.float64).ravel()
    mask = np.isfinite(f) & np.isfinite(xi)
    xi, f = xi[mask], f[mask]

    xi_min, xi_max = float(xi.min()), float(xi.max())
    edges = np.linspace(xi_min, xi_max, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kt = kT_kcal(temperature)
    beta = 1.0 / kt

    f1d = np.full(centers.shape, np.nan, dtype=np.float64)
    # stable: F_bin = -kT logsumexp(-β F) over cells in bin
    for i in range(len(centers)):
        sel = (xi >= edges[i]) & (xi < edges[i + 1] if i < len(centers) - 1 else xi <= edges[i + 1])
        if not np.any(sel):
            continue
        vals = -beta * f[sel]
        m = float(np.max(vals))
        f1d[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))

    finite = np.isfinite(f1d)
    if not np.any(finite):
        raise RuntimeError("1D projection produced no finite bins")
    f1d = f1d - np.nanmin(f1d[finite])
    return centers, f1d


def compute_grids(args: argparse.Namespace) -> dict[str, np.ndarray | float | str]:
    model, ckpt = load_free_energy_model(args.ckpt, device=args.device)
    temperature = float(ckpt.get("temperature", 300.0))

    metad = np.load(args.metad_fes)
    d1_m = np.asarray(metad["d1"], dtype=np.float64)
    d2_m = np.asarray(metad["d2"], dtype=np.float64)
    f_metad_raw = relative_fes(ev_to_kcal(np.asarray(metad["F"], dtype=np.float64)), None)

    if args.grid_source == "metad":
        d1, d2 = d1_m, d2_m
    else:
        lo = np.asarray(ckpt["bounds_lo"], dtype=np.float64)
        hi = np.asarray(ckpt["bounds_hi"], dtype=np.float64)
        d1 = np.linspace(lo[0], hi[0], int(args.n_grid))
        d2 = np.linspace(lo[1], hi[1], int(args.n_grid))

    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    cv_grid = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    f_mle = relative_fes(
        kt_to_kcal(
            evaluate_free_energy(model, ckpt, cv_grid, device=args.device).reshape(g1.shape),
            temperature,
        ),
        None,
    )

    if args.grid_source == "metad":
        f_meta = f_metad_raw
        if f_meta.shape != f_mle.shape:
            if f_meta.T.shape == f_mle.shape:
                f_meta = f_meta.T
            else:
                raise ValueError(f"shape mismatch MetaD {f_meta.shape} vs MLE {f_mle.shape}")
    else:
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (d2_m, d1_m), f_metad_raw, bounds_error=False, fill_value=np.nan
        )
        pts = np.stack([g2.ravel(), g1.ravel()], axis=-1)
        f_meta = relative_fes(interp(pts).reshape(g1.shape), None)

    xi_all = (g2 - g1).ravel()
    edges = np.linspace(float(xi_all.min()), float(xi_all.max()), int(args.n_bins_1d) + 1)
    xi_c = 0.5 * (edges[:-1] + edges[1:])

    def _project_fixed(f2d: np.ndarray) -> np.ndarray:
        xi_flat = (g2 - g1).ravel()
        f_flat = np.asarray(f2d, dtype=np.float64).ravel()
        kt = kT_kcal(temperature)
        beta = 1.0 / kt
        out = np.full(xi_c.shape, np.nan, dtype=np.float64)
        for i in range(len(xi_c)):
            if i < len(xi_c) - 1:
                sel = (xi_flat >= edges[i]) & (xi_flat < edges[i + 1])
            else:
                sel = (xi_flat >= edges[i]) & (xi_flat <= edges[i + 1])
            if not np.any(sel):
                continue
            vals = -beta * f_flat[sel]
            m = float(np.max(vals))
            out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
        finite = np.isfinite(out)
        out = out - np.nanmin(out[finite])
        return out

    f1d_mle = _project_fixed(f_mle)
    f1d_meta = _project_fixed(f_meta)

    return {
        "d1": np.asarray(d1, dtype=np.float64),
        "d2": np.asarray(d2, dtype=np.float64),
        "F_mle_kcal": np.asarray(f_mle, dtype=np.float64),
        "F_metad_kcal": np.asarray(f_meta, dtype=np.float64),
        "xi_d2_minus_d1": np.asarray(xi_c, dtype=np.float64),
        "F1d_mle_kcal": np.asarray(f1d_mle, dtype=np.float64),
        "F1d_metad_kcal": np.asarray(f1d_meta, dtype=np.float64),
        "temperature": float(temperature),
        "unit": "kcal/mol",
        "note": "2D F and 1D F are min-subtracted; not clipped",
    }


def load_grids(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    # scalars saved as 0-d arrays
    if "temperature" in data:
        data["temperature"] = float(np.asarray(data["temperature"]))
    return data


def mask_both_gt_and_reproject(
    data: dict,
    *,
    d1_gt: float,
    d2_gt: float,
    n_bins_1d: int = 80,
) -> dict:
    """Drop grid cells with ``d1 > d1_gt`` AND ``d2 > d2_gt``, then rebuild 1D FES."""
    d1 = np.asarray(data["d1"], dtype=np.float64)
    d2 = np.asarray(data["d2"], dtype=np.float64)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    drop = (g1 > float(d1_gt)) & (g2 > float(d2_gt))
    n_drop = int(np.count_nonzero(drop))
    print(
        f"[mask] remove cells with d1>{d1_gt:g} and d2>{d2_gt:g}: "
        f"{n_drop}/{drop.size}",
        flush=True,
    )

    f_mle = np.asarray(data["F_mle_kcal"], dtype=np.float64).copy()
    f_meta = np.asarray(data["F_metad_kcal"], dtype=np.float64).copy()
    f_mle[drop] = np.nan
    f_meta[drop] = np.nan
    # re-zero after masking
    f_mle = f_mle - np.nanmin(f_mle)
    f_meta = f_meta - np.nanmin(f_meta)

    temperature = float(data.get("temperature", 300.0))
    xi_flat = (g2 - g1).ravel()
    finite_xi = np.isfinite(xi_flat) & (~drop.ravel())
    edges = np.linspace(
        float(xi_flat[finite_xi].min()),
        float(xi_flat[finite_xi].max()),
        int(n_bins_1d) + 1,
    )
    xi_c = 0.5 * (edges[:-1] + edges[1:])

    def _project(f2d: np.ndarray) -> np.ndarray:
        f_flat = np.asarray(f2d, dtype=np.float64).ravel()
        kt = kT_kcal(temperature)
        beta = 1.0 / kt
        out = np.full(xi_c.shape, np.nan, dtype=np.float64)
        for i in range(len(xi_c)):
            if i < len(xi_c) - 1:
                sel = (xi_flat >= edges[i]) & (xi_flat < edges[i + 1])
            else:
                sel = (xi_flat >= edges[i]) & (xi_flat <= edges[i + 1])
            sel = sel & np.isfinite(f_flat)
            if not np.any(sel):
                continue
            vals = -beta * f_flat[sel]
            m = float(np.max(vals))
            out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
        finite = np.isfinite(out)
        out = out - np.nanmin(out[finite])
        return out

    out = dict(data)
    out["F_mle_kcal"] = f_mle
    out["F_metad_kcal"] = f_meta
    out["xi_d2_minus_d1"] = xi_c
    out["F1d_mle_kcal"] = _project(f_mle)
    out["F1d_metad_kcal"] = _project(f_meta)
    return out


def gaussian_smooth_and_reproject(
    data: dict,
    *,
    sigma: float,
    n_bins_1d: int = 80,
    sigma_1d: float | None = None,
) -> dict:
    """Light Gaussian blur on 2D FES (grid units), then rebuild 1D projection."""
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    if float(sigma) <= 0.0:
        return data

    d1 = np.asarray(data["d1"], dtype=np.float64)
    d2 = np.asarray(data["d2"], dtype=np.float64)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    temperature = float(data.get("temperature", 300.0))

    def _smooth2d(f: np.ndarray) -> np.ndarray:
        arr = np.asarray(f, dtype=np.float64)
        mask = np.isfinite(arr)
        fill = float(np.nanmean(arr[mask])) if np.any(mask) else 0.0
        filled = np.where(mask, arr, fill)
        sm = gaussian_filter(filled, sigma=float(sigma), mode="nearest")
        sm = np.where(mask, sm, np.nan)
        return sm - np.nanmin(sm)

    f_mle = _smooth2d(data["F_mle_kcal"])
    f_meta = _smooth2d(data["F_metad_kcal"])
    print(f"[smooth] 2D gaussian sigma={sigma:g} (grid units)", flush=True)

    xi_flat = (g2 - g1).ravel()
    valid = np.isfinite(f_mle.ravel())
    edges = np.linspace(
        float(xi_flat[valid].min()),
        float(xi_flat[valid].max()),
        int(n_bins_1d) + 1,
    )
    xi_c = 0.5 * (edges[:-1] + edges[1:])

    def _project(f2d: np.ndarray) -> np.ndarray:
        f_flat = np.asarray(f2d, dtype=np.float64).ravel()
        kt = kT_kcal(temperature)
        beta = 1.0 / kt
        out = np.full(xi_c.shape, np.nan, dtype=np.float64)
        for i in range(len(xi_c)):
            if i < len(xi_c) - 1:
                sel = (xi_flat >= edges[i]) & (xi_flat < edges[i + 1])
            else:
                sel = (xi_flat >= edges[i]) & (xi_flat <= edges[i + 1])
            sel = sel & np.isfinite(f_flat)
            if not np.any(sel):
                continue
            vals = -beta * f_flat[sel]
            m = float(np.max(vals))
            out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
        finite = np.isfinite(out)
        out = out - np.nanmin(out[finite])
        return out

    f1d_mle = _project(f_mle)
    f1d_meta = _project(f_meta)
    s1 = float(sigma if sigma_1d is None else sigma_1d)
    if s1 > 0.0:
        for key, arr in (("mle", f1d_mle), ("meta", f1d_meta)):
            m = np.isfinite(arr)
            fill = float(np.nanmean(arr[m]))
            sm = gaussian_filter1d(np.where(m, arr, fill), sigma=s1, mode="nearest")
            sm = np.where(m, sm, np.nan)
            sm = sm - np.nanmin(sm[np.isfinite(sm)])
            if key == "mle":
                f1d_mle = sm
            else:
                f1d_meta = sm
        print(f"[smooth] 1D gaussian sigma={s1:g}", flush=True)

    out = dict(data)
    out["F_mle_kcal"] = f_mle
    out["F_metad_kcal"] = f_meta
    out["xi_d2_minus_d1"] = xi_c
    out["F1d_mle_kcal"] = f1d_mle
    out["F1d_metad_kcal"] = f1d_meta
    return out


def save_grids(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        d1=data["d1"],
        d2=data["d2"],
        F_mle_kcal=data["F_mle_kcal"],
        F_metad_kcal=data["F_metad_kcal"],
        xi_d2_minus_d1=data["xi_d2_minus_d1"],
        F1d_mle_kcal=data["F1d_mle_kcal"],
        F1d_metad_kcal=data["F1d_metad_kcal"],
        temperature=np.asarray(data["temperature"], dtype=np.float64),
    )
    print(f"[ok] saved grids {path}", flush=True)


FE_LABEL = r"free energy / kcal mol$^{-1}$"
COLOR_META = "#2166ac"
COLOR_SFPT = "#e67e22"


def _rdbu_r_white_over():
    """RdBu_r (blue→white→red); values above vmax drawn white."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_over("white")
    cmap.set_bad("white")
    return cmap


def _parse_lim(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    vals = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"limit must be lo,hi got {text!r}")
    return float(vals[0]), float(vals[1])


def _savefig(fig, out: Path) -> None:
    """Save PNG+PDF at the figure's nominal size (double-column friendly)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # override global savefig.bbox=tight so width stays NATURE_DOUBLE_COL_MM
    kw = dict(bbox_inches=None, pad_inches=0.02)
    fig.savefig(out, **kw)
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, **kw)
    print(f"[ok] wrote {out}", flush=True)
    print(f"[ok] wrote {pdf}", flush=True)


def _set_panel_titles(
    ax,
    panel: str | None,
    title: str | None = None,
) -> None:
    """Left panel tag '(a)' etc.; optional centered method title."""
    # nonempty center title (or a blank) keeps identical top padding across panels
    ax.set_title(title if title else " ", loc="center", pad=6)
    if panel:
        # same left-title anchor as before; method name uses centered title
        ax.text(
            0.0,
            1.02,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=plt.rcParams["axes.titlesize"],
            clip_on=False,
        )


def _draw_fes2d_on_ax(
    ax,
    d1: np.ndarray,
    d2: np.ndarray,
    f: np.ndarray,
    *,
    levels: np.ndarray,
    norm: Normalize,
    cmap,
    vmax: float | None,
    panel: str | None = None,
    title: str | None = None,
    show_ylabel: bool = True,
    box_aspect: bool = True,
    xlabel: str = r"$d_1$ / Å",
    ylabel: str = r"$d_2$ / Å",
    d1_lim: tuple[float, float] | None = None,
    d2_lim: tuple[float, float] | None = None,
):
    vmin = float(levels[0])
    vmax_plot = float(levels[-1])
    f_ma = np.ma.masked_invalid(np.asarray(f, dtype=np.float64))
    im = ax.contourf(
        d1,
        d2,
        f_ma,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max" if vmax is not None else "neither",
    )
    ax.contour(
        d1,
        d2,
        np.clip(f_ma.filled(vmax_plot), vmin, vmax_plot),
        levels=levels[::2],
        colors="0.25",
        linewidths=0.3,
        alpha=0.35,
    )
    x0, x1 = d1_lim if d1_lim is not None else (float(d1.min()), float(d1.max()))
    y0, y1 = d2_lim if d2_lim is not None else (float(d2.min()), float(d2.max()))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    _set_panel_titles(ax, panel=panel, title=title)
    if box_aspect:
        ax.set_box_aspect(1)
    ax.set_aspect("auto")
    ax.tick_params(which="both", top=False, right=False)
    return im


def _draw_fes1d_on_ax(
    ax,
    xi: np.ndarray,
    f1d_mle: np.ndarray,
    f1d_meta: np.ndarray,
    *,
    vmax: float | None,
    xi_lim: tuple[float, float] | None,
    panel: str | None = None,
    title: str | None = None,
    box_aspect: bool = True,
    method_label: str = "sFPT",
):
    y_sfpt = relative_fes(f1d_mle, vmax)
    y_meta = relative_fes(f1d_meta, vmax)
    ymax = float(vmax) if vmax is not None else float(
        max(np.nanmax(y_sfpt), np.nanmax(y_meta))
    )
    ax.plot(xi, y_meta, color=COLOR_META, lw=1.4, label="WT-MetaD", zorder=2)
    ax.fill_between(xi, 0.0, y_meta, color=COLOR_META, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(xi, y_sfpt, color=COLOR_SFPT, lw=1.5, label=method_label, zorder=3)
    ax.set_xlabel(r"$d_2 - d_1$ / Å")
    ax.set_ylabel(FE_LABEL)
    if xi_lim is not None:
        ax.set_xlim(xi_lim[0], xi_lim[1])
    else:
        ax.set_xlim(float(np.nanmin(xi)), float(np.nanmax(xi)))
    ax.set_ylim(0.0, ymax * 1.05 if ymax > 0 else 1.0)
    _set_panel_titles(ax, panel=panel, title=title)
    if box_aspect:
        ax.set_box_aspect(1)
    ax.tick_params(which="both", top=False, right=False)
    ax.axhline(0.0, color="0.5", lw=0.5)
    ax.legend(loc="best", handlelength=1.6)


def plot_2d(
    d1: np.ndarray,
    d2: np.ndarray,
    f_mle: np.ndarray,
    f_meta: np.ndarray,
    *,
    vmax: float | None,
    n_levels: int,
    out: Path,
    method_label: str = "sFPT",
) -> None:
    f_mle_p = relative_fes(f_mle, None)
    f_meta_p = relative_fes(f_meta, None)
    vmin = 0.0
    vmax_plot = float(vmax) if vmax is not None else float(
        max(np.nanmax(f_mle_p), np.nanmax(f_meta_p))
    )
    levels = np.linspace(vmin, vmax_plot, int(n_levels))
    norm = Normalize(vmin=vmin, vmax=vmax_plot, clip=False)
    cmap = _rdbu_r_white_over()

    mm = 1.0 / 25.4
    w = NATURE_DOUBLE_COL_MM * mm
    fig = plt.figure(figsize=(w, w * 0.48))
    # a | gap | b | cbar (cbar height matched to b below)
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.0, 0.12, 1.0, 0.06],
        wspace=0.10,
        left=0.08,
        right=0.92,
        bottom=0.14,
        top=0.88,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    im = _draw_fes2d_on_ax(
        ax0,
        d1,
        d2,
        f_mle_p,
        levels=levels,
        norm=norm,
        cmap=cmap,
        vmax=vmax,
        panel="(a)",
        title=method_label,
    )
    _draw_fes2d_on_ax(
        ax1,
        d1,
        d2,
        f_meta_p,
        levels=levels,
        norm=norm,
        cmap=cmap,
        vmax=vmax,
        panel="(b)",
        title="WT-MetaD",
        show_ylabel=False,
    )
    fig.canvas.draw()
    pos_b = ax1.get_position()
    cbar_w = 0.018
    cax.set_position([pos_b.x1 + 0.012, pos_b.y0, cbar_w, pos_b.height])
    cbar = fig.colorbar(
        im, cax=cax, extend="max" if vmax is not None else "neither"
    )
    # keep label clear of tick numbers
    cbar.set_label(FE_LABEL, labelpad=10)
    cbar.ax.tick_params(width=0.6, length=3.25, direction="out", pad=2)
    cbar.outline.set_linewidth(0.6)
    _savefig(fig, out)
    plt.close(fig)


def plot_1d(
    xi: np.ndarray,
    f1d_mle: np.ndarray,
    f1d_meta: np.ndarray,
    *,
    vmax: float | None,
    out: Path,
    xi_lim: tuple[float, float] | None = None,
    method_label: str = "sFPT",
) -> None:
    mm = 1.0 / 25.4
    side = NATURE_DOUBLE_COL_MM * mm * 0.42
    fig = plt.figure(figsize=(side, side))
    ax = fig.add_subplot(111)
    _draw_fes1d_on_ax(
        ax,
        xi,
        f1d_mle,
        f1d_meta,
        vmax=vmax,
        xi_lim=xi_lim,
        title="1D FES",
        method_label=method_label,
    )
    fig.tight_layout()
    _savefig(fig, out)
    plt.close(fig)


def plot_combined_1x3(
    d1: np.ndarray,
    d2: np.ndarray,
    f_mle: np.ndarray,
    f_meta: np.ndarray,
    xi: np.ndarray,
    f1d_mle: np.ndarray,
    f1d_meta: np.ndarray,
    *,
    vmax: float | None,
    n_levels: int,
    xi_lim: tuple[float, float] | None,
    out: Path,
    d1_lim: tuple[float, float] | None = None,
    d2_lim: tuple[float, float] | None = None,
    xlabel: str = r"$d_1$ / Å",
    ylabel: str = r"$d_2$ / Å",
    method_label: str = "sFPT",
) -> None:
    """Double-column 1×3: method 2D | MetaD 2D (+cbar) | 1D FES."""
    # smaller type + larger panels for the combined figure only
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    ):
        f_mle_p = relative_fes(f_mle, None)
        f_meta_p = relative_fes(f_meta, None)
        vmin = 0.0
        vmax_plot = float(vmax) if vmax is not None else float(
            max(np.nanmax(f_mle_p), np.nanmax(f_meta_p))
        )
        levels = np.linspace(vmin, vmax_plot, int(n_levels))
        norm = Normalize(vmin=vmin, vmax=vmax_plot, clip=False)
        cmap = _rdbu_r_white_over()

        mm = 1.0 / 25.4
        fig_w = NATURE_DOUBLE_COL_MM * mm
        # taller canvas so equal square panels fill more of the double-col width
        fig_h = fig_w * 0.48
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax0 = fig.add_axes([0, 0, 1, 1])
        ax1 = fig.add_axes([0, 0, 1, 1])
        ax2 = fig.add_axes([0, 0, 1, 1])
        cax = fig.add_axes([0, 0, 1, 1])

        # Manual equal-square layout; tight margins → larger panels
        left, right = 0.055, 0.995
        bottom, top = 0.14, 0.90
        gap_ab = 0.035
        cbar_pad = 0.010
        cbar_w = 0.012
        cbar_label_w = 0.100
        gap_bc = 0.115
        reserved = gap_ab + cbar_pad + cbar_w + cbar_label_w + gap_bc
        w_p = (right - left - reserved) / 3.0
        h_p = w_p * fig_w / fig_h  # square in display inches
        h_avail = top - bottom
        if h_p > h_avail:
            scale = h_avail / h_p
            h_p *= scale
            w_p *= scale
        y0 = bottom + 0.5 * (h_avail - h_p)

        x0 = left
        x1 = x0 + w_p + gap_ab
        x_cbar = x1 + w_p + cbar_pad
        x2 = x_cbar + cbar_w + cbar_label_w + gap_bc

        ax0.set_position([x0, y0, w_p, h_p])
        ax1.set_position([x1, y0, w_p, h_p])
        cax.set_position([x_cbar, y0, cbar_w, h_p])
        ax2.set_position([x2, y0, w_p, h_p])

        im = _draw_fes2d_on_ax(
            ax0,
            d1,
            d2,
            f_mle_p,
            levels=levels,
            norm=norm,
            cmap=cmap,
            vmax=vmax,
            panel="(a)",
            title=method_label,
            box_aspect=False,
            xlabel=xlabel,
            ylabel=ylabel,
            d1_lim=d1_lim,
            d2_lim=d2_lim,
        )
        _draw_fes2d_on_ax(
            ax1,
            d1,
            d2,
            f_meta_p,
            levels=levels,
            norm=norm,
            cmap=cmap,
            vmax=vmax,
            panel="(b)",
            title="WT-MetaD",
            show_ylabel=False,
            box_aspect=False,
            xlabel=xlabel,
            ylabel=ylabel,
            d1_lim=d1_lim,
            d2_lim=d2_lim,
        )
        _draw_fes1d_on_ax(
            ax2,
            xi,
            f1d_mle,
            f1d_meta,
            vmax=vmax,
            xi_lim=xi_lim,
            panel="(c)",
            title=None,
            box_aspect=False,
            method_label=method_label,
        )
        ax1.tick_params(labelleft=False)
        ax2.yaxis.set_label_coords(-0.18, 0.5)
        cbar = fig.colorbar(
            im, cax=cax, extend="max" if vmax is not None else "neither"
        )
        cbar.set_ticks(np.linspace(0.0, float(vmax_plot), 5))
        cbar.ax.tick_params(width=0.6, length=2.8, direction="out", pad=2)
        cbar.outline.set_linewidth(0.6)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        x_right = float(cax.get_position().x1)
        for t in cbar.ax.get_yticklabels():
            if not t.get_text():
                continue
            x_right = max(
                x_right, float(inv.transform(t.get_window_extent(renderer).p1)[0])
            )
        fig.text(
            x_right + 0.028,
            y0 + 0.5 * h_p,
            FE_LABEL,
            rotation=90,
            va="center",
            ha="left",
            fontsize=plt.rcParams["axes.labelsize"],
        )
        _savefig(fig, out)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_mace/results/claisen_fpt_mh-0_g4_85889"
            "/fes_mle/fes_mle.pt"
        ),
    )
    p.add_argument(
        "--metad-fes",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_metadynamic/results/claisen_metad_mh-0_r32_85888"
            "/live/fes.npz"
        ),
    )
    p.add_argument(
        "--from-grid",
        type=Path,
        default=None,
        help="Load previously saved fes_grids.npz (skip model evaluation).",
    )
    p.add_argument(
        "--grid-npz",
        type=Path,
        default=None,
        help="Where to save grids (default: <out_dir>/fes_grids.npz).",
    )
    p.add_argument("--output", type=Path, default=None, help="2D compare PNG path.")
    p.add_argument(
        "--output-1d",
        type=Path,
        default=None,
        help="1D d2−d1 compare PNG (default: sibling of --output).",
    )
    p.add_argument(
        "--output-combined",
        type=Path,
        default=None,
        help="1×3 combined figure (default: sibling fes_compare_combined.png).",
    )
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument(
        "--xi-lim",
        type=str,
        default=None,
        help="1D x-axis lo,hi for d2−d1 (Å), e.g. '-2.5,4.5'.",
    )
    p.add_argument(
        "--mask-both-gt",
        type=str,
        default=None,
        help="Mask cells with d1>A and d2>B, e.g. '2.7,2.7'; reproject 1D after mask.",
    )
    p.add_argument(
        "--gauss-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma on 2D FES in grid-index units (0=off). Also lightly smooths 1D.",
    )
    p.add_argument(
        "--gauss-sigma-1d",
        type=float,
        default=None,
        help="Optional separate 1D gaussian sigma (default: same as --gauss-sigma).",
    )
    p.add_argument("--n-levels", type=int, default=26)
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--grid-source", type=str, default="metad", choices=("metad", "ckpt"))
    p.add_argument("--n-grid", type=int, default=96)
    return p


def main() -> None:
    args = build_parser().parse_args()
    plt.rcParams.update(nature_rcparams())
    xi_lim = _parse_lim(args.xi_lim)

    out_2d = (
        Path(args.output)
        if args.output
        else Path(args.ckpt).resolve().parent / "fes_compare_mle_vs_metad.png"
    )
    out_1d = (
        Path(args.output_1d)
        if args.output_1d
        else out_2d.with_name("fes_compare_1d_d2md1.png")
    )
    out_comb = (
        Path(args.output_combined)
        if args.output_combined
        else out_2d.with_name("fes_compare_combined.png")
    )
    grid_path = (
        Path(args.grid_npz)
        if args.grid_npz
        else out_2d.parent / "fes_grids.npz"
    )

    if args.from_grid is not None:
        data = load_grids(Path(args.from_grid))
        print(f"[load] grids from {args.from_grid}", flush=True)
    else:
        data = compute_grids(args)
        save_grids(grid_path, data)

    if args.mask_both_gt:
        a, b = [float(x) for x in args.mask_both_gt.replace(";", ",").split(",") if x.strip()]
        data = mask_both_gt_and_reproject(
            data, d1_gt=a, d2_gt=b, n_bins_1d=int(args.n_bins_1d)
        )

    if float(args.gauss_sigma) > 0.0:
        data = gaussian_smooth_and_reproject(
            data,
            sigma=float(args.gauss_sigma),
            n_bins_1d=int(args.n_bins_1d),
            sigma_1d=args.gauss_sigma_1d,
        )

    # if loaded from grid, still refresh npz when --grid-npz explicitly set
    if args.from_grid is not None and args.grid_npz is not None:
        save_grids(Path(args.grid_npz), data)

    plot_2d(
        data["d1"],
        data["d2"],
        data["F_mle_kcal"],
        data["F_metad_kcal"],
        vmax=args.vmax_kcal,
        n_levels=int(args.n_levels),
        out=out_2d,
    )
    plot_1d(
        data["xi_d2_minus_d1"],
        data["F1d_mle_kcal"],
        data["F1d_metad_kcal"],
        vmax=args.vmax_kcal,
        out=out_1d,
        xi_lim=xi_lim,
    )
    plot_combined_1x3(
        data["d1"],
        data["d2"],
        data["F_mle_kcal"],
        data["F_metad_kcal"],
        data["xi_d2_minus_d1"],
        data["F1d_mle_kcal"],
        data["F1d_metad_kcal"],
        vmax=args.vmax_kcal,
        n_levels=int(args.n_levels),
        xi_lim=xi_lim,
        out=out_comb,
    )

    print(
        f"[stats] 2D MLE max={float(np.nanmax(data['F_mle_kcal'])):.3f}  "
        f"MetaD max={float(np.nanmax(data['F_metad_kcal'])):.3f}  "
        f"1D MLE max={float(np.nanmax(data['F1d_mle_kcal'])):.3f}  "
        f"MetaD max={float(np.nanmax(data['F1d_metad_kcal'])):.3f} kcal/mol",
        flush=True,
    )


if __name__ == "__main__":
    main()
