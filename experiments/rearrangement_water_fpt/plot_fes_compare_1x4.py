#!/usr/bin/env python3
"""JCTC-style 1×4 figure: sFPT / WT-MetaD 2D FES, 1D FES, RMSE vs wall-clock.

sFPT 2D KDE uses the last 100 stages of 87173. Canvas is 14 in wide so
published at ACS double column (7 in) the type is ~8 pt (ACS min 4.5 pt).
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "rearrangement_water_fpt"
OUT = EXP / "convergence_analysis"
GRIDS = OUT / "fes_compare_last100_grids.npz"
RMSE = OUT / "fes_rmse_F30.npz"
FPT_DIR = EXP / "results" / "claisen_water_fpt_mh-0_g8_87173"
META_HILLS = [
    ROOT / "experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_87059/live/hills.npz",
    ROOT / "experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_87060/live/hills.npz",
]

FE_LABEL = r"Free energy (kcal/mol)"
COLOR_META = "#2166AC"
COLOR_SFPT = "#E67E22"
COLOR_OK = "#2CA02C"
FES_CMAP = "RdBu_r"
VMAX_2D = 55.0
LIM_2D = (1.0, 7.0)
XI_LIM = (-5.0, 5.0)
YLIM_RMSE = 12.0
OK_KCAL = 1.0
SMOOTH_1D_SIGMA = 1.25  # bins
N_FINE_1D = 500
N_BINS_1D = 140
GRID_N = 140
LAST_N = 100
P_FLOOR = 1e-6
LAM = 0.025
T_K = 300.0
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
SIGMA_METAD = np.array([0.2, 0.2], dtype=np.float64)
GAMMA_METAD = 30.0


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


def relative_fes(f: np.ndarray) -> np.ndarray:
    out = np.asarray(f, dtype=np.float64)
    return out - np.nanmin(out[np.isfinite(out)])


def fes_cmap():
    cmap = plt.get_cmap(FES_CMAP).copy()
    cmap.set_over("white")
    cmap.set_bad("white")
    return cmap


def metad_fes_kcal_fft(hills_path: Path, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    z = np.load(hills_path)
    n = int(z["n_hills"])
    centers = np.asarray(z["centers"][:n], dtype=np.float64)
    heights = np.asarray(z["heights"][:n], dtype=np.float64).reshape(-1)
    dx = 0.025
    pad = 1.2
    x0 = float(min(d1.min(), centers[:, 0].min()) - pad)
    x1 = float(max(d1.max(), centers[:, 0].max()) + pad)
    y0 = float(min(d2.min(), centers[:, 1].min()) - pad)
    y1 = float(max(d2.max(), centers[:, 1].max()) + pad)
    nx = int(np.round((x1 - x0) / dx)) + 1
    ny = int(np.round((y1 - y0) / dx)) + 1
    xe = x0 + dx * np.arange(nx + 1)
    ye = y0 + dx * np.arange(ny + 1)
    H, _, _ = np.histogram2d(
        centers[:, 0], centers[:, 1], bins=[xe, ye], weights=heights
    )
    krad = int(np.ceil(8.0 * float(np.max(SIGMA_METAD)) / dx))
    kx = np.arange(-krad, krad + 1, dtype=np.float64) * dx
    ky = np.arange(-krad, krad + 1, dtype=np.float64) * dx
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    ker = np.exp(
        -0.5 * ((KX / SIGMA_METAD[0]) ** 2 + (KY / SIGMA_METAD[1]) ** 2)
    )
    V = fftconvolve(H, ker, mode="same")
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    interp = RegularGridInterpolator((xc, yc), V, bounds_error=False, fill_value=0.0)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    Vq = interp(np.stack([g1.ravel(), g2.ravel()], axis=-1)).reshape(g1.shape)
    scale = GAMMA_METAD / (GAMMA_METAD - 1.0)
    return relative_fes((-scale * Vq) * EV_TO_KCAL)


def compute_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sys.path.insert(0, str(EXP))
    from plot_kde_fes import density_on_grid, discover_stages

    d1 = np.linspace(LIM_2D[0], LIM_2D[1], GRID_N)
    d2 = np.linspace(LIM_2D[0], LIM_2D[1], GRID_N)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)

    if GRIDS.is_file():
        z = np.load(GRIDS)
        if int(z["n_stages"]) == LAST_N and z["d1"].size == GRID_N:
            print(f"[reuse] {GRIDS}", flush=True)
            return d1, d2, np.asarray(z["F_sfpt"]), np.asarray(z["F_metad"])

    pairs = discover_stages(FPT_DIR, last_n=LAST_N, stage_list=None)
    print(
        f"[sFPT] last {len(pairs)} stages  {pairs[0][0].name} … {pairs[-1][0].name}",
        flush=True,
    )
    device = torch.device("cpu")
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    for i, (cv_p, bw_p, _lam) in enumerate(pairs):
        p_acc += density_on_grid(
            grid_xy, np.load(cv_p), np.load(bw_p), chunk=4096, device=device
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  kde [{i+1}/{len(pairs)}]", flush=True)
    p = np.clip(p_acc / float(len(pairs)), 1e-300, None)
    beta0 = 1.0 / (KB_EV_PER_K * T_K)
    f_sfpt = relative_fes((-np.log(p) / (beta0 * LAM) * EV_TO_KCAL).reshape(g1.shape))
    p2 = p.reshape(g1.shape)
    f_sfpt = f_sfpt.copy()
    f_sfpt[p2 < P_FLOOR * float(np.max(p2))] = np.nan

    f_meta_list = [metad_fes_kcal_fft(p, d1, d2) for p in META_HILLS]
    f_meta = relative_fes(np.nanmean(np.stack(f_meta_list, axis=0), axis=0))
    np.savez_compressed(
        GRIDS,
        d1=d1,
        d2=d2,
        F_sfpt=f_sfpt,
        F_metad=f_meta,
        n_stages=len(pairs),
        first_stage=pairs[0][0].name,
        last_stage=pairs[-1][0].name,
        lambda_=LAM,
    )
    print(f"[npz] {GRIDS}", flush=True)
    return d1, d2, f_sfpt, f_meta


def tag_title(ax, letter: str, title: str, *, letter_x: float = -0.08) -> None:
    ax.text(
        letter_x,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_title(
        title if title else " ",
        loc="center",
        pad=8.0,
        fontsize=17,
        fontweight="normal",
    )


def project_1d_from_2d(
    d1: np.ndarray,
    d2: np.ndarray,
    F: np.ndarray,
    *,
    d1_lim: tuple[float, float],
    d2_lim: tuple[float, float],
    n_bins: int,
    temperature: float = T_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann projection ξ=d2−d1 from the same 2D FES shown in panels a/b."""
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    f = np.asarray(F, dtype=np.float64)
    sel = (
        np.isfinite(f)
        & (g1 >= d1_lim[0])
        & (g1 <= d1_lim[1])
        & (g2 >= d2_lim[0])
        & (g2 <= d2_lim[1])
    )
    xi = (g2 - g1)[sel]
    fv = f[sel]
    edges = np.linspace(float(xi.min()), float(xi.max()), int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kt = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kt
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            m = (xi >= edges[i]) & (xi < edges[i + 1])
        else:
            m = (xi >= edges[i]) & (xi <= edges[i + 1])
        if not np.any(m):
            continue
        vals = -beta * fv[m]
        mx = float(np.max(vals))
        out[i] = -kt * (mx + np.log(np.sum(np.exp(vals - mx))))
    return centers, relative_fes(out)


def smooth_1d(xi: np.ndarray, f: np.ndarray, *, sigma: float, n_fine: int):
    xi = np.asarray(xi, dtype=np.float64)
    f = relative_fes(f)
    m = np.isfinite(f)
    fill = float(np.nanmean(f[m]))
    sm = gaussian_filter1d(np.where(m, f, fill), sigma=float(sigma), mode="nearest")
    sm = np.where(m, sm, np.nan)
    sm = relative_fes(sm)
    xi_f = np.linspace(float(xi[m].min()), float(xi[m].max()), int(n_fine))
    fi = interp1d(xi[m], sm[m], kind="cubic", bounds_error=False)
    y = np.asarray(fi(xi_f), dtype=np.float64)
    return xi_f, relative_fes(y)


def draw_fes2d(
    ax,
    d1,
    d2,
    F,
    *,
    vmax: float,
    title: str,
    letter: str,
    show_ylabel: bool = True,
):
    Z = np.ma.masked_invalid(relative_fes(F))
    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    im = ax.contourf(d1, d2, Z, levels=levels, cmap=cmap, norm=norm, extend="max")
    ax.contour(
        d1,
        d2,
        np.clip(Z.filled(float(vmax)), 0.0, float(vmax)),
        levels=levels[::3],
        colors="0.45",
        linewidths=0.35,
        alpha=0.45,
    )
    ax.set_xlim(*LIM_2D)
    ax.set_ylim(*LIM_2D)
    ax.set_xlabel(r"$d_1$ (Å)")
    ax.set_xticks([1, 3, 5, 7])
    ax.set_yticks([1, 3, 5, 7])
    if show_ylabel:
        ax.set_ylabel(r"$d_2$ (Å)")
        tag_title(ax, letter, title)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        tag_title(ax, letter, title, letter_x=0.02)
    ax.tick_params(which="both", top=False, right=False)
    ax.set_aspect("equal", adjustable="box")
    return im


def draw_fes1d(ax, xi_m, y_m, xi_s, y_s, *, letter: str) -> None:
    ax.plot(xi_m, y_m, color=COLOR_META, lw=1.8, label="WT-MetaD", zorder=2)
    ax.plot(xi_s, y_s, color=COLOR_SFPT, lw=2.1, label="sFPT", zorder=4)
    ax.set_xlim(*XI_LIM)
    ax.set_ylim(0.0, VMAX_2D)
    ax.set_xlabel(r"$d_2 - d_1$ (Å)")
    ax.set_ylabel(FE_LABEL)
    tag_title(ax, letter, r"$F(d_2-d_1)$")
    ax.tick_params(which="both", top=False, right=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend(
        loc="upper right",
        fontsize=12,
        handlelength=1.5,
        frameon=False,
        borderaxespad=0.15,
    )


def draw_rmse(ax, *, tw_m, y_m, e_m, tw_s, y_s, e_s, letter: str) -> None:
    ax.axhspan(
        0.0,
        OK_KCAL,
        color=COLOR_OK,
        alpha=0.18,
        lw=0,
        zorder=0,
        label=rf"{OK_KCAL:g} kcal/mol",
    )
    ax.axhline(0.0, color="0.45", ls="--", lw=0.9, zorder=1)
    ax.plot(tw_m, y_m, color=COLOR_META, lw=1.8, label="WT-MetaD", zorder=3)
    if e_m is not None:
        ax.fill_between(
            tw_m, y_m - e_m, y_m + e_m, color=COLOR_META, alpha=0.22, lw=0, zorder=2
        )
    ax.plot(tw_s, y_s, color=COLOR_SFPT, lw=2.1, label="sFPT", zorder=5)
    if e_s is not None:
        ax.fill_between(
            tw_s, y_s - e_s, y_s + e_s, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=4
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, YLIM_RMSE)
    ax.set_xlabel(r"wall-clock time (h)")
    ax.set_ylabel(r"1D FES RMSE (kcal/mol)")
    tag_title(ax, letter, "FES RMSE")
    ax.tick_params(which="both", top=False, right=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend(
        loc="upper right",
        fontsize=12,
        handlelength=1.5,
        frameon=False,
        borderaxespad=0.15,
    )


def main() -> None:
    plt.rcParams.update(nature_rcparams())
    d1, d2, F_sfpt, F_meta = compute_grids()
    xi_s0, f1d_s = project_1d_from_2d(
        d1, d2, F_sfpt, d1_lim=LIM_2D, d2_lim=LIM_2D, n_bins=N_BINS_1D
    )
    xi_m0, f1d_m = project_1d_from_2d(
        d1, d2, F_meta, d1_lim=LIM_2D, d2_lim=LIM_2D, n_bins=N_BINS_1D
    )
    xi_s, y_s = smooth_1d(xi_s0, f1d_s, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D)
    xi_m, y_m = smooth_1d(xi_m0, f1d_m, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D)
    print(
        f"[1d] ξ sFPT [{xi_s.min():.2f},{xi_s.max():.2f}]  "
        f"MetaD [{xi_m.min():.2f},{xi_m.max():.2f}]",
        flush=True,
    )

    r = np.load(RMSE)
    tw_m = np.asarray(r["tw_metad_h"], dtype=np.float64)
    tw_s = np.asarray(r["tw_kde_h"], dtype=np.float64)
    rmse_m = np.asarray(r["rmse_metad"], dtype=np.float64)
    rmse_s = np.asarray(r["rmse_kde"], dtype=np.float64)
    e_m = np.asarray(r["rmse_metad_std"], dtype=np.float64)
    e_s = np.asarray(r["rmse_kde_std"], dtype=np.float64)

    fig_w, fig_h = 14.0, 3.95
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = fig.add_axes([0, 0, 1, 1])
    ax_b = fig.add_axes([0, 0, 1, 1])
    ax_c = fig.add_axes([0, 0, 1, 1])
    ax_d = fig.add_axes([0, 0, 1, 1])
    cax = fig.add_axes([0, 0, 1, 1])

    left, right = 0.048, 0.992
    bottom, top = 0.175, 0.84
    gap_ab = 0.006
    cbar_pad = 0.006
    cbar_w = 0.010
    cbar_lab = 0.125
    gap_cd = 0.072
    reserved = gap_ab + cbar_pad + cbar_w + cbar_lab + gap_cd
    w_p = (right - left - reserved) / 4.0
    h_p = w_p * fig_w / fig_h
    h_avail = top - bottom
    if h_p > h_avail:
        scale = h_avail / h_p
        h_p *= scale
        w_p *= scale
    y0 = bottom + 0.5 * (h_avail - h_p)

    x_a = left
    x_b = x_a + w_p + gap_ab
    x_cb = x_b + w_p + cbar_pad
    x_c = x_cb + cbar_w + cbar_lab
    x_d = x_c + w_p + gap_cd
    ax_a.set_position([x_a, y0, w_p, h_p])
    ax_b.set_position([x_b, y0, w_p, h_p])
    cax.set_position([x_cb, y0, cbar_w, h_p])
    ax_c.set_position([x_c, y0, w_p, h_p])
    ax_d.set_position([x_d, y0, w_p, h_p])

    im = draw_fes2d(
        ax_a, d1, d2, F_sfpt, vmax=VMAX_2D, title="sFPT", letter="a"
    )
    draw_fes2d(
        ax_b,
        d1,
        d2,
        F_meta,
        vmax=VMAX_2D,
        title="WT-MetaD",
        letter="b",
        show_ylabel=False,
    )
    draw_fes1d(ax_c, xi_m, y_m, xi_s, y_s, letter="c")
    draw_rmse(
        ax_d,
        tw_m=tw_m,
        y_m=rmse_m,
        e_m=e_m,
        tw_s=tw_s,
        y_s=rmse_s,
        e_s=e_s,
        letter="d",
    )

    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(r"(kcal/mol)", fontsize=14, labelpad=4)
    cb.ax.tick_params(labelsize=13, length=3.2, width=1.0)
    cb.set_ticks([0, 15, 30, 45, 55])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    cb.outline.set_linewidth(1.1)

    png = OUT / "fes_compare_1x4.png"
    pdf = OUT / "fes_compare_1x4.pdf"
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0.06)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0.06)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)


if __name__ == "__main__":
    main()
