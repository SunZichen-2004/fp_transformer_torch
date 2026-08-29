#!/usr/bin/env python3
"""JCTC-style 1×4 main figure for Claisen rearrangement in water.

Layout matches ``convergence_analysis/fes_compare_1x4.png``.
Panels a–c use ``convergence_repeat3_gpu3`` last-100 / seed-averaged data
(same as ``fes_compare_abc.png``, with 1D mean±std across seeds).
Panel d is ΔG vs wall-clock (dive-in, same as ``deltaG_vs_wallclock_divein.png``).
A single outside legend covers c and d.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "rearrangement_water_fpt"
sys.path.insert(0, str(EXP))

from plot_fes_compare_1x4 import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    EV_TO_KCAL,
    FE_LABEL,
    KB_EV_PER_K,
    LAST_N,
    LIM_2D,
    N_BINS_1D,
    N_FINE_1D,
    SMOOTH_1D_SIGMA,
    T_K,
    VMAX_2D,
    XI_LIM,
    draw_fes2d,
    nature_rcparams,
    project_1d_from_2d,
    relative_fes,
    smooth_1d,
    tag_title,
)

SRC = EXP / "convergence_repeat3_gpu3"
OUT = EXP / "rearrangement_main_picture"
COLOR_US = "#2CA02C"
GROUP = 10
N_LAST_GROUPS = LAST_N // GROUP
DG_YLIM = (-20.0, 0.0)
KT_KCAL = KB_EV_PER_K * T_K * EV_TO_KCAL


def boltzmann_combine(f_rows: np.ndarray, kT: float = KT_KCAL) -> np.ndarray:
    f_rows = np.asarray(f_rows, dtype=np.float64)
    p = np.nanmean(np.exp(-np.clip(f_rows, None, 80.0) / kT), axis=0)
    out = -kT * np.log(np.clip(p, 1e-300, None))
    out[~np.isfinite(p)] = np.nan
    return relative_fes(out)


def mean_std_seeds(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=0)
    n = int(stack.shape[0])
    if n <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=0, ddof=1)


def smooth_mean_std(xi, y, e, *, sigma: float, n_fine: int):
    """Smooth mean and ±std without independently re-zeroing the error bounds."""
    xi = np.asarray(xi, dtype=np.float64)
    y = relative_fes(y)
    e = np.asarray(e, dtype=np.float64)
    m = np.isfinite(y)
    fill = float(np.nanmean(y[m]))
    sm = gaussian_filter1d(np.where(m, y, fill), sigma=float(sigma), mode="nearest")
    sm = relative_fes(np.where(m, sm, np.nan))
    e_ok = np.isfinite(e)
    e_fill = float(np.nanmean(e[e_ok])) if np.any(e_ok) else 0.0
    se = gaussian_filter1d(np.where(e_ok, e, e_fill), sigma=float(sigma), mode="nearest")
    se = np.where(m & e_ok, se, np.nan)
    xi_f = np.linspace(float(xi[m].min()), float(xi[m].max()), int(n_fine))
    yi = interp1d(xi[m], sm[m], kind="cubic", bounds_error=False)
    ei = interp1d(
        xi[m],
        np.where(np.isfinite(se[m]), se[m], 0.0),
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )
    yf = relative_fes(np.asarray(yi(xi_f), dtype=np.float64))
    ef = np.clip(np.asarray(ei(xi_f), dtype=np.float64), 0.0, None)
    return xi_f, yf, np.clip(yf - ef, 0.0, VMAX_2D), np.clip(yf + ef, 0.0, VMAX_2D)


def draw_fes1d(ax, *, xi_u, y_u, xi_m, y_m, lo_m, hi_m, xi_s, y_s, lo_s, hi_s) -> None:
    ax.fill_between(xi_m, lo_m, hi_m, color=COLOR_META, alpha=0.22, lw=0, zorder=1)
    ax.plot(xi_m, y_m, color=COLOR_META, lw=1.8, zorder=2)
    ax.fill_between(xi_s, lo_s, hi_s, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=3)
    ax.plot(xi_s, y_s, color=COLOR_SFPT, lw=2.1, zorder=4)
    ax.plot(xi_u, y_u, color=COLOR_US, lw=1.6, ls="--", zorder=5)
    ax.set_xlim(*XI_LIM)
    ax.set_ylim(0.0, VMAX_2D)
    ax.set_xlabel(r"$d_2 - d_1$ (Å)")
    ax.set_ylabel(FE_LABEL)
    tag_title(ax, "c", r"$F(d_2-d_1)$")
    ax.tick_params(which="both", top=False, right=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


def draw_dg(ax, *, tw_m, y_m, e_m, tw_s, y_s, e_s, dG_ref: float) -> None:
    ax.axhspan(
        dG_ref - 1.0,
        dG_ref + 1.0,
        color=COLOR_US,
        alpha=0.18,
        lw=0,
        zorder=0,
    )
    ax.axhline(dG_ref, color=COLOR_US, ls="--", lw=1.6, zorder=1)
    ax.fill_between(
        tw_m, y_m - e_m, y_m + e_m, color=COLOR_META, alpha=0.22, lw=0, zorder=2
    )
    ax.plot(tw_m, y_m, color=COLOR_META, lw=1.8, zorder=3)
    ax.fill_between(
        tw_s, y_s - e_s, y_s + e_s, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=4
    )
    ax.plot(tw_s, y_s, color=COLOR_SFPT, lw=2.1, zorder=5)
    ax.set_xlim(left=0.0)
    ax.set_ylim(*DG_YLIM)
    ax.set_xlabel(r"wall-clock time (h)")
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    tag_title(ax, "d", r"$\Delta G=k_BT\ln(P_1/P_2)$")
    ax.set_title(
        r"$\Delta G=k_BT\ln(P_1/P_2)$",
        loc="center",
        pad=8.0,
        fontsize=14,
        fontweight="normal",
    )
    ax.tick_params(which="both", top=False, right=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(nature_rcparams())

    grids = np.load(SRC / "fes_compare_last100_grids.npz")
    g1d = np.load(SRC / "fes_1d_all_groups.npz", allow_pickle=True)
    rmse = np.load(SRC / "fes_rmse_vs_time.npz")
    d1 = np.asarray(grids["d1"])
    d2 = np.asarray(grids["d2"])
    F_sfpt = np.asarray(grids["F_sfpt"])
    F_meta = np.asarray(grids["F_metad"])
    xi = np.asarray(g1d["xi"], dtype=np.float64)
    f_us = np.asarray(g1d["F1d_us"], dtype=np.float64)
    f_meta = np.asarray(g1d["F1d_metad"], dtype=np.float64)
    e_meta = np.asarray(g1d["F1d_metad_std"], dtype=np.float64)
    runs = np.asarray(g1d["F1d_kde_runs"], dtype=np.float64)
    tw_m = np.asarray(rmse["tw_metad_h"], dtype=np.float64)
    tw_s = np.asarray(rmse["tw_kde_h"], dtype=np.float64)
    dG_m = np.asarray(rmse["dG_metad"], dtype=np.float64)
    dG_m_e = np.asarray(rmse["dG_metad_std"], dtype=np.float64)
    dG_s = np.asarray(rmse["dG_kde"], dtype=np.float64)
    dG_s_e = np.asarray(rmse["dG_kde_std"], dtype=np.float64)
    dG_us = float(rmse["dG_us_ref"])
    grids.close()
    g1d.close()
    rmse.close()

    last = np.stack(
        [boltzmann_combine(runs[i, -N_LAST_GROUPS:]) for i in range(runs.shape[0])],
        axis=0,
    )
    f_sfpt_1d, e_sfpt = mean_std_seeds(last)
    # Mean curves match fes_compare_abc: Boltzmann projection of the a/b 2D FES.
    # Shaded band is seed std (MetaD: 2 hills; sFPT: last 100 stages / 10 groups).
    xi_s0, f1d_s = project_1d_from_2d(
        d1, d2, F_sfpt, d1_lim=LIM_2D, d2_lim=LIM_2D, n_bins=N_BINS_1D
    )
    xi_m0, f1d_m = project_1d_from_2d(
        d1, d2, F_meta, d1_lim=LIM_2D, d2_lim=LIM_2D, n_bins=N_BINS_1D
    )
    print(
        f"[1d] sFPT last {N_LAST_GROUPS} groups × {runs.shape[0]} seeds  "
        f"std max={float(np.nanmax(e_sfpt)):.3f}  "
        f"MetaD std max={float(np.nanmax(e_meta)):.3f}  US ΔG={dG_us:.3f}",
        flush=True,
    )

    m_us = np.isfinite(f_us)
    xi_u, y_u = smooth_1d(
        xi[m_us], f_us[m_us], sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D
    )
    y_u = np.minimum(y_u, VMAX_2D)
    xi_m, y_m = smooth_1d(xi_m0, f1d_m, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D)
    xi_s, y_s = smooth_1d(xi_s0, f1d_s, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D)
    xi_me, _, lo_m0, hi_m0 = smooth_mean_std(
        xi, f_meta, e_meta, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D
    )
    xi_se, _, lo_s0, hi_s0 = smooth_mean_std(
        xi, f_sfpt_1d, e_sfpt, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D
    )
    e_m_f = np.interp(xi_m, xi_me, 0.5 * (hi_m0 - lo_m0), left=0.0, right=0.0)
    e_s_f = np.interp(xi_s, xi_se, 0.5 * (hi_s0 - lo_s0), left=0.0, right=0.0)
    lo_m = np.clip(y_m - e_m_f, 0.0, VMAX_2D)
    hi_m = np.clip(y_m + e_m_f, 0.0, VMAX_2D)
    lo_s = np.clip(y_s - e_s_f, 0.0, VMAX_2D)
    hi_s = np.clip(y_s + e_s_f, 0.0, VMAX_2D)

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

    im = draw_fes2d(ax_a, d1, d2, F_sfpt, vmax=VMAX_2D, title="sFPT", letter="a")
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
    draw_fes1d(
        ax_c,
        xi_u=xi_u,
        y_u=y_u,
        xi_m=xi_m,
        y_m=y_m,
        lo_m=lo_m,
        hi_m=hi_m,
        xi_s=xi_s,
        y_s=y_s,
        lo_s=lo_s,
        hi_s=hi_s,
    )
    draw_dg(
        ax_d,
        tw_m=tw_m,
        y_m=dG_m,
        e_m=dG_m_e,
        tw_s=tw_s,
        y_s=dG_s,
        e_s=dG_s_e,
        dG_ref=dG_us,
    )

    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(r"(kcal/mol)", fontsize=14, labelpad=4)
    cb.ax.tick_params(labelsize=13, length=3.2, width=1.0)
    cb.set_ticks([0, 15, 30, 45, 55])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    cb.outline.set_linewidth(1.1)

    handles = [
        Line2D([0], [0], color=COLOR_US, lw=1.8, ls="--", label="umbrella sampling"),
        Line2D([0], [0], color=COLOR_META, lw=1.8, label="WT-MetaD"),
        Line2D([0], [0], color=COLOR_SFPT, lw=2.1, label="sFPT"),
    ]
    x_mid = 0.5 * (x_c + x_d + w_p)
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(x_mid, 0.975),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=13,
        handlelength=1.8,
        columnspacing=1.35,
        borderaxespad=0.0,
    )

    png = OUT / "fes_compare_1x4.png"
    pdf = OUT / "fes_compare_1x4.pdf"
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0.06)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0.06)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)


if __name__ == "__main__":
    main()
