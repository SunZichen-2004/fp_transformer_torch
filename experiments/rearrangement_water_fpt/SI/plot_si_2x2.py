#!/usr/bin/env python3
"""SI 2×2 for Claisen rearrangement in water.

Layout matches ``dialanine/SI_dialanine_vacuum/si_vacuum_2x2.png``.
a: ΔG vs GPU-hour cost from live MetaD hills (full x, dive-in y); sFPT wall-clock ×4
b: ΔG vs simulation time
c: 1D FES RMSE vs wall-clock
d: 1D FES RMSE vs simulation time
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "rearrangement_water_fpt"
sys.path.insert(0, str(EXP))

from plot_fes_compare_1x4 import COLOR_META, COLOR_SFPT, nature_rcparams  # noqa: E402

SRC = EXP / "convergence_repeat3_gpu3"
OUT = EXP / "SI"
PANEL_A = SRC / "deltaG_vs_cost_full_metad.npz"
COLOR_US = "#2CA02C"
DOUBLE_COL_IN = 180.0 / 25.4
SFPT_GPUS = 4.0
DG_YLIM = (-15.0, 0.0)
RMSE_YLIM = (0.0, 8.0)


def tag(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.025,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def series(ax, x_m, y_m, e_m, x_s, y_s, e_s) -> None:
    ax.plot(x_m, y_m, color=COLOR_META, lw=1.7, zorder=3)
    ax.fill_between(x_m, y_m - e_m, y_m + e_m, color=COLOR_META, alpha=0.22, lw=0, zorder=1)
    ax.plot(x_s, y_s, color=COLOR_SFPT, lw=2.0, zorder=4)
    ax.fill_between(x_s, y_s - e_s, y_s + e_s, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=2)


def us_ref(ax, dG_ref: float) -> None:
    ax.axhspan(dG_ref - 1.0, dG_ref + 1.0, color=COLOR_US, alpha=0.18, lw=0, zorder=0)
    ax.axhline(dG_ref, color=COLOR_US, ls=":", lw=1.4, zorder=1)


def style_ax(ax) -> None:
    ax.set_frame_on(True)
    ax.tick_params(which="both", top=False, right=False, labelsize=11, direction="out")
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color("0.15")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rmse = np.load(SRC / "fes_rmse_vs_time.npz")
    f30 = np.load(SRC / "fes_rmse_F30.npz")
    pa = np.load(PANEL_A)

    t_m = np.asarray(rmse["t_metad_ps"], dtype=np.float64)
    t_s = np.asarray(rmse["t_kde_ps"], dtype=np.float64)
    dG_m = np.asarray(rmse["dG_metad"], dtype=np.float64)
    dG_ms = np.asarray(rmse["dG_metad_std"], dtype=np.float64)
    dG_s = np.asarray(rmse["dG_kde"], dtype=np.float64)
    dG_ss = np.asarray(rmse["dG_kde_std"], dtype=np.float64)
    dG_us = float(rmse["dG_us_ref"])
    r_m = np.asarray(rmse["rmse_metad"], dtype=np.float64)
    r_ms = np.asarray(rmse["rmse_metad_std"], dtype=np.float64)
    r_s = np.asarray(rmse["rmse_kde"], dtype=np.float64)
    r_ss = np.asarray(rmse["rmse_kde_std"], dtype=np.float64)
    tw30_m = np.asarray(f30["tw_metad_h"], dtype=np.float64)
    tw30_s = np.asarray(f30["tw_kde_h"], dtype=np.float64)
    r30_m = np.asarray(f30["rmse_metad"], dtype=np.float64)
    r30_ms = np.asarray(f30["rmse_metad_std"], dtype=np.float64)
    r30_s = np.asarray(f30["rmse_kde"], dtype=np.float64)
    r30_ss = np.asarray(f30["rmse_kde_std"], dtype=np.float64)
    cost_m = np.asarray(pa["cost_metad_gpuh"], dtype=np.float64)
    cost_s = np.asarray(pa["cost_kde_gpuh"], dtype=np.float64)
    dG_m_a = np.asarray(pa["dG_metad"], dtype=np.float64)
    dG_ms_a = np.asarray(pa["dG_metad_std"], dtype=np.float64)
    dG_s_a = np.asarray(pa["dG_kde"], dtype=np.float64)
    dG_ss_a = np.asarray(pa["dG_kde_std"], dtype=np.float64)
    rmse.close()
    f30.close()
    pa.close()

    plt.rcParams.update(nature_rcparams())
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.05,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
        }
    )

    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.96))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        wspace=0.32,
        hspace=0.38,
        left=0.12,
        right=0.98,
        bottom=0.08,
        top=0.91,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    us_ref(ax_a, dG_us)
    series(ax_a, cost_m, dG_m_a, dG_ms_a, cost_s, dG_s_a, dG_ss_a)
    ax_a.set_xlim(0.0, None)
    ax_a.set_ylim(*DG_YLIM)
    ax_a.set_xlabel(r"cost / GPU$\cdot$h")
    ax_a.set_ylabel(r"$\Delta G$ (kcal/mol)")
    tag(ax_a, "a")
    style_ax(ax_a)

    us_ref(ax_b, dG_us)
    series(ax_b, t_m, dG_m, dG_ms, t_s, dG_s, dG_ss)
    ax_b.set_xlim(0.0, None)
    ax_b.set_ylim(*DG_YLIM)
    ax_b.set_xlabel(r"$t$ / ps")
    ax_b.set_ylabel(r"$\Delta G$ (kcal/mol)")
    tag(ax_b, "b")
    style_ax(ax_b)

    series(ax_c, tw30_m, r30_m, r30_ms, tw30_s, r30_s, r30_ss)
    ax_c.set_xlim(0.0, None)
    ax_c.set_ylim(*RMSE_YLIM)
    ax_c.set_xlabel(r"wall-clock time / h")
    ax_c.set_ylabel(r"1D FES RMSE (kcal/mol)")
    tag(ax_c, "c")
    style_ax(ax_c)

    series(ax_d, t_m, r_m, r_ms, t_s, r_s, r_ss)
    ax_d.set_xlim(0.0, None)
    ax_d.set_ylim(*RMSE_YLIM)
    ax_d.set_xlabel(r"$t$ / ps")
    ax_d.set_ylabel(r"1D FES RMSE (kcal/mol)")
    tag(ax_d, "d")
    style_ax(ax_d)

    handles = [
        Line2D([0], [0], color=COLOR_META, lw=1.8, label="WT-MetaD"),
        Line2D([0], [0], color=COLOR_SFPT, lw=2.0, label="sFPT"),
        Line2D(
            [0],
            [0],
            color=COLOR_US,
            ls=":",
            lw=1.5,
            label=rf"umbrella sampling  $\Delta G$={dG_us:.2f}",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.958),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.canvas.draw()
    for ax in (ax_c, ax_d):
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 + 0.016, p.width, p.height])

    png = OUT / "si_2x2.png"
    pdf = OUT / "si_2x2.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)
    print(
        f"[a] MetaD cost={float(cost_m[-1]):.2f} GPU·h  "
        f"sFPT cost={float(cost_s[-1]):.2f} GPU·h",
        flush=True,
    )


if __name__ == "__main__":
    main()
