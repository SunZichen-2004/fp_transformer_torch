#!/usr/bin/env python3
"""Main-picture 1×2 ΔG convergence: dive-in (a) + t ≤ 1500 ps (b).

Data from ``convergence_analysis/fes_rmse_vs_time.npz``.
Style follows ``rearrangement_water_fpt/SI/si_2x2.png``.
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
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SFP = HERE.parent
sys.path.insert(0, str(HERE))

from plot_main_picture import COLOR_META, COLOR_SFPT, COLOR_US, nature_rcparams  # noqa: E402

NPZ = SFP / "convergence_analysis" / "fes_rmse_vs_time.npz"
DIVEIN_PS = 200.0
TMAX_B_PS = 1500.0
REF_BAND = 0.5
DOUBLE_COL_IN = 180.0 / 25.4


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


def ref_band(ax, dG_ref: float) -> None:
    ax.axhspan(
        dG_ref - REF_BAND, dG_ref + REF_BAND,
        color=COLOR_US, alpha=0.18, lw=0, zorder=0,
    )
    ax.axhline(dG_ref, color=COLOR_US, ls=":", lw=1.4, zorder=1)


def style_ax(ax) -> None:
    ax.set_frame_on(True)
    ax.tick_params(which="both", top=False, right=False, labelsize=11, direction="out")
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color("0.15")


def clip_t(t, y, e, tmax):
    m = np.asarray(t, dtype=np.float64) <= float(tmax) + 1e-12
    return t[m], y[m], e[m]


def main() -> None:
    z = np.load(NPZ)
    t_m = np.asarray(z["t_dg_metad_ps"], dtype=np.float64)
    t_s = np.asarray(z["t_dg_kde_ps"], dtype=np.float64)
    dG_m = np.asarray(z["dG_metad"], dtype=np.float64)
    e_m = np.asarray(z["dG_metad_std"], dtype=np.float64)
    dG_s = np.asarray(z["dG_kde"], dtype=np.float64)
    e_s = np.asarray(z["dG_kde_std"], dtype=np.float64)
    dG_ref = float(z["dG_ref"])
    z.close()

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

    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.50))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.32, left=0.10, right=0.985, bottom=0.18, top=0.84,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    ta_m, ya_m, ea_m = clip_t(t_m, dG_m, e_m, DIVEIN_PS)
    ta_s, ya_s, ea_s = clip_t(t_s, dG_s, e_s, DIVEIN_PS)
    tb_m, yb_m, eb_m = clip_t(t_m, dG_m, e_m, TMAX_B_PS)
    tb_s, yb_s, eb_s = clip_t(t_s, dG_s, e_s, TMAX_B_PS)

    ylab = r"$\Delta G$ / kcal mol$^{-1}$"
    tmax_b_ns = TMAX_B_PS / 1000.0

    ref_band(ax_a, dG_ref)
    series(ax_a, ta_m, ya_m, ea_m, ta_s, ya_s, ea_s)
    ax_a.set_xlim(0.0, DIVEIN_PS)
    ax_a.set_ylim(dG_ref - 3.0, dG_ref + 3.0)
    ax_a.set_xlabel(r"$t$ / ps")
    ax_a.set_ylabel(ylab)
    tag(ax_a, "a")
    style_ax(ax_a)

    ref_band(ax_b, dG_ref)
    series(ax_b, tb_m / 1000.0, yb_m, eb_m, tb_s / 1000.0, yb_s, eb_s)
    ax_b.set_xlim(0.0, tmax_b_ns)
    ax_b.set_ylim(-1.5, 5.0)
    ax_b.set_xlabel(r"$t$ / ns")
    ax_b.set_ylabel(ylab)
    tag(ax_b, "b")
    style_ax(ax_b)

    handles = [
        Line2D([0], [0], color=COLOR_META, lw=1.8, label="WT-MetaD"),
        Line2D([0], [0], color=COLOR_SFPT, lw=2.0, label="sFPT"),
        Line2D(
            [0], [0], color=COLOR_US, ls=":", lw=1.5,
            label=rf"MetaD  $\Delta G$={dG_ref:.2f}",
        ),
        Patch(
            facecolor=COLOR_US, edgecolor="none", alpha=0.18,
            label=rf"final $\Delta G\pm {REF_BAND:g}$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.90),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    png = HERE / "convergence.png"
    pdf = HERE / "convergence.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)
    print(f"[ok] {pdf}", flush=True)
    print(f"[dG_ref]={dG_ref:.4f}  ±{REF_BAND:g}  a≤{DIVEIN_PS:g} ps  b≤{TMAX_B_PS:g} ps", flush=True)


if __name__ == "__main__":
    main()
