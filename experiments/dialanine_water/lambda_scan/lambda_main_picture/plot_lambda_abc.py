#!/usr/bin/env python3
"""1×3 λ-scan collage: ΔG(t), 2D FES RMSE(λ), ΔG MAE(λ)."""

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
EXP = ROOT / "experiments" / "dialanine_water"
OUT = EXP / "lambda_scan" / "lambda_main_picture"
DG_NPZ = EXP / "lambda_scan" / "convergence_analysis" / "deltaG_vs_time_by_lambda.npz"
RMSE_NPZ = EXP / "lambda_scan" / "convergence_analysis" / "fes_last50" / "fes_rmse_by_lambda.npz"
MAE_NPZ = OUT / "dG_mae_vs_lambda.npz"

sys.path.insert(0, str(EXP))
from plot_lambda_scan_fes_rmse import LAM_COLORS  # noqa: E402

COLOR_US = "#2CA02C"


def jctc_rcparams(*, scale: float = 1.0) -> dict[str, object]:
    s = float(scale)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Noto Sans CJK SC",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "font.size": 16 * s,
        "axes.labelsize": 16 * s,
        "axes.titlesize": 16 * s,
        "xtick.labelsize": 14 * s,
        "ytick.labelsize": 14 * s,
        "legend.fontsize": 12.5 * s,
        "axes.linewidth": 1.15,
        "xtick.major.width": 1.15,
        "ytick.major.width": 1.15,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }


def lam_tag(lam: float) -> str:
    return f"{float(lam):g}".replace(".", "p")


def lam_color(lam: float) -> str:
    return LAM_COLORS.get(float(lam), LAM_COLORS.get(round(float(lam), 2), "C0"))


def panel_letter(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.03,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def style_ax(ax) -> None:
    ax.tick_params(which="both", top=False, right=False)
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
        ax.spines[side].set_color("0.15")


def plot_errorbar_lambda(ax, lams, y, e) -> None:
    ax.errorbar(
        lams,
        y,
        yerr=e,
        fmt="o-",
        color="0.25",
        ecolor="0.45",
        elinewidth=1.1,
        capsize=3.2,
        lw=1.6,
        ms=5.8,
        zorder=2,
    )
    for x, yy in zip(lams, y):
        ax.plot(x, yy, "o", color=lam_color(float(x)), ms=7.2, zorder=3)
    ax.set_xlim(left=0.0)
    ax.set_xlabel(r"$\lambda$")


def plot_a(ax, z) -> list[Line2D]:
    dG_us = float(z["dG_us_ref"])
    lams = np.asarray(z["lambdas"], dtype=np.float64)
    ax.axhline(dG_us, color=COLOR_US, ls=":", lw=1.5, zorder=1)
    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_US,
            ls=":",
            lw=1.8,
            label=rf"US $\Delta G$={dG_us:.2f}",
        )
    ]
    for lam in lams:
        tag = lam_tag(float(lam))
        t = np.asarray(z[f"t_lam{tag}"], dtype=np.float64)
        dG = np.asarray(z[f"dG_lam{tag}"], dtype=np.float64)
        std = np.asarray(z[f"dG_std_lam{tag}"], dtype=np.float64)
        c = lam_color(float(lam))
        ax.plot(t, dG, color=c, lw=1.7, zorder=3)
        if np.any(std > 0):
            ax.fill_between(
                t, dG - std, dG + std, color=c, alpha=0.18, lw=0, zorder=2
            )
        handles.append(
            Line2D([0], [0], color=c, lw=1.8, label=rf"$\lambda$={float(lam):g}")
        )
    ax.set_xlim(0.0, None)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    return handles


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(jctc_rcparams(scale=1.0))

    fig = plt.figure(figsize=(14.0, 4.65))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        wspace=0.38,
        left=0.055,
        right=0.995,
        bottom=0.16,
        top=0.82,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    zd = np.load(DG_NPZ)
    handles = plot_a(ax_a, zd)
    zd.close()
    style_ax(ax_a)
    panel_letter(ax_a, "a")

    zr = np.load(RMSE_NPZ)
    lams_r = np.asarray(zr["lambdas"], dtype=np.float64)
    plot_errorbar_lambda(
        ax_b,
        lams_r,
        np.asarray(zr["rmse_2d"], dtype=np.float64),
        np.asarray(zr["rmse_2d_std"], dtype=np.float64),
    )
    zr.close()
    ax_b.set_ylim(0.3, 0.7)
    ax_b.set_title(r"2D $F(\phi,\psi)$", pad=6)
    ax_b.set_ylabel(r"RMSE vs US / kcal mol$^{-1}$")
    style_ax(ax_b)
    panel_letter(ax_b, "b")

    zm = np.load(MAE_NPZ)
    lams_m = np.asarray(zm["lambdas"], dtype=np.float64)
    plot_errorbar_lambda(
        ax_c,
        lams_m,
        np.asarray(zm["mae_mean"], dtype=np.float64),
        np.asarray(zm["mae_std"], dtype=np.float64),
    )
    zm.close()
    ax_c.set_ylim(bottom=0.0)
    ax_c.set_ylabel(r"$\Delta G$ MAE vs US / kcal mol$^{-1}$")
    style_ax(ax_c)
    panel_letter(ax_c, "c")

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=13,
        handlelength=1.8,
        columnspacing=1.35,
        borderaxespad=0.0,
    )

    png = OUT / "lambda_abc.png"
    pdf = OUT / "lambda_abc.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)


if __name__ == "__main__":
    main()
