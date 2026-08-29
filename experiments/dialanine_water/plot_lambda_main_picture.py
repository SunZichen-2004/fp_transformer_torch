#!/usr/bin/env python3
"""JCTC double-column λ-scan main figures.

1. 5×2 last-50% 1D F(φ), F(ψ) vs umbrella sampling, seed std as shade.
2. Flattening-law two-panel: overlay F_eff vs F_US, and slope vs λ/(1+λ).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(EXP))

from plot_fes_compare import relative_fes  # noqa: E402
from plot_lambda_scan_fes_rmse import LAM_COLORS  # noqa: E402

OUT = EXP / "lambda_scan" / "lambda_main_picture"
FES_NPZ = EXP / "lambda_scan" / "convergence_analysis" / "fes_last50" / "fes_rmse_by_lambda.npz"
FLAT_NPZ = EXP / "lambda_scan" / "flatten_law_validation" / "flatten_law_fit.npz"

COLOR_US = "#2CA02C"
COLOR_SFPT = "#E67E22"
LAMS = (0.02, 0.04, 0.06, 0.08, 0.10)
PI_TICKS = (-np.pi, 0.0, np.pi)
PI_LABELS = (r"$-\pi$", r"$0$", r"$\pi$")
FE_LABEL = r"Free energy (kcal/mol)"
XLABEL_US = r"$F_{\mathrm{US}}(x)$ / kcal/mol"
YLABEL_EFF = r"$F_{\mathrm{eff},\lambda}=-k_BT\ln P$ / kcal/mol"
VMAX = 16.0


def jctc_rcparams(*, scale: float = 1.0) -> dict[str, object]:
    # ACS/JCTC double-column = 7.0 in. Canvas is 14 in wide → ~2× type.
    s = float(scale)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "font.size": 16 * s,
        "axes.labelsize": 16 * s,
        "axes.titlesize": 17 * s,
        "xtick.labelsize": 14 * s,
        "ytick.labelsize": 14 * s,
        "legend.fontsize": 14 * s,
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


def panel_letter(ax, letter: str, *, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        clip_on=False,
    )


def plot_1d_curve(ax, xi, y, ystd, y_us) -> None:
    y_us = np.minimum(relative_fes(y_us, None), VMAX)
    y = np.minimum(relative_fes(y, None), VMAX)
    ax.plot(xi, y_us, color=COLOR_US, lw=3.2, ls="--", zorder=2)
    if ystd is not None and np.any(np.asarray(ystd) > 0):
        lo = np.clip(y - ystd, 0.0, VMAX)
        hi = np.clip(y + ystd, 0.0, VMAX)
        ax.fill_between(xi, lo, hi, color=COLOR_SFPT, alpha=0.32, lw=0, zorder=1)
    ax.plot(xi, y, color=COLOR_SFPT, lw=3.2, zorder=3)


def plot_fes_1d_grid(out: Path, z) -> None:
    plt.rcParams.update(jctc_rcparams(scale=1.35))
    fig = plt.figure(figsize=(14.0, 17.6))
    gs = gridspec.GridSpec(
        5,
        2,
        figure=fig,
        wspace=0.12,
        hspace=0.18,
        left=0.15,
        right=0.985,
        top=0.925,
        bottom=0.065,
    )
    xi_phi = np.asarray(z["xi_phi"], dtype=np.float64)
    xi_psi = np.asarray(z["xi_psi"], dtype=np.float64)
    f_phi_us = np.asarray(z["F_phi_us"], dtype=np.float64)
    f_psi_us = np.asarray(z["F_psi_us"], dtype=np.float64)

    axes = np.empty((5, 2), dtype=object)
    for i, lam in enumerate(LAMS):
        tag = lam_tag(lam)
        axes[i, 0] = fig.add_subplot(gs[i, 0], sharex=axes[0, 0] if i else None)
        axes[i, 1] = fig.add_subplot(
            gs[i, 1], sharex=axes[0, 1] if i else None, sharey=axes[i, 0]
        )
        plot_1d_curve(
            axes[i, 0],
            xi_phi,
            z[f"F_phi_lam{tag}"],
            z[f"F_phi_std_lam{tag}"],
            f_phi_us,
        )
        plot_1d_curve(
            axes[i, 1],
            xi_psi,
            z[f"F_psi_lam{tag}"],
            z[f"F_psi_std_lam{tag}"],
            f_psi_us,
        )
        axes[i, 0].text(
            0.04,
            0.90,
            rf"$\lambda={lam:.2f}$",
            transform=axes[i, 0].transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
        )

    for i in range(5):
        for j in range(2):
            ax = axes[i, j]
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0.0, VMAX)
            ax.set_xticks(PI_TICKS)
            ax.set_xticklabels(PI_LABELS if i == 4 else [])
            if j == 1:
                ax.tick_params(labelleft=False)
            ax.tick_params(which="both", top=False, right=False)
            ax.set_xlabel(r"$\phi$ / rad" if j == 0 else r"$\psi$ / rad" if i == 4 else "")
            if i != 4:
                ax.set_xlabel("")

    axes[0, 0].set_title(r"$\phi$", pad=10, fontsize=24)
    axes[0, 1].set_title(r"$\psi$", pad=10, fontsize=24)
    fig.text(0.032, 0.50, FE_LABEL, rotation=90, va="center", ha="center", fontsize=24)

    handles = [
        Line2D([0], [0], color=COLOR_US, lw=3.4, ls="--", label="umbrella sampling"),
        Line2D([0], [0], color=COLOR_SFPT, lw=3.4, label="sFPT"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=22,
        bbox_to_anchor=(0.55, 0.995),
        handlelength=2.6,
        columnspacing=2.0,
    )
    fig.savefig(out / "fes_1d_lambda_5x2.png", dpi=220)
    fig.savefig(out / "fes_1d_lambda_5x2.pdf")
    plt.close(fig)
    print("[ok] fes_1d_lambda_5x2", flush=True)


def lam_label(lam: float) -> str:
    return rf"$\lambda={float(lam):.2f}$"


def plot_flatten_law(out: Path, z) -> None:
    plt.rcParams.update(jctc_rcparams(scale=1.12))
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.55),
        layout="constrained",
        subplot_kw={"box_aspect": 1},
        gridspec_kw={"wspace": 0.18},
    )

    lams = np.asarray(z["lambdas"], dtype=np.float64)
    xmax = 0.0
    for lam in lams:
        tag = lam_tag(lam)
        xmax = max(xmax, float(np.nanmax(z[f"x_fit_lam{tag}"])))
    xmin, xmax = 0.0, max(xmax * 1.03, 1.0)
    xline = np.linspace(xmin, xmax, 250)

    for lam in lams:
        tag = lam_tag(lam)
        x = np.asarray(z[f"x_fit_lam{tag}"], dtype=np.float64)
        y = np.asarray(z[f"y_fit_lam{tag}"], dtype=np.float64)
        c = lam_color(lam)
        slope = float(z[f"slope_lam{tag}"])
        intercept = float(z[f"intercept_lam{tag}"])
        slope_th = float(lam) / (1.0 + float(lam))
        c_th = float(np.mean(y) - slope_th * np.mean(x))
        ax_a.scatter(x, y, s=7, color=c, alpha=0.22, linewidths=0, zorder=2, rasterized=True)
        ax_a.plot(
            xline,
            slope * xline + intercept,
            color=c,
            lw=2.1,
            zorder=4,
            label=lam_label(lam),
        )
        ax_a.plot(xline, slope_th * xline + c_th, color=c, lw=1.5, ls="--", zorder=3)

    h, _ = ax_a.get_legend_handles_labels()
    h.extend(
        [
            Line2D([0], [0], color="0.25", lw=2.0, ls="-", label="Linear fit"),
            Line2D(
                [0],
                [0],
                color="0.25",
                lw=1.5,
                ls="--",
                label=r"Theory: $\alpha=\lambda/(1+\lambda)$",
            ),
        ]
    )
    ax_a.legend(handles=h, frameon=False, fontsize=14, loc="upper left", handlelength=2.0)
    ax_a.set_xlabel(r"$F_{\mathrm{US}}(\phi,\psi)$ (kcal/mol)")
    ax_a.set_ylabel(r"$F_{\mathrm{eff},\lambda}(\phi,\psi)$ (kcal/mol)")
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_title(r"Free-energy scaling at different $\lambda$", pad=10)
    ax_a.tick_params(which="both", top=False, right=False)
    panel_letter(ax_a, "a", x=-0.14, y=1.05)

    # α = λ/(1+λ) as defined in panel a; theory line is y = x.
    xth = lams / (1.0 + lams)
    sl = np.asarray(z["slope"], dtype=np.float64)
    se = np.asarray(z["slope_se"], dtype=np.float64)
    lim = 0.12
    t = np.linspace(0.0, lim, 200)
    ax_b.plot(t, t, color="0.28", ls="--", lw=1.8, zorder=2, label=r"$\alpha_{\mathrm{fit}}=\alpha_{\mathrm{theory}}$")
    ax_b.errorbar(
        xth,
        sl,
        yerr=se,
        fmt="o",
        color="0.15",
        ecolor="0.40",
        elinewidth=1.2,
        capsize=3.5,
        ms=0,
        zorder=3,
    )
    for xt, yy, lam in zip(xth, sl, lams):
        ax_b.plot(xt, yy, "o", color=lam_color(lam), ms=10.5, zorder=4, label=lam_label(lam))
    ax_b.set_xlabel(r"$\alpha_{\mathrm{theory}}=\lambda/(1+\lambda)$")
    ax_b.set_ylabel(r"Fitted scaling factor $\alpha_{\mathrm{fit}}$")
    ax_b.set_xlim(0.0, lim)
    ax_b.set_ylim(0.0, lim)
    ax_b.set_title("Fitted versus theoretical scaling factor", pad=10)
    ax_b.legend(frameon=False, fontsize=14, loc="upper left", ncol=1)
    ax_b.tick_params(which="both", top=False, right=False)
    panel_letter(ax_b, "b", x=-0.14, y=1.05)

    fig.savefig(out / "flatten_law.png", dpi=220)
    fig.savefig(out / "flatten_law.pdf")
    plt.close(fig)
    print("[ok] flatten_law", flush=True)


def main() -> None:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    fes = np.load(FES_NPZ)
    flat = np.load(FLAT_NPZ)
    plot_fes_1d_grid(out, fes)
    plot_flatten_law(out, flat)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
