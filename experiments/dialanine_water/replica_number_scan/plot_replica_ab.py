#!/usr/bin/env python3
"""1×2 replica-scan collage: ΔG(t) and ΔG MAE vs R, plus WT-MetaD MAE.

Panel a: ΔG(t) for R=320/512/1024 (xlim 0–150 ps), same seeds as dG_mae_vs_R.
Panel b: sFPT ΔG MAE vs R, with a WT-MetaD ΔG MAE reference from the same
MetaD runs used in dialanine_main_picture (convergence_320_90535).

MetaD MAE is averaged from 80 ps to the end of the MetaD trajectory.
With two MetaD seeds, mean±std at each t uniquely determine the unordered
pair, so the mean of the two per-seed MAEs is recovered without identities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, ticker
from matplotlib.lines import Line2D

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "dialanine_water"
OUT = EXP / "replica_number_scan"
MAE_NPZ = OUT / "dG_mae_vs_R.npz"
WAT_DG = EXP / "sFPT" / "convergence_320_90535" / "fes_rmse_vs_time.npz"

sys.path.insert(0, str(OUT))
from plot_deltaG_vs_time_by_R import (  # noqa: E402
    COLOR_US,
    jctc_rcparams,
    load_series,
    panel_letter,
    plot_a,
    style_ax,
)
from plot_dG_mae_vs_R import COLORS, PANEL_A_R, T_LO  # noqa: E402

COLOR_META = "#2166AC"


def metad_mae_two_seeds(path: Path) -> tuple[float, float, float]:
    """Mean of per-seed MAEs from T_LO ps to the end of the MetaD trajectory."""
    z = np.load(path)
    t = np.asarray(z["t_metad_ps"], dtype=np.float64)
    dG = np.asarray(z["dG_metad"], dtype=np.float64)
    std = np.asarray(z["dG_metad_std"], dtype=np.float64)
    dG_us = float(z["dG_us_ref"])
    n_meta = len(z["metad_dirs"])
    z.close()
    sel = t >= T_LO
    if not np.any(sel):
        raise RuntimeError(f"MetaD: no samples with t >= {T_LO} ps")
    t_hi = float(t[sel][-1])
    dG_w = dG[sel]
    std_w = std[sel]
    mae_mean_curve = float(np.mean(np.abs(dG_w - dG_us)))
    if n_meta < 2:
        return mae_mean_curve, t_hi, mae_mean_curve
    # n=2, ddof=1: the two values at each t are mean ± std/sqrt(2)
    half = std_w / np.sqrt(2.0)
    mae_pair = 0.5 * (
        np.abs(dG_w + half - dG_us) + np.abs(dG_w - half - dG_us)
    )
    return float(np.mean(mae_pair)), t_hi, mae_mean_curve


def plot_b(ax, Rs, y, e, mae_meta: float) -> None:
    ax.axhline(
        mae_meta,
        color=COLOR_META,
        ls="--",
        lw=1.6,
        zorder=1,
        label="WT-MetaD",
    )
    ax.errorbar(
        Rs,
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
    for x, yy in zip(Rs, y):
        ax.plot(x, yy, "o", color=COLORS.get(int(x), "C0"), ms=7.2, zorder=3)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$\Delta G$ MAE vs US / kcal mol$^{-1}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(Rs))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.ticklabel_format(axis="x", style="plain")
    rmin, rmax = float(np.min(Rs)), float(np.max(Rs))
    ax.set_xlim(rmin / 1.2, rmax * 1.2)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper right", fontsize=12.5, handlelength=1.8)


def render(
    *,
    out_stem: Path,
    r_npz: dict | None = None,
    r_meta: dict | None = None,
    panel_a_r: tuple[int, ...] | None = None,
    subtitle: str | None = None,
    mae_npz_path: Path | None = None,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dG_us, series = load_series(r_npz, r_meta)
    if mae_npz_path is None:
        from plot_dG_mae_vs_R import mae_from_npz, R_NPZ as _N, R_META as _M

        Rs, y, e, _per = mae_from_npz(
            r_npz if r_npz is not None else _N,
            r_meta if r_meta is not None else _M,
        )
        Rs = np.asarray(Rs, dtype=np.float64)
    else:
        zm = np.load(mae_npz_path)
        Rs = np.asarray(zm["R"], dtype=np.float64)
        y = np.asarray(zm["mae_mean"], dtype=np.float64)
        e = np.asarray(zm["mae_std"], dtype=np.float64)
        zm.close()

    mae_meta, t_hi, mae_curve = metad_mae_two_seeds(WAT_DG)
    print(
        f"[MetaD] window {T_LO:.0f}–{t_hi:.0f} ps  "
        f"per-seed MAE mean={mae_meta:.3f}  "
        f"MAE of mean curve={mae_curve:.3f}",
        flush=True,
    )

    if panel_a_r is None:
        panel_a_r = PANEL_A_R
    series_a = {R: series[R] for R in panel_a_r if R in series}

    plt.rcParams.update(jctc_rcparams(scale=1.0))
    fig = plt.figure(figsize=(9.6, 5.05 if subtitle else 4.65))
    gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        wspace=0.34,
        left=0.08,
        right=0.995,
        bottom=0.15,
        top=0.78 if subtitle else 0.82,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    handles = plot_a(ax_a, dG_us, series_a)
    style_ax(ax_a)
    panel_letter(ax_a, "a")

    plot_b(ax_b, Rs, y, e, mae_meta)
    style_ax(ax_b)
    panel_letter(ax_b, "b")

    n_h = len(handles)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(n_h, 6),
        frameon=False,
        fontsize=12,
        handlelength=1.8,
        columnspacing=1.15,
        borderaxespad=0.0,
        title=subtitle,
        title_fontsize=12,
    )

    png = Path(out_stem).with_suffix(".png")
    pdf = Path(out_stem).with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)


def main() -> None:
    zm = MAE_NPZ if MAE_NPZ.is_file() else None
    render(out_stem=OUT / "replica_ab", mae_npz_path=zm)


if __name__ == "__main__":
    main()
