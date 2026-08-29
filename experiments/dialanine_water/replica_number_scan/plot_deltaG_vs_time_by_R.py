#!/usr/bin/env python3
"""ΔG(t) overlay vs replica number R, matching λ-scan lambda_abc.png panel a.

For each R, mean ± std of ΔG(t) across the same two seeds used in
plot_dG_mae_vs_R.py (prefer s0 and s2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "dialanine_water"
OUT = EXP / "replica_number_scan"

sys.path.insert(0, str(OUT))
from plot_dG_mae_vs_R import (  # noqa: E402
    COLORS,
    R_META,
    R_NPZ,
    dG_series,
    pick_two,
    seed_of,
)

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


def mean_std(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=0)
    if stack.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=0, ddof=1)


def load_series(
    r_npz: dict[int, Path] | None = None,
    r_meta: dict[int, Path] | None = None,
) -> tuple[float, dict[int, dict]]:
    r_npz = R_NPZ if r_npz is None else r_npz
    r_meta = R_META if r_meta is None else r_meta
    series: dict[int, dict] = {}
    dG_us = None
    for R in sorted(r_npz):
        if not Path(r_npz[R]).is_file() or not Path(r_meta[R]).is_file():
            print(f"[skip] R={R}: missing npz", flush=True)
            continue
        g1d = np.load(r_npz[R])
        meta = np.load(r_meta[R])
        xi = np.asarray(g1d["xi"], dtype=np.float64)
        t = np.asarray(g1d["t_ps"], dtype=np.float64)
        runs = np.asarray(g1d["F1d_kde_runs"], dtype=np.float64)
        dG_us = float(meta["dG_us_ref"])
        dirs = [str(x) for x in meta["fpt_dirs"]]
        seeds = [seed_of(d) for d in dirs]
        take = pick_two(seeds)
        dG_list = []
        used = []
        for i in take:
            dG_list.append(dG_series(xi, runs[i]))
            used.append(seeds[i])
        dG, std = mean_std(np.stack(dG_list, axis=0))
        series[R] = {
            "t": t,
            "dG": dG,
            "dG_std": std,
            "seeds": used,
        }
        print(
            f"[R={R}] n_t={len(t)} t={t[0]:.0f}–{t[-1]:.0f} ps  "
            f"seeds={used}  ΔG_end={dG[-1]:.3f}±{std[-1]:.3f}",
            flush=True,
        )
        g1d.close()
        meta.close()
    assert dG_us is not None
    return dG_us, series


def plot_a(ax, dG_us: float, series: dict[int, dict]) -> list[Line2D]:
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
    for R in sorted(series):
        s = series[R]
        t = s["t"]
        dG = s["dG"]
        std = s["dG_std"]
        c = COLORS.get(int(R), "C0")
        ax.plot(t, dG, color=c, lw=1.7, zorder=3)
        if np.any(std > 0):
            ax.fill_between(
                t, dG - std, dG + std, color=c, alpha=0.18, lw=0, zorder=2
            )
        handles.append(Line2D([0], [0], color=c, lw=1.8, label=rf"$R$={int(R)}"))
    ax.set_xlim(0.0, 150.0)
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    return handles


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dG_us, series = load_series()

    save: dict = {"dG_us_ref": np.float64(dG_us), "R": np.asarray(sorted(series), dtype=np.int64)}
    for R, s in series.items():
        save[f"t_R{R}"] = s["t"]
        save[f"dG_R{R}"] = s["dG"]
        save[f"dG_std_R{R}"] = s["dG_std"]
        save[f"seeds_R{R}"] = np.asarray(s["seeds"], dtype=np.int64)
    np.savez_compressed(OUT / "deltaG_vs_time_by_R.npz", **save)

    plt.rcParams.update(jctc_rcparams(scale=1.0))
    fig = plt.figure(figsize=(5.4, 4.65))
    ax = fig.add_axes((0.16, 0.16, 0.80, 0.66))
    handles = plot_a(ax, dG_us, series)
    style_ax(ax)
    panel_letter(ax, "a")
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=13,
        handlelength=1.8,
        columnspacing=1.35,
        borderaxespad=0.0,
    )

    png = OUT / "deltaG_vs_time_by_R.png"
    pdf = OUT / "deltaG_vs_time_by_R.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)


if __name__ == "__main__":
    main()
