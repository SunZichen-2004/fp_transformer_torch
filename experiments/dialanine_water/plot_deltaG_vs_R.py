#!/usr/bin/env python3
"""Overlay sFPT ΔG vs sim-time / wall-clock for different replica counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from plot_fes_compare import nature_rcparams  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import COLOR_US  # noqa: E402

# Distinct colors for R
COLORS = {
    320: "#1b9e77",
    512: "#d95f02",
    1024: "#7570b3",
    2048: "#e7298a",
}


def load_dg(path: Path) -> dict:
    z = np.load(path)
    dG_ref = float(z["dG_us_ref"]) if "dG_us_ref" in z.files else float(z["dG_metad_ref"])
    return {
        "t": np.asarray(z["t_kde_ps"], dtype=np.float64),
        "tw": np.asarray(z["tw_kde_h"], dtype=np.float64),
        "dG": np.asarray(z["dG_kde"], dtype=np.float64),
        "dG_std": (
            np.asarray(z["dG_kde_std"], dtype=np.float64)
            if "dG_kde_std" in z.files
            else np.zeros_like(z["dG_kde"], dtype=np.float64)
        ),
        "dG_ref": dG_ref,
    }


def plot_overlay(
    out: Path,
    *,
    series: dict[int, dict],
    xkey: str,
    xlabel: str,
    stem: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] = (-5.0, 5.0),
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    dG_ref = next(iter(series.values()))["dG_ref"]
    ax.axhspan(dG_ref - 1.0, dG_ref + 1.0, color=COLOR_US, alpha=0.15, lw=0, zorder=0)
    ax.axhline(
        dG_ref,
        color=COLOR_US,
        ls=":",
        lw=1.2,
        zorder=1,
        label=rf"US $\Delta G$={dG_ref:.2f}",
    )
    for R in sorted(series):
        s = series[R]
        x = s[xkey]
        y = s["dG"]
        e = s["dG_std"]
        c = COLORS.get(R, "C0")
        ax.plot(x, y, color=c, lw=1.5, label=f"R={R}", zorder=3)
        if np.any(e > 0):
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.18, lw=0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        r"$\Delta G=k_BT\ln(P_{\phi<0}/P_{\phi>0})$ / kcal mol$^{-1}$"
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(left=0.0)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--npz-320", type=Path, required=True)
    p.add_argument("--npz-512", type=Path, required=True)
    p.add_argument("--npz-1024", type=Path, required=True)
    p.add_argument("--npz-2048", type=Path, default=None)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    series = {
        320: load_dg(args.npz_320),
        512: load_dg(args.npz_512),
        1024: load_dg(args.npz_1024),
    }
    if args.npz_2048 is not None:
        series[2048] = load_dg(args.npz_2048)
    for R, s in series.items():
        print(
            f"[R={R}] n={len(s['t'])} t_end={s['t'][-1]:.1f} ps  "
            f"tw_end={s['tw'][-1]:.3f} h  ΔG_end={s['dG'][-1]:.3f}",
            flush=True,
        )

    plot_overlay(
        out,
        series=series,
        xkey="t",
        xlabel=r"$t$ / ps",
        stem="deltaG_vs_time_by_R",
        xlim=(0.0, 150.0),
        ylim=(2.0, 7.0),
    )
    plot_overlay(
        out,
        series=series,
        xkey="t",
        xlabel=r"$t$ / ps",
        stem="deltaG_vs_time_by_R_divein",
        xlim=(0.0, 40.0),
        ylim=(2.0, 7.0),
    )
    plot_overlay(
        out,
        series=series,
        xkey="tw",
        xlabel=r"wall-clock time / h",
        stem="deltaG_vs_wallclock_by_R",
        xlim=None,
        ylim=(2.0, 7.0),
    )
    plot_overlay(
        out,
        series=series,
        xkey="tw",
        xlabel=r"wall-clock time / h",
        stem="deltaG_vs_wallclock_by_R_divein",
        xlim=(0.0, 0.5),
        ylim=(2.0, 8.0),
    )
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
