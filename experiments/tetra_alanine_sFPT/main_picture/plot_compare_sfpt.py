#!/usr/bin/env python3
"""3-seed sFPT vs WT-MetaD: 3×3 panels a–i, fully aligned.

a–c sFPT 2D, d–f WT-MetaD 2D, g–i 1D mean±std.
Output: ``experiments/tetra_alanine_sFPT/main_picture/compare_sfpt.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SFP = HERE.parent
sys.path.insert(0, str(SFP))
sys.path.insert(0, str(HERE))

from fes_window import FE_LABEL, PHI_PAIRS, PI_LABELS, PI_TICKS, fes_cmap  # noqa: E402
from plot_main_picture import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    KT_KCAL,
    META_DIRS,
    REMD_PLOT_SMOOTH_2D,
    SFPT_DIRS,
    SFPT_SMOOTH_2D,
    VMAX,
    _wrap_1d,
    boltzmann_mean,
    fes_full_square,
    interp_1d,
    latest_sfpt_npz,
    mean_std,
    meta_live,
    nature_rcparams,
    relative_fes,
)


def fes_1d_from_2d(F: np.ndarray, *, kT: float, axis: int) -> np.ndarray:
    w = np.exp(-np.clip(relative_fes(F), 0.0, 80.0) / float(kT))
    p = np.clip(w.sum(axis=0 if axis == 0 else 1), 1e-300, None)
    return relative_fes(-float(kT) * np.log(p))


def meta_1d_from_pairs(f2: dict[tuple[int, int], np.ndarray], *, kT: float) -> np.ndarray:
    acc = [[] for _ in range(3)]
    acc[0].append(fes_1d_from_2d(f2[(0, 1)], kT=kT, axis=0))
    acc[0].append(fes_1d_from_2d(f2[(0, 2)], kT=kT, axis=0))
    acc[1].append(fes_1d_from_2d(f2[(0, 1)], kT=kT, axis=1))
    acc[1].append(fes_1d_from_2d(f2[(1, 2)], kT=kT, axis=0))
    acc[2].append(fes_1d_from_2d(f2[(0, 2)], kT=kT, axis=1))
    acc[2].append(fes_1d_from_2d(f2[(1, 2)], kT=kT, axis=1))
    out = np.zeros((3, acc[0][0].shape[0]), dtype=np.float64)
    for i in range(3):
        p = np.mean(
            [np.exp(-np.clip(a, 0.0, 80.0) / float(kT)) for a in acc[i]], axis=0
        )
        out[i] = relative_fes(-float(kT) * np.log(np.clip(p, 1e-300, None)))
    return out


def _pairs(z) -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 1): np.asarray(z["F_phi1_phi2"], dtype=np.float64),
        (0, 2): np.asarray(z["F_phi1_phi3"], dtype=np.float64),
        (1, 2): np.asarray(z["F_phi2_phi3"], dtype=np.float64),
    }


def load_seeds() -> tuple:
    ax1 = np.linspace(-np.pi, np.pi, 73)
    xx = None
    f2_s = {k: [] for k in ((0, 1), (0, 2), (1, 2))}
    f2_m = {k: [] for k in ((0, 1), (0, 2), (1, 2))}
    f1_s, f1_m = [], []
    t_s, t_m, n_h, lam = [], [], [], []
    for d in SFPT_DIRS:
        p = latest_sfpt_npz(d)
        z = np.load(p)
        ax = np.asarray(z["ax"], dtype=np.float64)
        fmap = _pairs(z)
        for key, F in fmap.items():
            xx, zf = fes_full_square(ax, F, sigma=SFPT_SMOOTH_2D)
            f2_s[key].append(zf)
        f1 = np.stack(
            [
                interp_1d(ax, relative_fes(z["F1_phi1"]), ax1),
                interp_1d(ax, relative_fes(z["F1_phi2"]), ax1),
                interp_1d(ax, relative_fes(z["F1_phi3"]), ax1),
            ],
            axis=0,
        )
        f1_s.append(f1)
        t_s.append(float(z["t_ps"]))
        lam.append(float(z["lam"]))
        print(f"[sFPT] {d.name} {p.name} t={t_s[-1]:g} ps", flush=True)
    for d in META_DIRS:
        p = meta_live(d)
        m = np.load(p)
        ax = np.asarray(m["ax"], dtype=np.float64)
        fmap = _pairs(m)
        for key, F in fmap.items():
            _, zf = fes_full_square(ax, F, sigma=REMD_PLOT_SMOOTH_2D)
            f2_m[key].append(zf)
        f1 = meta_1d_from_pairs(fmap, kT=KT_KCAL)
        f1_m.append(np.stack([interp_1d(ax, f1[i], ax1) for i in range(3)], axis=0))
        t_m.append(float(m["t_ps"]))
        n_h.append(int(m["n_hills"]))
        print(f"[Meta] {d.name} t={t_m[-1]:g} ps  nhills={n_h[-1]}", flush=True)
    assert xx is not None
    f2s_mean = {k: boltzmann_mean(v) for k, v in f2_s.items()}
    f2m_mean = {k: boltzmann_mean(v) for k, v in f2_m.items()}
    f1s = np.stack(f1_s, axis=0)
    f1m = np.stack(f1_m, axis=0)
    return xx, ax1, f2s_mean, f2m_mean, f1s, f1m, t_s, t_m, n_h, lam


def plot_compare(*, vmax: float, out: Path) -> None:
    xx, ax1, f2s, f2m, f1s, f1m, t_s, t_m, n_h, lam = load_seeds()
    f1s_mean, f1s_std = mean_std(f1s)
    f1m_mean, f1m_std = mean_std(f1m)

    plt.rcParams.update(nature_rcparams())
    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()

    fig_w, fig_h = 12.4, 13.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, bottom, top = 0.108, 0.048, 0.875
    right = 0.868
    gap_x_min, gap_y_min = 0.058, 0.108
    avail_w = right - left
    avail_h = top - bottom
    side = min(
        (avail_w - 2.0 * gap_x_min) / 3.0,
        (avail_h - 2.0 * gap_y_min) / 3.0,
    )
    extra_w = avail_w - 3.0 * side
    extra_h = avail_h - 3.0 * side
    gap_x = extra_w / 2.0
    gap_y = extra_h / 2.0
    xs = [left + col * (side + gap_x) for col in range(3)]
    ys = [top - (row + 1) * side - row * gap_y for row in range(3)]

    def _letter(ax, letter: str) -> None:
        ax.text(
            0.0, 1.06, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=16, fontweight="bold",
            color="0.08", clip_on=False, zorder=10,
        )

    def _draw_2d(ax, z2, *, xlab: str, ylab: str):
        z2 = np.minimum(relative_fes(z2), float(vmax))
        im = ax.contourf(xx, xx, z2, levels=levels, cmap=cmap, norm=norm, extend="max")
        ax.contour(
            xx, xx, np.clip(z2, 0.0, float(vmax)),
            levels=levels[::3], colors="0.45", linewidths=0.35, alpha=0.45,
        )
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.margins(0.0)
        ax.set_xticks(PI_TICKS)
        ax.set_xticklabels(PI_LABELS)
        ax.set_yticks(PI_TICKS)
        ax.set_yticklabels(PI_LABELS)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_aspect("auto")
        ax.tick_params(which="both", top=False, right=False)
        return im

    letters = "abcdefghi"
    im = None
    for row, fmap in enumerate((f2s, f2m)):
        for col, (_i, _j, ii, jj) in enumerate(PHI_PAIRS):
            ax = fig.add_axes([xs[col], ys[row], side, side])
            im = _draw_2d(
                ax, fmap[(_i, _j)],
                xlab=rf"$\phi_{ii}$ / rad",
                ylab=rf"$\phi_{jj}$ / rad",
            )
            _letter(ax, letters[row * 3 + col])

    for i in range(3):
        ax = fig.add_axes([xs[i], ys[2], side, side])
        xw, yw = _wrap_1d(ax1, f1s_mean[i])
        _, ys_lo = _wrap_1d(ax1, f1s_mean[i] - f1s_std[i])
        _, ys_hi = _wrap_1d(ax1, f1s_mean[i] + f1s_std[i])
        xm, ym = _wrap_1d(ax1, f1m_mean[i])
        _, ym_lo = _wrap_1d(ax1, f1m_mean[i] - f1m_std[i])
        _, ym_hi = _wrap_1d(ax1, f1m_mean[i] + f1m_std[i])
        ax.fill_between(xm, ym_lo, ym_hi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
        ax.plot(xm, ym, color=COLOR_META, lw=1.8, zorder=2)
        ax.fill_between(xw, ys_lo, ys_hi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
        ax.plot(xw, yw, color=COLOR_SFPT, lw=2.1, zorder=4)
        ax.axvline(0.0, color="0.55", lw=1.0, ls=":", zorder=0)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0.0, float(vmax))
        ax.set_xticks(PI_TICKS)
        ax.set_xticklabels(PI_LABELS)
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.set_xlabel(rf"$\phi_{i + 1}$ / rad")
        ax.set_ylabel(FE_LABEL)
        ax.set_aspect("auto")
        ax.tick_params(which="both", top=False, right=False)
        _letter(ax, letters[6 + i])

    fig.text(
        0.028, ys[0] + 0.5 * side, "sFPT",
        ha="center", va="center", rotation=90, fontsize=16,
    )
    fig.text(
        0.028, ys[1] + 0.5 * side, "WT-MetaD",
        ha="center", va="center", rotation=90, fontsize=16,
    )

    cax = fig.add_axes(
        [xs[2] + side + 0.012, ys[1], 0.016, 2.0 * side + gap_y]
    )
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(r"kcal mol$^{-1}$", fontsize=14, labelpad=3)
    cbar.ax.tick_params(labelsize=13, length=3.2, width=1.0)
    cbar.set_ticks(np.linspace(0.0, float(vmax), 5))

    fig.legend(
        handles=[
            Line2D([0], [0], color=COLOR_META, lw=2.4, label="WT-MetaD"),
            Line2D([0], [0], color=COLOR_SFPT, lw=2.6, label="sFPT"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.48, 0.995),
        ncol=2,
        frameon=False,
        fontsize=15,
        handlelength=1.8,
        borderaxespad=0.0,
        columnspacing=1.4,
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches=None, pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=None, pad_inches=0.08)
    plt.close(fig)
    print(
        f"[ok] {out}  sFPT t={t_s}  MetaD t={t_m}  "
        f"mean t_s={float(np.mean(t_s)):g} t_m={float(np.mean(t_m)):g}",
        flush=True,
    )

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vmax", type=float, default=VMAX)
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "compare_sfpt.png",
    )
    args = p.parse_args()
    plot_compare(vmax=float(args.vmax), out=Path(args.out))


if __name__ == "__main__":
    main()
