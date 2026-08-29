#!/usr/bin/env python3
"""Latest sFPT vs live WT-MetaD: 2×3 2D FES (row1 sFPT, row2 MetaD) + 1D overlay."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize

SFP = Path(__file__).resolve().parent.parent / "tetra_alanine_sFPT"
sys.path.insert(0, str(SFP))

from fes_window import (  # noqa: E402
    COLOR_SFPT,
    FE_LABEL,
    PHI_PAIRS,
    PI_LABELS,
    PI_TICKS,
    REMD_PLOT_SMOOTH_2D,
    SFPT_SMOOTH_2D,
    fes_cmap,
    relative_fes,
    smooth_fes_torus,
)

COLOR_META = "#2166ac"  # same blue as dialanine WT-MetaD lines

KT_KCAL = 0.00198720425864083 * 300.0
DEFAULT_SFPT_DIR = SFP / "results" / "tetra_alanine_phi3_r1024_s0_94092" / "fes_2d"
DEFAULT_META = (
    Path(__file__).resolve().parent
    / "results"
    / "tetra_alanine_metad_r64_s0_94100"
    / "live"
    / "fes_2d.npz"
)
T_RE = re.compile(r"fes_2d_t(\d+)ps\.npz$")


def latest_sfpt_npz(fes_dir: Path) -> Path:
    cands = []
    for p in fes_dir.glob("fes_2d_t*ps.npz"):
        m = T_RE.search(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        raise FileNotFoundError(f"no fes_2d_t*ps.npz in {fes_dir}")
    cands.sort()
    return cands[-1][1]


def _wrap_1d(xi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([xi, xi[:1] + 2.0 * np.pi]), np.concatenate([y, y[:1]])


def fes_1d_from_2d(F: np.ndarray, *, kT: float, axis: int) -> np.ndarray:
    w = np.exp(-np.clip(relative_fes(F), 0.0, 80.0) / float(kT))
    p = np.clip(w.sum(axis=0 if axis == 0 else 1), 1e-300, None)
    return relative_fes(-float(kT) * np.log(p))


def meta_1d_from_pairs(
    f2: dict[tuple[int, int], np.ndarray], *, kT: float
) -> np.ndarray:
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


def plot_compare(sfpt_npz: Path, meta_npz: Path, out: Path, *, vmax: float) -> None:
    z = np.load(sfpt_npz)
    m = np.load(meta_npz)
    ax_s = np.asarray(z["ax"], dtype=np.float64)
    ax_m = np.asarray(m["ax"], dtype=np.float64)
    f_s, f_m = _pairs(z), _pairs(m)
    f1_s = np.stack([z["F1_phi1"], z["F1_phi2"], z["F1_phi3"]]).astype(np.float64)
    f1_m = meta_1d_from_pairs(f_m, kT=KT_KCAL)
    t_s, lam = float(z["t_ps"]), float(z["lam"])
    t_m, n_h = float(m["t_ps"]), int(m["n_hills"])

    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    fig = plt.figure(figsize=(11.4, 11.0))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[2.15, 0.88],
        hspace=0.18, left=0.08, right=0.91, bottom=0.06, top=0.93,
    )
    gs2 = outer[0].subgridspec(
        2, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.18, hspace=0.28,
    )
    gs1 = outer[1].subgridspec(1, 3, wspace=0.32)
    im = None

    def _draw(a, ax, F, *, sigma: float, ylab_on: bool, xlab: str, ylab: str, title_s: str):
        nonlocal im
        z2 = np.minimum(smooth_fes_torus(ax, F, sigma=sigma), float(vmax))
        im = a.contourf(ax, ax, z2, levels=levels, cmap=cmap, norm=norm, extend="max")
        a.contour(
            ax, ax, np.clip(z2, 0.0, float(vmax)),
            levels=levels[::3], colors="0.45", linewidths=0.35, alpha=0.45,
        )
        a.set_xlim(-np.pi, np.pi)
        a.set_ylim(-np.pi, np.pi)
        a.set_xticks(PI_TICKS)
        a.set_xticklabels(PI_LABELS)
        a.set_yticks(PI_TICKS)
        a.set_yticklabels(PI_LABELS if ylab_on else [])
        a.set_xlabel(xlab)
        if ylab_on:
            a.set_ylabel(ylab)
        a.set_title(title_s, fontsize=11)
        a.set_aspect("equal")
        a.tick_params(which="both", top=False, right=False)

    rows = (
        (f_s, ax_s, "sFPT", SFPT_SMOOTH_2D),
        (f_m, ax_m, "MetaD", REMD_PLOT_SMOOTH_2D),
    )
    for row, (fmap, ax, lab, sigma) in enumerate(rows):
        for col, (i, j, ii, jj) in enumerate(PHI_PAIRS):
            a = fig.add_subplot(gs2[row, col])
            _draw(
                a, ax, fmap[(i, j)],
                sigma=sigma, ylab_on=(col == 0),
                xlab=rf"$\phi_{ii}$ / rad",
                ylab=rf"$\phi_{jj}$ / rad",
                title_s=rf"{lab}  $(\phi_{ii},\phi_{jj})$",
            )
    cax = fig.add_subplot(gs2[:, 3])
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(FE_LABEL)

    for i in range(3):
        a = fig.add_subplot(gs1[0, i])
        xs, ys = _wrap_1d(ax_s, relative_fes(f1_s[i]))
        xm, ym = _wrap_1d(ax_m, relative_fes(f1_m[i]))
        a.plot(xs, ys, color=COLOR_SFPT, lw=1.7, label="sFPT", zorder=3)
        a.plot(xm, ym, color=COLOR_META, lw=1.7, ls="-", label="MetaD", zorder=4)
        a.set_xlim(-np.pi, np.pi)
        a.set_ylim(0.0, float(vmax) * 1.02)
        a.set_xticks(PI_TICKS)
        a.set_xticklabels(PI_LABELS)
        a.set_xlabel(rf"$\phi_{i + 1}$ / rad")
        if i == 0:
            a.set_ylabel(FE_LABEL)
        if i == 1:
            a.legend(frameon=False, fontsize=8, loc="upper right")
        a.set_title(rf"$F(\phi_{i + 1})$", fontsize=11)
        a.tick_params(which="both", top=False, right=False)
        a.axvline(0.0, color="0.8", lw=0.6, ls=":", zorder=1)

    fig.suptitle(
        rf"sFPT $t={t_s:g}$ ps $\lambda={lam:g}$   vs   WT-MetaD $t={t_m:g}$ ps  "
        rf"($n_\mathrm{{hills}}={n_h}$)",
        fontsize=12,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}  sFPT t={t_s:g}ps  MetaD t={t_m:g}ps", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sfpt-dir", type=Path, default=DEFAULT_SFPT_DIR)
    p.add_argument("--sfpt-npz", type=Path, default=None)
    p.add_argument("--meta-npz", type=Path, default=DEFAULT_META)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--vmax", type=float, default=16.0)
    p.add_argument("--watch", type=float, default=0.0)
    args = p.parse_args()
    out = args.out or (Path(args.meta_npz).parent / "compare_sfpt_t0220ps.png")
    last = None
    while True:
        sfpt = Path(args.sfpt_npz) if args.sfpt_npz else latest_sfpt_npz(Path(args.sfpt_dir))
        meta = Path(args.meta_npz)
        stamp = (sfpt.stat().st_mtime, meta.stat().st_mtime)
        if stamp != last:
            plot_compare(sfpt, meta, out, vmax=float(args.vmax))
            last = stamp
        if float(args.watch) <= 0.0:
            break
        time.sleep(float(args.watch))


if __name__ == "__main__":
    main()
