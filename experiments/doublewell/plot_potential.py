#!/usr/bin/env python3
"""Plot the classic 2D double-well potential energy surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fp_transformer_torch.potential.double_well import (  # noqa: E402
    DoubleWell2D,
    DoubleWellParams,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h", type=float, default=5.0, help="barrier height")
    p.add_argument("--k", type=float, default=10.0, help="fast-DOF spring constant")
    p.add_argument("--x1-lim", type=float, default=2.0)
    p.add_argument("--x2-lim", type=float, default=2.0)
    p.add_argument("--n", type=int, default=241)
    p.add_argument("--vmax", type=float, default=12.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figs",
    )
    args = p.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pot = DoubleWell2D(DoubleWellParams(h=float(args.h), k=float(args.k)))
    x1 = np.linspace(-float(args.x1_lim), float(args.x1_lim), int(args.n))
    x2 = np.linspace(-float(args.x2_lim), float(args.x2_lim), int(args.n))
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    pts = torch.as_tensor(
        np.stack([X1.ravel(), X2.ravel()], axis=-1), dtype=torch.float64
    )
    with torch.no_grad():
        V = pot.energy(pts).reshape(X1.shape).cpu().numpy()

    vmax = float(args.vmax)
    levels = np.linspace(0.0, vmax, 25)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axes[0]
    im = ax.contourf(X1, X2, np.minimum(V, vmax), levels=levels, cmap="viridis")
    ax.contour(
        X1, X2, np.minimum(V, vmax), levels=levels[::3], colors="k",
        linewidths=0.35, alpha=0.4,
    )
    for (mx, my), lab in zip(pot.minima(), [r"$A$", r"$B$"]):
        ax.plot(mx, my, "o", color="white", ms=7, mec="k", mew=0.8, zorder=5)
        ax.text(mx, my - 0.22, lab, ha="center", va="top", color="white", fontsize=10)
    sx, sy = pot.saddle()
    ax.plot(sx, sy, "x", color="red", ms=9, mew=1.6, zorder=5)
    ax.text(sx + 0.08, sy + 0.12, "TS", color="red", fontsize=9)
    ax.set_xlabel(r"$x_1$ (slow / barrier)")
    ax.set_ylabel(r"$x_2$ (fast)")
    ax.set_aspect("equal")
    ax.set_title(
        rf"$V=h(x_1^2-1)^2+\frac{{1}}{{2}}k x_2^2$"
        "\n"
        rf"$h={args.h:g},\ k={args.k:g}$  (barrier$={pot.barrier_height():g}$)",
        fontsize=10,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$V$")

    ax = axes[1]
    # 1D slice along x2=0 and a few fixed-x2 cuts
    for x2v, ls in [(0.0, "-"), (0.5, "--"), (1.0, ":")]:
        pts1 = torch.as_tensor(
            np.stack([x1, np.full_like(x1, x2v)], axis=-1), dtype=torch.float64
        )
        with torch.no_grad():
            v1 = pot.energy(pts1).reshape(-1).cpu().numpy()
        ax.plot(x1, v1, ls=ls, lw=1.6, label=rf"$x_2={x2v:g}$")
    ax.axhline(pot.barrier_height(), color="0.5", ls=":", lw=1.0, label=r"barrier $h$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$V(x_1,x_2)$")
    ax.set_xlim(-float(args.x1_lim), float(args.x1_lim))
    ax.set_ylim(0.0, vmax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"1D slices vs $x_1$", fontsize=10)

    fig.tight_layout()
    for stem in ("potential_2d", "doublewell_pes"):
        fig.savefig(out / f"{stem}.png", dpi=200)
        fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {out / 'potential_2d.png'}", flush=True)
    print(
        f"[params] h={args.h:g}  k={args.k:g}  "
        f"minima={pot.minima()}  saddle={pot.saddle()}  ΔV‡={pot.barrier_height():g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
