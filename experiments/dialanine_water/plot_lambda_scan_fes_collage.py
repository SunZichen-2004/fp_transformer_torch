#!/usr/bin/env python3
"""λ-scan 1D F(φ) vs simulation-time collage (≤ max_panels), like convergence_320.

For each λ: two-seed mean±std of stage-group KDE F(φ), overlay US-MBAR.
``group_size`` stages per panel; use only the first ``group_size * max_panels``
stages (default 20×40) so there are at most ``max_panels`` panels.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DIAL = EXP.parent / "dialanine"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(DIAL))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    DIAL / "plot_convergence_analysis.py",
)
_dca = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dca)

from plot_fes_compare import (  # noqa: E402
    COLOR_SFPT,
    FE_LABEL,
    nature_rcparams,
    relative_fes,
)
from plot_g5_1d_grid_and_dg_zoom import COLOR_US  # noqa: E402

LAM_RE = re.compile(r"lam([0-9.]+)_", re.I)


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    if stack.shape[axis] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


def discover_runs(scan_dir: Path) -> dict[float, list[Path]]:
    out: dict[float, list[Path]] = {}
    for d in sorted(scan_dir.glob("dialanine_water_sfpt_lam*_r*_s*_*/")):
        if not d.is_dir():
            continue
        m = LAM_RE.search(d.name)
        if not m:
            continue
        out.setdefault(float(m.group(1)), []).append(d)
    return out


def load_us_f_phi(path: Path, xi_query: np.ndarray) -> np.ndarray:
    z = np.load(path)
    if "xi_phi" in z.files and "F_phi" in z.files:
        xi = np.asarray(z["xi_phi"], dtype=np.float64)
        f = relative_fes(np.asarray(z["F_phi"], dtype=np.float64), None)
    else:
        xi = np.asarray(z["xi"], dtype=np.float64)
        f = relative_fes(np.asarray(z["F_kcal"], dtype=np.float64), None)
    if len(xi) != len(xi_query) or not np.allclose(xi, xi_query):
        f = np.interp(xi_query, xi, f, left=np.nan, right=np.nan)
        f = relative_fes(f, None)
    return f


def n_stages(d: Path, lam: float, atol: float) -> int:
    return len(
        _dca.discover_dihedral_stages(
            d, lambda_target=float(lam), lambda_atol=float(atol)
        )
    )


def plot_phi_collage(
    out: Path,
    *,
    stem: str,
    xi: np.ndarray,
    f_us: np.ndarray,
    f_sfpt_mean: np.ndarray,
    f_sfpt_std: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
    group_size: int,
    lam: float,
) -> None:
    n = len(titles)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    plt.rcParams.update(nature_rcparams())
    fig = plt.figure(figsize=(2.35 * ncols, 1.95 * nrows))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.28,
        hspace=0.45,
        left=0.06,
        right=0.99,
        bottom=0.06,
        top=0.94,
    )
    y_us = relative_fes(f_us, None)
    if vmax > 0:
        y_us = np.minimum(y_us, float(vmax))
    ymax_auto = 0.0
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ym = relative_fes(f_sfpt_mean[ig], None)
        ylo = relative_fes(f_sfpt_mean[ig] - f_sfpt_std[ig], None)
        yhi = relative_fes(f_sfpt_mean[ig] + f_sfpt_std[ig], None)
        if vmax > 0:
            ym = np.minimum(ym, float(vmax))
            ylo = np.minimum(ylo, float(vmax))
            yhi = np.minimum(yhi, float(vmax))
        ax.fill_between(xi, ylo, yhi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=2)
        ax.plot(xi, ym, color=COLOR_SFPT, lw=1.25, label="sFPT mean", zorder=3)
        ax.plot(
            xi,
            y_us,
            color=COLOR_US,
            lw=1.55,
            ls="--",
            label=r"US-MBAR $F(\phi)$",
            zorder=5,
        )
        if vmax > 0:
            ax.set_ylim(0.0, float(vmax) * 1.02)
        else:
            ymax_auto = max(
                ymax_auto, float(np.nanmax(yhi)), float(np.nanmax(y_us))
            )
        ax.set_xlim(float(xi.min()), float(xi.max()))
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n:
            ax.set_xlabel(r"$\phi$ / rad", fontsize=8)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(FE_LABEL, fontsize=8)
        else:
            ax.set_yticklabels([])
        if ig == 0:
            ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    if vmax <= 0 and ymax_auto > 0:
        for ax in fig.axes:
            if ax.has_data():
                ax.set_ylim(0.0, ymax_auto * 1.05)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        rf"1D $F(\phi)$  $\lambda$={lam:g}  sFPT mean±std vs US  "
        rf"({group_size} stages / panel, n={n})",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-dir",
        type=Path,
        default=ROOT / "experiments/dialanine_water/lambda_scan",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water/lambda_scan/convergence_analysis",
    )
    p.add_argument(
        "--us-mbar",
        type=Path,
        default=ROOT
        / "experiments/dialanine_water_us/results"
        / "dialanine_water_us2d_r576_k100_s0_87367/mbar/fes_mbar.npz",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=20,
        help="Stages per panel",
    )
    p.add_argument("--dt-stage", type=float, default=0.1, help="ps per stage")
    p.add_argument(
        "--max-panels",
        type=int,
        default=40,
        help="Number of panels = first (group_size * max_panels) stages",
    )
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--collage-vmax", type=float, default=20.0)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-np.pi)
    p.add_argument("--xi-hi", type=float, default=np.pi)
    args = p.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    us_path = Path(args.us_mbar).resolve()
    if not us_path.is_file():
        raise FileNotFoundError(us_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda → cpu", flush=True)

    phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    xi = np.linspace(-np.pi, np.pi, int(args.n_bins_1d), endpoint=False)
    xi = xi + 0.5 * (xi[1] - xi[0])
    f_us = load_us_f_phi(us_path, xi)

    runs = discover_runs(Path(args.scan_dir))
    if not runs:
        raise RuntimeError(f"no lam runs under {args.scan_dir}")
    print(f"[scan] λs={sorted(runs)}", flush=True)

    max_panels = max(1, int(args.max_panels))
    gsz = max(1, int(args.group_size))
    for lam in sorted(runs):
        dirs = runs[lam]
        n_min = min(n_stages(d, lam, args.lambda_atol) for d in dirs)
        if n_min < gsz:
            print(f"[skip] λ={lam:g}: stages={n_min} < group_size={gsz}", flush=True)
            continue
        n_groups = min(max_panels, n_min // gsz)
        dt_group = float(gsz) * float(args.dt_stage)
        print(
            f"[λ={lam:g}] stages_min={n_min} use_first={n_groups * gsz} "
            f"group_size={gsz} dt_group={dt_group:g} ps  "
            f"n_groups={n_groups}  seeds={len(dirs)}",
            flush=True,
        )

        f_list = []
        titles = None
        t_ref = None
        for d in dirs:
            t_s, _, _, f_s, titles_s = _dca.kde_groups_phi(
                d,
                phi=phi,
                psi=psi,
                f1d_ref=f_us,
                xi=xi,
                group_size=gsz,
                dt_group=dt_group,
                temperature=float(args.temperature),
                vmax=None,
                device=device,
                chunk=int(args.chunk),
                n_bins=int(args.n_bins_1d),
                xi_cut=float(args.xi_cut),
                xi_lo=float(args.xi_lo),
                xi_hi=float(args.xi_hi),
                max_groups=int(n_groups),
                lambda_target=float(lam),
                lambda_atol=float(args.lambda_atol),
            )
            f_list.append(f_s)
            if titles is None:
                titles = titles_s
                t_ref = t_s
            print(f"  [done] {d.name} groups={len(t_s)}", flush=True)

        stack = np.stack(f_list, axis=0)
        f_mean, f_std = mean_std(stack, axis=0)
        tag = f"{lam:g}".replace(".", "p")
        stem = f"fes_1d_all_groups_lam{tag}"
        np.savez_compressed(
            out / f"{stem}.npz",
            xi=xi,
            F_us=f_us,
            F1d_mean=f_mean,
            F1d_std=f_std,
            F1d_runs=stack,
            t_ps=t_ref,
            titles=np.asarray(titles),
            lambda_=np.float64(lam),
            group_size=np.int64(gsz),
            dt_group=np.float64(dt_group),
        )
        plot_phi_collage(
            out,
            stem=stem,
            xi=xi,
            f_us=f_us,
            f_sfpt_mean=f_mean,
            f_sfpt_std=f_std,
            titles=list(titles),
            vmax=float(args.collage_vmax),
            ncols=int(args.ncols),
            group_size=gsz,
            lam=lam,
        )

    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
