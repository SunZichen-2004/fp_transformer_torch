#!/usr/bin/env python3
"""Compare KDE FES (from FPT CV+bandwidth) vs WT-MetaD; same layout as MLE compare.

Produces:
  - fes_compare_kde_vs_metad.png/.pdf   (2D side-by-side)
  - fes_compare_1d_d2md1.png/.pdf       (1D ξ = d2−d1)
  - fes_compare_combined.png/.pdf       (1×3 combined)

Default: load existing ``kde_fes_last3.npz`` under the water FPT run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))

from plot_fes_compare import (  # noqa: E402
    ev_to_kcal,
    gaussian_smooth_and_reproject,
    mask_both_gt_and_reproject,
    nature_rcparams,
    plot_1d,
    plot_2d,
    plot_combined_1x3,
    relative_fes,
    save_grids,
    _parse_lim,
)

DEFAULT_KDE = Path(
    "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
    "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736"
    "/kde_fes/kde_fes_last3.npz"
)
DEFAULT_METAD = Path(
    "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
    "/experiments/rearrangement_water_metadynamics/results"
    "/claisen_water_metad_mh-0_r64_86711/live/fes.npz"
)


def interpolate_metad_to_grid(
    metad_path: Path,
    d1: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray:
    """MetaD F [eV] → kcal/mol on (d1, d2) mesh (xy indexing), min-subtracted."""
    from scipy.interpolate import RegularGridInterpolator

    metad = np.load(metad_path)
    d1_m = np.asarray(metad["d1"], dtype=np.float64)
    d2_m = np.asarray(metad["d2"], dtype=np.float64)
    f_m = relative_fes(ev_to_kcal(np.asarray(metad["F"], dtype=np.float64)), None)
    # MetaD live/fes.npz is usually (len(d2), len(d1)) with axes (d2, d1)
    if f_m.shape == (len(d2_m), len(d1_m)):
        values = f_m
    elif f_m.shape == (len(d1_m), len(d2_m)):
        values = f_m.T
    else:
        raise ValueError(
            f"MetaD F shape {f_m.shape} incompatible with d1={d1_m.shape} d2={d2_m.shape}"
        )
    interp = RegularGridInterpolator(
        (d2_m, d1_m), values, bounds_error=False, fill_value=np.nan
    )
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    pts = np.stack([g2.ravel(), g1.ravel()], axis=-1)
    return relative_fes(interp(pts).reshape(g1.shape), None)


def build_compare_data(
    kde_npz: Path,
    metad_fes: Path,
    *,
    n_bins_1d: int,
) -> dict:
    z = np.load(kde_npz)
    d1 = np.asarray(z["d1"], dtype=np.float64)
    d2 = np.asarray(z["d2"], dtype=np.float64)
    f_kde = relative_fes(np.asarray(z["F_kcal"], dtype=np.float64), None)
    temperature = float(np.asarray(z["temperature"])) if "temperature" in z.files else 300.0
    f_meta = interpolate_metad_to_grid(metad_fes, d1, d2)

    # Shared ξ bins from both maps' finite cells
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    xi_all = (g2 - g1).ravel()
    finite = np.isfinite(f_kde.ravel()) & np.isfinite(f_meta.ravel())
    edges = np.linspace(
        float(xi_all[finite].min()),
        float(xi_all[finite].max()),
        int(n_bins_1d) + 1,
    )
    xi_c = 0.5 * (edges[:-1] + edges[1:])

    def _project(f2d: np.ndarray) -> np.ndarray:
        # reuse same edges as MLE compare
        from plot_fes_compare import kT_kcal

        xi_flat = (g2 - g1).ravel()
        f_flat = np.asarray(f2d, dtype=np.float64).ravel()
        kt = kT_kcal(temperature)
        beta = 1.0 / kt
        out = np.full(xi_c.shape, np.nan, dtype=np.float64)
        for i in range(len(xi_c)):
            if i < len(xi_c) - 1:
                sel = (xi_flat >= edges[i]) & (xi_flat < edges[i + 1])
            else:
                sel = (xi_flat >= edges[i]) & (xi_flat <= edges[i + 1])
            sel = sel & np.isfinite(f_flat)
            if not np.any(sel):
                continue
            vals = -beta * f_flat[sel]
            m = float(np.max(vals))
            out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
        finite_o = np.isfinite(out)
        out = out - np.nanmin(out[finite_o])
        return out

    return {
        "d1": d1,
        "d2": d2,
        "F_mle_kcal": f_kde,  # left panel / orange curve slot
        "F_metad_kcal": f_meta,
        "xi_d2_minus_d1": xi_c,
        "F1d_mle_kcal": _project(f_kde),
        "F1d_metad_kcal": _project(f_meta),
        "temperature": temperature,
        "unit": "kcal/mol",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kde-fes", type=Path, default=DEFAULT_KDE)
    p.add_argument("--metad-fes", type=Path, default=DEFAULT_METAD)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as kde npz parent).",
    )
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument("--n-levels", type=int, default=26)
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--xi-lim", type=str, default=None)
    p.add_argument(
        "--mask-both-gt",
        type=str,
        default=None,
        help="Mask d1>A and d2>B then reproject 1D, e.g. '4.5,4.5'.",
    )
    p.add_argument("--gauss-sigma", type=float, default=0.0)
    p.add_argument("--gauss-sigma-1d", type=float, default=None)
    p.add_argument(
        "--d1-lim",
        type=str,
        default=None,
        help="2D x-axis lo,hi (Å), e.g. '1,4.5'.",
    )
    p.add_argument(
        "--d2-lim",
        type=str,
        default=None,
        help="2D y-axis lo,hi (Å), e.g. '1,6.5'.",
    )
    p.add_argument("--method-label", type=str, default="KDE")
    return p


def main() -> None:
    args = build_parser().parse_args()
    plt.rcParams.update(nature_rcparams())
    kde_path = args.kde_fes.resolve()
    if not kde_path.is_file():
        raise FileNotFoundError(
            f"missing {kde_path}; run plot_kde_fes.py first "
            "(e.g. --stages 14,15,16)"
        )
    metad_path = args.metad_fes.resolve()
    out_dir = (args.out_dir or kde_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_compare_data(kde_path, metad_path, n_bins_1d=int(args.n_bins_1d))
    print(
        f"[load] KDE={kde_path.name}  MetaD={metad_path}  "
        f"grid={data['d1'].size}x{data['d2'].size}",
        flush=True,
    )

    if args.mask_both_gt:
        a, b = [
            float(x)
            for x in args.mask_both_gt.replace(";", ",").split(",")
            if x.strip()
        ]
        data = mask_both_gt_and_reproject(
            data, d1_gt=a, d2_gt=b, n_bins_1d=int(args.n_bins_1d)
        )
    if float(args.gauss_sigma) > 0.0:
        data = gaussian_smooth_and_reproject(
            data,
            sigma=float(args.gauss_sigma),
            n_bins_1d=int(args.n_bins_1d),
            sigma_1d=args.gauss_sigma_1d,
        )

    grid_path = out_dir / "fes_grids_kde_vs_metad.npz"
    save_grids(grid_path, data)

    out_2d = out_dir / "fes_compare_kde_vs_metad.png"
    out_1d = out_dir / "fes_compare_1d_d2md1.png"
    out_comb = out_dir / "fes_compare_combined.png"
    xi_lim = _parse_lim(args.xi_lim)
    d1_lim = _parse_lim(args.d1_lim)
    d2_lim = _parse_lim(args.d2_lim)
    label = str(args.method_label)

    plot_2d(
        data["d1"],
        data["d2"],
        data["F_mle_kcal"],
        data["F_metad_kcal"],
        vmax=args.vmax_kcal,
        n_levels=int(args.n_levels),
        out=out_2d,
        method_label=label,
    )
    plot_1d(
        data["xi_d2_minus_d1"],
        data["F1d_mle_kcal"],
        data["F1d_metad_kcal"],
        vmax=args.vmax_kcal,
        out=out_1d,
        xi_lim=xi_lim,
        method_label=label,
    )
    plot_combined_1x3(
        data["d1"],
        data["d2"],
        data["F_mle_kcal"],
        data["F_metad_kcal"],
        data["xi_d2_minus_d1"],
        data["F1d_mle_kcal"],
        data["F1d_metad_kcal"],
        vmax=args.vmax_kcal,
        n_levels=int(args.n_levels),
        xi_lim=xi_lim,
        out=out_comb,
        d1_lim=d1_lim,
        d2_lim=d2_lim,
        method_label=label,
    )
    print(
        f"[stats] 2D KDE max={float(np.nanmax(data['F_mle_kcal'])):.3f}  "
        f"MetaD max={float(np.nanmax(data['F_metad_kcal'])):.3f}  "
        f"1D KDE max={float(np.nanmax(data['F1d_mle_kcal'])):.3f}  "
        f"MetaD max={float(np.nanmax(data['F1d_metad_kcal'])):.3f} kcal/mol",
        flush=True,
    )


if __name__ == "__main__":
    main()
