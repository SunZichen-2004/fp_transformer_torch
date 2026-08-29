#!/usr/bin/env python3
"""Replace WT-MetaD in convergence_1024 with s0/s2 from convergence_320_90535 and replot.

Keeps R=1024 sFPT arrays; overlays MetaD time series, 1D FES, and 2D FES from
the same MetaD runs used in dialanine_main_picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "dialanine_water"
SFPT = EXP / "sFPT"
OUT = SFPT / "convergence_1024"
SRC = SFPT / "convergence_320_90535"

sys.path.insert(0, str(EXP))
from plot_convergence_multi import (  # noqa: E402
    plot_1d_collage,
    plot_2d_metad_mean,
    plot_2d_sfpt_vs_refs,
    plot_series_err,
)

META_KEYS = (
    "t_metad_ps",
    "tw_metad_h",
    "cost_metad_gpuh",
    "rmse_metad",
    "rmse_metad_std",
    "dG_metad",
    "dG_metad_std",
    "F1d_metad_mean",
    "F1d_metad_std",
    "metad_dirs",
)


def _npz_dict(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    z.close()
    return out


def patch_and_save() -> dict:
    dst = _npz_dict(OUT / "fes_rmse_vs_time.npz")
    src = _npz_dict(SRC / "fes_rmse_vs_time.npz")
    for k in META_KEYS:
        dst[k] = src[k]
    np.savez_compressed(OUT / "fes_rmse_vs_time.npz", **dst)

    g1 = _npz_dict(OUT / "fes_1d_all_groups.npz")
    g1["F1d_metad_mean"] = src["F1d_metad_mean"]
    g1["F1d_metad_std"] = src["F1d_metad_std"]
    np.savez_compressed(OUT / "fes_1d_all_groups.npz", **g1)

    f2s = _npz_dict(SRC / "fes_2d_last_frac.npz")
    f2d = _npz_dict(OUT / "fes_2d_last_frac.npz")
    f2d["F_metad"] = f2s["F_metad"]
    np.savez_compressed(OUT / "fes_2d_last_frac.npz", **f2d)

    m2 = _npz_dict(SRC / "fes_2d_metad_mean.npz")
    np.savez_compressed(OUT / "fes_2d_metad_mean.npz", **m2)

    print(
        "[patch] MetaD dirs:\n  "
        + "\n  ".join(str(x) for x in np.asarray(dst["metad_dirs"])),
        flush=True,
    )
    return {"rmse": dst, "g1": g1, "f2": f2d, "m2": m2}


def replot(data: dict) -> None:
    z = data["rmse"]
    g1 = data["g1"]
    f2 = data["f2"]
    m2 = data["m2"]
    t_meta = np.asarray(z["t_metad_ps"], dtype=np.float64)
    t_sfpt = np.asarray(z["t_kde_ps"], dtype=np.float64)
    tw_meta = np.asarray(z["tw_metad_h"], dtype=np.float64)
    tw_sfpt = np.asarray(z["tw_kde_h"], dtype=np.float64)
    cost_meta = np.asarray(z["cost_metad_gpuh"], dtype=np.float64)
    cost_sfpt = np.asarray(z["cost_kde_gpuh"], dtype=np.float64)
    rmse_meta = np.asarray(z["rmse_metad"], dtype=np.float64)
    rmse_meta_std = np.asarray(z["rmse_metad_std"], dtype=np.float64)
    rmse_sfpt = np.asarray(z["rmse_kde"], dtype=np.float64)
    rmse_sfpt_std = np.asarray(z["rmse_kde_std"], dtype=np.float64)
    dG_meta = np.asarray(z["dG_metad"], dtype=np.float64)
    dG_meta_std = np.asarray(z["dG_metad_std"], dtype=np.float64)
    dG_sfpt = np.asarray(z["dG_kde"], dtype=np.float64)
    dG_sfpt_std = np.asarray(z["dG_kde_std"], dtype=np.float64)
    dG_ref = float(z["dG_us_ref"])
    xi_lo, xi_hi = -np.pi, np.pi

    ylab_rmse = (
        rf"1D $F(\phi)$ RMSE vs US ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim, xm, xs, xlab, xlim in [
        ("fes_rmse_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", None),
        (
            "fes_rmse_vs_time_divein",
            (0.0, 3.0),
            t_meta,
            t_sfpt,
            r"$t$ / ps",
            (0.0, 100.0),
        ),
        ("fes_rmse_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            "fes_rmse_vs_wallclock_divein",
            (0.0, 3.0),
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
            (0.0, 1.0),
        ),
        ("fes_rmse_vs_cost", None, cost_meta, cost_sfpt, r"cost / GPU$\cdot$h", None),
        (
            "fes_rmse_vs_cost_divein",
            (0.0, 8.0),
            cost_meta,
            cost_sfpt,
            r"cost / GPU$\cdot$h",
            None,
        ),
    ]:
        plot_series_err(
            OUT,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab_rmse,
            t_meta=xm,
            y_meta=rmse_meta,
            e_meta=rmse_meta_std,
            t_sfpt=xs,
            y_sfpt=rmse_sfpt,
            e_sfpt=rmse_sfpt_std,
            y0_line=True,
            ylim=ylim,
            xlim=xlim,
        )

    ylab_dg = (
        rf"$\Delta G=k_BT\ln(P_{{\phi<0}}/P_{{\phi>0}})$ "
        rf"($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim, xm, xs, xlab, xlim in [
        ("deltaG_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", None),
        (
            "deltaG_vs_time_divein",
            (2.5, 6.5),
            t_meta,
            t_sfpt,
            r"$t$ / ps",
            (0.0, 200.0),
        ),
        ("deltaG_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            "deltaG_vs_wallclock_divein",
            (2.5, 6.5),
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
            None,
        ),
    ]:
        plot_series_err(
            OUT,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab_dg,
            t_meta=xm,
            y_meta=dG_meta,
            e_meta=dG_meta_std,
            t_sfpt=xs,
            y_sfpt=dG_sfpt,
            e_sfpt=dG_sfpt_std,
            dG_ref=dG_ref,
            ref_label="US",
            ylim=ylim,
            xlim=xlim,
        )

    plot_1d_collage(
        OUT,
        xi=g1["xi"],
        f_meta_mean=g1["F1d_metad_mean"],
        f_meta_std=g1["F1d_metad_std"],
        f_us=g1["F1d_us"],
        f_sfpt_mean=g1["F1d_kde_mean"],
        f_sfpt_std=g1["F1d_kde_std"],
        titles=g1["titles"],
        vmax=20.0,
        ncols=4,
        group_size=int(z["group_size"]),
    )
    plot_2d_metad_mean(
        OUT,
        m2["phi"],
        m2["psi"],
        m2["F_mean"],
        m2["F_std"],
        vmax=20.0,
    )
    plot_2d_sfpt_vs_refs(
        OUT,
        phi=f2["phi"],
        psi=f2["psi"],
        F_sfpt=f2["F_sfpt"],
        F_sfpt_std=f2["F_sfpt_std"],
        F_meta=f2["F_metad"],
        F_us_2d=f2["F_us"],
        vmax=20.0,
    )
    print(f"[done] {OUT}", flush=True)


def main() -> None:
    data = patch_and_save()
    replot(data)


if __name__ == "__main__":
    main()
