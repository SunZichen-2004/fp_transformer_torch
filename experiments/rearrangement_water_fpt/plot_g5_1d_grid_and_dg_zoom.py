#!/usr/bin/env python3
"""For rmse_vs_time_g5: (1) collage of all 1D FES MetaD vs sFPT-KDE;
(2) zoomed ΔG–t with final ΔG±1 band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from plot_fes_compare import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    FE_LABEL,
    ev_to_kcal,
    nature_rcparams,
    relative_fes,
)
from plot_fes_rmse_vs_time import (  # noqa: E402
    delta_g_two_basins,
    discover_stage_files,
    make_1d_edges_from_grid,
    project_1d_fixed_edges,
)
from plot_kde_fes import density_on_grid  # noqa: E402

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
# US MBAR reference (distinct from MetaD blue / sFPT orange)
COLOR_US = "#2ca02c"

DEFAULT_US_MBAR = Path(
    "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
    "/experiments/rearrangement_water_US/results"
    "/claisen_water_us1d_mh-0_r64_g4_86936/mbar/fes_mbar.npz"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736"
            "/kde_fes/rmse_vs_time_g5"
        ),
    )
    p.add_argument(
        "--fpt-dir",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736"
        ),
    )
    p.add_argument(
        "--metad-fes",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_metadynamics/results"
            "/claisen_water_metad_mh-0_r64_86711/live/fes.npz"
        ),
    )
    p.add_argument(
        "--us-mbar",
        type=Path,
        default=DEFAULT_US_MBAR,
        help="1D US MBAR FES (ξ=d1−d2); flipped to d2−d1 for plots",
    )
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=2.5)
    p.add_argument("--bounds", type=str, default="1.0,4.0,1.0,4.0")
    p.add_argument("--grid-n", type=int, default=120)
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--ncols", type=int, default=4)
    return p


def load_metad_1d_ref(
    metad_fes: Path,
    bounds: tuple[float, float, float, float],
    n_bins: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fes = np.load(metad_fes)
    d1_m = np.asarray(fes["d1"], dtype=np.float64)
    d2_m = np.asarray(fes["d2"], dtype=np.float64)
    d1_lo, d1_hi, d2_lo, d2_hi = bounds
    m1 = (d1_m >= d1_lo) & (d1_m <= d1_hi)
    m2 = (d2_m >= d2_lo) & (d2_m <= d2_hi)
    d1_use, d2_use = d1_m[m1], d2_m[m2]
    f_full = relative_fes(ev_to_kcal(np.asarray(fes["F"], dtype=np.float64)), None)
    if f_full.shape == (len(d2_m), len(d1_m)):
        f_ref = f_full[np.ix_(m2, m1)]
    elif f_full.shape == (len(d1_m), len(d2_m)):
        f_ref = f_full.T[np.ix_(m2, m1)]
    else:
        raise ValueError(f"unexpected F shape {f_full.shape}")
    f_ref = relative_fes(f_ref, None)
    edges = make_1d_edges_from_grid(d1_use, d2_use, n_bins)
    xi, f1d = project_1d_fixed_edges(
        d1_use, d2_use, f_ref, edges, temperature=temperature
    )
    return xi, f1d, edges


def load_us_mbar_1d_d2_minus_d1(
    us_mbar: Path,
    xi_query: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load US MBAR F(ξ) with ξ_US=d1−d2 and return F on ξ=d2−d1.

    Basin convention (same as FPT/MetaD plots)::

        ξ = d2 − d1
        basin 1: ξ < 0  (product-like: d2 short / d1 long)
        basin 2: ξ > 0  (reactant-like: d2 long / d1 short)
    """
    z = np.load(Path(us_mbar))
    xi_us = np.asarray(z["xi"], dtype=np.float64)  # d1 − d2
    f_us = np.asarray(z["F_kcal"], dtype=np.float64)
    ok = np.isfinite(xi_us) & np.isfinite(f_us)
    xi_us, f_us = xi_us[ok], f_us[ok]
    # flip: ξ_fpt = −ξ_us ; reverse for increasing order
    xi = -xi_us[::-1]
    f = f_us[::-1].copy()
    f = f - float(np.nanmin(f))
    if xi_query is not None:
        xq = np.asarray(xi_query, dtype=np.float64)
        fq = np.interp(xq, xi, f, left=np.nan, right=np.nan)
        return xq, fq
    return xi, f


def us_delta_g_ref(
    us_mbar: Path,
    *,
    temperature: float = 300.0,
    xi_cut: float = 0.0,
    xi_lo: float | None = None,
    xi_hi: float | None = None,
    xi_query: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    xi, f1d = load_us_mbar_1d_d2_minus_d1(us_mbar, xi_query=xi_query)
    dG = delta_g_two_basins(
        xi,
        f1d,
        temperature=float(temperature),
        xi_cut=float(xi_cut),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )
    return float(dG), xi, relative_fes(f1d, None)


def fpt_wall_hours_from_throughput(
    t_ps: np.ndarray,
    *,
    timestep_ps: float = 0.001,
    rate_it_s: float = 4.20,
) -> np.ndarray:
    """Equal-spaced wall-clock hours from sim time and constant job throughput.

    Prefer this over stage mtimes: NFS/resume gaps make mtime intervals uneven.
    """
    t = np.asarray(t_ps, dtype=np.float64)
    return (t / float(timestep_ps)) / float(rate_it_s) / 3600.0


def plot_deltaG_dive_in(
    npz_path: Path,
    out_dir: Path,
    *,
    dG_ref: float | None = None,
    ref_label: str = "US",
) -> None:
    z = np.load(npz_path)
    t_m = np.asarray(z["t_metad_ps"], dtype=np.float64)
    dG_m = np.asarray(z["dG_metad"], dtype=np.float64)
    t_k = np.asarray(z["t_kde_ps"], dtype=np.float64)
    dG_k = np.asarray(z["dG_kde"], dtype=np.float64)
    if dG_ref is None:
        if "dG_us_ref" in z.files:
            dG_ref = float(np.asarray(z["dG_us_ref"]))
        else:
            dG_ref = float(np.asarray(z["dG_metad_ref"]))
            ref_label = "MetaD"

    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # reference ΔG ± 1 band (US MBAR by default)
    ax.axhspan(
        dG_ref - 1.0,
        dG_ref + 1.0,
        color=COLOR_US,
        alpha=0.18,
        lw=0,
        zorder=0,
        label=rf"final $\Delta G\pm 1$",
    )
    ax.axhline(
        dG_ref,
        color=COLOR_US,
        ls=":",
        lw=1.1,
        alpha=0.95,
        zorder=1,
        label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
    )
    ax.plot(t_m, dG_m, color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
    ax.plot(
        t_k,
        dG_k,
        color=COLOR_SFPT,
        lw=1.4,
        marker="o",
        ms=4.5,
        label="sFPT-KDE",
        zorder=3,
    )
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$\Delta G = k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(-20.0, 0.0)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    out = out_dir / "deltaG_vs_time_divein.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] wrote {out}  ref={ref_label} ΔG={dG_ref:.4f}", flush=True)


def _fpt_wall_hours_from_stages(
    fpt_dir: Path,
    *,
    group_size: int,
    n_groups: int,
    steps_per_stage: int = 500,
    fallback_rate_it_s: float = 4.20,
) -> np.ndarray:
    """Wall-clock hours at the end of each KDE group (mtime of last stage)."""
    import re

    stages = sorted(
        fpt_dir.glob("stage[0-9]*_cv.npy"),
        key=lambda p: int(re.search(r"stage(\d+)", p.name).group(1)),
    )
    if len(stages) < group_size * n_groups:
        # Constant throughput fallback (job log ~4.2 it/s).
        n_steps = np.arange(1, n_groups + 1) * group_size * steps_per_stage
        return n_steps / float(fallback_rate_it_s) / 3600.0
    t0 = stages[0].stat().st_mtime
    out = np.zeros(n_groups, dtype=np.float64)
    for ig in range(n_groups):
        last = stages[(ig + 1) * group_size - 1]
        out[ig] = (last.stat().st_mtime - t0) / 3600.0
    return out


def plot_deltaG_vs_wall_clock(
    npz_path: Path,
    out_dir: Path,
    *,
    fpt_dir: Path,
    metad_rate_it_s: float = 13.74,
    metad_gpus: int = 1,
    fpt_gpus: int = 8,
    ylim: tuple[float, float] | None = None,
    stem: str = "deltaG_vs_wallclock",
    dG_ref: float | None = None,
    ref_label: str = "US",
) -> None:
    """ΔG vs job wall-clock (hours). MetaD from throughput; FPT from stage mtimes."""
    z = np.load(npz_path)
    t_m = np.asarray(z["t_metad_ps"], dtype=np.float64)
    dG_m = np.asarray(z["dG_metad"], dtype=np.float64)
    t_k = np.asarray(z["t_kde_ps"], dtype=np.float64)
    dG_k = np.asarray(z["dG_kde"], dtype=np.float64)
    if dG_ref is None:
        if "dG_us_ref" in z.files:
            dG_ref = float(np.asarray(z["dG_us_ref"]))
        else:
            dG_ref = float(np.asarray(z["dG_metad_ref"]))
            ref_label = "MetaD"
    dt = float(np.asarray(z["timestep_ps"])) if "timestep_ps" in z.files else 0.001
    gsz = int(np.asarray(z["group_size"])) if "group_size" in z.files else 5

    # MetaD: wall_h = n_steps / rate / 3600, n_steps = t_ps / dt
    tw_m = (t_m / dt) / float(metad_rate_it_s) / 3600.0
    tw_k = _fpt_wall_hours_from_stages(fpt_dir, group_size=gsz, n_groups=len(t_k))

    np.savez_compressed(
        out_dir / "deltaG_vs_wallclock.npz",
        t_metad_h=tw_m,
        dG_metad=dG_m,
        t_kde_h=tw_k,
        dG_kde=dG_k,
        dG_metad_ref=float(np.asarray(z["dG_metad_ref"])) if "dG_metad_ref" in z.files else dG_ref,
        dG_us_ref=dG_ref,
        t_metad_ps=t_m,
        t_kde_ps=t_k,
        metad_rate_it_s=metad_rate_it_s,
        metad_gpus=metad_gpus,
        fpt_gpus=fpt_gpus,
    )

    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if ylim is not None:
        ax.axhspan(
            dG_ref - 1.0,
            dG_ref + 1.0,
            color=COLOR_US,
            alpha=0.18,
            lw=0,
            zorder=0,
            label=r"final $\Delta G\pm 1$",
        )
    ax.axhline(
        dG_ref,
        color=COLOR_US,
        ls=":",
        lw=1.1,
        alpha=0.95,
        zorder=1,
        label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
    )
    ax.plot(
        tw_m,
        dG_m,
        color=COLOR_META,
        lw=1.3,
        label=rf"WT-MetaD ({metad_gpus}$\times$GPU)",
        zorder=2,
    )
    ax.plot(
        tw_k,
        dG_k,
        color=COLOR_SFPT,
        lw=1.4,
        marker="o",
        ms=4.5,
        label=rf"sFPT-KDE ({fpt_gpus}$\times$GPU)",
        zorder=3,
    )
    ax.set_xlabel(r"wall-clock time / h")
    ax.set_ylabel(r"$\Delta G = k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vals = np.concatenate([dG_k, dG_m, np.asarray([dG_ref, 0.0])])
        vals = vals[np.isfinite(vals)]
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    out = out_dir / f"{stem}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(
        f"[ok] wrote {out}  ref={ref_label} ΔG={dG_ref:.4f}  "
        f"MetaD wall∈[{tw_m[0]:.3f},{tw_m[-1]:.3f}] h  "
        f"sFPT wall∈[{tw_k[0]:.3f},{tw_k[-1]:.3f}] h",
        flush=True,
    )


def _draw_1d_collage(
    *,
    out_dir: Path,
    xi: np.ndarray,
    f1d_metad: np.ndarray,
    f1d_us: np.ndarray | None,
    f1d_kde: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
) -> None:
    n_groups = int(f1d_kde.shape[0])
    ncols = int(ncols)
    nrows = int(np.ceil(n_groups / ncols))
    plt.rcParams.update(nature_rcparams())
    fig_w = 2.35 * ncols
    fig_h = 1.95 * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
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
    y_meta = np.minimum(relative_fes(f1d_metad, None), float(vmax))
    y_us = None
    if f1d_us is not None:
        y_us = np.minimum(relative_fes(f1d_us, None), float(vmax))
    for ig in range(n_groups):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        y_kde = np.minimum(relative_fes(f1d_kde[ig], None), float(vmax))
        ax.plot(xi, y_meta, color=COLOR_META, lw=1.1, label="WT-MetaD", zorder=2)
        ax.fill_between(xi, 0.0, y_meta, color=COLOR_META, alpha=0.10, lw=0, zorder=1)
        if y_us is not None:
            ax.plot(
                xi,
                y_us,
                color=COLOR_US,
                lw=1.2,
                ls="--",
                label="US-MBAR",
                zorder=3,
            )
        ax.plot(xi, y_kde, color=COLOR_SFPT, lw=1.2, label="sFPT-KDE", zorder=4)
        ax.set_ylim(0.0, float(vmax) * 1.02)
        ax.set_xlim(float(xi.min()), float(xi.max()))
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n_groups:
            ax.set_xlabel(r"$d_2-d_1$ / Å", fontsize=8)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(FE_LABEL, fontsize=8)
        else:
            ax.set_yticklabels([])
        if ig == 0:
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    for j in range(n_groups, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        "1D FES  WT-MetaD / US-MBAR / sFPT-KDE  (5 stages / panel)",
        fontsize=11,
        y=0.995,
    )
    out = out_dir / "fes_1d_all_groups.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] wrote {out}", flush=True)


def plot_1d_collage(
    *,
    fpt_dir: Path,
    metad_fes: Path,
    out_dir: Path,
    group_size: int,
    dt_group: float,
    bounds: tuple[float, float, float, float],
    grid_n: int,
    n_bins: int,
    temperature: float,
    vmax: float,
    device: torch.device,
    chunk: int,
    ncols: int,
    us_mbar: Path | None = None,
) -> None:
    xi, f1d_ref, edges = load_metad_1d_ref(
        metad_fes, bounds, n_bins, temperature
    )
    f1d_us = None
    if us_mbar is not None and Path(us_mbar).is_file():
        _, f1d_us = load_us_mbar_1d_d2_minus_d1(us_mbar, xi_query=xi)
        f1d_us = relative_fes(f1d_us, None)
    rows = discover_stage_files(fpt_dir)
    n_groups = len(rows) // int(group_size)
    if n_groups < 1:
        raise RuntimeError("no complete stage groups")

    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    f1d_groups: list[np.ndarray] = []
    titles: list[str] = []
    for ig in range(n_groups):
        chunk_rows = rows[ig * group_size : (ig + 1) * group_size]
        t_ps = float(ig + 1) * float(dt_group)
        d1 = np.linspace(bounds[0], bounds[1], int(grid_n))
        d2 = np.linspace(bounds[2], bounds[3], int(grid_n))
        g1, g2 = np.meshgrid(d1, d2, indexing="xy")
        grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
        lam = float(chunk_rows[0][3])
        p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
        for _, cv_p, bw_p, _ in chunk_rows:
            p_acc += density_on_grid(
                grid_xy, np.load(cv_p), np.load(bw_p), chunk=chunk, device=device
            )
        p = np.clip(p_acc / float(len(chunk_rows)), 1e-300, None)
        f_kcal = relative_fes(
            (-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None
        )
        _, f1d = project_1d_fixed_edges(
            d1, d2, f_kcal, edges, temperature=temperature
        )
        f1d_groups.append(f1d)
        s0, s1 = chunk_rows[0][0], chunk_rows[-1][0]
        titles.append(f"t={t_ps:g} ps  stage{s0:04d}–{s1:04d}")
        print(f"[1d] {titles[-1]}", flush=True)

    save_kw = dict(
        xi=xi,
        F1d_metad=f1d_ref,
        F1d_kde=np.stack(f1d_groups, axis=0),
        t_ps=np.arange(1, n_groups + 1, dtype=np.float64) * float(dt_group),
        titles=np.asarray(titles),
    )
    if f1d_us is not None:
        save_kw["F1d_us"] = f1d_us
    np.savez_compressed(out_dir / "fes_1d_all_groups.npz", **save_kw)

    _draw_1d_collage(
        out_dir=out_dir,
        xi=xi,
        f1d_metad=f1d_ref,
        f1d_us=f1d_us,
        f1d_kde=np.stack(f1d_groups, axis=0),
        titles=titles,
        vmax=vmax,
        ncols=ncols,
    )


def refresh_1d_collage_metad_only(
    *,
    out_dir: Path,
    metad_fes: Path,
    bounds: tuple[float, float, float, float],
    n_bins: int,
    temperature: float,
    vmax: float,
    ncols: int,
    us_mbar: Path | None = None,
) -> None:
    """Keep cached sFPT 1D curves; replace MetaD/US reference and redraw."""
    cache = out_dir / "fes_1d_all_groups.npz"
    if not cache.is_file():
        raise FileNotFoundError(f"missing {cache}; cannot reuse sFPT 1D curves")
    old = np.load(cache, allow_pickle=True)
    f1d_kde = np.asarray(old["F1d_kde"], dtype=np.float64)
    t_ps = np.asarray(old["t_ps"], dtype=np.float64)
    titles = [str(x) for x in old["titles"].tolist()]
    n_groups = int(f1d_kde.shape[0])
    xi, f1d_ref, _edges = load_metad_1d_ref(metad_fes, bounds, n_bins, temperature)
    if f1d_kde.shape[1] != len(xi):
        raise ValueError(
            f"xi length mismatch: cached F1d_kde {f1d_kde.shape} vs new MetaD {len(xi)}"
        )
    f1d_us = None
    if us_mbar is not None and Path(us_mbar).is_file():
        _, f1d_us = load_us_mbar_1d_d2_minus_d1(us_mbar, xi_query=xi)
        f1d_us = relative_fes(f1d_us, None)
    save_kw = dict(
        xi=xi,
        F1d_metad=f1d_ref,
        F1d_kde=f1d_kde,
        t_ps=t_ps,
        titles=np.asarray(titles),
    )
    if f1d_us is not None:
        save_kw["F1d_us"] = f1d_us
    np.savez_compressed(cache, **save_kw)

    _draw_1d_collage(
        out_dir=out_dir,
        xi=xi,
        f1d_metad=f1d_ref,
        f1d_us=f1d_us,
        f1d_kde=f1d_kde,
        titles=titles,
        vmax=vmax,
        ncols=ncols,
    )
    print(f"[ok] refreshed MetaD/US collage ({n_groups} groups)", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = out_dir / "fes_rmse_vs_time.npz"
    fpt_dir = args.fpt_dir.resolve()
    us_mbar = Path(args.us_mbar).resolve()
    dG_us = None
    f1d_us_on_xi = None
    if us_mbar.is_file() and npz.is_file():
        z0 = np.load(npz)
        xi0 = np.asarray(z0["xi"], dtype=np.float64) if "xi" in z0.files else None
        dG_us, _, f1d_us_on_xi = us_delta_g_ref(
            us_mbar,
            temperature=float(args.temperature),
            xi_cut=float(args.xi_cut),
            xi_query=xi0,
        )
        # persist US reference into the shared npz (keep all prior keys)
        payload = {k: z0[k] for k in z0.files}
        payload["dG_us_ref"] = np.asarray(dG_us, dtype=np.float64)
        if f1d_us_on_xi is not None and xi0 is not None:
            payload["F1d_us_ref"] = np.asarray(f1d_us_on_xi, dtype=np.float64)
        np.savez_compressed(npz, **payload)
        print(f"[us] ΔG_ref={dG_us:.4f} kcal/mol  (ξ=d2−d1, cut={args.xi_cut})", flush=True)
    elif not us_mbar.is_file():
        print(f"[warn] missing US MBAR {us_mbar}", flush=True)

    if npz.is_file():
        plot_deltaG_dive_in(npz, out_dir, dG_ref=dG_us, ref_label="US")
        plot_deltaG_vs_wall_clock(
            npz, out_dir, fpt_dir=fpt_dir, dG_ref=dG_us, ref_label="US"
        )
        plot_deltaG_vs_wall_clock(
            npz,
            out_dir,
            fpt_dir=fpt_dir,
            ylim=(-20.0, 0.0),
            stem="deltaG_vs_wallclock_divein",
            dG_ref=dG_us,
            ref_label="US",
        )
    else:
        print(f"[warn] missing {npz}; skip dive-in / wall-clock", flush=True)

    bounds = tuple(
        float(x) for x in args.bounds.replace(";", ",").split(",") if x.strip()
    )
    if len(bounds) != 4:
        raise ValueError("--bounds must be 4 floats")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    # Prefer fast refresh (reuse KDE) when cache exists
    cache = out_dir / "fes_1d_all_groups.npz"
    if cache.is_file():
        refresh_1d_collage_metad_only(
            out_dir=out_dir,
            metad_fes=args.metad_fes.resolve(),
            bounds=bounds,  # type: ignore[arg-type]
            n_bins=int(args.n_bins_1d),
            temperature=float(args.temperature),
            vmax=float(args.vmax_kcal),
            ncols=int(args.ncols),
            us_mbar=us_mbar if us_mbar.is_file() else None,
        )
    else:
        plot_1d_collage(
            fpt_dir=args.fpt_dir.resolve(),
            metad_fes=args.metad_fes.resolve(),
            out_dir=out_dir,
            group_size=int(args.group_size),
            dt_group=float(args.dt_group),
            bounds=bounds,  # type: ignore[arg-type]
            grid_n=int(args.grid_n),
            n_bins=int(args.n_bins_1d),
            temperature=float(args.temperature),
            vmax=float(args.vmax_kcal),
            device=device,
            chunk=int(args.chunk),
            ncols=int(args.ncols),
            us_mbar=us_mbar if us_mbar.is_file() else None,
        )


if __name__ == "__main__":
    main()
