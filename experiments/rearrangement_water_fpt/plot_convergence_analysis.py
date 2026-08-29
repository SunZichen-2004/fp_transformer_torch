#!/usr/bin/env python3
"""Convergence analysis: 3×sFPT + 2×WT-MetaD (mean±std) + US-MBAR reference.

Mirrors ``rmse_vs_time_g5`` figure set under ``convergence_analysis/``.
sFPT groups truncated to the shortest run; MetaD shown as mean±std of two seeds.
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
    delta_g_from_2d,
    delta_g_two_basins,
    discover_stage_files,
    incremental_metad_metrics,
    make_1d_edges_from_grid,
    project_1d_fixed_edges,
    rmse_1d,
)
from plot_g5_1d_grid_and_dg_zoom import (  # noqa: E402
    COLOR_US,
    fpt_wall_hours_from_throughput,
    load_us_mbar_1d_d2_minus_d1,
    us_delta_g_ref,
)
from plot_kde_fes import density_on_grid  # noqa: E402

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample std (ddof=1) along axis; std=0 if n=1."""
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    n = stack.shape[axis]
    if n <= 1:
        return mean, np.zeros_like(mean)
    std = np.nanstd(stack, axis=axis, ddof=1)
    return mean, std


def load_metad_1d_from_fes(
    fes_path: Path,
    bounds: tuple[float, float, float, float],
    n_bins: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fes = np.load(fes_path)
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
    return xi, f1d, edges, d1_use, d2_use, f_ref


def metrics_from_f1d(
    xi: np.ndarray,
    f1d: np.ndarray,
    f1d_ref: np.ndarray,
    *,
    temperature: float,
    vmax: float | None,
    xi_cut: float,
    xi_lo: float | None,
    xi_hi: float | None,
) -> tuple[float, float]:
    rmse = rmse_1d(
        f1d, f1d_ref, vmax=vmax, xi=xi, xi_lo=xi_lo, xi_hi=xi_hi
    )
    dG = delta_g_two_basins(
        xi,
        f1d,
        temperature=temperature,
        xi_cut=xi_cut,
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )
    return rmse, dG


def kde_1d_groups(
    fpt_dir: Path,
    *,
    edges: np.ndarray,
    f1d_ref: np.ndarray,
    bounds: tuple[float, float, float, float],
    grid_n: int,
    group_size: int,
    dt_group: float,
    temperature: float,
    vmax: float,
    device: torch.device,
    chunk: int,
    max_groups: int | None = None,
    xi_cut: float = 0.0,
    xi_lo: float | None = -2.0,
    xi_hi: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = discover_stage_files(fpt_dir)
    n_groups = len(rows) // int(group_size)
    if max_groups is not None:
        n_groups = min(n_groups, int(max_groups))
    if n_groups < 1:
        raise RuntimeError(f"no complete groups in {fpt_dir}")
    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    xi = 0.5 * (edges[:-1] + edges[1:])
    t_list, r_list, g_list, f_list, titles = [], [], [], [], []
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
        rmse = rmse_1d(
            f1d,
            f1d_ref,
            vmax=vmax,
            xi=xi,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
        )
        dG = delta_g_from_2d(
            d1, d2, f_kcal, temperature=temperature, xi_cut=xi_cut
        )
        t_list.append(t_ps)
        r_list.append(rmse)
        g_list.append(dG)
        f_list.append(f1d)
        s0, s1 = chunk_rows[0][0], chunk_rows[-1][0]
        titles.append(f"t={t_ps:g} ps  stage{s0:04d}–{s1:04d}")
        print(
            f"[kde {fpt_dir.name}] {titles[-1]}  RMSE={r_list[-1]:.4f}  ΔG={g_list[-1]:.4f}",
            flush=True,
        )
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(g_list),
        np.stack(f_list, axis=0),
        titles,
    )


def kde_1d_cumulative(
    fpt_dir: Path,
    *,
    edges: np.ndarray,
    f1d_ref: np.ndarray,
    bounds: tuple[float, float, float, float],
    grid_n: int,
    dt_stage: float,
    discard_frac: float,
    eval_stride: int,
    temperature: float,
    vmax: float,
    device: torch.device,
    chunk: int,
    max_stages: int | None = None,
    xi_cut: float = 0.0,
    xi_lo: float | None = -2.0,
    xi_hi: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Cumulative KDE: at each checkpoint use stages [⌊f·n⌋, n) (discard first f)."""
    rows = discover_stage_files(fpt_dir)
    if max_stages is not None:
        rows = rows[: int(max_stages)]
    n_tot = len(rows)
    if n_tot < 2:
        raise RuntimeError(f"need ≥2 stages in {fpt_dir}, got {n_tot}")
    frac = float(discard_frac)
    if not (0.0 <= frac < 1.0):
        raise ValueError(f"discard_frac must be in [0,1), got {frac}")
    stride = max(1, int(eval_stride))
    ends = list(range(stride, n_tot + 1, stride))
    if not ends or ends[-1] != n_tot:
        ends.append(n_tot)
    # require at least one kept stage after discard
    ends = [n for n in ends if n - int(np.floor(frac * n)) >= 1]
    if not ends:
        raise RuntimeError(f"no valid cumulative windows in {fpt_dir}")

    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    xi = 0.5 * (edges[:-1] + edges[1:])
    d1 = np.linspace(bounds[0], bounds[1], int(grid_n))
    d2 = np.linspace(bounds[2], bounds[3], int(grid_n))
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    lam0 = float(rows[0][3])

    # prefix[k] = sum of densities for first k stages
    prefix: list[np.ndarray] = [
        np.zeros(grid_xy.shape[0], dtype=np.float64)
    ]
    end_set = set(ends)
    t_list, r_list, g_list, f_list, titles = [], [], [], [], []
    for k, (_, cv_p, bw_p, lam) in enumerate(rows, start=1):
        if abs(float(lam) - lam0) > 1e-12:
            raise ValueError(f"mixed λ in {fpt_dir}: {lam0} vs {lam}")
        prefix.append(
            prefix[-1]
            + density_on_grid(
                grid_xy, np.load(cv_p), np.load(bw_p), chunk=chunk, device=device
            )
        )
        if k not in end_set:
            continue
        i0 = int(np.floor(frac * k))
        n_keep = k - i0
        p = np.clip((prefix[k] - prefix[i0]) / float(n_keep), 1e-300, None)
        f_kcal = relative_fes(
            (-np.log(p) / (beta0 * lam0) * EV_TO_KCAL).reshape(g1.shape), None
        )
        _, f1d = project_1d_fixed_edges(
            d1, d2, f_kcal, edges, temperature=temperature
        )
        rmse = rmse_1d(
            f1d,
            f1d_ref,
            vmax=vmax,
            xi=xi,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
        )
        dG = delta_g_from_2d(
            d1, d2, f_kcal, temperature=temperature, xi_cut=xi_cut
        )
        t_ps = float(k) * float(dt_stage)
        s0, s1 = rows[i0][0], rows[k - 1][0]
        title = f"t={t_ps:g} ps  stage{s0:04d}–{s1:04d} (cum discard {frac:.0%})"
        t_list.append(t_ps)
        r_list.append(rmse)
        g_list.append(dG)
        f_list.append(f1d)
        titles.append(title)
        print(
            f"[kde-cum {fpt_dir.name}] {title}  n={n_keep}/{k}  "
            f"RMSE={rmse:.4f}  ΔG={dG:.4f}",
            flush=True,
        )
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(g_list),
        np.stack(f_list, axis=0),
        titles,
    )


def plot_series_mean_err(
    out: Path,
    *,
    stem: str,
    xlabel: str,
    ylabel: str,
    t_meta: np.ndarray,
    y_meta: np.ndarray,
    e_meta: np.ndarray | None,
    t_sfpt: np.ndarray,
    y_sfpt: np.ndarray,
    e_sfpt: np.ndarray,
    dG_ref: float | None = None,
    ref_label: str = "US",
    ylim: tuple[float, float] | None = None,
    y0_line: bool = False,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if dG_ref is not None and ylim is not None:
        ax.axhspan(
            dG_ref - 1.0,
            dG_ref + 1.0,
            color=COLOR_US,
            alpha=0.18,
            lw=0,
            zorder=0,
            label=r"final $\Delta G\pm 1$",
        )
    if dG_ref is not None:
        ax.axhline(
            dG_ref,
            color=COLOR_US,
            ls=":",
            lw=1.1,
            alpha=0.95,
            zorder=1,
            label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
        )
    if y0_line:
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    if e_meta is not None:
        ax.plot(t_meta, y_meta, color=COLOR_META, lw=1.3, label="WT-MetaD mean", zorder=2)
        ax.fill_between(
            t_meta,
            y_meta - e_meta,
            y_meta + e_meta,
            color=COLOR_META,
            alpha=0.22,
            lw=0,
            zorder=1,
            label=r"MetaD $\pm$std",
        )
    else:
        ax.plot(t_meta, y_meta, color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
    ax.plot(t_sfpt, y_sfpt, color=COLOR_SFPT, lw=1.5, label="sFPT mean", zorder=4)
    ax.fill_between(
        t_sfpt,
        y_sfpt - e_sfpt,
        y_sfpt + e_sfpt,
        color=COLOR_SFPT,
        alpha=0.25,
        lw=0,
        zorder=3,
        label=r"sFPT $\pm$std",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        parts = [y_meta, y_sfpt]
        if e_meta is not None:
            parts += [y_meta - e_meta, y_meta + e_meta]
        parts += [y_sfpt - e_sfpt, y_sfpt + e_sfpt]
        if dG_ref is not None:
            parts.append(np.asarray([dG_ref, 0.0]))
        vals = np.concatenate([np.ravel(p) for p in parts])
        vals = vals[np.isfinite(vals)]
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(ymax - ymin, 1.0)
        if y0_line:
            ax.set_ylim(bottom=-0.02 * max(ymax, 1.0), top=ymax * 1.08 if ymax > 0 else 1.0)
        else:
            ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def plot_rmse_suite(
    out: Path,
    *,
    stem_prefix: str,
    ylab: str,
    t_meta: np.ndarray,
    tw_meta: np.ndarray,
    cost_meta: np.ndarray,
    rmse_meta: np.ndarray,
    rmse_meta_std: np.ndarray,
    t_sfpt: np.ndarray,
    tw_sfpt: np.ndarray,
    cost_sfpt: np.ndarray,
    rmse_sfpt: np.ndarray,
    rmse_sfpt_std: np.ndarray,
    fpt_gpus: int,
    metad_gpus: int,
    divein_ylim: tuple[float, float] = (0.0, 8.0),
) -> None:
    """Write RMSE vs time / wall-clock / cost (+ dive-in)."""
    for stem, ylim, xm, xs, xlab in [
        (f"{stem_prefix}_vs_time", None, t_meta, t_sfpt, r"$t$ / ps"),
        (f"{stem_prefix}_vs_time_divein", divein_ylim, t_meta, t_sfpt, r"$t$ / ps"),
        (
            f"{stem_prefix}_vs_wallclock",
            None,
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
        ),
        (
            f"{stem_prefix}_vs_wallclock_divein",
            divein_ylim,
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
        ),
        (
            f"{stem_prefix}_vs_cost",
            None,
            cost_meta,
            cost_sfpt,
            rf"cost / GPU$\cdot$h  (sFPT $\times${fpt_gpus}, MetaD $\times${metad_gpus})",
        ),
        (
            f"{stem_prefix}_vs_cost_divein",
            divein_ylim,
            cost_meta,
            cost_sfpt,
            rf"cost / GPU$\cdot$h  (sFPT $\times${fpt_gpus}, MetaD $\times${metad_gpus})",
        ),
    ]:
        plot_series_mean_err(
            out,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab,
            t_meta=xm,
            y_meta=rmse_meta,
            e_meta=rmse_meta_std,
            t_sfpt=xs,
            y_sfpt=rmse_sfpt,
            e_sfpt=rmse_sfpt_std,
            y0_line=True,
            ylim=ylim,
        )


def plot_1d_collage(
    out: Path,
    *,
    xi: np.ndarray,
    f_meta_mean: np.ndarray,
    f_meta_lo: np.ndarray,
    f_meta_hi: np.ndarray,
    f_us: np.ndarray | None,
    f_sfpt_mean: np.ndarray,
    f_sfpt_lo: np.ndarray,
    f_sfpt_hi: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
) -> None:
    n = len(titles)
    ncols = int(ncols)
    nrows = int(np.ceil(n / ncols))
    plt.rcParams.update(nature_rcparams())
    fig = plt.figure(figsize=(2.35 * ncols, 1.95 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, wspace=0.28, hspace=0.45,
        left=0.06, right=0.99, bottom=0.06, top=0.94,
    )
    f_meta_mean = np.asarray(f_meta_mean, dtype=np.float64)
    f_meta_lo = np.asarray(f_meta_lo, dtype=np.float64)
    f_meta_hi = np.asarray(f_meta_hi, dtype=np.float64)
    meta_series = f_meta_mean.ndim == 2
    if not meta_series:
        y_meta = np.minimum(relative_fes(f_meta_mean, None), float(vmax))
        y_mlo = np.minimum(relative_fes(f_meta_lo, None), float(vmax))
        y_mhi = np.minimum(relative_fes(f_meta_hi, None), float(vmax))
    y_us = (
        np.minimum(relative_fes(f_us, None), float(vmax)) if f_us is not None else None
    )
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ym = np.minimum(relative_fes(f_sfpt_mean[ig], None), float(vmax))
        ylo = np.minimum(relative_fes(f_sfpt_lo[ig], None), float(vmax))
        yhi = np.minimum(relative_fes(f_sfpt_hi[ig], None), float(vmax))
        if meta_series:
            y_meta = np.minimum(relative_fes(f_meta_mean[ig], None), float(vmax))
            y_mlo = np.minimum(relative_fes(f_meta_lo[ig], None), float(vmax))
            y_mhi = np.minimum(relative_fes(f_meta_hi[ig], None), float(vmax))
        ax.fill_between(xi, y_mlo, y_mhi, color=COLOR_META, alpha=0.20, lw=0, zorder=1)
        ax.plot(xi, y_meta, color=COLOR_META, lw=1.1, label="WT-MetaD", zorder=2)
        if y_us is not None:
            ax.plot(xi, y_us, color=COLOR_US, lw=1.15, ls="--", label="US-MBAR", zorder=3)
        ax.fill_between(xi, ylo, yhi, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=3)
        ax.plot(xi, ym, color=COLOR_SFPT, lw=1.3, label="sFPT mean", zorder=5)
        ax.set_ylim(0.0, float(vmax) * 1.02)
        ax.set_xlim(float(xi.min()), float(xi.max()))
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n:
            ax.set_xlabel(r"$d_2-d_1$ / Å", fontsize=8)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(FE_LABEL, fontsize=8)
        else:
            ax.set_yticklabels([])
        if ig == 0:
            ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        "1D FES  WT-MetaD (same t) / US / sFPT mean±std"
        if meta_series
        else "1D FES  MetaD mean±std / US / sFPT mean±std  (5 stages / panel)",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print("[ok] fes_1d_all_groups", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "experiments/rearrangement_water_fpt/convergence_analysis",
    )
    p.add_argument(
        "--fpt-dirs",
        type=str,
        default=",".join(
            [
                str(
                    ROOT
                    / "experiments/rearrangement_water_fpt/results"
                    / "claisen_water_fpt_mh-0_g8_86736"
                ),
                str(
                    ROOT
                    / "experiments/rearrangement_water_fpt/results"
                    / "claisen_water_fpt_mh-0_g8_86938"
                ),
                str(
                    ROOT
                    / "experiments/rearrangement_water_fpt/results"
                    / "claisen_water_fpt_mh-0_g8_87055"
                ),
            ]
        ),
    )
    p.add_argument(
        "--metad-dirs",
        type=str,
        default=",".join(
            [
                str(
                    ROOT
                    / "experiments/rearrangement_water_metadynamics/results"
                    / "claisen_water_metad_mh-0_r64_87059/live"
                ),
                str(
                    ROOT
                    / "experiments/rearrangement_water_metadynamics/results"
                    / "claisen_water_metad_mh-0_r64_87060/live"
                ),
            ]
        ),
    )
    p.add_argument(
        "--us-mbar",
        type=Path,
        default=ROOT
        / "experiments/rearrangement_water_US/results"
        / "claisen_water_us1d_mh-0_r64_g1_87040/mbar/fes_mbar.npz",
    )
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=2.5)
    p.add_argument(
        "--sfpt-mode",
        type=str,
        choices=("grouped", "cumulative"),
        default="grouped",
        help="grouped: non-overlapping stage blocks; cumulative: growing window",
    )
    p.add_argument(
        "--discard-frac",
        type=float,
        default=0.2,
        help="For cumulative mode: discard first this fraction of stages at each checkpoint",
    )
    p.add_argument(
        "--dt-stage",
        type=float,
        default=0.5,
        help="ps per FPT stage (cumulative time axis)",
    )
    p.add_argument(
        "--eval-stride",
        type=int,
        default=5,
        help="For cumulative mode: evaluate every N stages (and at the end)",
    )
    p.add_argument("--bounds", type=str, default="1.0,8.5,1.0,8.5")
    p.add_argument("--grid-n", type=int, default=120)
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument(
        "--rmse-vmax-extra",
        type=float,
        default=30.0,
        help="Also plot RMSE suite restricted to F_ref ≤ this (kcal/mol); 0 to disable",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--n-replicas", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=30.0)
    p.add_argument("--sigma", type=str, default="0.2,0.2")
    p.add_argument("--metad-stride-deposits", type=int, default=10)
    p.add_argument("--metad-rate", type=float, default=13.7)
    p.add_argument(
        "--fpt-rate",
        type=float,
        default=4.20,
        help="sFPT wall-clock throughput (it/s); used for equal-spaced wall hours",
    )
    p.add_argument("--fpt-gpus", type=int, default=8, help="sFPT GPUs per job (cost)")
    p.add_argument("--metad-gpus", type=int, default=1, help="MetaD GPUs per job (cost)")
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-2.0)
    p.add_argument("--xi-hi", type=float, default=2.0)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument(
        "--reuse-kde-npz",
        type=Path,
        default=None,
        help="Reuse F1d_kde_runs from prior fes_1d_all_groups.npz (skip GPU KDE)",
    )
    args = p.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fpt_dirs = [Path(x.strip()) for x in args.fpt_dirs.split(",") if x.strip()]
    metad_dirs = [Path(x.strip()) for x in args.metad_dirs.split(",") if x.strip()]
    bounds = tuple(float(x) for x in args.bounds.replace(";", ",").split(",") if x.strip())
    assert len(bounds) == 4
    sigma = tuple(float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda → cpu", flush=True)
    xi_lo = float(args.xi_lo)
    xi_hi = float(args.xi_hi)
    xi_cut = float(args.xi_cut)
    print(
        f"[window] RMSE on ξ∈[{xi_lo:g},{xi_hi:g}]; "
        f"ΔG from full 2D FES split at d2−d1={xi_cut:g} (no 2D crop, no ξ window); "
        f"RMSE ref=US",
        flush=True,
    )

    # shortest sFPT length
    n_stages_each = [len(discover_stage_files(d)) for d in fpt_dirs]
    n_stages = int(min(n_stages_each))
    if args.sfpt_mode == "grouped":
        n_groups_each = [n // int(args.group_size) for n in n_stages_each]
        n_groups = int(min(n_groups_each))
        print(
            f"[sFPT grouped] stages={n_stages_each} → groups={n_groups_each} "
            f"→ use {n_groups} ({n_groups * args.dt_group:g} ps)",
            flush=True,
        )
    else:
        n_groups = None  # unused
        print(
            f"[sFPT cumulative] stages={n_stages_each} → use {n_stages} "
            f"({n_stages * args.dt_stage:g} ps), discard={args.discard_frac:.0%}, "
            f"eval_stride={args.eval_stride}",
            flush=True,
        )

    # 2D grid for KDE / MetaD hills / ΔG: full (d1,d2) box, not the old [1,4]² crop.
    d1_use = np.linspace(bounds[0], bounds[1], int(args.grid_n))
    d2_use = np.linspace(bounds[2], bounds[3], int(args.grid_n))
    edges = make_1d_edges_from_grid(d1_use, d2_use, int(args.n_bins_1d))
    xi = 0.5 * (edges[:-1] + edges[1:])
    print(
        f"[2d] KDE/hills grid d1∈[{bounds[0]:g},{bounds[1]:g}] "
        f"d2∈[{bounds[2]:g},{bounds[3]:g}] n={args.grid_n}",
        flush=True,
    )

    # MetaD 1D (collage); ΔG from dumped 2D FES with no extra crop
    f1d_meta_final = []
    dG_meta_runs = []
    for md in metad_dirs:
        xi_i, f1d_i, _, d1_i, d2_i, f2_i = load_metad_1d_from_fes(
            md / "fes.npz", bounds, int(args.n_bins_1d), float(args.temperature)
        )
        f1d_i = np.interp(xi, xi_i, f1d_i, left=np.nan, right=np.nan)
        finite = np.isfinite(f1d_i)
        if np.any(finite):
            f1d_i = f1d_i - np.nanmin(f1d_i[finite])
        f1d_meta_final.append(f1d_i)
        dG_i = delta_g_from_2d(
            d1_i, d2_i, f2_i, temperature=float(args.temperature), xi_cut=xi_cut
        )
        dG_meta_runs.append(dG_i)
        print(
            f"[metad] {md.parent.name} Fmax={np.nanmax(f1d_i):.2f}  "
            f"ΔG_2D={dG_i:.4f}",
            flush=True,
        )
    f_meta_stack = np.stack(f1d_meta_final, axis=0)
    f1d_meta, f1d_meta_std = mean_std(f_meta_stack, axis=0)
    f1d_meta = relative_fes(f1d_meta, None)
    dG_meta = float(np.mean(dG_meta_runs))
    print(f"[metad] mean ΔG (full 2D)={dG_meta:.4f}", flush=True)

    # US: ΔG on the full 1D MBAR F(ξ); RMSE reference interpolated onto ξ
    us_path = Path(args.us_mbar)
    if not us_path.is_file():
        raise FileNotFoundError(f"missing US MBAR {us_path}; run mbar_fes.py first")
    dG_us, _, _ = us_delta_g_ref(
        us_path,
        temperature=float(args.temperature),
        xi_cut=xi_cut,
        xi_lo=None,
        xi_hi=None,
        xi_query=None,
    )
    _, f_us = load_us_mbar_1d_d2_minus_d1(us_path, xi_query=xi)
    f1d_ref = relative_fes(f_us, None)
    print(f"[us] ΔG (full 1D MBAR, no ξ window)={dG_us:.4f}", flush=True)

    # MetaD time series vs US
    t_m_list, rmse_m_list, dG_m_list = [], [], []
    for md in metad_dirs:
        t_m, rmse_m, dG_m = incremental_metad_metrics(
            md / "hills.npz",
            d1=d1_use,
            d2=d2_use,
            f1d_ref=f1d_ref,
            edges=edges,
            xi_centers=xi,
            n_replicas=int(args.n_replicas),
            deposit_pace=int(args.deposit_pace),
            timestep=float(args.timestep),
            bias_factor=float(args.bias_factor),
            sigma=sigma,
            temperature=float(args.temperature),
            vmax_kcal=float(args.vmax_kcal),
            stride_deposits=int(args.metad_stride_deposits),
            xi_cut=xi_cut,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
        )
        t_m_list.append(t_m)
        rmse_m_list.append(rmse_m)
        dG_m_list.append(dG_m)
        print(f"[metad] {md.parent.name} t→{t_m[-1]:.1f} ps n={len(t_m)}", flush=True)

    n_m = min(len(t) for t in t_m_list)
    t_meta = t_m_list[0][:n_m]
    rmse_meta_mean, rmse_meta_std = mean_std(np.stack([r[:n_m] for r in rmse_m_list]))
    dG_meta_mean, dG_meta_std = mean_std(np.stack([g[:n_m] for g in dG_m_list]))

    # sFPT: reuse KDE FES if available, else recompute
    reuse = args.reuse_kde_npz
    if reuse is None and args.sfpt_mode == "grouped":
        cand = out / "fes_1d_all_groups.npz"
        if cand.is_file():
            reuse = cand
    titles = None
    if reuse is not None and Path(reuse).is_file():
        z = np.load(reuse, allow_pickle=True)
        xi_z = np.asarray(z["xi"], dtype=np.float64)
        if not np.allclose(xi_z, xi):
            raise RuntimeError("reuse npz xi grid mismatch; recompute KDE")
        f_sfpt_stack = np.asarray(z["F1d_kde_runs"], dtype=np.float64)
        if f_sfpt_stack.shape[0] != len(fpt_dirs):
            raise RuntimeError(
                f"reuse runs={f_sfpt_stack.shape[0]} != n_fpt={len(fpt_dirs)}"
            )
        n_eval = f_sfpt_stack.shape[1]
        if args.sfpt_mode == "grouped" and n_groups is not None:
            n_eval = min(int(n_groups), n_eval)
        f_sfpt_stack = f_sfpt_stack[:, :n_eval, :]
        t_sfpt = np.asarray(z["t_ps"], dtype=np.float64)[:n_eval]
        titles = [str(x) for x in np.asarray(z["titles"])[:n_eval]]
        r_s_list, g_s_list = [], []
        for ir in range(f_sfpt_stack.shape[0]):
            rr, gg = [], []
            for ig in range(n_eval):
                rmse, dG = metrics_from_f1d(
                    xi,
                    f_sfpt_stack[ir, ig],
                    f1d_ref,
                    temperature=float(args.temperature),
                    vmax=float(args.vmax_kcal),
                    xi_cut=xi_cut,
                    xi_lo=xi_lo,
                    xi_hi=xi_hi,
                )
                rr.append(rmse)
                gg.append(dG)
            r_s_list.append(rr)
            g_s_list.append(gg)
            print(
                f"[reuse {fpt_dirs[ir].name}] last RMSE={rr[-1]:.4f} ΔG={gg[-1]:.4f}",
                flush=True,
            )
        print(f"[sFPT] reused KDE FES from {reuse}  n_eval={n_eval}", flush=True)
    else:
        t_s_list, r_s_list, g_s_list, f_s_list = [], [], [], []
        for d in fpt_dirs:
            if args.sfpt_mode == "cumulative":
                t_s, r_s, g_s, f_s, titles_i = kde_1d_cumulative(
                    d,
                    edges=edges,
                    f1d_ref=f1d_ref,
                    bounds=bounds,  # type: ignore[arg-type]
                    grid_n=int(args.grid_n),
                    dt_stage=float(args.dt_stage),
                    discard_frac=float(args.discard_frac),
                    eval_stride=int(args.eval_stride),
                    temperature=float(args.temperature),
                    vmax=float(args.vmax_kcal),
                    device=device,
                    chunk=int(args.chunk),
                    max_stages=n_stages,
                    xi_cut=xi_cut,
                    xi_lo=xi_lo,
                    xi_hi=xi_hi,
                )
            else:
                t_s, r_s, g_s, f_s, titles_i = kde_1d_groups(
                    d,
                    edges=edges,
                    f1d_ref=f1d_ref,
                    bounds=bounds,  # type: ignore[arg-type]
                    grid_n=int(args.grid_n),
                    group_size=int(args.group_size),
                    dt_group=float(args.dt_group),
                    temperature=float(args.temperature),
                    vmax=float(args.vmax_kcal),
                    device=device,
                    chunk=int(args.chunk),
                    max_groups=n_groups,
                    xi_cut=xi_cut,
                    xi_lo=xi_lo,
                    xi_hi=xi_hi,
                )
            t_s_list.append(t_s)
            r_s_list.append(r_s)
            g_s_list.append(g_s)
            f_s_list.append(f_s)
            if titles is None:
                titles = titles_i
        # align checkpoint count (cumulative ends are identical if max_stages shared)
        n_eval = min(len(t) for t in t_s_list)
        t_sfpt = t_s_list[0][:n_eval]
        r_s_list = [r[:n_eval] for r in r_s_list]
        g_s_list = [g[:n_eval] for g in g_s_list]
        f_sfpt_stack = np.stack([f[:n_eval] for f in f_s_list], axis=0)
        titles = titles[:n_eval]

    n_groups = len(t_sfpt)  # checkpoint count for logging / F30 suite

    rmse_sfpt_mean, rmse_sfpt_std = mean_std(np.stack(r_s_list))
    dG_sfpt_mean, dG_sfpt_std = mean_std(np.stack(g_s_list))
    f_sfpt_mean, f_sfpt_std = mean_std(f_sfpt_stack, axis=0)

    # wall-clock: constant throughput → equal spacing (avoid mtime gaps)
    tw_sfpt = fpt_wall_hours_from_throughput(
        t_sfpt,
        timestep_ps=float(args.timestep),
        rate_it_s=float(args.fpt_rate),
    )
    tw_meta = (t_meta / float(args.timestep)) / float(args.metad_rate) / 3600.0
    cost_sfpt = tw_sfpt * float(args.fpt_gpus)
    cost_meta = tw_meta * float(args.metad_gpus)
    print(
        f"[wall] sFPT Δt≈{float(np.mean(np.diff(tw_sfpt))):.4f} h/group  "
        f"rate={args.fpt_rate:g} it/s ×{args.fpt_gpus} GPU; "
        f"MetaD rate={args.metad_rate:g} it/s ×{args.metad_gpus} GPU",
        flush=True,
    )

    np.savez_compressed(
        out / "fes_rmse_vs_time.npz",
        t_metad_ps=t_meta,
        rmse_metad=rmse_meta_mean,
        rmse_metad_std=rmse_meta_std,
        dG_metad=dG_meta_mean,
        dG_metad_std=dG_meta_std,
        dG_metad_ref=dG_meta,
        dG_us_ref=dG_us,
        xi=xi,
        F1d_metad_ref=f1d_meta,
        F1d_us_ref=f1d_ref,
        t_kde_ps=t_sfpt,
        rmse_kde=rmse_sfpt_mean,
        rmse_kde_std=rmse_sfpt_std,
        dG_kde=dG_sfpt_mean,
        dG_kde_std=dG_sfpt_std,
        tw_kde_h=tw_sfpt,
        tw_metad_h=tw_meta,
        cost_kde_gpuh=cost_sfpt,
        cost_metad_gpuh=cost_meta,
        fpt_gpus=int(args.fpt_gpus),
        metad_gpus=int(args.metad_gpus),
        n_groups=n_groups,
        group_size=int(args.group_size),
        dt_group_ps=float(args.dt_group),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
        xi_cut=xi_cut,
        fpt_rate_it_s=float(args.fpt_rate),
        metad_rate_it_s=float(args.metad_rate),
        timestep_ps=float(args.timestep),
        fpt_dirs=np.asarray([str(d) for d in fpt_dirs]),
        metad_dirs=np.asarray([str(d) for d in metad_dirs]),
        sfpt_mode=str(args.sfpt_mode),
        discard_frac=float(args.discard_frac),
        dt_stage_ps=float(args.dt_stage),
        eval_stride=int(args.eval_stride),
    )
    np.savez_compressed(
        out / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad=f1d_meta,
        F1d_metad_std=f1d_meta_std,
        F1d_us=f1d_ref,
        F1d_kde=f_sfpt_mean,
        F1d_kde_std=f_sfpt_std,
        F1d_kde_runs=f_sfpt_stack,
        t_ps=t_sfpt,
        titles=np.asarray(titles),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
        sfpt_mode=str(args.sfpt_mode),
        discard_frac=float(args.discard_frac),
        dt_stage_ps=float(args.dt_stage),
        eval_stride=int(args.eval_stride),
    )

    ylab_rmse = rf"1D FES RMSE ($\xi\in[{xi_lo:g},{xi_hi:g}]$) / kcal mol$^{{-1}}$"
    plot_rmse_suite(
        out,
        stem_prefix="fes_rmse",
        ylab=ylab_rmse,
        t_meta=t_meta,
        tw_meta=tw_meta,
        cost_meta=cost_meta,
        rmse_meta=rmse_meta_mean,
        rmse_meta_std=rmse_meta_std,
        t_sfpt=t_sfpt,
        tw_sfpt=tw_sfpt,
        cost_sfpt=cost_sfpt,
        rmse_sfpt=rmse_sfpt_mean,
        rmse_sfpt_std=rmse_sfpt_std,
        fpt_gpus=int(args.fpt_gpus),
        metad_gpus=int(args.metad_gpus),
    )

    # Extra RMSE suite: only bins with F_US ≤ rmse_vmax_extra (still in ξ window)
    vmax_extra = float(args.rmse_vmax_extra)
    if vmax_extra > 0:
        print(f"[rmse] extra suite with F_ref ≤ {vmax_extra:g} kcal/mol", flush=True)
        r30_m_list, r30_s_list = [], []
        for md in metad_dirs:
            t_m30, rmse_m30, _ = incremental_metad_metrics(
                md / "hills.npz",
                d1=d1_use,
                d2=d2_use,
                f1d_ref=f1d_ref,
                edges=edges,
                xi_centers=xi,
                n_replicas=int(args.n_replicas),
                deposit_pace=int(args.deposit_pace),
                timestep=float(args.timestep),
                bias_factor=float(args.bias_factor),
                sigma=sigma,
                temperature=float(args.temperature),
                vmax_kcal=vmax_extra,
                stride_deposits=int(args.metad_stride_deposits),
                xi_cut=xi_cut,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
            )
            r30_m_list.append(rmse_m30[:n_m])
            print(f"[metad F≤{vmax_extra:g}] {md.parent.name} last RMSE={rmse_m30[n_m-1]:.4f}", flush=True)
        rmse30_meta_mean, rmse30_meta_std = mean_std(np.stack(r30_m_list))
        for ir in range(f_sfpt_stack.shape[0]):
            rr = [
                metrics_from_f1d(
                    xi,
                    f_sfpt_stack[ir, ig],
                    f1d_ref,
                    temperature=float(args.temperature),
                    vmax=vmax_extra,
                    xi_cut=xi_cut,
                    xi_lo=xi_lo,
                    xi_hi=xi_hi,
                )[0]
                for ig in range(n_groups)
            ]
            r30_s_list.append(rr)
            print(
                f"[sFPT F≤{vmax_extra:g}] {fpt_dirs[ir].name} last RMSE={rr[-1]:.4f}",
                flush=True,
            )
        rmse30_sfpt_mean, rmse30_sfpt_std = mean_std(np.stack(r30_s_list))
        np.savez_compressed(
            out / f"fes_rmse_F{int(vmax_extra)}.npz",
            t_metad_ps=t_meta,
            t_kde_ps=t_sfpt,
            tw_metad_h=tw_meta,
            tw_kde_h=tw_sfpt,
            cost_metad_gpuh=cost_meta,
            cost_kde_gpuh=cost_sfpt,
            rmse_metad=rmse30_meta_mean,
            rmse_metad_std=rmse30_meta_std,
            rmse_kde=rmse30_sfpt_mean,
            rmse_kde_std=rmse30_sfpt_std,
            vmax_kcal=vmax_extra,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
        )
        plot_rmse_suite(
            out,
            stem_prefix=f"fes_rmse_F{int(vmax_extra)}",
            ylab=(
                rf"1D FES RMSE ($\xi\in[{xi_lo:g},{xi_hi:g}]$, "
                rf"$F_{{\mathrm{{US}}}}\leq {vmax_extra:g}$) / kcal mol$^{{-1}}$"
            ),
            t_meta=t_meta,
            tw_meta=tw_meta,
            cost_meta=cost_meta,
            rmse_meta=rmse30_meta_mean,
            rmse_meta_std=rmse30_meta_std,
            t_sfpt=t_sfpt,
            tw_sfpt=tw_sfpt,
            cost_sfpt=cost_sfpt,
            rmse_sfpt=rmse30_sfpt_mean,
            rmse_sfpt_std=rmse30_sfpt_std,
            fpt_gpus=int(args.fpt_gpus),
            metad_gpus=int(args.metad_gpus),
        )

    for stem, ylim in [
        ("deltaG_vs_time", None),
        ("deltaG_vs_time_divein", (-20.0, 0.0)),
    ]:
        plot_series_mean_err(
            out,
            stem=stem,
            xlabel=r"$t$ / ps",
            ylabel=r"$\Delta G=k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$",
            t_meta=t_meta,
            y_meta=dG_meta_mean,
            e_meta=dG_meta_std,
            t_sfpt=t_sfpt,
            y_sfpt=dG_sfpt_mean,
            e_sfpt=dG_sfpt_std,
            dG_ref=float(dG_us),
            ref_label="US",
            ylim=ylim,
        )
    for stem, ylim in [
        ("deltaG_vs_wallclock", None),
        ("deltaG_vs_wallclock_divein", (-20.0, 0.0)),
    ]:
        plot_series_mean_err(
            out,
            stem=stem,
            xlabel=r"wall-clock time / h",
            ylabel=r"$\Delta G=k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$",
            t_meta=tw_meta,
            y_meta=dG_meta_mean,
            e_meta=dG_meta_std,
            t_sfpt=tw_sfpt,
            y_sfpt=dG_sfpt_mean,
            e_sfpt=dG_sfpt_std,
            dG_ref=float(dG_us),
            ref_label="US",
            ylim=ylim,
        )

    plot_1d_collage(
        out,
        xi=xi,
        f_meta_mean=f1d_meta,
        f_meta_lo=f1d_meta - f1d_meta_std,
        f_meta_hi=f1d_meta + f1d_meta_std,
        f_us=f1d_ref,
        f_sfpt_mean=f_sfpt_mean,
        f_sfpt_lo=f_sfpt_mean - f_sfpt_std,
        f_sfpt_hi=f_sfpt_mean + f_sfpt_std,
        titles=titles,
        vmax=float(args.vmax_kcal),
        ncols=int(args.ncols),
    )
    print(
        f"[done] out={out}  sFPT n_groups={n_groups}  "
        f"US ΔG={dG_us:.4f}  MetaD ΔG={dG_meta:.4f}  "
        f"RMSE window=[{xi_lo:g},{xi_hi:g}]  ΔG=full 2D",
        flush=True,
    )


if __name__ == "__main__":
    main()
