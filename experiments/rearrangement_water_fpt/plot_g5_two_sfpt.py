#!/usr/bin/env python3
"""Combine two sFPT runs (86736 + 86938) with original MetaD live/ + US ΔG ref.

Outputs under ``rmse_vs_time_g5``:
  - mean ± error (half-range of 2 runs) for RMSE / ΔG vs t and wall-clock
  - both individual sFPT curves
  - 1D FES collage: MetaD, US, mean sFPT ± band, optional both runs
"""

from __future__ import annotations

import argparse
import os
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
    nature_rcparams,
    relative_fes,
)
from plot_fes_rmse_vs_time import (  # noqa: E402
    delta_g_two_basins,
    discover_stage_files,
    incremental_metad_metrics,
    make_1d_edges_from_grid,
    project_1d_fixed_edges,
    rmse_1d,
)
from plot_g5_1d_grid_and_dg_zoom import (  # noqa: E402
    COLOR_US,
    DEFAULT_US_MBAR,
    _fpt_wall_hours_from_stages,
    load_metad_1d_ref,
    load_us_mbar_1d_d2_minus_d1,
    us_delta_g_ref,
)
from plot_kde_fes import density_on_grid  # noqa: E402

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
COLOR_SFPT_MEAN = "#d35400"


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return t_ps, rmse, dG, F1d[n_groups,n_bins], titles."""
    rows = discover_stage_files(fpt_dir)
    n_groups = len(rows) // int(group_size)
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
        t_list.append(t_ps)
        r_list.append(rmse_1d(f1d, f1d_ref, vmax=vmax))
        g_list.append(
            delta_g_two_basins(xi, f1d, temperature=temperature, xi_cut=0.0)
        )
        f_list.append(f1d)
        s0, s1 = chunk_rows[0][0], chunk_rows[-1][0]
        titles.append(f"t={t_ps:g} ps  stage{s0:04d}–{s1:04d}")
        print(
            f"[kde {fpt_dir.name}] {titles[-1]}  RMSE={r_list[-1]:.4f}  "
            f"ΔG={g_list[-1]:.4f}",
            flush=True,
        )
    return (
        np.asarray(t_list, dtype=np.float64),
        np.asarray(r_list, dtype=np.float64),
        np.asarray(g_list, dtype=np.float64),
        np.stack(f_list, axis=0),
        titles,
    )


def mean_err_two(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and half-range error for two runs (aligned)."""
    mean = 0.5 * (a + b)
    err = 0.5 * np.abs(a - b)
    return mean, err


def plot_rmse_two(
    out: Path,
    *,
    t_m: np.ndarray,
    rmse_m: np.ndarray,
    t: np.ndarray,
    rmse_mean: np.ndarray,
    rmse_err: np.ndarray,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.plot(t_m, rmse_m, color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
    ax.errorbar(
        t,
        rmse_mean,
        yerr=rmse_err,
        color=COLOR_SFPT,
        lw=1.5,
        marker="o",
        ms=4.0,
        capsize=2.5,
        elinewidth=1.0,
        label=r"sFPT mean$\pm\frac{1}{2}$range",
        zorder=4,
    )
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"1D FES RMSE / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    ymax = float(np.nanmax(np.concatenate([rmse_m, rmse_mean + rmse_err])))
    ax.set_ylim(bottom=-0.02 * max(ymax, 1.0), top=ymax * 1.08 if ymax > 0 else 1.0)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "fes_rmse_vs_time.png", dpi=200)
    fig.savefig(out / "fes_rmse_vs_time.pdf")
    plt.close(fig)


def plot_dg_two(
    out: Path,
    *,
    t_m: np.ndarray,
    dG_m: np.ndarray,
    t: np.ndarray,
    dG_mean: np.ndarray,
    dG_err: np.ndarray,
    dG_ref: float,
    ref_label: str,
    ylim: tuple[float, float] | None,
    stem: str,
) -> None:
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
    ax.plot(t_m, dG_m, color=COLOR_META, lw=1.3, label="WT-MetaD", zorder=2)
    ax.errorbar(
        t,
        dG_mean,
        yerr=dG_err,
        color=COLOR_SFPT,
        lw=1.5,
        marker="o",
        ms=4.0,
        capsize=2.5,
        elinewidth=1.0,
        label=r"sFPT mean$\pm\frac{1}{2}$range",
        zorder=4,
    )
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$\Delta G = k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vals = np.concatenate([dG_mean, dG_m, np.asarray([dG_ref, 0.0])])
        vals = vals[np.isfinite(vals)]
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def plot_dg_wall_two(
    out: Path,
    *,
    tw_m: np.ndarray,
    dG_m: np.ndarray,
    tw: np.ndarray,
    dG_mean: np.ndarray,
    dG_err: np.ndarray,
    dG_ref: float,
    ref_label: str,
    ylim: tuple[float, float] | None,
    stem: str,
) -> None:
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
        label=r"WT-MetaD (1$\times$GPU)",
        zorder=2,
    )
    ax.errorbar(
        tw,
        dG_mean,
        yerr=dG_err,
        color=COLOR_SFPT,
        lw=1.5,
        marker="o",
        ms=4.0,
        capsize=2.5,
        elinewidth=1.0,
        label=r"sFPT mean$\pm\frac{1}{2}$range (8$\times$GPU)",
        zorder=4,
    )
    ax.set_xlabel(r"wall-clock time / h")
    ax.set_ylabel(r"$\Delta G = k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vals = np.concatenate([dG_mean, dG_m, np.asarray([dG_ref, 0.0])])
        vals = vals[np.isfinite(vals)]
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def plot_1d_collage_two(
    out: Path,
    *,
    xi: np.ndarray,
    f_meta: np.ndarray,
    f_us: np.ndarray | None,
    f_a: np.ndarray,
    f_b: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
) -> None:
    n = min(f_a.shape[0], f_b.shape[0], len(titles))
    ncols = int(ncols)
    nrows = int(np.ceil(n / ncols))
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
    y_meta = np.minimum(relative_fes(f_meta, None), float(vmax))
    y_us = (
        np.minimum(relative_fes(f_us, None), float(vmax)) if f_us is not None else None
    )
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ya = np.minimum(relative_fes(f_a[ig], None), float(vmax))
        yb = np.minimum(relative_fes(f_b[ig], None), float(vmax))
        ymean = 0.5 * (ya + yb)
        ylo = np.minimum(ya, yb)
        yhi = np.maximum(ya, yb)
        ax.plot(xi, y_meta, color=COLOR_META, lw=1.1, label="WT-MetaD", zorder=2)
        ax.fill_between(xi, 0.0, y_meta, color=COLOR_META, alpha=0.08, lw=0, zorder=1)
        if y_us is not None:
            ax.plot(xi, y_us, color=COLOR_US, lw=1.15, ls="--", label="US-MBAR", zorder=3)
        ax.fill_between(
            xi, ylo, yhi, color=COLOR_SFPT, alpha=0.25, lw=0, zorder=3, label="sFPT ±range"
        )
        ax.plot(xi, ymean, color=COLOR_SFPT, lw=1.35, label="sFPT mean", zorder=5)
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
            ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        "1D FES  MetaD / US / sFPT mean±range  (5 stages / panel)",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print(f"[ok] wrote {out / 'fes_1d_all_groups.png'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fpt-a",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736"
        ),
    )
    p.add_argument(
        "--fpt-b",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86938"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <fpt-a>/kde_fes/rmse_vs_time_g5",
    )
    p.add_argument(
        "--metad-dir",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_metadynamics/results"
            "/claisen_water_metad_mh-0_r64_86711/live"
        ),
    )
    p.add_argument("--us-mbar", type=Path, default=DEFAULT_US_MBAR)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=2.5)
    p.add_argument("--bounds", type=str, default="1.0,4.0,1.0,4.0")
    p.add_argument("--grid-n", type=int, default=120)
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--n-replicas", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=30.0)
    p.add_argument("--sigma", type=str, default="0.2,0.2")
    p.add_argument("--metad-stride-deposits", type=int, default=10)
    p.add_argument("--metad-rate", type=float, default=13.74)
    p.add_argument("--reuse-f1d-a", type=Path, default=None, help="fes_1d_all_groups.npz")
    p.add_argument("--ncols", type=int, default=4)
    args = p.parse_args()

    fpt_a = args.fpt_a.resolve()
    fpt_b = args.fpt_b.resolve()
    out = (args.out_dir or (fpt_a / "kde_fes" / "rmse_vs_time_g5")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metad_dir = args.metad_dir.resolve()
    bounds = tuple(
        float(x) for x in args.bounds.replace(";", ",").split(",") if x.strip()
    )
    assert len(bounds) == 4
    sigma = tuple(float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    # MetaD reference from original live/
    xi, f1d_ref, edges = load_metad_1d_ref(
        metad_dir / "fes.npz", bounds, int(args.n_bins_1d), float(args.temperature)
    )
    dG_meta_ref = delta_g_two_basins(
        xi, f1d_ref, temperature=float(args.temperature), xi_cut=0.0
    )
    print(f"[metad-ref] ΔG={dG_meta_ref:.4f}  from {metad_dir}", flush=True)

    d1 = np.linspace(bounds[0], bounds[1], int(args.grid_n))
    d2 = np.linspace(bounds[2], bounds[3], int(args.grid_n))
    # use same cropped axes as load_metad_1d_ref projection domain
    fes = np.load(metad_dir / "fes.npz")
    d1_m = np.asarray(fes["d1"])
    d2_m = np.asarray(fes["d2"])
    m1 = (d1_m >= bounds[0]) & (d1_m <= bounds[1])
    m2 = (d2_m >= bounds[2]) & (d2_m <= bounds[3])
    d1_use, d2_use = d1_m[m1], d2_m[m2]

    t_m, rmse_m, dG_m = incremental_metad_metrics(
        metad_dir / "hills.npz",
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
        xi_cut=0.0,
    )

    # sFPT A: reuse cached 1D if available and xi matches
    reuse_a = args.reuse_f1d_a or (out / "fes_1d_all_groups.npz")
    got_a = False
    if Path(reuse_a).is_file():
        old = np.load(reuse_a, allow_pickle=True)
        if "F1d_kde" in old.files and np.allclose(old["xi"], xi):
            f_a = np.asarray(old["F1d_kde"], dtype=np.float64)
            t_a = np.asarray(old["t_ps"], dtype=np.float64)
            titles_a = [str(x) for x in old["titles"].tolist()]
            # if old F1d was vs rerun MetaD, recompute rmse/dG vs live MetaD
            rmse_a = np.array(
                [rmse_1d(f_a[i], f1d_ref, vmax=float(args.vmax_kcal)) for i in range(len(t_a))]
            )
            dG_a = np.array(
                [
                    delta_g_two_basins(xi, f_a[i], temperature=float(args.temperature), xi_cut=0.0)
                    for i in range(len(t_a))
                ]
            )
            got_a = True
            print(f"[reuse-A] {reuse_a}  n={len(t_a)}", flush=True)
    if not got_a:
        t_a, rmse_a, dG_a, f_a, titles_a = kde_1d_groups(
            fpt_a,
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
        )

    t_b, rmse_b, dG_b, f_b, titles_b = kde_1d_groups(
        fpt_b,
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
    )

    n_common = min(len(t_a), len(t_b))
    titles = titles_a[:n_common]
    dG_us, _, f_us = us_delta_g_ref(
        args.us_mbar, temperature=float(args.temperature), xi_cut=0.0, xi_query=xi
    )

    mean_r, err_r = mean_err_two(rmse_a[:n_common], rmse_b[:n_common])
    mean_g, err_g = mean_err_two(dG_a[:n_common], dG_b[:n_common])
    t_common = t_a[:n_common]

    tw_a = _fpt_wall_hours_from_stages(
        fpt_a, group_size=int(args.group_size), n_groups=len(t_a)
    )
    tw_b = _fpt_wall_hours_from_stages(
        fpt_b, group_size=int(args.group_size), n_groups=len(t_b)
    )
    tw_m = (t_m / float(args.timestep)) / float(args.metad_rate) / 3600.0
    tw_common = 0.5 * (tw_a[:n_common] + tw_b[:n_common])

    np.savez_compressed(
        out / "fes_rmse_vs_time.npz",
        t_metad_ps=t_m,
        rmse_metad=rmse_m,
        dG_metad=dG_m,
        dG_metad_ref=dG_meta_ref,
        dG_us_ref=dG_us,
        xi=xi,
        F1d_metad_ref=f1d_ref,
        F1d_us_ref=f_us,
        t_kde_ps=t_common,
        rmse_kde=mean_r,
        rmse_kde_err=err_r,
        dG_kde=mean_g,
        dG_kde_err=err_g,
        t_kde_a_ps=t_a,
        rmse_kde_a=rmse_a,
        dG_kde_a=dG_a,
        t_kde_b_ps=t_b,
        rmse_kde_b=rmse_b,
        dG_kde_b=dG_b,
        tw_kde_a_h=tw_a,
        tw_kde_b_h=tw_b,
        tw_kde_h=tw_common,
        tw_metad_h=tw_m,
        group_size=int(args.group_size),
        dt_group_ps=float(args.dt_group),
        xi_cut=0.0,
        vmax_kcal=float(args.vmax_kcal),
        timestep_ps=float(args.timestep),
        n_replicas=int(args.n_replicas),
        deposit_pace=int(args.deposit_pace),
    )
    np.savez_compressed(
        out / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad=f1d_ref,
        F1d_us=f_us,
        F1d_kde_a=f_a[:n_common],
        F1d_kde_b=f_b[:n_common],
        F1d_kde=0.5 * (f_a[:n_common] + f_b[:n_common]),
        t_ps=t_common,
        titles=np.asarray(titles),
    )

    plot_rmse_two(
        out,
        t_m=t_m,
        rmse_m=rmse_m,
        t=t_common,
        rmse_mean=mean_r,
        rmse_err=err_r,
    )
    plot_dg_two(
        out,
        t_m=t_m,
        dG_m=dG_m,
        t=t_common,
        dG_mean=mean_g,
        dG_err=err_g,
        dG_ref=float(dG_us),
        ref_label="US",
        ylim=None,
        stem="deltaG_vs_time",
    )
    plot_dg_two(
        out,
        t_m=t_m,
        dG_m=dG_m,
        t=t_common,
        dG_mean=mean_g,
        dG_err=err_g,
        dG_ref=float(dG_us),
        ref_label="US",
        ylim=(-20.0, 0.0),
        stem="deltaG_vs_time_divein",
    )
    plot_dg_wall_two(
        out,
        tw_m=tw_m,
        dG_m=dG_m,
        tw=tw_common,
        dG_mean=mean_g,
        dG_err=err_g,
        dG_ref=float(dG_us),
        ref_label="US",
        ylim=None,
        stem="deltaG_vs_wallclock",
    )
    plot_dg_wall_two(
        out,
        tw_m=tw_m,
        dG_m=dG_m,
        tw=tw_common,
        dG_mean=mean_g,
        dG_err=err_g,
        dG_ref=float(dG_us),
        ref_label="US",
        ylim=(-20.0, 0.0),
        stem="deltaG_vs_wallclock_divein",
    )
    plot_1d_collage_two(
        out,
        xi=xi,
        f_meta=f1d_ref,
        f_us=f_us,
        f_a=f_a,
        f_b=f_b,
        titles=titles,
        vmax=float(args.vmax_kcal),
        ncols=int(args.ncols),
    )
    print(
        f"[done] out={out}  n_common={n_common}  "
        f"US ΔG={dG_us:.4f}  MetaD ΔG={dG_meta_ref:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
