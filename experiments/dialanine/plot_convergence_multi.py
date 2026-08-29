#!/usr/bin/env python3
"""Dialanine vacuum convergence: 3×sFPT + 3×WT-MetaD (mean±std).

Same figure suite as ``plot_convergence_analysis.py``, but RMSE / ΔG / F(φ)
are mean±std over seeds. Reference = mean MetaD F(φ) from the last
``metad-ref-last-ps`` of hills (no US for vacuum dialanine).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(EXP))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    EXP / "plot_convergence_analysis.py",
)
_dca = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dca)

from plot_fes_compare import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    FE_LABEL,
    nature_rcparams,
    relative_fes,
)
from plot_fes_rmse_vs_time import delta_g_two_basins, rmse_1d  # noqa: E402
from plot_g5_1d_grid_and_dg_zoom import COLOR_US, fpt_wall_hours_from_throughput  # noqa: E402


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    n = stack.shape[axis]
    if n <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


def plot_series_err(
    out: Path,
    *,
    stem: str,
    xlabel: str,
    ylabel: str,
    t_meta,
    y_meta,
    e_meta,
    t_sfpt,
    y_sfpt,
    e_sfpt,
    dG_ref: float | None = None,
    ref_label: str = "MetaD",
    ylim=None,
    y0_line: bool = False,
    xlim_right: float | None = None,
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
            zorder=1,
            label=rf"{ref_label} $\Delta G$={dG_ref:.2f}",
        )
    if y0_line:
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.plot(t_meta, y_meta, color=COLOR_META, lw=1.3, label="WT-MetaD mean", zorder=2)
    if e_meta is not None:
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
    ax.plot(t_sfpt, y_sfpt, color=COLOR_SFPT, lw=1.5, label="sFPT mean", zorder=4)
    if e_sfpt is not None:
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
    ax.set_xlim(left=0.0, right=xlim_right)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def plot_1d_collage(
    out: Path,
    *,
    xi,
    f_meta_mean,
    f_meta_std,
    f_sfpt_mean,
    f_sfpt_std,
    titles,
    vmax: float,
    ncols: int,
    stages_per_panel: int,
) -> None:
    n = len(titles)
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
    clip = float(vmax) > 0
    y_meta = relative_fes(f_meta_mean, None)
    y_mlo = relative_fes(f_meta_mean - f_meta_std, None)
    y_mhi = relative_fes(f_meta_mean + f_meta_std, None)
    if clip:
        y_meta = np.minimum(y_meta, float(vmax))
        y_mlo = np.minimum(y_mlo, float(vmax))
        y_mhi = np.minimum(y_mhi, float(vmax))
        y_hi = float(vmax) * 1.02
    else:
        y_hi = None
    ymax_auto = 0.0
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ym = relative_fes(f_sfpt_mean[ig], None)
        ylo = relative_fes(f_sfpt_mean[ig] - f_sfpt_std[ig], None)
        yhi = relative_fes(f_sfpt_mean[ig] + f_sfpt_std[ig], None)
        if clip:
            ym = np.minimum(ym, float(vmax))
            ylo = np.minimum(ylo, float(vmax))
            yhi = np.minimum(yhi, float(vmax))
        ax.fill_between(xi, y_mlo, y_mhi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
        ax.plot(xi, y_meta, color=COLOR_META, lw=1.05, label="WT-MetaD mean", zorder=2)
        ax.fill_between(xi, ylo, yhi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
        ax.plot(xi, ym, color=COLOR_SFPT, lw=1.25, label="sFPT mean", zorder=4)
        ax.axvline(0.0, color="0.5", ls=":", lw=0.7, zorder=0)
        if clip:
            ax.set_ylim(0.0, y_hi)
        else:
            ymax_auto = max(
                ymax_auto, float(np.nanmax(yhi)), float(np.nanmax(y_mhi))
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
    if not clip and ymax_auto > 0:
        for ax in fig.axes:
            if ax.has_data():
                ax.set_ylim(0.0, ymax_auto * 1.05)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        rf"1D $F(\phi)$  MetaD / sFPT mean$\pm$std  ({int(stages_per_panel)} stages / panel)",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print("[ok] fes_1d_all_groups", flush=True)


def split_dirs(text: str) -> list[Path]:
    return [Path(x.strip()).resolve() for x in str(text).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dirs", type=str, required=True)
    p.add_argument("--metad-dirs", type=str, required=True)
    p.add_argument("--out-dir", type=Path, default=EXP / "convergence_analysis")
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=0.5)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--vmax-kcal", type=float, default=0.0)
    p.add_argument("--rmse-vmax-extra", type=float, default=8.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--n-replicas", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=15.0)
    p.add_argument("--sigma", type=str, default="0.35,0.35")
    p.add_argument("--metad-stride-deposits", type=int, default=10)
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-np.pi)
    p.add_argument("--xi-hi", type=float, default=np.pi)
    p.add_argument("--fpt-rate", type=float, default=64.4)
    p.add_argument("--metad-rate", type=float, default=139.0)
    p.add_argument("--fpt-gpus", type=int, default=1)
    p.add_argument("--metad-gpus", type=int, default=1)
    p.add_argument("--collage-ncols", type=int, default=4)
    p.add_argument("--collage-vmax", type=float, default=12.0)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--metad-ref-last-ps", type=float, default=200.0)
    p.add_argument("--t-max", type=float, default=25.0)
    p.add_argument("--lambda-target", type=float, default=0.1)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fpt_dirs = split_dirs(args.fpt_dirs)
    metad_dirs = split_dirs(args.metad_dirs)
    if not fpt_dirs or not metad_dirs:
        raise ValueError("need at least one --fpt-dirs and one --metad-dirs")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)
    sigma = tuple(
        float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip()
    )
    xi_lo, xi_hi, xi_cut = float(args.xi_lo), float(args.xi_hi), float(args.xi_cut)
    vmax = float(args.vmax_kcal) if args.vmax_kcal > 0 else None
    t_max = float(args.t_max)

    phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    fes0 = metad_dirs[0] / "fes.npz"
    if fes0.is_file():
        zfes = np.load(fes0)
        phi_f = np.asarray(zfes["phi"], dtype=np.float64)
        psi_f = np.asarray(zfes["psi"], dtype=np.float64)
        if len(phi_f) == int(args.grid_n) and len(psi_f) == int(args.grid_n):
            phi, psi = phi_f, psi_f

    # MetaD last-N-ps F(φ) per seed → mean reference
    f1d_ref_list = []
    xi = None
    t_meta_tot = None
    for md in metad_dirs:
        hills = md / "hills.npz"
        if not hills.is_file():
            raise FileNotFoundError(hills)
        xi_i, f1d_i, t_tot = _dca.metad_ref_from_last_ps(
            hills,
            phi=phi,
            psi=psi,
            n_replicas=int(args.n_replicas),
            deposit_pace=int(args.deposit_pace),
            timestep=float(args.timestep),
            bias_factor=float(args.bias_factor),
            sigma=sigma,
            last_ps=float(args.metad_ref_last_ps),
            n_bins=int(args.n_bins_1d),
            temperature=float(args.temperature),
            device=device,
        )
        if xi is None:
            xi = xi_i
            t_meta_tot = t_tot
        f1d_ref_list.append(relative_fes(f1d_i, None))
        print(f"[ref] {md.name}  traj={t_tot:.1f} ps", flush=True)
    f1d_ref, f1d_ref_std = mean_std(np.stack(f1d_ref_list, 0), 0)
    f1d_ref = relative_fes(f1d_ref, None)
    dG_ref = delta_g_two_basins(
        xi,
        f1d_ref,
        temperature=float(args.temperature),
        xi_cut=xi_cut,
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )
    print(
        f"[ref] MetaD F(φ) mean of {len(metad_dirs)} seeds, "
        f"last {args.metad_ref_last_ps:g} ps (traj={t_meta_tot:.1f} ps)  "
        f"ΔG_ref={dG_ref:.4f} kcal/mol",
        flush=True,
    )

    max_groups = args.max_groups
    if max_groups is None and t_max > 0:
        max_groups = max(1, int(np.floor(t_max / float(args.dt_group) + 1e-12)))

    n_stages_each = [
        len(
            _dca.discover_dihedral_stages(
                d,
                lambda_target=float(args.lambda_target),
                lambda_atol=float(args.lambda_atol),
            )
        )
        for d in fpt_dirs
    ]
    n_groups_each = [n // int(args.group_size) for n in n_stages_each]
    n_groups = int(min(n_groups_each))
    if max_groups is not None:
        n_groups = min(n_groups, int(max_groups))
    print(
        f"[sFPT] stages={n_stages_each} groups={n_groups_each} → use {n_groups} "
        f"({n_groups * args.dt_group:g} ps)",
        flush=True,
    )

    # Incremental MetaD vs shared ref
    t_m_list, rmse_m_list, dG_m_list, f_m_list = [], [], [], []
    for md in metad_dirs:
        t_m, rmse_m, dG_m, f_m = _dca.incremental_metad_phi(
            md / "hills.npz",
            phi=phi,
            psi=psi,
            f1d_ref=f1d_ref,
            xi=xi,
            n_replicas=int(args.n_replicas),
            deposit_pace=int(args.deposit_pace),
            timestep=float(args.timestep),
            bias_factor=float(args.bias_factor),
            sigma=sigma,
            temperature=float(args.temperature),
            n_bins=int(args.n_bins_1d),
            vmax_kcal=vmax,
            stride_deposits=int(args.metad_stride_deposits),
            xi_cut=xi_cut,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
            device=device,
        )
        t_m_list.append(t_m)
        rmse_m_list.append(rmse_m)
        dG_m_list.append(dG_m)
        f_m_list.append(f_m)
        print(f"[metad-inc] {md.name} n={len(t_m)} t_end={t_m[-1]:.1f}", flush=True)

    # sFPT groups vs shared ref
    f_sfpt_runs, rmse_runs, dG_runs = [], [], []
    t_sfpt = titles = None
    for d in fpt_dirs:
        t_s, r_s, g_s, f_s, titles_s = _dca.kde_groups_phi(
            d,
            phi=phi,
            psi=psi,
            f1d_ref=f1d_ref,
            xi=xi,
            group_size=int(args.group_size),
            dt_group=float(args.dt_group),
            temperature=float(args.temperature),
            vmax=vmax,
            device=device,
            chunk=int(args.chunk),
            n_bins=int(args.n_bins_1d),
            xi_cut=xi_cut,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
            max_groups=n_groups,
            lambda_target=float(args.lambda_target),
            lambda_atol=float(args.lambda_atol),
        )
        f_sfpt_runs.append(f_s)
        rmse_runs.append(r_s)
        dG_runs.append(g_s)
        if t_sfpt is None:
            t_sfpt = t_s
            titles = titles_s
        print(f"[kde] {d.name} groups={len(t_s)} last RMSE={r_s[-1]:.4f}", flush=True)

    t_sfpt = np.asarray(t_sfpt, dtype=np.float64)
    if t_max > 0:
        keep = t_sfpt <= t_max + 1e-12
        t_sfpt = t_sfpt[keep]
        titles = titles[: int(keep.sum())]
        f_sfpt_runs = [f[keep] for f in f_sfpt_runs]
        rmse_runs = [r[keep] for r in rmse_runs]
        dG_runs = [g[keep] for g in dG_runs]

    # Align each MetaD seed onto the sFPT t grid, then mean±std
    rmse_m_al, dG_m_al, f_m_al = [], [], []
    for t_m, r_m, g_m, f_m in zip(t_m_list, rmse_m_list, dG_m_list, f_m_list):
        if t_max > 0:
            t_m, r_m, g_m, f_m = _dca.truncate_by_tmax(t_m, t_max, r_m, g_m, f_m)
        rmse_m_al.append(_dca.align_series_to_times(t_m, r_m, t_sfpt))
        dG_m_al.append(_dca.align_series_to_times(t_m, g_m, t_sfpt))
        f_m_al.append(_dca.align_series_to_times(t_m, f_m, t_sfpt))
    t_meta = np.asarray(t_sfpt, dtype=np.float64).copy()
    rmse_meta, rmse_meta_std = mean_std(np.stack(rmse_m_al, 0), 0)
    dG_meta, dG_meta_std = mean_std(np.stack(dG_m_al, 0), 0)
    f_meta_traj, _ = mean_std(np.stack(f_m_al, 0), 0)

    f_sfpt, f_sfpt_std = mean_std(np.stack(f_sfpt_runs, 0), 0)
    rmse_sfpt, rmse_sfpt_std = mean_std(np.stack(rmse_runs, 0), 0)
    dG_sfpt, dG_sfpt_std = mean_std(np.stack(dG_runs, 0), 0)

    tw_sfpt = fpt_wall_hours_from_throughput(
        t_sfpt, timestep_ps=float(args.timestep), rate_it_s=float(args.fpt_rate)
    )
    tw_meta_own = fpt_wall_hours_from_throughput(
        t_meta, timestep_ps=float(args.timestep), rate_it_s=float(args.metad_rate)
    )
    tw_meta = tw_sfpt.copy()
    cost_sfpt = tw_sfpt * float(args.fpt_gpus)
    cost_meta = cost_sfpt.copy()
    print(
        f"[align] MetaD → sFPT t  n={len(t_sfpt)}  "
        f"t={t_sfpt[0]:g}–{t_sfpt[-1]:g} ps  "
        f"(MetaD own wall {tw_meta_own[-1]:.4g} h)",
        flush=True,
    )

    np.savez_compressed(
        out / "fes_rmse_vs_time.npz",
        t_metad_ps=t_meta,
        t_kde_ps=t_sfpt,
        tw_metad_h=tw_meta,
        tw_kde_h=tw_sfpt,
        cost_metad_gpuh=cost_meta,
        cost_kde_gpuh=cost_sfpt,
        rmse_metad=rmse_meta,
        rmse_metad_std=rmse_meta_std,
        rmse_kde=rmse_sfpt,
        rmse_kde_std=rmse_sfpt_std,
        dG_metad=dG_meta,
        dG_metad_std=dG_meta_std,
        dG_kde=dG_sfpt,
        dG_kde_std=dG_sfpt_std,
        dG_metad_ref=dG_ref,
        xi=xi,
        F1d_metad_ref=f1d_ref,
        F1d_metad_ref_std=f1d_ref_std,
        group_size=int(args.group_size),
        dt_group_ps=float(args.dt_group),
        fpt_dirs=[str(d) for d in fpt_dirs],
        metad_dirs=[str(d) for d in metad_dirs],
        metad_ref_last_ps=float(args.metad_ref_last_ps),
        t_max_ps=float(t_max),
        tw_metad_own_h=tw_meta_own,
    )
    np.savez_compressed(
        out / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad_mean=f1d_ref,
        F1d_metad_std=f1d_ref_std,
        F1d_kde_mean=f_sfpt,
        F1d_kde_std=f_sfpt_std,
        t_ps=t_sfpt,
        titles=np.asarray(titles),
    )

    ylab_rmse = (
        rf"1D $F(\phi)$ RMSE ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim, xm, xs, xlab, xright in [
        ("fes_rmse_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", t_max if t_max > 0 else None),
        (
            "fes_rmse_vs_time_divein",
            (0.0, 8.0),
            t_meta,
            t_sfpt,
            r"$t$ / ps",
            t_max if t_max > 0 else None,
        ),
        ("fes_rmse_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            "fes_rmse_vs_wallclock_divein",
            (0.0, 8.0),
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
            None,
        ),
        (
            "fes_rmse_vs_cost",
            None,
            cost_meta,
            cost_sfpt,
            rf"cost / GPU$\cdot$h  (sFPT $\times${args.fpt_gpus}, MetaD $\times${args.metad_gpus})",
            None,
        ),
        (
            "fes_rmse_vs_cost_divein",
            (0.0, 8.0),
            cost_meta,
            cost_sfpt,
            rf"cost / GPU$\cdot$h  (sFPT $\times${args.fpt_gpus}, MetaD $\times${args.metad_gpus})",
            None,
        ),
    ]:
        plot_series_err(
            out,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab_rmse,
            t_meta=xm,
            y_meta=rmse_meta,
            e_meta=rmse_meta_std,
            t_sfpt=xs,
            y_sfpt=rmse_sfpt,
            e_sfpt=rmse_sfpt_std,
            ylim=ylim,
            y0_line=True,
            xlim_right=xright,
        )

    vmax_extra = float(args.rmse_vmax_extra)
    if vmax_extra > 0:
        r30_m = [
            np.asarray(
                [
                    rmse_1d(
                        f_m_al[s][i],
                        f1d_ref,
                        vmax=vmax_extra,
                        xi=xi,
                        xi_lo=xi_lo,
                        xi_hi=xi_hi,
                    )
                    for i in range(len(t_meta))
                ]
            )
            for s in range(len(f_m_al))
        ]
        r30_s = [
            np.asarray(
                [
                    rmse_1d(
                        f_sfpt_runs[s][i],
                        f1d_ref,
                        vmax=vmax_extra,
                        xi=xi,
                        xi_lo=xi_lo,
                        xi_hi=xi_hi,
                    )
                    for i in range(len(t_sfpt))
                ]
            )
            for s in range(len(f_sfpt_runs))
        ]
        r30_m_mean, r30_m_std = mean_std(np.stack(r30_m, 0), 0)
        r30_s_mean, r30_s_std = mean_std(np.stack(r30_s, 0), 0)
        np.savez_compressed(
            out / f"fes_rmse_F{int(vmax_extra)}.npz",
            t_metad_ps=t_meta,
            t_kde_ps=t_sfpt,
            tw_metad_h=tw_meta,
            tw_kde_h=tw_sfpt,
            cost_metad_gpuh=cost_meta,
            cost_kde_gpuh=cost_sfpt,
            rmse_metad=r30_m_mean,
            rmse_metad_std=r30_m_std,
            rmse_kde=r30_s_mean,
            rmse_kde_std=r30_s_std,
            vmax_kcal=vmax_extra,
        )
        ylab_r30 = (
            rf"1D $F(\phi)$ RMSE ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$, "
            rf"$F_{{\mathrm{{ref}}}}\leq {vmax_extra:g}$) / kcal mol$^{{-1}}$"
        )
        for stem, ylim, xm, xs, xlab, xright in [
            (
                f"fes_rmse_F{int(vmax_extra)}_vs_time",
                None,
                t_meta,
                t_sfpt,
                r"$t$ / ps",
                t_max if t_max > 0 else None,
            ),
            (
                f"fes_rmse_F{int(vmax_extra)}_vs_time_divein",
                (0.0, 8.0),
                t_meta,
                t_sfpt,
                r"$t$ / ps",
                t_max if t_max > 0 else None,
            ),
            (
                f"fes_rmse_F{int(vmax_extra)}_vs_wallclock",
                None,
                tw_meta,
                tw_sfpt,
                r"wall-clock time / h",
                None,
            ),
            (
                f"fes_rmse_F{int(vmax_extra)}_vs_wallclock_divein",
                (0.0, 8.0),
                tw_meta,
                tw_sfpt,
                r"wall-clock time / h",
                None,
            ),
            (
                f"fes_rmse_F{int(vmax_extra)}_vs_cost",
                None,
                cost_meta,
                cost_sfpt,
                rf"cost / GPU$\cdot$h  (sFPT $\times${args.fpt_gpus}, MetaD $\times${args.metad_gpus})",
                None,
            ),
            (
                f"fes_rmse_F{int(vmax_extra)}_vs_cost_divein",
                (0.0, 8.0),
                cost_meta,
                cost_sfpt,
                rf"cost / GPU$\cdot$h  (sFPT $\times${args.fpt_gpus}, MetaD $\times${args.metad_gpus})",
                None,
            ),
        ]:
            plot_series_err(
                out,
                stem=stem,
                xlabel=xlab,
                ylabel=ylab_r30,
                t_meta=xm,
                y_meta=r30_m_mean,
                e_meta=r30_m_std,
                t_sfpt=xs,
                y_sfpt=r30_s_mean,
                e_sfpt=r30_s_std,
                ylim=ylim,
                y0_line=True,
                xlim_right=xright,
            )

    ylab_dg = (
        rf"$\Delta G=k_BT\ln(P_{{\phi<0}}/P_{{\phi>0}})$ "
        rf"($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim, xm, xs, xlab, xright in [
        ("deltaG_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", t_max if t_max > 0 else None),
        (
            "deltaG_vs_time_divein",
            (-5.0, 5.0),
            t_meta,
            t_sfpt,
            r"$t$ / ps",
            t_max if t_max > 0 else None,
        ),
        ("deltaG_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            "deltaG_vs_wallclock_divein",
            (-5.0, 5.0),
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
            None,
        ),
    ]:
        plot_series_err(
            out,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab_dg,
            t_meta=xm,
            y_meta=dG_meta,
            e_meta=dG_meta_std,
            t_sfpt=xs,
            y_sfpt=dG_sfpt,
            e_sfpt=dG_sfpt_std,
            dG_ref=float(dG_ref),
            ref_label="MetaD",
            ylim=ylim,
            xlim_right=xright,
        )

    plot_1d_collage(
        out,
        xi=xi,
        f_meta_mean=f1d_ref,
        f_meta_std=f1d_ref_std,
        f_sfpt_mean=f_sfpt,
        f_sfpt_std=f_sfpt_std,
        titles=titles,
        vmax=float(args.collage_vmax),
        ncols=int(args.collage_ncols),
        stages_per_panel=int(args.group_size),
    )
    print(
        f"[done] out={out}  sFPT groups={len(t_sfpt)}  "
        f"RMSE={rmse_sfpt[-1]:.4f}±{rmse_sfpt_std[-1]:.4f}  "
        f"ΔG={dG_sfpt[-1]:.4f}±{dG_sfpt_std[-1]:.4f}  "
        f"MetaD RMSE={rmse_meta[-1]:.4f}±{rmse_meta_std[-1]:.4f}  "
        f"ΔG={dG_meta[-1]:.4f}±{dG_meta_std[-1]:.4f}  ref={dG_ref:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
