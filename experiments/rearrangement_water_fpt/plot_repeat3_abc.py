#!/usr/bin/env python3
"""Recompute convergence_repeat3_gpu3 with fes_compare_1x4 a/b/c methods.

MetaD: FFT-convolved hills on (d1,d2)∈[1,7]², not dumped fes.npz.
sFPT: KDE on the same grid, P-floor mask.
1D: Boltzmann projection ξ=d2−d1 from that 2D FES (+ gaussian smooth for display).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))

from plot_convergence_analysis import (  # noqa: E402
    mean_std,
    plot_1d_collage,
    plot_rmse_suite,
    plot_series_mean_err,
)
from plot_fes_compare_1x4 import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    EV_TO_KCAL,
    FE_LABEL,
    GAMMA_METAD,
    GRID_N,
    LAST_N,
    LIM_2D,
    N_BINS_1D,
    N_FINE_1D,
    P_FLOOR,
    SIGMA_METAD,
    SMOOTH_1D_SIGMA,
    T_K,
    VMAX_2D,
    XI_LIM,
    draw_fes1d,
    draw_fes2d,
    metad_fes_kcal_fft,
    nature_rcparams as abc_rcparams,
    project_1d_from_2d,
    relative_fes,
    smooth_1d,
    tag_title,
)
from plot_fes_rmse_vs_time import (  # noqa: E402
    delta_g_from_2d,
    discover_stage_files,
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
from fp_transformer_torch.kernel.gaussian import (  # noqa: E402
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)

EXP = Path(__file__).resolve().parent
OUT = EXP / "convergence_repeat3_gpu3"
FPT_DIRS = [
    EXP / "results" / "claisen_water_fpt_mh-0_g4_89108",
    EXP / "results" / "claisen_water_fpt_mh-0_g4_89109",
    EXP / "results" / "claisen_water_fpt_mh-0_g4_89110",
]
META_HILLS = [
    ROOT
    / "experiments/rearrangement_water_metadynamics/results"
    / "claisen_water_metad_mh-0_r64_90440/live/hills.npz",
    ROOT
    / "experiments/rearrangement_water_metadynamics/results"
    / "claisen_water_metad_mh-0_r64_90441/live/hills.npz",
]
US_MBAR = (
    ROOT
    / "experiments/rearrangement_water_US/results"
    / "claisen_water_us1d_mh-0_merged_87040_90364_90365/mbar/fes_mbar.npz"
)

LAM = 0.025
GROUP = 10
DT_GROUP = 1.0
VMAX_RMSE = 60.0
VMAX30 = 30.0
N_WORKERS = 12
KDE_CHUNK = 4096
KB_EV_PER_K = 8.617330337217213e-5
BETA0 = 1.0 / (KB_EV_PER_K * T_K)
N_REP = 64
DEPOSIT_PACE = 50
TIMESTEP = 0.001
METAD_STRIDE = 10
METAD_RATE = 13.34  # MetaD 1×H800, from 90440/90441 logs (~13.34 it/s)
# 4-GPU sFPT 89108/89109/89110: mean ~3.93 it/s (median 3.95). Not the 8-GPU 4.23 it/s.
SFPT_RATE = 3.93
FPT_GPUS = 4.0
METAD_GPUS = 1.0


def density_f32(grid_xy: np.ndarray, sample_cv: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    samples = torch.as_tensor(np.asarray(sample_cv, dtype=np.float32).reshape(-1, 2))
    bw = torch.as_tensor(np.asarray(bandwidth, dtype=np.float32).reshape(-1))
    n = float(samples.shape[0])
    feature_dim = 2.0
    s2 = torch.sum(samples * samples, dim=1)
    m = int(grid_xy.shape[0])
    out = np.empty((m,), dtype=np.float64)
    log_norm = gaussian_kernel_log_normalization(bw, feature_dim)
    bw2 = bandwidth_column_squared(bw)
    for start in range(0, m, KDE_CHUNK):
        stop = min(start + KDE_CHUNK, m)
        grid = torch.as_tensor(np.asarray(grid_xy[start:stop], dtype=np.float32).reshape(-1, 2))
        g2 = torch.sum(grid * grid, dim=1, keepdim=True)
        dist = torch.clamp(g2 + s2.unsqueeze(0) - 2.0 * (grid @ samples.T), min=0.0) / feature_dim
        log_k = (-0.5 * dist) / bw2 + log_norm
        log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
        out[start:stop] = torch.exp(log_p).numpy()
    return out


def kde_fes_from_stages(
    stages: list[tuple[str, str, float]],
    *,
    d1: np.ndarray,
    d2: np.ndarray,
    grid_xy: np.ndarray,
    g1_shape: tuple[int, int],
) -> np.ndarray:
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    lam = LAM
    for cv_p, bw_p, lam_i in stages:
        lam = float(lam_i) if float(lam_i) > 0 else LAM
        p_acc += density_f32(grid_xy, np.load(cv_p), np.load(bw_p))
    p = np.clip(p_acc / float(len(stages)), 1e-300, None)
    f2 = relative_fes((-np.log(p) / (BETA0 * lam) * EV_TO_KCAL).reshape(g1_shape))
    p2 = p.reshape(g1_shape)
    f2 = f2.copy()
    f2[p2 < P_FLOOR * float(np.max(p2))] = np.nan
    return f2


_W: dict = {}


def _worker_init(shared: dict) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _W.update(shared)


def _kde_job(payload: dict):
    d1 = _W["d1"]
    d2 = _W["d2"]
    f2 = kde_fes_from_stages(
        payload["stages"],
        d1=d1,
        d2=d2,
        grid_xy=_W["grid_xy"],
        g1_shape=_W["g1_shape"],
    )
    _, f1d = project_1d_fixed_edges(d1, d2, f2, _W["edges"], temperature=T_K)
    xi = _W["xi"]
    f1d_ref = _W["f1d_ref"]
    rmse = rmse_1d(f1d, f1d_ref, vmax=VMAX_RMSE, xi=xi, xi_lo=None, xi_hi=None)
    rmse30 = rmse_1d(f1d, f1d_ref, vmax=VMAX30, xi=xi, xi_lo=None, xi_hi=None)
    dG = delta_g_from_2d(payload["d1"], payload["d2"], f2, temperature=T_K, xi_cut=0.0)
    kind = payload.get("kind", "group")
    if kind == "last100":
        return "last100", payload["ir"], f2, f1d, rmse, rmse30, dG
    return (
        "group",
        payload["ir"],
        payload["ig"],
        f1d,
        rmse,
        rmse30,
        dG,
        f2 if payload.get("keep_f2") else None,
    )


def _metad_fft_setup(centers: np.ndarray, d1: np.ndarray, d2: np.ndarray):
    dx = 0.025
    pad = 1.2
    x0 = float(min(d1.min(), centers[:, 0].min()) - pad)
    x1 = float(max(d1.max(), centers[:, 0].max()) + pad)
    y0 = float(min(d2.min(), centers[:, 1].min()) - pad)
    y1 = float(max(d2.max(), centers[:, 1].max()) + pad)
    nx = int(np.round((x1 - x0) / dx)) + 1
    ny = int(np.round((y1 - y0) / dx)) + 1
    xe = x0 + dx * np.arange(nx + 1)
    ye = y0 + dx * np.arange(ny + 1)
    krad = int(np.ceil(8.0 * float(np.max(SIGMA_METAD)) / dx))
    kx = np.arange(-krad, krad + 1, dtype=np.float64) * dx
    ky = np.arange(-krad, krad + 1, dtype=np.float64) * dx
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    ker = np.exp(-0.5 * ((KX / SIGMA_METAD[0]) ** 2 + (KY / SIGMA_METAD[1]) ** 2))
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return xe, ye, ker, xc, yc


def _fes_from_hist(H, ker, xc, yc, d1, d2) -> np.ndarray:
    V = fftconvolve(H, ker, mode="same")
    interp = RegularGridInterpolator((xc, yc), V, bounds_error=False, fill_value=0.0)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    Vq = interp(np.stack([g1.ravel(), g2.ravel()], axis=-1)).reshape(g1.shape)
    scale = GAMMA_METAD / (GAMMA_METAD - 1.0)
    return relative_fes((-scale * Vq) * EV_TO_KCAL)


def incremental_metad_fft(
    hills_path: Path,
    *,
    d1: np.ndarray,
    d2: np.ndarray,
    edges: np.ndarray,
    xi: np.ndarray,
    f1d_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(hills_path)
    n = int(z["n_hills"])
    centers = np.asarray(z["centers"][:n], dtype=np.float64)
    heights = np.asarray(z["heights"][:n], dtype=np.float64).reshape(-1)
    xe, ye, ker, xc, yc = _metad_fft_setup(centers, d1, d2)
    n_dep = n // N_REP
    H = np.zeros((xe.size - 1, ye.size - 1), dtype=np.float64)
    t_list, r_list, r30_list, g_list = [], [], [], []
    for k in range(1, n_dep + 1):
        i0 = (k - 1) * N_REP
        c = centers[i0 : i0 + N_REP]
        w = heights[i0 : i0 + N_REP]
        Hk, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=[xe, ye], weights=w)
        H += Hk
        if (k % METAD_STRIDE) != 0 and k != n_dep:
            continue
        f2 = _fes_from_hist(H, ker, xc, yc, d1, d2)
        _, f1d = project_1d_fixed_edges(d1, d2, f2, edges, temperature=T_K)
        t_list.append(float(k) * DEPOSIT_PACE * TIMESTEP)
        r_list.append(rmse_1d(f1d, f1d_ref, vmax=VMAX_RMSE, xi=xi, xi_lo=None, xi_hi=None))
        r30_list.append(rmse_1d(f1d, f1d_ref, vmax=VMAX30, xi=xi, xi_lo=None, xi_hi=None))
        g_list.append(delta_g_from_2d(d1, d2, f2, temperature=T_K, xi_cut=0.0))
        if k == METAD_STRIDE or k == n_dep or k % 500 == 0:
            print(
                f"[metad-fft] {hills_path.parent.parent.name} k={k}/{n_dep}  "
                f"RMSE={r_list[-1]:.3f} ΔG={g_list[-1]:.3f}",
                flush=True,
            )
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(r30_list),
        np.asarray(g_list),
    )


def plot_last_abc(
    out: Path,
    *,
    d1: np.ndarray,
    d2: np.ndarray,
    F_sfpt: np.ndarray,
    F_meta: np.ndarray,
    xi_m,
    y_m,
    xi_s,
    y_s,
    xi_u=None,
    y_u=None,
) -> None:
    plt.rcParams.update(abc_rcparams())
    fig_w, fig_h = 11.2, 3.95
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = fig.add_axes([0, 0, 1, 1])
    ax_b = fig.add_axes([0, 0, 1, 1])
    ax_c = fig.add_axes([0, 0, 1, 1])
    cax = fig.add_axes([0, 0, 1, 1])
    left, right = 0.055, 0.985
    bottom, top = 0.175, 0.84
    gap_ab = 0.006
    cbar_pad = 0.006
    cbar_w = 0.012
    cbar_lab = 0.13
    gap_c = 0.075
    reserved = gap_ab + cbar_pad + cbar_w + cbar_lab + gap_c
    w_p = (right - left - reserved) / 3.0
    h_p = w_p * fig_w / fig_h
    h_avail = top - bottom
    if h_p > h_avail:
        scale = h_avail / h_p
        h_p *= scale
        w_p *= scale
    y0 = bottom + 0.5 * (h_avail - h_p)
    x_a = left
    x_b = x_a + w_p + gap_ab
    x_cb = x_b + w_p + cbar_pad
    x_c = x_cb + cbar_w + cbar_lab
    ax_a.set_position([x_a, y0, w_p, h_p])
    ax_b.set_position([x_b, y0, w_p, h_p])
    cax.set_position([x_cb, y0, cbar_w, h_p])
    ax_c.set_position([x_c, y0, w_p, h_p])

    im = draw_fes2d(ax_a, d1, d2, F_sfpt, vmax=VMAX_2D, title="sFPT", letter="a")
    draw_fes2d(
        ax_b, d1, d2, F_meta, vmax=VMAX_2D, title="WT-MetaD", letter="b", show_ylabel=False
    )
    draw_fes1d(ax_c, xi_m, y_m, xi_s, y_s, letter="c")
    if xi_u is not None and y_u is not None:
        ax_c.plot(xi_u, y_u, color=COLOR_US, lw=1.5, ls="--", label="US-MBAR", zorder=3)
        ax_c.legend(loc="upper right", fontsize=12, handlelength=1.5, frameon=False)

    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(r"(kcal/mol)", fontsize=14, labelpad=4)
    cb.ax.tick_params(labelsize=13, length=3.2, width=1.0)
    cb.set_ticks([0, 15, 30, 45, 55])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    cb.outline.set_linewidth(1.1)
    png = out / "fes_compare_abc.png"
    fig.savefig(png, dpi=400, bbox_inches=None, pad_inches=0.06)
    fig.savefig(out / "fes_compare_abc.pdf", bbox_inches=None, pad_inches=0.06)
    plt.close(fig)
    print(f"[ok] {png}", flush=True)


def plot_fes1d_last(out: Path, xi_m, y_m, xi_s, y_s, xi_u=None, y_u=None) -> None:
    plt.rcParams.update(abc_rcparams())
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    draw_fes1d(ax, xi_m, y_m, xi_s, y_s, letter="")
    ax.set_title("1D FES  last 100 stages")
    if xi_u is not None and y_u is not None:
        ax.plot(xi_u, y_u, color=COLOR_US, lw=1.5, ls="--", label="US-MBAR", zorder=3)
        ax.legend(loc="upper right", fontsize=12, handlelength=1.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fes_1d_last.png", dpi=250)
    fig.savefig(out / "fes_1d_last.pdf")
    plt.close(fig)
    print("[ok] fes_1d_last", flush=True)


def _persist_wallclock(
    out: Path,
    *,
    tw_sfpt: np.ndarray,
    tw_meta: np.ndarray,
    cost_sfpt: np.ndarray,
    cost_meta: np.ndarray,
) -> None:
    extra = dict(
        fpt_rate_it_s=np.float64(SFPT_RATE),
        fpt_gpus=np.float64(FPT_GPUS),
        metad_rate_it_s=np.float64(METAD_RATE),
        metad_gpus=np.float64(METAD_GPUS),
    )
    for name in ("fes_rmse_vs_time.npz", "fes_rmse_F30.npz"):
        path = out / name
        z = dict(np.load(path, allow_pickle=True))
        z["tw_kde_h"] = np.asarray(tw_sfpt, dtype=np.float64)
        z["tw_metad_h"] = np.asarray(tw_meta, dtype=np.float64)
        z["cost_kde_gpuh"] = np.asarray(cost_sfpt, dtype=np.float64)
        z["cost_metad_gpuh"] = np.asarray(cost_meta, dtype=np.float64)
        z.update(extra)
        np.savez_compressed(path, **z)


def plot_all_from_npz(out: Path) -> None:
    """Draw all gpu3 figures from npz written by run_repeat3_abc.py."""
    grids = np.load(out / "fes_compare_last100_grids.npz")
    g1d = np.load(out / "fes_1d_all_groups.npz", allow_pickle=True)
    rmse = np.load(out / "fes_rmse_vs_time.npz")
    f30 = np.load(out / "fes_rmse_F30.npz")
    d1 = np.asarray(grids["d1"])
    d2 = np.asarray(grids["d2"])
    F_sfpt = np.asarray(grids["F_sfpt"])
    F_meta = np.asarray(grids["F_metad"])
    xi = np.asarray(g1d["xi"])
    f1d_meta = np.asarray(g1d["F1d_metad"])
    f1d_meta_std = np.asarray(g1d["F1d_metad_std"])
    if "F1d_metad_t" in g1d.files:
        f1d_meta = np.asarray(g1d["F1d_metad_t"])
        f1d_meta_std = np.asarray(g1d["F1d_metad_t_std"])
    f1d_ref = np.asarray(g1d["F1d_us"])
    f_sfpt_mean = np.asarray(g1d["F1d_kde"])
    f_sfpt_std = np.asarray(g1d["F1d_kde_std"])
    titles = [str(t) for t in g1d["titles"]]
    t_sfpt = np.asarray(rmse["t_kde_ps"])
    t_meta = np.asarray(rmse["t_metad_ps"])
    rmse_meta = np.asarray(rmse["rmse_metad"])
    rmse_meta_std = np.asarray(rmse["rmse_metad_std"])
    rmse_sfpt = np.asarray(rmse["rmse_kde"])
    rmse_sfpt_std = np.asarray(rmse["rmse_kde_std"])
    dG_us = float(rmse["dG_us_ref"])
    dG_meta_t = np.asarray(rmse["dG_metad"])
    dG_meta_std = np.asarray(rmse["dG_metad_std"])
    dG_sfpt = np.asarray(rmse["dG_kde"])
    dG_sfpt_std = np.asarray(rmse["dG_kde_std"])
    rmse30_meta = np.asarray(f30["rmse_metad"])
    rmse30_meta_std = np.asarray(f30["rmse_metad_std"])
    rmse30_sfpt = np.asarray(f30["rmse_kde"])
    rmse30_sfpt_std = np.asarray(f30["rmse_kde_std"])
    grids.close()
    g1d.close()
    rmse.close()
    f30.close()
    # Wall-clock from sim time / constant job throughput (4-GPU sFPT, 1-GPU MetaD).
    tw_sfpt = (t_sfpt / TIMESTEP) / SFPT_RATE / 3600.0
    tw_meta = (t_meta / TIMESTEP) / METAD_RATE / 3600.0
    cost_sfpt = tw_sfpt * FPT_GPUS
    cost_meta = tw_meta * METAD_GPUS
    _persist_wallclock(out, tw_sfpt=tw_sfpt, tw_meta=tw_meta, cost_sfpt=cost_sfpt, cost_meta=cost_meta)

    xi_s0, f1d_s = project_1d_from_2d(
        d1, d2, F_sfpt, d1_lim=LIM_2D, d2_lim=LIM_2D, n_bins=N_BINS_1D
    )
    xi_m0, f1d_m = project_1d_from_2d(
        d1, d2, F_meta, d1_lim=LIM_2D, d2_lim=LIM_2D, n_bins=N_BINS_1D
    )
    xi_s, y_s = smooth_1d(xi_s0, f1d_s, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D)
    xi_m, y_m = smooth_1d(xi_m0, f1d_m, sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D)
    m_us = np.isfinite(f1d_ref)
    xi_u = y_u = None
    if np.any(m_us):
        xi_u, y_u = smooth_1d(
            xi[m_us], f1d_ref[m_us], sigma=SMOOTH_1D_SIGMA, n_fine=N_FINE_1D
        )
        y_u = np.minimum(y_u, VMAX_2D)

    plot_last_abc(
        out,
        d1=d1,
        d2=d2,
        F_sfpt=F_sfpt,
        F_meta=F_meta,
        xi_m=xi_m,
        y_m=y_m,
        xi_s=xi_s,
        y_s=y_s,
        xi_u=xi_u,
        y_u=y_u,
    )
    plot_fes1d_last(out, xi_m, y_m, xi_s, y_s, xi_u=xi_u, y_u=y_u)
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
        vmax=float(VMAX_2D),
        ncols=4,
    )
    plot_rmse_suite(
        out,
        stem_prefix="fes_rmse",
        ylab=r"1D FES RMSE ($d_1,d_2\in[1,7]$) / kcal mol$^{-1}$",
        t_meta=t_meta,
        tw_meta=tw_meta,
        cost_meta=cost_meta,
        rmse_meta=rmse_meta,
        rmse_meta_std=rmse_meta_std,
        t_sfpt=t_sfpt,
        tw_sfpt=tw_sfpt,
        cost_sfpt=cost_sfpt,
        rmse_sfpt=rmse_sfpt,
        rmse_sfpt_std=rmse_sfpt_std,
        fpt_gpus=4,
        metad_gpus=1,
    )
    plot_rmse_suite(
        out,
        stem_prefix="fes_rmse_F30",
        ylab=r"1D FES RMSE ($F\leq 30$, $d_1,d_2\in[1,7]$) / kcal mol$^{-1}$",
        t_meta=t_meta,
        tw_meta=tw_meta,
        cost_meta=cost_meta,
        rmse_meta=rmse30_meta,
        rmse_meta_std=rmse30_meta_std,
        t_sfpt=t_sfpt,
        tw_sfpt=tw_sfpt,
        cost_sfpt=cost_sfpt,
        rmse_sfpt=rmse30_sfpt,
        rmse_sfpt_std=rmse30_sfpt_std,
        fpt_gpus=4,
        metad_gpus=1,
    )
    ylab_dg = r"$\Delta G=k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$"
    for stem, ylim, xm, xs, xlab in [
        ("deltaG_vs_time", None, t_meta, t_sfpt, r"$t$ / ps"),
        ("deltaG_vs_time_divein", (-20.0, 0.0), t_meta, t_sfpt, r"$t$ / ps"),
        ("deltaG_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h"),
        ("deltaG_vs_wallclock_divein", (-20.0, 0.0), tw_meta, tw_sfpt, r"wall-clock time / h"),
    ]:
        plot_series_mean_err(
            out,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab_dg,
            t_meta=xm,
            y_meta=dG_meta_t,
            e_meta=dG_meta_std,
            t_sfpt=xs,
            y_sfpt=dG_sfpt,
            e_sfpt=dG_sfpt_std,
            dG_ref=dG_us,
            ref_label="US",
            ylim=ylim,
        )
    print(f"[done] {out}", flush=True)


def main() -> None:
    plot_all_from_npz(OUT)


if __name__ == "__main__":
    main()
