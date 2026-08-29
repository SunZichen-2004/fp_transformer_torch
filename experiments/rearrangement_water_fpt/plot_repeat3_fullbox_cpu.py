#!/usr/bin/env python3
"""Login-node full-(d1,d2) RMSE/ΔG for 89108/89109/89110 (no [1,4] crop, no ξ window)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from plot_convergence_analysis import (  # noqa: E402
    mean_std,
    plot_rmse_suite,
    plot_series_mean_err,
)
from plot_fes_compare import ev_to_kcal, relative_fes  # noqa: E402
from plot_fes_rmse_vs_time import (  # noqa: E402
    delta_g_from_2d,
    discover_stage_files,
    make_1d_edges_from_grid,
    project_1d_fixed_edges,
    rmse_1d,
)
from plot_g5_1d_grid_and_dg_zoom import (  # noqa: E402
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
META_DIRS = [
    ROOT / "experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_87059/live",
    ROOT / "experiments/rearrangement_water_metadynamics/results/claisen_water_metad_mh-0_r64_87060/live",
]
US_MBAR = (
    ROOT
    / "experiments/rearrangement_water_US/results"
    / "claisen_water_us1d_mh-0_r64_g1_87040/mbar/fes_mbar.npz"
)

BOUNDS = (1.0, 8.5, 1.0, 8.5)
GRID_N = 72
N_BINS = 80
GROUP = 10
DT_GROUP = 1.0
T_K = 300.0
VMAX = 60.0
VMAX30 = 30.0
N_WORKERS = 32
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
BETA0 = 1.0 / (KB_EV_PER_K * T_K)


def density_f32(grid_xy: np.ndarray, sample_cv: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    samples = torch.as_tensor(np.asarray(sample_cv, dtype=np.float32).reshape(-1, 2))
    bw = torch.as_tensor(np.asarray(bandwidth, dtype=np.float32).reshape(-1))
    grid = torch.as_tensor(np.asarray(grid_xy, dtype=np.float32).reshape(-1, 2))
    n = float(samples.shape[0])
    feature_dim = 2.0
    g2 = torch.sum(grid * grid, dim=1, keepdim=True)
    s2 = torch.sum(samples * samples, dim=1)
    dist = torch.clamp(g2 + s2.unsqueeze(0) - 2.0 * (grid @ samples.T), min=0.0) / feature_dim
    log_k = (-0.5 * dist) / bandwidth_column_squared(bw) + gaussian_kernel_log_normalization(
        bw, feature_dim
    )
    log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
    return torch.exp(log_p).numpy().astype(np.float64)


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _kde_group_job(payload: dict) -> tuple[int, int, np.ndarray, float, float, float]:
    _worker_init()
    d1 = payload["d1"]
    d2 = payload["d2"]
    grid_xy = payload["grid_xy"]
    edges = payload["edges"]
    xi = payload["xi"]
    f1d_ref = payload["f1d_ref"]
    g1_shape = payload["g1_shape"]
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    lam = None
    n_st = 0
    for cv_p, bw_p, lam_i in payload["stages"]:
        lam = float(lam_i)
        p_acc += density_f32(grid_xy, np.load(cv_p), np.load(bw_p))
        n_st += 1
    p = np.clip(p_acc / float(n_st), 1e-300, None)
    f2 = relative_fes((-np.log(p) / (BETA0 * lam) * EV_TO_KCAL).reshape(g1_shape), None)
    _, f1d = project_1d_fixed_edges(d1, d2, f2, edges, temperature=T_K)
    rmse = rmse_1d(f1d, f1d_ref, vmax=VMAX, xi=xi, xi_lo=None, xi_hi=None)
    rmse30 = rmse_1d(f1d, f1d_ref, vmax=VMAX30, xi=xi, xi_lo=None, xi_hi=None)
    dG = delta_g_from_2d(d1, d2, f2, temperature=T_K, xi_cut=0.0)
    return payload["ir"], payload["ig"], f1d, rmse, rmse30, dG


def metad_one(
    hills_path: Path,
    *,
    d1: np.ndarray,
    d2: np.ndarray,
    edges: np.ndarray,
    xi: np.ndarray,
    f1d_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hills = np.load(hills_path)
    n_hills = int(hills["n_hills"]) if "n_hills" in hills.files else int(hills["centers"].shape[0])
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float64)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    scale = 30.0 / 29.0
    sig = np.array([0.2, 0.2], dtype=np.float64)
    n_rep = 64
    n_dep = n_hills // n_rep
    stride = 10
    v = np.zeros(grid_xy.shape[0], dtype=np.float64)
    t_list, r_list, r30_list, g_list = [], [], [], []
    for k in range(1, n_dep + 1):
        i0 = (k - 1) * n_rep
        c = centers[i0 : i0 + n_rep]
        w = heights[i0 : i0 + n_rep]
        diff = (grid_xy[:, None, :] - c[None, :, :]) / sig.reshape(1, 1, -1)
        v += np.exp(-0.5 * np.sum(diff * diff, axis=-1)) @ w
        if (k % stride) != 0 and k != n_dep:
            continue
        f2 = relative_fes((-scale * v).reshape(g1.shape) * EV_TO_KCAL, None)
        _, f1d = project_1d_fixed_edges(d1, d2, f2, edges, temperature=T_K)
        t_list.append(float(k) * 50 * 0.001)
        r_list.append(rmse_1d(f1d, f1d_ref, vmax=VMAX, xi=xi, xi_lo=None, xi_hi=None))
        r30_list.append(rmse_1d(f1d, f1d_ref, vmax=VMAX30, xi=xi, xi_lo=None, xi_hi=None))
        g_list.append(delta_g_from_2d(d1, d2, f2, temperature=T_K, xi_cut=0.0))
        if k == 1 or k == n_dep or k % 500 == 0:
            print(
                f"[metad] {hills_path.parent.parent.name} k={k}/{n_dep}  "
                f"RMSE={r_list[-1]:.3f} ΔG={g_list[-1]:.3f}",
                flush=True,
            )
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(r30_list),
        np.asarray(g_list),
    )


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = np.linspace(BOUNDS[0], BOUNDS[1], GRID_N)
    d2 = np.linspace(BOUNDS[2], BOUNDS[3], GRID_N)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    edges = make_1d_edges_from_grid(d1, d2, N_BINS)
    xi = 0.5 * (edges[:-1] + edges[1:])
    print(
        f"[fullbox] d1∈[{BOUNDS[0]},{BOUNDS[1]}] d2∈[{BOUNDS[2]},{BOUNDS[3]}] "
        f"grid={GRID_N}  ξ∈[{xi.min():.2f},{xi.max():.2f}]  no ξ window",
        flush=True,
    )

    dG_us, _, _ = us_delta_g_ref(
        US_MBAR, temperature=T_K, xi_cut=0.0, xi_lo=None, xi_hi=None, xi_query=None
    )
    _, f_us = load_us_mbar_1d_d2_minus_d1(US_MBAR, xi_query=xi)
    f1d_ref = relative_fes(f_us, None)
    print(f"[us] ΔG full 1D={dG_us:.4f}", flush=True)

    rows_all = [discover_stage_files(d) for d in FPT_DIRS]
    n_groups = min(len(r) // GROUP for r in rows_all)
    t_sfpt = (np.arange(n_groups) + 1) * DT_GROUP
    print(f"[sFPT] groups={n_groups}  stages={[len(r) for r in rows_all]}", flush=True)

    jobs = []
    for ir, rows in enumerate(rows_all):
        for ig in range(n_groups):
            chunk = rows[ig * GROUP : (ig + 1) * GROUP]
            jobs.append(
                {
                    "ir": ir,
                    "ig": ig,
                    "d1": d1,
                    "d2": d2,
                    "grid_xy": grid_xy,
                    "edges": edges,
                    "xi": xi,
                    "f1d_ref": f1d_ref,
                    "g1_shape": g1.shape,
                    "stages": [(str(cv), str(bw), lam) for _, cv, bw, lam in chunk],
                }
            )

    f_stack = np.empty((len(FPT_DIRS), n_groups, xi.size), dtype=np.float64)
    r_stack = np.empty((len(FPT_DIRS), n_groups), dtype=np.float64)
    r30_stack = np.empty((len(FPT_DIRS), n_groups), dtype=np.float64)
    g_stack = np.empty((len(FPT_DIRS), n_groups), dtype=np.float64)
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        for ir, ig, f1d, rmse, rmse30, dG in ex.map(_kde_group_job, jobs, chunksize=2):
            f_stack[ir, ig] = f1d
            r_stack[ir, ig] = rmse
            r30_stack[ir, ig] = rmse30
            g_stack[ir, ig] = dG
            done += 1
            if done == 1 or done % 20 == 0 or done == len(jobs):
                print(
                    f"[kde] {done}/{len(jobs)}  last RMSE={rmse:.3f} ΔG={dG:.3f}",
                    flush=True,
                )

    f_sfpt_mean, f_sfpt_std = mean_std(f_stack, axis=0)
    rmse_sfpt, rmse_sfpt_std = mean_std(r_stack)
    rmse30_sfpt, rmse30_sfpt_std = mean_std(r30_stack)
    dG_sfpt, dG_sfpt_std = mean_std(g_stack)
    print(
        f"[sFPT] last RMSE={rmse_sfpt[-1]:.4f}±{rmse_sfpt_std[-1]:.4f}  "
        f"ΔG={dG_sfpt[-1]:.4f}±{dG_sfpt_std[-1]:.4f}",
        flush=True,
    )

    t_m_list, rmse_m_list, rmse30_m_list, dG_m_list, f_meta_final = [], [], [], [], []
    for md in META_DIRS:
        t_m, rmse_m, rmse_m30, dG_m = metad_one(
            md / "hills.npz", d1=d1, d2=d2, edges=edges, xi=xi, f1d_ref=f1d_ref
        )
        t_m_list.append(t_m)
        rmse_m_list.append(rmse_m)
        rmse30_m_list.append(rmse_m30)
        dG_m_list.append(dG_m)
        fes = np.load(md / "fes.npz")
        d1m = np.asarray(fes["d1"], dtype=np.float64)
        d2m = np.asarray(fes["d2"], dtype=np.float64)
        f_full = relative_fes(ev_to_kcal(np.asarray(fes["F"], dtype=np.float64)), None)
        if f_full.shape == (len(d1m), len(d2m)):
            f_full = f_full.T
        _, f1d_m = project_1d_fixed_edges(d1m, d2m, f_full, edges, temperature=T_K)
        f_meta_final.append(f1d_m)
        print(f"[metad] {md.parent.name} t→{t_m[-1]:.1f} ps  ΔG_end={dG_m[-1]:.4f}", flush=True)

    n_m = min(len(t) for t in t_m_list)
    t_meta = t_m_list[0][:n_m]
    rmse_meta, rmse_meta_std = mean_std(np.stack([r[:n_m] for r in rmse_m_list]))
    rmse30_meta, rmse30_meta_std = mean_std(np.stack([r[:n_m] for r in rmse30_m_list]))
    dG_meta_t, dG_meta_std = mean_std(np.stack([g[:n_m] for g in dG_m_list]))
    f1d_meta, f1d_meta_std = mean_std(np.stack(f_meta_final), axis=0)
    dG_meta = float(dG_meta_t[-1])
    print(f"[metad] mean ΔG_end={dG_meta:.4f}  US={dG_us:.4f}", flush=True)

    tw_sfpt = fpt_wall_hours_from_throughput(t_sfpt, timestep_ps=0.001, rate_it_s=3.95)
    tw_meta = (t_meta / 0.001) / 13.7 / 3600.0
    cost_sfpt = tw_sfpt * 4.0
    cost_meta = tw_meta * 1.0

    np.savez_compressed(
        OUT / "fes_rmse_vs_time.npz",
        t_metad_ps=t_meta,
        rmse_metad=rmse_meta,
        rmse_metad_std=rmse_meta_std,
        dG_metad=dG_meta_t,
        dG_metad_std=dG_meta_std,
        dG_metad_ref=dG_meta,
        dG_us_ref=dG_us,
        xi=xi,
        F1d_metad_ref=f1d_meta,
        F1d_us_ref=f1d_ref,
        t_kde_ps=t_sfpt,
        rmse_kde=rmse_sfpt,
        rmse_kde_std=rmse_sfpt_std,
        dG_kde=dG_sfpt,
        dG_kde_std=dG_sfpt_std,
        tw_kde_h=tw_sfpt,
        tw_metad_h=tw_meta,
        cost_kde_gpuh=cost_sfpt,
        cost_metad_gpuh=cost_meta,
        bounds=np.asarray(BOUNDS),
        xi_window="none",
    )
    titles = [f"t={t:g} ps" for t in t_sfpt]
    np.savez_compressed(
        OUT / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad=f1d_meta,
        F1d_metad_std=f1d_meta_std,
        F1d_us=f1d_ref,
        F1d_kde=f_sfpt_mean,
        F1d_kde_std=f_sfpt_std,
        F1d_kde_runs=f_stack,
        t_ps=t_sfpt,
        titles=np.asarray(titles),
    )

    ylab_rmse = r"1D FES RMSE (full $d_1,d_2$) / kcal mol$^{-1}$"
    plot_rmse_suite(
        OUT,
        stem_prefix="fes_rmse",
        ylab=ylab_rmse,
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
    np.savez_compressed(
        OUT / "fes_rmse_F30.npz",
        t_metad_ps=t_meta,
        t_kde_ps=t_sfpt,
        tw_metad_h=tw_meta,
        tw_kde_h=tw_sfpt,
        cost_metad_gpuh=cost_meta,
        cost_kde_gpuh=cost_sfpt,
        rmse_metad=rmse30_meta,
        rmse_metad_std=rmse30_meta_std,
        rmse_kde=rmse30_sfpt,
        rmse_kde_std=rmse30_sfpt_std,
    )
    plot_rmse_suite(
        OUT,
        stem_prefix="fes_rmse_F30",
        ylab=r"1D FES RMSE ($F\leq 30$, full $d_1,d_2$) / kcal mol$^{-1}$",
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
            OUT,
            stem=stem,
            xlabel=xlab,
            ylabel=ylab_dg,
            t_meta=xm,
            y_meta=dG_meta_t,
            e_meta=dG_meta_std,
            t_sfpt=xs,
            y_sfpt=dG_sfpt,
            e_sfpt=dG_sfpt_std,
            dG_ref=float(dG_us),
            ref_label="US",
            ylim=ylim,
        )

    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
