#!/usr/bin/env python3
"""Light spawn entry: KDE workers only import this file + repeat3_abc_worker."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

from repeat3_abc_worker import kde_job, worker_init

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

LIM = 1.0, 7.0
GRID_N = 140
N_BINS_1D = 140
LAST_N = 100
GROUP = 10
DT_GROUP = 1.0
T_K = 300.0
LAM = 0.025
N_WORKERS = 12
KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
SIGMA_METAD = np.array([0.2, 0.2], dtype=np.float64)
GAMMA_METAD = 30.0
N_REP = 64
DEPOSIT_PACE = 50
TIMESTEP = 0.001
METAD_STRIDE = 10
METAD_RATE = 13.34  # MetaD 1×H800, from 90440/90441 logs (~13.34 it/s)
# 4-GPU sFPT 89108/89109/89110: mean ~3.93 it/s (not 8-GPU 4.23).
SFPT_RATE = 3.93
STAGE_RE = re.compile(r"stage(\d+)", re.I)
LAMBDA_RE = re.compile(r"lambda(\d+)p(\d+)", re.I)


def relative_fes(f: np.ndarray) -> np.ndarray:
    out = np.asarray(f, dtype=np.float64)
    return out - np.nanmin(out[np.isfinite(out)])


def mean_std(stack: np.ndarray, axis: int = 0):
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    n = stack.shape[axis]
    if n <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


def discover_stage_files(data_dir: Path):
    rows = []
    for cv in sorted(data_dir.glob("stage[0-9]*_lambda*_cv.npy")):
        sm = STAGE_RE.search(cv.name)
        lm = LAMBDA_RE.search(cv.name)
        if sm is None or lm is None:
            continue
        bw = cv.with_name(cv.name.replace("_cv.npy", "_bandwidth.npy"))
        if not bw.is_file():
            continue
        rows.append((int(sm.group(1)), cv, bw, float(f"{lm.group(1)}.{lm.group(2)}")))
    rows.sort(key=lambda r: r[0])
    return rows


def load_us_1d(path: Path, xi_query: np.ndarray):
    z = np.load(path)
    xi_us = np.asarray(z["xi"], dtype=np.float64)
    f_us = np.asarray(z["F_kcal"], dtype=np.float64)
    ok = np.isfinite(xi_us) & np.isfinite(f_us)
    xi_us, f_us = xi_us[ok], f_us[ok]
    xi = -xi_us[::-1]
    f = relative_fes(f_us[::-1].copy())
    fq = np.interp(np.asarray(xi_query, dtype=np.float64), xi, f, left=np.nan, right=np.nan)
    return fq, xi, f


def us_delta_g(xi, f, *, temperature=T_K, xi_cut=0.0) -> float:
    kt = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kt
    m = np.isfinite(xi) & np.isfinite(f)
    xi_a, f = np.asarray(xi)[m], np.asarray(f)[m]
    b1 = xi_a < float(xi_cut)
    b2 = xi_a > float(xi_cut)
    if not np.any(b1) or not np.any(b2):
        return float("nan")

    def _logZ(sel):
        vals = -beta * f[sel]
        mval = float(np.max(vals))
        return mval + float(np.log(np.sum(np.exp(vals - mval))))

    return float(kt * (_logZ(b1) - _logZ(b2)))


def project_1d_fixed_edges(d1, d2, f_kcal, edges, *, temperature=T_K):
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    xi = (g2 - g1).ravel()
    f = np.asarray(f_kcal, dtype=np.float64).ravel()
    centers = 0.5 * (edges[:-1] + edges[1:])
    kt = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kt
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            sel = (xi >= edges[i]) & (xi < edges[i + 1])
        else:
            sel = (xi >= edges[i]) & (xi <= edges[i + 1])
        sel = sel & np.isfinite(f)
        if not np.any(sel):
            continue
        vals = -beta * f[sel]
        m = float(np.max(vals))
        out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
    finite = np.isfinite(out)
    if np.any(finite):
        out = out - np.nanmin(out[finite])
    return centers, out


def delta_g_from_2d(d1, d2, f_kcal, *, temperature=T_K, xi_cut=0.0) -> float:
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    f = np.asarray(f_kcal, dtype=np.float64)
    kt = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kt
    xi = g2 - g1
    msk = np.isfinite(f)
    b1 = msk & (xi < float(xi_cut))
    b2 = msk & (xi > float(xi_cut))
    if not np.any(b1) or not np.any(b2):
        return float("nan")

    def _logZ(sel):
        vals = -beta * f[sel]
        mval = float(np.max(vals))
        return mval + float(np.log(np.sum(np.exp(vals - mval))))

    return float(kt * (_logZ(b1) - _logZ(b2)))


def rmse_1d(f_a, f_b, *, vmax) -> float:
    a = np.asarray(f_a, dtype=np.float64)
    b = np.asarray(f_b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if vmax is not None:
        m = m & (b <= float(vmax))
    if not np.any(m):
        return float("nan")
    aa = a[m] - np.min(a[m])
    bb = b[m] - np.min(b[m])
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def metad_fft_setup(centers, d1, d2):
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


def fes_from_hist(H, ker, xc, yc, d1, d2):
    V = fftconvolve(H, ker, mode="same")
    interp = RegularGridInterpolator((xc, yc), V, bounds_error=False, fill_value=0.0)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    Vq = interp(np.stack([g1.ravel(), g2.ravel()], axis=-1)).reshape(g1.shape)
    scale = GAMMA_METAD / (GAMMA_METAD - 1.0)
    return relative_fes((-scale * Vq) * EV_TO_KCAL)


def metad_fes_final(hills_path: Path, d1, d2):
    z = np.load(hills_path)
    n = int(z["n_hills"])
    centers = np.asarray(z["centers"][:n], dtype=np.float64)
    heights = np.asarray(z["heights"][:n], dtype=np.float64).reshape(-1)
    xe, ye, ker, xc, yc = metad_fft_setup(centers, d1, d2)
    H, _, _ = np.histogram2d(centers[:, 0], centers[:, 1], bins=[xe, ye], weights=heights)
    return fes_from_hist(H, ker, xc, yc, d1, d2)


def incremental_metad_fft(hills_path: Path, *, d1, d2, edges, f1d_ref):
    z = np.load(hills_path)
    n = int(z["n_hills"])
    centers = np.asarray(z["centers"][:n], dtype=np.float64)
    heights = np.asarray(z["heights"][:n], dtype=np.float64).reshape(-1)
    xe, ye, ker, xc, yc = metad_fft_setup(centers, d1, d2)
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
        f2 = fes_from_hist(H, ker, xc, yc, d1, d2)
        _, f1d = project_1d_fixed_edges(d1, d2, f2, edges, temperature=T_K)
        t_list.append(float(k) * DEPOSIT_PACE * TIMESTEP)
        r_list.append(rmse_1d(f1d, f1d_ref, vmax=60.0))
        r30_list.append(rmse_1d(f1d, f1d_ref, vmax=30.0))
        g_list.append(delta_g_from_2d(d1, d2, f2, temperature=T_K, xi_cut=0.0))
        if k == METAD_STRIDE or k == n_dep or k % 500 == 0:
            print(
                f"[metad-fft] {hills_path.stem} k={k}/{n_dep}  "
                f"RMSE={r_list[-1]:.3f} ΔG={g_list[-1]:.3f}",
                flush=True,
            )
    return tuple(np.asarray(x) for x in (t_list, r_list, r30_list, g_list))


def snapshot_hills(src_paths, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in src_paths:
        dest = dest_dir / f"{p.parent.parent.name}_hills.npz"
        shutil.copy2(p, dest)
        z = np.load(dest)
        n = int(z["n_hills"])
        print(f"[hills] {p.parent.parent.name} n_hills={n} -> {dest.name}", flush=True)
        out.append(dest)
    return out


def load_sfpt_from_npz(out: Path) -> dict:
    g1d = np.load(out / "fes_1d_all_groups.npz", allow_pickle=True)
    rmse = np.load(out / "fes_rmse_vs_time.npz")
    f30 = np.load(out / "fes_rmse_F30.npz")
    grids = np.load(out / "fes_compare_last100_grids.npz")
    rec = {
        "t_sfpt": np.asarray(rmse["t_kde_ps"]),
        "rmse_sfpt": np.asarray(rmse["rmse_kde"]),
        "rmse_sfpt_std": np.asarray(rmse["rmse_kde_std"]),
        "dG_sfpt": np.asarray(rmse["dG_kde"]),
        "dG_sfpt_std": np.asarray(rmse["dG_kde_std"]),
        "rmse30_sfpt": np.asarray(f30["rmse_kde"]),
        "rmse30_sfpt_std": np.asarray(f30["rmse_kde_std"]),
        "f_sfpt_mean": np.asarray(g1d["F1d_kde"]),
        "f_sfpt_std": np.asarray(g1d["F1d_kde_std"]),
        "f_stack": np.asarray(g1d["F1d_kde_runs"]),
        "F_sfpt": np.asarray(grids["F_sfpt"]),
        "xi_us_full": np.asarray(g1d["f_us_full_xi"]) if "f_us_full_xi" in g1d.files else None,
        "f_us_full": np.asarray(g1d["f_us_full"]) if "f_us_full" in g1d.files else None,
    }
    g1d.close()
    rmse.close()
    f30.close()
    grids.close()
    rec["titles"] = [f"t={t:g} ps" for t in rec["t_sfpt"]]
    print(
        f"[sFPT] reuse {out}  groups={len(rec['t_sfpt'])}  "
        f"ΔG={rec['dG_sfpt'][-1]:.4f}±{rec['dG_sfpt_std'][-1]:.4f}",
        flush=True,
    )
    return rec


def _incremental_one(payload):
    hp, d1, d2, edges, f1d_ref = payload
    return str(hp), incremental_metad_fft(
        Path(hp), d1=d1, d2=d2, edges=edges, f1d_ref=f1d_ref
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metad-only",
        action="store_true",
        help="reuse existing sFPT npz; only recompute MetaD from hills and replot",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d1 = np.linspace(LIM[0], LIM[1], GRID_N)
    d2 = np.linspace(LIM[0], LIM[1], GRID_N)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    xi_all = (g2 - g1).ravel()
    edges = np.linspace(float(xi_all.min()), float(xi_all.max()), N_BINS_1D + 1)
    xi = 0.5 * (edges[:-1] + edges[1:])
    print(
        f"[abc] d1,d2∈{LIM} grid={GRID_N}  ξ∈[{xi.min():.2f},{xi.max():.2f}]  "
        f"MetaD=hills FFT  workers={N_WORKERS}  metad_only={args.metad_only}",
        flush=True,
    )

    f_us, xi_us_full, f_us_full = load_us_1d(US_MBAR, xi)
    f1d_ref = relative_fes(f_us)
    dG_us = us_delta_g(xi_us_full, f_us_full)
    print(f"[us] ΔG={dG_us:.4f}", flush=True)

    if args.metad_only:
        cache = load_sfpt_from_npz(OUT)
        t_sfpt = cache["t_sfpt"]
        rmse_sfpt = cache["rmse_sfpt"]
        rmse_sfpt_std = cache["rmse_sfpt_std"]
        dG_sfpt = cache["dG_sfpt"]
        dG_sfpt_std = cache["dG_sfpt_std"]
        rmse30_sfpt = cache["rmse30_sfpt"]
        rmse30_sfpt_std = cache["rmse30_sfpt_std"]
        f_sfpt_mean = cache["f_sfpt_mean"]
        f_sfpt_std = cache["f_sfpt_std"]
        f_stack = cache["f_stack"]
        F_sfpt = cache["F_sfpt"]
        titles = cache["titles"]
        if cache["xi_us_full"] is not None:
            xi_us_full = cache["xi_us_full"]
            f_us_full = cache["f_us_full"]
    else:
        rows_all = [discover_stage_files(d) for d in FPT_DIRS]
        n_groups = min(len(r) // GROUP for r in rows_all)
        t_sfpt = (np.arange(n_groups) + 1) * DT_GROUP
        print(f"[sFPT] groups={n_groups}  stages={[len(r) for r in rows_all]}", flush=True)

        shared = {
            "d1": d1,
            "d2": d2,
            "grid_xy": grid_xy,
            "g1_shape": g1.shape,
            "edges": edges,
            "xi": xi,
            "f1d_ref": f1d_ref,
        }
        jobs = []
        for ir, rows in enumerate(rows_all):
            for ig in range(n_groups):
                chunk = rows[ig * GROUP : (ig + 1) * GROUP]
                jobs.append(
                    {
                        "kind": "group",
                        "ir": ir,
                        "ig": ig,
                        "keep_f2": False,
                        "stages": [(str(cv), str(bw), lam) for _, cv, bw, lam in chunk],
                    }
                )
            last = rows[-LAST_N:]
            jobs.append(
                {
                    "kind": "last100",
                    "ir": ir,
                    "stages": [(str(cv), str(bw), lam) for _, cv, bw, lam in last],
                }
            )

        f_stack = np.empty((len(FPT_DIRS), n_groups, xi.size), dtype=np.float64)
        r_stack = np.empty((len(FPT_DIRS), n_groups), dtype=np.float64)
        r30_stack = np.empty((len(FPT_DIRS), n_groups), dtype=np.float64)
        g_stack = np.empty((len(FPT_DIRS), n_groups), dtype=np.float64)
        F_last = [None] * len(FPT_DIRS)
        done = 0
        with ProcessPoolExecutor(
            max_workers=N_WORKERS, initializer=worker_init, initargs=(shared,)
        ) as ex:
            for rec in ex.map(kde_job, jobs, chunksize=1):
                done += 1
                if rec[0] == "last100":
                    _, ir, f2, f1d, rmse, rmse30, dG = rec
                    F_last[ir] = f2
                    print(
                        f"[kde] {done}/{len(jobs)} last100 run{ir}  RMSE={rmse:.3f} ΔG={dG:.3f}",
                        flush=True,
                    )
                    continue
                _, ir, ig, f1d, rmse, rmse30, dG, _ = rec
                f_stack[ir, ig] = f1d
                r_stack[ir, ig] = rmse
                r30_stack[ir, ig] = rmse30
                g_stack[ir, ig] = dG
                if done == 1 or done % 20 == 0 or done == len(jobs):
                    print(
                        f"[kde] {done}/{len(jobs)}  last RMSE={rmse:.3f} ΔG={dG:.3f}",
                        flush=True,
                    )

        f_sfpt_mean, f_sfpt_std = mean_std(f_stack, axis=0)
        rmse_sfpt, rmse_sfpt_std = mean_std(r_stack)
        rmse30_sfpt, rmse30_sfpt_std = mean_std(r30_stack)
        dG_sfpt, dG_sfpt_std = mean_std(g_stack)
        F_sfpt = relative_fes(np.nanmean(np.stack(F_last, axis=0), axis=0))
        print(
            f"[sFPT] last RMSE={rmse_sfpt[-1]:.4f}±{rmse_sfpt_std[-1]:.4f}  "
            f"ΔG={dG_sfpt[-1]:.4f}±{dG_sfpt_std[-1]:.4f}",
            flush=True,
        )

    hills = snapshot_hills(META_HILLS, OUT / "_hills_snap")
    f_meta_list = [metad_fes_final(p, d1, d2) for p in hills]
    F_meta = relative_fes(np.nanmean(np.stack(f_meta_list, axis=0), axis=0))
    f1d_meta_runs = []
    for f2 in f_meta_list:
        _, f1d_m = project_1d_fixed_edges(d1, d2, f2, edges, temperature=T_K)
        f1d_meta_runs.append(f1d_m)
    f1d_meta, f1d_meta_std = mean_std(np.stack(f1d_meta_runs, axis=0), axis=0)
    f1d_meta = relative_fes(f1d_meta)

    t_m_list, rmse_m_list, rmse30_m_list, dG_m_list = [], [], [], []
    payloads = [(str(hp), d1, d2, edges, f1d_ref) for hp in hills]
    with ProcessPoolExecutor(max_workers=min(2, len(payloads))) as ex:
        for hp_s, (t_m, rmse_m, rmse_m30, dG_m) in ex.map(_incremental_one, payloads):
            t_m_list.append(t_m)
            rmse_m_list.append(rmse_m)
            rmse30_m_list.append(rmse_m30)
            dG_m_list.append(dG_m)
            print(
                f"[metad] {Path(hp_s).stem} t→{t_m[-1]:.1f} ps  ΔG_end={dG_m[-1]:.4f}",
                flush=True,
            )
    n_m = min(len(t) for t in t_m_list)
    t_meta = t_m_list[0][:n_m]
    rmse_meta, rmse_meta_std = mean_std(np.stack([r[:n_m] for r in rmse_m_list]))
    rmse30_meta, rmse30_meta_std = mean_std(np.stack([r[:n_m] for r in rmse30_m_list]))
    dG_meta_t, dG_meta_std = mean_std(np.stack([g[:n_m] for g in dG_m_list]))
    dG_meta = float(dG_meta_t[-1])
    print(f"[metad] mean ΔG_end={dG_meta:.4f}  US={dG_us:.4f}", flush=True)

    tw_sfpt = (t_sfpt / TIMESTEP) / SFPT_RATE / 3600.0
    tw_meta = (t_meta / TIMESTEP) / METAD_RATE / 3600.0
    cost_sfpt = tw_sfpt * 4.0
    cost_meta = tw_meta * 1.0
    titles = [f"t={t:g} ps" for t in t_sfpt]

    np.savez_compressed(
        OUT / "fes_compare_last100_grids.npz",
        d1=d1,
        d2=d2,
        F_sfpt=F_sfpt,
        F_metad=F_meta,
        n_stages=LAST_N,
        lambda_=LAM,
        bounds=np.asarray(LIM),
    )
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
        bounds=np.asarray([LIM[0], LIM[1], LIM[0], LIM[1]]),
        method="fes_compare_1x4_abc",
        fpt_rate_it_s=np.float64(SFPT_RATE),
        fpt_gpus=np.float64(4.0),
        metad_rate_it_s=np.float64(METAD_RATE),
        metad_gpus=np.float64(1.0),
    )
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
        f_us_full_xi=xi_us_full,
        f_us_full=f_us_full,
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
    print("[npz] saved, plotting", flush=True)

    from plot_repeat3_abc import plot_all_from_npz

    plot_all_from_npz(OUT)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
