#!/usr/bin/env python3
"""ΔG vs GPU-hour cost from live MetaD hills (full trajectory, not the old snap).

Writes standalone panel-a figure + npz under convergence_repeat3_gpu3.
sFPT cost = wall-clock × 4; MetaD cost = wall-clock × 1.
Y-axis is the dive-in window [-20, 0]; x-axis is the full MetaD series.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from plot_convergence_analysis import plot_series_mean_err  # noqa: E402
from run_repeat3_abc import (  # noqa: E402
    DEPOSIT_PACE,
    GRID_N,
    LIM,
    METAD_RATE,
    METAD_STRIDE,
    META_HILLS,
    N_REP,
    OUT,
    T_K,
    TIMESTEP,
    delta_g_from_2d,
    fes_from_hist,
    mean_std,
    metad_fft_setup,
)

SFPT_GPUS = 4.0
DG_YLIM = (-20.0, 0.0)
NPZ = OUT / "deltaG_vs_cost_full_metad.npz"


def incremental_metad_dg(hills_path: Path, d1: np.ndarray, d2: np.ndarray):
    z = np.load(hills_path)
    n = int(z["n_hills"])
    centers = np.asarray(z["centers"][:n], dtype=np.float64)
    heights = np.asarray(z["heights"][:n], dtype=np.float64).reshape(-1)
    z.close()
    xe, ye, ker, xc, yc = metad_fft_setup(centers, d1, d2)
    n_dep = n // N_REP
    H = np.zeros((xe.size - 1, ye.size - 1), dtype=np.float64)
    t_list, g_list = [], []
    for k in range(1, n_dep + 1):
        i0 = (k - 1) * N_REP
        c = centers[i0 : i0 + N_REP]
        w = heights[i0 : i0 + N_REP]
        Hk, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=[xe, ye], weights=w)
        H += Hk
        if (k % METAD_STRIDE) != 0 and k != n_dep:
            continue
        f2 = fes_from_hist(H, ker, xc, yc, d1, d2)
        t_list.append(float(k) * DEPOSIT_PACE * TIMESTEP)
        g_list.append(delta_g_from_2d(d1, d2, f2, temperature=T_K, xi_cut=0.0))
        if k == METAD_STRIDE or k == n_dep or k % 500 == 0:
            print(
                f"[metad-dg] {hills_path.parent.parent.name} k={k}/{n_dep}  "
                f"t={t_list[-1]:.1f} ps  ΔG={g_list[-1]:.3f}",
                flush=True,
            )
    return np.asarray(t_list), np.asarray(g_list)


def _one(payload):
    hp, d1, d2 = payload
    return str(hp), incremental_metad_dg(Path(hp), d1, d2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = np.linspace(LIM[0], LIM[1], GRID_N)
    d2 = np.linspace(LIM[0], LIM[1], GRID_N)
    old = np.load(OUT / "fes_rmse_vs_time.npz")
    t_sfpt = np.asarray(old["t_kde_ps"], dtype=np.float64)
    tw_sfpt = np.asarray(old["tw_kde_h"], dtype=np.float64)
    dG_sfpt = np.asarray(old["dG_kde"], dtype=np.float64)
    dG_sfpt_std = np.asarray(old["dG_kde_std"], dtype=np.float64)
    dG_us = float(old["dG_us_ref"])
    old.close()
    cost_sfpt = tw_sfpt * SFPT_GPUS

    for p in META_HILLS:
        z = np.load(p)
        n = int(z["n_hills"])
        z.close()
        print(f"[hills] {p.parent.parent.name} n_hills={n}  {p}", flush=True)

    t_m_list, dG_m_list = [], []
    payloads = [(str(hp), d1, d2) for hp in META_HILLS]
    with ProcessPoolExecutor(max_workers=min(2, len(payloads))) as ex:
        for hp_s, (t_m, dG_m) in ex.map(_one, payloads):
            t_m_list.append(t_m)
            dG_m_list.append(dG_m)
            tw = (t_m[-1] / TIMESTEP) / METAD_RATE / 3600.0
            print(
                f"[metad] {Path(hp_s).parent.parent.name} t→{t_m[-1]:.1f} ps  "
                f"wall={tw:.2f} h  ΔG_end={dG_m[-1]:.4f}",
                flush=True,
            )
    n_m = min(len(t) for t in t_m_list)
    t_meta = t_m_list[0][:n_m]
    dG_meta, dG_meta_std = mean_std(np.stack([g[:n_m] for g in dG_m_list]))
    tw_meta = (t_meta / TIMESTEP) / METAD_RATE / 3600.0
    cost_meta = tw_meta.copy()

    np.savez_compressed(
        NPZ,
        t_metad_ps=t_meta,
        tw_metad_h=tw_meta,
        cost_metad_gpuh=cost_meta,
        dG_metad=dG_meta,
        dG_metad_std=dG_meta_std,
        t_kde_ps=t_sfpt,
        tw_kde_h=tw_sfpt,
        cost_kde_gpuh=cost_sfpt,
        dG_kde=dG_sfpt,
        dG_kde_std=dG_sfpt_std,
        dG_us_ref=np.float64(dG_us),
        fpt_gpus=np.float64(SFPT_GPUS),
        metad_gpus=np.float64(1.0),
        metad_rate_it_s=np.float64(METAD_RATE),
    )
    print(f"[npz] {NPZ}", flush=True)

    ylab = r"$\Delta G=k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$"
    plot_series_mean_err(
        OUT,
        stem="deltaG_vs_cost_divein",
        xlabel=rf"cost / GPU$\cdot$h  (sFPT $\times${SFPT_GPUS:g}, MetaD $\times$1)",
        ylabel=ylab,
        t_meta=cost_meta,
        y_meta=dG_meta,
        e_meta=dG_meta_std,
        t_sfpt=cost_sfpt,
        y_sfpt=dG_sfpt,
        e_sfpt=dG_sfpt_std,
        dG_ref=dG_us,
        ref_label="US",
        ylim=DG_YLIM,
    )
    print(
        f"[a] MetaD t={float(t_meta[-1]):.1f} ps  cost={float(cost_meta[-1]):.2f} GPU·h  "
        f"sFPT cost={float(cost_sfpt[-1]):.2f} GPU·h",
        flush=True,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
