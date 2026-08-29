#!/usr/bin/env python3
"""RMSE and ΔG of 1D FES vs simulation time: FPT-KDE windows vs WT-MetaD hills.

FPT (sFPT / KDE)
  - Group every ``group_size`` stages (default 3); each group spans ``dt_group`` ps
    (default 1.5 ps = 3 × 500 steps × 0.001 ps).
  - KDE 2D FES on (d1,d2)∈[1,4]² from that group's CV+bandwidth, then Boltzmann
    project to ξ = d2−d1; RMSE vs MetaD reference 1D FES.
  - ΔG = k_B T ln(P1/P2) with basins ξ < xi_cut (1) and ξ > xi_cut (2).

MetaD
  - Reconstruct F ≈ −γ/(γ−1) V from cumulative hills (same formula as
    ``WellTemperedMetadynamicsBias.free_energy_grid``).
  - Deposit every ``deposit_pace`` MD steps (default 50); each deposit adds
    ``n_replicas`` hills.  Time after deposit k:
        t = k * deposit_pace * timestep   [ps]
  - RMSE / ΔG of 1D projection vs final ``live/fes.npz``.

Plots: RMSE–t and ΔG–t (scatter+line), horizontal dashed y=0.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from plot_fes_compare import (  # noqa: E402
    ev_to_kcal,
    kT_kcal,
    nature_rcparams,
    relative_fes,
)
from plot_kde_fes import density_on_grid, parse_lambda, parse_stage  # noqa: E402

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
STAGE_RE = re.compile(r"stage(\d+)", re.I)


def project_1d_fixed_edges(
    d1: np.ndarray,
    d2: np.ndarray,
    f_kcal: np.ndarray,
    edges: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann projection onto ξ=d2−d1 using fixed bin edges; min-subtract."""
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    xi = (g2 - g1).ravel()
    f = np.asarray(f_kcal, dtype=np.float64).ravel()
    centers = 0.5 * (edges[:-1] + edges[1:])
    kt = kT_kcal(temperature)
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


def make_1d_edges_from_grid(
    d1: np.ndarray, d2: np.ndarray, n_bins: int
) -> np.ndarray:
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    xi = (g2 - g1).ravel()
    return np.linspace(float(xi.min()), float(xi.max()), int(n_bins) + 1)


def rmse_1d(
    f_a: np.ndarray,
    f_b: np.ndarray,
    *,
    vmax: float | None,
    xi: np.ndarray | None = None,
    xi_lo: float | None = None,
    xi_hi: float | None = None,
) -> float:
    a = np.asarray(f_a, dtype=np.float64)
    b = np.asarray(f_b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if xi is not None and (xi_lo is not None or xi_hi is not None):
        xi_a = np.asarray(xi, dtype=np.float64)
        if xi_a.shape != a.shape:
            raise ValueError(f"xi shape {xi_a.shape} != F shape {a.shape}")
        if xi_lo is not None:
            m = m & (xi_a >= float(xi_lo))
        if xi_hi is not None:
            m = m & (xi_a <= float(xi_hi))
    if vmax is not None and float(vmax) > 0:
        # only compare below vmax on the reference curve
        m = m & (b <= float(vmax))
    if not np.any(m):
        return float("nan")
    aa = a[m] - np.min(a[m])
    bb = b[m] - np.min(b[m])
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def delta_g_two_basins(
    xi: np.ndarray,
    f_kcal: np.ndarray,
    *,
    temperature: float,
    xi_cut: float = 0.0,
    xi_lo: float | None = None,
    xi_hi: float | None = None,
) -> float:
    """ΔG = (1/β) ln(P1/P2) in kcal/mol.

    Optionally restrict to ξ ∈ [xi_lo, xi_hi] (inclusive), then split basins at xi_cut.
    Basin 1: ξ < xi_cut (product-like for Claisen: d2−d1 < 0).
    Basin 2: ξ > xi_cut (reactant-like: d2−d1 > 0).
    P_i ∝ Σ_{ξ∈basin i} exp(−β F(ξ)) (equal bin width cancels in the ratio).
    """
    xi_a = np.asarray(xi, dtype=np.float64)
    f = np.asarray(f_kcal, dtype=np.float64)
    kt = kT_kcal(temperature)
    beta = 1.0 / kt
    m = np.isfinite(xi_a) & np.isfinite(f)
    if xi_lo is not None:
        m = m & (xi_a >= float(xi_lo))
    if xi_hi is not None:
        m = m & (xi_a <= float(xi_hi))
    xi_a, f = xi_a[m], f[m]
    b1 = xi_a < float(xi_cut)
    b2 = xi_a > float(xi_cut)
    if not np.any(b1) or not np.any(b2):
        return float("nan")

    def _logZ(sel: np.ndarray) -> float:
        vals = -beta * f[sel]
        mval = float(np.max(vals))
        return mval + float(np.log(np.sum(np.exp(vals - mval))))

    # ΔG = kT ln(P1/P2) = kT (logZ1 - logZ2)
    return float(kt * (_logZ(b1) - _logZ(b2)))


def delta_g_from_2d(
    d1: np.ndarray,
    d2: np.ndarray,
    f_kcal: np.ndarray,
    *,
    temperature: float,
    xi_cut: float = 0.0,
) -> float:
    """ΔG from the full 2D FES (no ξ window, no extra 2D crop).

    P1: d2 − d1 < xi_cut (product-like); P2: d2 − d1 > xi_cut (reactant-like).
    Uniform (d1, d2) grid → cell area cancels in the ratio.
    """
    d1_a = np.asarray(d1, dtype=np.float64)
    d2_a = np.asarray(d2, dtype=np.float64)
    f = np.asarray(f_kcal, dtype=np.float64)
    g1, g2 = np.meshgrid(d1_a, d2_a, indexing="xy")
    if f.shape != g1.shape:
        raise ValueError(f"F shape {f.shape} != meshgrid {g1.shape}")
    kt = kT_kcal(temperature)
    beta = 1.0 / kt
    xi = g2 - g1
    m = np.isfinite(f) & np.isfinite(xi)
    b1 = m & (xi < float(xi_cut))
    b2 = m & (xi > float(xi_cut))
    if not np.any(b1) or not np.any(b2):
        return float("nan")

    def _logZ(sel: np.ndarray) -> float:
        vals = -beta * f[sel]
        mval = float(np.max(vals))
        return mval + float(np.log(np.sum(np.exp(vals - mval))))

    return float(kt * (_logZ(b1) - _logZ(b2)))


def incremental_metad_metrics(
    hills_path: Path,
    *,
    d1: np.ndarray,
    d2: np.ndarray,
    f1d_ref: np.ndarray,
    edges: np.ndarray,
    xi_centers: np.ndarray,
    n_replicas: int,
    deposit_pace: int,
    timestep: float,
    bias_factor: float,
    sigma: tuple[float, float],
    temperature: float,
    vmax_kcal: float | None,
    stride_deposits: int,
    xi_cut: float,
    xi_lo: float | None = None,
    xi_hi: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative-hills MetaD: return t_ps, 1D-RMSE, ΔG."""
    hills = np.load(hills_path)
    n_hills = (
        int(hills["n_hills"]) if "n_hills" in hills.files else int(hills["centers"].shape[0])
    )
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float32)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float32)

    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    scale = float(bias_factor) / (float(bias_factor) - 1.0)
    sig = np.asarray(sigma, dtype=np.float64)

    n_dep = n_hills // int(n_replicas)
    if n_dep * int(n_replicas) != n_hills:
        print(
            f"[warn] n_hills={n_hills} not divisible by n_replicas={n_replicas}; "
            f"using n_dep={n_dep}",
            flush=True,
        )
    stride = max(1, int(stride_deposits))
    v = np.zeros(grid_xy.shape[0], dtype=np.float64)
    t_list: list[float] = []
    r_list: list[float] = []
    g_list: list[float] = []
    hills_per_dep = int(n_replicas)

    for k in range(1, n_dep + 1):
        i0 = (k - 1) * hills_per_dep
        i1 = k * hills_per_dep
        c = centers[i0:i1].astype(np.float64)
        w = heights[i0:i1].astype(np.float64)
        diff = (grid_xy[:, None, :] - c[None, :, :]) / sig.reshape(1, 1, -1)
        gauss = np.exp(-0.5 * np.sum(diff * diff, axis=-1))
        v += gauss @ w
        if (k % stride) != 0 and k != n_dep:
            continue
        f2 = relative_fes((-scale * v).reshape(g1.shape) * EV_TO_KCAL, None)
        _, f1d = project_1d_fixed_edges(d1, d2, f2, edges, temperature=temperature)
        t_ps = float(k) * float(deposit_pace) * float(timestep)
        t_list.append(t_ps)
        r_list.append(
            rmse_1d(
                f1d,
                f1d_ref,
                vmax=vmax_kcal,
                xi=xi_centers,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
            )
        )
        g_list.append(
            delta_g_from_2d(
                d1,
                d2,
                f2,
                temperature=temperature,
                xi_cut=xi_cut,
            )
        )
        if k == 1 or k % max(1, n_dep // 10) == 0 or k == n_dep:
            print(
                f"[metad] deposit {k}/{n_dep}  t={t_ps:.3f} ps  "
                f"RMSE={r_list[-1]:.4f}  ΔG={g_list[-1]:.4f} kcal/mol",
                flush=True,
            )

    return (
        np.asarray(t_list, dtype=np.float64),
        np.asarray(r_list, dtype=np.float64),
        np.asarray(g_list, dtype=np.float64),
    )


def discover_stage_files(data_dir: Path) -> list[tuple[int, Path, Path, float]]:
    rows: list[tuple[int, Path, Path, float]] = []
    for cv in sorted(data_dir.glob("stage[0-9]*_lambda*_cv.npy")):
        idx = parse_stage(cv)
        lam = parse_lambda(cv)
        if idx is None or lam is None:
            continue
        bw = cv.with_name(cv.name.replace("_cv.npy", "_bandwidth.npy"))
        if not bw.is_file():
            continue
        rows.append((idx, cv, bw, float(lam)))
    rows.sort(key=lambda r: r[0])
    return rows


def kde_group_rmse(
    stages: list[tuple[int, Path, Path, float]],
    *,
    edges: np.ndarray,
    f1d_ref: np.ndarray,
    bounds: tuple[float, float, float, float],
    grid_n: int,
    temperature: float,
    beta0: float,
    vmax_kcal: float | None,
    device: torch.device,
    chunk: int,
) -> float:
    d1_lo, d1_hi, d2_lo, d2_hi = bounds
    d1 = np.linspace(d1_lo, d1_hi, int(grid_n))
    d2 = np.linspace(d2_lo, d2_hi, int(grid_n))
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)

    lams = [lam for _, _, _, lam in stages]
    if max(lams) - min(lams) > 1e-12:
        raise ValueError(f"mixed λ in group: {lams}")
    lam = float(lams[0])

    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    for _, cv_path, bw_path, _ in stages:
        cv = np.load(cv_path)
        bw = np.load(bw_path)
        p_acc += density_on_grid(grid_xy, cv, bw, chunk=chunk, device=device)
    p = np.clip(p_acc / float(len(stages)), 1e-300, None)
    f_ev = -np.log(p) / (beta0 * lam)
    f_kcal = relative_fes(f_ev * EV_TO_KCAL, None).reshape(g1.shape)
    _, f1d = project_1d_fixed_edges(d1, d2, f_kcal, edges, temperature=temperature)
    return rmse_1d(f1d, f1d_ref, vmax=vmax_kcal)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fpt-dir",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736"
        ),
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
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--dt-group", type=float, default=1.5, help="ps per stage group")
    p.add_argument("--bounds", type=str, default="1.0,4.0,1.0,4.0")
    p.add_argument("--grid-n", type=int, default=120)
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    # MetaD
    p.add_argument("--n-replicas", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=30.0)
    p.add_argument("--sigma", type=str, default="0.2,0.2")
    p.add_argument(
        "--metad-stride-deposits",
        type=int,
        default=10,
        help="Evaluate MetaD RMSE every N deposits (1=every deposit).",
    )
    p.add_argument(
        "--xi-cut",
        type=float,
        default=0.0,
        help="Split basins on ξ=d2−d1: basin1 ξ<cut, basin2 ξ>cut.",
    )
    p.add_argument(
        "--reuse-kde-npz",
        type=Path,
        default=None,
        help="Reuse sFPT/KDE arrays from an existing fes_rmse_vs_time.npz; "
        "only recompute MetaD curves from --metad-dir.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    fpt_dir = args.fpt_dir.resolve()
    metad_dir = args.metad_dir.resolve()
    out_dir = (args.out_dir or (fpt_dir / "kde_fes" / "rmse_vs_time")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hills_path = metad_dir / "hills.npz"
    fes_path = metad_dir / "fes.npz"
    if not hills_path.is_file() or not fes_path.is_file():
        raise FileNotFoundError(f"need {hills_path} and {fes_path}")

    bounds_v = [float(x) for x in args.bounds.replace(";", ",").split(",") if x.strip()]
    if len(bounds_v) != 4:
        raise ValueError("--bounds must be d1_lo,d1_hi,d2_lo,d2_hi")
    bounds = (bounds_v[0], bounds_v[1], bounds_v[2], bounds_v[3])
    sigma = tuple(
        float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip()
    )
    if len(sigma) != 2:
        raise ValueError("--sigma must be s1,s2")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    vmax = float(args.vmax_kcal) if args.vmax_kcal > 0 else None

    # Reference 1D from final MetaD FES; use same ξ edges for all RMSE
    fes = np.load(fes_path)
    d1_m = np.asarray(fes["d1"], dtype=np.float64)
    d2_m = np.asarray(fes["d2"], dtype=np.float64)
    # Restrict reference projection domain to the same CV box used for KDE
    d1_lo, d1_hi, d2_lo, d2_hi = bounds
    m1 = (d1_m >= d1_lo) & (d1_m <= d1_hi)
    m2 = (d2_m >= d2_lo) & (d2_m <= d2_hi)
    d1_use = d1_m[m1]
    d2_use = d2_m[m2]
    f_ref_full = relative_fes(ev_to_kcal(np.asarray(fes["F"], dtype=np.float64)), None)
    # F shape is (len(d2), len(d1)) with xy mesh
    if f_ref_full.shape == (len(d2_m), len(d1_m)):
        f_ref = f_ref_full[np.ix_(m2, m1)]
    elif f_ref_full.shape == (len(d1_m), len(d2_m)):
        f_ref = f_ref_full.T[np.ix_(m2, m1)]
    else:
        raise ValueError(f"unexpected F shape {f_ref_full.shape}")
    f_ref = relative_fes(f_ref, None)
    edges = make_1d_edges_from_grid(d1_use, d2_use, int(args.n_bins_1d))
    xi, f1d_ref = project_1d_fixed_edges(
        d1_use, d2_use, f_ref, edges, temperature=float(args.temperature)
    )

    print(
        f"[ref] MetaD 1D bins={len(xi)}  ξ∈[{edges[0]:.2f},{edges[-1]:.2f}]  "
        f"Fmax={float(np.nanmax(f1d_ref)):.2f} kcal/mol",
        flush=True,
    )
    dG_ref = delta_g_two_basins(
        xi, f1d_ref, temperature=float(args.temperature), xi_cut=float(args.xi_cut)
    )
    print(
        f"[ref] ΔG=(1/β)ln(P1/P2)  xi_cut={args.xi_cut:g}  "
        f"ΔG_MetaD_final={dG_ref:.4f} kcal/mol",
        flush=True,
    )
    print(
        f"[metad] deposit_pace={args.deposit_pace} steps  "
        f"Δt_hill={args.deposit_pace * args.timestep:g} ps  "
        f"replicas={args.n_replicas}  γ={args.bias_factor}",
        flush=True,
    )

    # MetaD cumulative hills on the same cropped CV grid as the reference 1D
    t_m, rmse_m, dG_m = incremental_metad_metrics(
        hills_path,
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
        vmax_kcal=vmax,
        stride_deposits=int(args.metad_stride_deposits),
        xi_cut=float(args.xi_cut),
    )

    reuse = args.reuse_kde_npz
    if reuse is not None:
        reuse = Path(reuse).resolve()
        if not reuse.is_file():
            raise FileNotFoundError(f"--reuse-kde-npz not found: {reuse}")
        old = np.load(reuse)
        t_k_a = np.asarray(old["t_kde_ps"], dtype=np.float64)
        r_k_a = np.asarray(old["rmse_kde"], dtype=np.float64)
        g_k_a = np.asarray(old["dG_kde"], dtype=np.float64)
        gsz = (
            int(np.asarray(old["group_size"]))
            if "group_size" in old.files
            else int(args.group_size)
        )
        dt_g = (
            float(np.asarray(old["dt_group_ps"]))
            if "dt_group_ps" in old.files
            else float(args.dt_group)
        )
        print(
            f"[reuse-kde] {reuse}  n={len(t_k_a)}  "
            f"group_size={gsz}  dt_group={dt_g} ps  (sFPT unchanged)",
            flush=True,
        )
    else:
        rows = discover_stage_files(fpt_dir)
        gsz = int(args.group_size)
        n_groups = len(rows) // gsz
        t_k: list[float] = []
        r_k: list[float] = []
        g_k: list[float] = []
        for ig in range(n_groups):
            chunk_rows = rows[ig * gsz : (ig + 1) * gsz]
            t_ps = float(ig + 1) * float(args.dt_group)
            print(
                f"[kde] group {ig + 1}/{n_groups}  stages "
                f"{chunk_rows[0][0]}-{chunk_rows[-1][0]}  t={t_ps:.2f} ps",
                flush=True,
            )
            d1 = np.linspace(bounds[0], bounds[1], int(args.grid_n))
            d2 = np.linspace(bounds[2], bounds[3], int(args.grid_n))
            g1, g2 = np.meshgrid(d1, d2, indexing="xy")
            grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
            lam = float(chunk_rows[0][3])
            p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
            for _, cv_p, bw_p, _ in chunk_rows:
                p_acc += density_on_grid(
                    grid_xy, np.load(cv_p), np.load(bw_p), chunk=args.chunk, device=device
                )
            p = np.clip(p_acc / float(len(chunk_rows)), 1e-300, None)
            f_kcal = relative_fes(
                (-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None
            )
            _, f1d = project_1d_fixed_edges(
                d1, d2, f_kcal, edges, temperature=float(args.temperature)
            )
            err = rmse_1d(f1d, f1d_ref, vmax=vmax)
            dG = delta_g_two_basins(
                xi, f1d, temperature=float(args.temperature), xi_cut=float(args.xi_cut)
            )
            t_k.append(t_ps)
            r_k.append(err)
            g_k.append(dG)
            print(f"       RMSE={err:.4f}  ΔG={dG:.4f} kcal/mol", flush=True)

        t_k_a = np.asarray(t_k, dtype=np.float64)
        r_k_a = np.asarray(r_k, dtype=np.float64)
        g_k_a = np.asarray(g_k, dtype=np.float64)
        gsz = int(args.group_size)
        dt_g = float(args.dt_group)

    np.savez_compressed(
        out_dir / "fes_rmse_vs_time.npz",
        t_kde_ps=t_k_a,
        rmse_kde=r_k_a,
        dG_kde=g_k_a,
        t_metad_ps=t_m,
        rmse_metad=rmse_m,
        dG_metad=dG_m,
        dG_metad_ref=np.asarray(dG_ref, dtype=np.float64),
        xi=xi,
        F1d_metad_ref=f1d_ref,
        xi_cut=float(args.xi_cut),
        group_size=gsz,
        dt_group_ps=dt_g,
        deposit_pace=int(args.deposit_pace),
        timestep_ps=float(args.timestep),
        n_replicas=int(args.n_replicas),
        vmax_kcal=float(args.vmax_kcal),
    )

    plt.rcParams.update(nature_rcparams())

    # --- RMSE vs t ---
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.plot(t_m, rmse_m, color="#2166ac", lw=1.3, label="WT-MetaD", zorder=2)
    ax.plot(
        t_k_a,
        r_k_a,
        color="#e67e22",
        lw=1.4,
        marker="o",
        ms=4.5,
        label="sFPT-KDE",
        zorder=3,
    )
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"1D FES RMSE / kcal mol$^{-1}$")
    ax.set_xlim(left=0.0)
    ymax = float(np.nanmax(np.concatenate([r_k_a, rmse_m])))
    ax.set_ylim(bottom=-0.02 * max(ymax, 1.0), top=ymax * 1.08 if ymax > 0 else 1.0)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    out_png = out_dir / "fes_rmse_vs_time.png"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] wrote {out_png} and {out_png.with_suffix('.pdf')}", flush=True)

    # --- ΔG vs t ---
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.axhline(
        dG_ref,
        color="#2166ac",
        ls=":",
        lw=1.0,
        alpha=0.85,
        label=rf"MetaD final $\Delta G$={dG_ref:.2f}",
        zorder=1,
    )
    ax.plot(t_m, dG_m, color="#2166ac", lw=1.3, label="WT-MetaD", zorder=2)
    ax.plot(
        t_k_a,
        g_k_a,
        color="#e67e22",
        lw=1.4,
        marker="o",
        ms=4.5,
        label="sFPT-KDE",
        zorder=3,
    )
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(
        r"$\Delta G = k_BT\ln(P_1/P_2)$ / kcal mol$^{-1}$"
    )
    ax.set_xlim(left=0.0)
    vals = np.concatenate([g_k_a, dG_m, np.asarray([dG_ref, 0.0])])
    vals = vals[np.isfinite(vals)]
    ymin, ymax = float(np.min(vals)), float(np.max(vals))
    pad = 0.08 * max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    out_dg = out_dir / "deltaG_vs_time.png"
    fig.savefig(out_dg, dpi=200)
    fig.savefig(out_dg.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] wrote {out_dg} and {out_dg.with_suffix('.pdf')}", flush=True)
    print(f"[ok] wrote {out_dir / 'fes_rmse_vs_time.npz'}", flush=True)


if __name__ == "__main__":
    main()
