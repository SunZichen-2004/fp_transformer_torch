#!/usr/bin/env python3
"""1D KDE FES for fair-compete R=64/128, then refresh replica_ab panel b.

Uses the same US ΔG reference and group protocol (20 stages / 2 ps) as
convergence_512/1024. Does not rebuild MetaD FES.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master")
EXP = ROOT / "experiments" / "dialanine_water"
SFPT = EXP / "sFPT"
DIAL = ROOT / "experiments" / "dialanine"
SCAN = EXP / "replica_number_scan"
REF_1D = SFPT / "convergence_320" / "fes_1d_all_groups.npz"
REF_META = SFPT / "convergence_320" / "fes_rmse_vs_time.npz"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(DIAL))
sys.path.insert(0, str(SCAN))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    DIAL / "plot_convergence_analysis.py",
)
_dca = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dca)

from plot_fes_compare import relative_fes  # noqa: E402

RUNS = {
    64: [
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r64_g1_s0_94293",
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r64_g1_s1_94294",
    ],
    128: [
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r128_g1_s0_94291",
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r128_g1_s1_94292",
    ],
}


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    if stack.shape[axis] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group-size", type=int, default=20)
    p.add_argument("--dt-group", type=float, default=2.0)
    p.add_argument("--max-groups", type=int, default=80)
    p.add_argument("--lambda-target", type=float, default=0.0625)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-np.pi)
    p.add_argument("--xi-hi", type=float, default=np.pi)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda → cpu", flush=True)

    gref = np.load(REF_1D)
    mref = np.load(REF_META)
    xi = np.asarray(gref["xi"], dtype=np.float64)
    f1d_ref = relative_fes(np.asarray(gref["F1d_us"], dtype=np.float64), None)
    dG_us = float(mref["dG_us_ref"])
    gref.close()
    mref.close()
    if int(args.n_bins_1d) != len(xi):
        raise RuntimeError(f"n_bins_1d={args.n_bins_1d} != len(xi)={len(xi)}")

    phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    kde_kw = dict(
        phi=phi,
        psi=psi,
        f1d_ref=f1d_ref,
        xi=xi,
        group_size=int(args.group_size),
        dt_group=float(args.dt_group),
        temperature=float(args.temperature),
        vmax=None,
        device=device,
        chunk=int(args.chunk),
        n_bins=int(args.n_bins_1d),
        xi_cut=float(args.xi_cut),
        xi_lo=float(args.xi_lo),
        xi_hi=float(args.xi_hi),
        max_groups=int(args.max_groups),
        lambda_target=float(args.lambda_target),
        lambda_atol=float(args.lambda_atol),
    )

    for R, dirs in RUNS.items():
        for d in dirs:
            n = len(list(d.glob("stage*_lambda*_dihedral.npy")))
            print(f"[R={R}] {d.name}  n_stage={n}", flush=True)
            if n < int(args.group_size):
                raise RuntimeError(f"{d} has too few stages ({n})")

        f_runs, titles = [], None
        t_ps = None
        for d in dirs:
            t_s, _r, _g, f_s, titles_s = _dca.kde_groups_phi(d, **kde_kw)
            f_runs.append(f_s)
            if t_ps is None:
                t_ps = t_s
                titles = titles_s
            print(f"[kde] {d.name} groups={len(t_s)} t={t_s[0]:.0f}–{t_s[-1]:.0f} ps", flush=True)

        n_t = min(len(x) for x in f_runs)
        f_stack = np.stack([x[:n_t] for x in f_runs], axis=0)
        t_ps = np.asarray(t_ps[:n_t], dtype=np.float64)
        f_mean, f_std = mean_std(f_stack, axis=0)
        out = SFPT / "fair_compete_result" / f"convergence_{R}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "fes_1d_all_groups.npz",
            xi=xi,
            F1d_us=f1d_ref,
            F1d_kde_mean=f_mean,
            F1d_kde_std=f_std,
            F1d_kde_runs=f_stack,
            t_ps=t_ps,
            titles=np.asarray(titles[:n_t]),
        )
        np.savez_compressed(
            out / "fes_rmse_vs_time.npz",
            dG_us_ref=np.float64(dG_us),
            fpt_dirs=np.asarray([str(d) for d in dirs]),
            t_kde_ps=t_ps,
            group_size=np.int64(args.group_size),
            dt_group_ps=np.float64(args.dt_group),
            lambda_target=np.float64(args.lambda_target),
        )
        print(f"[ok] {out}", flush=True)

    import plot_dG_mae_vs_R as mae_mod
    import plot_replica_ab as ab_mod

    mae_mod.main()
    ab_mod.main()
    print("[ok] replica_ab refreshed", flush=True)


if __name__ == "__main__":
    main()
