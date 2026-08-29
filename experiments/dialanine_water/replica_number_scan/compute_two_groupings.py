#!/usr/bin/env python3
"""Two replica-scan ΔG groupings for dialanine-water sFPT.

Scheme equalN
    Each ΔG uses the same number of replica samples as R=1024 with 2 ps:
    N = 1024 × (2 ps / 0.1 ps) = 20480.  Then Δt(R) = 20480 / R × 0.1 ps
    (R=128 → 16 ps, R=64 → 32 ps, …).

Scheme dt3ps
    Every R uses a 3 ps window (30 stages) per ΔG.
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
from plot_dG_mae_vs_R import mae_from_npz  # noqa: E402
from plot_replica_ab import render  # noqa: E402

DT_STAGE = 0.1
EQUAL_REF_R = 1024
EQUAL_REF_DT = 2.0
FIXED_DT = 3.0
T_MAX = 160.0

RUNS: dict[int, list[Path]] = {
    64: [
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r64_g1_s0_94293",
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r64_g1_s1_94294",
    ],
    128: [
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r128_g1_s0_94291",
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r128_g1_s1_94292",
    ],
    320: [
        SFPT / "results" / "dialanine_water_sfpt_r320_g1_s0_87371",
        SFPT / "results" / "dialanine_water_sfpt_r320_g1_s2_87373",
    ],
    512: [
        SFPT / "results" / "dialanine_water_sfpt_r512_g1_s0_87344",
        SFPT / "results" / "dialanine_water_sfpt_r512_g1_s2_87368",
    ],
    1024: [
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r1024_g4_s0_91354",
        SFPT / "fair_compete_result" / "dialanine_water_sfpt_r1024_g4_s2_91355",
    ],
}

ALL_R = (64, 128, 320, 512, 1024)


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    if stack.shape[axis] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


def equal_n_group(R: int) -> tuple[int, float]:
    n_samples = int(round(EQUAL_REF_R * EQUAL_REF_DT / DT_STAGE))
    gsz = n_samples // int(R)
    if gsz * int(R) != n_samples:
        raise RuntimeError(f"R={R} does not divide N={n_samples}")
    return gsz, gsz * DT_STAGE


def scheme_group(scheme: str, R: int) -> tuple[int, float]:
    if scheme == "equalN":
        return equal_n_group(R)
    if scheme == "dt3ps":
        gsz = int(round(FIXED_DT / DT_STAGE))
        return gsz, gsz * DT_STAGE
    raise ValueError(scheme)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schemes", type=str, default="equalN,dt3ps")
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
    p.add_argument("--t-max", type=float, default=T_MAX)
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

    phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    n_eff = int(round(EQUAL_REF_R * EQUAL_REF_DT / DT_STAGE))
    print(f"[equalN] N_replicas×stages = {n_eff}  (1024 × 2 ps)", flush=True)
    for R in ALL_R:
        gsz, dt = equal_n_group(R)
        print(f"  R={R:4d}  Δt={dt:g} ps  group_size={gsz}", flush=True)

    subtitles = {
        "equalN": rf"$R\times\Delta t = {EQUAL_REF_R}\times{EQUAL_REF_DT:g}$ ps",
        "dt3ps": None,
    }

    for scheme in schemes:
        out_root = SCAN / scheme
        out_root.mkdir(parents=True, exist_ok=True)
        r_npz: dict[int, Path] = {}
        r_meta: dict[int, Path] = {}
        print(f"\n===== scheme {scheme} =====", flush=True)
        for R, dirs in RUNS.items():
            gsz, dt_group = scheme_group(scheme, R)
            max_groups = max(1, int(np.floor(float(args.t_max) / dt_group + 1e-9)))
            for d in dirs:
                n = len(list(d.glob("stage*_lambda*_dihedral.npy")))
                print(
                    f"[R={R}] {d.name}  n_stage={n}  group={gsz}  "
                    f"dt={dt_group:g} ps  max_groups={max_groups}",
                    flush=True,
                )
                if n < gsz:
                    raise RuntimeError(f"{d} has too few stages ({n} < {gsz})")

            kde_kw = dict(
                phi=phi,
                psi=psi,
                f1d_ref=f1d_ref,
                xi=xi,
                group_size=int(gsz),
                dt_group=float(dt_group),
                temperature=float(args.temperature),
                vmax=None,
                device=device,
                chunk=int(args.chunk),
                n_bins=int(args.n_bins_1d),
                xi_cut=float(args.xi_cut),
                xi_lo=float(args.xi_lo),
                xi_hi=float(args.xi_hi),
                max_groups=int(max_groups),
                lambda_target=float(args.lambda_target),
                lambda_atol=float(args.lambda_atol),
            )
            f_runs, titles = [], None
            t_ps = None
            for d in dirs:
                t_s, _r, _g, f_s, titles_s = _dca.kde_groups_phi(d, **kde_kw)
                f_runs.append(f_s)
                if t_ps is None:
                    t_ps = t_s
                    titles = titles_s
                print(
                    f"[kde] {d.name} groups={len(t_s)} t={t_s[0]:.2f}–{t_s[-1]:.2f} ps",
                    flush=True,
                )

            n_t = min(len(x) for x in f_runs)
            f_stack = np.stack([x[:n_t] for x in f_runs], axis=0)
            t_ps = np.asarray(t_ps[:n_t], dtype=np.float64)
            f_mean, f_std = mean_std(f_stack, axis=0)
            out = out_root / f"convergence_{R}"
            out.mkdir(parents=True, exist_ok=True)
            p1d = out / "fes_1d_all_groups.npz"
            pmeta = out / "fes_rmse_vs_time.npz"
            np.savez_compressed(
                p1d,
                xi=xi,
                F1d_us=f1d_ref,
                F1d_kde_mean=f_mean,
                F1d_kde_std=f_std,
                F1d_kde_runs=f_stack,
                t_ps=t_ps,
                titles=np.asarray(titles[:n_t]),
                group_size=np.int64(gsz),
                dt_group_ps=np.float64(dt_group),
            )
            np.savez_compressed(
                pmeta,
                dG_us_ref=np.float64(dG_us),
                fpt_dirs=np.asarray([str(d) for d in dirs]),
                t_kde_ps=t_ps,
                group_size=np.int64(gsz),
                dt_group_ps=np.float64(dt_group),
                lambda_target=np.float64(args.lambda_target),
            )
            r_npz[R] = p1d
            r_meta[R] = pmeta
            print(f"[ok] {out}", flush=True)

        print(f"\n[mae] {scheme}", flush=True)
        Rs, y, e, per_r = mae_from_npz(r_npz, r_meta)
        kw = {
            "R": np.asarray(Rs, dtype=np.float64),
            "mae_mean": y,
            "mae_std": e,
            "scheme": np.asarray(scheme),
        }
        for R in Rs:
            kw[f"mae_R{R}"] = per_r[R]["mae_seeds"]
            kw[f"seeds_R{R}"] = np.asarray(per_r[R]["seeds"], dtype=np.int64)
            kw[f"dt_R{R}"] = np.float64(scheme_group(scheme, R)[1])
        np.savez_compressed(out_root / "dG_mae_vs_R.npz", **kw)

        render(
            out_stem=SCAN / f"replica_ab_{scheme}",
            r_npz=r_npz,
            r_meta=r_meta,
            panel_a_r=ALL_R,
            subtitle=subtitles[scheme],
            mae_npz_path=out_root / "dG_mae_vs_R.npz",
        )


if __name__ == "__main__":
    main()
