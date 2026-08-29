#!/usr/bin/env python3
"""Tetra-alanine last-frac 6D torus-KDE → MC samples → subspace FES vs REMD.

Same density recipe as dialanine-water, in 6D:

  1. Load last ``--last-frac`` of λ-target stages (dihedral + bandwidth).
  2. Each stage is a 6D torus-KDE with per-replica σ from ``*_bandwidth.npy``.
  3. Average stage densities:  P = (1/S) Σ_s KDE_s
     (equivalent to the uniform mixture of all S×R kernels).
  4. Physical density is π ∝ P^{1/λ} = exp(−β F_6) with F_6 = −kT ln P / λ.
     Draw MC from the mixture of **sharpened** torus kernels
     ``κ → κ/λ = 1/(d σ² λ)`` (exact for one kernel; good when replicas
     are localized). Do **not** sample P then divide the marginal F by λ —
     that projects first and stays as smooth as the KDE.
  5. Histogram those samples in 1D/2D;  F = −kT ln π_sub  (no extra /λ).

REMD (300 K): histogram the replica trajectory in the same bins (no KDE, no MC).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from fp_transformer_torch.units import KB_KCAL_PER_MOL_K  # noqa: E402
from plot_fes_rmse_vs_time import rmse_1d  # noqa: E402
from plot_fes_compare import relative_fes  # noqa: E402


def _load_tetra_conv():
    path = EXP / "plot_convergence_analysis.py"
    spec = importlib.util.spec_from_file_location("tetra_conv_grid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tc = _load_tetra_conv()
N_CV = _tc.N_CV
PHI_IDX = _tc.PHI_IDX
PHI_PAIRS = _tc.PHI_PAIRS
as_cv6 = _tc.as_cv6
discover_dihedral_stages = _tc.discover_dihedral_stages
load_remd_300k = _tc.load_remd_300k
plot_1d_phi_overlay = _tc.plot_1d_phi_overlay
plot_2d_pair = _tc.plot_2d_pair
plot_rama_grid = _tc.plot_rama_grid
plot_phi_ij_grid = _tc.plot_phi_ij_grid
plot_rmse_vs_time = _tc.plot_rmse_vs_time

DEFAULT_FPT = EXP / "results" / "tetra_alanine_sfpt_r1024_s0_88562"
DEFAULT_REMD = EXP / "remd_280_400_n64_90605"


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def load_mixture(rows: list) -> tuple[np.ndarray, np.ndarray, float]:
    cvs = []
    bws = []
    lams = []
    for _, dih_p, bw_p, lam in rows:
        cvs.append(as_cv6(np.load(dih_p)))
        bws.append(np.asarray(np.load(bw_p), dtype=np.float64).reshape(-1))
        lams.append(float(lam))
    cv = np.stack(cvs, axis=0)
    bw = np.stack(bws, axis=0)
    if cv.shape[0] != bw.shape[0] or cv.shape[1] != bw.shape[1]:
        raise ValueError(f"mixture shape mismatch cv={cv.shape} bw={bw.shape}")
    return cv, bw, float(np.mean(lams))


def sample_mixture_kde(
    cv: np.ndarray,
    bw: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """Draw from the mixture of torus kernels, optionally sharpened by λ.

    Pick (stage, replica) uniformly, then 6 independent von Mises with
    ``κ = 1 / (d σ² λ)`` matching ``dist = 2 − 2 mean(cos Δ)``.
    ``lam=1`` samples P; ``lam<1`` samples the sharpened mixture that
    approximates π ∝ P^{1/λ}. Histogram those samples with ``F = −kT ln π``
    and do **not** divide by λ again.
    """
    n_stage, n_rep, n_dim = cv.shape
    s_idx = rng.integers(0, n_stage, size=int(n_mc))
    r_idx = rng.integers(0, n_rep, size=int(n_mc))
    mu = cv[s_idx, r_idx]
    sig = np.clip(bw[s_idx, r_idx], 1e-6, None)
    kappa = np.clip(1.0 / (float(n_dim) * sig * sig * float(lam)), 1e-6, 1e6)
    theta = np.empty((int(n_mc), n_dim), dtype=np.float64)
    for k in range(n_dim):
        theta[:, k] = rng.vonmises(mu[:, k], kappa)
    return wrap_pi(theta)


def fes_from_hist(
    counts: np.ndarray,
    *,
    kT: float,
    empty_nan: bool,
) -> np.ndarray:
    p = np.asarray(counts, dtype=np.float64)
    tot = float(np.sum(np.clip(p, 0.0, None)))
    if tot <= 0.0:
        return np.full(p.shape, np.nan if empty_nan else 1e6, dtype=np.float64)
    pos = p > 0.0
    f = np.full(p.shape, np.nan if empty_nan else 1e6, dtype=np.float64)
    f[pos] = -float(kT) * np.log(p[pos] / tot)
    return relative_fes(f, None)


def fes_hist_1d(
    x: np.ndarray, *, n_bins: int, kT: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    counts, _ = np.histogram(np.asarray(x, dtype=np.float64), bins=edges)
    xi = 0.5 * (edges[:-1] + edges[1:])
    return xi, fes_from_hist(counts, kT=kT, empty_nan=False)


def fes_hist_2d(
    x: np.ndarray, y: np.ndarray, *, n_bins: int, kT: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    counts, _, _ = np.histogram2d(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        bins=[edges, edges],
    )
    f = fes_from_hist(counts, kT=kT, empty_nan=True).T
    ax = 0.5 * (edges[:-1] + edges[1:])
    return ax, f


def project_phi(
    samples: np.ndarray, *, n_bins: int, kT: float
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((3, int(n_bins)), dtype=np.float64)
    xi = None
    for i, col in enumerate(PHI_IDX):
        xi, out[i] = fes_hist_1d(samples[:, col], n_bins=n_bins, kT=kT)
    assert xi is not None
    return xi, out


def project_rama(
    samples: np.ndarray, *, n_bins: int, kT: float
) -> tuple[np.ndarray, np.ndarray]:
    ax = None
    out = np.zeros((3, int(n_bins), int(n_bins)), dtype=np.float64)
    for i, (cphi, cpsi) in enumerate(zip(PHI_IDX, (1, 3, 5))):
        ax, out[i] = fes_hist_2d(
            samples[:, cphi], samples[:, cpsi], n_bins=n_bins, kT=kT
        )
    assert ax is not None
    return ax, out


def project_phi_ij(
    samples: np.ndarray, *, n_bins: int, kT: float
) -> tuple[np.ndarray, np.ndarray]:
    ax = None
    out = np.zeros((len(PHI_PAIRS), int(n_bins), int(n_bins)), dtype=np.float64)
    for k, (ci, cj, _i, _j) in enumerate(PHI_PAIRS):
        ax, out[k] = fes_hist_2d(
            samples[:, ci], samples[:, cj], n_bins=n_bins, kT=kT
        )
    assert ax is not None
    return ax, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dir", type=Path, default=DEFAULT_FPT)
    p.add_argument("--remd-dir", type=Path, default=DEFAULT_REMD)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--lambda-target", type=float, default=0.08)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--last-frac", type=float, default=0.5)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--dt-group", type=float, default=1.0)
    p.add_argument("--remd-discard-frac", type=float, default=0.2)
    p.add_argument("--n-mc", type=int, default=1_048_576, help="MC draws for last-frac mixture")
    p.add_argument("--n-mc-group", type=int, default=262144, help="MC draws per time-group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-plot-1d", type=int, default=72)
    p.add_argument("--n-plot-2d", type=int, default=64)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--rmse-vmax", type=float, default=8.0)
    args = p.parse_args()

    fpt = Path(args.fpt_dir).resolve()
    remd_dir = Path(args.remd_dir).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else fpt / "convergence_mc"
    out.mkdir(parents=True, exist_ok=True)

    kT = float(KB_KCAL_PER_MOL_K) * float(args.temperature)
    rmse_vmax = float(args.rmse_vmax) if float(args.rmse_vmax) > 0 else None
    rng = np.random.default_rng(int(args.seed))
    print(
        f"[init] 6D torus-KDE MC  n_mc={args.n_mc}  n_mc_group={args.n_mc_group}  "
        f"kT={kT:.4g}  out={out}",
        flush=True,
    )

    remd_cv, t_slot, i_t = load_remd_300k(
        remd_dir,
        temperature=float(args.temperature),
        discard_frac=float(args.remd_discard_frac),
    )
    xi, f_remd_phi = project_phi(remd_cv, n_bins=int(args.n_plot_1d), kT=kT)
    ax_r, f_remd_rama = project_rama(remd_cv, n_bins=int(args.n_plot_2d), kT=kT)
    ax2, f_remd_phij = project_phi_ij(remd_cv, n_bins=int(args.n_plot_2d), kT=kT)
    for i in range(3):
        print(
            f"[remd] hist Fmax  φ{i+1}={float(np.nanmax(f_remd_phi[i])):.3g}  "
            f"φ{i+1}ψ{i+1}={float(np.nanmax(f_remd_rama[i])):.3g}",
            flush=True,
        )
    for k, (_ci, _cj, i, j) in enumerate(PHI_PAIRS):
        print(
            f"[remd] hist Fmax  φ{i}–φ{j}={float(np.nanmax(f_remd_phij[k])):.3g}",
            flush=True,
        )

    rows = discover_dihedral_stages(
        fpt, lambda_target=float(args.lambda_target), lambda_atol=float(args.lambda_atol)
    )
    if not rows:
        raise RuntimeError(f"no λ≈{args.lambda_target:g} stages in {fpt}")
    n_keep = max(1, int(np.ceil(float(args.last_frac) * len(rows))))
    last_rows = rows[-n_keep:]
    print(
        f"[sfpt] {len(rows)} stages total  last {n_keep}  "
        f"{last_rows[0][0]}–{last_rows[-1][0]}  λ≈{last_rows[0][3]:g}…{last_rows[-1][3]:g}",
        flush=True,
    )

    gsize = int(args.group_size)
    n_groups = len(rows) // gsize
    t = np.zeros(max(n_groups, 0), dtype=np.float64)
    rmse_t = np.zeros((max(n_groups, 0), 3), dtype=np.float64)
    for ig in range(n_groups):
        chunk = rows[ig * gsize : (ig + 1) * gsize]
        cv_g, bw_g, lam_g = load_mixture(chunk)
        samp_g = sample_mixture_kde(
            cv_g, bw_g, int(args.n_mc_group), rng, lam=lam_g
        )
        _, f1 = project_phi(samp_g, n_bins=int(args.n_plot_1d), kT=kT)
        t[ig] = float(ig + 1) * float(args.dt_group)
        for i in range(3):
            rmse_t[ig, i] = rmse_1d(
                f1[i], f_remd_phi[i], vmax=rmse_vmax, xi=xi,
                xi_lo=float(xi.min()), xi_hi=float(xi.max()),
            )
        if ig == 0 or (ig + 1) % 10 == 0 or ig + 1 == n_groups:
            print(
                f"[rmse] group {ig+1}/{n_groups}  t={t[ig]:.2f} ps  "
                f"RMSE φ={rmse_t[ig, 0]:.3f}/{rmse_t[ig, 1]:.3f}/{rmse_t[ig, 2]:.3f}  "
                f"mean={rmse_t[ig].mean():.3f}  S={cv_g.shape[0]} R={cv_g.shape[1]}",
                flush=True,
            )
    if n_groups >= 1:
        plot_rmse_vs_time(
            out / "fes_rmse_phi_1d_vs_time.png",
            t,
            rmse_t,
            ylabel=r"RMSE vs REMD $F(\phi)$ / kcal mol$^{-1}$",
        )

    cv, bw, lam = load_mixture(last_rows)
    sig = np.clip(bw.reshape(-1), 1e-6, None)
    kappa = 1.0 / (float(N_CV) * sig * sig)
    kappa_phys = kappa / float(lam)
    print(
        f"[kde] stages={cv.shape[0]}  R={cv.shape[1]}  kernels={cv.shape[0]*cv.shape[1]}  "
        f"λ={lam:g}  σ min/med/max={sig.min():.4g}/{np.median(sig):.4g}/{sig.max():.4g}  "
        f"κ med={float(np.median(kappa)):.4g}  κ/λ med={float(np.median(kappa_phys)):.4g}",
        flush=True,
    )
    print(
        f"[mc] drawing {args.n_mc} samples from P^{{1/λ}} (sharpened κ/λ)",
        flush=True,
    )
    samp = sample_mixture_kde(cv, bw, int(args.n_mc), rng, lam=lam)

    _, f_phi = project_phi(samp, n_bins=int(args.n_plot_1d), kT=kT)
    rmse = np.array(
        [
            rmse_1d(
                f_phi[i], f_remd_phi[i], vmax=rmse_vmax, xi=xi,
                xi_lo=float(xi.min()), xi_hi=float(xi.max()),
            )
            for i in range(3)
        ],
        dtype=np.float64,
    )
    print(f"[sfpt] RMSE φ={rmse}  mean={rmse.mean():.4f}", flush=True)
    plot_1d_phi_overlay(
        out / "fes_1d_phi_last_frac_vs_remd.png",
        xi,
        f_phi,
        f_remd_phi,
        vmax=float(args.vmax),
        title=(
            rf"6D torus-KDE MC $P^{{1/\lambda}}$ → $F(\phi)$  last {float(args.last_frac):.0%}  "
            rf"stage {last_rows[0][0]}–{last_rows[-1][0]}  RMSE={rmse.mean():.3g}"
        ),
    )

    _, f_rama = project_rama(samp, n_bins=int(args.n_plot_2d), kT=kT)
    for i in range(3):
        plot_2d_pair(
            out / f"fes_2d_phi{i+1}_psi{i+1}_last_frac_vs_remd.png",
            ax_r, ax_r, f_rama[i], f_remd_rama[i],
            vmax=float(args.vmax), n_levels=21,
            xlabel=rf"$\phi_{i+1}$ / rad", ylabel=rf"$\psi_{i+1}$ / rad",
            title_a=r"sFPT $P^{1/\lambda}$ MC", title_b="REMD hist",
        )
    plot_rama_grid(
        out / "fes_2d_phi_psi_all_last_frac_vs_remd.png",
        ax_r, ax_r, f_rama, f_remd_rama,
        vmax=float(args.vmax), n_levels=21,
    )

    _, f_phij = project_phi_ij(samp, n_bins=int(args.n_plot_2d), kT=kT)
    for k, (_ci, _cj, i, j) in enumerate(PHI_PAIRS):
        plot_2d_pair(
            out / f"fes_2d_phi{i}_phi{j}_last_frac_vs_remd.png",
            ax2, ax2, f_phij[k], f_remd_phij[k],
            vmax=float(args.vmax), n_levels=21,
            xlabel=rf"$\phi_{i}$ / rad", ylabel=rf"$\phi_{j}$ / rad",
            title_a=r"sFPT $P^{1/\lambda}$ MC", title_b="REMD hist",
        )
    plot_phi_ij_grid(
        out / "fes_2d_phi_ij_all_last_frac_vs_remd.png",
        ax2, ax2, f_phij, f_remd_phij,
        vmax=float(args.vmax), n_levels=21,
    )

    np.savez_compressed(
        out / "convergence_phi_remd_mc.npz",
        xi=xi,
        ax_2d=ax2,
        t=t,
        rmse_phi=rmse_t,
        F1d_sfpt_last=f_phi,
        F1d_remd=f_remd_phi,
        F2d_phi_ij_sfpt=f_phij,
        F2d_phi_ij_remd=f_remd_phij,
        F2d_rama_sfpt=f_rama,
        F2d_rama_remd=f_remd_rama,
        rmse_last=rmse,
        last_frac=np.float64(args.last_frac),
        stage_lo=np.int64(last_rows[0][0]),
        stage_hi=np.int64(last_rows[-1][0]),
        n_stages_last=np.int64(n_keep),
        n_mc=np.int64(args.n_mc),
        n_mc_group=np.int64(args.n_mc_group),
        lambda_last=np.float64(lam),
        bw_median=np.float64(np.median(sig)),
        kappa_median=np.float64(np.median(kappa)),
        kappa_phys_median=np.float64(np.median(kappa_phys)),
        method=np.asarray("torus_kde6d_mc_p_pow_1_over_lambda"),
        temperature=np.float64(args.temperature),
        remd_T_slot=np.float64(t_slot),
        remd_T_index=np.int64(i_t),
        group_size=np.int64(gsize),
        dt_group=np.float64(args.dt_group),
        kT=np.float64(kT),
        fpt_dir=str(fpt),
        remd_dir=str(remd_dir),
    )
    print(f"[done] last RMSE φ mean={rmse.mean():.4f}  {out}", flush=True)


if __name__ == "__main__":
    main()
