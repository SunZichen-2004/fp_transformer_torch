#!/usr/bin/env python3
"""Dialanine-water convergence: sFPT torus-KDE vs WT-MetaD.

Same figure suite as ``experiments/dialanine/plot_convergence_analysis.py``,
plus a 2D φ–ψ FES from the last ``--last-frac`` of λ=0.1 stages (default 20%).

Defaults target:
  sFPT  ``.../dialanine_water_sfpt_r1024_s0_87246``
  MetaD ``.../dialanine_water_metad_r64_s0_87241/live``
  out   ``<fpt-dir>/convergence``
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DIAL = EXP.parent / "dialanine"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))
sys.path.insert(0, str(DIAL))

_spec = importlib.util.spec_from_file_location(
    "dialanine_plot_convergence_analysis",
    DIAL / "plot_convergence_analysis.py",
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
from plot_g5_1d_grid_and_dg_zoom import fpt_wall_hours_from_throughput  # noqa: E402

KB_EV_PER_K = _dca.KB_EV_PER_K
EV_TO_KCAL = _dca.EV_TO_KCAL

DEFAULT_FPT = (
    EXP
    / "sFPT"
    / "results"
    / "dialanine_water_sfpt_r1024_s0_87246"
)
DEFAULT_METAD = (
    EXP.parent
    / "dialanine_water_meta"
    / "results"
    / "dialanine_water_metad_r64_s0_87241"
    / "live"
)


def kde_2d_from_stages(
    rows: list,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    temperature: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """Return relative F(φ,ψ) / kcal·mol⁻¹ from torus KDE over stage rows."""
    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    lam = float(rows[0][3])
    for _, dih_p, bw_p, _ in rows:
        p_acc += _dca.density_on_grid_torus(
            grid_xy,
            np.load(dih_p),
            np.load(bw_p),
            chunk=chunk,
            device=device,
        )
    p = np.clip(p_acc / float(len(rows)), 1e-300, None)
    return relative_fes(
        (-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None
    )


def metad_2d_from_last_ps(
    hills_path: Path,
    *,
    phi: np.ndarray,
    psi: np.ndarray,
    n_replicas: int,
    deposit_pace: int,
    timestep: float,
    bias_factor: float,
    sigma: tuple[float, float],
    last_ps: float,
    device: torch.device,
) -> np.ndarray:
    """2D F(φ,ψ) from MetaD hills in the last ``last_ps`` of trajectory."""
    hills = np.load(hills_path)
    n_hills = (
        int(hills["n_hills"])
        if "n_hills" in hills.files
        else int(hills["centers"].shape[0])
    )
    centers = np.asarray(hills["centers"][:n_hills], dtype=np.float64)
    heights = np.asarray(hills["heights"][:n_hills], dtype=np.float64)
    dt_dep = float(deposit_pace) * float(timestep)
    n_dep = n_hills // int(n_replicas)
    n_keep_dep = int(np.ceil(float(last_ps) / dt_dep)) if last_ps > 0 else n_dep
    n_keep_dep = min(max(n_keep_dep, 1), n_dep)
    i0_dep = n_dep - n_keep_dep
    i0 = i0_dep * int(n_replicas)
    c = torch.as_tensor(centers[i0:], dtype=torch.float64, device=device)
    w = torch.as_tensor(heights[i0:], dtype=torch.float64, device=device)
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid = torch.as_tensor(
        np.stack([g1.ravel(), g2.ravel()], axis=-1),
        dtype=torch.float64,
        device=device,
    )
    sig = torch.as_tensor(sigma, dtype=torch.float64, device=device).view(1, 1, 2)
    v = torch.zeros(grid.shape[0], dtype=torch.float64, device=device)
    chunk_h = 4096
    for s in range(0, c.shape[0], chunk_h):
        cc = c[s : s + chunk_h]
        ww = w[s : s + chunk_h]
        diff = grid[:, None, :] - cc[None, :, :]
        diff = diff - _dca.TWO_PI * torch.round(diff / _dca.TWO_PI)
        gauss = torch.exp(-0.5 * ((diff / sig) ** 2).sum(-1))
        v = v + (gauss @ ww)
    scale = float(bias_factor) / (float(bias_factor) - 1.0)
    return relative_fes(
        (-scale * v.detach().cpu().numpy()).reshape(g1.shape) * EV_TO_KCAL, None
    )


def plot_2d_fes_last_frac(
    out: Path,
    *,
    fpt_dir: Path,
    hills_path: Path,
    phi: np.ndarray,
    psi: np.ndarray,
    last_frac: float,
    temperature: float,
    device: torch.device,
    chunk: int,
    n_replicas: int,
    deposit_pace: int,
    timestep: float,
    bias_factor: float,
    sigma: tuple[float, float],
    metad_ref_last_ps: float,
    vmax: float,
    lambda_target: float = 0.1,
) -> None:
    rows = _dca.discover_dihedral_stages(fpt_dir, lambda_target=float(lambda_target))
    if not rows:
        raise RuntimeError(
            f"no λ={lambda_target:g} stages in {fpt_dir}"
        )
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    s0, s1 = use[0][0], use[-1][0]
    print(
        f"[2d] sFPT last {last_frac:.0%} stages: {n_keep}/{len(rows)}  "
        f"stage{s0:04d}–{s1:04d}",
        flush=True,
    )
    f_sfpt = kde_2d_from_stages(
        use,
        phi=phi,
        psi=psi,
        temperature=temperature,
        device=device,
        chunk=chunk,
    )
    f_meta = metad_2d_from_last_ps(
        hills_path,
        phi=phi,
        psi=psi,
        n_replicas=n_replicas,
        deposit_pace=deposit_pace,
        timestep=timestep,
        bias_factor=bias_factor,
        sigma=sigma,
        last_ps=metad_ref_last_ps,
        device=device,
    )
    np.savez_compressed(
        out / "fes_2d_last_frac.npz",
        phi=phi,
        psi=psi,
        F_sfpt=f_sfpt,
        F_metad=f_meta,
        last_frac=float(last_frac),
        stage_lo=int(s0),
        stage_hi=int(s1),
        n_stages=int(n_keep),
        metad_ref_last_ps=float(metad_ref_last_ps),
    )

    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    levels = np.linspace(0.0, float(vmax), 25)
    for ax, F, title, cmap in [
        (
            axes[0],
            f_sfpt,
            rf"sFPT KDE  last {last_frac:.0%}  "
            rf"(stage {s0}–{s1})",
            "viridis",
        ),
        (
            axes[1],
            f_meta,
            (
                "WT-MetaD  ALL hills"
                if float(metad_ref_last_ps) <= 0
                else rf"WT-MetaD  last {metad_ref_last_ps:g} ps hills"
            ),
            "viridis",
        ),
    ]:
        Z = np.minimum(relative_fes(F, None), float(vmax))
        im = ax.contourf(phi, psi, Z, levels=levels, cmap=cmap)
        ax.contour(phi, psi, Z, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.35)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"$\phi$ / rad")
        ax.set_ylabel(r"$\psi$ / rad")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(which="both", top=False, right=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(FE_LABEL, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        r"2D $F(\phi,\psi)$  sFPT (last stages) vs MetaD ref",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "fes_2d_last_frac.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fes_2d_last_frac.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fes_2d_last_frac", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dir", type=Path, default=DEFAULT_FPT)
    p.add_argument("--metad-dir", type=Path, default=DEFAULT_METAD)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <fpt-dir>/convergence",
    )
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--dt-group", type=float, default=1.0, help="ps per stage group")
    p.add_argument("--dt-stage", type=float, default=0.1, help="ps per stage")
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
    p.add_argument("--fpt-rate", type=float, default=7.59)
    p.add_argument("--metad-rate", type=float, default=50.0)
    p.add_argument("--fpt-gpus", type=int, default=1)
    p.add_argument("--metad-gpus", type=int, default=1)
    p.add_argument("--collage-ncols", type=int, default=4)
    p.add_argument(
        "--collage-vmax",
        type=float,
        default=0.0,
        help="Cap 1D collage F(φ); <=0 means no clip (auto y-max from data)",
    )
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument(
        "--metad-ref-last-ps",
        type=float,
        default=0.0,
        help="MetaD FES ref from last this many ps of hills; <=0 means ALL hills",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=0.0,
        help="Truncate sim-time axes to ≤ this (ps); 0 = no truncate (plot all available)",
    )
    p.add_argument(
        "--align-metad-to-sfpt",
        action="store_true",
        help="If set, resample MetaD onto sFPT t-grid (truncates MetaD to sFPT length)",
    )
    p.add_argument(
        "--last-frac",
        type=float,
        default=0.2,
        help="Fraction of last λ=0.1 stages for 2D FES panel",
    )
    p.add_argument("--fes2d-vmax", type=float, default=12.0)
    p.add_argument(
        "--lambda-target",
        type=float,
        default=0.1,
        help="Select sFPT stages with this λ (filename lambdaXpY)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    fpt_dir = args.fpt_dir.resolve()
    metad_dir = args.metad_dir.resolve()
    out = (args.out_dir or (fpt_dir / "convergence")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    hills_path = metad_dir / "hills.npz"
    fes_path = metad_dir / "fes.npz"
    if not hills_path.is_file():
        raise FileNotFoundError(f"need {hills_path}")
    if fes_path.is_file():
        zfes = np.load(fes_path)
        phi = np.asarray(zfes["phi"], dtype=np.float64)
        psi = np.asarray(zfes["psi"], dtype=np.float64)
        if int(args.grid_n) != len(phi):
            phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
            psi = np.linspace(-np.pi, np.pi, int(args.grid_n))
    else:
        phi = np.linspace(-np.pi, np.pi, int(args.grid_n))
        psi = np.linspace(-np.pi, np.pi, int(args.grid_n))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    sigma = tuple(
        float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip()
    )
    if len(sigma) != 2:
        raise ValueError("--sigma must be s1,s2")

    xi_lo = float(args.xi_lo)
    xi_hi = float(args.xi_hi)
    xi_cut = float(args.xi_cut)
    vmax = float(args.vmax_kcal) if args.vmax_kcal > 0 else None
    t_max = float(args.t_max)

    xi, f1d_ref, t_meta_tot = _dca.metad_ref_from_last_ps(
        hills_path,
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
    # Cap ref window if MetaD shorter than requested
    if float(args.metad_ref_last_ps) > 0 and float(args.metad_ref_last_ps) > t_meta_tot:
        print(
            f"[ref] requested last {args.metad_ref_last_ps:g} ps but traj="
            f"{t_meta_tot:.1f} ps → using full trajectory for ref",
            flush=True,
        )
    dG_ref = delta_g_two_basins(
        xi,
        f1d_ref,
        temperature=float(args.temperature),
        xi_cut=xi_cut,
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )
    ref_tag = (
        "ALL hills"
        if float(args.metad_ref_last_ps) <= 0
        else f"last {args.metad_ref_last_ps:g} ps"
    )
    print(
        f"[ref] MetaD F(φ) {ref_tag} "
        f"(traj={t_meta_tot:.1f} ps)  ΔG_ref={dG_ref:.4f} kcal/mol",
        flush=True,
    )
    print(f"[fpt] {fpt_dir}", flush=True)
    print(f"[metad] {metad_dir}  out={out}", flush=True)

    max_groups = args.max_groups
    if max_groups is None and t_max > 0:
        max_groups = max(1, int(np.floor(t_max / float(args.dt_group) + 1e-12)))

    t_meta, rmse_meta, dG_meta, f_meta_traj = _dca.incremental_metad_phi(
        hills_path,
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
    f_meta_plot = f1d_ref

    t_sfpt, rmse_sfpt, dG_sfpt, f_sfpt, titles = _dca.kde_groups_phi(
        fpt_dir,
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
        max_groups=max_groups,
        lambda_target=float(args.lambda_target),
    )

    if t_max > 0:
        t_meta, rmse_meta, dG_meta, f_meta_traj = _dca.truncate_by_tmax(
            t_meta, t_max, rmse_meta, dG_meta, f_meta_traj
        )
        t_sfpt, rmse_sfpt, dG_sfpt, f_sfpt = _dca.truncate_by_tmax(
            t_sfpt, t_max, rmse_sfpt, dG_sfpt, f_sfpt
        )
        titles = titles[: len(t_sfpt)]
        print(
            f"[t-max] truncated to ≤{t_max:g} ps: "
            f"sFPT n={len(t_sfpt)} MetaD n={len(t_meta)}",
            flush=True,
        )

    if args.align_metad_to_sfpt:
        rmse_meta = _dca.align_series_to_times(t_meta, rmse_meta, t_sfpt)
        dG_meta = _dca.align_series_to_times(t_meta, dG_meta, t_sfpt)
        f_meta_traj = _dca.align_series_to_times(t_meta, f_meta_traj, t_sfpt)
        t_meta = np.asarray(t_sfpt, dtype=np.float64).copy()
        print(
            f"[align] MetaD → sFPT t grid  n={len(t_sfpt)}  "
            f"t={t_sfpt[0]:g}–{t_sfpt[-1]:g} ps",
            flush=True,
        )
    else:
        print(
            f"[no-align] MetaD keep native t  n={len(t_meta)}  "
            f"t={t_meta[0]:g}–{t_meta[-1]:g} ps; "
            f"sFPT n={len(t_sfpt)} t={t_sfpt[0]:g}–{t_sfpt[-1]:g} ps",
            flush=True,
        )

    tw_sfpt = fpt_wall_hours_from_throughput(
        t_sfpt, timestep_ps=float(args.timestep), rate_it_s=float(args.fpt_rate)
    )
    tw_meta = fpt_wall_hours_from_throughput(
        t_meta, timestep_ps=float(args.timestep), rate_it_s=float(args.metad_rate)
    )
    cost_sfpt = tw_sfpt * float(args.fpt_gpus)
    cost_meta = tw_meta * float(args.metad_gpus)
    tw_meta_own = tw_meta
    cost_meta_own = cost_meta

    np.savez_compressed(
        out / "fes_rmse_vs_time.npz",
        t_metad_ps=t_meta,
        t_kde_ps=t_sfpt,
        tw_metad_h=tw_meta,
        tw_kde_h=tw_sfpt,
        cost_metad_gpuh=cost_meta,
        cost_kde_gpuh=cost_sfpt,
        rmse_metad=rmse_meta,
        rmse_kde=rmse_sfpt,
        dG_metad=dG_meta,
        dG_kde=dG_sfpt,
        dG_metad_ref=dG_ref,
        xi=xi,
        F1d_metad_ref=f1d_ref,
        group_size=int(args.group_size),
        dt_group_ps=float(args.dt_group),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
        xi_cut=xi_cut,
        fpt_dir=str(fpt_dir),
        metad_dir=str(metad_dir),
        fpt_rate_it_s=float(args.fpt_rate),
        metad_rate_it_s=float(args.metad_rate),
        metad_ref_last_ps=float(args.metad_ref_last_ps),
        t_max_ps=float(t_max),
        tw_metad_own_h=tw_meta_own,
        cost_metad_own_gpuh=cost_meta_own,
    )
    np.savez_compressed(
        out / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad=f_meta_plot,
        F1d_kde=f_sfpt,
        t_ps=t_sfpt,
        titles=np.asarray(titles),
        xi_lo=xi_lo,
        xi_hi=xi_hi,
    )

    ylab_rmse = (
        rf"1D $F(\phi)$ RMSE ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    _dca.plot_rmse_suite(
        out,
        stem_prefix="fes_rmse",
        ylab=ylab_rmse,
        t_meta=t_meta,
        tw_meta=tw_meta,
        cost_meta=cost_meta,
        rmse_meta=rmse_meta,
        t_sfpt=t_sfpt,
        tw_sfpt=tw_sfpt,
        cost_sfpt=cost_sfpt,
        rmse_sfpt=rmse_sfpt,
        fpt_gpus=int(args.fpt_gpus),
        metad_gpus=int(args.metad_gpus),
        t_max=t_max if t_max > 0 else None,
    )

    vmax_extra = float(args.rmse_vmax_extra)
    if vmax_extra > 0:
        r30_m = [
            rmse_1d(
                f_meta_traj[i],
                f1d_ref,
                vmax=vmax_extra,
                xi=xi,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
            )
            for i in range(len(t_meta))
        ]
        r30_s = [
            rmse_1d(
                f_sfpt[i], f1d_ref, vmax=vmax_extra, xi=xi, xi_lo=xi_lo, xi_hi=xi_hi
            )
            for i in range(len(t_sfpt))
        ]
        np.savez_compressed(
            out / f"fes_rmse_F{int(vmax_extra)}.npz",
            t_metad_ps=t_meta,
            t_kde_ps=t_sfpt,
            tw_metad_h=tw_meta,
            tw_kde_h=tw_sfpt,
            cost_metad_gpuh=cost_meta,
            cost_kde_gpuh=cost_sfpt,
            rmse_metad=np.asarray(r30_m),
            rmse_kde=np.asarray(r30_s),
            vmax_kcal=vmax_extra,
        )
        _dca.plot_rmse_suite(
            out,
            stem_prefix=f"fes_rmse_F{int(vmax_extra)}",
            ylab=(
                rf"1D $F(\phi)$ RMSE ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$, "
                rf"$F_{{\mathrm{{ref}}}}\leq {vmax_extra:g}$) / kcal mol$^{{-1}}$"
            ),
            t_meta=t_meta,
            tw_meta=tw_meta,
            cost_meta=cost_meta,
            rmse_meta=np.asarray(r30_m),
            t_sfpt=t_sfpt,
            tw_sfpt=tw_sfpt,
            cost_sfpt=cost_sfpt,
            rmse_sfpt=np.asarray(r30_s),
            fpt_gpus=int(args.fpt_gpus),
            metad_gpus=int(args.metad_gpus),
            t_max=t_max if t_max > 0 else None,
        )

    ylab_dg = (
        rf"$\Delta G=k_BT\ln(P_{{\phi<0}}/P_{{\phi>0}})$ "
        rf"($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim in [
        ("deltaG_vs_time", None),
        ("deltaG_vs_time_divein", (-5.0, 5.0)),
    ]:
        _dca.plot_series(
            out,
            stem=stem,
            xlabel=r"$t$ / ps",
            ylabel=ylab_dg,
            t_meta=t_meta,
            y_meta=dG_meta,
            t_sfpt=t_sfpt,
            y_sfpt=dG_sfpt,
            dG_ref=float(dG_ref),
            ref_label="MetaD",
            ylim=ylim,
            xlim_right=t_max if t_max > 0 else None,
        )
    for stem, ylim in [
        ("deltaG_vs_wallclock", None),
        ("deltaG_vs_wallclock_divein", (-5.0, 5.0)),
    ]:
        _dca.plot_series(
            out,
            stem=stem,
            xlabel=r"wall-clock time / h",
            ylabel=ylab_dg,
            t_meta=tw_meta,
            y_meta=dG_meta,
            t_sfpt=tw_sfpt,
            y_sfpt=dG_sfpt,
            dG_ref=float(dG_ref),
            ref_label="MetaD",
            ylim=ylim,
        )

    _dca.plot_1d_collage(
        out,
        xi=xi,
        f_meta=f_meta_plot,
        f_sfpt=f_sfpt,
        titles=titles,
        vmax=float(args.collage_vmax),
        ncols=int(args.collage_ncols),
        stages_per_panel=int(args.group_size),
    )

    # Extra: 2D FES from last fraction of stages vs MetaD ref
    plot_2d_fes_last_frac(
        out,
        fpt_dir=fpt_dir,
        hills_path=hills_path,
        phi=phi,
        psi=psi,
        last_frac=float(args.last_frac),
        temperature=float(args.temperature),
        device=device,
        chunk=int(args.chunk),
        n_replicas=int(args.n_replicas),
        deposit_pace=int(args.deposit_pace),
        timestep=float(args.timestep),
        bias_factor=float(args.bias_factor),
        sigma=sigma,
        metad_ref_last_ps=float(args.metad_ref_last_ps),
        vmax=float(args.fes2d_vmax),
        lambda_target=float(args.lambda_target),
    )

    print(
        f"[done] out={out}  sFPT groups={len(t_sfpt)}  "
        f"last RMSE={rmse_sfpt[-1]:.4f} ΔG={dG_sfpt[-1]:.4f}  "
        f"MetaD last RMSE={rmse_meta[-1]:.4f} ΔG={dG_meta[-1]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
