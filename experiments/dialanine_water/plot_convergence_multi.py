#!/usr/bin/env python3
"""Multi-seed dialanine-water convergence: 3×sFPT + 2×MetaD (mean±std) + US-MBAR ref.

- sFPT: torus-KDE F(φ) per stage-group, truncated to shortest run; mean±std
- MetaD: incremental F(φ) mean±std; final 2D FES averaged over seeds
- 1D RMSE / ΔG reference: US-MBAR F(φ) from ``mbar_fes_2d.py``
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
from plot_g5_1d_grid_and_dg_zoom import COLOR_US, fpt_wall_hours_from_throughput  # noqa: E402

KB_EV_PER_K = _dca.KB_EV_PER_K
EV_TO_KCAL = _dca.EV_TO_KCAL
TWO_PI = _dca.TWO_PI


def mean_std(stack: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=np.float64)
    mean = np.nanmean(stack, axis=axis)
    n = stack.shape[axis]
    if n <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(stack, axis=axis, ddof=1)


def load_us_f1d(path: Path, xi_query: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if "xi_phi" in z.files and "F_phi" in z.files:
        xi = np.asarray(z["xi_phi"], dtype=np.float64)
        f = relative_fes(np.asarray(z["F_phi"], dtype=np.float64), None)
    elif "xi" in z.files and "F_kcal" in z.files:
        xi = np.asarray(z["xi"], dtype=np.float64)
        f = relative_fes(np.asarray(z["F_kcal"], dtype=np.float64), None)
    else:
        raise ValueError(f"{path} missing xi_phi/F_phi or xi/F_kcal")
    if xi_query is not None and (
        len(xi) != len(xi_query) or not np.allclose(xi, xi_query)
    ):
        f = np.interp(xi_query, xi, f, left=np.nan, right=np.nan)
        f = relative_fes(f, None)
        xi = np.asarray(xi_query, dtype=np.float64)
    return xi, f


def load_metad_2d(fes_path: Path, grid_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(fes_path)
    phi = np.asarray(z["phi"], dtype=np.float64)
    psi = np.asarray(z["psi"], dtype=np.float64)
    f_ev = np.asarray(z["F"], dtype=np.float64)
    f_kcal = relative_fes(f_ev * EV_TO_KCAL, None)
    if f_kcal.shape == (len(phi), len(psi)) and f_kcal.shape != (len(psi), len(phi)):
        f_kcal = f_kcal.T
    if int(grid_n) != len(phi) or int(grid_n) != len(psi):
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (psi, phi), f_kcal, bounds_error=False, fill_value=None
        )
        phi = np.linspace(-np.pi, np.pi, int(grid_n))
        psi = np.linspace(-np.pi, np.pi, int(grid_n))
        g_phi, g_psi = np.meshgrid(phi, psi, indexing="xy")
        f_kcal = relative_fes(
            interp(np.stack([g_psi.ravel(), g_phi.ravel()], axis=-1)).reshape(
                g_phi.shape
            ),
            None,
        )
    return phi, psi, f_kcal


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
    ref_label: str = "US",
    ylim=None,
    y0_line: bool = False,
    xlim: tuple[float, float] | None = None,
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
            alpha=0.95,
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
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
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
    f_us,
    f_sfpt_mean,
    f_sfpt_std,
    titles,
    vmax: float,
    ncols: int,
    group_size: int,
    subtitle: str | None = None,
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
    y_meta = np.minimum(relative_fes(f_meta_mean, None), float(vmax)) if vmax > 0 else relative_fes(f_meta_mean, None)
    y_mlo = relative_fes(f_meta_mean - f_meta_std, None)
    y_mhi = relative_fes(f_meta_mean + f_meta_std, None)
    if vmax > 0:
        y_mlo = np.minimum(y_mlo, float(vmax))
        y_mhi = np.minimum(y_mhi, float(vmax))
    y_us = relative_fes(f_us, None)
    if vmax > 0:
        y_us = np.minimum(y_us, float(vmax))
    ymax_auto = 0.0
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ym = relative_fes(f_sfpt_mean[ig], None)
        ylo = relative_fes(f_sfpt_mean[ig] - f_sfpt_std[ig], None)
        yhi = relative_fes(f_sfpt_mean[ig] + f_sfpt_std[ig], None)
        if vmax > 0:
            ym = np.minimum(ym, float(vmax))
            ylo = np.minimum(ylo, float(vmax))
            yhi = np.minimum(yhi, float(vmax))
        ax.fill_between(xi, y_mlo, y_mhi, color=COLOR_META, alpha=0.18, lw=0, zorder=1)
        ax.plot(xi, y_meta, color=COLOR_META, lw=1.05, label="WT-MetaD mean", zorder=2)
        ax.fill_between(xi, ylo, yhi, color=COLOR_SFPT, alpha=0.22, lw=0, zorder=3)
        ax.plot(xi, ym, color=COLOR_SFPT, lw=1.25, label="sFPT mean", zorder=4)
        # US on top (often close to MetaD mean — keep dashed + higher zorder)
        ax.plot(
            xi,
            y_us,
            color=COLOR_US,
            lw=1.55,
            ls="--",
            label="US-MBAR $F(\\phi)$",
            zorder=6,
        )
        if vmax > 0:
            ax.set_ylim(0.0, float(vmax) * 1.02)
        else:
            ymax_auto = max(ymax_auto, float(np.nanmax(yhi)), float(np.nanmax(y_mhi)), float(np.nanmax(y_us)))
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
    if vmax <= 0 and ymax_auto > 0:
        for ax in fig.axes:
            if ax.has_data():
                ax.set_ylim(0.0, ymax_auto * 1.05)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        subtitle
        if subtitle
        else f"1D $F(\\phi)$  MetaD/US/sFPT mean±std  ({group_size} stages / panel)",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print("[ok] fes_1d_all_groups", flush=True)


def plot_2d_metad_mean(
    out: Path,
    phi,
    psi,
    F_mean,
    F_std,
    vmax: float,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    levels = np.linspace(0.0, float(vmax), 25)
    Z = np.minimum(relative_fes(F_mean, None), float(vmax))
    im0 = axes[0].contourf(phi, psi, Z, levels=levels, cmap="viridis")
    axes[0].contour(phi, psi, Z, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.35)
    axes[0].set_title("WT-MetaD mean $F(\\phi,\\psi)$", fontsize=9)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(FE_LABEL, fontsize=8)
    S = np.minimum(np.nan_to_num(F_std, nan=0.0), float(vmax))
    im1 = axes[1].contourf(phi, psi, S, levels=25, cmap="magma")
    axes[1].set_title(r"WT-MetaD std across seeds", fontsize=9)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"std / kcal mol$^{-1}$", fontsize=8)
    for ax in axes:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"$\phi$ / rad")
        ax.set_ylabel(r"$\psi$ / rad")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "fes_2d_metad_mean.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fes_2d_metad_mean.pdf", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out / "fes_2d_metad_mean.npz", phi=phi, psi=psi, F_mean=F_mean, F_std=F_std)
    print("[ok] fes_2d_metad_mean", flush=True)


def plot_2d_sfpt_vs_refs(
    out: Path,
    *,
    phi,
    psi,
    F_sfpt,
    F_sfpt_std,
    F_meta,
    F_us_2d,
    vmax: float,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), sharex=True, sharey=True)
    levels = np.linspace(0.0, float(vmax), 25)
    panels = [
        (F_sfpt, "sFPT mean (last 20%)"),
        (F_meta, "WT-MetaD mean"),
        (F_us_2d, "US-MBAR"),
    ]
    for ax, (F, title) in zip(axes, panels):
        Z = np.minimum(relative_fes(F, None), float(vmax))
        im = ax.contourf(phi, psi, Z, levels=levels, cmap="viridis")
        ax.contour(phi, psi, Z, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.35)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"$\phi$ / rad")
        ax.set_ylabel(r"$\psi$ / rad")
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(FE_LABEL, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fes_2d_last_frac.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fes_2d_last_frac.pdf", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        out / "fes_2d_last_frac.npz",
        phi=phi,
        psi=psi,
        F_sfpt=F_sfpt,
        F_sfpt_std=F_sfpt_std,
        F_metad=F_meta,
        F_us=F_us_2d,
    )
    print("[ok] fes_2d_last_frac", flush=True)


def kde_2d_last_frac(
    fpt_dir: Path,
    *,
    phi,
    psi,
    last_frac,
    temperature,
    device,
    chunk,
    lambda_target,
    lambda_atol: float = 1e-3,
):
    rows = _dca.discover_dihedral_stages(
        fpt_dir,
        lambda_target=float(lambda_target),
        lambda_atol=float(lambda_atol),
    )
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    beta0 = 1.0 / (KB_EV_PER_K * float(temperature))
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    lam = float(use[0][3])
    for _, dih_p, bw_p, _ in use:
        p_acc += _dca.density_on_grid_torus(
            grid_xy, np.load(dih_p), np.load(bw_p), chunk=chunk, device=device
        )
    p = np.clip(p_acc / float(len(use)), 1e-300, None)
    return relative_fes((-np.log(p) / (beta0 * lam) * EV_TO_KCAL).reshape(g1.shape), None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fpt-dirs", type=str, required=True)
    p.add_argument("--metad-dirs", type=str, required=True)
    p.add_argument("--us-mbar", type=Path, required=True)
    p.add_argument("--group-size", type=int, default=20)
    p.add_argument("--dt-group", type=float, default=2.0)
    p.add_argument("--dt-stage", type=float, default=0.1)
    p.add_argument("--grid-n", type=int, default=73)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--vmax-kcal", type=float, default=0.0)
    p.add_argument("--rmse-vmax-extra", type=float, default=0.0)
    p.add_argument("--collage-vmax", type=float, default=20.0)
    p.add_argument("--fes2d-vmax", type=float, default=20.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--n-replicas", type=int, default=64)
    p.add_argument("--deposit-pace", type=int, default=50)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--bias-factor", type=float, default=15.0)
    p.add_argument("--sigma", type=str, default="0.35,0.35")
    p.add_argument("--metad-stride-deposits", type=int, default=10)
    p.add_argument("--metad-ref-last-ps", type=float, default=0.0)
    p.add_argument("--xi-cut", type=float, default=0.0)
    p.add_argument("--xi-lo", type=float, default=-np.pi)
    p.add_argument("--xi-hi", type=float, default=np.pi)
    p.add_argument("--fpt-rate", type=float, default=19.7)
    p.add_argument("--metad-rate", type=float, default=44.0)
    p.add_argument("--fpt-gpus", type=int, default=1)
    p.add_argument("--metad-gpus", type=int, default=1)
    p.add_argument("--collage-ncols", type=int, default=4)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--last-frac", type=float, default=0.2)
    p.add_argument(
        "--sfpt-window",
        type=str,
        default="group",
        choices=("group", "lastfrac"),
        help="group: disjoint group_size stages; lastfrac: last last-frac of stages up to t",
    )
    p.add_argument("--lambda-target", type=float, default=0.0625)
    p.add_argument(
        "--lambda-atol",
        type=float,
        default=1e-3,
        help="|λ_file − λ_target| tolerance when discovering stages",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fpt_dirs = [Path(x.strip()).resolve() for x in args.fpt_dirs.split(",") if x.strip()]
    metad_dirs = [Path(x.strip()).resolve() for x in args.metad_dirs.split(",") if x.strip()]
    us_path = Path(args.us_mbar).resolve()
    if not us_path.is_file():
        raise FileNotFoundError(f"missing US MBAR {us_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda → cpu", flush=True)
    sigma = tuple(float(x) for x in args.sigma.replace(";", ",").split(",") if x.strip())
    xi_lo, xi_hi, xi_cut = float(args.xi_lo), float(args.xi_hi), float(args.xi_cut)
    vmax = float(args.vmax_kcal) if args.vmax_kcal > 0 else None

    # MetaD 2D mean±std
    f2_meta_list = []
    phi = psi = None
    for md in metad_dirs:
        fes = md / "fes.npz"
        if not fes.is_file():
            raise FileNotFoundError(fes)
        phi_i, psi_i, f2 = load_metad_2d(fes, int(args.grid_n))
        if phi is None:
            phi, psi = phi_i, psi_i
        f2_meta_list.append(f2)
        print(f"[metad2d] {md}  Fmax={np.nanmax(f2):.2f}", flush=True)
    f2_stack = np.stack(f2_meta_list, axis=0)
    f2_meta, f2_meta_std = mean_std(f2_stack, axis=0)
    f2_meta = relative_fes(f2_meta, None)
    plot_2d_metad_mean(out, phi, psi, f2_meta, f2_meta_std, float(args.fes2d_vmax))

    # MetaD final 1D mean±std (for collage)
    f1d_meta_list = []
    xi = None
    for f2 in f2_meta_list:
        xi_i, f1d_i = _dca.project_1d_axis(
            phi, psi, f2, axis="phi", n_bins=int(args.n_bins_1d), temperature=float(args.temperature)
        )
        if xi is None:
            xi = xi_i
        else:
            if not np.allclose(xi_i, xi):
                f1d_i = np.interp(xi, xi_i, f1d_i, left=np.nan, right=np.nan)
                f1d_i = relative_fes(f1d_i, None)
        f1d_meta_list.append(f1d_i)
    f_meta_stack = np.stack(f1d_meta_list, axis=0)
    f1d_meta, f1d_meta_std = mean_std(f_meta_stack, axis=0)
    f1d_meta = relative_fes(f1d_meta, None)

    # US 1D reference
    xi_us, f1d_us = load_us_f1d(us_path, xi_query=xi)
    f1d_ref = relative_fes(f1d_us, None)
    dG_ref = delta_g_two_basins(
        xi, f1d_ref, temperature=float(args.temperature), xi_cut=xi_cut, xi_lo=xi_lo, xi_hi=xi_hi
    )
    print(f"[us] ΔG_ref={dG_ref:.4f} kcal/mol  (RMSE reference)", flush=True)

    # US 2D for panel (resample if needed)
    zus = np.load(us_path)
    if "F" in zus.files and "phi" in zus.files:
        from scipy.interpolate import RegularGridInterpolator

        Fu = relative_fes(np.asarray(zus["F"], dtype=np.float64), None)
        pu, su = np.asarray(zus["phi"]), np.asarray(zus["psi"])
        if Fu.shape == (len(pu), len(su)):
            Fu = Fu.T
        interp = RegularGridInterpolator((su, pu), Fu, bounds_error=False, fill_value=None)
        g_phi, g_psi = np.meshgrid(phi, psi, indexing="xy")
        F_us_2d = relative_fes(
            interp(np.stack([g_psi.ravel(), g_phi.ravel()], axis=-1)).reshape(g_phi.shape),
            None,
        )
    else:
        F_us_2d = np.full_like(f2_meta, np.nan)

    # MetaD incremental mean±std vs US
    t_m_list, rmse_m_list, dG_m_list = [], [], []
    for md in metad_dirs:
        hills = md / "hills.npz"
        t_m, rmse_m, dG_m, _ = _dca.incremental_metad_phi(
            hills,
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
        print(f"[metad-inc] {md.name} n={len(t_m)} t_end={t_m[-1]:.1f}", flush=True)

    n_m = min(len(t) for t in t_m_list)
    t_meta = t_m_list[0][:n_m]
    rmse_meta, rmse_meta_std = mean_std(np.stack([r[:n_m] for r in rmse_m_list], 0), 0)
    dG_meta, dG_meta_std = mean_std(np.stack([g[:n_m] for g in dG_m_list], 0), 0)

    # sFPT groups mean±std
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
    if args.max_groups is not None:
        n_groups = min(n_groups, int(args.max_groups))
    window = str(args.sfpt_window)
    print(
        f"[sFPT] stages={n_stages_each} groups={n_groups_each} → use {n_groups} "
        f"({n_groups * args.dt_group:g} ps)  λ={args.lambda_target}  window={window}"
        + (f" last_frac={args.last_frac}" if window == "lastfrac" else ""),
        flush=True,
    )

    kde_kw = dict(
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
    kde_fn = (
        _dca.kde_rolling_lastfrac_phi if window == "lastfrac" else _dca.kde_groups_phi
    )
    if window == "lastfrac":
        kde_kw["last_frac"] = float(args.last_frac)

    f_sfpt_runs, rmse_runs, dG_runs, titles = [], [], [], None
    t_sfpt = None
    for d in fpt_dirs:
        t_s, r_s, g_s, f_s, titles_s = kde_fn(d, **kde_kw)
        f_sfpt_runs.append(f_s)
        rmse_runs.append(r_s)
        dG_runs.append(g_s)
        if t_sfpt is None:
            t_sfpt = t_s
            titles = titles_s
        print(f"[kde] {d.name} groups={len(t_s)} last RMSE={r_s[-1]:.4f}", flush=True)

    f_sfpt_stack = np.stack(f_sfpt_runs, axis=0)
    f_sfpt, f_sfpt_std = mean_std(f_sfpt_stack, axis=0)
    rmse_sfpt, rmse_sfpt_std = mean_std(np.stack(rmse_runs, 0), 0)
    dG_sfpt, dG_sfpt_std = mean_std(np.stack(dG_runs, 0), 0)

    tw_sfpt = fpt_wall_hours_from_throughput(
        t_sfpt, timestep_ps=float(args.timestep), rate_it_s=float(args.fpt_rate)
    )
    tw_meta = fpt_wall_hours_from_throughput(
        t_meta, timestep_ps=float(args.timestep), rate_it_s=float(args.metad_rate)
    )
    cost_sfpt = tw_sfpt * float(args.fpt_gpus)
    cost_meta = tw_meta * float(args.metad_gpus)

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
        dG_us_ref=dG_ref,
        xi=xi,
        F1d_us_ref=f1d_ref,
        F1d_metad_mean=f1d_meta,
        F1d_metad_std=f1d_meta_std,
        group_size=int(args.group_size),
        dt_group_ps=float(args.dt_group),
        fpt_dirs=[str(d) for d in fpt_dirs],
        metad_dirs=[str(d) for d in metad_dirs],
        us_mbar=str(us_path),
        lambda_target=float(args.lambda_target),
        sfpt_window=window,
        last_frac=float(args.last_frac),
        fpt_rate=float(args.fpt_rate),
        fpt_gpus=int(args.fpt_gpus),
        metad_rate=float(args.metad_rate),
        metad_gpus=int(args.metad_gpus),
    )
    np.savez_compressed(
        out / "fes_1d_all_groups.npz",
        xi=xi,
        F1d_metad_mean=f1d_meta,
        F1d_metad_std=f1d_meta_std,
        F1d_us=f1d_ref,
        F1d_kde_mean=f_sfpt,
        F1d_kde_std=f_sfpt_std,
        F1d_kde_runs=f_sfpt_stack,
        t_ps=t_sfpt,
        titles=np.asarray(titles),
    )

    ylab_rmse = (
        rf"1D $F(\phi)$ RMSE vs US ($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim, xm, xs, xlab, xlim in [
        ("fes_rmse_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", None),
        (
            "fes_rmse_vs_time_divein",
            (0.0, 3.0),
            t_meta,
            t_sfpt,
            r"$t$ / ps",
            (0.0, 100.0),
        ),
        ("fes_rmse_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            "fes_rmse_vs_wallclock_divein",
            (0.0, 3.0),
            tw_meta,
            tw_sfpt,
            r"wall-clock time / h",
            (0.0, 1.0),
        ),
        ("fes_rmse_vs_cost", None, cost_meta, cost_sfpt, r"cost / GPU$\cdot$h", None),
        ("fes_rmse_vs_cost_divein", (0.0, 8.0), cost_meta, cost_sfpt, r"cost / GPU$\cdot$h", None),
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
            y0_line=True,
            ylim=ylim,
            xlim=xlim,
        )

    ylab_dg = (
        rf"$\Delta G=k_BT\ln(P_{{\phi<0}}/P_{{\phi>0}})$ "
        rf"($\phi\in[{xi_lo:.3g},{xi_hi:.3g}]$) / kcal mol$^{{-1}}$"
    )
    for stem, ylim, xm, xs, xlab, xlim in [
        ("deltaG_vs_time", None, t_meta, t_sfpt, r"$t$ / ps", None),
        (
            "deltaG_vs_time_divein",
            (2.5, 6.5),
            t_meta,
            t_sfpt,
            r"$t$ / ps",
            (0.0, 200.0),
        ),
        ("deltaG_vs_wallclock", None, tw_meta, tw_sfpt, r"wall-clock time / h", None),
        (
            "deltaG_vs_wallclock_divein",
            (2.5, 6.5),
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
            ref_label="US",
            ylim=ylim,
            xlim=xlim,
        )

    plot_1d_collage(
        out,
        xi=xi,
        f_meta_mean=f1d_meta,
        f_meta_std=f1d_meta_std,
        f_us=f1d_ref,
        f_sfpt_mean=f_sfpt,
        f_sfpt_std=f_sfpt_std,
        titles=titles,
        vmax=float(args.collage_vmax),
        ncols=int(args.collage_ncols),
        group_size=int(args.group_size),
        subtitle=(
            rf"1D $F(\phi)$  MetaD/US/sFPT  (last {100.0 * float(args.last_frac):.0f}% of trajectory to $t$)"
            if window == "lastfrac"
            else None
        ),
    )

    # 2D sFPT last-frac mean across seeds
    f2_s_list = []
    for d in fpt_dirs:
        f2_s_list.append(
            kde_2d_last_frac(
                d,
                phi=phi,
                psi=psi,
                last_frac=float(args.last_frac),
                temperature=float(args.temperature),
                device=device,
                chunk=int(args.chunk),
                lambda_target=float(args.lambda_target),
                lambda_atol=float(args.lambda_atol),
            )
        )
    f2_s, f2_s_std = mean_std(np.stack(f2_s_list, 0), 0)
    f2_s = relative_fes(f2_s, None)
    plot_2d_sfpt_vs_refs(
        out,
        phi=phi,
        psi=psi,
        F_sfpt=f2_s,
        F_sfpt_std=f2_s_std,
        F_meta=f2_meta,
        F_us_2d=F_us_2d,
        vmax=float(args.fes2d_vmax),
    )

    print(
        f"[done] out={out}  sFPT groups={len(t_sfpt)}  "
        f"RMSE={rmse_sfpt[-1]:.4f}±{rmse_sfpt_std[-1]:.4f}  "
        f"ΔG={dG_sfpt[-1]:.4f}±{dG_sfpt_std[-1]:.4f}  US_ref={dG_ref:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
