#!/usr/bin/env python3
"""Dialanine φ–ψ KDE FES from sFPT stages vs optional reference.

Default: ``dialanine_gaff_s0_87188``, λ=0.1 stages, **last 50%**, average
torus-Gaussian densities → 2D FES + 1D projections F(φ), F(ψ).

NOTE: ``experiments/dialanine/parity_ref/`` is **not** a hills/FES dump.
It only holds BAOAB-FP JAX↔torch **parity metadata** (``meta.npz``: dt, γ, λ,
bandwidth0, …). Trajectory arrays for that test are mostly absent here.
Pass ``--ref-fes`` if you have a real reference FES npz.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from fp_transformer_torch.kernel.gaussian import (  # noqa: E402
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)
from plot_fes_compare import (  # noqa: E402
    COLOR_META,
    COLOR_SFPT,
    FE_LABEL,
    NATURE_DOUBLE_COL_MM,
    _draw_fes2d_on_ax,
    _rdbu_r_white_over,
    kT_kcal,
    nature_rcparams,
    relative_fes,
)
from plot_kde_fes import to_kernel_cv  # noqa: E402

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
STAGE_RE = re.compile(r"stage(\d+)", re.I)
LAMBDA_RE = re.compile(r"lambda(\d+)p(\d+)", re.I)

DEFAULT_RUN = EXP / "results" / "dialanine_gaff_s0_87188"
DEFAULT_OUT = DEFAULT_RUN / "kde_fes"


def parse_lambda(path: Path) -> float | None:
    m = LAMBDA_RE.search(path.name)
    if m is None:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def parse_stage(path: Path) -> int | None:
    m = STAGE_RE.search(path.name)
    return int(m.group(1)) if m else None


def torus_cross_distance(grid_cv: torch.Tensor, sample_cv: torch.Tensor) -> torch.Tensor:
    """(M,N) torus distance matching ``torus_pairwise_distance`` algebra."""
    m = int(grid_cv.shape[0])
    n = int(sample_cv.shape[0])
    g = grid_cv.reshape(m, -1)
    s = sample_cv.reshape(n, -1)
    feat = float(g.shape[1])
    dots = torch.cos(g) @ torch.cos(s).T + torch.sin(g) @ torch.sin(s).T
    return 2.0 - 2.0 * (dots / feat)


def density_on_grid_torus(
    grid_xy: np.ndarray,
    sample_cv: np.ndarray,
    bandwidth: np.ndarray,
    *,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    samples = to_kernel_cv(sample_cv).to(device=device)
    bw = torch.as_tensor(bandwidth, dtype=torch.float64, device=device).reshape(-1)
    n = float(samples.shape[0])
    m = int(grid_xy.shape[0])
    feature_dim = float(samples.shape[-2] * samples.shape[-1])
    out = np.empty((m,), dtype=np.float64)
    for start in range(0, m, int(chunk)):
        stop = min(start + int(chunk), m)
        grid = to_kernel_cv(grid_xy[start:stop]).to(device=device)
        dist = torus_cross_distance(grid, samples)
        log_k = (-0.5 * dist) / bandwidth_column_squared(bw) + gaussian_kernel_log_normalization(
            bw, feature_dim
        )
        log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
        out[start:stop] = torch.exp(log_p).detach().cpu().numpy()
    return out


def discover_lambda_stages(
    data_dir: Path,
    *,
    lambda_target: float,
    frac_last: float,
) -> list[tuple[int, Path, Path, float]]:
    """Return (stage_idx, dihedral, bandwidth, λ) for last ``frac_last`` of matching stages."""
    rows: list[tuple[int, Path, Path, float]] = []
    for dih in sorted(data_dir.glob("stage[0-9]*_lambda*_dihedral.npy")):
        idx = parse_stage(dih)
        lam = parse_lambda(dih)
        if idx is None or lam is None:
            continue
        if abs(float(lam) - float(lambda_target)) > 1e-6:
            continue
        bw = dih.with_name(dih.name.replace("_dihedral.npy", "_bandwidth.npy"))
        if not bw.is_file():
            continue
        rows.append((idx, dih, bw, float(lam)))
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise FileNotFoundError(
            f"no stage*_lambda≈{lambda_target}_dihedral+bandwidth under {data_dir}"
        )
    n = len(rows)
    i0 = int(np.floor((1.0 - float(frac_last)) * n))
    i0 = min(max(i0, 0), n - 1)
    use = rows[i0:]
    print(
        f"[stages] λ={lambda_target:g}: {n} total → use last {frac_last:.0%} "
        f"= {len(use)} (stage {use[0][0]:04d}–{use[-1][0]:04d})",
        flush=True,
    )
    return use


def project_1d_axis(
    phi: np.ndarray,
    psi: np.ndarray,
    f2d: np.ndarray,
    *,
    axis: str,
    n_bins: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann-project 2D FES onto φ or ψ."""
    g_phi, g_psi = np.meshgrid(phi, psi, indexing="xy")
    if axis == "phi":
        x_flat = g_phi.ravel()
        edges = np.linspace(float(phi.min()), float(phi.max()), int(n_bins) + 1)
    elif axis == "psi":
        x_flat = g_psi.ravel()
        edges = np.linspace(float(psi.min()), float(psi.max()), int(n_bins) + 1)
    else:
        raise ValueError(axis)
    f_flat = np.asarray(f2d, dtype=np.float64).ravel()
    centers = 0.5 * (edges[:-1] + edges[1:])
    kt = kT_kcal(temperature)
    beta = 1.0 / kt
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            sel = (x_flat >= edges[i]) & (x_flat < edges[i + 1])
        else:
            sel = (x_flat >= edges[i]) & (x_flat <= edges[i + 1])
        sel = sel & np.isfinite(f_flat)
        if not np.any(sel):
            continue
        vals = -beta * f_flat[sel]
        m = float(np.max(vals))
        out[i] = -kt * (m + np.log(np.sum(np.exp(vals - m))))
    out = relative_fes(out, None)
    return centers, out


def load_ref_fes_on_grid(
    ref_path: Path,
    phi: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray | None:
    """Load reference FES if present; return kcal/mol on (psi,phi) xy grid or None."""
    if not ref_path.is_file():
        return None
    z = np.load(ref_path)
    keys = set(z.files)
    # Reject parity meta.npz (no F)
    if "F" not in keys and "F_kcal" not in keys and "fes" not in keys:
        print(
            f"[ref] {ref_path} has no F/F_kcal (keys={sorted(keys)[:12]}…) — skip",
            flush=True,
        )
        return None
    if "F_kcal" in keys:
        f = np.asarray(z["F_kcal"], dtype=np.float64)
        x = np.asarray(z["phi"] if "phi" in keys else z["d1"], dtype=np.float64)
        y = np.asarray(z["psi"] if "psi" in keys else z["d2"], dtype=np.float64)
    else:
        f_raw = np.asarray(z["F"] if "F" in keys else z["fes"], dtype=np.float64)
        # Heuristic: if max |F| < 5 treat as eV
        if np.nanmax(np.abs(f_raw)) < 5.0:
            f = f_raw * EV_TO_KCAL
        else:
            f = f_raw
        x = np.asarray(z["phi"] if "phi" in keys else z["d1"], dtype=np.float64)
        y = np.asarray(z["psi"] if "psi" in keys else z["d2"], dtype=np.float64)
    f = relative_fes(f, None)
    from scipy.interpolate import RegularGridInterpolator

    if f.shape == (len(y), len(x)):
        values = f
    elif f.shape == (len(x), len(y)):
        values = f.T
    else:
        raise ValueError(f"ref F shape {f.shape} vs phi={x.shape} psi={y.shape}")
    interp = RegularGridInterpolator(
        (y, x), values, bounds_error=False, fill_value=np.nan
    )
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    pts = np.stack([g2.ravel(), g1.ravel()], axis=-1)
    return relative_fes(interp(pts).reshape(g1.shape), None)


def plot_2d_phi_psi(
    phi: np.ndarray,
    psi: np.ndarray,
    f_kde: np.ndarray,
    f_ref: np.ndarray | None,
    *,
    vmax: float,
    n_levels: int,
    out: Path,
) -> None:
    f_kde_p = relative_fes(f_kde, None)
    levels = np.linspace(0.0, float(vmax), int(n_levels))
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = _rdbu_r_white_over()
    mm = 1.0 / 25.4
    w = NATURE_DOUBLE_COL_MM * mm
    if f_ref is None:
        fig, ax = plt.subplots(figsize=(w * 0.55, w * 0.48))
        im = _draw_fes2d_on_ax(
            ax,
            phi,
            psi,
            f_kde_p,
            levels=levels,
            norm=norm,
            cmap=cmap,
            vmax=vmax,
            panel="(a)",
            title="sFPT-KDE",
            xlabel=r"$\phi$ / rad",
            ylabel=r"$\psi$ / rad",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="max")
        cbar.set_label(FE_LABEL)
        fig.tight_layout()
    else:
        f_ref_p = relative_fes(f_ref, None)
        fig = plt.figure(figsize=(w, w * 0.48))
        gs = gridspec.GridSpec(
            1, 4, figure=fig, width_ratios=[1.0, 0.12, 1.0, 0.06],
            wspace=0.10, left=0.08, right=0.92, bottom=0.14, top=0.88,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])
        im = _draw_fes2d_on_ax(
            ax0, phi, psi, f_kde_p, levels=levels, norm=norm, cmap=cmap,
            vmax=vmax, panel="(a)", title="sFPT-KDE",
            xlabel=r"$\phi$ / rad", ylabel=r"$\psi$ / rad",
        )
        _draw_fes2d_on_ax(
            ax1, phi, psi, f_ref_p, levels=levels, norm=norm, cmap=cmap,
            vmax=vmax, panel="(b)", title="reference",
            xlabel=r"$\phi$ / rad", ylabel=r"$\psi$ / rad", show_ylabel=False,
        )
        fig.canvas.draw()
        pos_b = ax1.get_position()
        cax.set_position([pos_b.x1 + 0.012, pos_b.y0, 0.018, pos_b.height])
        cbar = fig.colorbar(im, cax=cax, extend="max")
        cbar.set_label(FE_LABEL, labelpad=10)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def plot_1d_two(
    x_phi: np.ndarray,
    f_phi: np.ndarray,
    x_psi: np.ndarray,
    f_psi: np.ndarray,
    *,
    f_phi_ref: np.ndarray | None,
    f_psi_ref: np.ndarray | None,
    vmax: float,
    out: Path,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, x, y, yref, lab in [
        (axes[0], x_phi, f_phi, f_phi_ref, r"$\phi$ / rad"),
        (axes[1], x_psi, f_psi, f_psi_ref, r"$\psi$ / rad"),
    ]:
        ax.plot(x, y, color=COLOR_SFPT, lw=1.5, label="sFPT-KDE")
        if yref is not None:
            ax.plot(x, yref, color=COLOR_META, lw=1.3, label="reference")
        ax.set_xlabel(lab)
        ax.set_ylim(0.0, float(vmax))
        ax.axhline(0.0, color="0.5", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(which="both", top=False, right=False)
    axes[0].set_ylabel(FE_LABEL)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] {out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    p.add_argument("--frac-last", type=float, default=0.8, help="Use last this fraction of λ stages")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--grid-n", type=int, default=120)
    p.add_argument("--phi-lim", type=str, default=f"{-np.pi:.6f},{np.pi:.6f}")
    p.add_argument("--psi-lim", type=str, default=f"{-np.pi:.6f},{np.pi:.6f}")
    p.add_argument("--n-bins-1d", type=int, default=80)
    p.add_argument("--vmax-kcal", type=float, default=12.0)
    p.add_argument("--n-levels", type=int, default=25)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--ref-fes",
        type=Path,
        default=None,
        help="Optional reference FES npz (F or F_kcal + phi/psi or d1/d2). "
        "parity_ref/meta.npz is NOT a FES.",
    )
    p.add_argument(
        "--parity-ref",
        type=Path,
        default=EXP / "parity_ref" / "meta.npz",
        help="Inspected and reported; not used as FES.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    plt.rcParams.update(nature_rcparams())
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explain parity_ref
    pref = Path(args.parity_ref)
    if pref.is_file():
        z = np.load(pref)
        print(
            f"[parity_ref] {pref}\n"
            f"  → This is BAOAB-FP JAX↔torch parity metadata, NOT hills and NOT FES.\n"
            f"  → keys={z.files}",
            flush=True,
        )
    else:
        print(f"[parity_ref] missing {pref}", flush=True)

    rows = discover_lambda_stages(
        data_dir, lambda_target=float(args.lambda_), frac_last=float(args.frac_last)
    )
    phi_lo, phi_hi = [float(x) for x in args.phi_lim.split(",")]
    psi_lo, psi_hi = [float(x) for x in args.psi_lim.split(",")]
    phi = np.linspace(phi_lo, phi_hi, int(args.grid_n))
    psi = np.linspace(psi_lo, psi_hi, int(args.grid_n))
    g1, g2 = np.meshgrid(phi, psi, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[kde] device={device}  grid={args.grid_n}²  λ={args.lambda_}", flush=True)

    beta0 = 1.0 / (KB_EV_PER_K * float(args.temperature))
    lam = float(args.lambda_)
    p_acc = np.zeros(grid_xy.shape[0], dtype=np.float64)
    for idx, dih_p, bw_p, _ in rows:
        cv = np.load(dih_p)
        bw = np.load(bw_p)
        if cv.ndim == 3:
            cv = cv.reshape(cv.shape[0], -1)
        p_acc += density_on_grid_torus(
            grid_xy, cv, bw, chunk=int(args.chunk), device=device
        )
        print(f"[kde] stage{idx:04d} done", flush=True)
    p = np.clip(p_acc / float(len(rows)), 1e-300, None)
    f_ev = -np.log(p) / (beta0 * lam)
    f_kcal = relative_fes((f_ev * EV_TO_KCAL).reshape(g1.shape), None)

    f_ref = None
    if args.ref_fes is not None:
        f_ref = load_ref_fes_on_grid(Path(args.ref_fes), phi, psi)

    xi_phi, f1d_phi = project_1d_axis(
        phi, psi, f_kcal, axis="phi", n_bins=int(args.n_bins_1d), temperature=float(args.temperature)
    )
    xi_psi, f1d_psi = project_1d_axis(
        phi, psi, f_kcal, axis="psi", n_bins=int(args.n_bins_1d), temperature=float(args.temperature)
    )
    f1d_phi_ref = f1d_psi_ref = None
    if f_ref is not None:
        _, f1d_phi_ref = project_1d_axis(
            phi, psi, f_ref, axis="phi", n_bins=int(args.n_bins_1d), temperature=float(args.temperature)
        )
        _, f1d_psi_ref = project_1d_axis(
            phi, psi, f_ref, axis="psi", n_bins=int(args.n_bins_1d), temperature=float(args.temperature)
        )

    np.savez_compressed(
        out_dir / "kde_fes_last50pct_lam0p1.npz",
        phi=phi,
        psi=psi,
        F_kcal=f_kcal,
        F_ref_kcal=f_ref if f_ref is not None else np.array([]),
        xi_phi=xi_phi,
        F1d_phi_kcal=f1d_phi,
        xi_psi=xi_psi,
        F1d_psi_kcal=f1d_psi,
        temperature=float(args.temperature),
        lambda_=lam,
        stages=np.asarray([r[0] for r in rows]),
        frac_last=float(args.frac_last),
        data_dir=str(data_dir),
    )

    plot_2d_phi_psi(
        phi, psi, f_kcal, f_ref,
        vmax=float(args.vmax_kcal), n_levels=int(args.n_levels),
        out=out_dir / "fes_compare_2d_phi_psi.png",
    )
    plot_1d_two(
        xi_phi, f1d_phi, xi_psi, f1d_psi,
        f_phi_ref=f1d_phi_ref, f_psi_ref=f1d_psi_ref,
        vmax=float(args.vmax_kcal),
        out=out_dir / "fes_compare_1d_phi_psi.png",
    )
    print(
        f"[done] out={out_dir}  2D Fmax={np.nanmax(f_kcal):.2f}  "
        f"1D φ Fmax={np.nanmax(f1d_phi):.2f}  ψ Fmax={np.nanmax(f1d_psi):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
