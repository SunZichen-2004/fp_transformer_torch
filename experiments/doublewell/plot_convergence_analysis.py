#!/usr/bin/env python3
"""Double-well sFPT convergence: 1D KDE F(x1) vs analytical free energy.

Mirrors the dialanine_water ``convergence_320`` figure suite for the toy
system (CV=``x1``, reduced units ``k_B=T=1``):

- F(x1) collage over stage-groups
- RMSE(t) / |ΔG|(t) vs analytical reference
  ``F_ref(x1)=h(x1²−1)²`` (x2 integrates out to a constant)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_mace"))
sys.path.insert(0, str(ROOT / "experiments" / "rearrangement_water_fpt"))

from fp_transformer_torch.kernel.gaussian import (  # noqa: E402
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)
from plot_fes_compare import COLOR_SFPT, nature_rcparams, relative_fes  # noqa: E402
from plot_fes_rmse_vs_time import rmse_1d  # noqa: E402

STAGE_RE = re.compile(
    r"stage(\d+)_lambda([0-9p]+)_x1\.npy$", re.I
)
COLOR_REF = "#d62728"


def parse_lambda_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def discover_x1_stages(
    data_dir: Path,
    *,
    lambda_target: float = 0.1,
    lambda_atol: float = 1e-3,
) -> list[tuple[int, Path, Path, float]]:
    rows: list[tuple[int, Path, Path, float]] = []
    for x1_p in sorted(data_dir.glob("stage[0-9]*_lambda*_x1.npy")):
        m = STAGE_RE.search(x1_p.name)
        if not m:
            continue
        idx = int(m.group(1))
        lam = parse_lambda_tag(m.group(2))
        if abs(lam - float(lambda_target)) > float(lambda_atol):
            continue
        bw_p = x1_p.with_name(x1_p.name.replace("_x1.npy", "_bandwidth.npy"))
        if not bw_p.is_file():
            continue
        rows.append((idx, x1_p, bw_p, lam))
    rows.sort(key=lambda r: r[0])
    return rows


def density_on_grid_1d(
    grid_x: np.ndarray,
    sample_x: np.ndarray,
    bandwidth: np.ndarray,
    *,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    """Normalized Gaussian KDE on 1D grid; returns ``(M,)``."""
    sx = np.asarray(sample_x, dtype=np.float64).reshape(-1)
    bw = np.asarray(bandwidth, dtype=np.float64).reshape(-1)
    if sx.shape[0] != bw.shape[0]:
        raise ValueError(f"x1 len {sx.shape[0]} != bw len {bw.shape[0]}")
    # CV layout matching CartesianComponent: (N, 1, 1)
    samples = torch.as_tensor(sx, dtype=torch.float64, device=device).reshape(-1, 1, 1)
    bw_t = torch.as_tensor(bw, dtype=torch.float64, device=device)
    n = float(samples.shape[0])
    feature_dim = 1.0
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    m = int(gx.shape[0])
    out = np.empty((m,), dtype=np.float64)
    for start in range(0, m, int(chunk)):
        stop = min(start + int(chunk), m)
        grid = torch.as_tensor(
            gx[start:stop], dtype=torch.float64, device=device
        ).reshape(-1, 1, 1)
        g = grid.reshape(stop - start, -1)
        s = samples.reshape(int(n), -1)
        g2 = torch.sum(g * g, dim=1, keepdim=True)
        s2 = torch.sum(s * s, dim=1)
        dist = g2 + s2.unsqueeze(0) - 2.0 * (g @ s.T)
        dist = torch.clamp(dist, min=0.0) / feature_dim
        log_k = (-0.5 * dist) / bandwidth_column_squared(bw_t) + gaussian_kernel_log_normalization(
            bw_t, feature_dim
        )
        log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
        out[start:stop] = torch.exp(log_p).detach().cpu().numpy()
    return out


def analytical_f_x1(xi: np.ndarray, h: float) -> np.ndarray:
    """Exact F(x1) up to additive constant (x2 is independent harmonic)."""
    x = np.asarray(xi, dtype=np.float64)
    return relative_fes(float(h) * (x * x - 1.0) ** 2, None)


def delta_g_two_wells(
    xi: np.ndarray,
    f: np.ndarray,
    *,
    temperature: float,
    xi_cut: float = 0.0,
) -> float:
    """ΔG = kT ln(P_{x1<cut} / P_{x1>cut}); k_B=1."""
    xi_a = np.asarray(xi, dtype=np.float64)
    f_a = np.asarray(f, dtype=np.float64)
    m = np.isfinite(f_a)
    kt = float(temperature)
    beta = 1.0 / kt
    left = m & (xi_a < float(xi_cut))
    right = m & (xi_a > float(xi_cut))
    if not (np.any(left) and np.any(right)):
        return float("nan")
    # log-sum-exp for numerical stability
    def _logZ(sel: np.ndarray) -> float:
        v = -beta * f_a[sel]
        mval = float(np.max(v))
        return mval + float(np.log(np.sum(np.exp(v - mval))))

    return float(kt * (_logZ(left) - _logZ(right)))


def kde_groups_x1(
    fpt_dir: Path,
    *,
    xi: np.ndarray,
    f_ref: np.ndarray,
    group_size: int,
    dt_group: float,
    temperature: float,
    beta0: float,
    vmax: float | None,
    device: torch.device,
    chunk: int,
    max_groups: int | None,
    lambda_target: float,
    lambda_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = discover_x1_stages(
        fpt_dir, lambda_target=lambda_target, lambda_atol=lambda_atol
    )
    n_groups = len(rows) // int(group_size)
    if max_groups is not None:
        n_groups = min(n_groups, int(max_groups))
    if n_groups < 1:
        raise RuntimeError(f"no complete groups in {fpt_dir} (stages={len(rows)})")
    print(f"[scan] stages={len(rows)}  groups={n_groups}  group_size={group_size}", flush=True)

    t_list, r_list, g_list, f_list, titles = [], [], [], [], []
    for ig in range(n_groups):
        chunk_rows = rows[ig * group_size : (ig + 1) * group_size]
        t = float(ig + 1) * float(dt_group)
        lam = float(chunk_rows[0][3])
        p_acc = np.zeros(xi.shape[0], dtype=np.float64)
        for _, x1_p, bw_p, _ in chunk_rows:
            p_acc += density_on_grid_1d(
                xi,
                np.load(x1_p),
                np.load(bw_p),
                chunk=chunk,
                device=device,
            )
        p = np.clip(p_acc / float(len(chunk_rows)), 1e-300, None)
        f1d = relative_fes((-np.log(p) / (float(beta0) * lam)), None)
        rmse = rmse_1d(
            f1d, f_ref, vmax=vmax, xi=xi, xi_lo=float(xi.min()), xi_hi=float(xi.max())
        )
        dG = delta_g_two_wells(xi, f1d, temperature=temperature, xi_cut=0.0)
        s0, s1 = chunk_rows[0][0], chunk_rows[-1][0]
        title = f"t={t:g}  stage{s0:04d}–{s1:04d}"
        t_list.append(t)
        r_list.append(rmse)
        g_list.append(dG)
        f_list.append(f1d)
        titles.append(title)
        print(f"[kde] {title}  RMSE={rmse:.4f}  ΔG={dG:.4f}", flush=True)
    return (
        np.asarray(t_list),
        np.asarray(r_list),
        np.asarray(g_list),
        np.stack(f_list, axis=0),
        titles,
    )


def subsample_panels(
    t, rmse, dG, F, titles, max_panels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    n = len(titles)
    if max_panels <= 0 or n <= max_panels:
        idx = np.arange(n)
    else:
        idx = np.unique(
            np.round(np.linspace(0, n - 1, int(max_panels))).astype(int)
        )
    return t[idx], rmse[idx], dG[idx], F[idx], [titles[i] for i in idx], idx


def plot_collage(
    out: Path,
    *,
    xi: np.ndarray,
    f_ref: np.ndarray,
    f_sfpt: np.ndarray,
    titles: list[str],
    vmax: float,
    ncols: int,
    group_size: int,
) -> None:
    n = len(titles)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    plt.rcParams.update(nature_rcparams())
    fig = plt.figure(figsize=(2.35 * ncols, 1.95 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, wspace=0.28, hspace=0.45,
        left=0.06, right=0.99, bottom=0.06, top=0.94,
    )
    y_ref = relative_fes(f_ref, None)
    if vmax > 0:
        y_ref = np.minimum(y_ref, float(vmax))
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ys = relative_fes(f_sfpt[ig], None)
        if vmax > 0:
            ys = np.minimum(ys, float(vmax))
        ax.plot(xi, y_ref, color=COLOR_REF, lw=1.2, ls="--", label=r"$F_{\mathrm{exact}}$", zorder=2)
        ax.plot(xi, ys, color=COLOR_SFPT, lw=1.3, label="sFPT", zorder=3)
        ax.axvline(0.0, color="0.5", ls=":", lw=0.7, zorder=0)
        ax.set_xlim(float(xi.min()), float(xi.max()))
        if vmax > 0:
            ax.set_ylim(0.0, float(vmax) * 1.02)
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n:
            ax.set_xlabel(r"$x_1$", fontsize=8)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel(r"$F(x_1)$ / $k_BT$", fontsize=8)
        else:
            ax.set_yticklabels([])
        if ig == 0:
            ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
    fig.suptitle(
        rf"1D $F(x_1)$  exact / sFPT-KDE  ({group_size} stages / panel)",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print("[ok] fes_1d_all_groups", flush=True)


def kde_last_frac_x1(
    fpt_dir: Path,
    *,
    xi: np.ndarray,
    last_frac: float,
    beta0: float,
    device: torch.device,
    chunk: int,
    lambda_target: float,
    lambda_atol: float,
) -> tuple[np.ndarray, int, int, int, float]:
    rows = discover_x1_stages(
        fpt_dir, lambda_target=lambda_target, lambda_atol=lambda_atol
    )
    if not rows:
        raise RuntimeError(f"no λ≈{lambda_target:g} stages in {fpt_dir}")
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    lam = float(use[0][3])
    p_acc = np.zeros(xi.shape[0], dtype=np.float64)
    for i, (_, x1_p, bw_p, _) in enumerate(use):
        p_acc += density_on_grid_1d(
            xi, np.load(x1_p), np.load(bw_p), chunk=chunk, device=device
        )
        if (i + 1) % 20 == 0 or i + 1 == len(use):
            print(f"[last-frac] kde {i+1}/{len(use)}", flush=True)
    p = np.clip(p_acc / float(len(use)), 1e-300, None)
    f1d = relative_fes((-np.log(p) / (float(beta0) * lam)), None)
    return f1d, int(use[0][0]), int(use[-1][0]), int(n_keep), lam


def plot_1d_last_frac(
    out: Path,
    *,
    xi: np.ndarray,
    f_sfpt: np.ndarray,
    f_ref: np.ndarray,
    vmax: float,
    last_frac: float,
    stage_lo: int,
    stage_hi: int,
    n_keep: int,
    rmse: float,
    dG: float,
    dG_ref: float,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y_ref = relative_fes(f_ref, None)
    y = relative_fes(f_sfpt, None)
    if vmax > 0:
        y_ref = np.minimum(y_ref, float(vmax))
        y = np.minimum(y, float(vmax))
    ax.plot(xi, y_ref, color=COLOR_REF, lw=1.6, ls="--", label=r"$F_{\mathrm{exact}}$", zorder=2)
    ax.plot(xi, y, color=COLOR_SFPT, lw=1.6, label="sFPT KDE", zorder=3)
    ax.axvline(0.0, color="0.5", ls=":", lw=0.8, zorder=0)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$F(x_1)$ / $k_BT$")
    ax.set_xlim(float(xi.min()), float(xi.max()))
    if vmax > 0:
        ax.set_ylim(0.0, float(vmax) * 1.02)
    ax.set_title(
        rf"last {float(last_frac):.0%}  stage {stage_lo}–{stage_hi} (n={n_keep})"
        "\n"
        rf"RMSE={rmse:.3g}  $\Delta G$={dG:.3g} (exact {dG_ref:.3g})",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "fes_1d_last_frac.png", dpi=200)
    fig.savefig(out / "fes_1d_last_frac.pdf")
    plt.close(fig)
    print("[ok] fes_1d_last_frac", flush=True)


def plot_series(
    out: Path,
    *,
    stem: str,
    t: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    y0: float | None = None,
    ylim: tuple[float, float] | None = None,
    y0_line: bool = False,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if y0 is not None:
        ax.axhline(y0, color=COLOR_REF, ls=":", lw=1.2, label=rf"exact={y0:g}", zorder=1)
        ax.axhspan(y0 - 1.0, y0 + 1.0, color=COLOR_REF, alpha=0.12, lw=0, zorder=0)
    if y0_line:
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.plot(t, y, color=COLOR_SFPT, lw=1.5, marker="o", ms=3.0, label="sFPT", zorder=3)
    ax.set_xlabel(r"$t$ (reduced)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"[ok] {stem}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--dt-group", type=float, default=1.0, help="time per group (stage_steps*dt*group_size)")
    p.add_argument("--barrier-h", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kB", type=float, default=1.0)
    p.add_argument("--lambda-target", type=float, default=0.1)
    p.add_argument("--lambda-atol", type=float, default=1e-3)
    p.add_argument("--n-grid", type=int, default=257)
    p.add_argument("--x1-lim", type=float, default=2.0)
    p.add_argument("--vmax", type=float, default=20.0, help="plot/RMSE clip; 0=off for RMSE")
    p.add_argument("--rmse-vmax", type=float, default=12.0)
    p.add_argument("--collage-vmax", type=float, default=20.0)
    p.add_argument("--collage-ncols", type=int, default=4)
    p.add_argument("--max-panels", type=int, default=40)
    p.add_argument("--last-frac", type=float, default=0.5)
    p.add_argument(
        "--only-last-frac",
        action="store_true",
        help="Skip group collage / RMSE(t); only last-frac 1D FES",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()

    fpt = Path(args.fpt_dir).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else fpt / "convergence"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    xi = np.linspace(-float(args.x1_lim), float(args.x1_lim), int(args.n_grid))
    f_ref = analytical_f_x1(xi, float(args.barrier_h))
    dG_ref = delta_g_two_wells(xi, f_ref, temperature=float(args.temperature), xi_cut=0.0)
    print(
        f"[ref] h={args.barrier_h:g}  β0={beta0:g}  ΔG_exact={dG_ref:.4g}  "
        f"device={device}  out={out}",
        flush=True,
    )

    rmse_vmax = float(args.rmse_vmax) if float(args.rmse_vmax) > 0 else None

    if not bool(args.only_last_frac):
        t, rmse, dG, F, titles = kde_groups_x1(
            fpt,
            xi=xi,
            f_ref=f_ref,
            group_size=int(args.group_size),
            dt_group=float(args.dt_group),
            temperature=float(args.temperature),
            beta0=beta0,
            vmax=rmse_vmax,
            device=device,
            chunk=int(args.chunk),
            max_groups=None,
            lambda_target=float(args.lambda_target),
            lambda_atol=float(args.lambda_atol),
        )
        _, _, _, F_pan, titles_pan, _ = subsample_panels(
            t, rmse, dG, F, titles, int(args.max_panels)
        )

        np.savez_compressed(
            out / "fes_rmse_vs_time.npz",
            xi=xi,
            F_ref=f_ref,
            t=t,
            rmse=rmse,
            dG=dG,
            F1d=F,
            titles=np.asarray(titles),
            dG_ref=np.float64(dG_ref),
            barrier_h=np.float64(args.barrier_h),
            group_size=np.int64(args.group_size),
            dt_group=np.float64(args.dt_group),
        )
        np.savez_compressed(
            out / "fes_1d_all_groups.npz",
            xi=xi,
            F_ref=f_ref,
            F1d=F_pan,
            t=t,
            titles=np.asarray(titles_pan),
        )

        plot_collage(
            out,
            xi=xi,
            f_ref=f_ref,
            f_sfpt=F_pan,
            titles=titles_pan,
            vmax=float(args.collage_vmax),
            ncols=int(args.collage_ncols),
            group_size=int(args.group_size),
        )
        plot_series(
            out,
            stem="fes_rmse_vs_time",
            t=t,
            y=rmse,
            ylabel=r"RMSE vs exact $F(x_1)$ / $k_BT$",
            y0_line=True,
            ylim=(0.0, None) if False else None,
        )
        plt.rcParams.update(nature_rcparams())
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(t, rmse, color=COLOR_SFPT, lw=1.5, marker="o", ms=3.0)
        ax.set_xlabel(r"$t$ (reduced)")
        ax.set_ylabel(r"RMSE vs exact $F(x_1)$ / $k_BT$")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(which="both", top=False, right=False)
        fig.tight_layout()
        fig.savefig(out / "fes_rmse_vs_time.png", dpi=200)
        fig.savefig(out / "fes_rmse_vs_time.pdf")
        plt.close(fig)

        plot_series(
            out,
            stem="deltaG_vs_time",
            t=t,
            y=dG,
            ylabel=r"$\Delta G=k_BT\ln(P_{x_1<0}/P_{x_1>0})$",
            y0=float(dG_ref),
        )
        plot_series(
            out,
            stem="fes_rmse_vs_time_divein",
            t=t,
            y=rmse,
            ylabel=r"RMSE vs exact $F(x_1)$ / $k_BT$",
            y0_line=True,
        )
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(t, rmse, color=COLOR_SFPT, lw=1.5, marker="o", ms=3.0)
        ax.set_xlabel(r"$t$ (reduced)")
        ax.set_ylabel(r"RMSE / $k_BT$")
        ax.set_xlim(0.0, min(20.0, float(np.max(t)) if len(t) else 20.0))
        ax.set_ylim(bottom=0.0)
        ax.tick_params(which="both", top=False, right=False)
        fig.tight_layout()
        fig.savefig(out / "fes_rmse_vs_time_divein.png", dpi=200)
        fig.savefig(out / "fes_rmse_vs_time_divein.pdf")
        plt.close(fig)

        plot_series(
            out,
            stem="deltaG_vs_time_divein",
            t=t,
            y=dG,
            ylabel=r"$\Delta G$ / $k_BT$",
            y0=float(dG_ref),
        )
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.axhline(dG_ref, color=COLOR_REF, ls=":", lw=1.2, label=rf"exact={dG_ref:g}")
        ax.axhspan(dG_ref - 1.0, dG_ref + 1.0, color=COLOR_REF, alpha=0.12, lw=0)
        ax.plot(t, dG, color=COLOR_SFPT, lw=1.5, marker="o", ms=3.0, label="sFPT")
        ax.set_xlabel(r"$t$ (reduced)")
        ax.set_ylabel(r"$\Delta G$ / $k_BT$")
        ax.set_xlim(0.0, min(20.0, float(np.max(t)) if len(t) else 20.0))
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(which="both", top=False, right=False)
        fig.tight_layout()
        fig.savefig(out / "deltaG_vs_time_divein.png", dpi=200)
        fig.savefig(out / "deltaG_vs_time_divein.pdf")
        plt.close(fig)
        print(
            f"[done-groups] groups={len(t)}  RMSE_end={rmse[-1]:.4f}  "
            f"ΔG_end={dG[-1]:.4f}",
            flush=True,
        )

    if float(args.last_frac) > 0:
        f_last, s0, s1, n_keep, lam = kde_last_frac_x1(
            fpt,
            xi=xi,
            last_frac=float(args.last_frac),
            beta0=beta0,
            device=device,
            chunk=int(args.chunk),
            lambda_target=float(args.lambda_target),
            lambda_atol=float(args.lambda_atol),
        )
        rmse_last = rmse_1d(
            f_last,
            f_ref,
            vmax=rmse_vmax,
            xi=xi,
            xi_lo=float(xi.min()),
            xi_hi=float(xi.max()),
        )
        dG_last = delta_g_two_wells(
            xi, f_last, temperature=float(args.temperature), xi_cut=0.0
        )
        np.savez_compressed(
            out / "fes_1d_last_frac.npz",
            xi=xi,
            F_sfpt=f_last,
            F_ref=f_ref,
            last_frac=np.float64(args.last_frac),
            stage_lo=np.int64(s0),
            stage_hi=np.int64(s1),
            n_stages=np.int64(n_keep),
            lambda_=np.float64(lam),
            rmse=np.float64(rmse_last),
            dG=np.float64(dG_last),
            dG_ref=np.float64(dG_ref),
        )
        plot_1d_last_frac(
            out,
            xi=xi,
            f_sfpt=f_last,
            f_ref=f_ref,
            vmax=float(args.collage_vmax),
            last_frac=float(args.last_frac),
            stage_lo=s0,
            stage_hi=s1,
            n_keep=n_keep,
            rmse=rmse_last,
            dG=dG_last,
            dG_ref=dG_ref,
        )
        print(
            f"[last-frac] stages={n_keep} ({s0}–{s1}) λ={lam:g}  "
            f"RMSE={rmse_last:.4f}  ΔG={dG_last:.4f}",
            flush=True,
        )

    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
