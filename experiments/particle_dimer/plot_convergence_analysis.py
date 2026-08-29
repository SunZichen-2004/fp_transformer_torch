#!/usr/bin/env python3
"""Particle-dimer sFPT convergence: 1D KDE F(d) with F(1+λ)/λ reconstruction.

``F = F_KDE · (1+λ) / λ`` with ``F_KDE = −log p / β0``.

Reference is the WT-MetaD 1D FES (mean over replica counts, latest dump).
RMSE / collage overlays use that MetaD curve. ``fes_1d_last_frac`` pools the
last 50% of sFPT stages and overlays MetaD. Also reports ΔG and T_kin.
"""

from __future__ import annotations

import argparse
import csv
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

from fp_transformer_torch.kernel.gaussian import (  # noqa: E402
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)

STAGE_RE = re.compile(r"stage(\d+)_lambda([0-9p]+)_d\.npy$", re.I)
META_DIR_RE = re.compile(r"particle_dimer_meta_r(\d+)_T(.+)_s\d+_")
COLOR_SFPT = "#e67e22"
COLOR_REF = "#2166ac"
COLOR_META = "#2166ac"
COLOR_META_R = {
    "64": "#2166ac",
    "128": "#67a9cf",
}


def _temp_tag(temperature: float) -> str:
    t = float(temperature)
    if abs(t - round(t)) < 1e-12:
        return str(int(round(t)))
    return f"{t:g}"


def find_meta_dirs(meta_root: Path, temperature: float) -> list[Path]:
    """Match ``..._T{tag}_s0_`` so T=1 does not hit T=1.5."""
    root = Path(meta_root)
    if not root.is_dir():
        return []
    needle = f"_T{_temp_tag(temperature)}_s0_"
    out = [
        p
        for p in sorted(root.glob("particle_dimer_meta_r*_T*_s0_*"))
        if p.is_dir() and needle in p.name
    ]
    return out


def latest_meta_fes_path(meta_dir: Path) -> Path:
    final = Path(meta_dir) / "fes_final.npz"
    if final.is_file():
        return final
    cands = list(Path(meta_dir).glob("step*_fes.npz"))
    if not cands:
        raise FileNotFoundError(f"no fes npz in {meta_dir}")

    def _step(p: Path) -> int:
        m = re.search(r"step(\d+)_fes\.npz$", p.name)
        return int(m.group(1)) if m else -1

    return max(cands, key=_step)


def load_meta_fes_mean(
    meta_dirs: list[Path], xi: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Interpolate each latest MetaD F(d) onto ``xi``, min-shift, then mean."""
    curves: list[np.ndarray] = []
    labels: list[str] = []
    for d in meta_dirs:
        fes_p = latest_meta_fes_path(d)
        z = np.load(fes_p)
        d_grid = np.asarray(z["d"], dtype=np.float64).reshape(-1)
        f_raw = np.asarray(z["F"], dtype=np.float64).reshape(-1)
        fi = np.interp(xi, d_grid, f_raw, left=np.nan, right=np.nan)
        curves.append(relative_fes(fi, None))
        m = META_DIR_RE.search(d.name)
        r = m.group(1) if m else "?"
        step = fes_p.stem.replace("_fes", "")
        labels.append(f"R={r} {step}")
        print(f"[meta] {d.name}  {fes_p.name}", flush=True)
    stacked = np.stack(curves, axis=0)
    mean = np.nanmean(stacked, axis=0)
    return relative_fes(mean, None), labels


def load_meta_deltag_vs_time(
    meta_dirs: list[Path],
    *,
    temperature: float,
    dt: float,
    xi_cut: float = 1.5,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Per-run WT-MetaD ΔG(t). ``t = step · dt``. One series per replica count."""
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _r_num(p: Path) -> int:
        rm = META_DIR_RE.search(p.name)
        return int(rm.group(1)) if rm else 0

    for d in sorted(meta_dirs, key=_r_num):
        steps: list[int] = []
        dgs: list[float] = []
        fes_files = list(Path(d).glob("step*_fes.npz"))
        fes_files.sort(
            key=lambda p: int(re.search(r"step(\d+)_fes\.npz$", p.name).group(1))
        )
        for fes_p in fes_files:
            m = re.search(r"step(\d+)_fes\.npz$", fes_p.name)
            if not m:
                continue
            z = np.load(fes_p)
            dg = delta_g_wells(
                np.asarray(z["d"]),
                np.asarray(z["F"]),
                temperature=float(temperature),
                xi_cut=float(xi_cut),
            )
            steps.append(int(m.group(1)))
            dgs.append(float(dg))
        if not steps:
            continue
        rm = META_DIR_RE.search(d.name)
        r = rm.group(1) if rm else d.name
        label = rf"WT-MetaD $R={r}$"
        t = np.asarray(steps, dtype=np.float64) * float(dt)
        y = np.asarray(dgs, dtype=np.float64)
        out.append((label, t, y))
        print(
            f"[meta-dG] {d.name}  R={r}  n={len(steps)}  "
            f"t={t[0]:g}→{t[-1]:g}  ΔG_end={y[-1]:.4f}",
            flush=True,
        )
    return out


def nature_rcparams() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
    }


def relative_fes(f: np.ndarray, vmax: float | None = None) -> np.ndarray:
    out = np.asarray(f, dtype=np.float64) - np.nanmin(f)
    if vmax is not None and float(vmax) > 0:
        out = np.clip(out, 0.0, float(vmax))
    return out


def parse_lambda_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def reconstruct_fes(p: np.ndarray, *, beta0: float, lam: float) -> np.ndarray:
    """``F = (−log p / β0) · (1+λ) / λ``, shifted to min=0."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, None)
    f_kde = -np.log(p) / float(beta0)
    f = f_kde * (1.0 + float(lam)) / float(lam)
    return relative_fes(f, None)


def discover_d_stages(
    data_dir: Path,
    *,
    lambda_target: float = 0.05,
    lambda_atol: float = 5e-3,
) -> list[tuple[int, Path, Path, float]]:
    rows: list[tuple[int, Path, Path, float]] = []
    for d_p in sorted(data_dir.glob("stage[0-9]*_lambda*_d.npy")):
        m = STAGE_RE.search(d_p.name)
        if not m:
            continue
        idx = int(m.group(1))
        lam = parse_lambda_tag(m.group(2))
        if abs(lam - float(lambda_target)) > float(lambda_atol):
            continue
        bw_p = d_p.with_name(d_p.name.replace("_d.npy", "_bandwidth.npy"))
        if not bw_p.is_file():
            continue
        rows.append((idx, d_p, bw_p, lam))
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
    sx = np.asarray(sample_x, dtype=np.float64).reshape(-1)
    bw = np.asarray(bandwidth, dtype=np.float64).reshape(-1)
    if sx.shape[0] != bw.shape[0]:
        raise ValueError(f"d len {sx.shape[0]} != bw len {bw.shape[0]}")
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


def delta_g_wells(
    xi: np.ndarray,
    f: np.ndarray,
    *,
    temperature: float,
    xi_cut: float = 1.5,
) -> float:
    """ΔG = kT ln(P_{d<cut} / P_{d>cut}); compact vs extended."""
    xi_a = np.asarray(xi, dtype=np.float64)
    f_a = np.asarray(f, dtype=np.float64)
    m = np.isfinite(f_a)
    kt = float(temperature)
    beta = 1.0 / kt
    left = m & (xi_a < float(xi_cut))
    right = m & (xi_a > float(xi_cut))
    if not (np.any(left) and np.any(right)):
        return float("nan")

    def _logZ(sel: np.ndarray) -> float:
        v = -beta * f_a[sel]
        mval = float(np.max(v))
        return mval + float(np.log(np.sum(np.exp(v - mval))))

    return float(kt * (_logZ(left) - _logZ(right)))


def rmse_1d(f_a: np.ndarray, f_b: np.ndarray, *, vmax: float | None) -> float:
    a = np.asarray(f_a, dtype=np.float64)
    b = np.asarray(f_b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if vmax is not None and float(vmax) > 0:
        m = m & (a <= float(vmax)) & (b <= float(vmax))
    if not np.any(m):
        return float("nan")
    a0 = a[m] - np.min(a[m])
    b0 = b[m] - np.min(b[m])
    return float(np.sqrt(np.mean((a0 - b0) ** 2)))


def kde_groups(
    fpt_dir: Path,
    *,
    xi: np.ndarray,
    group_size: int,
    dt_group: float,
    temperature: float,
    beta0: float,
    device: torch.device,
    chunk: int,
    lambda_target: float,
    lambda_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    rows = discover_d_stages(
        fpt_dir, lambda_target=lambda_target, lambda_atol=lambda_atol
    )
    n_groups = len(rows) // int(group_size)
    if n_groups < 1:
        # fall back to one-stage groups if the run is short
        group_size = 1
        n_groups = len(rows)
    if n_groups < 1:
        raise RuntimeError(f"no λ≈{lambda_target:g} stages in {fpt_dir}")
    print(
        f"[scan] stages={len(rows)}  groups={n_groups}  group_size={group_size}",
        flush=True,
    )

    t_list, g_list, f_list, titles = [], [], [], []
    for ig in range(n_groups):
        chunk_rows = rows[ig * group_size : (ig + 1) * group_size]
        t = float(ig + 1) * float(dt_group)
        lam = float(chunk_rows[0][3])
        p_acc = np.zeros(xi.shape[0], dtype=np.float64)
        for _, d_p, bw_p, _ in chunk_rows:
            p_acc += density_on_grid_1d(
                xi, np.load(d_p), np.load(bw_p), chunk=chunk, device=device
            )
        p = p_acc / float(len(chunk_rows))
        f1d = reconstruct_fes(p, beta0=beta0, lam=lam)
        dG = delta_g_wells(xi, f1d, temperature=temperature, xi_cut=1.5)
        s0, s1 = chunk_rows[0][0], chunk_rows[-1][0]
        title = f"t={t:g}  stage{s0:04d}–{s1:04d}"
        t_list.append(t)
        g_list.append(dG)
        f_list.append(f1d)
        titles.append(title)
        print(f"[kde] {title}  ΔG={dG:.4f}", flush=True)
    F = np.stack(f_list, axis=0)
    return np.asarray(t_list), np.asarray(g_list), F, titles, np.asarray([group_size])


def kde_last_frac(
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
    rows = discover_d_stages(
        fpt_dir, lambda_target=lambda_target, lambda_atol=lambda_atol
    )
    if not rows:
        raise RuntimeError(f"no λ≈{lambda_target:g} stages in {fpt_dir}")
    n_keep = max(1, int(np.ceil(float(last_frac) * len(rows))))
    use = rows[-n_keep:]
    lam = float(use[0][3])
    p_acc = np.zeros(xi.shape[0], dtype=np.float64)
    for i, (_, d_p, bw_p, _) in enumerate(use):
        p_acc += density_on_grid_1d(
            xi, np.load(d_p), np.load(bw_p), chunk=chunk, device=device
        )
        if (i + 1) % 10 == 0 or i + 1 == len(use):
            print(f"[last-frac] kde {i+1}/{len(use)}", flush=True)
    p = p_acc / float(len(use))
    f1d = reconstruct_fes(p, beta0=beta0, lam=lam)
    return f1d, int(use[0][0]), int(use[-1][0]), int(n_keep), lam


def plot_collage(
    out: Path,
    *,
    xi: np.ndarray,
    f_sfpt: np.ndarray,
    f_ref: np.ndarray | None,
    titles: list[str],
    vmax: float,
    ncols: int,
) -> None:
    n = len(titles)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    plt.rcParams.update(nature_rcparams())
    fig = plt.figure(figsize=(2.5 * ncols, 2.05 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, wspace=0.32, hspace=0.48,
        left=0.07, right=0.99, bottom=0.08, top=0.90,
    )
    y_ref = None if f_ref is None else relative_fes(f_ref, None)
    if y_ref is not None and vmax > 0:
        y_ref = np.minimum(y_ref, float(vmax))
    for ig in range(n):
        r, c = divmod(ig, ncols)
        ax = fig.add_subplot(gs[r, c])
        ys = relative_fes(f_sfpt[ig], None)
        if vmax > 0:
            ys = np.minimum(ys, float(vmax))
        if y_ref is not None:
            ax.plot(xi, y_ref, color=COLOR_META, lw=1.1, ls="--", label="WT-MetaD", zorder=2)
        ax.plot(xi, ys, color=COLOR_SFPT, lw=1.3, label="sFPT", zorder=3)
        ax.axvline(1.5, color="0.5", ls=":", lw=0.7, zorder=0)
        ax.set_xlim(float(xi.min()), float(xi.max()))
        if vmax > 0:
            ax.set_ylim(0.0, float(vmax) * 1.02)
        ax.set_title(titles[ig], fontsize=8, pad=3)
        ax.tick_params(labelsize=7, which="both", top=False, right=False)
        if r == nrows - 1 or ig + ncols >= n:
            ax.set_xlabel(r"$d$", fontsize=8)
        if c == 0:
            ax.set_ylabel(r"$F(d)$ ($k_BT$)", fontsize=8)
        if ig == 0:
            ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    fig.suptitle(
        r"1D $F(d)$  sFPT-KDE  $F_{\mathrm{KDE}}(1+\lambda)/\lambda$",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(out / "fes_1d_all_groups.png", dpi=200)
    fig.savefig(out / "fes_1d_all_groups.pdf")
    plt.close(fig)
    print("[ok] fes_1d_all_groups", flush=True)


def plot_last_frac(
    out: Path,
    *,
    xi: np.ndarray,
    f_sfpt: np.ndarray,
    f_meta: np.ndarray | None,
    vmax: float,
    last_frac: float,
    stage_lo: int,
    stage_hi: int,
    n_keep: int,
    dG: float,
    lam: float,
    meta_label: str | None = None,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = relative_fes(f_sfpt, None)
    if vmax > 0:
        y = np.minimum(y, float(vmax))
    if f_meta is not None:
        ym = relative_fes(f_meta, None)
        if vmax > 0:
            ym = np.minimum(ym, float(vmax))
        ax.plot(
            xi,
            ym,
            color=COLOR_META,
            lw=1.6,
            ls="--",
            label=meta_label or "WT-MetaD",
            zorder=2,
        )
        ax.fill_between(xi, 0.0, ym, color=COLOR_META, alpha=0.10, lw=0, zorder=1)
    ax.plot(xi, y, color=COLOR_SFPT, lw=1.8, label="sFPT last 50%", zorder=3)
    ax.axvline(0.97, color="0.55", ls=":", lw=0.7)
    ax.axvline(2.03, color="0.55", ls=":", lw=0.7)
    ax.axvline(1.5, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"dimer distance $d$")
    ax.set_ylabel(r"$F(d)$ ($k_BT$)")
    ax.set_xlim(float(xi.min()), float(xi.max()))
    if vmax > 0:
        ax.set_ylim(0.0, float(vmax) * 1.02)
    ax.set_title(
        rf"last {float(last_frac):.0%}  stage {stage_lo}–{stage_hi} (n={n_keep})"
        "\n"
        rf"$\lambda$={lam:g}  $F(1+\lambda)/\lambda$  $\Delta G$={dG:.3g}",
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
    t_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    ref_label: str = "WT-MetaD",
    ref_curves: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    markers = ["s", "^", "D", "v"]
    if ref_curves:
        for i, (lab, tr, yr) in enumerate(ref_curves):
            rkey = "128" if "128" in lab else ("64" if "64" in lab else "")
            color = COLOR_META_R.get(rkey, COLOR_META)
            ax.plot(
                tr,
                yr,
                color=color,
                lw=1.45,
                marker=markers[i % len(markers)],
                ms=2.8,
                markevery=max(1, int(np.asarray(tr).size) // 24),
                label=lab,
                zorder=2,
            )
    elif (
        t_ref is not None
        and y_ref is not None
        and int(np.asarray(t_ref).size) > 0
        and int(np.asarray(y_ref).size) > 0
    ):
        ax.plot(
            t_ref,
            y_ref,
            color=COLOR_META,
            lw=1.5,
            marker="s",
            ms=2.8,
            label=ref_label,
            zorder=2,
        )
    elif y0 is not None and np.isfinite(y0):
        ax.axhline(y0, color=COLOR_META, ls=":", lw=1.2, label=rf"WT-MetaD={y0:.3g}")
    ax.plot(t, y, color=COLOR_SFPT, lw=1.6, marker="o", ms=3.5, label="sFPT", zorder=3)
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


def plot_temperature(out: Path, metrics_csv: Path, t_target: float) -> None:
    if not metrics_csv.is_file():
        return
    rows = list(csv.DictReader(metrics_csv.open()))
    if not rows:
        return
    prod = [r for r in rows if r.get("phase") == "anneal"]
    use = prod or rows
    t_kin = np.array([float(r["T_kin_mean"]) for r in use], dtype=np.float64)
    t_std = np.array([float(r["T_kin_std"]) for r in use], dtype=np.float64)
    x = np.arange(1, len(use) + 1, dtype=np.float64)
    plt.rcParams.update(nature_rcparams())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(float(t_target), color=COLOR_REF, ls=":", lw=1.2, label=rf"$T={t_target:g}$")
    ax.fill_between(x, t_kin - t_std, t_kin + t_std, color=COLOR_SFPT, alpha=0.18, lw=0)
    ax.plot(x, t_kin, color=COLOR_SFPT, lw=1.5, marker="o", ms=3.0, label=r"$T_{\mathrm{kin}}$")
    ax.set_xlabel("stage")
    ax.set_ylabel(r"$T_{\mathrm{kin}}$")
    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(out / "T_kin_vs_stage.png", dpi=200)
    fig.savefig(out / "T_kin_vs_stage.pdf")
    plt.close(fig)
    print(
        f"[ok] T_kin_vs_stage  mean={t_kin.mean():.3g}  last={t_kin[-1]:.3g}  "
        f"target={t_target:g}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fpt-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--dt-group", type=float, default=None)
    p.add_argument("--stage-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kB", type=float, default=1.0)
    p.add_argument("--lambda-target", type=float, default=0.1)
    p.add_argument("--lambda-atol", type=float, default=5e-3)
    p.add_argument("--n-grid", type=int, default=257)
    p.add_argument("--d-lo", type=float, default=0.6)
    p.add_argument("--d-hi", type=float, default=2.8)
    p.add_argument("--vmax", type=float, default=20.0)
    p.add_argument("--collage-ncols", type=int, default=4)
    p.add_argument("--last-frac", type=float, default=0.5)
    p.add_argument(
        "--meta-root",
        type=Path,
        default=None,
        help="Directory of particle_dimer_meta results (auto-match by T).",
    )
    p.add_argument(
        "--meta-dirs",
        type=str,
        default=None,
        help="Comma-separated MetaD result dirs; overrides --meta-root.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()

    fpt = Path(args.fpt_dir).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else fpt / "convergence_analysis"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    xi = np.linspace(float(args.d_lo), float(args.d_hi), int(args.n_grid))
    dt_group = args.dt_group
    if dt_group is None:
        dt_group = float(args.stage_steps) * float(args.dt) * float(args.group_size)
    print(
        f"[cfg] β0={beta0:g}  λ={args.lambda_target:g}  F ← F_KDE(1+λ)/λ  "
        f"group_size={args.group_size}  last_frac={args.last_frac:g}  "
        f"device={device}  out={out}",
        flush=True,
    )

    meta_dirs: list[Path] = []
    if args.meta_dirs:
        meta_dirs = [
            Path(x.strip()).resolve()
            for x in str(args.meta_dirs).replace(";", ",").split(",")
            if x.strip()
        ]
    else:
        meta_root = (
            Path(args.meta_root).resolve()
            if args.meta_root
            else ROOT / "experiments" / "particle_dimer_meta" / "results"
        )
        meta_dirs = find_meta_dirs(meta_root, float(args.temperature))
    f_meta: np.ndarray | None = None
    meta_labels: list[str] = []
    dG_meta = float("nan")
    if meta_dirs:
        f_meta, meta_labels = load_meta_fes_mean(meta_dirs, xi)
        dG_meta = delta_g_wells(
            xi, f_meta, temperature=float(args.temperature), xi_cut=1.5
        )
        print(
            f"[meta] n={len(meta_dirs)}  mean of {meta_labels}  ΔG={dG_meta:.4f}",
            flush=True,
        )
    else:
        print("[meta] no MetaD dirs found; RMSE falls back to last-frac", flush=True)

    f_last, s0, s1, n_keep, lam = kde_last_frac(
        fpt,
        xi=xi,
        last_frac=float(args.last_frac),
        beta0=beta0,
        device=device,
        chunk=int(args.chunk),
        lambda_target=float(args.lambda_target),
        lambda_atol=float(args.lambda_atol),
    )
    dG_last = delta_g_wells(
        xi, f_last, temperature=float(args.temperature), xi_cut=1.5
    )
    np.savez_compressed(
        out / "fes_1d_last_frac.npz",
        xi=xi,
        F_sfpt=f_last,
        F_meta=np.asarray(f_meta) if f_meta is not None else np.array([]),
        last_frac=np.float64(args.last_frac),
        stage_lo=np.int64(s0),
        stage_hi=np.int64(s1),
        n_stages=np.int64(n_keep),
        lambda_=np.float64(lam),
        dG=np.float64(dG_last),
        dG_meta=np.float64(dG_meta),
        meta_labels=np.asarray(meta_labels),
        formula=np.asarray("F_KDE*(1+lambda)/lambda"),
    )
    plot_last_frac(
        out,
        xi=xi,
        f_sfpt=f_last,
        f_meta=f_meta,
        vmax=float(args.vmax),
        last_frac=float(args.last_frac),
        stage_lo=s0,
        stage_hi=s1,
        n_keep=n_keep,
        dG=dG_last,
        lam=lam,
        meta_label="WT-MetaD mean" if meta_labels else None,
    )

    t, dG, F, titles, gs = kde_groups(
        fpt,
        xi=xi,
        group_size=int(args.group_size),
        dt_group=float(dt_group),
        temperature=float(args.temperature),
        beta0=beta0,
        device=device,
        chunk=int(args.chunk),
        lambda_target=float(args.lambda_target),
        lambda_atol=float(args.lambda_atol),
    )
    f_ref = f_meta if f_meta is not None else f_last
    dG_ref = dG_meta if f_meta is not None else dG_last
    rmse = np.array([rmse_1d(Fi, f_ref, vmax=float(args.vmax)) for Fi in F])
    np.savez_compressed(
        out / "fes_rmse_vs_time.npz",
        xi=xi,
        F_ref=f_ref,
        F_sfpt_last_frac=f_last,
        F_meta=np.asarray(f_meta) if f_meta is not None else np.array([]),
        t=t,
        rmse=rmse,
        dG=dG,
        F1d=F,
        titles=np.asarray(titles),
        dG_ref=np.float64(dG_ref),
        dG_last_frac=np.float64(dG_last),
        dG_meta=np.float64(dG_meta),
        meta_labels=np.asarray(meta_labels),
        group_size=np.int64(int(gs[0])),
        dt_group=np.float64(dt_group),
        formula=np.asarray("F_KDE*(1+lambda)/lambda"),
    )
    plot_collage(
        out,
        xi=xi,
        f_sfpt=F,
        f_ref=f_ref,
        titles=titles,
        vmax=float(args.vmax),
        ncols=int(args.collage_ncols),
    )
    plot_series(
        out,
        stem="fes_rmse_vs_time",
        t=t,
        y=rmse,
        ylabel=r"RMSE vs WT-MetaD $F(d)$ ($k_BT$)",
        ylim=(0.0, None),
    )
    meta_dg_curves = (
        load_meta_deltag_vs_time(
            meta_dirs, temperature=float(args.temperature), dt=float(args.dt)
        )
        if meta_dirs
        else []
    )
    np.savez_compressed(
        out / "deltaG_meta_vs_time.npz",
        labels=np.asarray([c[0] for c in meta_dg_curves]),
        **{
            f"t_{i}": c[1]
            for i, c in enumerate(meta_dg_curves)
        },
        **{
            f"dG_{i}": c[2]
            for i, c in enumerate(meta_dg_curves)
        },
    )
    plot_series(
        out,
        stem="deltaG_vs_time",
        t=t,
        y=dG,
        ylabel=r"$\Delta G=k_BT\ln(P_{d<1.5}/P_{d>1.5})$",
        y0=float(dG_ref) if np.isfinite(dG_ref) else None,
        ref_curves=meta_dg_curves,
    )
    plot_temperature(out, fpt / "stage_metrics.csv", float(args.temperature))
    print(
        f"[done] groups={len(t)}  RMSE_end={rmse[-1]:.4f}  ΔG_end={dG[-1]:.4f}  "
        f"ΔG_last-frac={dG_last:.4f}  ΔG_meta={dG_meta:.4f}  {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
