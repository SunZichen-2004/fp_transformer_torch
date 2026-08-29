#!/usr/bin/env python3
"""KDE FES from FPT stage CV + per-replica bandwidth (same Gaussian kernel as source).

Loads ``stage*_cv.npy`` / ``stage*_bandwidth.npy``, evaluates normalized Gaussian
kernel density on a 2D (d1, d2) grid with the same convention as
``GaussianKernel.kernel_matrix`` + ``pairwise_distance``:

    D_ji = ||x_j - y_i||^2 / (A D)
    K_ji = (2 π σ_i²)^{-AD/2} exp( -0.5 D_ji / σ_i² )

Then

    P(x) = (1/N) Σ_i K(x, x_i; σ_i)
    F(x) = - ln P(x) / (β₀ λ) = - (k_B T / λ) ln P(x)     [eV]

(with ``β₀ = 1/(k_B T)`` as in ``sample.py``). Multiple stages: average ``P``
on the same grid, then take ``ln`` once. λ is parsed from filenames
(e.g. ``lambda0p030000`` → 0.03) unless ``--lambda`` is set.
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

from fp_transformer_torch.kernel.gaussian import (  # noqa: E402
    bandwidth_column_squared,
    gaussian_kernel_log_normalization,
)

KB_EV_PER_K = 8.617330337217213e-5
EV_TO_KCAL = 23.06054783061903
STAGE_RE = re.compile(r"stage(\d+)", re.I)
LAMBDA_RE = re.compile(r"lambda(\d+)p(\d+)", re.I)


def parse_lambda(path: Path) -> float | None:
    m = LAMBDA_RE.search(path.name)
    if m is None:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def parse_stage(path: Path) -> int | None:
    m = STAGE_RE.search(path.name)
    return int(m.group(1)) if m else None


def to_kernel_cv(cv: np.ndarray | torch.Tensor) -> torch.Tensor:
    """``(N, 2)`` → ``(N, 2, 1)`` matching ``MultiParticlePairDistance``."""
    t = torch.as_tensor(cv, dtype=torch.float64)
    if t.ndim == 2 and t.shape[-1] == 2:
        return t.reshape(t.shape[0], 2, 1)
    if t.ndim == 3 and t.shape[-2:] == (2, 1):
        return t.to(dtype=torch.float64)
    if t.ndim == 3 and t.shape[-2:] == (1, 2):
        return t.reshape(t.shape[0], 2, 1)
    raise ValueError(f"expected CV (N,2) or (N,2,1), got {tuple(t.shape)}")


def cross_log_kernel(
    grid_cv: torch.Tensor,
    sample_cv: torch.Tensor,
    bandwidth: torch.Tensor,
) -> torch.Tensor:
    """Log normalized Gaussian ``log K_ji`` for grid ``j`` vs sample ``i``.

    Same algebra as ``GaussianKernel.kernel_matrix``, but grid ≠ sample set.
    Distance uses the pairwise convention ``||·||^2 / (A*D)``.
    """
    if grid_cv.ndim != 3 or sample_cv.ndim != 3:
        raise ValueError("grid/sample must be (batch, atoms, dim)")
    m = int(grid_cv.shape[0])
    n = int(sample_cv.shape[0])
    feature_dim = float(sample_cv.shape[-2] * sample_cv.shape[-1])
    g = grid_cv.reshape(m, -1)
    s = sample_cv.reshape(n, -1)
    g2 = torch.sum(g * g, dim=1, keepdim=True)
    s2 = torch.sum(s * s, dim=1)
    dist = g2 + s2.unsqueeze(0) - 2.0 * (g @ s.T)
    dist = torch.clamp(dist, min=0.0) / feature_dim
    bw = bandwidth.reshape(-1).to(dtype=grid_cv.dtype, device=grid_cv.device)
    if bw.numel() != n:
        raise ValueError(f"bandwidth length {bw.numel()} != n_samples {n}")
    log_k = (-0.5 * dist) / bandwidth_column_squared(bw) + gaussian_kernel_log_normalization(
        bw, feature_dim
    )
    return log_k


def density_on_grid(
    grid_xy: np.ndarray,
    sample_cv: np.ndarray,
    bandwidth: np.ndarray,
    *,
    chunk: int = 4096,
    device: torch.device,
) -> np.ndarray:
    """``P`` on flat grid points ``(M, 2)``; returns shape ``(M,)``."""
    samples = to_kernel_cv(sample_cv).to(device=device)
    bw = torch.as_tensor(bandwidth, dtype=torch.float64, device=device).reshape(-1)
    n = float(samples.shape[0])
    m = int(grid_xy.shape[0])
    out = np.empty((m,), dtype=np.float64)
    for start in range(0, m, int(chunk)):
        stop = min(start + int(chunk), m)
        grid = to_kernel_cv(grid_xy[start:stop]).to(device=device)
        log_k = cross_log_kernel(grid, samples, bw)
        log_p = torch.logsumexp(log_k, dim=1) - np.log(n)
        out[start:stop] = torch.exp(log_p).detach().cpu().numpy()
    return out


def discover_stages(
    data_dir: Path,
    *,
    last_n: int | None,
    stage_list: list[int] | None,
) -> list[tuple[Path, Path, float]]:
    cvs = sorted(data_dir.glob("stage[0-9]*_lambda*_cv.npy"))
    rows: list[tuple[Path, Path, float, int]] = []
    for cv_path in cvs:
        idx = parse_stage(cv_path)
        if idx is None:
            continue
        lam = parse_lambda(cv_path)
        if lam is None:
            continue
        bw_path = cv_path.with_name(cv_path.name.replace("_cv.npy", "_bandwidth.npy"))
        if not bw_path.is_file():
            print(f"[skip] missing bandwidth for {cv_path.name}", flush=True)
            continue
        rows.append((cv_path, bw_path, float(lam), idx))
    rows.sort(key=lambda r: r[3])
    if stage_list is not None:
        want = set(stage_list)
        rows = [r for r in rows if r[3] in want]
    elif last_n is not None and last_n > 0:
        rows = rows[-int(last_n) :]
    if not rows:
        raise FileNotFoundError(f"no stage cv+bandwidth pairs under {data_dir}")
    return [(a, b, lam) for a, b, lam, _ in rows]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/storage/gaoyiqinLab/sunzicheng/work/torch_fpt/fp_transformer_torch-master"
            "/experiments/rearrangement_water_fpt/results/claisen_water_fpt_mh-0_g8_86736"
        ),
    )
    p.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma stages, e.g. 14,15,16. Default: --last-n.",
    )
    p.add_argument("--last-n", type=int, default=1, help="Use last N stages (default 1).")
    p.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=None,
        help="Override λ (default: parse from filenames; all stages must match).",
    )
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument(
        "--bounds",
        type=str,
        default="1.0,6.5,1.0,6.5",
        help="d1_lo,d1_hi,d2_lo,d2_hi",
    )
    p.add_argument("--grid-n", type=int, default=200)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--vmax-kcal", type=float, default=60.0)
    p.add_argument(
        "--p-floor",
        type=float,
        default=1e-6,
        help="Mask cells with P < p_floor * max(P) in the plot (NaN).",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output png (default: DATA/kde_fes/kde_fes_2d.png)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    stage_list = None
    if args.stages:
        stage_list = [int(x) for x in args.stages.replace(";", ",").split(",") if x.strip()]
    pairs = discover_stages(data_dir, last_n=args.last_n, stage_list=stage_list)

    lams = [lam for _, _, lam in pairs]
    if args.lambda_ is not None:
        lam = float(args.lambda_)
    else:
        if max(lams) - min(lams) > 1e-12:
            raise ValueError(
                f"stages have different λ={lams}; pass a single --lambda or select same-λ stages"
            )
        lam = float(lams[0])
    if lam <= 0:
        raise ValueError(f"λ must be > 0, got {lam}")

    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    vals = [float(x) for x in args.bounds.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("--bounds must be d1_lo,d1_hi,d2_lo,d2_hi")
    d1_lo, d1_hi, d2_lo, d2_hi = vals
    n = int(args.grid_n)
    d1 = np.linspace(d1_lo, d1_hi, n)
    d2 = np.linspace(d2_lo, d2_hi, n)
    g1, g2 = np.meshgrid(d1, d2, indexing="xy")
    grid_xy = np.stack([g1.ravel(), g2.ravel()], axis=-1)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda unavailable → cpu", flush=True)

    p_acc = np.zeros((grid_xy.shape[0],), dtype=np.float64)
    for cv_path, bw_path, file_lam in pairs:
        cv = np.load(cv_path)
        bw = np.load(bw_path)
        print(
            f"[load] {cv_path.name}  λ_file={file_lam:g}  N={cv.shape[0]}  "
            f"bw med={np.median(bw):.4g}",
            flush=True,
        )
        p_acc += density_on_grid(
            grid_xy, cv, bw, chunk=args.chunk, device=device
        )
    p_avg = p_acc / float(len(pairs))
    p_avg = np.clip(p_avg, 1e-300, None)

    # F [eV] = -ln P / (β₀ λ) = -(kT/λ) ln P ; plot in kcal/mol, min = 0
    f_ev = -np.log(p_avg) / (beta0 * lam)
    f_kcal = (f_ev - np.nanmin(f_ev)) * EV_TO_KCAL
    f2 = f_kcal.reshape(g1.shape)
    p2 = p_avg.reshape(g1.shape)
    mask = p2 < float(args.p_floor) * float(np.max(p2))
    f_plot = np.ma.array(np.minimum(f2, float(args.vmax_kcal)), mask=mask)

    out = args.output
    if out is None:
        out = data_dir / "kde_fes" / "kde_fes_2d.png"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"),
        d1=d1,
        d2=d2,
        P=p2,
        F_eV=f_ev.reshape(g1.shape),
        F_kcal=f2,
        lambda_=lam,
        beta0=beta0,
        temperature=float(args.temperature),
        formula="F = -ln(P)/(beta0*lambda)  [eV]",
        stages=[p.name for p, _, _ in pairs],
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    pcm = ax.pcolormesh(
        g1,
        g2,
        f_plot,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=float(args.vmax_kcal),
    )
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$F$ / kcal mol$^{-1}$")
    ax.set_xlabel(r"$d_1$ / Å")
    ax.set_ylabel(r"$d_2$ / Å")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        rf"KDE FES  $\lambda={lam:g}$  $N_\mathrm{{stage}}={len(pairs)}$  "
        rf"$T={args.temperature:g}\,\mathrm{{K}}$"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(
        f"[ok] wrote {out}  and {out.with_suffix('.npz')}  "
        f"F_kcal max={float(np.nanmax(f_kcal)):.3f}  (clipped plot vmax={args.vmax_kcal})",
        flush=True,
    )


if __name__ == "__main__":
    main()
