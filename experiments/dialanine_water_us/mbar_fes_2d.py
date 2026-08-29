#!/usr/bin/env python3
"""2D MBAR/WHAM for dialanine_water φ/ψ umbrella sampling.

Reads a US result from ``sample.py``::

    trajectory_cv.npy          (n_frames, n_windows, 2)
    umbrella_centers.npy       (n_windows, 2)
    umbrella_kappa.npy         (n_windows, 2)   eV / rad²
    run_info.txt

Bias uses torus minimum-image::

    U_k(s) = ½ Σ_α κ_{kα} wrap(s_α − s0_{kα})²

Writes ``mbar/fes_mbar.npz`` with 2D F(φ,ψ) and projected 1D F(φ), F(ψ).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.06054783061903
KB_EV_PER_K = 8.617330337217213e-5
TWO_PI = 2.0 * np.pi


def _parse_run_info(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not path.is_file():
        return info
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


def wrap(d: np.ndarray, period: float = TWO_PI) -> np.ndarray:
    return d - period * np.round(d / period)


def relative_fes(F: np.ndarray) -> np.ndarray:
    out = np.asarray(F, dtype=np.float64).copy()
    ok = np.isfinite(out)
    if np.any(ok):
        out = out - np.nanmin(out[ok])
    return out


def project_1d(phi, psi, F2, *, axis: str, n_bins: int, temperature: float):
    """Boltzmann-project 2D FES onto φ or ψ."""
    kT = KB_EV_PER_K * float(temperature) * EV_TO_KCAL
    beta = 1.0 / kT
    F2 = np.asarray(F2, dtype=np.float64)
    # F2 shape (n_psi, n_phi) with indexing xy
    if axis == "phi":
        # integrate over ψ
        w = np.exp(-beta * np.where(np.isfinite(F2), F2, 1e6))
        p = np.nansum(w, axis=0)
        edges = np.linspace(float(phi.min()), float(phi.max()), int(n_bins) + 1)
        # resample onto n_bins if needed
        centers = 0.5 * (edges[:-1] + edges[1:])
        if len(phi) == int(n_bins) and np.allclose(phi, centers):
            p1 = p
            xi = phi
        else:
            p1 = np.interp(centers, phi, p, left=0.0, right=0.0)
            xi = centers
    else:
        w = np.exp(-beta * np.where(np.isfinite(F2), F2, 1e6))
        p = np.nansum(w, axis=1)
        edges = np.linspace(float(psi.min()), float(psi.max()), int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if len(psi) == int(n_bins) and np.allclose(psi, centers):
            p1 = p
            xi = psi
        else:
            p1 = np.interp(centers, psi, p, left=0.0, right=0.0)
            xi = centers
    p1 = np.clip(p1, 0.0, None)
    s = p1.sum()
    if s <= 0:
        raise RuntimeError("empty 1D projection")
    p1 = p1 / s
    F1 = -kT * np.log(np.clip(p1, 1e-300, None))
    return xi, relative_fes(F1)


def mbar_pmf_2d(
    samples: list[np.ndarray],
    centers: np.ndarray,
    kappa: np.ndarray,
    *,
    beta: float,
    kT_kcal: float,
    n_bins: int,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    from pymbar import MBAR

    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    kappa = np.asarray(kappa, dtype=np.float64)
    if kappa.ndim == 1:
        kappa = np.broadcast_to(kappa.reshape(1, -1), centers.shape).copy()
    if kappa.shape != centers.shape:
        raise ValueError(f"kappa {kappa.shape} != centers {centers.shape}")
    K = centers.shape[0]

    parts = []
    N_k = np.zeros(K, dtype=np.int64)
    for k, s in enumerate(samples):
        s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
        parts.append(s)
        N_k[k] = s.shape[0]
    s_n = np.concatenate(parts, axis=0)
    N = int(s_n.shape[0])
    print(f"[mbar] windows={K}  samples={N}  N_k min/med/max="
          f"{N_k.min()}/{int(np.median(N_k))}/{N_k.max()}", flush=True)

    # u_kn[k,n] = β U_k(s_n)
    u_kn = np.zeros((K, N), dtype=np.float64)
    for k in range(K):
        d = wrap(s_n - centers[k])
        u_kn[k] = beta * 0.5 * np.sum(kappa[k] * d * d, axis=-1)

    mbar = MBAR(u_kn, N_k, verbose=verbose)
    f_k = np.asarray(mbar.f_k, dtype=np.float64)
    fe = mbar.compute_free_energy_differences()
    Delta_f = np.asarray(fe["Delta_f"], dtype=np.float64)

    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    phi = 0.5 * (edges[:-1] + edges[1:])
    psi = phi.copy()
    # 2D histogram counts
    H, _, _ = np.histogram2d(
        s_n[:, 0], s_n[:, 1], bins=[edges, edges]
    )  # (n_phi, n_psi) with x=φ
    # WHAM density on bin centers; store as (n_psi, n_phi) for xy plots
    g_phi, g_psi = np.meshgrid(phi, psi, indexing="xy")  # (nψ, nφ)
    grid = np.stack([g_phi.ravel(), g_psi.ravel()], axis=-1)  # (G, 2)
    counts_flat = H.T.ravel()  # (nψ, nφ) → flat
    p_flat = np.zeros(grid.shape[0], dtype=np.float64)
    log_N = np.log(np.maximum(N_k.astype(np.float64), 1.0))
    occupied = np.where(counts_flat > 0)[0]
    # chunk occupied bins to limit memory (G_occ × K × 2)
    chunk = 256
    for s0 in range(0, occupied.size, chunk):
        idx = occupied[s0 : s0 + chunk]
        g = grid[idx]  # (C, 2)
        d = wrap(g[:, None, :] - centers[None, :, :])  # (C, K, 2)
        u_s = beta * 0.5 * np.sum(kappa[None, :, :] * d * d, axis=-1)  # (C, K)
        log_den = log_N[None, :] + f_k[None, :] - u_s
        m = np.max(log_den, axis=1, keepdims=True)
        den = np.exp(m)[:, 0] * np.exp(log_den - m).sum(axis=1)
        p_flat[idx] = counts_flat[idx] / den
    p = p_flat.reshape(int(n_bins), int(n_bins))

    ok = p > 0
    if not np.any(ok):
        raise RuntimeError("no occupied 2D bins")
    p_norm = p.copy()
    p_norm[ok] = p[ok] / p[ok].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -kT_kcal * np.log(p_norm)
    F = relative_fes(F)
    F[~ok] = np.nan

    try:
        overlap = mbar.compute_overlap()
        O = np.asarray(
            overlap.get("matrix", overlap.get("overlap", []))
            if isinstance(overlap, dict)
            else overlap,
            dtype=np.float64,
        )
    except Exception:
        O = np.full((K, K), np.nan)

    return {
        "phi": phi,
        "psi": psi,
        "bin_edges": edges,
        "F_kcal": F,
        "p": p_norm,
        "counts": H.T.astype(np.float64),  # (psi, phi)
        "f_k_kT": f_k,
        "Delta_f_kT": Delta_f,
        "N_k": N_k.astype(np.float64),
        "overlap": O,
        "centers": centers,
        "kappa": kappa,
    }


def plot_fes(out: Path, phi, psi, F, F_phi, xi_phi, F_psi, xi_psi) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    Z = np.minimum(np.nan_to_num(F, nan=np.nanmax(F[np.isfinite(F)])), 20.0)
    im = axes[0].contourf(phi, psi, Z, levels=25, cmap="viridis")
    axes[0].set_xlabel(r"$\phi$ / rad")
    axes[0].set_ylabel(r"$\psi$ / rad")
    axes[0].set_title("US-MBAR $F(\\phi,\\psi)$")
    axes[0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].plot(xi_phi, F_phi, c="C0", lw=1.8)
    axes[1].set_xlabel(r"$\phi$ / rad")
    axes[1].set_ylabel(r"$F$ / kcal mol$^{-1}$")
    axes[1].set_title(r"$F(\phi)$")
    axes[1].set_xlim(-np.pi, np.pi)
    axes[2].plot(xi_psi, F_psi, c="C1", lw=1.8)
    axes[2].set_xlabel(r"$\psi$ / rad")
    axes[2].set_ylabel(r"$F$ / kcal mol$^{-1}$")
    axes[2].set_title(r"$F(\psi)$")
    axes[2].set_xlim(-np.pi, np.pi)
    fig.tight_layout()
    fig.savefig(out / "fes_mbar.png", dpi=160)
    fig.savefig(out / "fes_mbar.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="US result directory (has trajectory_cv.npy)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-bins", type=int, default=72)
    p.add_argument("--n-bins-1d", type=int, default=72)
    p.add_argument("--discard-frac", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = Path(args.result_dir).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else result / "mbar"
    out.mkdir(parents=True, exist_ok=True)

    info = _parse_run_info(result / "run_info.txt")
    T = float(args.temperature or info.get("temperature_K", 300.0))
    beta = 1.0 / (KB_EV_PER_K * T)
    kT_kcal = KB_EV_PER_K * T * EV_TO_KCAL

    cv = np.load(result / "trajectory_cv.npy")  # (T, W, 2)
    centers = np.load(result / "umbrella_centers.npy")
    kappa = np.load(result / "umbrella_kappa.npy")
    if cv.ndim != 3 or cv.shape[-1] != 2:
        raise ValueError(f"trajectory_cv shape {cv.shape}, expect (T,W,2)")
    n_frames, n_win, _ = cv.shape
    i0 = int(np.floor(float(args.discard_frac) * n_frames))
    cv = cv[i0:]
    print(
        f"[mbar] {result.name}  frames {i0}:{n_frames} kept={cv.shape[0]}  "
        f"windows={n_win}  T={T:g} K",
        flush=True,
    )

    samples = [cv[:, k, :] for k in range(n_win)]
    res = mbar_pmf_2d(
        samples,
        centers,
        kappa,
        beta=beta,
        kT_kcal=kT_kcal,
        n_bins=int(args.n_bins),
        verbose=bool(args.verbose),
    )
    xi_phi, F_phi = project_1d(
        res["phi"],
        res["psi"],
        res["F_kcal"],
        axis="phi",
        n_bins=int(args.n_bins_1d),
        temperature=T,
    )
    xi_psi, F_psi = project_1d(
        res["phi"],
        res["psi"],
        res["F_kcal"],
        axis="psi",
        n_bins=int(args.n_bins_1d),
        temperature=T,
    )

    np.savez_compressed(
        out / "fes_mbar.npz",
        phi=res["phi"],
        psi=res["psi"],
        F=res["F_kcal"],
        p=res["p"],
        counts=res["counts"],
        xi_phi=xi_phi,
        F_phi=F_phi,
        xi_psi=xi_psi,
        F_psi=F_psi,
        # aliases used by convergence scripts
        xi=xi_phi,
        F_kcal=F_phi,
        f_k_kT=res["f_k_kT"],
        N_k=res["N_k"],
        centers=res["centers"],
        kappa=res["kappa"],
        temperature_K=T,
        discard_frac=float(args.discard_frac),
    )
    plot_fes(out, res["phi"], res["psi"], res["F_kcal"], F_phi, xi_phi, F_psi, xi_psi)
    meta = {
        "result_dir": str(result),
        "temperature_K": T,
        "n_windows": int(n_win),
        "n_frames_kept": int(cv.shape[0]),
        "discard_frac": float(args.discard_frac),
        "F2d_span_kcal": float(np.nanmax(res["F_kcal"]) - np.nanmin(res["F_kcal"])),
        "F_phi_span_kcal": float(np.nanmax(F_phi) - np.nanmin(F_phi)),
    }
    (out / "mbar_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[mbar] F2d span={meta['F2d_span_kcal']:.2f}  F(φ) span={meta['F_phi_span_kcal']:.2f}", flush=True)
    print(f"[mbar] wrote {out}/fes_mbar.npz", flush=True)


if __name__ == "__main__":
    main()
